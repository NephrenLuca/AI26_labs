"""任务三主入口：BERT (Transformer) + 手写 CRF 完成 NER。

主要选项
--------
* ``--language``           : Chinese / English / all
* ``--device``             : cpu / cuda
* ``--gpu``                : 仅 ``cuda`` 时生效，逗号分隔的物理 GPU id
                             （0-7），多卡时自动启用 ``DataParallel``
* ``--model_name_or_path`` : 预训练模型名或本地路径；不指定时按语言取默认值
* ``--epochs``             : 训练轮数
* ``--batch_size``         : 训练 batch size（推理为 ``--eval_batch_size``）
* ``--lr_bert`` / ``--lr_head`` : 分组学习率（BERT vs CRF/分类头）
* ``--use_bilstm``         : 在 BERT 与 CRF 间插入一层 BiLSTM
* ``--max_seq_length``     : 子词序列最大长度（≤ 512）
* ``--max_words_per_chunk``: 一个 chunk 最多包含多少 word
* ``--seed``               : 随机种子

预训练模型下载
--------------
* 中文：``bert-base-chinese``
  - 官方     : https://huggingface.co/bert-base-chinese
  - HF 镜像  : https://hf-mirror.com/bert-base-chinese
* 英文：``bert-base-cased``
  - 官方     : https://huggingface.co/google-bert/bert-base-cased
  - HF 镜像  : https://hf-mirror.com/google-bert/bert-base-cased

使用 ``transformers.AutoTokenizer.from_pretrained`` /
``AutoModel.from_pretrained`` 时若网络不通，可设
``HF_ENDPOINT=https://hf-mirror.com`` 或先 ``huggingface-cli download``
到本地，再用 ``--model_name_or_path /path/to/local`` 指向本地目录。

示例
----
    # 单卡训练 + 评测
    python run.py --language Chinese --device cuda --gpu 0 --epochs 3

    # 多卡训练（DataParallel）
    python run.py --language all --device cuda --gpu 0,1,2,3 --batch_size 64

    # CPU 调试
    python run.py --language English --device cpu --epochs 1 --batch_size 8
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
NER_DIR = os.path.normpath(os.path.join(HERE, "..", "NER"))


@contextmanager
def timed(name: str):
    t0 = time.time()
    yield
    print(f"[time] {name}: {time.time() - t0:.2f}s")


# ----------------------------------------------------------------------
# 设备处理（必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES）
# ----------------------------------------------------------------------


def setup_devices(device_arg: str, gpu_arg: str) -> None:
    """根据 ``--device``/``--gpu`` 配置可见 GPU 并打印信息。"""
    if device_arg == "cuda":
        ids = [s.strip() for s in gpu_arg.split(",") if s.strip()]
        if not ids or not all(s.isdigit() for s in ids):
            raise SystemExit(
                f"--gpu 必须是逗号分隔的非负整数（0-7），得到: {gpu_arg!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        print(
            f"[设备] 请求 cuda；CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
        )
    else:
        # 显式禁用 CUDA，避免 torch 默认占用
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        print("[设备] 请求 cpu")


# ----------------------------------------------------------------------
# 默认预训练模型
# ----------------------------------------------------------------------


DEFAULT_MODEL = {
    "Chinese": "bert-base-chinese",
    "English": "bert-base-cased",
}


# ----------------------------------------------------------------------
# 训练 / 验证 / 预测主流程
# ----------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch  # 局部 import：确保已经设过 CUDA_VISIBLE_DEVICES

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer_and_scheduler(
    model,
    num_training_steps: int,
    lr_bert: float,
    lr_head: float,
    weight_decay: float,
    warmup_ratio: float,
):
    import torch
    from torch.optim import AdamW

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    bert_decay, bert_no_decay, head_decay, head_no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bert = n.startswith("bert.") or n.startswith("module.bert.")
        is_no_decay = any(nd in n for nd in no_decay)
        if is_bert:
            (bert_no_decay if is_no_decay else bert_decay).append(p)
        else:
            (head_no_decay if is_no_decay else head_decay).append(p)
    param_groups = [
        {"params": bert_decay, "lr": lr_bert, "weight_decay": weight_decay},
        {"params": bert_no_decay, "lr": lr_bert, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups)
    num_warmup = int(num_training_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = float(step - num_warmup) / float(
            max(1, num_training_steps - num_warmup)
        )
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def predict(
    model,
    dataset,
    collate_fn,
    device,
    batch_size: int,
) -> List[List[int]]:
    """对整个 dataset 进行 Viterbi 解码，按 dataset 顺序返回。"""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    underlying = model.module if hasattr(model, "module") else model
    underlying.eval()
    all_preds: List[List[int]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            first_subword_idx = batch["first_subword_idx"].to(device, non_blocking=True)
            word_mask = batch["word_mask"].to(device, non_blocking=True)
            # 直接使用 underlying（不要走 DataParallel）以保证返回 list
            preds = underlying(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_idx=first_subword_idx,
                word_mask=word_mask,
                labels=None,
            )
            all_preds.extend(preds)
    return all_preds


def run_one(language: str, args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print(
        f"=== 语言: {language} | device={args.device} | gpu={args.gpu} "
        f"| model={args.model_name_or_path or DEFAULT_MODEL[language]} ==="
    )
    print("=" * 60)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    from data_utils import (
        build_dataset,
        build_inference_dataset,
        build_label_maps,
        get_label_list,
        group_into_sentences,
        make_collate_fn,
        read_conll,
        read_lines_with_blanks,
        reassemble_predictions,
        write_predictions,
    )
    from evaluate import evaluate, micro_f1
    from model import BertCRFForNER

    # -------- 数据 --------
    train_path = os.path.join(args.data_dir, language, "train.txt")
    valid_path = os.path.join(args.data_dir, language, "validation.txt")
    if not os.path.isfile(train_path) or not os.path.isfile(valid_path):
        raise FileNotFoundError(f"找不到数据文件：{train_path} 或 {valid_path}")

    label_list = get_label_list(language)
    label2id, id2label = build_label_maps(label_list)
    print(f"标签数: {len(label_list)}")

    with timed("加载训练数据"):
        train_sents = read_conll(train_path)
    print(f"训练句子数: {len(train_sents)}")

    # -------- 设备与模型 --------
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[警告] cuda 不可用，自动回退到 cpu")
            device = torch.device("cpu")
            n_gpu = 0
        else:
            device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES 之后总是从 0 开始
            n_gpu = torch.cuda.device_count()
            print(f"[设备] 实际可见 GPU 数: {n_gpu}")
    else:
        device = torch.device("cpu")
        n_gpu = 0

    model_name = args.model_name_or_path or DEFAULT_MODEL[language]
    print(f"加载预训练模型: {model_name}")
    with timed("加载 tokenizer / 模型"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, do_lower_case=False
        )
        bert = AutoModel.from_pretrained(model_name)

    model = BertCRFForNER(
        bert_model=bert,
        num_tags=len(label_list),
        hidden_dropout_prob=args.dropout,
        use_bilstm=args.use_bilstm,
        bilstm_hidden=args.bilstm_hidden,
        bilstm_layers=args.bilstm_layers,
        bilstm_dropout=args.dropout,
    )
    model.to(device)
    if n_gpu > 1:
        print(f"[多卡] 启用 DataParallel，n_gpu={n_gpu}")
        model = torch.nn.DataParallel(model)

    # -------- 数据集 --------
    pad_token_id = tokenizer.pad_token_id or 0
    collate_fn = make_collate_fn(pad_token_id)

    with timed("构造训练集 / 验证集"):
        train_dataset = build_dataset(
            train_sents,
            tokenizer,
            label2id,
            max_seq_length=args.max_seq_length,
            max_words_per_chunk=args.max_words_per_chunk,
            have_labels=True,
        )
        valid_records = read_lines_with_blanks(valid_path)
        valid_sentences = group_into_sentences(valid_records)
        valid_dataset = build_inference_dataset(
            valid_sentences,
            tokenizer,
            max_seq_length=args.max_seq_length,
            max_words_per_chunk=args.max_words_per_chunk,
        )
    print(
        f"训练 chunk 数: {len(train_dataset)} | "
        f"验证句子数: {len(valid_sentences)} | "
        f"验证 chunk 数: {len(valid_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -------- 优化器与调度器 --------
    num_training_steps = max(1, len(train_loader) * args.epochs)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        num_training_steps=num_training_steps,
        lr_bert=args.lr_bert,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    # -------- 训练循环 --------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{language}_pred.txt")
    best_f1 = -1.0
    best_epoch = -1
    log_interval = max(1, len(train_loader) // 10)

    for epoch in range(1, args.epochs + 1):
        underlying = model.module if hasattr(model, "module") else model
        underlying.train()
        running_loss = 0.0
        running_steps = 0
        t_epoch = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch_on_device = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            loss = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                first_subword_idx=batch_on_device["first_subword_idx"],
                word_mask=batch_on_device["word_mask"],
                labels=batch_on_device["labels"],
            )
            if loss.dim() > 0:
                # DataParallel 会返回每卡一个标量
                loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            running_steps += 1
            if step % log_interval == 0 or step == len(train_loader):
                avg = running_loss / running_steps
                lr0 = optimizer.param_groups[0]["lr"]
                lr2 = optimizer.param_groups[2]["lr"]
                print(
                    f"  [epoch {epoch}/{args.epochs}] step {step:>5d}/"
                    f"{len(train_loader)}  loss={avg:.4f}  "
                    f"lr_bert={lr0:.2e}  lr_head={lr2:.2e}"
                )
                running_loss = 0.0
                running_steps = 0
        print(f"  [epoch {epoch}] 用时 {time.time() - t_epoch:.1f}s")

        # 每个 epoch 末做一次预测 + 评测，保存最优
        with timed(f"  [epoch {epoch}] Viterbi 解码验证集"):
            chunk_preds = predict(
                model,
                valid_dataset,
                collate_fn,
                device,
                batch_size=args.eval_batch_size,
            )
        sent_preds = reassemble_predictions(
            n_sentences=len(valid_sentences),
            encoded_chunks=valid_dataset.encoded_chunks,
            chunk_predictions=chunk_preds,
            id2label=id2label,
        )
        write_predictions(out_path, valid_records, sent_preds)
        f1 = micro_f1(language, valid_path, out_path)
        print(f"  [epoch {epoch}] dev micro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            # 同时把当前 best 预测保留为单独副本
            best_path = os.path.join(args.output_dir, f"{language}_pred_best.txt")
            write_predictions(best_path, valid_records, sent_preds)
            if args.save_model:
                state = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                ckpt_path = os.path.join(args.output_dir, f"{language}_best.pt")
                torch.save(state, ckpt_path)
                print(f"  [epoch {epoch}] 保存 best ckpt -> {ckpt_path}")

    print(
        f"\n[Final] best epoch = {best_epoch}, best dev micro-F1 = {best_f1:.4f}"
    )
    # 用 best 副本覆盖最终输出
    best_path = os.path.join(args.output_dir, f"{language}_pred_best.txt")
    if os.path.isfile(best_path):
        # 把 best 预测复制到最终输出（如果存在）
        with open(best_path, "r", encoding="utf-8") as fr:
            content = fr.read()
        with open(out_path, "w", encoding="utf-8", newline="\n") as fw:
            fw.write(content)
    print(f"最终预测输出: {out_path}")
    print("\n--- 评测 (与 NER/check.py 等价；按 micro avg F1 评分) ---")
    evaluate(language=language, gold_path=valid_path, my_path=out_path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transformer + CRF NER —— 任务三",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--language", choices=["Chinese", "English", "all"], default="all"
    )
    parser.add_argument(
        "--data_dir",
        default=NER_DIR,
        help="NER 数据集根目录，包含 Chinese/ 与 English/ 子目录。",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(HERE, "outputs"),
        help="预测结果与 ckpt 输出目录。",
    )

    # 设备
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--gpu",
        default="0",
        help=("（仅 --device cuda 时生效）指定可见 GPU 物理 id（0-7），"
              "可单个或逗号分隔，例如 '0' 或 '0,1,2,3'；多卡时启用 DataParallel。"),
    )

    # 模型
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help=("HuggingFace 模型名或本地路径；不指定时按语言取默认值："
              " Chinese -> bert-base-chinese, English -> bert-base-cased。"),
    )
    parser.add_argument("--use_bilstm", action="store_true")
    parser.add_argument("--bilstm_hidden", type=int, default=256)
    parser.add_argument("--bilstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr_bert", type=float, default=3e-5)
    parser.add_argument(
        "--lr_head", type=float, default=1e-3,
        help="非 BERT 参数（CRF + 分类头 + BiLSTM）的学习率，需要远大于 BERT。",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 数据
    parser.add_argument(
        "--max_seq_length", type=int, default=512,
        help="子词序列最大长度，必须 <= BERT 的 512。",
    )
    parser.add_argument(
        "--max_words_per_chunk", type=int, default=200,
        help="一段切片内最多多少 word；中文 token=字符可放更大，英文略保守。",
    )

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    setup_devices(args.device, args.gpu)
    set_seed(args.seed)

    if args.max_seq_length > 512:
        print("[警告] BERT 仅支持 max_seq_length <= 512，自动截断到 512")
        args.max_seq_length = 512

    languages = ["Chinese", "English"] if args.language == "all" else [args.language]
    for lang in languages:
        run_one(lang, args)


if __name__ == "__main__":
    main()

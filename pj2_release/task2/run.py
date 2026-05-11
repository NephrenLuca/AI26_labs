"""任务二主入口：训练手写线性链 CRF、在验证集上预测、调用 check.py 等价评测。

主要选项
--------
- ``--language``       : Chinese / English / all
- ``--device``         : cpu / cuda
- ``--gpu``            : 物理 GPU id 列表（仅 --device cuda 时生效，例如 ``--gpu 3``
                         或 ``--gpu 0,2``，将通过 ``CUDA_VISIBLE_DEVICES`` 限制可见
                         设备；本任务模型规模小，单卡训练即可，多卡场景将使用
                         可见设备中的第 0 张）
- ``--epochs``         : 训练轮数
- ``--batch_size``     : 每批句子数
- ``--lr``             : AdamW 学习率
- ``--weight_decay``   : AdamW L2 正则强度
- ``--feat_min_count`` : 训练集中频次 < 该阈值的特征会被丢弃
- ``--max_grad_norm``  : 梯度裁剪阈值
- ``--seed``           : 随机种子
- ``--save_model``     : 是否保存模型权重（``outputs/{lang}_crf.pt``）

示例
----
    python run.py --language all
    python run.py --language English --device cuda --gpu 3 --epochs 15
    python run.py --language Chinese --device cuda --gpu 0,1 --batch_size 64
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import List, Sequence, Tuple

# 让 ``check.py`` 可被导入
HERE = os.path.dirname(os.path.abspath(__file__))
NER_DIR = os.path.normpath(os.path.join(HERE, "..", "NER"))
if NER_DIR not in sys.path:
    sys.path.insert(0, NER_DIR)


@contextmanager
def timed(name: str):
    t0 = time.time()
    yield
    print(f"[time] {name}: {time.time() - t0:.2f}s")


# ---------------------------- 单语言训练流程 ----------------------------


def run_one(language: str, args: argparse.Namespace) -> None:
    """对单一语言（Chinese / English）跑一遍 train -> predict -> evaluate。"""
    # 这些模块依赖 torch（features 不依赖，但放在一起便于阅读）；在 main() 中
    # 已经设置好 CUDA_VISIBLE_DEVICES，此处再导入 torch 安全。
    import torch
    from torch.utils.data import DataLoader

    from crf import LinearChainCRF, collate
    from data_utils import (
        group_into_sentences,
        read_conll,
        read_lines_with_blanks,
        write_predictions,
    )
    from evaluate import (
        SORTED_LABELS_CHN,
        SORTED_LABELS_ENG,
        evaluate,
    )
    from features import FeatureExtractor, build_tag_vocab

    print("\n" + "=" * 60)
    print(
        f"=== 语言: {language} | 设备: {args.device} "
        f"| epochs={args.epochs} | bs={args.batch_size} | lr={args.lr} ==="
    )
    print("=" * 60)

    # ---------------- 数据 ----------------
    train_path = os.path.join(args.data_dir, language, "train.txt")
    valid_path = os.path.join(args.data_dir, language, "validation.txt")
    if not os.path.isfile(train_path) or not os.path.isfile(valid_path):
        raise FileNotFoundError(
            f"找不到数据文件：{train_path} 或 {valid_path}"
        )

    with timed("加载训练数据"):
        train_sents = read_conll(train_path)
    print(f"训练句子数: {len(train_sents)}")

    canonical = SORTED_LABELS_ENG if language == "English" else SORTED_LABELS_CHN
    id2tag, tag2id = build_tag_vocab(train_sents, canonical=canonical)
    print(f"标签数: {len(id2tag)}")

    # ---------------- 特征 ----------------
    extractor = FeatureExtractor(
        language=language,
        use_shape=not args.no_shape,
        use_affix=not args.no_affix,
        max_affix_len=args.max_affix_len,
    )
    with timed("构建特征词表"):
        extractor.build_vocab(train_sents, min_count=args.feat_min_count)
    print(
        f"特征数: {len(extractor.vocab)} "
        f"(min_count={args.feat_min_count})"
    )

    # 为每个训练句子提取 (feat_ids per position, tag_ids)
    with timed("提取训练特征"):
        train_data: List[Tuple[List[List[int]], List[int]]] = []
        for tokens, tags in train_sents:
            feats = extractor.feat_ids(tokens)
            tag_ids = [tag2id[t] for t in tags]
            train_data.append((feats, tag_ids))

    # ---------------- 模型 ----------------
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[警告] 请求 cuda 但 torch.cuda.is_available()==False，自动回退到 cpu")
        device_str = "cpu"
    device = torch.device(device_str)

    model = LinearChainCRF(
        num_features=len(extractor.vocab), num_tags=len(id2tag)
    ).to(device)
    print(
        f"模型: LinearChainCRF | 参数量 = "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    if args.optimizer == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:  # sgd
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )

    # 简单的随机打散 + DataLoader（collate 在 crf.py 中实现）
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    def make_loader(data, shuffle: bool):
        return DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

    train_loader = make_loader(train_data, shuffle=True)

    # ---------------- 训练 ----------------
    with timed("训练 CRF"):
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            n_sents = 0
            t0 = time.time()
            for batch in train_loader:
                batch = batch.to(device)
                optim.zero_grad(set_to_none=True)
                loss = model.neg_log_likelihood(batch)
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optim.step()
                bs = batch.batch_size
                total_loss += float(loss.item()) * bs
                n_sents += bs
            print(
                f"  epoch {epoch:>2d}/{args.epochs}  "
                f"avg NLL/sent = {total_loss / max(1, n_sents):.4f}  "
                f"({time.time() - t0:.1f}s)"
            )

    # ---------------- 验证 ----------------
    with timed("加载验证数据"):
        records = read_lines_with_blanks(valid_path)
        valid_sents_tokens = group_into_sentences(records)
    print(
        f"验证句子数: {len(valid_sents_tokens)} | "
        f"验证 token 数: {sum(len(s) for s in valid_sents_tokens)}"
    )

    with timed("Viterbi 解码"):
        model.eval()
        valid_data = [
            (extractor.feat_ids(toks), [0] * len(toks))  # 占位 tag，仅满足 collate 形状
            for toks in valid_sents_tokens
        ]
        valid_loader = make_loader(valid_data, shuffle=False)
        predictions: List[List[str]] = []
        for batch in valid_loader:
            batch = batch.to(device)
            decoded = model.decode(batch)
            for tag_ids in decoded:
                predictions.append([id2tag[i] for i in tag_ids])

    # 安全检查：预测序列长度需与原 token 序列一致
    for orig, pred in zip(valid_sents_tokens, predictions):
        if len(orig) != len(pred):
            raise RuntimeError(
                f"句子长度不一致：orig={len(orig)} vs pred={len(pred)}"
            )

    out_path = os.path.join(args.output_dir, f"{language}_pred.txt")
    write_predictions(out_path, records, predictions)
    print(f"预测结果已写入: {out_path}")

    if args.save_model:
        model_path = os.path.join(args.output_dir, f"{language}_crf.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "id2tag": id2tag,
                "feat2id": extractor.vocab.feat2id,
                "language": language,
                "args": vars(args),
            },
            model_path,
        )
        print(f"模型权重已保存: {model_path}")

    print("\n--- 评测 (与 NER/check.py 等价；按 micro avg F1 评分) ---")
    evaluate(language=language, gold_path=valid_path, my_path=out_path)


# ---------------------------- 入口 ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRF NER —— 任务二（手写线性链 CRF + PyTorch）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--language", choices=["Chinese", "English", "all"], default="all"
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.normpath(os.path.join(HERE, "..", "NER")),
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(HERE, "outputs")
    )

    # 设备
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--gpu",
        default="0",
        help="（仅 --device cuda 时生效）指定可见 GPU 物理 id（0-7），"
             "可单个或逗号分隔；通过 CUDA_VISIBLE_DEVICES 限制可见设备。",
    )

    # 训练超参
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam", "sgd"],
        default="adamw",
        help="优化器；adamw 推荐，sgd 用于规避部分 Windows 环境上 torch._dynamo "
             "兼容问题（不影响 Linux GPU 服务器）。",
    )

    # 特征
    parser.add_argument(
        "--feat_min_count",
        type=int,
        default=2,
        help="训练集中出现频次 < 该阈值的特征会被丢弃（L0 正则）。",
    )
    parser.add_argument(
        "--max_affix_len",
        type=int,
        default=3,
        help="英文前/后缀最长长度（中文按字符级 token，无前后缀概念）。",
    )
    parser.add_argument(
        "--no_shape", action="store_true", help="关闭形态学（shape）特征。"
    )
    parser.add_argument(
        "--no_affix", action="store_true", help="关闭前/后缀特征。"
    )

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_model", action="store_true", help="把模型权重保存到 outputs/。"
    )

    args = parser.parse_args()

    # 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES
    if args.device == "cuda":
        ids = [s.strip() for s in args.gpu.split(",") if s.strip()]
        if not ids or not all(s.isdigit() for s in ids):
            parser.error(
                f"--gpu 必须是逗号分隔的非负整数，得到: {args.gpu!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        print(
            f"[设备] 请求 cuda，CUDA_VISIBLE_DEVICES="
            f"{os.environ['CUDA_VISIBLE_DEVICES']}"
            "（若 cuda 不可用会自动回退到 cpu）"
        )
    else:
        print(f"[设备] 请求 {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    languages = ["Chinese", "English"] if args.language == "all" else [args.language]
    for lang in languages:
        run_one(lang, args)


if __name__ == "__main__":
    main()

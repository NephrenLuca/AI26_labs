"""任务一主入口：训练 HMM、在验证集上预测、调用 check.py 等价评测。

主要选项
--------
- ``--language``            : Chinese / English / all
- ``--device``              : cpu / cuda（用 cupy）
- ``--gpu``                 : 物理 GPU id 列表
- ``--order``               : 1（一阶 HMM）/ 2（二阶 trigram HMM）
- ``--alpha_pi/trans/emit`` : π / A / B 的 Laplace α
- ``--rare_threshold``      : 训练集中 freq ≤ 该阈值的词参与 TnT 后缀树
- ``--max_suffix_len``      : 后缀最大长度（中文自动 = 1，英文 10）
- ``--suffix_min_count``    : 一个后缀至少出现这么多次才进 SUF shape vocab
- ``--theta_tnt``           : TnT 后缀树递归平滑权重 θ
- ``--weight_tnt``          : OOV 推理时 TnT 相对发射的混合权重，0 = 关 TnT
- ``--no_tnt_tree``         : 关掉 TnT 后缀树（消融）
- ``--no_suffix_oov``       : 关掉整个 shape vocab（消融）
- ``--no_constraints``      : 关掉 BIO/BMES 结构约束（消融）
- ``--tune``                : 在 train 切 mini-dev 做超参网格搜索

示例
----
    python run.py --language all --order 2
    python run.py --language English --device cuda --gpu 3 --order 2 --tune
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

# 让 ``check.py`` 可被导入
HERE = os.path.dirname(os.path.abspath(__file__))
NER_DIR = os.path.normpath(os.path.join(HERE, "..", "NER"))
if NER_DIR not in sys.path:
    sys.path.insert(0, NER_DIR)

from data_utils import (  # noqa: E402
    group_into_sentences,
    read_conll,
    read_lines_with_blanks,
    write_predictions,
)
from evaluate import evaluate  # noqa: E402
from hmm import HMM  # noqa: E402


@contextmanager
def timed(name: str):
    t0 = time.time()
    yield
    print(f"[time] {name}: {time.time() - t0:.2f}s")


# ---------------------------- 网格搜索 ---------------------------------


def split_train_dev(
    sentences: Sequence[Tuple[List[str], List[str]]],
    dev_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]]]:
    rng = random.Random(seed)
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    n_dev = max(1, int(len(sentences) * dev_ratio))
    dev_idx = set(indices[:n_dev])
    train_split, dev_split = [], []
    for i, s in enumerate(sentences):
        (dev_split if i in dev_idx else train_split).append(s)
    return train_split, dev_split


def _f1_micro(
    y_true: Iterable[str], y_pred: Iterable[str], labels: Sequence[str]
) -> float:
    from sklearn.metrics import f1_score

    return f1_score(
        list(y_true), list(y_pred), labels=list(labels), average="micro",
        zero_division=0,
    )


def _build_hmm(args: argparse.Namespace, **overrides) -> HMM:
    """根据 args 构造一个 HMM；``overrides`` 用于网格搜索覆盖个别超参。"""
    kw = dict(
        device=args.device,
        order=args.order,
        alpha_pi=args.alpha_pi,
        alpha_trans=args.alpha_trans,
        alpha_emit=args.alpha_emit,
        min_word_freq=args.min_word_freq,
        use_suffix_oov=not args.no_suffix_oov,
        rare_threshold=args.rare_threshold,
        max_suffix_len=args.max_suffix_len,
        suffix_min_count=args.suffix_min_count,
        use_tnt_tree=not args.no_tnt_tree,
        theta_tnt=args.theta_tnt,
        weight_tnt=args.weight_tnt,
        use_constraints=not args.no_constraints,
    )
    kw.update(overrides)
    return HMM(**kw)


def grid_search(
    train_split,
    dev_split,
    args: argparse.Namespace,
) -> Tuple[dict, float]:
    """5D 网格：α_π × α_A × α_B × weight_tnt × suffix_min_count。"""
    pi_grid = tuple(args.alpha_pi_grid)
    at_grid = tuple(args.alpha_trans_grid)
    ae_grid = tuple(args.alpha_emit_grid)
    wt_grid = tuple(args.weight_tnt_grid)
    sm_grid = tuple(args.suffix_min_count_grid)
    total = len(pi_grid) * len(at_grid) * len(ae_grid) * len(wt_grid) * len(sm_grid)
    print(
        f"\n[Grid Search] α_π × α_A × α_B × w_tnt × suf_min "
        f"= {len(pi_grid)} × {len(at_grid)} × {len(ae_grid)} × {len(wt_grid)} × {len(sm_grid)} "
        f"= {total} 组"
    )
    dev_tokens = [t for t, _ in dev_split]
    dev_gold_flat = [tag for _, tags in dev_split for tag in tags]

    best = (-1.0, None)
    n = 0
    for ap in pi_grid:
        for at in at_grid:
            for ae in ae_grid:
                for wt in wt_grid:
                    for sm in sm_grid:
                        n += 1
                        m = _build_hmm(
                            args,
                            alpha_pi=ap,
                            alpha_trans=at,
                            alpha_emit=ae,
                            weight_tnt=wt,
                            suffix_min_count=sm,
                        )
                        m.fit(train_split)
                        preds = m.predict(dev_tokens, batch_size=args.batch_size)
                        pred_flat = [tag for sent in preds for tag in sent]
                        labels = [t for t in m.id2tag if t != "O"]
                        f1 = _f1_micro(dev_gold_flat, pred_flat, labels)
                        marker = ""
                        if f1 > best[0]:
                            best = (
                                f1,
                                dict(
                                    alpha_pi=ap,
                                    alpha_trans=at,
                                    alpha_emit=ae,
                                    weight_tnt=wt,
                                    suffix_min_count=sm,
                                ),
                            )
                            marker = "  <-- best"
                        print(
                            f"  [{n:>3d}/{total}] α_π={ap:<6g} α_A={at:<8g} "
                            f"α_B={ae:<10g} w_tnt={wt:<5g} suf_min={sm:<4g}  "
                            f"dev F1 = {f1:.4f}{marker}"
                        )
    assert best[1] is not None
    cfg = best[1]
    print(
        f"[Grid Search] 最佳：α_π={cfg['alpha_pi']:g} α_A={cfg['alpha_trans']:g} "
        f"α_B={cfg['alpha_emit']:g} w_tnt={cfg['weight_tnt']:g} "
        f"suf_min={cfg['suffix_min_count']:g} -> dev F1 = {best[0]:.4f}"
    )
    return cfg, best[0]


# ---------------------------- 主流程 -----------------------------------


def run_one(language: str, args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print(
        f"=== 语言: {language} | 设备: {args.device} | order={args.order} "
        f"| suffix_oov={not args.no_suffix_oov} "
        f"| tnt_tree={not args.no_tnt_tree} "
        f"| constraints={not args.no_constraints} ==="
    )
    print("=" * 60)

    train_path = os.path.join(args.data_dir, language, "train.txt")
    valid_path = os.path.join(args.data_dir, language, "validation.txt")
    if not os.path.isfile(train_path) or not os.path.isfile(valid_path):
        raise FileNotFoundError(f"找不到数据文件：{train_path} 或 {valid_path}")

    with timed("加载训练数据"):
        train_sents = read_conll(train_path)
    print(f"训练句子数: {len(train_sents)}")

    overrides: dict = {}
    if args.tune:
        train_split, dev_split = split_train_dev(
            train_sents, dev_ratio=args.dev_ratio, seed=args.seed
        )
        print(
            f"切分 mini-dev: train={len(train_split)}, dev={len(dev_split)} "
            f"(seed={args.seed}, ratio={args.dev_ratio})"
        )
        with timed("网格搜索"):
            best_cfg, _ = grid_search(train_split, dev_split, args)
        overrides.update(best_cfg)

    print(
        "\n[Final] α_π={alpha_pi:g}  α_A={alpha_trans:g}  α_B={alpha_emit:g}  "
        "w_tnt={weight_tnt:g}  suf_min={suffix_min_count:g}".format(
            alpha_pi=overrides.get("alpha_pi", args.alpha_pi),
            alpha_trans=overrides.get("alpha_trans", args.alpha_trans),
            alpha_emit=overrides.get("alpha_emit", args.alpha_emit),
            weight_tnt=overrides.get("weight_tnt", args.weight_tnt),
            suffix_min_count=overrides.get("suffix_min_count", args.suffix_min_count),
        )
    )
    model = _build_hmm(args, **overrides)
    print(f"实际数值后端: {model.xp.__name__} ({model.device})")
    with timed("训练 HMM (在全部 train 上)"):
        model.fit(train_sents)
    print(
        f"语言: {model.language} | 标签方案: {model.scheme} | "
        f"标签数: {len(model.id2tag)} | 词表大小(无 shape, 仅 freq≥{model.min_word_freq}): {len(model.id2word)}"
    )
    if model.order == 2 and model.lambdas is not None:
        l1, l2, l3 = model.lambdas
        print(f"Brants 插值权重: λ1(uni)={l1:.3f}, λ2(bi)={l2:.3f}, λ3(tri)={l3:.3f}")

    with timed("加载验证数据"):
        records = read_lines_with_blanks(valid_path)
        valid_sents = group_into_sentences(records)
    print(
        f"验证句子数: {len(valid_sents)} | 验证 token 数: {sum(len(s) for s in valid_sents)}"
    )
    with timed("Viterbi 解码"):
        predictions: List[List[str]] = model.predict(
            valid_sents, batch_size=args.batch_size
        )

    out_path = os.path.join(args.output_dir, f"{language}_pred.txt")
    write_predictions(out_path, records, predictions)
    print(f"预测结果已写入: {out_path}")

    if args.save_model:
        model_path = os.path.join(args.output_dir, f"{language}_hmm.pkl")
        model.save(model_path)
        print(f"模型参数已保存: {model_path}")

    print("\n--- 评测 (与 NER/check.py 等价；按 micro avg F1 评分) ---")
    evaluate(language=language, gold_path=valid_path, my_path=out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM NER —— 任务一",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--language", choices=["Chinese", "English", "all"], default="all")
    parser.add_argument("--data_dir", default=os.path.normpath(os.path.join(HERE, "..", "NER")))
    parser.add_argument("--output_dir", default=os.path.join(HERE, "outputs"))

    # 设备
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--gpu", default="0",
        help="（仅 --device cuda 时生效）指定可见 GPU 物理 id（0-7），可单个或逗号分隔。",
    )

    # 模型超参
    parser.add_argument(
        "--order", type=int, default=2, choices=[1, 2],
        help="HMM 阶数：1 = bigram，2 = trigram (推荐)。",
    )
    parser.add_argument("--alpha_pi", type=float, default=1.0)
    parser.add_argument("--alpha_trans", type=float, default=0.01)
    parser.add_argument("--alpha_emit", type=float, default=1e-4)
    parser.add_argument("--min_word_freq", type=int, default=2)

    # 后缀模型
    parser.add_argument("--rare_threshold", type=int, default=10,
                        help="freq ≤ 该阈值的训练词都参与后缀分类器。")
    parser.add_argument("--max_suffix_len", type=int, default=10,
                        help="后缀最大长度。中文是字符级，自动设为 1。")
    parser.add_argument("--suffix_min_count", type=int, default=200,
                        help="一个后缀至少出现这么多次才视为有效 SUF shape；"
                             "其余落到固定类别 CAT shape。")
    parser.add_argument("--theta_tnt", type=float, default=1.0,
                        help="TnT 后缀树递归平滑权重 θ。")
    parser.add_argument("--weight_tnt", type=float, default=0.3,
                        help="OOV 推理时 TnT 相对发射的混合权重，0 = 关 TnT。")

    # 消融
    parser.add_argument("--no_suffix_oov", action="store_true",
                        help="关掉 shape vocab。")
    parser.add_argument("--no_tnt_tree", action="store_true",
                        help="关掉 TnT 后缀树。")
    parser.add_argument("--no_constraints", action="store_true",
                        help="关掉 BIO/BMES 结构化约束。")

    # 网格搜索
    parser.add_argument("--tune", action="store_true",
                        help="在 train 切 mini-dev 做超参网格搜索。")
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha_pi_grid", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--alpha_trans_grid", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--alpha_emit_grid", type=float, nargs="+", default=[1e-4, 1e-3])
    parser.add_argument("--weight_tnt_grid", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--suffix_min_count_grid", type=int, nargs="+", default=[100, 200])

    # 其他
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    # 中文默认换成 max_suffix_len=1（字符级，单字本身就是最长可能后缀）
    if args.language in ("Chinese",) and args.max_suffix_len > 2:
        # 不强制覆盖，仅给出提示
        print(
            f"[提示] 当前 --max_suffix_len={args.max_suffix_len}；"
            "中文是字符级 token，建议 1 或 2，可在 CLI 显式指定。"
        )

    if args.device == "cuda":
        ids = [s.strip() for s in args.gpu.split(",") if s.strip()]
        if not ids or not all(s.isdigit() for s in ids):
            parser.error(f"--gpu 必须是逗号分隔的非负整数，得到: {args.gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        print(
            f"[设备] 请求 cuda，CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
            "（cuda 不可用时会自动回退到 cpu）"
        )
    else:
        print(f"[设备] 请求 {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    languages = ["Chinese", "English"] if args.language == "all" else [args.language]
    for lang in languages:
        # 中文用更短的 suffix 默认（仅当用户没显式覆盖时）
        if lang == "Chinese" and args.max_suffix_len == 10:
            args.max_suffix_len_resolved = 1
        else:
            args.max_suffix_len_resolved = args.max_suffix_len
        # 临时把 max_suffix_len 重定向为 resolved 值，run_one 用的是 args.max_suffix_len
        original = args.max_suffix_len
        args.max_suffix_len = args.max_suffix_len_resolved
        try:
            run_one(lang, args)
        finally:
            args.max_suffix_len = original


if __name__ == "__main__":
    main()

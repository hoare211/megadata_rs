"""第三方分词器吞吐对照基准（方案 §8）。

三向对照，回答"用第三方分词器到底慢多少、慢在哪一步"：

| 口径 | 语义 | 用途 |
|---|---|---|
| **A. bbpe 基线** | `tokenizer_path=` 指到 bbpe 模型（Rust 直连） | 默认路径的性能上界 |
| **B. HF 单条循环** | `for t in texts: hf.encode(t)` | 模拟 naive 的 Python 侧逐条调用 |
| **C. megadata + HF 批量** | `DataBuilder(tokenizer=HfTokenizerAdapter(...))` | 本方案的真实构建吞吐 |

A 与 C 需要同一份分词器的两种形态，用 `--bbpe-model` / `--hf-model` 分别给出
（`tests/test_tokenizer_integration.py` 的 `write_bbpe_model` 可以产出前者）；
只给了 `--hf-model` 时跳过 A。

用法：
    python bench/third_party_tokenizer_bench.py \\
        --hf-model /path/to/tokenizer.json \\
        [--bbpe-model /path/to/model.bbpe] \\
        --docs 20000 --out /tmp/bench_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

CHAT_TEMPLATE = (
    "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"
)


def make_corpus(n: int) -> List[str]:
    """合成中文 + 英文 + emoji 的对话语料（三向口径共用，保证可比）。"""
    return [
        json.dumps({
            "messages": [
                {"role": "user", "content": f"问题{i}：机器学习需要多少数据？😀"},
                {"role": "assistant", "content": f"回答{i}：越多越好，但质量优先，中文语料尤其如此。"},
            ]
        }, ensure_ascii=False)
        for i in range(n)
    ]


def rendered_texts(n: int) -> List[str]:
    """口径 B 的输入：模板渲染后的文本（近似值，避免基准依赖 Jinja）。"""
    return [
        f"<|im_start|>user\n问题{i}：机器学习需要多少数据？😀<|im_end|>\n"
        f"<|im_start|>assistant\n回答{i}：越多越好，但质量优先，中文语料尤其如此。<|im_end|>\n"
        for i in range(n)
    ]


def bench_hf_single(hf, texts: List[str]) -> float:
    """口径 B：Python 侧逐条调用（naive 基线，仅 tokenize）。"""
    t0 = time.perf_counter()
    for t in texts:
        hf.encode(t, add_special_tokens=False)
    return time.perf_counter() - t0


def bench_hf_batch(hf, texts: List[str]) -> float:
    """口径 B′：HF 原生批量（仅 tokenize）—— 把"批处理本身的收益"单独量出来。

    与 B 的差值 = `encode_batch` 相对逐条调用的收益（HF fast 引擎内部自并行）；
    与 C 的差值 = 渲染 / pack / 压缩 / 写盘 / 索引这些 Rust 阶段的开销。
    """
    t0 = time.perf_counter()
    hf.encode_batch(texts, add_special_tokens=False)
    return time.perf_counter() - t0


def bench_build(builder_kwargs: dict, jsonl: str, out_dir: str, docs: int) -> float:
    """口径 A / C：走完整 megadata 流水线（渲染 / pack / 压缩 / 写盘 / 索引）。"""
    from megadata import DataBuilder

    builder = DataBuilder(max_seq_len=1024, **builder_kwargs)
    t0 = time.perf_counter()
    builder.build_dataset(
        input_paths=[jsonl],
        output_dir=out_dir,
        mode="sft",
        template_str=CHAT_TEMPLATE,
        batch_size=8,
        pad_id=0,
    )
    dt = time.perf_counter() - t0
    print(f"    → {docs} docs / {dt:.2f}s = {docs / dt:,.0f} docs/s")
    return dt


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="megadata 第三方分词器吞吐三向对照基准")
    ap.add_argument("--hf-model", required=True, help="HF tokenizer.json（或所在目录）")
    ap.add_argument("--bbpe-model", default=None, help="等价的 bbpe 模型文件（口径 A 需要）")
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--out", default=None, help="产物目录（默认临时目录）")
    args = ap.parse_args(argv)

    from megadata import HfTokenizerAdapter
    from tokenizers import Tokenizer

    tmp = args.out or tempfile.mkdtemp(prefix="megadata_tokbench_")
    os.makedirs(tmp, exist_ok=True)
    jsonl = os.path.join(tmp, "bench.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write("\n".join(make_corpus(args.docs)) + "\n")

    print(f"语料：{args.docs} docs → {jsonl}")
    rows = []

    if args.bbpe_model:
        print("[A] bbpe 基线（Rust 直连，完整流水线）")
        rows.append(("A bbpe 基线", bench_build(
            {"tokenizer_path": args.bbpe_model},
            jsonl, os.path.join(tmp, "a_bbpe"), args.docs,
        )))
    else:
        print("[A] 跳过（未给 --bbpe-model）")

    print("[B] HF 单条循环（仅 tokenize，不含渲染/pack/写盘）")
    hf_file = args.hf_model if os.path.isfile(args.hf_model) \
        else os.path.join(args.hf_model, "tokenizer.json")
    hf = Tokenizer.from_file(hf_file)
    dt_b = bench_hf_single(hf, rendered_texts(args.docs))
    print(f"    → {args.docs} docs / {dt_b:.2f}s = {args.docs / dt_b:,.0f} docs/s")
    rows.append(("B HF 单条", dt_b))

    print("[B′] HF 批量 encode_batch（仅 tokenize，诊断用）")
    dt_b2 = bench_hf_batch(hf, rendered_texts(args.docs))
    print(f"    → {args.docs} docs / {dt_b2:.2f}s = {args.docs / dt_b2:,.0f} docs/s")
    rows.append(("B′ HF 批量", dt_b2))

    print("[C] megadata + HfTokenizerAdapter 批量（完整流水线）")
    rows.append(("C 本方案", bench_build(
        {"tokenizer": HfTokenizerAdapter.from_file(args.hf_model)},
        jsonl, os.path.join(tmp, "c_python"), args.docs,
    )))

    print("\n=== 汇总（越小越好）===")
    base = next((dt for name, dt in rows if name.startswith("A")), None)
    for name, dt in rows:
        ratio = f"{dt / base:.2f}x" if base else "-"
        print(f"  {name:<12} {dt:8.2f}s   相对 A：{ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

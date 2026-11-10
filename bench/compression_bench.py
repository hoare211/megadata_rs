#!/usr/bin/env python
"""
zstd 压缩 benchmark：决定 PT 裸流（Zstd-MDS）压缩默认值。

落地前必须用真实训练语料 benchmark，不要预设 30-50%。本脚本用
**真实训练语料 + 真实 tokenizer 的预分词 token 流**测量：

数据源（优先级递减）：
1. 真实语料（可选）：Megatron-core 格式 idx/bin（int32/int16/int64 LE），
   由 `--corpus-bin/--corpus-idx` 或环境变量 `MEGADATA_BENCH_CORPUS_BIN/IDX` 指定；
2. 合成对照：均匀随机 / Zipf(1.07) 的 151k 词表 token 流，
   用于校验脚本与最坏情况上界。

测量维度（模拟 Zstd-MDS 的 per-batch 独立 frame 设计）：
- 压缩率（compressed / original）
- 压缩吞吐（构建侧，MB/s）
- 解压吞吐（训练热路径顺序读，MB/s）
- 随机访问延迟（随机 seek 单 batch：裸流 vs 解压单帧，µs/batch）
- 参照组：整 shard 单帧压缩率（上界参考，我们不用它：牺牲随机寻址）

用法：
    python bench/compression_bench.py [--tokens 4000000] [--batch-size 8] [--seq-len 2048]
"""
import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import zstandard as zstd
except ImportError:
    sys.exit("缺少 zstandard 包：uv pip install zstandard")

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_BIN = Path(os.environ.get("MEGADATA_BENCH_CORPUS_BIN", ""))
CORPUS_IDX = Path(os.environ.get("MEGADATA_BENCH_CORPUS_IDX", ""))

QWEN_VOCAB = 151_936  # Qwen2 tokenizer vocab_size
LEVELS = [1, 3, 7, 12, 19]


def load_corpus_tokens() -> np.ndarray:
    """读 Megatron-core 格式 idx/bin，返回全部 token（int32 → int32 原样）。"""
    with open(CORPUS_IDX, "rb") as f:
        assert f.read(9) == b"MMIDIDX\x00\x00", "不是 Megatron idx 格式"
        (version,) = struct.unpack("<Q", f.read(8))
        assert version == 1
        (dtype_code,) = struct.unpack("<B", f.read(1))
        dtype = {3: np.int16, 4: np.int32, 5: np.int64}[dtype_code]
        (seq_count,) = struct.unpack("<Q", f.read(8))
        (doc_count,) = struct.unpack("<Q", f.read(8))
        lengths = np.frombuffer(f.read(4 * seq_count), dtype=np.int32)
    tokens = np.memmap(CORPUS_BIN, dtype=dtype, mode="r")
    n = int(lengths.sum())
    assert n <= len(tokens), f"idx 声称 {n} tokens > bin 实际 {len(tokens)}"
    print(f"[corpus] {seq_count} 序列 / {n:,} tokens / dtype={np.dtype(dtype).name}")
    return np.asarray(tokens[:n], dtype=np.int32)


def synthesize_tokens(n: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if kind == "uniform":
        return rng.integers(0, QWEN_VOCAB, size=n, dtype=np.int32)
    if kind == "zipf":  # 附录 A 口径：Zipf(1.07) 词典分布
        ranks = np.minimum((rng.zipf(1.07, size=n) - 1), QWEN_VOCAB - 1)
        perm = rng.permutation(QWEN_VOCAB)  # 打散 rank→id 映射，贴近真实词频序
        return perm[ranks].astype(np.int32)
    raise ValueError(kind)


def chunk_batches(tokens: np.ndarray, batch_size: int, seq_len: int, eos_id: int = 151643):
    """模拟 GreedyPacker 落盘：token 流切成 batch_size×seq_len 的定宽 batch，
    末尾用 EOS 补齐。返回 [n_batches, batch_size*seq_len] 的 int32 矩阵视图（copy）。"""
    capacity = batch_size * seq_len
    n_full = len(tokens) // capacity
    body = tokens[: n_full * capacity].reshape(n_full, capacity)
    batches = list(body)
    rest = tokens[n_full * capacity:]
    if len(rest) > 0:
        tail = np.full(capacity, eos_id, dtype=np.int32)
        tail[: len(rest)] = rest
        batches.append(tail)
    return batches


def bench_dataset(name: str, batches, levels=LEVELS, random_probes: int = 1000):
    raw_per_batch = [b.tobytes() for b in batches]
    raw_total = sum(len(x) for x in raw_per_batch)
    raw_blob = b"".join(raw_per_batch)
    print(f"\n## {name}")
    print(f"batches={len(batches)}  batch_bytes={len(raw_per_batch[0]):,}  raw_total={raw_total / 1e6:.1f}MB")
    print(f"| 配置 | 压缩后 | 压缩率 | 压缩吞吐 | 解压吞吐 | 随机访问 µs/batch |")
    print(f"|---|---|---|---|---|---|")

    # —— 未压缩基线 ——
    t0 = time.perf_counter()
    seq_reads = [raw_blob[i * len(raw_per_batch[0]): (i + 1) * len(raw_per_batch[0])] for i in range(0, len(batches))]
    seq_s = time.perf_counter() - t0
    idxs = np.random.default_rng(0).integers(0, len(batches), size=random_probes)
    stride = len(raw_per_batch[0])
    t0 = time.perf_counter()
    for i in idxs:
        _ = raw_blob[i * stride:(i + 1) * stride]
    rand_s = time.perf_counter() - t0
    print(f"| 裸流（不压缩） | {raw_total / 1e6:.1f}MB | 100% | "
          f"{raw_total / 1e6 / seq_s:.0f}MB/s | {raw_total / 1e6 / seq_s:.0f}MB/s | "
          f"{rand_s / random_probes * 1e6:.1f} |")

    # —— per-batch 独立 frame（Zstd-MDS 设计）——
    results = {}
    for level in levels:
        cctx = zstd.ZstdCompressor(level=level)
        dctx = zstd.ZstdDecompressor()
        t0 = time.perf_counter()
        frames = [cctx.compress(x) for x in raw_per_batch]
        comp_s = time.perf_counter() - t0
        comp_total = sum(len(f) for f in frames)

        t0 = time.perf_counter()
        for f in frames:
            dctx.decompress(f, max_output_size=len(raw_per_batch[0]))
        decomp_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in idxs:
            dctx.decompress(frames[i], max_output_size=len(raw_per_batch[0]))
        rand_s = time.perf_counter() - t0

        ratio = comp_total / raw_total * 100
        results[level] = (comp_total, ratio)
        print(f"| zstd:{level} (per-batch frame) | {comp_total / 1e6:.1f}MB | {ratio:.1f}% | "
              f"{raw_total / 1e6 / comp_s:.0f}MB/s | {raw_total / 1e6 / decomp_s:.0f}MB/s | "
              f"{rand_s / random_probes * 1e6:.1f} |")

    # —— 参照：整 shard 单帧（64 batches；牺牲随机寻址，仅作上界参考）——
    shard_size = 64
    cctx = zstd.ZstdCompressor(level=3)
    whole = []
    for s in range(0, len(raw_per_batch), shard_size):
        whole.append(cctx.compress(b"".join(raw_per_batch[s:s + shard_size])))
    whole_total = sum(len(w) for w in whole)
    print(f"| 参照: zstd:3 整shard单帧 | {whole_total / 1e6:.1f}MB | {whole_total / raw_total * 100:.1f}% | "
          f"— | — | 不可随机寻址 |")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4_000_000, help="每种数据集使用的 token 数")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--corpus-bin", default=os.environ.get("MEGADATA_BENCH_CORPUS_BIN", ""),
                    help="可选：Megatron-core 格式 .bin（真实语料 token 流）")
    ap.add_argument("--corpus-idx", default=os.environ.get("MEGADATA_BENCH_CORPUS_IDX", ""),
                    help="可选：与 --corpus-bin 配对的 .idx")
    args = ap.parse_args()

    global CORPUS_BIN, CORPUS_IDX
    CORPUS_BIN = Path(args.corpus_bin) if args.corpus_bin else CORPUS_BIN
    CORPUS_IDX = Path(args.corpus_idx) if args.corpus_idx else CORPUS_IDX

    print(f"# zstd 压缩 benchmark（batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"per-batch frame 设计 = Zstd-MDS）")

    if CORPUS_BIN.exists() and CORPUS_IDX.exists():
        tokens = load_corpus_tokens()
        have = min(len(tokens), args.tokens)
        batches = chunk_batches(np.asarray(tokens[:have]), args.batch_size, args.seq_len)
        bench_dataset("真实语料（预分词 token 流）", batches)
    else:
        print("\n[跳过真实语料：未提供 --corpus-bin/--corpus-idx（或文件不存在）]")

    for kind in ("uniform", "zipf"):
        tokens = synthesize_tokens(args.tokens, kind)
        batches = chunk_batches(tokens, args.batch_size, args.seq_len)
        bench_dataset(f"合成对照：{kind}（151k 词表，{args.tokens:,} tokens）", batches)

    print(
        "\n结论判据（决定 PT 默认压缩）：真实语料下 zstd:3 若满足\n"
        "  ① 压缩后 ≤ 75%（节省 ≥25% 磁盘/对象存储流量）；\n"
        "  ② 解压吞吐 ≥ 500MB/s（6.5MB/batch 解压 < 13ms，远小于训练 step，\n"
        "     且可与 GPU 计算重叠）；\n"
        "  ③ 随机访问增量 ≤ 1ms/batch（varlen/seek 场景无感）；\n"
        "则 DataBuilder PT 模式默认 compression='zstd'，否则保持 None 并在文档给出建议值。"
    )


if __name__ == "__main__":
    main()

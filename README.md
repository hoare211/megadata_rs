# megadata_rs: The Ultimate LLM Data Pipeline Engine

**English** | [简体中文](README.zh-CN.md)

[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyO3](https://img.shields.io/badge/Binding-PyO3-red.svg)](https://github.com/PyO3/pyo3)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`megadata_rs` is a **Rust-native, zero-copy data pipeline engine** for LLM and
VLM training at cluster scale. It removes the usual causes of GPU starvation —
JSON parsing, the Python GIL, IPC copies between dataloader processes, and
disk I/O congestion — and moves samples from raw corpora to GPU tensors with
**zero memory copies** in between.

## Key Features

* **Dual packing engine**
  * **SFT mode**: native Jinja rendering of HuggingFace ChatML-style templates,
    a document-aware 2-D bin-packing algorithm, precise `-100` masking, and
    `cu_seqlens` output that plugs straight into FlashAttention-2 varlen
    (no cross-document attention leakage).
  * **PT mode**: greedy packing that concatenates pretraining corpora into a
    seamless token stream to maximise MFU. The stream never loses or duplicates
    a token; if your loop computes loss over the whole `(B, S)` tile, use
    `cu_seqlens` to cut the EOS padding at the tail, or run varlen attention.
  * ⚠️ **Long PT documents are split across batches.** A document longer than
    `batch_size × max_seq_len` is laid out over several consecutive batches, and
    each piece becomes its own `cu_seqlens` segment — no data is lost, but under
    varlen training the **cross-batch context is silently cut**. The builder
    aggregates a WARN and records `global_stats.split_docs` / `split_tokens` in
    `index.json`; the loader warns again. Mitigation: raise `max_seq_len` until
    it covers the vast majority of documents.

* **Zero-copy FFI**: a purpose-built **DLPack C ABI** plus the
  **Arrow C Data Interface** hand Rust heap buffers to PyTorch tensors in
  $O(1)$ time. No matter how large the batch, Python never copies a single byte.

* **Temperature mixer for multiple sources**: mount Code / Math / Web streams at
  runtime; **temperature scaling** samples them concurrently underneath so you
  can steer the knowledge mix smoothly instead of shuffling statically.

* **Deterministic sharding and second-level resumption (MMP protocol)**: no
  centralised RPC. Thousands of ranks derive shard ownership from a
  deterministic PRNG with round-robin assignment, so every rank sees the same
  reproducible shard sequence within an epoch. File offsets and the PRNG state
  are captured too, so a crashed job **resumes in milliseconds** and replays the
  exact cross-source sampling order.

* **Production control plane**
  * **Shard aggregation**: one shard file holds N batches (~128 MB/shard by
    default), cutting file count from "one per batch" to a few hundred; the
    loader hard-checks `shards >= world_size` so no rank idles.
  * **Integrity**: every shard is read back and sha256-ed into `index.json`.
    Cloud downloads are atomic (`.tmp` + rename), verified after download,
    re-downloaded on failure, retried with jittered exponential backoff,
    de-duplicated across processes (flock) and bounded by an LRU cache
    (`MEGADATA_CACHE_LIMIT`). **Local datasets are not verified by default**
    (zero overhead); opt in with `MEGADATA_VERIFY_LOCAL=1` or `=sample:P`
    (results cached by size+mtime, so each file is hashed once).
  * **Errors are never silent**: I/O, decompression and corruption failures raise
    `OSError` with the shard name and batch index instead of looking like
    "end of data" or hanging in an infinite retry. Local reads get **bounded
    retries** (2 attempts, 50/200 ms backoff). EOF and user cancellation still
    end normally, and the reader stays alive after EOF so resumption can seek.
  * **Prefetch-thread panics surface too**: the prefetch thread is wrapped in
    `catch_unwind`, so any panic (including lock poisoning) becomes a fatal
    error at the consumer rather than a truncated epoch.
  * **Optional compression**: PT raw streams use per-batch zstd frames
    (`Zstd-MDS`, preserving O(1) random access) — **on by default** (real-corpus
    benchmark: 42% of original size on a 151k vocab, 629 MB/s decompression,
    +103 µs random access; disable with `compression="none"`). SFT uses Arrow
    IPC's native zstd/lz4 (explicit).
  * ⚠️ Automatic `batches_per_shard` is computed on **uncompressed** bytes
    (PT+zstd lands at ~42% of that); plan disk capacity at ~0.42×.
  * **Prefetch window throttling**: `predownload` caps in-flight batches
    (default 16), smoothing I/O bursts and bounding prefetch memory.
  * **Permutation shuffle**: Rust builds a deterministic batch-level permutation
    from `(seed, epoch)`, shrinking `state_dict` from gigabytes of buffer to a
    few dozen bytes and making it reproducible across processes.
  * **Elastic scaling**: `relaxed` partitioning with `num_canonical_nodes`
    anchors, so global sample order is independent of cluster topology and a job
    can resume after a resize.
  * **3-D world topology**: node × rank × worker; with `num_workers > 0` each
    worker gets a disjoint shard subset (readers rebuilt lazily after fork).
  * **Drop-in ecosystem**: `StreamingDataset(IterableDataset)` and
    `MegadataDataLoader(DataLoader)` work with HF Trainer / Lightning as-is.

## Installation

Requires Rust 1.85+ (`edition = "2024"`), Python 3.8+ and `maturin`.
`bbpe_rs` (the tokenizer) is a regular crates.io dependency and is fetched
automatically — nothing to place by hand:

```bash
pip install maturin pytest torch pyarrow
maturin develop --release
```

To train or export your own tokenizer with `bbpe_tokenizer`:

```bash
pip install "git+https://github.com/hoare211/bbpe_rs@v0.1.0"
```

## Quick Start

### 1. Build a dataset offline (`DataBuilder`)

`DataBuilder` uses Tokio async I/O and a Rayon thread pool to swallow JSONL
corpora and write the on-disk format in one pass.

```python
from megadata import DataBuilder

builder = DataBuilder(tokenizer_path="./model_tokenizer", max_seq_len=8192)

# SFT chat dataset (written as Arrow IPC)
builder.build_dataset(
    input_paths=["data/sft/raw.jsonl"],
    output_dir="data/sft_processed",
    mode="sft",
    template_str="{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}",
    batch_size=16,
)

# PT pretraining dataset (Megatron-style raw-stream .bin/.idx)
builder.build_dataset(
    input_paths=["data/pt/raw.jsonl"],
    output_dir="data/pt_processed",
    mode="pt",
    batch_size=16,
    # optional control-plane knobs:
    # batches_per_shard=0   # 0 = auto (~128MB/shard); loader checks shards >= world_size
    # compression="zstd"    # PT default (per-batch frame, Zstd-MDS); "none" to disable
)
```

### 2. Multi-source zero-copy loading

```python
import torch
from megadata import MixBatchStream

# temperature-controlled multi-source stream (Code/Math/Web, curriculum, annealing)
loader = MixBatchStream(
    [
        {"path": "data/pt_processed/index.json", "weight": 0.7},
        {"path": "data/sft_processed/index.json", "weight": 0.3},
    ],
    batch_size=16,
    max_seq_len=8192,
    world_size=1024,
    rank=0,                  # DDP topology: no DistributedSampler needed
    steps_per_epoch=1000,    # required under DDP for a deterministic epoch length
)
loader.update_temperature(2.0)   # smooth the mix towards smaller sources

# resume (millisecond-level jump)
# loader.load_state_dict(checkpoint["megadata_state"])

for batch in loader:
    # tensors map Rust memory directly — zero copies
    input_ids = batch["input_ids"].cuda()

    loss = model(input_ids).loss
    loss.backward()

# exact snapshot of file cursors + PRNG state
# checkpoint["megadata_state"] = loader.state_dict()
```

> Single-source training loops should use `BatchStream`; random access for
> evaluation uses `RandomAccessBatchReader`; HF Trainer / Lightning integration
> uses `StreamingDataset` + `MegadataDataLoader`. See the usage guide.

## Documentation

- **[`docs/guide/usage.md`](docs/guide/usage.md)** — English quickstart: build → load → train → resume → tune → troubleshoot
- **[`docs/guide/使用指南.md`](docs/guide/使用指南.md)** — full reference guide (Chinese, most detailed)
- **[`MIGRATION.md`](MIGRATION.md)** — v1/v2 → v3 rename, snapshot and behaviour changes
- **[`docs/README.md`](docs/README.md)** — documentation map

## CI (GitHub Actions)

`.github/workflows/ci.yml` runs four jobs: `lint` (ruff over `python/ tests/ tools/`),
`docs` (`docs_lint.py check` + `self-test`, stdlib only), `rust-test`
(`cargo clippy --all-targets -p megadata_rs -- -D warnings` + `cargo test --lib`)
and `python-test` (maturin build + pytest). `bench.yml` runs throughput
benchmarks on demand; `nightly.yml` runs the full proptest fuzz suite nightly.

No repository variables are required: `bbpe_rs` comes from crates.io.

## Quality Defense

Four test dimensions back the claims above (see `tests/`):

1. **E2E parity** — the FlashAttention topology drops no document.
2. **Distributed integrity** — ranks and workers are strictly disjoint.
3. **Time travel** — after an interrupt/restart, byte-level content and the
   PRNG sampling order match 100%.
4. **Memory leaks** — high-frequency C FFI traffic and Python GC pressure leak
   nothing.

## Contributing

Issues and PRs are welcome — please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
(local self-check commands included). For security reports use the private
channel in [`SECURITY.md`](SECURITY.md).

## License

Released under the [MIT](LICENSE) license.

## Acknowledgements

Tokenization is provided by the sister project
[bbpe_rs](https://github.com/hoare211/bbpe_rs) (a Rust BPE tokenizer
trainer/encoder). `object_store`, `arrow`, `pyo3`, `tokio` and `rayon` form the
infrastructure this engine is built on.

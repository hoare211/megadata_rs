# megadata_rs — Usage Guide (English quickstart)

> **状态**：现行英文上手文档（随代码更新；完整参考见中文版 `使用指南.md`）。

A condensed, task-oriented walkthrough: install → build → load → train → resume →
tune → troubleshoot.
For the exhaustive reference (parameter-by-parameter rationale, edge cases,
benchmark tables) see the Chinese full guide [`使用指南.md`](使用指南.md).

- [1. Requirements and install](#1-requirements-and-install)
- [2. Five-minute quickstart](#2-five-minute-quickstart)
- [3. Building datasets (`DataBuilder`)](#3-building-datasets-databuilder)
- [4. Loading datasets](#4-loading-datasets)
- [5. Training integration](#5-training-integration)
- [6. Runtime behaviour and environment variables](#6-runtime-behaviour-and-environment-variables)
- [7. Capacity planning](#7-capacity-planning)
- [8. Troubleshooting](#8-troubleshooting)
- [9. API cheat sheet](#9-api-cheat-sheet)

---

## 1. Requirements and install

| Component | Requirement |
|---|---|
| Rust | 1.85+ (`edition = "2024"`) |
| Python | 3.8+ |
| Build tool | `maturin >= 1.0` |
| Runtime deps | `torch >= 2.0`, `pyarrow >= 14` |

`bbpe_rs` is a crates.io dependency and is pulled by `cargo` automatically.

```bash
# 1) virtualenv
uv venv --python 3.12 .venv && source .venv/bin/activate

# 2) optional: the tokenizer trainer/exporter (only needed to train/export vocab)
uv pip install "git+https://github.com/hoare211/bbpe_rs@v0.1.0"

# 3) build the extension
uv pip install maturin "torch>=2.0" "pyarrow>=14"
maturin develop --release

# 4) self-check
python -c "import megadata; print(megadata.__version__)"
```

Running from source without installing:

```bash
cargo build --features python-extension
cp target/debug/libmegadata_rs.so python/megadata/_rust.cpython-312-x86_64-linux-gnu.so
PYTHONPATH=$PWD/python python -c "import megadata; print(megadata.__version__)"
```

> The `.so` name must match your interpreter ABI tag (`python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`).

---

## 2. Five-minute quickstart

The repository ships two synthetic corpora (`data/pretrain_t2t_mini.jsonl`,
`data/sft_t2t_mini.jsonl`), so the flow below runs out of the box once you have a
tokenizer directory.

### 2.1 Build a PT dataset

```python
from megadata import DataBuilder

builder = DataBuilder(tokenizer_path="path/to/tokenizer", max_seq_len=512)
builder.build_dataset(
    input_paths=["data/pretrain_t2t_mini.jsonl"],
    output_dir="out/pt",
    mode="pt",
    batch_size=8,
    batches_per_shard=8,   # 0 = auto (~128MB/shard)
    compression="zstd",    # PT default is zstd (per-batch frame, Zstd-MDS)
)
```

### 2.2 Build an SFT dataset

```python
CHATML = (
    "{% for msg in messages %}<|im_start|>{{ msg.role }}\n"
    "{{ msg.content }}<|im_end|>\n{% endfor %}"
)

builder = DataBuilder(tokenizer_path="path/to/tokenizer", max_seq_len=512)
builder.build_dataset(
    input_paths=["data/sft_t2t_mini.jsonl"],
    output_dir="out/sft",
    mode="sft",
    template_str=CHATML,
    batch_size=8,
)
```

### 2.3 Read and train

```python
from megadata import BatchStream

loader = BatchStream("out/sft/index.json", batch_size=8, max_seq_len=512)
for batch in loader:
    ids = batch["input_ids"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    out = model(input_ids=ids, labels=labels, cu_seqlens=batch["cu_seqlens"])
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
```

### 2.4 Save and restore

```python
ckpt = {"model": model.state_dict(), "optim": opt.state_dict(),
        "dataloader": loader.state_dict(), "step": global_step}
torch.save(ckpt, f"ckpt_{global_step}.pt")

state = torch.load("ckpt_1000.pt")
model.load_state_dict(state["model"]); opt.load_state_dict(state["optim"])
loader.load_state_dict(state["dataloader"])   # seeks to the exact batch, O(1)
```

---

## 3. Building datasets (`DataBuilder`)

### 3.1 Input format

| Mode | One JSON object per line | Notes |
|---|---|---|
| `mode="pt"` | `{"text": "..."}` | tokenized, then greedily packed into one flat stream (documents may share a row) |
| `mode="sft"` | `{"conversations": [{"role", "content"}, ...]}` or `{"messages": [...]}` | rendered through `template_str`, then encoded; a `tools` field is passed to the template |

### 3.2 Recommended: `BuildConfig`

```python
from megadata import DataBuilder
from megadata.builder import BuildConfig, Marker

cfg = BuildConfig(
    input_paths=["raw.jsonl"],
    output_dir="out/sft",
    mode="sft",                 # "pt" | "sft" — typos fail at construction time
    template_str="...",         # required for SFT (minijinja template)
    batch_size=16,              # ★ baked into shards: the loader must match
    pad_id=0,
    eos=Marker(),               # PT only (defaults to "<|endoftext|>" derivation)
    turn_start=Marker(),        # SFT only (default "<|im_start|>", must be exactly 1 token)
    turn_end=Marker(),
    assistant_str="assistant",
    batches_per_shard=0,        # 0 = auto (~128MB/shard)
    compression="auto",         # "auto" | "zstd" | "lz4" | "none"
    extra_columns=False,        # document-level side columns doc_ids / loss_weights
)
DataBuilder("path/to/tokenizer", max_seq_len=8192).build_dataset(cfg)
```

Key rules:

- **`batch_size` / `max_seq_len` are frozen into the shards.** Loading with
  different values raises instead of silently mis-reading.
- **Mode relevance is validated at construction time**: passing `eos=` to
  `mode="sft"`, or `turn_start=` to `mode="pt"`, raises `ValueError` rather than
  being ignored.
- **`Marker(text=None, id=None, strict=True)`**: `id` is derived from `text` when
  omitted. `strict=True` (turn markers) requires exactly one token; `strict=False`
  (EOS default) warns and takes the first token. Rationale: a wrong EOS only
  moves a document boundary, a wrong turn marker corrupts supervision labels.
- Mixing new and legacy forms (`eos=` together with `eos_str=`/`eos_id=`) is an
  error — no silent precedence.
- `progress_cb` lives on `build_dataset`, not in `BuildConfig` (config feeds the
  manifest fingerprint; callbacks are runtime side effects).

Machine-readable knowledge belongs in `index.json`: per-shard sha256, batch
counts, token counts and `global_stats` (including PT `split_docs`/`split_tokens`).

### 3.3 Side columns: `doc_ids` / `loss_weights`

With `extra_columns=True` (PT), each batch carries two extra zero-copy tensors:

- `doc_ids`: `(B, S)` int32 — which document a token belongs to, derived from
  `cu_seqlens` (useful for detecting documents split across batches).
- `loss_weights`: `(B, S)` float16 — per-token loss weights.

### 3.4 Third-party tokenizers

`tokenizer_path` accepts more than bbpe models:

- a bbpe_rs model directory / model file (default, native Rust path);
- a HuggingFace `tokenizer.json` (or a directory containing it) — routed through
  `megadata.tokenizer_adapter` (`HfTokenizerAdapter`, `load_tokenizer_file`,
  `adapt_tokenizer`).

Both paths are validated up front: the error message names **both** stages when
neither accepts the file.

### 3.5 Progress and cancellation

```python
builder.build_dataset(cfg, progress_cb=lambda p: print(p["docs_done"]))
```

The callback fires roughly every 100 ms. `Ctrl+C` during a build cancels through
the same signal plumbing as the loaders (see §5.5).

---

## 4. Loading datasets

### 4.1 Which entry point?

```
Dataset already built — need to read it?
├─ Random access by index (eval / debug / spot checks)  → RandomAccessBatchReader
├─ Main training loop (recommended)                     → BatchStream
│   ├─ shuffled (default)                                → shuffle="permutation"
│   ├─ no batch-level permutation (debug)                → shuffle="none"
│   └─ strict global shard order (token-by-token audit)  → shuffle="none" + shard_order="sequential"
├─ Multi-source mix / curriculum / annealing             → MixBatchStream
└─ HF Trainer / Lightning takes over                     → StreamingDataset + MegadataDataLoader
```

### 4.2 Common parameters (shared by all loaders)

| Parameter | Default | Meaning |
|---|---|---|
| `index_json_path` | — | path to `index.json`; local, `http(s)://`, `s3://`, `gs://` |
| `batch_size` | from manifest | must match the build-time value |
| `max_seq_len` | from manifest | must match the build-time value |
| `world_size` / `rank` | `None` | fall back to `torch.distributed`, then `WORLD_SIZE`/`RANK` (warns once); `MEGADATA_STRICT_TOPOLOGY=1` forbids the fallback |
| `seed` | `42` | drives shard shuffling and the permutation table; **must match on resume** |
| `shuffle` | `"permutation"` | `"permutation"` or `"none"` (turns off batch-level permutation only; shard order still follows `(seed, epoch)`). `"buffer"` was removed in v3 |
| `partition` | `"round_robin"` | or `"relaxed"` for elastic resizing |
| `shard_order` | `"random"` | or `"sequential"` for a stable global shard order; orthogonal to `shuffle` |
| `num_canonical_nodes` | `None` | `relaxed` only; must be repeated explicitly on resume |
| `predownload` | `16` (Rust default) | in-flight prefetch cap; values `< 4` warn |
| `epoch` | `0` | starting epoch; `load_state_dict` overrides it from the snapshot |

`len(loader)` is "how many batches the *next* iteration will actually yield" —
after `load_state_dict` it is `epoch_length - resume_position`, which keeps a
`len()`-driven LR scheduler aligned with real steps. One `for` loop == exactly
one epoch (epoch counting belongs to your training script).

### 4.3 Batch layout

| Mode | Keys | Notes |
|---|---|---|
| PT | `input_ids`, `cu_seqlens`, `format` | `input_ids`: `(B, S)` int32 tensor via DLPack; `cu_seqlens`: int32 tensor, length = documents + 1, covering the whole batch; `format="pt"` |
| SFT | `input_ids`, `labels`, `position_ids`, `cu_seqlens`, `row_lens`, `format` | `labels` are `-100` on unsupervised positions; `cu_seqlens` are logical offsets (documents never straddle rows); `row_lens` is the valid length per row |

All paths (single source, homogeneous mix, heterogeneous mix) carry the `format`
key, so `mask_adapter` can pick the right packer via `packer=batch.get("format")`.

### 4.4 Distributed and elastic scaling

- Hand `world_size`/`rank` to megadata and **do not wrap it in
  `torch.utils.data.DistributedSampler`** — that would shard twice.
- `MixBatchStream` under DDP needs `steps_per_epoch`; it performs a
  communication-free static check at construction and exposes
  `verify_ddp_consistency()` for a definitive (collective) answer.
- For elastic resizing use `partition="relaxed"` plus a fixed
  `num_canonical_nodes`; keep both identical when resuming.
- With `num_workers > 0` use `StreamingDataset` + `MegadataDataLoader`; each
  worker gets a disjoint shard subset automatically.

### 4.5 HF Trainer / Lightning

```python
from megadata import StreamingDataset, MegadataDataLoader

ds = StreamingDataset("out/sft/index.json", batch_size=16, max_seq_len=8192)
dl = MegadataDataLoader(ds, num_workers=4)
trainer.fit(model, train_dataloaders=dl)
```

> Iterate through `MegadataDataLoader`. Running `for b in ds:` directly bypasses
> epoch advancement, so every epoch would replay the same order.

---

## 5. Training integration

### 5.1 Why not wrap it in `torch.utils.data.DataLoader`?

The engine already does sharding, permutation, prefetching and zero-copy
transfer. A `DataLoader` on top adds process hops and copies, and its
`DistributedSampler` would double-shard. Use `BatchStream` / `MixBatchStream`
directly in the training loop, or the Tier-1 wrappers when a framework insists
on the `DataLoader` protocol.

### 5.2 DDP loop

```python
import os, torch
from megadata import BatchStream

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

loader = BatchStream("out/sft/index.json", batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                     world_size=world_size, rank=rank, seed=42)

for epoch in range(num_epochs):
    for batch in loader:
        input_ids = batch["input_ids"].to("cuda", non_blocking=True)
        labels = batch["labels"].to("cuda", non_blocking=True)
        out = model(input_ids=input_ids, labels=labels, cu_seqlens=batch["cu_seqlens"])
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
```

### 5.3 Checkpoint and resume

| Loader | Snapshot contents | Restore semantics |
|---|---|---|
| `BatchStream` | `{schema, seed, epoch, in_epoch_pos, manifest_fingerprint, topology}` | permutation recomputed from `(seed, epoch)` + `seek_to_epoch_batch(...)`, O(1) |
| `MixBatchStream` | the above + `{weights, base_weights, temperature, steps_yielded, sub_stream_cursors, rng_word_pos}` | per-substream seek + PRNG jump to `word_pos`, O(1); weights/temperature restored too |
| `StreamingDataset` / `MegadataDataLoader` | engine snapshot + `{num_batches_yielded, dataset, tier1, resume_granularity}` | rebuild the loader from the snapshot; `resume_granularity="epoch"` means multi-worker resumes replay from the epoch start |

Rules that matter:

1. Resume with the **same `seed`, `world_size`/`rank`, `partition`** as the
   original run (use `relaxed` + `num_canonical_nodes` for resizes).
2. After `load_state_dict`, the very next `next()` is the batch after the
   checkpoint — stale prefetches are discarded.
3. The snapshot overrides the constructor's `epoch=`; the restored epoch finishes
   and then advances to `epoch + 1`.
4. Optimizer / AMP state is yours to save; it is independent.
5. **`break` differs between layers**: breaking out of an engine-level loop does
   *not* advance the epoch (the next iteration replays the epoch from batch 0);
   the Tier-1 `MegadataDataLoader` teardown advances unconditionally. Use
   `state_dict()` + `load_state_dict()` when you need exact "continue where I
   stopped" semantics.

### 5.4 Interrupt and cancel

- On first instance construction, megadata installs idempotent SIGINT/SIGTERM
  handlers that chain to the previous handler; `MEGADATA_DISABLE_SIGNAL_HANDLER=1`
  disables this.
- `Ctrl+C` while blocked on I/O exits within one batch period and raises
  `MegadataInterrupted` (a `KeyboardInterrupt` subclass).
- Explicit cancellation: `loader.cancel()`; `close()` is terminal, `release()`
  frees only the current read head.

---

## 6. Runtime behaviour and environment variables

| Situation | Behaviour |
|---|---|
| Epoch finished | iteration ends normally (`StopIteration`) |
| User cancel (`cancel()` / `Ctrl+C`) | `MegadataInterrupted`; cancellation is terminal |
| Poll timeout, no data yet | retried internally, no exception |
| **I/O / decompression / corruption** | `OSError` including shard name and batch index |

Local reads retry twice (50 ms / 200 ms backoff) before reporting; cloud reads use
three jittered exponential retries (500 ms base) plus atomic write and sha256
verification. Mismatched `batch_size`/`max_seq_len`, `shards < world_size`, and
unsupported `index.json` versions all fail **at loader construction**, not
mid-training.

| Variable | Default | Purpose |
|---|---|---|
| `MEGADATA_CACHE_DIR` | `/tmp/megadata_cache` | cloud file cache directory |
| `MEGADATA_CACHE_LIMIT` | `256GiB` | LRU bound (binary units, `0` = unlimited); eviction uses **mtime** |
| `MEGADATA_VALIDATE_CACHE` | off | also sha256 cache hits (bit-rot defence) |
| `MEGADATA_VERIFY_LOCAL` | off | `1` = verify local datasets, `sample:P` = deterministic sampling |
| `MEGADATA_DISABLE_SIGNAL_HANDLER` | off | leave signals to your framework |
| `WORLD_SIZE` / `RANK` | — | topology fallback (warns once) |
| `MEGADATA_STRICT_TOPOLOGY` | off | `1` = forbid the env fallback |
| `MEGADATA_SKIP_DDP_CHECK` | off | skip the `MixBatchStream` DDP length check |
| `BBPE_TOKENIZER_DIR` | — | tokenizer directory used by the test suite |

---

## 7. Capacity planning

`python -m megadata.simulation` replays the real read order (same partitioning and
permutation functions as the reader) to model **download bandwidth → node cache →
starvation**, answering "what should I set these to?" without running training.

```bash
# one pass
python -m megadata.simulation out/pt/index.json --world-size 64 --num-nodes 8

# sweep per-node cache = 2/4/8/16/32 shards and print a comparison table
python -m megadata.simulation out/pt/index.json --world-size 64 --num-nodes 8 --sweep-cache
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `shards (N) < world_size (M)` | aggregation too coarse / `world_size` too large | rebuild with smaller `batches_per_shard`, or lower `world_size` |
| `batch_size mismatch: built X, loading Y` | loader disagrees with the build | share one `BATCH_SIZE` constant |
| `max_seq_len mismatch` | same | same |
| `index.json version unsupported` | dataset built by an older version | rebuild with the current version |
| SFT build ends with `total_shards=0` / "no shard records" | empty `template_str` produced empty renders | pass the chat template (a `UserWarning` is emitted) |
| `Build produced 0 shards (docs_read=…, docs_tokenized=…)` | every row filtered: template variables do not match the data keys, empty input, or PT rows missing `text` | align template and data keys; ensure inputs are non-empty |
| `Checksum failure (local): …` | local dataset corrupted and `MEGADATA_VERIFY_LOCAL` is on | re-copy or rebuild the dataset |
| Build warns `SFT truncation summary` / `truncated_docs>0` | conversations exceed `max_seq_len` and were tail-truncated | raise `max_seq_len` or clean overly long conversations |
| Build warns `Turn-marker injection summary` | turn-marker text embedded in user content leaked into supervision | clean/escape the data; check `docs_flagged_injection` |
| Mid-iteration `OSError: … shard … read failed` | truncated shard / disk corruption / half cache | repair or re-download; enable `MEGADATA_VALIDATE_CACHE=1` |
| Training stalls, GPU idle | prefetch/cache pressure | tune with `python -m megadata.simulation … --sweep-cache`; ensure `predownload >= 16` |
| `len(loader)` smaller than expected after resume | by design: it counts remaining batches | use it directly for the LR scheduler |
| `KeyError: 'labels'` in a mixed loop | heterogeneous schemas | branch on `"labels" in batch` or `batch["format"] == "sft"` |
| `update_temperature` raises `ValueError` | a dataset has `global_stats.total_tokens == 0`, or invalid weights | set explicit weights / call `update_weights`, or rebuild with token stats |
| Order differs after resume | `seed` / `world_size` / `partition` changed | keep them identical, or use `relaxed` + `num_canonical_nodes` |
| Single-node job sharded unexpectedly | stray `WORLD_SIZE`/`RANK` in the environment | unset them, or set `MEGADATA_STRICT_TOPOLOGY=1` |
| `Ctrl+C` does nothing | your framework also traps signals | set `MEGADATA_DISABLE_SIGNAL_HANDLER=1` |

---

## 9. API cheat sheet

```python
from megadata import (
    DataBuilder, BuildConfig, Marker,          # build
    BatchStream, MixBatchStream,               # engine-level streams
    RandomAccessBatchReader,                   # random access
    StreamingDataset, MegadataDataLoader,      # Tier-1 (torch)
    to_flash_varlen, flatten_and_unpad,        # attention adapters
)

# build
DataBuilder(tokenizer_path, max_seq_len=8192).build_dataset(config_or_kwargs)

# engine streams (path first, everything else keyword-only)
BatchStream(index_json_path, *, batch_size=None, max_seq_len=None,
            world_size=None, rank=None, seed=42, shuffle="permutation",
            partition="round_robin", shard_order="random",
            num_canonical_nodes=None, predownload=None, epoch=0)

MixBatchStream(configs, *, weights=None, steps_per_epoch=None, batch_size=None,
               max_seq_len=None, world_size=None, rank=None, seed=42,
               shuffle=True, partition="round_robin", ...)

# Tier-1
StreamingDataset(index_json_path, *, batch_size=None, max_seq_len=None, ...)
MegadataDataLoader(dataset, num_workers=0, ...)

# lifecycle: close() is terminal, release() only drops the current read head
# snapshots: loader.state_dict() / loader.load_state_dict(state)
```

Capacity planning (`megadata.simulation`) and dataset validation
(`megadata.validate`, CLI `python -m megadata.validate`) are submodules and are
deliberately **not** exported at the package top level.

---

For deeper detail — parameter rationale, `cu_seqlens`/attention backend mapping,
benchmark methodology and the full FAQ — read the Chinese full guide:
[`使用指南.md`](使用指南.md).

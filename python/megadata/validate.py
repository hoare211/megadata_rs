# python/megadata/validate.py
"""数据集自检入口（F-32）：把"常见陷阱速查"变成可程序化的断言。

用法::

    from megadata.validate import validate
    report = validate("out/pt/index.json", world_size=8)
    assert report["ok"], report["checks"]

CLI::

    python -m megadata.validate out/pt/index.json --world-size 8

检查项（静态、不读 shard 数据，对应 USAGE.md 的"常见陷阱速查"）：

- manifest 可读 + 版本受支持；
- ``global_stats.total_tokens > 0``（空数据集/构建失败检测）；
- ``batch_size`` / ``max_seq_len`` 已记录且与期望一致（如传入）；
- shard 数 >= ``world_size``（传入时；不足会在加载端被拒，高位 rank 无数据）；
- ``total_batches == Σ shards.num_batches``（manifest 被截断/手改检测）；
- shard 文件与**有 hash 记录的**旁路文件（``.cu`` / ``.idx``）存在性（本地路径）；
- ``tokenizer`` 指纹块（可选）提示。

**深度校验**（``deep=True`` / CLI ``--deep``，仅本地路径）：抽 2 个 shard 做内容级抽查——

- ``.bin`` / ``.cu`` / ``.idx`` 的 sha256 与 manifest 记录比对（有记录时）；
- PT Memmap：解析 ``.cu``（length-prefixed 记录），校验记录数 == ``num_batches``、
  cu 首值为 0 / 非递减 / 末值 <= ``batch_size * max_seq_len``；``.idx`` 单调不减
  且条数 == ``num_batches + 1``；
- ArrowIpc：pyarrow 读首个 batch，校验 batch 级列（cu_seqlens / row_lens）
  契约（与 Rust 读取侧 ``validate_arrow_batch_level_columns`` 同口径）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from megadata._rust import read_manifest_json

try:
    # 版本白名单的**唯一事实源**在 Rust 写入端（src/builder/metadata.rs）——
    # 从这里读，改写入端后不会与 Python 侧手抄的第二份静默漂移。
    from megadata._rust import SUPPORTED_MANIFEST_VERSIONS as _SUPPORTED_MANIFEST_VERSIONS
except ImportError:  # pragma: no cover - 旧扩展兜底（重新构建后不再走到这里）
    _SUPPORTED_MANIFEST_VERSIONS = ("1.0", "2.0")

__all__ = ["validate", "main"]


def _check(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail}


def validate(index_path: str, *,
             world_size: Optional[int] = None,
             expect_batch_size: Optional[int] = None,
             expect_max_seq_len: Optional[int] = None,
             deep: bool = False) -> Dict[str, Any]:
    """对 ``index.json`` 做静态自检，返回结构化报告（不抛异常）。

    Returns:
        ``{"ok": bool, "index_path": str, "checks": [{"check","ok","detail"}, ...]}``
    """
    checks: List[Dict[str, Any]] = []
    path = str(index_path)
    try:
        manifest = json.loads(read_manifest_json(path))
    except Exception as e:
        return {"ok": False, "index_path": path,
                "checks": [_check("manifest readable", False, f"{type(e).__name__}: {e}")]}

    checks.append(_check("manifest readable", True, path))
    version = str(manifest.get("version", ""))
    checks.append(_check("version supported", version in set(_SUPPORTED_MANIFEST_VERSIONS),
                         f"version={version!r} (supported: {list(_SUPPORTED_MANIFEST_VERSIONS)})"))

    stats = manifest.get("global_stats", {}) or {}
    dt = stats.get("total_tokens")
    checks.append(_check("total_tokens > 0", (dt or 0) > 0,
                         f"total_tokens={dt} (0 will make the Mixer fail on token-based mixing)"))

    bs = int(stats.get("batch_size") or 0)
    checks.append(_check("batch_size recorded", bs > 0, f"batch_size={bs} (may be missing in old datasets)"))
    if expect_batch_size is not None:
        checks.append(_check("batch_size matches expectation", bs == int(expect_batch_size),
                             f"manifest={bs}, expected={expect_batch_size}"))
    msl = int(manifest.get("max_seq_len") or 0)
    checks.append(_check("max_seq_len recorded", msl > 0, f"max_seq_len={msl}"))
    if expect_max_seq_len is not None:
        checks.append(_check("max_seq_len matches expectation", msl == int(expect_max_seq_len),
                             f"manifest={msl}, expected={expect_max_seq_len}"))

    # ★ 评审 §3.5（F2-PT）：把"PT 长文档被跨 batch 切分"变成 validate 可见项。
    #   该问题不报错、但会让 varlen 训练静默丢掉跨 batch 上下文，属"必须在训练前
    #   看到"的类别，因此作为**告警级**检查项（非失败）出现在报告中。
    #   与 `truncated_docs`（SFT 丢数据）严格区分：那是数据损失，这是语义变化。
    split_docs = int(stats.get("split_docs") or 0)
    if split_docs > 0:
        split_tokens = int(stats.get("split_tokens") or 0)
        avg = split_tokens / split_docs if split_docs else 0
        checks.append(_check(
            "PT long docs fit in a single batch", False,
            f"{split_docs} doc(s) (avg {avg:.0f} tokens) exceed batch_size×max_seq_len "
            f"({bs}×{msl}={bs * msl}) and were split across batches — NO data was dropped, but "
            f"FlashAttention varlen treats each segment as an independent sequence (position "
            f"restarts at 0), so cross-batch context inside one document is LOST. "
            f"Raise max_seq_len, or accept it and use standard attention instead of varlen."))
    truncated_docs = int(stats.get("truncated_docs") or 0)
    if truncated_docs > 0:
        truncated_tokens = int(stats.get("truncated_tokens") or 0)
        checks.append(_check(
            "SFT truncation free", False,
            f"{truncated_docs} doc(s) were tail-truncated ({truncated_tokens} tokens DROPPED). "
            f"Raise max_seq_len to avoid data loss."))

    shards = manifest.get("shards") or []
    checks.append(_check("shard records present", len(shards) > 0, f"shards={len(shards)}"))
    if world_size is not None:
        checks.append(_check("shard count >= world_size", len(shards) >= int(world_size),
                             f"shards={len(shards)} world_size={world_size} (the loader rejects it when insufficient)"))

    total_batches = sum(int(s.get("num_batches") or 0) for s in shards)
    recorded = int(stats.get("total_batches") or 0)
    checks.append(_check("total_batches matches shard sum", recorded in (0, total_batches),
                         f"recorded={recorded} Σshards={total_batches} (a mismatch means the manifest was truncated or hand-edited)"))

    is_local = not path.startswith(("http://", "https://", "s3://", "gs://"))
    if is_local and shards:
        base = os.path.dirname(path)
        missing: List[str] = []
        for s in shards:
            fn = s.get("filename")
            if not fn:
                continue
            if not os.path.exists(os.path.join(base, fn)):
                missing.append(fn)
            hashes = s.get("hashes") or {}
            # 旁路文件按 stem 命名：shard_00000.bin → shard_00000.cu / shard_00000.idx
            stem = os.path.splitext(fn)[0]
            for ext, hkey in ((".cu", "sha256:cu"), (".idx", "sha256:idx")):
                if hashes.get(hkey) and not os.path.exists(os.path.join(base, stem + ext)):
                    missing.append(stem + ext)
        checks.append(_check("shard files exist", not missing,
                             f"missing {len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}"))

    tok = manifest.get("tokenizer")
    checks.append(_check("tokenizer fingerprint (optional)", True,
                         json.dumps(tok, ensure_ascii=False) if tok
                         else "manifest has no tokenizer block (write-only; normal for old datasets)"))

    # ------------------------------------------------------------
    # 深度校验（deep=True）：抽样 shard 做内容级抽查（仅本地路径）
    # ------------------------------------------------------------
    if deep:
        if not is_local:
            checks.append(_check("deep sampling", True,
                                 "skipped: deep mode supports local index.json only"))
        elif not shards:
            checks.append(_check("deep sampling", True, "skipped: no shards"))
        else:
            ds_format = str(manifest.get("format") or "Memmap")
            ds_schema = manifest.get("schema") or {}
            sample = _sample_shards(shards)
            for s in sample:
                checks.extend(_deep_check_shard(
                    os.path.dirname(path), s, ds_format, ds_schema, bs, msl,
                ))

    return {"ok": all(c["ok"] for c in checks), "index_path": path, "checks": checks}


# =============================================================
# 深度校验（二轮 P3-10）：抽样 shard 内容级抽查
# =============================================================

def _sample_shards(shards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """确定性抽样：首、尾两个 shard（不足 2 个时有多少抽多少）。"""
    idx = sorted({0, len(shards) - 1})
    return [shards[i] for i in idx]


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _deep_check_shard(base: str, shard: Dict[str, Any], fmt: str,
                      schema: Dict[str, Any],
                      batch_size: int, max_seq_len: int) -> List[Dict[str, Any]]:
    """对单个 shard 做 sha256 + 内容契约抽查；单项失败不中断其余检查。"""
    fn = shard.get("filename") or ""
    shard_name = fn or "<unnamed>"
    hashes = shard.get("hashes") or {}
    stem = os.path.splitext(fn)[0]

    def _file_hash_check(label: str, path: str, key: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return _check(f"[{shard_name}] {label} sha256", False, f"file missing: {path}")
        if not hashes.get(key):
            return _check(f"[{shard_name}] {label} sha256", True,
                          "no hash recorded (old manifest), skipped")
        try:
            got = _sha256_file(path)
        except OSError as e:
            return _check(f"[{shard_name}] {label} sha256", False, f"read failed: {e}")
        want = hashes[key]
        return _check(f"[{shard_name}] {label} sha256", got == want,
                      "match" if got == want else f"manifest={want[:16]}… file={got[:16]}…")

    out: List[Dict[str, Any]] = []
    out.append(_file_hash_check(".bin", os.path.join(base, fn), "sha256"))
    for ext, key in ((".cu", "sha256:cu"), (".idx", "sha256:idx")):
        if hashes.get(key):
            out.append(_file_hash_check(ext, os.path.join(base, stem + ext), key))

    n_expected = int(shard.get("num_batches") or 0)
    if fmt == "ArrowIpc":
        out.extend(_deep_check_arrow_shard(base, shard_name, fn, n_expected, batch_size, max_seq_len))
    else:
        out.extend(_deep_check_pt_shard(base, shard_name, fn, stem, n_expected, batch_size, max_seq_len))
        # 文档级旁路列（列扩展）：schema 声明 ⇒ 列文件必须存在且定宽（否则读取端会报错）
        for key, ext in (("doc_ids", ".col_doc_ids"), ("loss_weights", ".col_loss_weights")):
            if key not in schema:
                continue
            col_path = os.path.join(base, stem + ext)
            if not os.path.exists(col_path):
                out.append(_check(f"[{shard_name}] {ext} exists", False,
                                  f"schema declares {key} but the column file is missing"))
                continue
            # 定宽校验：doc_ids 每 token 4 字节、loss_weights 每 token 2 字节
            elem = 4 if key == "doc_ids" else 2
            want = n_expected * batch_size * max_seq_len * elem
            got = os.path.getsize(col_path)
            out.append(_check(
                f"[{shard_name}] {ext} fixed width", want == 0 or got == want,
                f"{got} bytes, expected {want} ({n_expected} batches × {batch_size}×{max_seq_len}×{elem})",
            ))
    return out


def _deep_check_pt_shard(base: str, shard_name: str, fn: str, stem: str, n_expected: int,
                         batch_size: int, max_seq_len: int) -> List[Dict[str, Any]]:
    """PT Memmap：.cu length-prefixed 记录契约 + .idx 单调性。"""
    import struct
    out: List[Dict[str, Any]] = []
    cu_path = os.path.join(base, stem + ".cu")
    idx_path = os.path.join(base, stem + ".idx")

    if not os.path.exists(cu_path):
        out.append(_check(f"[{shard_name}] cu contract", True,
                          "no .cu sidecar (old dataset), skipped"))
    else:
        try:
            with open(cu_path, "rb") as f:
                data = f.read()
            records: List[List[int]] = []
            off = 0
            while off < len(data):
                (count,) = struct.unpack_from("<I", data, off)
                off += 4
                values = list(struct.unpack_from(f"<{count}i", data, off)) if count else []
                off += 4 * count
                records.append(values)
            problems: List[str] = []
            if n_expected and len(records) != n_expected:
                problems.append(f"record count {len(records)} != num_batches {n_expected}")
            for i, cu in enumerate(records):
                if not cu or cu[0] != 0:
                    problems.append(f"batch {i}: cu must start at 0, got {cu[:3]}")
                    continue
                if any(b < a for a, b in zip(cu, cu[1:])):
                    problems.append(f"batch {i}: cu not non-decreasing")
                    continue
                if batch_size and max_seq_len and cu[-1] > batch_size * max_seq_len:
                    problems.append(f"batch {i}: cu[-1]={cu[-1]} > batch_size*max_seq_len"
                                    f"={batch_size * max_seq_len}")
            out.append(_check(f"[{shard_name}] cu contract", not problems,
                              "; ".join(problems[:3]) if problems
                              else f"{len(records)} records OK"))
        except (OSError, struct.error) as e:
            out.append(_check(f"[{shard_name}] cu contract", False, f"parse failed: {e}"))

    if os.path.exists(idx_path):
        try:
            with open(idx_path, "rb") as f:
                raw = f.read()
            offsets = list(struct.unpack(f"<{len(raw) // 8}Q", raw))
            problems: List[str] = []
            if n_expected and len(offsets) != n_expected + 1:
                problems.append(f"{len(offsets)} entries != num_batches+1 ({n_expected + 1})")
            if any(b < a for a, b in zip(offsets, offsets[1:])):
                problems.append("offsets not monotonically non-decreasing")
            out.append(_check(f"[{shard_name}] idx offsets", not problems,
                              "; ".join(problems) if problems else f"{len(offsets)} entries OK"))
        except (OSError, struct.error) as e:
            out.append(_check(f"[{shard_name}] idx offsets", False, f"parse failed: {e}"))
    return out


def _deep_check_arrow_shard(base: str, shard_name: str, fn: str, n_expected: int,
                            batch_size: int, max_seq_len: int) -> List[Dict[str, Any]]:
    """ArrowIpc：pyarrow 读首个 batch，校验 batch 级列契约（与 Rust 读取侧同口径）。"""
    out: List[Dict[str, Any]] = []
    try:
        import pyarrow as pa
    except ImportError:
        out.append(_check(f"[{shard_name}] arrow contract", True,
                          "pyarrow not installed, skipped"))
        return out
    path = os.path.join(base, fn)
    try:
        with pa.memory_map(path, "rb") as src:
            reader = pa.ipc.open_file(src)
            if n_expected and reader.num_record_batches != n_expected:
                out.append(_check(f"[{shard_name}] arrow contract", False,
                                  f"batches in file={reader.num_record_batches} "
                                  f"!= num_batches {n_expected}"))
                return out
            batch = reader.get_batch(0)
            problems: List[str] = []
            cu_arr = batch.column("cu_seqlens") if "cu_seqlens" in batch.schema.names else None
            row_arr = batch.column("row_lens") if "row_lens" in batch.schema.names else None
            if cu_arr is None or row_arr is None:
                problems.append("missing batch-level columns (cu_seqlens/row_lens)")
            else:
                # 写侧约定：一个 RecordBatch = 一个 packed batch，行数 == batch_size；
                # batch 级列只填第 0 行，其余行是空列表
                if batch_size and (cu_arr.length() != batch_size or row_arr.length() != batch_size):
                    problems.append(f"batch-level columns must have batch_size rows "
                                    f"({batch_size}), got cu={cu_arr.length()} row={row_arr.length()}")
                elif not cu_arr.length():
                    problems.append("batch-level columns are empty")
                else:
                    cu = cu_arr.field(0).to_pylist()
                    row = row_arr.field(0).to_pylist()
                    if not cu or cu[0] != 0:
                        problems.append(f"cu must start at 0, got {cu[:3]}")
                    elif any(b < a for a, b in zip(cu, cu[1:])):
                        problems.append("cu not non-decreasing")
                    elif batch_size and len(row) != batch_size:
                        problems.append(f"row_lens len {len(row)} != batch_size {batch_size}")
                    elif batch_size and sum(row) != cu[-1]:
                        problems.append(f"sum(row_lens)={sum(row)} != cu[-1]={cu[-1]}")
            out.append(_check(f"[{shard_name}] arrow contract", not problems,
                              "; ".join(problems[:3]) if problems else "first batch OK"))
    except Exception as e:  # noqa: BLE001 - 校验器不应向调用方抛异常
        out.append(_check(f"[{shard_name}] arrow contract", False,
                          f"{type(e).__name__}: {e}"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m megadata.validate",
        description="megadata dataset self-check (programmatic assertions for common pitfalls)",
    )
    p.add_argument("index_path", help="path to the dataset index.json")
    p.add_argument("--world-size", type=int, default=None, help="check shard count >= world_size")
    p.add_argument("--batch-size", type=int, default=None, help="check batch_size matches expectation")
    p.add_argument("--max-seq-len", type=int, default=None, help="check max_seq_len matches expectation")
    p.add_argument("--deep", action="store_true",
                   help="deep sampling: verify sha256 + shard content contract (local index only)")
    args = p.parse_args(argv)

    report = validate(args.index_path, world_size=args.world_size,
                      expect_batch_size=args.batch_size,
                      expect_max_seq_len=args.max_seq_len,
                      deep=args.deep)
    for c in report["checks"]:
        print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"\nVerdict: {'PASS' if report['ok'] else 'ISSUES FOUND'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

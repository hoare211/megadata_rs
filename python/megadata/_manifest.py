# python/megadata/_manifest.py
"""manifest 读取 / 进程内缓存 / 指纹（P2-11 自 `dataloader.py` 原样拆出，零行为变更）。

依赖方向：`_manifest` → `_rust`（无 torch / pyarrow / engine 依赖）。
"""
import json
import os
import warnings
from typing import Any, Dict, List

from megadata._rust import read_manifest_json


def read_manifest(index_path: str) -> Dict[str, Any]:
    """读取数据集 manifest（本地 / http(s) / s3），带重试退避（走 Rust IO 层）。

    公共 API 保持"每次真实读取"语义；加载器内部走 `_load_manifest_cached`。
    """
    return json.loads(read_manifest_json(str(index_path)))


# F-26/F-30：加载器内部的 manifest 进程内缓存（**FIFO 有界**，`_MANIFEST_CACHE_MAX` 条，
# 超限淘汰最旧）。`StreamingDataset` 每轮重建 loader（epoch-per-iter 语义）会重走
# read_manifest（Rust IO + 重试）；本缓存避免每轮重复 IO。
# 失效策略：本地文件按 mtime_ns 失效；非本地路径（http/s3）进程内不失效
# （同一进程内数据集视为不变）；远程条目被 FIFO 淘汰后会重新远程读取
# （多数据集长生命周期进程的内存 vs 重复 IO 权衡，容量默认 128）。
# fork 后各进程独立副本，无跨进程一致性问题。
_MANIFEST_CACHE_MAX = 128
_MANIFEST_CACHE: Dict[str, Any] = {}


def _load_manifest_cached(index_path: str) -> Dict[str, Any]:
    """内部使用：带进程内缓存的 manifest 读取（返回值为**只读共享**，调用方不得修改）。"""
    path = str(index_path)
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        stamp = None
    entry = _MANIFEST_CACHE.get(path)
    if entry is not None and (stamp is None or entry[0] == stamp):
        return entry[1]
    manifest = read_manifest(path)
    if path not in _MANIFEST_CACHE and len(_MANIFEST_CACHE) >= _MANIFEST_CACHE_MAX:
        _MANIFEST_CACHE.pop(next(iter(_MANIFEST_CACHE)))   # FIFO：淘汰最旧条目
    _MANIFEST_CACHE[path] = (stamp, manifest)
    return manifest


def _read_total_tokens(index_path: str) -> int:
    """取 manifest 的 global_stats.total_tokens（Mixer 按 token 数配比权重用）。

    F-30：走 manifest 缓存，`update_temperature` 不再重复触发网络/磁盘 IO。
    """
    return int(_load_manifest_cached(index_path)["global_stats"]["total_tokens"])


def _effective_modality_tokens(stats: Dict[str, Any]) -> "tuple[int, int, int, int]":
    """模态拆分的读取期解释（sidecar 格式冻结 §8，Rust `effective_modality_tokens`
    的 Python 侧透出，语义必须与 Rust 保持一致）：

    - `stats_version >= 2`：直接用拆分字段；
    - 旧 manifest（字段缺失 = 0）：**明确解释**为 `text = total_tokens`、
      其余模态 = 0，不静默错解。**不改写 manifest**（读不改写纪律）。

    返回 `(text, image, audio, video)` 有效 token 四元组。
    """
    total = int(stats.get("total_tokens") or 0)
    if int(stats.get("stats_version") or 0) >= 2:
        return (
            int(stats.get("total_text_tokens") or 0),
            int(stats.get("total_image_tokens") or 0),
            int(stats.get("total_audio_tokens") or 0),
            int(stats.get("total_video_tokens") or 0),
        )
    return (total, 0, 0, 0)


def _manifest_format(manifest: Dict[str, Any]) -> str:
    """manifest 的存储格式 → 逻辑 format（"pt" / "sft"）。"""
    return "sft" if manifest.get("format") == "Arrow-IPC" else "pt"


def _fingerprint_of(path: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    stats = manifest.get("global_stats", {})
    return {
        "index_path": str(path),
        "version": manifest.get("version"),
        "batch_size": int(stats.get("batch_size") or 0),
        "max_seq_len": int(manifest.get("max_seq_len") or 0),
        "total_batches": int(stats.get("total_batches") or 0),
    }


def _combined_manifest_fingerprint(paths: List[str], manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """F-09：MixBatchStream 用——**全子流**的合并指纹（顺序敏感）。

    旧实现只取 `paths[0]` 的指纹，子流 2..N 被替换（身份字段不同）无法检测，
    续训会静默读到错位数据。现在逐个记录，由 `_check_fingerprint` 逐子流比对。
    """
    return {"sub_streams": [_fingerprint_of(p, m) for p, m in zip(paths, manifests)]}


def _check_one_fingerprint(saved: Dict[str, Any], current: Dict[str, Any], *, label: str) -> None:
    identity = ("version", "batch_size", "max_seq_len", "total_batches")
    diffs = [k for k in identity
             if saved.get(k) not in (None, 0) and current.get(k) not in (None, 0)
             and saved.get(k) != current.get(k)]
    if diffs:
        raise ValueError(
            f"{label}manifest fingerprint mismatch (" + "; ".join(f"{k}: {saved.get(k)} → {current.get(k)}" for k in diffs)
            + "): the checkpoint and the current dataset are not the same dataset; refusing to resume (avoid silently reading misaligned data)."
        )
    if saved.get("index_path") and current.get("index_path") and saved["index_path"] != current["index_path"]:
        warnings.warn(
            f"{label}manifest path changed ({saved['index_path']} → {current['index_path']}): "
            "identity fields match, treating it as the same dataset; if it is actually a different dataset, retrain.",
            stacklevel=3,
        )


def _check_fingerprint(saved: Dict[str, Any], current: Dict[str, Any]) -> None:
    """B6'：身份字段不一致 → 拒绝（抛错）；index_path 变化 → 仅告警。

    F-09：支持 MixBatchStream 的"全子流合并指纹"（`{"sub_streams": [...]}`）——逐子流比较，
    子流数量或任一子流的身份字段不一致都拒绝。旧格式（单子流指纹 dict）仍按其原语义
    比对（子流级字段在旧快照里缺失时跳过，保持向后兼容）。
    """
    saved_subs = saved.get("sub_streams") if isinstance(saved, dict) else None
    current_subs = current.get("sub_streams") if isinstance(current, dict) else None
    if isinstance(saved_subs, list) and isinstance(current_subs, list):
        if len(saved_subs) != len(current_subs):
            raise ValueError(
                f"sub-stream count mismatch (checkpoint {len(saved_subs)} → current {len(current_subs)}): "
                "refusing to resume (the sub-stream set changed; weight and cursor semantics no longer correspond)."
            )
        for i, (s, c) in enumerate(zip(saved_subs, current_subs)):
            _check_one_fingerprint(s, c, label=f"sub-stream {i} ")
        return
    _check_one_fingerprint(saved, current, label="")

# python/megadata/_topology.py
"""DDP / DataLoader worker 拓扑探测与跨 rank 轮长判定（P2-11 自 `dataloader.py` 原样拆出，零行为变更）。

torch 只在函数内**惰性导入**（`torch.distributed` / `get_worker_info`），
本模块顶层不引入 torch。依赖方向：`_topology` → 标准库（engine / tier1 在其上）。
"""
import os
import warnings
from typing import Any, Dict, List, Optional

_ENV_TOPOLOGY_WARNED = False


def _detect_dist(world_size: Optional[int], rank: Optional[int]) -> "tuple[int, int]":
    """拓扑探测：显式参数 > torch.distributed（已初始化）> 环境变量 > 单机默认。

    F-14：从环境变量回退时**告警一次**（单机调试残留 `WORLD_SIZE`/`RANK` 会被静默分片）；
    显式传参无告警。`MEGADATA_STRICT_TOPOLOGY=1` 时**禁止 env 回退**（未显式传且
    torch.distributed 未初始化则直接报错）。
    """
    global _ENV_TOPOLOGY_WARNED
    if world_size is None or rank is None:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                if world_size is None:
                    world_size = dist.get_world_size()
                if rank is None:
                    rank = dist.get_rank()
        except Exception:
            pass
    strict = os.environ.get("MEGADATA_STRICT_TOPOLOGY", "") in ("1", "true", "True")
    env_used: List[str] = []
    if world_size is None:
        env_ws = os.environ.get("WORLD_SIZE")
        if env_ws:
            if strict:
                raise RuntimeError(
                    "MEGADATA_STRICT_TOPOLOGY=1: world_size was not passed explicitly and env fallback is forbidden "
                    f"(detected WORLD_SIZE={env_ws}). Pass world_size=... explicitly, or unset the variable in single-machine runs."
                )
            env_used.append(f"WORLD_SIZE={env_ws}")
            world_size = int(env_ws)
        else:
            world_size = 1
    if rank is None:
        env_rank = os.environ.get("RANK")
        if env_rank:
            if strict:
                raise RuntimeError(
                    "MEGADATA_STRICT_TOPOLOGY=1: rank was not passed explicitly and env fallback is forbidden "
                    f"(detected RANK={env_rank}). Pass rank=... explicitly, or unset the variable in single-machine runs."
                )
            env_used.append(f"RANK={env_rank}")
            rank = int(env_rank)
        else:
            rank = 0
    if env_used and not _ENV_TOPOLOGY_WARNED:
        _ENV_TOPOLOGY_WARNED = True
        warnings.warn(
            "world_size / rank were not passed explicitly; fell back to environment variables (" + "; ".join(env_used) + "): "
            "leftover variables in single-machine debugging cause silent sharding. Pass world_size/rank explicitly or unset "
            "the variables (MEGADATA_STRICT_TOPOLOGY=1 disables the fallback).",
            stacklevel=2,
        )
    return int(world_size), int(rank)


def _detect_workers() -> "tuple[int, int]":
    """在 DataLoader worker 进程内从 get_worker_info() 取 worker 维度。

    v3（P4）：引擎层**没有** worker 参数——worker 维度只存在于 DataLoader 层，
    worker 内由本函数自动探测（主进程返回 (0, 0)）。
    """
    try:
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        if info is not None:
            return int(info.num_workers), int(info.id)
    except Exception:
        pass
    return 0, 0


def _warn_if_topology_changed(state: Dict[str, Any], *, world_size: int, rank: int,
                              num_workers: Optional[int] = None,
                              worker_id: Optional[int] = None,
                              partition: Optional[str] = None,
                              shard_order: Optional[str] = None) -> None:
    diffs: List[str] = []

    def _cmp(key: str, current: Any) -> None:
        if key in state and state[key] is not None and state[key] != current:
            diffs.append(f"{key}: {state[key]} → {current}")

    _cmp("world_size", world_size)
    _cmp("rank", rank)
    if num_workers is not None:
        _cmp("num_workers", num_workers)
    if worker_id is not None:
        _cmp("worker_id", worker_id)
    if partition is not None:
        _cmp("partition", partition)
    if shard_order is not None:
        _cmp("shard_order", shard_order)
    if diffs:
        warnings.warn(
            "resume topology differs from the snapshot (" + "; ".join(diffs) + "): per-rank consumption progress cannot be "
            "translated across topologies into a global position; resuming may repeat or skip a small amount of data. "
            "Note: changing num_workers changes the rank-level shard split itself (not just the scheduling granularity), "
            "so each rank consumes a different set of shards. For exact resume keep world_size / rank / num_workers / "
            "partition unchanged (relaxed only guarantees a stable global sample order); changing shard_order "
            "(random / sequential) changes both the shard order inside each epoch and the per-shard batch order seen, "
            "so a checkpoint cannot be translated across shard_order values either.",
            stacklevel=3,
        )


def _ddp_world_size() -> int:
    """`torch.distributed` 已初始化时的 world_size，否则 0（**不做任何集合通信**）。"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 0


#: 自然模式 + DDP 的静态判定无法覆盖"shard 尺寸不齐"时的告警去重（每进程一次）
_DDP_NATURAL_WARNED = False


def _natural_mode_length_verdict(
    manifests: List[Dict[str, Any]], *, world_size: int, num_workers: int,
) -> "tuple[str, str]":
    """自然模式 DDP 轮长恒等性的**纯本地**判定（零集合通信、零额外 IO）。

    返回 ``("safe" | "danger" | "unknown", 说明)``。

    判据（等长 shard 时可精确证真/证伪）：每个 (rank, worker) 每轮分到的 **shard 数**
    由其在扁平编号中的位置决定——round_robin 取 `下标 % T`、relaxed 取连续切片——
    跨 epoch 恒定；每 shard 的 batch 数相等时，轮长 = shard 数 × 每 shard batch 数。
    因此可以本地精确算出每个 rank 的轮长：
    - 全相等 → `safe`（DDP 轮末同步安全）；
    - 不全相等 → `danger`（可证明不等，必须报错）；
    - 任一子流 shard 尺寸不齐 → `unknown`（步数还取决于"分到哪些 shard"，静态不可判）。
    """
    W = max(1, int(world_size))
    K = max(1, int(num_workers))
    T = W * K
    for i, manifest in enumerate(manifests):
        shards = manifest.get("shards") or []
        n = len(shards)
        counts = {int(s.get("num_batches") or 0) for s in shards}
        if n == 0 or not counts or counts == {0}:
            return "unknown", f"sub-stream {i} manifest has no usable shard/num_batches records"
        if len(counts) != 1:
            return "unknown", f"sub-stream {i} has {n} shards of unequal sizes (num_batches has {len(counts)} distinct values)"
        per_shard = next(iter(counts))
        base, rem = divmod(n, T)
        rank_batches = [
            (base * K + sum(1 for w in range(r * K, (r + 1) * K) if w < rem)) * per_shard
            for r in range(W)
        ]
        if len(set(rank_batches)) != 1:
            return "danger", (
                f"sub-stream {i}: {n} equal-sized shards ({per_shard} batches each) distribute unevenly across "
                f"{T} (rank, worker) pairs; per-rank epoch lengths = {rank_batches}"
            )
    return "safe", "all sub-streams have equal-sized shards evenly distributed across (rank, worker) pairs; epoch lengths are constant across ranks and epochs"

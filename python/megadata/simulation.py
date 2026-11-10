"""megadata 仿真器：不跑真实训练，回答"shard 多大 / cache_limit 设多少 / predownload 设多少"。

定位：按 step 步进的吞吐 / 缓存 / 饥饿仿真，
数据源是 Rust 侧 ``plan_shard_sequence``（与 ``RustStreamReader`` 同一套分区 / 置换纯函数）
+ ``index.json`` 的 shard 元数据（``size_bytes`` 是磁盘字节数，含压缩，因此下载量统计对压缩感知）。

仿真模型与假设（规划级近似，结论用于选参而非精确预测）：

1. **step = 消费一个 batch**，纯计算耗时 ``compute_ms_per_batch``；步骤被拉长的唯一原因是数据未就绪。
2. **数据源**：``plan[worker][step]`` 给出该 worker 该步要读的全局 shard 下标；
   一个 shard 含多个 batch，故连续多步可能落在同一 shard（命中缓存）。
3. **worker → 节点映射**：``node = worker // workers_per_node``，假设 rank 按 node 连续铺开
   （``workers_per_node = ceil(total_workers / num_nodes)``，末节点不足则收敛到末节点）。
4. **每节点串行 FIFO 下载队列**（保守近似，不做精细带宽分片）：
   ``start = max(request_time, node.queue_free_at)``，``finish = start + size_bytes / bandwidth``，
   ``queue_free_at = finish``。带宽为节点共享下行带宽（单位 MB/s，按 1e6 bytes 计）。
5. **predownload 预取窗口**：worker 处理第 i 步时，向前看 ``predownload`` 步，把"将要用到且
   未缓存、未在途"的 shard 立即入队（``request_time = 当前 worker 时钟``），
   即预取提前量 ≈ ``ahead × compute_ms``。
6. **饥饿判定**：消费第 i 步时若 shard 的 ``finish > 当前时钟`` → 该步拉长到
   ``finish + compute_ms``，计一次 starved。
7. **节点 LRU 缓存**：条目 ``shard_id → size``，插入 / 命中时移到 LRU 末端，超 ``cache_limit_bytes``
   从最久未用端驱逐。**不建模"正在被读"的引用计数**（规划级近似）。超大 shard（单条即超限）保留。
8. **时钟**：每个 worker 独立推进；事件按"最早时钟"合并处理（时钟相同按 worker 编号），
   保证跨 worker 的下载排队顺序与时间线一致，且结果确定可复现。

由此可得两条读表须知：

- 本模型下**任何 miss 都必然产生等待**（下载耗时 > 0 且串行队列），因此
  ``starved_steps == cache_misses``、``starve_ratio == 1 - hit_rate``；两列都保留是为与
  streaming 的指标口径对齐，也可在后续引入并行下载 / 带宽分片后自然分化。
- ``peak_cache_bytes`` 是各节点峰值缓存之和（全局保守上界）。

CLI 用法::

    python -m megadata.simulation data/pt/index.json --world-size 64 --num-workers 2 \\
        --bandwidth-mbps 500 --cache-limit-gb 64 --epochs 2
    python -m megadata.simulation data/pt/index.json --world-size 64 --sweep-cache
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

try:
    # ★ 评审 §3.8(d)：Rust 侧唯一事实源（`stream.rs::DEFAULT_PREDOWNLOAD`，
    #   经 PyO3 导出）。原先本文件独立硬编码 16，两份各自维护。
    from megadata._rust import DEFAULT_PREDOWNLOAD
except ImportError:  # pragma: no cover - 仅在扩展未重建（旧 .so）时触发
    # 兼容兜底，**刻意不作为第二份事实源**：
    # tests/test_simulation.py::test_predownload_default_matches_rust 会在扩展
    # 重建后校验两者一致，一旦Rust 侧改动而本兜底未跟上，测试即失败。
    DEFAULT_PREDOWNLOAD = 16

from ._rust import plan_shard_sequence as _rust_plan_shard_sequence
from ._options import normalize_partition, normalize_shard_order, normalize_shuffle

# 扫描缓存档位：k 个 shard / 节点（None = 不限）
SWEEP_K: Tuple[Optional[int], ...] = (2, 4, 8, 16, 32, None)


@dataclass
class SimulationConfig:
    """仿真输入参数（与 Rust 侧控制面参数一一对应，便于直接对照调参）。

    F-12：构造期校验（`__post_init__`）——`shuffle` 走与 `BatchStream`/`MixBatchStream`
    同一归一化实现（取值集合一致，拼错不再静默当开启）；拓扑与性能参数必须非负/合法。
    """

    index_path: str
    world_size: int = 8
    num_nodes: int = 1
    num_workers: int = 0
    seed: int = 42
    epochs: int = 1
    shuffle: str = "permutation"      # "permutation" | "none"（对应 Rust shuffle bool；True/False 为等价简写）
    partition: str = "round_robin"    # round_robin | relaxed
    shard_order: str = "random"       # random | sequential（epoch 内 shard 序，F6；与 shuffle 正交）
    num_canonical_nodes: Optional[int] = None
    bandwidth_mbps: float = 1000.0    # 每节点共享下行带宽（MB/s，按 1e6 bytes）
    cache_limit_bytes: int = 0        # 0 = 不限（用于对照）
    compute_ms_per_batch: float = 250.0
    # ★ 评审 §3.8(d)：默认值取自 Rust 的 `DEFAULT_PREDOWNLOAD`（经 PyO3 导出），
    #   不再独立硬编码 16。原先两份各自维护，改Rust 侧不会同步仿真器 →
    #   容量规划结论会失真（仿真报告"16 够用"，真实引擎其实是另一个值）。
    predownload: int = DEFAULT_PREDOWNLOAD
    max_steps: int = 0                # 0 = 跑满

    def __post_init__(self) -> None:
        try:
            normalize_shuffle(self.shuffle)
        except ValueError as e:
            raise ValueError(f"invalid SimulationConfig.shuffle: {e}") from None
        try:
            normalize_partition(self.partition)
        except ValueError as e:
            raise ValueError(f"invalid SimulationConfig.partition: {e}") from None
        try:
            normalize_shard_order(self.shard_order)
        except ValueError as e:
            raise ValueError(f"invalid SimulationConfig.shard_order: {e}") from None
        for name, value, lo in (
            ("world_size", self.world_size, 1),
            ("num_nodes", self.num_nodes, 1),
            ("num_workers", self.num_workers, 0),
            ("seed", self.seed, 0),
            ("epochs", self.epochs, 1),
            ("predownload", self.predownload, 0),
            ("max_steps", self.max_steps, 0),
            ("cache_limit_bytes", self.cache_limit_bytes, 0),
        ):
            if int(value) < lo:
                raise ValueError(f"SimulationConfig.{name} must be >= {lo}, got {value!r}")
        if float(self.bandwidth_mbps) <= 0:
            raise ValueError(f"SimulationConfig.bandwidth_mbps must be > 0, got {self.bandwidth_mbps!r}")
        if float(self.compute_ms_per_batch) < 0:
            raise ValueError(f"SimulationConfig.compute_ms_per_batch must be >= 0, got {self.compute_ms_per_batch!r}")


@dataclass
class SimulationReport:
    """仿真输出（纯标量，便于断言 / 打印 / 序列化）。"""

    steps: int = 0
    total_download_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    peak_cache_bytes: int = 0
    starved_steps: int = 0
    starve_ratio: float = 0.0
    busy_time_s: float = 0.0          # 各 worker 纯计算时间之和
    wall_time_s: float = 0.0          # 最慢 worker 的墙钟（全局完成时间）
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    avg_gpu_util: float = 0.0         # busy_time_s / (wall × worker 数) ≤ 1，平均每设备利用率


class _NodeSim:
    """单节点：串行 FIFO 下载队列 + 在途表 + LRU 缓存。"""

    __slots__ = (
        "sizes", "bandwidth", "limit", "queue_free_at", "cache", "cache_bytes",
        "inflight", "_heap", "peak_cache_bytes", "downloaded_bytes",
    )

    def __init__(self, sizes: Sequence[int], bandwidth: float, limit: int) -> None:
        self.sizes = sizes
        self.bandwidth = bandwidth
        self.limit = limit
        self.queue_free_at = 0.0
        # shard_id -> size；OrderedDict 头部 = 最久未用（LRU 驱逐端）
        self.cache: "OrderedDict[int, int]" = OrderedDict()
        self.cache_bytes = 0
        self.inflight: Dict[int, float] = {}
        self._heap: List[Tuple[float, int]] = []
        self.peak_cache_bytes = 0
        self.downloaded_bytes = 0

    def schedule(self, shard: int, request_time: float) -> float:
        """确保 shard 在途（返回其完成时刻）；已缓存返回 -inf（无需下载）。"""
        if shard in self.cache:
            return -math.inf
        fin = self.inflight.get(shard)
        if fin is not None:
            return fin
        start = max(request_time, self.queue_free_at)
        fin = start + self.sizes[shard] / self.bandwidth
        self.queue_free_at = fin
        self.inflight[shard] = fin
        self.downloaded_bytes += self.sizes[shard]
        heapq.heappush(self._heap, (fin, shard))
        return fin

    def apply_completions(self, now: float) -> None:
        """把 now 之前完成的下载落进缓存（懒应用，保证缓存可见性与时间线一致）。"""
        while self._heap and self._heap[0][0] <= now:
            _fin, shard = heapq.heappop(self._heap)
            self.inflight.pop(shard, None)
            if shard in self.cache:
                continue
            size = self.sizes[shard]
            self.cache[shard] = size
            self.cache_bytes += size
            if self.limit > 0:
                # 先驱逐再看峰值：LRU 头部是最久未用（刚插入的在末端，不会被驱逐）
                while self.cache_bytes > self.limit and len(self.cache) > 1:
                    _victim, victim_size = self.cache.popitem(last=False)
                    self.cache_bytes -= victim_size
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)

    def touch(self, shard: int) -> None:
        """缓存命中：移到 LRU 末端。"""
        self.cache.move_to_end(shard)


def _steps_for(cfg: SimulationConfig, seq: Sequence[int]) -> int:
    if cfg.max_steps > 0:
        return min(len(seq), cfg.max_steps)
    return len(seq)


def simulate_plan(
    cfg: SimulationConfig,
    plan: Sequence[Sequence[int]],
    shard_sizes: Sequence[int],
) -> SimulationReport:
    """纯函数仿真核心：给定 plan（worker → 每步 shard）与 shard 大小，产出报告。

    与 Rust 侧解耦，便于用合成 plan 做单元测试；真实 plan 由 :func:`simulate` 提供。
    """
    num_nodes = max(int(cfg.num_nodes), 1)
    total_workers = len(plan)
    if total_workers == 0:
        return SimulationReport()

    bandwidth = max(float(cfg.bandwidth_mbps), 1e-9) * 1e6  # bytes/s
    compute_s = max(float(cfg.compute_ms_per_batch), 0.0) / 1000.0
    limit = max(int(cfg.cache_limit_bytes), 0)

    workers_per_node = max(1, math.ceil(total_workers / num_nodes))
    node_of = [min(w // workers_per_node, num_nodes - 1) for w in range(total_workers)]
    nodes = [_NodeSim(shard_sizes, bandwidth, limit) for _ in range(num_nodes)]

    report = SimulationReport()
    step_durations: List[float] = []
    clocks = [0.0] * total_workers
    cursor = [0] * total_workers

    # 事件堆：(worker 时钟, worker 编号)；时钟相同按编号，保证确定性
    heap: List[Tuple[float, int]] = [
        (0.0, w) for w in range(total_workers) if _steps_for(cfg, plan[w]) > 0
    ]
    heapq.heapify(heap)

    predownload = max(int(cfg.predownload), 0)
    while heap:
        now, w = heapq.heappop(heap)
        seq = plan[w]
        i = cursor[w]
        if i >= _steps_for(cfg, seq):
            continue
        node = nodes[node_of[w]]
        node.apply_completions(now)

        # 预取窗口：[i, i + predownload)（含当前步——当前步"立即需要"是窗口最小值）
        for j in range(i, min(i + predownload, len(seq))):
            node.schedule(seq[j], now)

        shard = seq[i]
        if shard in node.cache:
            node.touch(shard)
            report.cache_hits += 1
            step_end = now + compute_s
        else:
            report.cache_misses += 1
            fin = node.schedule(shard, now)
            if fin > now:
                report.starved_steps += 1
                step_end = fin + compute_s
            else:
                step_end = now + compute_s

        step_durations.append(step_end - now)
        clocks[w] = step_end
        cursor[w] = i + 1
        if cursor[w] < _steps_for(cfg, seq):
            heapq.heappush(heap, (step_end, w))

    report.steps = len(step_durations)
    report.total_download_bytes = sum(node.downloaded_bytes for node in nodes)

    if report.steps:
        report.starve_ratio = report.starved_steps / report.steps
        total_lookups = report.cache_hits + report.cache_misses
        report.hit_rate = report.cache_hits / total_lookups if total_lookups else 0.0
        report.busy_time_s = report.steps * compute_s
        report.wall_time_s = max(clocks) if clocks else 0.0
        # 按 worker 数摊平：worker 们并行消费各自的 GPU，总利用率不应超过 1
        report.avg_gpu_util = (
            report.busy_time_s / (report.wall_time_s * total_workers)
            if report.wall_time_s > 0 else 1.0
        )
        srt = sorted(step_durations)
        report.p50_ms = _percentile(srt, 0.50) * 1000.0
        report.p99_ms = _percentile(srt, 0.99) * 1000.0
        report.max_ms = srt[-1] * 1000.0
    report.peak_cache_bytes = sum(node.peak_cache_bytes for node in nodes)
    return report


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """最近秩法分位数（离散、确定性，不插值）。"""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def load_shard_sizes(index_path: str) -> List[int]:
    """从 index.json 读 shard 元数据（只读元数据，不碰 shard 数据）。"""
    with open(index_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return [int(s["size_bytes"]) for s in manifest["shards"]]


def plan_from_rust(cfg: SimulationConfig) -> List[List[int]]:
    """调用 Rust 纯函数产出与真实读取序一致的 plan（worker → 每步全局 shard 下标）。

    F-11：shuffle 走共享归一化（不再用 `!= "none"` 的"非 none 即开启"近似）。
    F6：shard_order 透传（epoch 内 shard 访问顺序，与 shuffle 正交）。
    """
    shuffle = normalize_shuffle(cfg.shuffle)
    shard_order = normalize_shard_order(cfg.shard_order)
    ncn = int(cfg.num_canonical_nodes or 0)
    raw = _rust_plan_shard_sequence(
        cfg.index_path, cfg.world_size, cfg.num_workers, cfg.seed, cfg.epochs,
        shuffle, cfg.partition, shard_order, ncn,
    )
    return [list(seq) for seq in raw]


def simulate(cfg: SimulationConfig) -> SimulationReport:
    """读取数据集元数据 + Rust plan，跑一次仿真。"""
    plan = plan_from_rust(cfg)
    sizes = load_shard_sizes(cfg.index_path)
    return simulate_plan(cfg, plan, sizes)


def sweep_cache(
    cfg: SimulationConfig, ks: Sequence[Optional[int]] = SWEEP_K
) -> List[Tuple[str, SimulationReport]]:
    """对每节点缓存 = k 个 shard（k ∈ ks，None = 不限）各跑一遍。

    返回 [(档位标签, report)]；缓存字节数 = ``k × 平均 shard 大小``。
    """
    plan = plan_from_rust(cfg)
    sizes = load_shard_sizes(cfg.index_path)
    mean_size = (sum(sizes) / len(sizes)) if sizes else 0.0
    out: List[Tuple[str, SimulationReport]] = []
    for k in ks:
        label = "∞" if k is None else f"{k}×shard"
        limit = 0 if k is None else int(k * mean_size)
        out.append((label, simulate_plan(replace(cfg, cache_limit_bytes=limit), plan, sizes)))
    return out


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000.0:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}PB"


def format_report_markdown(
    report: SimulationReport, title: str = "Simulation Report"
) -> str:
    rows = [
        ("steps", str(report.steps)),
        ("downloaded", _fmt_bytes(report.total_download_bytes)),
        ("cache hits / misses", f"{report.cache_hits} / {report.cache_misses}"),
        ("hit_rate", f"{report.hit_rate:.3f}"),
        ("peak cache", _fmt_bytes(report.peak_cache_bytes)),
        ("starved_steps", str(report.starved_steps)),
        ("starve_ratio", f"{report.starve_ratio:.3f}"),
        ("busy_time", f"{report.busy_time_s:.1f}s"),
        ("wall_time", f"{report.wall_time_s:.1f}s"),
        ("step p50 / p99 / max", f"{report.p50_ms:.1f} / {report.p99_ms:.1f} / {report.max_ms:.1f} ms"),
        ("avg_gpu_util", f"{report.avg_gpu_util:.3f}"),
    ]
    lines = [f"### {title}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def format_sweep_markdown(
    cfg: SimulationConfig, results: Sequence[Tuple[str, SimulationReport]]
) -> str:
    header = (
        f"world_size={cfg.world_size} num_workers={cfg.num_workers} nodes={cfg.num_nodes} "
        f"epochs={cfg.epochs} bandwidth={cfg.bandwidth_mbps}MB/s "
        f"compute={cfg.compute_ms_per_batch}ms/batch predownload={cfg.predownload}"
    )
    lines = [
        f"### cache_limit sweep ({header})",
        "",
        "| per-node cache | steps | downloaded | hit_rate | starve_ratio | wall_time | p99 | gpu_util |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, r in results:
        lines.append(
            f"| {label} | {r.steps} | {_fmt_bytes(r.total_download_bytes)} | "
            f"{r.hit_rate:.3f} | {r.starve_ratio:.3f} | {r.wall_time_s:.1f}s | "
            f"{r.p99_ms:.1f}ms | {r.avg_gpu_util:.3f} |"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m megadata.simulation",
        description="megadata data-loading simulator: estimate shard sizes / cache_limit / predownload without a real training run.",
    )
    p.add_argument("index_path", help="path to the dataset index.json")
    p.add_argument("--world-size", type=int, default=8, help="number of global ranks (GPUs), default 8")
    p.add_argument("--num-nodes", type=int, default=1, help="number of physical nodes, default 1")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per rank, 0 = 1")
    p.add_argument("--seed", type=int, default=42, help="global random seed, default 42")
    p.add_argument("--epochs", type=int, default=1, help="number of epochs to simulate, default 1")
    p.add_argument("--shuffle", choices=("permutation", "none"), default="permutation")
    p.add_argument("--partition", choices=("round_robin", "relaxed"), default="round_robin")
    p.add_argument("--num-canonical-nodes", type=int, default=0, help="canonical nodes for relaxed partition, 0 = same as --num-nodes")
    p.add_argument("--bandwidth-mbps", type=float, default=1000.0, help="per-node shared download bandwidth (MB/s), default 1000")
    p.add_argument("--cache-limit-gb", type=float, default=0.0, help="per-node cache limit (GB), 0 = unlimited")
    p.add_argument("--compute-ms-per-batch", type=float, default=250.0, help="pure compute time per batch (ms), default 250")
    p.add_argument("--predownload", type=int, default=DEFAULT_PREDOWNLOAD, help=f"prefetch window (steps), default {DEFAULT_PREDOWNLOAD} (the Rust engine's own default)")
    p.add_argument("--max-steps", type=int, default=0, help="max steps simulated per worker, 0 = run to completion")
    p.add_argument("--sweep-cache", action="store_true", help="sweep per-node cache tiers (2/4/8/16/32/∞ shards)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = SimulationConfig(
        index_path=args.index_path,
        world_size=args.world_size,
        num_nodes=args.num_nodes,
        num_workers=args.num_workers,
        seed=args.seed,
        epochs=args.epochs,
        shuffle=args.shuffle,
        partition=args.partition,
        num_canonical_nodes=args.num_canonical_nodes or None,
        bandwidth_mbps=args.bandwidth_mbps,
        cache_limit_bytes=int(args.cache_limit_gb * 1e9),
        compute_ms_per_batch=args.compute_ms_per_batch,
        predownload=args.predownload,
        max_steps=args.max_steps,
    )
    if not os.path.exists(cfg.index_path):
        print(f"index.json does not exist: {cfg.index_path}")
        return 2

    if args.sweep_cache:
        results = sweep_cache(cfg)
        print(format_sweep_markdown(cfg, results))
    else:
        report = simulate(cfg)
        print(format_report_markdown(report))
    print()
    print("> The model is a planning-grade approximation (per-node serial FIFO download queue + LRU cache; see the simulation.py module docstring for details).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

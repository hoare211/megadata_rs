# python/megadata/tier1.py
"""Tier-1 torch 适配层（P2-11 自 `dataloader.py` 原样拆出，零行为变更）。

`StreamingDataset` / `MixStreamingDataset` / `MegadataDataLoader`：把引擎层加载器
接入 `torch.utils.data` 协议（IterableDataset / DataLoader），服务 HF Trainer /
Lightning。torch / torch.utils.data 依赖收敛在本层。

依赖方向：`tier1` → `engine`（→ _manifest / _topology / ...），无反向依赖。
"""
import os
import warnings
from typing import Any, Dict, List, Optional

from torch.utils.data import IterableDataset, DataLoader

from megadata._rust import epoch_total_batches as _epoch_total_batches
from megadata._topology import _detect_dist, _detect_workers
from megadata.engine import (
    BatchStream, DatasetConfig, LoaderOptions, MixBatchStream,
    _resolve_options,
)

# 中断异常 / 信号 / 活跃实例注册表见 `megadata._signals`（F-36 抽出）。
# F-18 的"直接迭代告警"用 `StreamingDataset._loader_iterated` 实例标志实现
# （不用线程本地变量：生成器被暂停/丢弃时 TLS 会残留，导致跨调用误判）。


# ============================================================
# Tier 1：HF Trainer / Lightning（StreamingDataset + MegadataDataLoader）
# ============================================================
def _identity_collate(batch):
    """batch 已在构建期预打包：DataLoader 不再做任何 collate。"""
    return batch


class StreamingDataset(IterableDataset):
    """megadata 流式数据集（`torch.utils.data.IterableDataset` 协议）。

    ```python
    ds = StreamingDataset(index_json_path, batch_size=..., max_seq_len=...)
    dl = MegadataDataLoader(ds, num_workers=4)
    trainer.fit(model, train_dataloaders=dl)
    ```

    生命周期（A1：close 只有一种语义——终态）：
    - `release()`：释放**当前轮**读头（幂等、非终态），下一轮 `__iter__` 自动重建；
    - `close()`：**终态**（与引擎层同语义），之后任何迭代/查询明确报错；
    - 支持 `with ds:`（退出时 `close()`）。

    **必须经 `MegadataDataLoader` 迭代**（C24-③）：直接 `for b in ds:`（不经 DataLoader）
    不会推进轮号 → 每轮数据顺序相同。

    轮号推进（C17"消费侧递增" + C26）：
    - `num_workers=0` / `persistent_workers=False`：主进程由 `MegadataDataLoader.__iter__`
      的 teardown 递增，`fork` 把新轮号带给下一轮 worker；
    - `persistent_workers=True`：worker 不再 fork，由本进程内生成器结束时自增
      （由 `_worker_epoch_bump` 显式分派，见 `MegadataDataLoader.__init__`）。
    """

    #: 类属性哨兵：`MixStreamingDataset` 不经 `StreamingDataset.__init__` 也能安全读
    _closed = False
    #: worker 内是否推进轮号（由 MegadataDataLoader 按 persistent_workers 设置）
    _worker_epoch_bump = False

    def __init__(self, index_json_path, *,
                 batch_size: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 42,
                 shuffle: Any = True,
                 partition: str = "round_robin",
                 shard_order: str = "random",
                 num_canonical_nodes: Optional[int] = None,
                 predownload: Optional[int] = None,
                 epoch: int = 0,
                 options: Optional["LoaderOptions"] = None):
        super().__init__()
        opts = _resolve_options(
            "StreamingDataset", options, epoch=epoch,
            batch_size=batch_size, max_seq_len=max_seq_len,
            world_size=world_size, rank=rank, seed=seed, shuffle=shuffle,
            partition=partition, shard_order=shard_order,
            num_canonical_nodes=num_canonical_nodes,
            predownload=predownload,
        )
        # 归一化后统一存为扁平 kwargs（每轮 `_build_loader()` 用它构造引擎）
        self._kwargs = dict(
            index_json_path=os.fspath(index_json_path),
            **opts.as_kwargs(),
        )
        # Tier 1 的轮号（dataset 计数为唯一权威，C26/H2）
        self._tier1_epoch = int(epoch)
        self._current_loader: Optional[BatchStream] = None
        self._resume_pending = False
        self._direct_iter_warned = False  # F-18：直接迭代告警只发一次
        self._loader_iterated = False     # F-18：是否曾（经 MegadataDataLoader）迭代过

    def _build_loader(self) -> BatchStream:
        """C20/G1：轮号经 P2 的 `epoch=` 构造参数传入（不写私有属性）。"""
        return BatchStream(**self._kwargs, epoch=self._tier1_epoch)

    def _query_epoch_total(self, epoch: int) -> int:
        """纯函数查询该轮轮长（A3：零副作用——不构造读头、不注册信号）。

        拓扑解析规则与 `_build_loader` 一致（worker 维度动态探测：主进程为 (0,0)，
        得到的正是"本 rank 整轮步数"，即 LR scheduler 要的数）。
        """
        kwargs = self._kwargs
        world_size, rank = _detect_dist(kwargs["world_size"], kwargs["rank"])
        num_workers, worker_id = _detect_workers()
        return int(_epoch_total_batches(
            kwargs["index_json_path"], int(epoch), world_size, rank,
            int(num_workers), int(worker_id), int(kwargs["seed"]),
            kwargs["partition"], int(kwargs["num_canonical_nodes"] or 0),
        ))

    # ---- 生命周期（A1：close 终态 / release 非终态）----
    def release(self) -> None:
        """释放**当前轮**读头（幂等、非终态）：下一轮 `__iter__` 会自动重建。

        正常迭代路径由 `__iter__` 自动调用；训练间隙 / 长评估等资源敏感场景
        可主动调用让出内存与下载线程。
        """
        loader, self._current_loader = self._current_loader, None
        self._resume_pending = False
        if loader is not None:
            loader.close()

    def close(self) -> None:
        """**终态**：释放读头并标记本 dataset 不再可用（与引擎层 close 同语义）。

        之后 `__iter__` / `len()` / `state_dict()` 等明确报错，需重建实例；
        只想释放读头请用 `release()`。
        """
        if self._closed:
            return
        self.release()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"{type(self).__name__} is closed() (terminal state): rebuild the instance; "
                "use release() to free only the current reader."
            )

    def __iter__(self):
        self._ensure_open()
        # F-18：直接迭代（不经 MegadataDataLoader、也不在 worker 内）不会推进轮号——
        # 给出告警（每实例一次），避免"每轮数据顺序相同"被静默接受。
        if (not self._in_worker_context()
                and not self._loader_iterated
                and not self._direct_iter_warned):
            self._direct_iter_warned = True
            warnings.warn(
                "Iterating StreamingDataset directly (without MegadataDataLoader) does **not** advance the epoch counter — "
                "every epoch yields the same data order (known boundary C24-③). Iterate via MegadataDataLoader, "
                "or call `ds.set_epoch(k)` manually and manage epochs yourself.",
                stacklevel=2,
            )
        # load_state_dict 之后首次迭代复用已 seek 好的 loader
        if self._resume_pending and self._current_loader is not None:
            self._resume_pending = False
            return self._iter_with_bump(self._current_loader)
        # 建新一轮 loader 前无条件释放旧读头（不再等 GC 回收 Rust 后台线程）
        self.release()
        self._current_loader = self._build_loader()
        return self._iter_with_bump(self._current_loader)

    def _iter_with_bump(self, loader: BatchStream):
        """worker 上下文：生成器结束时推进本进程副本的轮号（仅 persistent worker 需要）。

        推进分工由 `_worker_epoch_bump` 显式决定（构造 MegadataDataLoader 时按
        `persistent_workers` 设置并随 fork 继承）：非 persistent 的 worker 副本短命，
        轮号由主进程 teardown 推进、随下一次 fork 带入；persistent worker 不再 fork，
        必须由本进程自推进。
        """
        in_worker = self._in_worker_context()
        try:
            for batch in loader:
                yield batch
        finally:
            if in_worker and self._worker_epoch_bump:
                self._advance_tier1_epoch()

    @staticmethod
    def _in_worker_context() -> bool:
        try:
            from torch.utils.data import get_worker_info
            return get_worker_info() is not None
        except Exception:
            return False

    def _cancel(self):
        if self._current_loader is not None:
            self._current_loader._cancel()

    def cancel(self):
        self._cancel()

    def state_dict(self):
        """⚠️ `num_workers>0` 时主进程拿不到 worker 内的 loader，这里返回 `{}`。

        精度信息由 `MegadataDataLoader.state_dict()["resume_granularity"]` 承载；
        此处**不返回 partial dict**——非空 dict 是 truthy，会误导下游的
        `if dataset_state:` 判断（A5：宁可返回空，不可返回假信号）。
        """
        if self._current_loader is not None:
            return self._current_loader.state_dict()
        return {}

    def load_state_dict(self, state):
        # F-01：先把 state["epoch"] 回填进 Tier-1 轮号（设计文档 P5/C27-H3——
        # dataset.epoch 是唯一权威，必须覆盖构造期 epoch=），再构造 loader。
        # 缺了这一步：恢复轮跑完后 teardown 的 `+= 1` 会让下一轮回到已消费的轮号，整轮重放。
        self._ensure_open()
        if isinstance(state, dict) and "epoch" in state:
            self._tier1_epoch = int(state["epoch"])
        self._current_loader = self._build_loader()
        self._current_loader.load_state_dict(state)
        self._resume_pending = True

    def __len__(self) -> int:
        """当前轮的 batch 数（A3：纯函数查询，不构造读头）。

        口径与 `BatchStream.__len__` 一致：恢复待消费时为"剩余步数"，
        否则为整轮长；`num_workers>0` 时仍是**本 rank 整轮步数**
        （worker 内切分由 Rust 侧完成，主进程探测到的 worker 维度为 (0,0)）。
        """
        self._ensure_open()
        if self._current_loader is not None:
            return len(self._current_loader)
        return self._query_epoch_total(self._tier1_epoch)

    def __getstate__(self) -> Dict[str, Any]:
        """spawn 平台（Windows / macOS）的序列化：**丢弃当前读头**（pyo3 对象不可 pickle）。

        spawn 模式下 `DataLoader(num_workers>0)` 会把 dataset 序列化给 worker：
        `load_state_dict` 之后 `_current_loader` 持有 Rust 读头，直接 pickle 会在 worker
        启动时崩（`cannot pickle 'megadata_rs.RustStreamReader' object`）。这里丢掉读头、
        清掉 `_resume_pending`，worker 内 `__iter__` 会按 `epoch` 重建——续训精度退化为
        "从该轮开头重读"（与 `num_workers>0` 既有的轮级精度一致，MIGRATION 文档已声明）。
        fork 平台不走本方法（不序列化），行为不变。
        """
        state = self.__dict__.copy()
        dropped = state.pop("_current_loader", None) is not None
        if dropped and state.get("_resume_pending"):
            already_warned = self.__dict__.get("_spawn_drop_warned", False)
            # 同一 dataset 会被多个 worker / 多个 epoch 反复序列化：告警按实例只发一次
            # （否则 2 epoch × 2 worker 会重复 4 次）。降级标记置位于原对象与副本：
            # 供 `MegadataDataLoader.state_dict()` 自述续训精度（spawn 下 worker 从轮头重读）。
            state["_spawn_drop_warned"] = True
            state["_spawn_resume_degraded"] = True
            self.__dict__["_spawn_drop_warned"] = True
            self.__dict__["_spawn_resume_degraded"] = True
            if not already_warned:
                warnings.warn(
                    "spawn serialization dropped the seeked reader (Rust handles cannot be pickled across processes): "
                    f"the worker will re-read from the start of epoch {self._tier1_epoch}; for batch-precise resume use "
                    "num_workers=0 or a fork platform (Linux).",
                    stacklevel=2,
                )
        state["_current_loader"] = None
        state["_resume_pending"] = False
        return state

    def _advance_tier1_epoch(self) -> None:
        """Tier-1 轮号推进的**唯一**实现点（主进程 teardown 与 persistent worker 共用）。"""
        self._tier1_epoch += 1

    @property
    def epoch(self) -> int:
        """Tier 1 轮号（dataset 计数，唯一权威；mode C 下主进程值不可靠）。"""
        return self._tier1_epoch

    def set_epoch(self, epoch: int) -> None:
        """F-18：手动设置轮号——下一次迭代从该轮开始（用于自定义 loop / 手动管理轮号）。

        活跃迭代会话中调用抛 `RuntimeError`（生效时序未定义）。
        """
        self._ensure_open()
        if int(epoch) < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        loader = self._current_loader
        if loader is not None and loader._active_session:
            raise RuntimeError("cannot set_epoch() during an active iteration session: finish the current session first")
        if loader is not None:
            # 已建 loader 的轮号冻结在构造期：释放，让下一轮按新轮号重建
            self.release()
        self._tier1_epoch = int(epoch)


class MixStreamingDataset(StreamingDataset):
    """多源混采的 Tier-1 适配（F-31）：`MixBatchStream` 的 `IterableDataset` 封装。

    与 `StreamingDataset` 相同的轮号推进 / `state_dict` / `load_state_dict` 语义
    （直接继承其 `__iter__` / `_iter_with_bump` / `set_epoch` / `close`），
    区别只在 `_build_loader()` 构造的是 `MixBatchStream`——多源场景因此也能拿到
    `num_workers` 与 HF Trainer 集成。

    ```python
    ds = MixStreamingDataset(
        [{"path": "out/pt/index.json", "weight": 0.8},
         {"path": "out/sft/index.json", "weight": 0.2}],
        batch_size=8, max_seq_len=4096, steps_per_epoch=1000,
    )
    dl = MegadataDataLoader(ds, num_workers=2)
    ```

    注意：多 worker 下与 `StreamingDataset` 相同的限制（主进程拿不到 worker 内的
    引擎快照 → `state_dict()["dataset"] == {}`，精确续训请用 `num_workers=0`）。
    """

    def __init__(self, configs: List[DatasetConfig], *,
                 weights: Optional[List[float]] = None,
                 steps_per_epoch: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 42,
                 shuffle: Any = True,
                 partition: str = "round_robin",
                 shard_order: str = "random",
                 num_canonical_nodes: Optional[int] = None,
                 predownload: Optional[int] = None,
                 epoch: int = 0,
                 options: Optional["LoaderOptions"] = None):
        IterableDataset.__init__(self)  # 不经 StreamingDataset.__init__（无单源 index 路径）
        opts = _resolve_options(
            "MixStreamingDataset", options, epoch=epoch,
            batch_size=batch_size, max_seq_len=max_seq_len,
            world_size=world_size, rank=rank, seed=seed, shuffle=shuffle,
            partition=partition, shard_order=shard_order,
            num_canonical_nodes=num_canonical_nodes,
            predownload=predownload,
        )
        self._mix_configs = [dict(c) for c in configs]
        self._mix_paths = [c["path"] for c in self._mix_configs]
        self._mix_weights = None if weights is None else list(weights)
        self._mix_steps_per_epoch = steps_per_epoch
        self._mix_common = opts.as_kwargs()
        # Tier 1 的轮号（dataset 计数为唯一权威，与 StreamingDataset 同约定；F-01 回填同样生效）
        self._tier1_epoch = int(epoch)
        self._current_loader: Optional[MixBatchStream] = None
        self._resume_pending = False
        self._direct_iter_warned = False
        self._loader_iterated = False     # F-18：同 StreamingDataset

    def _build_loader(self) -> MixBatchStream:
        return MixBatchStream(
            self._mix_configs,
            weights=self._mix_weights,
            steps_per_epoch=self._mix_steps_per_epoch,
            epoch=self._tier1_epoch,
            **self._mix_common,
        )

    def __len__(self) -> int:
        """当前轮的 batch 数。

        确定性模式（`steps_per_epoch=N`）直接由 N 给出；自然模式未建 loader 时按
        **纯函数**（`epoch_total_batches`，与 Rust 读取序同源）逐子流求和——不构造
        mixer / 读头、不注册信号、不做跨 rank 校验（旧实现在这里临时构造完整
        `MixBatchStream`，含集合通信：rank 非对称调用 `len()` 会 hang，且每次
        `len()` 都重跑构造期全部重活）。
        """
        self._ensure_open()
        if self._current_loader is not None:
            return len(self._current_loader)
        if self._mix_steps_per_epoch is not None:
            return int(self._mix_steps_per_epoch)
        world_size, rank = _detect_dist(self._mix_common["world_size"], self._mix_common["rank"])
        num_workers, worker_id = _detect_workers()
        return int(sum(
            _epoch_total_batches(
                p, self._tier1_epoch, world_size, rank,
                int(num_workers), int(worker_id), int(self._mix_common["seed"]),
                self._mix_common["partition"],
                int(self._mix_common["num_canonical_nodes"] or 0),
            )
            for p in self._mix_paths
        ))

    def verify_ddp_consistency(self) -> None:
        """精确的跨 rank 轮长校验（**集合通信**：所有 rank 必须对称调用）。

        自然模式 + DDP 的构造期静态判定无法覆盖"shard 尺寸不齐"的数据集（只告警），
        需要确凿结论时在训练循环开始前的同步点显式调用；已建 loader 时直接委托，
        否则临时构造一个（本方法本来就是显式重操作，不进构造器 / `len()`）。
        """
        self._ensure_open()
        if self._mix_steps_per_epoch is not None:
            return  # 确定性模式天然同步
        loader = self._current_loader
        if loader is not None:
            loader.verify_ddp_consistency()
            return
        tmp = self._build_loader()
        try:
            tmp.verify_ddp_consistency()
        finally:
            tmp.close()


class MegadataDataLoader(DataLoader):
    """megadata 专属 DataLoader（`torch.utils.data.DataLoader` 的**受限子集**）。

    保留 DataLoader 身份是刻意取舍：HF `Trainer` / Lightning 依赖 `DataLoader`
    类型做 `num_workers` / `len()` / 迭代协议协商。但本类是**预打包契约**的实现——
    batch 已在构建期按 (batch_size, max_seq_len) 烧进 shard，因此：

    - 拒绝 `batch_size` / `shuffle` / `sampler`（传了直接 `ValueError`，不静默忽略）；
    - 强制 `batch_size=None` + identity collate（不再二次打包）；
    - 额外提供 `state_dict()` / `load_state_dict()` / `set_epoch()` / `cancel()` /
      `close()`（终态，转发 `dataset.close()`）/ `with`。

    **逃生门**（自担契约风险；直接 `torch.utils.data.DataLoader` 会失去 teardown
    轮号推进，每轮数据顺序相同且无提示）::

        dl = torch.utils.data.DataLoader(ds, batch_size=None, collate_fn=lambda b: b)
        ds.set_epoch(k)      # ⚠️ 必须自行推进轮号

    增值：
    - 追踪 `num_batches_yielded` 并透出 state_dict/load_state_dict；
    - `num_workers>0` 可用（`StreamingDataset` 在 worker 内构造 reader）；
    - **teardown 推进 dataset 轮号**（C17 mode A/B）——保证 HF/Lightning 每轮
      `iter(dl)` 拿到的是新的一轮，而不是永远 epoch 0。
    """

    # F-03：这三个参数会静默破坏"batch 已预打包"契约（USAGE.md 头号陷阱），显式拒绝
    _REJECTED_KWARGS = ("batch_size", "shuffle", "sampler")

    def __init__(self, dataset, num_workers: int = 0, **kwargs):
        conflicts = [k for k in self._REJECTED_KWARGS if k in kwargs]
        if conflicts:
            raise ValueError(
                f"MegadataDataLoader does not accept {conflicts}: batches are pre-packed at build time "
                "(batch_size × max_seq_len baked into shards); passing them again risks nested batches / misaligned dims. "
                "Shuffling, sharding and sampling are owned by megadata's own seed/partition. "
                "If you really need these arguments, use torch.utils.data.DataLoader directly (it breaks the pre-packed contract)."
            )
        kwargs["batch_size"] = None            # 硬置：用户传了上面已报错，这里无覆盖风险
        kwargs.setdefault("collate_fn", _identity_collate)
        kwargs["num_workers"] = num_workers
        super().__init__(dataset, **kwargs)
        self._num_batches_yielded = 0
        # 轮号推进分派（A1：由显式配置决定，不用幂等键事后去重）：
        # persistent worker 不再 fork，其 dataset 副本必须自推进轮号（否则永远 epoch 0）；
        # 非 persistent 的 worker 副本短命，轮号统一由主进程 teardown 推进、随 fork 带入。
        if isinstance(dataset, StreamingDataset):
            dataset._worker_epoch_bump = bool(kwargs.get("persistent_workers", False))

    def __iter__(self):
        # F-18：标记"本 dataset 经 DataLoader 迭代过"，StreamingDataset 据此不误报
        # "直接迭代不推进轮号"的告警（实例标志，无 TLS 残留问题）。
        ds = getattr(self, "dataset", None)
        if isinstance(ds, StreamingDataset):
            ds._ensure_open()
            ds._loader_iterated = True
        try:
            for batch in super().__iter__():
                self._num_batches_yielded += 1
                yield batch
        finally:
            # 任何结束方式（自然读完 / break / max_steps / 异常）都把该轮计为已消费
            ds = getattr(self, "dataset", None)
            if isinstance(ds, StreamingDataset):
                ds._advance_tier1_epoch()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def cancel(self):
        cancel = getattr(self.dataset, "cancel", None)
        if callable(cancel):
            cancel()

    def close(self):
        """转发到 `dataset.close()`（**终态**：之后迭代/查询明确报错，需重建实例）。"""
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()

    def set_epoch(self, epoch: int) -> None:
        """F-18：转发到 dataset.set_epoch（HF/Lightning 手动管理轮号的约定入口）。"""
        fn = getattr(self.dataset, "set_epoch", None)
        if callable(fn):
            fn(epoch)

    def verify_ddp_consistency(self) -> None:
        """转发到 dataset 的**精确**跨 rank 轮长校验（集合通信：所有 rank 必须对称调用）。

        仅 `MixStreamingDataset` 自然模式 + DDP 有意义（其他情况 no-op）；
        详见 `MixBatchStream.verify_ddp_consistency`。
        """
        fn = getattr(self.dataset, "verify_ddp_consistency", None)
        if callable(fn):
            fn()

    def state_dict(self):
        """快照：轮号 + 已产出批数 + dataset 引擎状态（若有）+ **精度自述**。

        `resume_granularity`（A5 能力缺口显式化）：
        - `"batch"`：`dataset` 部分非空且未被 spawn 降级（`num_workers=0`），
          恢复可精确到 batch；
        - `"epoch"`：主进程拿不到 worker 内的读头位置（`num_workers>0`），
          或 dataset 已被 spawn 序列化降级（`_spawn_resume_degraded`）——
          恢复从该轮开头重读（方向为重放而非跳过）。

        键结构（P2-9）：Tier-1 轮号嵌套为 `state["tier1"]["epoch"]`——
        原点号键 `"dataset.epoch"` 已移除（dict 键伪装属性路径，易与
        `state["dataset"]["epoch"]` 混淆）；`load_state_dict` 对旧格式静默兜底。
        """
        dataset_state = None
        if hasattr(self.dataset, "state_dict"):
            dataset_state = self.dataset.state_dict()
        degraded = bool(getattr(self.dataset, "_spawn_resume_degraded", False))
        return {
            "num_batches_yielded": self._num_batches_yielded,
            "dataset": dataset_state,
            "tier1": {"epoch": getattr(self.dataset, "epoch", 0)},
            "resume_granularity": "batch" if (dataset_state and not degraded) else "epoch",
        }

    def load_state_dict(self, state):
        dataset_state = state.get("dataset") if isinstance(state, dict) else None
        yielded = int(state.get("num_batches_yielded", 0)) if isinstance(state, dict) else 0
        if isinstance(self.dataset, StreamingDataset):
            self.dataset._ensure_open()
        # A5：精度自述 + 只在"确实丢过位置"时告警（未迭代就 save 的轮级快照不误报）
        gran = state.get("resume_granularity") if isinstance(state, dict) else None
        if gran is None:
            gran = "batch" if dataset_state else "epoch"
        if gran == "epoch" and yielded and not dataset_state:
            warnings.warn(
                "snapshot has **epoch-level** granularity only: with `num_workers>0` the reader is built inside the worker "
                "process, so the main process cannot know this epoch's reader position. After resume it re-reads from the "
                f"start of the epoch; the {yielded} batches already consumed in this epoch will repeat (data order unchanged; "
                "replay, not skip). For batch-precise resume use num_workers=0.",
                stacklevel=2,
            )
        if dataset_state and hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(dataset_state)
        # P2-9：Tier-1 轮号优先读新嵌套键 `tier1.epoch`；旧点号键 `dataset.epoch`
        # 静默兜底（pre-release 零外部用户，兜底只为跨版本过渡：不告警、不设旋钮）。
        t1 = state.get("tier1") if isinstance(state, dict) else None
        t1_epoch = None
        if isinstance(t1, dict) and "epoch" in t1:
            t1_epoch = int(t1["epoch"])
        elif isinstance(state, dict) and "dataset.epoch" in state:
            t1_epoch = int(state["dataset.epoch"])
        if t1_epoch is not None and isinstance(self.dataset, StreamingDataset):
            # F-01（复核精确化）：Tier-1 轮号是**唯一权威**（C26/H2），
            # 在引擎快照就位后统一回填/覆盖（两个分支——精确续训与多 worker——
            # 以及 MixStreamingDataset 都走这一条）。引擎快照里的 epoch 只是
            # "直接调 dataset.load_state_dict(引擎状态)" 场景的兜底。
            self.dataset._tier1_epoch = t1_epoch
        self._num_batches_yielded = state.get("num_batches_yielded", 0) if isinstance(state, dict) else 0

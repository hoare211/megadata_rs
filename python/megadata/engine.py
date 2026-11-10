# python/megadata/engine.py
"""引擎层加载器（P2-11 自 `dataloader.py` 原样拆出，零行为变更）。

包含：`_EngineBase` 公共基类 + 三个引擎类（BatchStream / MixBatchStream /
RandomAccessBatchReader）、P1 batch 转换、`LoaderOptions` 及其派生的扁平 kwargs
冲突检测表（`_OPTION_DEFAULTS` / `_OPTION_NORMALIZERS` / `_derive_option_defaults` /
`_resolve_options`）。

依赖方向：`engine` → {_manifest, _topology, _options, _signals, _rust}；
Tier-1 torch 适配（tier1.py）在本层之上。
"""
import contextlib
import math
import os
import warnings
import pyarrow as pa
from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, List, Optional, TypedDict

import torch

import megadata._topology as _topology
from megadata._manifest import (
    _check_fingerprint,
    _combined_manifest_fingerprint,
    _effective_modality_tokens,
    _fingerprint_of,
    _load_manifest_cached,
    _manifest_format,
)
from megadata._options import normalize_partition, normalize_shard_order, normalize_shuffle
from megadata._rust import (
    RustStreamReader, RustStreamMixer,
    epoch_total_batches as _epoch_total_batches,
    read_vid_blob as _read_vid_blob,
)
from megadata._signals import MegadataInterrupted, _ACTIVE_INSTANCES, _register
from megadata._topology import (
    _ddp_world_size,
    _detect_dist,
    _detect_workers,
    _natural_mode_length_verdict,
    _warn_if_topology_changed,
)

SCHEMA_VERSION = 3


class DatasetConfig(TypedDict, total=False):
    """单个子流的配置（MixBatchStream 用）。运行期等价于 dict，向后兼容。"""
    path: str
    weight: float


#: 引擎层/Tier-1 共用扁平参数的默认哨兵（`options=` 冲突检测用）：
#: **由下方 `LoaderOptions` 自动推导**（单一事实源），不在此手抄。


@dataclass(frozen=True)
class LoaderOptions:
    """加载参数值对象（#5 配置收敛）：三个引擎类 + 两个 Tier-1 类共用。

    价值：一份参数集合可复用于多个 loader（train/val 同参）；归一化与校验只在一处
    （`shuffle` / `partition` 构造期即归一化，拼错立即 `ValueError`）。

    ```python
    opts = LoaderOptions(batch_size=8, max_seq_len=4096, world_size=64, rank=rank)
    loader = BatchStream("out/pt/index.json", options=opts)
    ```

    与扁平参数的关系：**互斥**（同时传入且值偏离默认 → `ValueError`；
    配置来源必须唯一，避免哪份生效的静默歧义）。旧扁平写法完全可用。
    """
    batch_size: Optional[int] = None
    max_seq_len: Optional[int] = None
    world_size: Optional[int] = None
    rank: Optional[int] = None
    seed: int = 42
    shuffle: Any = True
    partition: str = "round_robin"
    #: F6：epoch 内 shard 访问顺序（与 shuffle 正交；"random" = 历史行为）
    shard_order: str = "random"
    num_canonical_nodes: Optional[int] = None
    predownload: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shuffle", normalize_shuffle(self.shuffle))
        object.__setattr__(self, "partition", normalize_partition(self.partition))
        object.__setattr__(self, "shard_order", normalize_shard_order(self.shard_order))
        # 0 与 None 同义（"取默认 / manifest 值"，与既有扁平签名一致）：predownload=0
        # 表示用 Rust 默认值，batch_size/max_seq_len=0 表示从 manifest 读。
        for name in ("batch_size", "max_seq_len", "predownload"):
            v = getattr(self, name)
            if v is not None and int(v) < 0:
                raise ValueError(
                    f"LoaderOptions.{name} must not be negative (None / 0 = use default or manifest value), got {v!r}"
                )
        if self.num_canonical_nodes is not None and int(self.num_canonical_nodes) < 1:
            raise ValueError(
                f"LoaderOptions.num_canonical_nodes must be >= 1 (None = unspecified), got {self.num_canonical_nodes!r}"
            )
        if int(self.seed) < 0:
            raise ValueError(f"LoaderOptions.seed must be >= 0, got {self.seed!r}")

    def as_kwargs(self) -> Dict[str, Any]:
        """展开为引擎构造的扁平 kwargs（shuffle / partition / shard_order 已是归一化形式）。"""
        return {
            "batch_size": self.batch_size, "max_seq_len": self.max_seq_len,
            "world_size": self.world_size, "rank": self.rank, "seed": self.seed,
            "shuffle": self.shuffle, "partition": self.partition,
            "shard_order": self.shard_order,
            "num_canonical_nodes": self.num_canonical_nodes,
            "predownload": self.predownload,
        }


def _derive_option_defaults() -> Dict[str, Any]:
    """从 `LoaderOptions` 推导"偏离默认即视为显式传入"的对照表（单一事实源）。

    手抄两份默认值会在改 `LoaderOptions` 默认时静默漂移：冲突检测要么误报、
    要么把"显式传值"当成默认值放过（配置来源静默歧义）。
    """
    out: Dict[str, Any] = {}
    for f in fields(LoaderOptions):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:  # pragma: no cover - 当前无 factory 字段
            out[f.name] = f.default_factory()
    return out


_OPTION_DEFAULTS: Dict[str, Any] = _derive_option_defaults()

#: 冲突比较前先归一化（与 `LoaderOptions.__post_init__` 同一套实现）——
#: `shuffle=True` 与 `shuffle="permutation"` 语义相同，必须同判（否则推荐写法会误报冲突）。
_OPTION_NORMALIZERS = {
    "shuffle": normalize_shuffle,
    "partition": normalize_partition,
    "shard_order": normalize_shard_order,
}


def _resolve_options(cls_name: str, options: Optional[LoaderOptions], *,
                     epoch: int = 0, **flat) -> LoaderOptions:
    """五个入口构造器的**统一公共前置**（P1-1 收敛点）：
    `options=` / 扁平参数互斥解析 + 归一化收敛到 `LoaderOptions` + `epoch` 构造期校验。

    冲突判定：扁平实参"**归一化后**偏离默认"即视为显式传入（显式传默认值无行为差异、
    不判冲突；非法值在此处即报带可选值的 `ValueError`）。

    公共参数集与差异声明（五入口 = `BatchStream` / `MixBatchStream` /
    `RandomAccessBatchReader` / `StreamingDataset` / `MixStreamingDataset`）：

    - **公共参数集**（全部五入口同名同义）：`batch_size` / `max_seq_len` /
      `world_size` / `rank` / `seed` / `shuffle` / `partition` / `shard_order` /
      `num_canonical_nodes` / `predownload` / `epoch` / `options`（见使用指南 §4.2）；
    - **差异字段**：`RandomAccessBatchReader` **无 `shuffle` / `shard_order`**
      （无置换表与 shard 访问序语义，`LoaderOptions` 照单接收但有意不消费）；
      `Mix*` 两入口额外有
      `weights` / `steps_per_epoch`（仅 `MixBatchStream` 消费，Tier-1 透传）；
    - 新增/调整公共参数时改三处即可：`LoaderOptions` 字段（+`__post_init__` 校验）、
      `as_kwargs`、各入口显式签名（保持 IDE 可见，不 `**kwargs` 化）。
    """
    if int(epoch) < 0:
        # C31/J2：构造期参数校验（负值会让 pyo3 抛难读的 OverflowError）
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if options is None:
        return LoaderOptions(**flat)
    conflicts: List[str] = []
    for k, v in flat.items():
        if k not in _OPTION_DEFAULTS:
            continue
        norm = _OPTION_NORMALIZERS.get(k)
        if norm is not None:
            v = norm(v)
        if v != _OPTION_DEFAULTS[k]:
            conflicts.append(k)
    if conflicts:
        raise ValueError(
            f"{cls_name} cannot take both options= and flat arguments {conflicts}: "
            "the configuration source must be unique (to avoid silent ambiguity about which one takes effect)."
        )
    return options


def _safe_from_dlpack(obj):
    """
    智能 DLPack 提取器：
    - 针对 PyArrow Array：调用高层 torch.from_dlpack，自动触发 __dlpack__()
    - 针对 Rust 传来的裸 PyCapsule：调用底层 torch._C._from_dlpack 强行消费
    """
    if hasattr(obj, "__dlpack__"):
        return torch.from_dlpack(obj)
    if hasattr(torch._C, "_from_dlpack"):
        return torch._C._from_dlpack(obj)
    from torch.utils.dlpack import from_dlpack as _torch_from_dlpack
    return _torch_from_dlpack(obj)


def read_video_blob(entry: Dict[str, Any]):
    """按需读一条 video 记录的 blob 字节（P3b；D8 "video blob 不进预取窗口"）。

    `entry` 是 batch dict ``videos`` 的条目（含 ``blob_ref=(store_id, offset,
    len)``）；字节经 FFI ``read_vid_blob`` 在 collate 时点读取，返回 per-clip
    Owned DLPack tensor（所有权转移，copy_count=0）。shard 已切换（seek/新
    epoch）后旧条目的 blob_ref 会 fail-fast——需重新取 batch。
    """
    store_id, offset, length = entry["blob_ref"]
    return _safe_from_dlpack(_read_vid_blob(store_id, offset, length))


# ============================================================
# P1：batch 转换
# ============================================================
def _unpack_arrow_zero_copy(pyarrow_batch: pa.RecordBatch) -> Dict[str, torch.Tensor]:
    """通过 O(1) 视图转换，将嵌套 Arrow 结构剥离为 Tensor（零拷贝）。

    写读约定（P1/C9）：`cu_seqlens` 列是 ListArray，**batch 级前缀和只写第 0 行**、
    其余行为空列表（offsets `[0,K,K,...,K]`），故 `arr.values` 恰好还原该前缀和。
    禁止任何"逐行拼接"的重解释。
    """
    batch_tensors = {}
    for column_name in pyarrow_batch.schema.names:
        col = pyarrow_batch.column(column_name)
        num_chunks = getattr(col, "num_chunks", 0)
        if num_chunks > 1:
            # F-27：写侧约定单 chunk（batch 级前缀和只写第 0 行）；多 chunk 时旧实现
            # 只取 chunk(0) 会静默丢数据——这里响亮拒绝，交由上游排查。
            raise ValueError(
                f"column {column_name!r} returned {num_chunks} Arrow chunks (writer contract is a single chunk): "
                "refusing to silently drop data (taking only chunk(0) would drop the rest). Check the dataset / read path."
            )
        arr = col.chunk(0) if num_chunks > 0 else col

        if isinstance(arr.type, pa.FixedSizeListType):
            flat_tensor = _safe_from_dlpack(arr.values)
            batch_tensors[column_name] = flat_tensor.view(len(arr), -1)
        elif isinstance(arr.type, pa.ListType):
            batch_tensors[column_name] = _safe_from_dlpack(arr.values)
        else:
            batch_tensors[column_name] = _safe_from_dlpack(arr)

    return batch_tensors


def _convert_raw_batch(raw_dict: Dict[str, Any], fmt: Optional[str] = None) -> Dict[str, Any]:
    """Rust 返回的 capsule/arrow dict → tensor dict（零拷贝；cu_seqlens 为 int32 tensor）。

    F-10：所有 loader 的 batch 统一携带 `"format"` 键（"pt" / "sft"）——让
    `mask_adapter` 建议的 `packer=batch.get("format")` 防护在单源 / 同构多源 /
    异构混采三种路径下都生效（此前只有异构混采才带该键，单源路径防护永不生效）。
    """
    if "arrow_batch" in raw_dict:
        batch = dict(_unpack_arrow_zero_copy(raw_dict["arrow_batch"]))
    elif "input_ids" in raw_dict:
        batch: Dict[str, Any] = {"input_ids": _safe_from_dlpack(raw_dict["input_ids"])}
        if raw_dict.get("cu_seqlens"):
            batch["cu_seqlens"] = raw_dict["cu_seqlens"]
        # 文档级旁路列（列扩展，构建期 extra_columns=True 才有）：
        # doc_ids = int32，loss_weights = float16，两者都以 DLPack capsule 零拷贝透出。
        for k in ("doc_ids", "loss_weights"):
            if raw_dict.get(k) is not None:
                batch[k] = _safe_from_dlpack(raw_dict[k])
        # 多模态图像 sidecar（P1-T4b）：条目 list[dict]（bytes 为 DLPack capsule）
        if raw_dict.get("images") is not None:
            batch["images"] = raw_dict["images"]
        # 多模态视频 sidecar（P3b）：条目 list[dict]（**不含字节**——blob_ref
        # 按需读；D8 "video blob 不进预取窗口"的引擎层形态）
        if raw_dict.get("videos") is not None:
            batch["videos"] = raw_dict["videos"]
        # 三模态对齐视图（P3c）：条目 list[dict]（每窗口一条；请求 align 才有）
        if raw_dict.get("av_align") is not None:
            batch["av_align"] = raw_dict["av_align"]
    else:
        raise RuntimeError("unrecognized data format returned")

    # P1：cu_seqlens / row_lens 张量化（int32；下游 mask_adapter 会再归一化为 int64）
    # row_lens：SFT 数据集自描述的"每行有效长度"列（A4），旧数据集无该键。
    for k in ("cu_seqlens", "row_lens"):
        if k in batch:
            batch[k] = torch.as_tensor(batch[k], dtype=torch.int32)
    if fmt is not None:
        batch["format"] = fmt
    return batch


# ============================================================
# 引擎层公共基类
# ============================================================
class _SessionIterator:
    """包装迭代生成器，保证"活动会话"标志一定被释放。

    ★ 评审 §3.9(b)：`_EngineBase.__iter__` 用单个 Rust 游标，必须阻止并发会话，
    靠的是 `_active_session` 标志。但若把它交给裸生成器：

    - 生成器**正常跑完 / 被 close()**（for 循环 break 后 CPython 会close）→
      `finally` 执行，标志清掉；
    - 生成器**从未启动就被丢弃**（`it = stream.__iter__()` 之后什么都不做）→
      CPython 对未启动生成器调 `close()` **不会执行函数体**，`finally` 不跑，
      标志永久卡在 True → 后续所有 `__iter__` / `load_state_dict` 抛 RuntimeError，
      且**无法恢复**（只能重建实例）。

    `__del__` 在对象被 GC 时兜底清除标志，覆盖上面两种形态。`close()` 供显式使用。
    重复释放是幂等的（生成器自己的 `finally` 也会清一次）。
    """

    __slots__ = ("_gen", "_on_end", "_closed")

    def __init__(self, gen, on_end) -> None:
        self._gen = gen
        self._on_end = on_end
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        # 生成器跑完时会走它自己的 finally；这里只在 StopIteration 路径兜底，
        # 保证"正常结束但 finally 尚未执行"（例如被 close）时标志也清掉。
        try:
            return next(self._gen)
        except StopIteration:
            self._release()
            raise

    def close(self) -> None:
        """显式结束会话（幂等）。"""
        if not self._closed:
            self._closed = True
            try:
                self._gen.close()
            finally:
                self._on_end()

    def _release(self) -> None:
        if not self._closed:
            self._closed = True
            self._on_end()

    def __del__(self) -> None:
        # 解释器退出时属性可能已被拆除，用 contextlib.suppress 兜住一切
        with contextlib.suppress(Exception):
            self._release()
            self._gen.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _EngineBase:
    """引擎层公共基类：fork 安全的引擎生命周期（模板方法）+ 轮号协议 + 快照骨架。

    A1 单一状态权威：Rust 引擎对象的持有、pid 检查、重建、释放只在基类实现一次；
    子类只声明"怎么造"（`_make_engine`）与"重建后修什么"（`_on_engine_rebuilt`），
    消除三份 fork 安全样板各自的漂移风险。
    """
    #: P1-1 差异声明：本入口是否消费 `shuffle`。仅 `RandomAccessBatchReader` 为 False
    #:（无置换表语义——`LoaderOptions` 照单接收该字段但有意不存实例属性，
    #: `_make_engine` 固定传 False；检测面声明见其类 docstring）。
    _consumes_shuffle: bool = True
    #: F6 差异声明：本入口是否消费 `shard_order`（同样仅 `RandomAccessBatchReader`
    #: 为 False——按 idx 直取，无"shard 访问顺序"语义）。
    _consumes_shard_order: bool = True
    #: 以下四个字段是**类属性哨兵**：仅作为"子类构造器尚未执行到 super().__init__()
    #: / 未显式赋值时"的安全默认值（读到它们不会 AttributeError）。
    #: 子类约定（P3 文档化）：每个子类必须在自己的构造器里显式写成**实例属性**
    #: （如 `self._engine_obj = None`），否则多个实例会共享类属性状态
    #: （一个实例 close() 后另一个实例可能读到 `_closed=True`）。
    _engine_obj = None  # 子类填 Rust 引擎对象（reader / mixer）
    _engine_pid = -1
    _closed = False
    _active_session = False

    # ---- 子类接口 ----
    def _make_engine(self):
        """构造 Rust 引擎对象（reader / mixer）。子类必须实现。"""
        raise NotImplementedError

    def _store_common_options(self, opts: LoaderOptions) -> None:
        """`LoaderOptions` → 实例属性的**公共拷贝**（P1-1 收敛点）。

        `batch_size` / `max_seq_len` 不在此列：各入口的 manifest fallback 与
        跨子流一致性校验逻辑不同，留在各自构造器。`shuffle` / `shard_order` 按
        `_consumes_shuffle` / `_consumes_shard_order` 差异声明决定是否落实例属性
        （`RandomAccessBatchReader` 有意不消费）。
        """
        self._seed = int(opts.seed)
        self._partition = opts.partition            # 已归一化（"round_robin"/"relaxed"）
        self._num_canonical_nodes = opts.num_canonical_nodes
        self._predownload = opts.predownload
        if self._consumes_shuffle:
            self._shuffle = opts.shuffle            # LoaderOptions 已归一化（bool）
        if self._consumes_shard_order:
            self._shard_order = opts.shard_order    # F6：已归一化（"random"/"sequential"）

    def _on_engine_rebuilt(self, *, forked: bool) -> None:
        """引擎重建后的状态修复钩子（默认无操作）。

        `forked=True` 表示"之前有引擎对象、现在 pid 变了"（fork 后重建），
        此时游标已归零，派生状态（会话计数 / 顺序游标）需要同步失效。
        """

    # ---- 引擎生命周期（唯一实现点） ----
    @property
    def _engine(self):
        """fork 安全地取（必要时重建）Rust 引擎对象。

        同 pid：直接复用；pid 变化（fork 到 worker）：cancel 旧对象并重建，
        重建后游标归零（子类在 `_on_engine_rebuilt` 里同步派生状态）。
        """
        pid = os.getpid()
        if self._engine_obj is not None and self._engine_pid == pid:
            return self._engine_obj
        forked = self._engine_obj is not None
        self._dispose_engine()
        self._engine_obj = self._make_engine()
        self._engine_pid = pid
        self._on_engine_rebuilt(forked=forked)
        return self._engine_obj

    def _dispose_engine(self) -> None:
        """cancel + 置空（幂等、吞异常；重建与 close 共用同一释放路径）。"""
        obj, self._engine_obj, self._engine_pid = self._engine_obj, None, -1
        if obj is not None:
            with contextlib.suppress(Exception):
                obj.cancel()

    def _cancel(self):
        obj = self._engine_obj
        if obj is not None:
            with contextlib.suppress(Exception):
                obj.cancel()

    def _end_session(self) -> None:
        """结束活动会话（幂等）。由 `_SessionIterator.__del__` / `close` / `__next__`
        的 StopIteration 路径调用。

        ★ 评审 §3.9(b)：放在 `_EngineBase` 而不是 `BatchStream`，因为
        `BatchStream` 与 `MixBatchStream` 两个入口都要用（单引擎 reader 与多源
        mixer 共用同一个"单游标、禁并发"的会话协议）。
        """
        self._active_session = False

    def _close_engine(self):
        self._dispose_engine()
        self._active_session = False

    def cancel(self):
        """请求取消：正在阻塞的下一次取数会抛 MegadataInterrupted（可重复调用）。

        **取消是终态**：被取消的实例不再产数据，需继续训练请重建实例
        （每实例取消令牌无对外 reset 入口，见 F-33/使用指南 §5.4）。
        """
        self._cancel()

    def close(self):
        """释放底层读头并从信号注册表注销。

        **close 是终态**：之后 `__iter__` / `get_batch` 明确报错，需继续使用请重建实例。
        设计为可重复调用（幂等）；`with BatchStream(...) as l:` 退出时自动调用。
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._close_engine()
        finally:
            _ACTIVE_INSTANCES.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"{type(self).__name__} is closed(): the megadata reader enters a terminal state after close; "
                "rebuild the instance (see the usage guide for close semantics)."
            )

    # ---- 轮号协议（P2：set_epoch 下沉；推进收敛到唯一实现点） ----
    def set_epoch(self, epoch: int) -> None:
        """手动设定轮号（**下一次 `__iter__` 将产出的轮**）。

        活跃迭代会话中调用抛 `RuntimeError`（生效时序未定义——与
        `load_state_dict` 同款"响亮拒绝优于静默歧义"）。
        `MixBatchStream` 会重建 mixer（Rust 侧 `start_epoch` 是构造期落点）。
        """
        self._ensure_open()
        if self._active_session:
            raise RuntimeError(
                "cannot set_epoch() during an active iteration session: effect ordering would be undefined "
                "(finish the current session first)"
            )
        v = int(epoch)
        if v < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self._set_epoch_checked(v)

    def _set_epoch_checked(self, epoch: int) -> None:
        """子类实现：写入轮号并失效派生状态（续位点 / 引擎重建 / 轮长重算）。"""
        self._epoch = epoch

    def _advance_epoch(self) -> int:
        """轮号推进的**唯一**实现点（轮末由引擎内部调用）。"""
        self._epoch += 1
        return self._epoch

    # ---- 快照骨架（P5）----
    def _base_state(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "seed": self._seed,
            # 拷贝：避免调用方改 state 时把 loader 自己的指纹也改掉（指纹校验会失效）
            "manifest_fingerprint": dict(self._fingerprint),
            "topology": {
                "world_size": self._world_size,
                "rank": self._rank,
                "partition": self._partition,
                # F6：shard 访问顺序进快照，恢复时检测变更（读序 → 断点不可平移）
                "shard_order": self._shard_order,
                "num_canonical_nodes": self._num_canonical_nodes,
                "num_workers": self._worker_dims[0] if self._worker_dims else 0,
                "worker_id": self._worker_dims[1] if self._worker_dims else 0,
            },
        }

    @staticmethod
    def _require_schema(state: Dict[str, Any]) -> None:
        schema = state.get("schema") if isinstance(state, dict) else None
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"incompatible state_dict (schema={schema}, expected {SCHEMA_VERSION}). "
                "v3 is a full reset with no legacy-format compat path: retrain with v3."
            )


# ============================================================
# 1. BatchStream：单源流式（epoch-per-iter）
# ============================================================
class BatchStream(_EngineBase):
    """单源流式加载器（训练主力）。

    ```python
    for epoch in range(3):        # 轮数谁用谁数
        for batch in loader:      # 每个 for = 恰好一个 epoch
            train_step(batch)     # 置换表按 (seed, epoch) 自动更换
    ```
    """

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
        opts = _resolve_options(
            "BatchStream", options, epoch=epoch,
            batch_size=batch_size, max_seq_len=max_seq_len,
            world_size=world_size, rank=rank, seed=seed, shuffle=shuffle,
            partition=partition, shard_order=shard_order,
            num_canonical_nodes=num_canonical_nodes,
            predownload=predownload,
        )
        self._index_json_path = os.fspath(index_json_path)
        # F-26：走进程内缓存（StreamingDataset 每轮重建本类，不再每轮重读 manifest）
        manifest = _load_manifest_cached(self._index_json_path)

        self._batch_size = int(opts.batch_size or manifest["global_stats"].get("batch_size") or 0)
        self._max_seq_len = int(opts.max_seq_len or manifest.get("max_seq_len") or 0)
        if not self._batch_size or not self._max_seq_len:
            raise ValueError(
                "batch_size / max_seq_len were not passed explicitly and are missing from the manifest (old dataset?). "
                "Pass the loading parameters explicitly."
            )
        self._store_common_options(opts)  # P1-1 公共拷贝（seed/shuffle/partition/ncn/predownload）
        self._format = _manifest_format(manifest)
        self._fingerprint = _fingerprint_of(self._index_json_path, manifest)
        self._world_size, self._rank = _detect_dist(opts.world_size, opts.rank)

        # P3 游标双轨制
        self._epoch = int(epoch)
        self._restored_pos: Optional[int] = None
        self._produced = 0
        self._session_start = 0
        self._active_session = False

        # F-30：`__len__` 轮长缓存（同 pid 内同 epoch 的轮长不变；fork 后按 pid 失效）
        self._len_cache: Dict[int, int] = {}
        self._len_cache_pid = -1

        self._engine_obj = None
        self._engine_pid = -1
        self._worker_dims: Optional[tuple] = None
        self._ensure_reader()
        _register(self)

    # ---- P2 ----
    @staticmethod
    def _normalize_shuffle(value: Any) -> bool:
        """shuffle 归一化：`True/"permutation" → True`；`False/"none" → False`；其余报错。

        F-11：收敛到 `megadata._options.normalize_shuffle` 的唯一实现，与 `MixBatchStream` /
        `SimulationConfig` 共享同一取值集合（保留本方法仅为兼容既有调用点/测试）。
        """
        return normalize_shuffle(value)

    # ---- 引擎生命周期（fork 安全；模板方法见 _EngineBase）----
    def _make_engine(self) -> RustStreamReader:
        num_workers, worker_id = _detect_workers()
        self._worker_dims = (num_workers, worker_id)
        return RustStreamReader(
            self._index_json_path, self._batch_size, self._max_seq_len,
            self._world_size, self._rank, self._seed,
            num_workers, worker_id, self._shuffle,
            partition=self._partition,
            shard_order=self._shard_order,
            num_canonical_nodes=self._num_canonical_nodes or 0,
            predownload=self._predownload or 0,
        )

    def _ensure_reader(self) -> RustStreamReader:
        """类型化别名：fork 安全的读头获取（实现见 `_EngineBase._engine`）。
        """
        return self._engine

    def _on_engine_rebuilt(self, *, forked: bool) -> None:
        """fork 后读头重建、游标归零：会话内计数同步失效（P5 reset_cursor 语义）。"""
        if forked:
            self._session_start = 0
            self._produced = 0

    def _set_epoch_checked(self, epoch: int) -> None:
        # 轮号变了：一次性续位点失效（否则下一次迭代会 seek 到错误位置）
        self._epoch = epoch
        self._restored_pos = None

    # ---- P1 ----
    @property
    def format(self) -> str:
        """数据集逻辑格式（"pt" / "sft"），来自 manifest。"""
        return self._format

    @property
    def epoch(self) -> int:
        """当前轮号（下一次 `__iter__` 将产出的轮）。"""
        return self._epoch

    # ---- P3 ----
    def __iter__(self):
        self._ensure_open()
        if self._active_session:
            raise RuntimeError(
                "an active iteration session already exists: the megadata reader is a single Rust cursor and "
                "concurrent sessions would interleave seeks. Let the current session finish first "
                "(read to end / break then close / GC)."
            )
        start = self._restored_pos if self._restored_pos is not None else 0
        self._restored_pos = None          # 一次性续位，消费即清空
        # ★ 评审 §3.9(b)：seek 从生成器体内提到这里（原来在 `_iterate` 的第一次
        #   `__next__` 才执行）。这样 seek 失败在 `__iter__` 当场抛出（此时还没置位），
        #   不会留下"会话已开但游标没动"的残留状态。
        self._ensure_reader().seek_to_epoch_batch(self._epoch, start)
        self._session_start = start
        self._produced = 0
        self._active_session = True
        # ★ 评审 §3.9(b)（续）：把生成器包进 `_SessionIterator`，解决"迭代器被创建
        #   却从未迭代 → `finally: _active_session = False` 永不执行 → 实例被永久
        #   卡在活动会话、后续 `__iter__`/`load_state_dict` 全部 RuntimeError"。
        #
        #   为什么裸生成器解决不了：CPython 对**从未启动**的生成器调用 `close()`
        #   不会执行函数体（连 `try` 块都不进），所以 `finally` 不会跑。实测形态就是
        #   `it = stream.__iter__` 后不做任何事，这个句柄就废了。
        #   包装类用 `__del__` 释放会话标志，覆盖"从未启动"与"中途丢弃"两种情况。
        return _SessionIterator(self._iterate(), self._end_session)

    def _iterate(self):
        reader = self._ensure_reader()
        try:
            while True:
                try:
                    raw_dict = reader.next_batch_capsules()
                except KeyboardInterrupt as e:
                    raise MegadataInterrupted("megadata stream interrupted") from e
                if raw_dict is None:
                    # 本轮读完（Rust at_eof）：推进轮号，游标回到下一轮开头
                    self._epoch += 1
                    self._session_start = 0
                    self._produced = 0
                    return
                # 先计数再产出：生成器在 yield 处挂起，"yield 之后自增"会让
                # state_dict() 少算 1（已交付但未计数）——checkpoint 最典型的 ±1 静默 bug。
                self._produced += 1
                yield _convert_raw_batch(raw_dict, self._format)
        finally:
            self._active_session = False

    def _query_epoch_total(self, epoch: int) -> int:
        """纯函数查询第 `epoch` 轮本 (rank, worker) 的轮长（A3：零副作用）。

        参数与 `_make_engine` 的读头构造保持一致（worker 维度动态探测），
        不构造读头、不启动预取线程、不注册信号。
        """
        num_workers, worker_id = _detect_workers()
        return int(_epoch_total_batches(
            self._index_json_path, int(epoch), self._world_size, self._rank,
            int(num_workers), int(worker_id), self._seed, self._partition,
            int(self._num_canonical_nodes or 0),
        ))

    def _epoch_total_for(self, epoch: int) -> int:
        """第 `epoch` 轮本 (rank, worker) 的轮长（F-30：进程内缓存，fork 后按 pid 失效）。"""
        pid = os.getpid()
        if pid != self._len_cache_pid:
            self._len_cache.clear()
            self._len_cache_pid = pid
        total = self._len_cache.get(epoch)
        if total is None:
            total = self._query_epoch_total(epoch)
            self._len_cache[epoch] = total
        return total

    def __len__(self) -> int:
        """当前轮将产出的 batch 数（R3）。

        F-08（协议决策 ①）：口径统一为"**下一次迭代实际会产出多少**"——
        `load_state_dict` 恢复后返回 `轮长 - 恢复位置`（与 MixBatchStream 确定性模式
        同口径：HF 侧按 len 排的 LR scheduler 不会与真实步数错位）；恢复位置被消费
        （`__iter__` 清空）后恢复为整轮长。

        轮末边界（`in_epoch_pos == 轮长`）由 `load_state_dict` / `state_dict` 统一
        归一化为下一轮开头（C22/G3 同款口径），因此恢复后恒 `>= 0`；这里用
        `max(0, ...)` 兜底，不再返回"说谎的 1"（旧实现 `max(1, ...)` 会让恰好
        轮末的 checkpoint 声称还剩 1 步，而实际产出 0 个 batch）。
        """
        self._ensure_open()
        total = self._epoch_total_for(self._epoch)
        if self._restored_pos is not None:
            return max(0, total - self._restored_pos)
        return total

    # ---- P5 ----
    def _epoch_total_live(self, epoch: int) -> int:
        """第 `epoch` 轮轮长：**读头内的纯计算**（R3 `epoch_total_batches`）。

        与 `_query_epoch_total`（模块级纯函数，会重新读 manifest——云路径可能触发网络）
        的区别：这里直接用已构造的读头（manifest / sampler 都在内存里），checkpoint
        路径因此不引入额外 IO；fork 后读头按 worker 维度重建，口径与 `_produced` 一致。
        """
        return int(self._ensure_reader().epoch_total_batches(int(epoch)))

    def _next_position(self) -> "tuple[int, int]":
        """`(epoch, in_epoch_pos)`：**下一个待产出位置**口径（Rust seek 的唯一合法输入）。

        与 `MixBatchStream` 的 C22/G3 同款归一化：本轮的 batch 已全部产出
        （`in_epoch_pos == 轮长`）时，下一位置是 `(epoch + 1, 0)`——否则快照会在恢复时
        被 Rust 拒绝（`seek 越界：… 耗尽时应归一化为 (epoch+1, 0)`）。
        轮长查询用读头内的纯计算，只在已消费过（`pos > 0`）时发生。
        """
        pos = int(self._session_start + self._produced)
        if pos == 0:
            return self._epoch, 0
        total = self._epoch_total_live(self._epoch)
        if pos >= total:
            return self._epoch + 1, 0
        return self._epoch, pos

    def state_dict(self) -> Dict[str, Any]:
        self._ensure_open()
        state = self._base_state()
        epoch, pos = self._next_position()
        state.update({
            "epoch": epoch,
            "in_epoch_pos": pos,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._ensure_open()
        if self._active_session:
            raise RuntimeError("cannot load_state_dict during an active iteration session: effect ordering would be undefined (fail loudly rather than leave silent ambiguity)")
        self._require_schema(state)
        _check_fingerprint(state.get("manifest_fingerprint", {}), self._fingerprint)
        self._ensure_reader()
        _warn_if_topology_changed(
            state.get("topology", {}),
            world_size=self._world_size, rank=self._rank,
            partition=self._partition,
            shard_order=self._shard_order,
        )
        # C27/H3：state 覆盖构造期的 epoch（两个来源，state 优先）
        self._epoch = int(state["epoch"])
        pos = int(state["in_epoch_pos"])
        # 防御性归一化（旧版本的轮末快照写成 (epoch, 轮长)，恢复时会被 Rust 拒绝）：
        # `pos == 轮长` 是合法的轮末边界，等价于 (epoch+1, 0)；`pos > 轮长` 属快照不自洽。
        if pos > 0:
            total = self._epoch_total_live(self._epoch)
            if pos > total:
                warnings.warn(
                    f"snapshot in_epoch_pos={pos} exceeds the length {total} of epoch {self._epoch} (inconsistent snapshot): "
                    "normalized as end-of-epoch to the start of the next epoch.",
                    stacklevel=2,
                )
                pos = 0
                self._epoch += 1
            elif pos == total:
                pos = 0
                self._epoch += 1
        self._restored_pos = pos
        self._session_start = pos
        self._produced = 0


# ============================================================
# 2. MixBatchStream：多源加权混采（双模式轮长）
# ============================================================
class MixBatchStream(_EngineBase):
    """多源加权混采加载器（PT/SFT 异构、课程学习/退火）。

    两种轮长模式（OD1'）：
    - `steps_per_epoch=None`（自然模式）：被采样的子流耗尽即本轮结束；`len()` 是上界。
      **DDP 下必须配 `steps_per_epoch`**（C14）：构造期只做**无集合通信**的静态判定——
      等长 shard 且分配均衡可证明安全；等长 shard 分配不均直接报错；shard 尺寸不齐
      告警一次，需要确凿结论请显式调 `verify_ddp_consistency()`（需所有 rank 对称调用）。
    - `steps_per_epoch=N`（确定性模式）：恰好 N 步成功产出后结束，子流耗尽就地 wrap。

    权重来源（三选一，互斥）：显式 `weights=` / configs 内 `weight` 键 /
    按口径自动配比。**锚点 2（D3/R9）口径防错**：自动配比的基数由
    `weight_basis` 显式可选——`"total_tokens"`（默认 = 历史行为）/ 
    `"text_tokens"`（模态拆分后的文本有效 token，旧 manifest 解释为
    total_tokens）/ `"samples"`（total_batches × batch_size 代理）。含
    visual token 的数据集未显式选 basis 时 WARN 一次（纯文本不刷屏）。
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
                 weight_basis: Optional[str] = None,
                 options: Optional["LoaderOptions"] = None):
        opts = _resolve_options(
            "MixBatchStream", options, epoch=epoch,
            batch_size=batch_size, max_seq_len=max_seq_len,
            world_size=world_size, rank=rank, seed=seed, shuffle=shuffle,
            partition=partition, shard_order=shard_order,
            num_canonical_nodes=num_canonical_nodes,
            predownload=predownload,
        )
        if steps_per_epoch is not None and int(steps_per_epoch) < 1:
            raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")

        self._configs = list(configs)
        paths = [cfg["path"] for cfg in self._configs]
        if not paths:
            raise ValueError("datasets_config must not be empty")
        self._paths = paths
        self._steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)

        manifests = [_load_manifest_cached(p) for p in paths]  # F-26：进程内缓存
        # 尺寸一致性：异构指的是 PT/SFT 之别，不是尺寸之别
        bs = opts.batch_size or manifests[0]["global_stats"].get("batch_size") or 0
        msl = opts.max_seq_len or manifests[0].get("max_seq_len") or 0
        for p, m in zip(paths, manifests):
            if opts.batch_size is None and (m["global_stats"].get("batch_size") or 0) not in (0, bs):
                raise ValueError(f"inconsistent batch_size across sub-stream manifests: {p}")
            if opts.max_seq_len is None and (m.get("max_seq_len") or 0) not in (0, msl):
                raise ValueError(f"inconsistent max_seq_len across sub-stream manifests: {p}")
        self._batch_size = int(bs)
        self._max_seq_len = int(msl)
        self._source_formats = [_manifest_format(m) for m in manifests]
        self._heterogeneous = len(set(self._source_formats)) > 1

        # F-11：`shuffle` 控制**子流内**的 batch 级置换表（子流之间的顺序由加权随机采样
        # 决定，不可关闭；`seed` 保证可复现）。与 BatchStream 接受同一取值集合（同一
        # 归一化实现：LoaderOptions）——公共字段拷贝见 `_store_common_options`（P1-1）。
        self._store_common_options(opts)
        # F-09：全子流合并指纹（旧实现只覆盖 paths[0]，子流 2..N 被替换无法检测）
        self._fingerprint = _combined_manifest_fingerprint(paths, manifests)
        self._world_size, self._rank = _detect_dist(opts.world_size, opts.rank)

        # A2 意图与生效分离：`_base_weights` = 用户意图（显式 weights= / configs.weight /
        # token 配比），只被 update_weights 替换；`_temperature` 是作用在 base 上的算子，
        # **不覆写 base**（修复：update_temperature 不再静默重置用户显式权重）。
        # 锚点 2（D3/R9）口径防错：weight_basis 显式可选（默认 total_tokens = 现行为）；
        # 含 visual token 的多模态数据集不显式选 basis 时 WARN 一次（见 _resolve_weight_basis）。
        self._weight_basis = self._resolve_weight_basis(weight_basis, manifests)
        self._base_weights = self._resolve_weights(weights, manifests)
        self._temperature: Optional[float] = None
        # 先按原始权重构造 mixer（零长度掩码在 mixer 建好后才能算轮长）
        self.current_weights = list(self._base_weights)
        self._active_session = False
        # 各子流的共同起始轮号（C25/H1：由 Rust 侧 `start_epoch` 落点）
        self._epoch = int(epoch)
        self._session_started = False

        # 恢复后"本轮已走过的步数"：确定性模式下决定本次迭代还要产出多少步
        # （不修的话恢复后会再产出完整的 steps_per_epoch 步，轮边界与 LR 调度错位）。
        self._steps_before_session = 0

        self._engine_obj = None
        self._engine_pid = -1
        self._worker_dims: Optional[tuple] = None
        self._ensure_mixer()
        # 零长度子流（本 rank 无任何 shard）：有效权重置 0，否则采样只会得到 exhausted
        self._apply_zero_length_mask()
        self._check_cross_rank()
        _register(self)

    # ---- 权重 ----
    #: weight basis 的合法取值（锚点 2 / D3；默认 = 现行为）
    _WEIGHT_BASES = ("total_tokens", "text_tokens", "samples")

    def _resolve_weight_basis(self, weight_basis: Optional[str], manifests) -> str:
        """口径解析（锚点 2 / D3 / R9）：

        - ``total_tokens``（默认）：``global_stats.total_tokens``，与历史行为逐字节一致；
        - ``text_tokens``：模态拆分后的文本有效 token（`_effective_modality_tokens`，
          旧 manifest 解释为 == total_tokens）；
        - ``samples``：样本口径。manifest 尚无真实样本计数字段（GlobalStats 无
          total_samples），以 ``total_batches × batch_size`` 为跨格式代理
          （SFT 语义下即精确样本数；PT 为打包流的同型代理——诚实标注，P2 落
          total_samples 后可切换真实值）。

        WARN：**仅当任一子流含 visual token（total_image_tokens > 0）且用户未
        显式选 basis** 时告警一次（多模态数据集的"一图顶千 token"会静默改变
        token 配比的语义；纯文本不刷屏）。显式指定（哪怕与默认相同）不告警。
        """
        if weight_basis is None:
            has_visual = any(
                _effective_modality_tokens(m["global_stats"])[1] > 0 for m in manifests
            )
            if has_visual:
                warnings.warn(
                    "weight_basis not specified while a sub-stream contains visual tokens "
                    "(total_image_tokens > 0): token-based weighting uses total_tokens "
                    "(text + visual placeholders). If you want text-only or per-sample "
                    "weighting, pass weight_basis='text_tokens' / 'samples' explicitly.",
                    stacklevel=3,
                )
            return "total_tokens"
        basis = str(weight_basis)
        if basis not in self._WEIGHT_BASES:
            raise ValueError(
                f"weight_basis must be one of {self._WEIGHT_BASES}, got {weight_basis!r}"
            )
        return basis

    def _weight_basis_counts(self, manifests) -> List[int]:
        """按 `_weight_basis` 取各子流的配比基数（0 = 无法按该口径配比，报错）。"""
        counts: List[int] = []
        for m in manifests:
            stats = m["global_stats"]
            if self._weight_basis == "total_tokens":
                counts.append(int(stats.get("total_tokens") or 0))
            elif self._weight_basis == "text_tokens":
                counts.append(_effective_modality_tokens(stats)[0])
            else:  # "samples"
                counts.append(int(stats.get("total_batches") or 0) * int(stats.get("batch_size") or 0))
        return counts

    def _resolve_weights(self, weights, manifests) -> List[float]:
        """权重来源解析：``weights=`` 参数 / configs 内 ``weight`` 键 / token 配比，三选一。

        修复（复核追加）：``weights=`` 构造参数此前**被静默忽略**（用户传入后被无声丢弃，
        实际走 token 配比）——属于与 F-02 同类的静默失效，现改为真正生效，
        且与 configs 内的 ``weight`` 互斥（同时指定报错，权重来源必须唯一）。
        """
        cfg_weights = [cfg.get("weight") for cfg in self._configs]
        n_explicit = sum(w is not None for w in cfg_weights)
        if weights is not None and n_explicit:
            raise ValueError(
                "cannot specify both weights= and the \"weight\" key in datasets_config: "
                "the weight source must be unique (pick one; avoid silent ambiguity about which takes effect)."
            )
        if weights is not None:
            resolved = [float(w) for w in weights]
            if len(resolved) != len(self._paths):
                raise ValueError(
                    f"number of weights ({len(resolved)}) does not match the number of sub-streams ({len(self._paths)})"
                )
            if any(not math.isfinite(w) or w < 0 for w in resolved):
                raise ValueError(f"weights must be finite and non-negative (no NaN/Inf), got {resolved}")
            if not any(w > 0 for w in resolved):
                raise ValueError(f"at least one weight must be positive, got: {resolved}")
            return resolved
        if 0 < n_explicit < len(self._paths):
            raise ValueError(
                f"datasets_config weights must be all specified or all omitted (currently {n_explicit}/{len(self._paths)})"
            )
        if n_explicit == len(self._paths):
            resolved = [float(w) for w in cfg_weights]
            if not any(w > 0 for w in resolved):
                raise ValueError(f"at least one weight must be positive, got: {resolved}")
            return resolved
        # 按所选口径配比（锚点 2 / D3：basis 显式可选，默认 total_tokens = 现行为；
        # text_tokens 走模态拆分解释，samples 为 batches×batch_size 代理）
        basis_counts = self._weight_basis_counts(manifests)
        zeros = [p for p, c in zip(self._paths, basis_counts) if c == 0]
        if zeros:
            raise ValueError(
                f"these datasets have a zero {self._weight_basis} count (empty dataset or failed build); "
                f"cannot weight by {self._weight_basis}: {zeros}. Pass weight explicitly, "
                "or rebuild the dataset with the current version."
            )
        # F-29：原 `c ** (1.0 / 1.0)` 是温度参数被烘死后的死代码（恒等于 c），
        # 误导读者以为此处有温度；直接按基数配比。
        total = sum(basis_counts)
        return [c / total for c in basis_counts]

    def _effective_weights(self) -> List[float]:
        """base 经温度算子后的相对权重（归一化；零长度掩码**前**）。

        A2：温度是纯函数算子——同一 `_base_weights` 上重复调用结果恒等（幂等），
        且不修改 base 本身。
        """
        if self._temperature is None:
            return list(self._base_weights)
        t = self._temperature
        powered = [w ** (1.0 / t) for w in self._base_weights]
        denom = sum(powered)
        return [w / denom for w in powered]

    def _apply_zero_length_mask(self) -> None:
        """C19-②：把本 rank 轮长为 0 的子流权重置 0（update_* 都要重新应用）。

        F-02 原子性：**先让 Rust 落库成功，再提交 Python 侧 `current_weights`**——
        失败路径不得把实例留在"Python 状态已改、引擎分布未变"的自相矛盾状态。
        """
        mixer = self._ensure_mixer()
        totals = list(mixer.substream_epoch_totals())
        self._zero_mask = [t == 0 for t in totals]
        effective = [0.0 if zero else w for zero, w in zip(self._zero_mask, self._effective_weights())]
        if not any(w > 0 for w in effective):
            raise ValueError(
                f"all sub-streams on this rank have 0 available batches (empty shard assignment: "
                f"world_size={self._world_size} too large?): {totals}"
            )
        mixer.update_weights(effective)   # 先 Rust（可能抛 ValueError）
        self.current_weights = effective  # 后提交 Python 状态

    def _check_cross_rank(self) -> None:
        """C14/C19-①：自然模式 + DDP 的轮长一致性检查（**构造期不做集合通信**）。

        旧实现在构造器里 `all_gather_object`——那是死锁向量：rank 间配置不一致
        （如 rank0 传了 `steps_per_epoch`）或只在部分 rank 构造（包括
        `MixStreamingDataset.__len__` 自然模式的临时构造）都会让部分 rank 永久卡在
        集合通信里，且 `try/except` 接不住 hang。现在：

        - **本地静态判定**（只用已加载的 manifest，零额外 IO、零集合通信）：
          等长 shard 且分配均衡 → 可证明安全，静默通过；等长 shard 但分配不均 →
          可证明不等，直接 `RuntimeError`；shard 尺寸不齐 → 静态不可判，**告警一次**
          并指向精确认证入口 `verify_ddp_consistency()`（显式调用、所有 rank 对称）。
        - `MEGADATA_SKIP_DDP_CHECK=1` 关闭本检查（含告警）。
        - **检测面边界**：本判定只读本 rank 的 manifest，证明的是"轮长恒等"；
          **不覆盖各 rank manifest 内容漂移**（各 rank 的 index.json 不同但各自自洽时，
          本地判定仍可能通过）。该场景请在训练循环开始前的同步点显式调用
          `verify_ddp_consistency()`（`all_gather` 逐子流轮长向量，可发现 rank 间差异）。
        """
        if self._steps_per_epoch is not None:
            return  # 确定性模式天然同步
        if os.environ.get("MEGADATA_SKIP_DDP_CHECK", "") in ("1", "true", "True"):
            return
        if _ddp_world_size() <= 1:
            return  # 单进程 / dist 未初始化：没有需要同步的轮长
        num_workers, _ = self._worker_dims or (0, 0)
        verdict, why = _natural_mode_length_verdict(
            [_load_manifest_cached(p) for p in self._paths],  # F-26：进程内缓存，无额外 IO
            world_size=self._world_size, num_workers=num_workers,
        )
        if verdict == "safe":
            return
        if verdict == "danger":
            raise RuntimeError(
                f"natural mode (steps_per_epoch=None) provably has unequal per-rank epoch lengths under DDP: {why}. "
                "The end-of-epoch collective in the training loop would hang. Pass steps_per_epoch (deterministic mode) "
                "explicitly, or set MEGADATA_SKIP_DDP_CHECK=1 at your own risk."
            )
        # 告警去重标志在 `_topology`（P2-11 拆出）：跨模块读写同一份"每进程一次"状态
        if not _topology._DDP_NATURAL_WARNED:
            _topology._DDP_NATURAL_WARNED = True
            warnings.warn(
                f"natural mode (steps_per_epoch=None) cannot statically prove equal per-rank epoch lengths under DDP: {why}. "
                "If per-rank epoch lengths do differ, the end-of-epoch collective would hang. Pass steps_per_epoch explicitly; "
                "or call verify_ddp_consistency() first for an exact check (all ranks must call symmetrically); "
                "or set MEGADATA_SKIP_DDP_CHECK=1 to disable this check.",
                stacklevel=3,
            )

    def verify_ddp_consistency(self) -> None:
        """精确的跨 rank 轮长校验（**集合通信**：所有 rank 必须对称调用）。

        构造期只做无需通信的静态判定，shard 尺寸不齐的数据集只能告警（无法静态证明）。
        需要确凿结论时，在训练循环开始前的同步点显式调用本方法：它 all_gather 各 rank
        的逐子流轮长向量，不一致直接 `RuntimeError`。

        - 确定性模式（`steps_per_epoch=N`）/ 单进程 / dist 未初始化 / 已设
          `MEGADATA_SKIP_DDP_CHECK=1`：no-op；
        - **禁止在 DataLoader worker 内调用**（fork 出来的 worker 复用父进程通信子不安全）；
        - 只在部分 rank 调用会 hang——这正是不把它放进构造器/`len()` 的原因。
        """
        self._ensure_open()
        if self._steps_per_epoch is not None:
            return
        if os.environ.get("MEGADATA_SKIP_DDP_CHECK", "") in ("1", "true", "True"):
            return
        num_workers, _ = self._worker_dims or (0, 0)
        if num_workers > 0:
            raise RuntimeError(
                "verify_ddp_consistency() must not be called inside a DataLoader worker: "
                "a forked worker reusing the parent communicator is unsafe. Call it in the main process "
                "(after construction, before entering the training loop)."
            )
        try:
            import torch.distributed as dist
            if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
                return
            local = list(self._ensure_mixer().substream_epoch_totals())
            gathered: List[Optional[List[int]]] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
        except Exception as e:
            raise RuntimeError(
                f"cross-rank epoch-length check failed ({type(e).__name__}: {e}): `all_gather` requires all ranks "
                "to call verify_ddp_consistency() symmetrically (calling on only some ranks hangs). Check whether "
                "a rank was skipped / the communicator is unhealthy."
            ) from e
        if any(g is not None and list(g) != local for g in gathered):
            raise RuntimeError(
                "per-rank sub-stream epoch-length vectors are inconsistent (caused by tail-shard ownership changing per epoch): "
                "natural mode may have different per-rank epoch lengths under DDP, and the end-of-epoch collective would hang. "
                "Pass steps_per_epoch (deterministic mode) explicitly, or set MEGADATA_SKIP_DDP_CHECK=1 at your own risk."
            )

    # ---- 引擎生命周期（fork 安全；模板方法见 _EngineBase）----
    def _make_engine(self) -> RustStreamMixer:
        num_workers, worker_id = _detect_workers()
        self._worker_dims = (num_workers, worker_id)
        return RustStreamMixer(
            index_paths=self._paths, weights=list(self.current_weights),
            batch_size=self._batch_size, max_seq_len=self._max_seq_len,
            world_size=self._world_size, rank=self._rank, seed=self._seed,
            num_workers=num_workers, worker_id=worker_id,
            shuffle=self._shuffle,
            partition=self._partition,
            shard_order=self._shard_order,
            num_canonical_nodes=self._num_canonical_nodes or 0,
            predownload=self._predownload or 0,
            start_epoch=self._epoch,
        )

    def _ensure_mixer(self) -> RustStreamMixer:
        """类型化别名：fork 安全的 mixer 获取（实现见 `_EngineBase._engine`）。"""
        return self._engine

    def _set_epoch_checked(self, epoch: int) -> None:
        """手动设轮号：Rust mixer 的 `start_epoch` 是构造期落点 → 置空触发按新轮重建。

        重建后 mixer 的 `steps_yielded` / 游标归零（"从该轮开头重新开始"的主动
        重置语义），同时清掉恢复步数残留与重入 wrap 状态。
        """
        self._epoch = epoch
        self._session_started = False
        self._steps_before_session = 0
        self._dispose_engine()

    # ---- 属性 ----
    @property
    def source_formats(self) -> List[str]:
        return list(self._source_formats)

    @property
    def epoch(self) -> int:
        """当前轮号（动态查询——不再是构造期冻结值；F-24）。

        - **确定性模式**（`steps_per_epoch=N`）：`构造轮号 + steps_yielded // N`（逻辑轮，
          与 Tier-1 / 用户视角一致；子流物理游标可能远未 wrap，故不能用游标）；
        - **自然模式**：各子流游标 epoch 的**最小值**（子流独立 wrap，取最保守口径）。

        `load_state_dict` 恢复后 `steps_yielded`/游标均来自快照，返回值即恢复到的轮号。
        """
        if self._engine_obj is None:
            return self._epoch
        try:
            mixer_state = self._ensure_mixer().state_dict()
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as e:
            # ★ 评审 §3.9(a)：原为裸 `except Exception`，是全文件唯一一处——
            #   与项目"错误不静默"的整体基调不一致。收窄到引擎实际会抛的类型，
            #   真正的编程错误（AttributeError / NameError 等）不再被吞掉。
            #   仍降级返回构造期轮号：该property 的契约是"当前轮号"的**只读查询**，
            #   不能因为一次查询失败就让训练循环崩掉；但必须留痕。
            warnings.warn(
                f"megadata: MixBatchStream.epoch could not query the Rust mixer ({e!r}); "
                f"falling back to the construction-time epoch ({self._epoch}). "
                f"Training data position is unaffected (state_dict() still reads the mixer directly), "
                f"but this epoch value may be stale.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._epoch
        if self._steps_per_epoch is not None:
            return self._epoch + int(mixer_state["steps_yielded"]) // self._steps_per_epoch
        cursors = mixer_state["sub_stream_cursors"]
        if not cursors:
            return self._epoch
        return min(int(c[0]) for c in cursors)

    def update_weights(self, new_weights: List[float]) -> None:
        """设定**基础权重**（意图层）并清除温度——用户显式接管，温度不再作用。

        相对权重口径：只影响子流间的抽样比例（`[0.5, 0.5]` 与 `[5, 5]` 等价），
        不是"绝对抽取概率"；与构造期按 `total_tokens` 配比的路径同口径。

        失败原子性（F-02）：长度 / 有限性 / 非负 / 至少一个正数 在入口校验，
        非法输入抛 `ValueError` 且**不改动任何内部状态**。
        """
        ws = [float(w) for w in new_weights]
        self._validate_weights(ws, source="update_weights")
        old = (self._base_weights, self._temperature)
        self._base_weights, self._temperature = ws, None
        try:
            self._apply_zero_length_mask()
        except Exception:
            self._base_weights, self._temperature = old   # 失败回滚，保持状态一致
            raise

    def _validate_weights(self, ws: List[float], *, source: str) -> None:
        """F-02：长度 / 有限性 / 非负 / 至少一个正数 的统一校验（Rust 侧不校验长度）。"""
        n = len(self._paths)
        if len(ws) != n:
            raise ValueError(
                f"number of weights ({len(ws)}) does not match the number of sub-streams ({n}) ({source}): "
                "give one weight per sub-stream — a length mismatch would be silently truncated and starve some sub-streams."
            )
        if any(not math.isfinite(w) for w in ws):
            raise ValueError(f"weights must all be finite (no NaN/Inf), got {ws} ({source})")
        if any(w < 0 for w in ws):
            raise ValueError(f"weights must not be negative, got {ws} ({source})")
        if not any(w > 0 for w in ws):
            raise ValueError(f"at least one weight must be positive, got {ws} ({source})")

    def update_temperature(self, new_temperature: float) -> None:
        """在**基础权重**上应用温度平滑（预训练退火）：w_i ∝ base_i ** (1/T)。

        A2 意图与生效分离：
        - 温度只改"生效值"，**不覆写 base**——显式权重（构造 `weights=` /
          configs 的 `weight`）不会被静默重置为 token 配比；
        - 重复调用幂等：`update_temperature(2.0)` 后再 `update_temperature(4.0)`
          等价于直接 `(4.0)`（不复合）；
        - `update_weights(x)` 会替换 base 并清除温度（用户显式接管优先）。

        base 为 token 配比（构造期省略权重）时行为与原实现一致。
        注：token 配比使用 v1.0 manifest 的 total_tokens 口径（含 padding）时
        结果会偏向高 padding 源。
        """
        t = float(new_temperature)
        if t <= 0:
            raise ValueError(f"temperature must be > 0, got {new_temperature}")
        old = self._temperature
        self._temperature = t
        try:
            self._apply_zero_length_mask()
        except Exception:
            self._temperature = old
            raise

    # ---- P3 ----
    def __iter__(self):
        self._ensure_open()
        if self._active_session:
            raise RuntimeError("an active iteration session already exists (the megadata reader is not shared): finish the current session first")
        mixer = self._ensure_mixer()
        # B9'：重入时 wrap **所有**已耗尽子流（不止触发边界那一个）
        if self._session_started:
            mixer.wrap_all_exhausted()
        self._session_started = True
        self._active_session = True
        # 一次迭代 = 当前轮的**剩余**步数（_steps_before_session 只对恢复后的
        # 第一次迭代生效，消费即清空；后续迭代都是完整的 steps_per_epoch）
        carried = self._steps_before_session
        self._steps_before_session = 0
        # ★ 评审 §3.9(b)：与 `BatchStream.__iter__` 同款——用 `_SessionIterator`
        #   包装，保证"迭代器被创建却从未迭代"时活动会话标志仍能被释放
        #   （裸生成器在"从未启动即丢弃"时 `finally` 不会执行）。
        return _SessionIterator(self._iterate(carried), self._end_session)

    def _iterate(self, carried: int = 0):
        mixer = self._ensure_mixer()
        # 确定性模式的停止条件按**轮内全局步数**算：恢复后只剩 N - carried 步
        # （口径与 __len__ 统一为 max(0, ...)：carried 由 load_state_dict 的模运算
        #  归一化到 [0, N-1]，轮末快照应报告 0 步而不是"说谎的 1"）
        target = None
        if self._steps_per_epoch is not None:
            target = max(0, self._steps_per_epoch - carried)
        yielded = 0
        # ★ 评审 §3.9(c)："连续 wrap 无进展"守卫。
        #   确定性模式下若某个子流本轮轮长为 0（例如 shard 归属为空、或数据集被
        #   重建为0 batch），`exhausted → wrap → 立即又 exhausted → wrap …` 会变成
        #   **无进展忙循环**，占满一个 CPU 且永不产出，训练看起来就是"卡住"。
        #   这里统计"没有产出任何 batch 的连续 wrap 次数"，超过阈值就响亮报错，
        #   把"卡住"变成"一条可操作的错误信息"。
        #   阈值按子流数取：每个子流至少允许 wrap 一次（覆盖"各子流轮长不同"的
        #   正常场景——一个子流先耗尽、wrap 后继续供数，不会被误判）。
        no_progress_wraps = 0
        max_no_progress_wraps = max(len(self._paths) * 4, 1024)
        try:
            while True:
                try:
                    status, stream_idx, raw_dict = mixer.next_batch_capsules()
                except KeyboardInterrupt as e:
                    raise MegadataInterrupted("megadata mix stream interrupted") from e

                if status == "exhausted":
                    # 子流本轮读完：自然模式=轮边界；确定性模式=wrap 后继续
                    if self._steps_per_epoch is None:
                        return
                    no_progress_wraps += 1
                    if no_progress_wraps > max_no_progress_wraps:
                        raise RuntimeError(
                            f"megadata mix stream made no progress for {no_progress_wraps} consecutive "
                            f"wrap operations on {len(self._paths)} sub-stream(s) (threshold "
                            f"{max_no_progress_wraps}). This means at least one sub-stream has 0 batches "
                            f"available, so wrapping it can never yield data and the loop would spin forever. "
                            f"Check the sub-streams' manifests (global_stats.total_batches) and the "
                            f"partitioning (world_size / partition / shard_order)."
                        )
                    mixer.wrap_stream(stream_idx)
                    continue
                no_progress_wraps = 0

                # F-10：无条件带 format（不再只在异构混采时带；单源/同构也一致）
                batch = _convert_raw_batch(raw_dict, self._source_formats[stream_idx])
                yielded += 1
                yield batch

                if target is not None and yielded >= target:
                    return
        finally:
            self._active_session = False

    def __len__(self) -> int:
        self._ensure_open()
        if self._steps_per_epoch is not None:
            # 恢复后本轮只剩 N - k 步：len() 必须与"下一次迭代实际会产出多少"一致
            # （否则 HF 侧按 len 排的 LR scheduler 会与实际步数错位）。
            # k = steps_yielded % N ∈ [0, N-1]（load_state_dict 已用模运算归一化，轮末
            # 快照自然回到下一轮开头），故 N-k >= 1。
            # 口径与 BatchStream.__len__ 统一为 max(0, ...)：不返回"说谎的 1"
            # （恰好轮末的快照应报告 0 而不是 1），与 _iterate 的 target 同口径。
            return max(0, self._steps_per_epoch - self._steps_before_session)
        # C19-③：名义值随"当前游标所在轮"重算（尾 shard 逐轮换归属）
        return int(sum(self._ensure_mixer().substream_epoch_totals()))

    # ---- P5 ----
    def state_dict(self) -> Dict[str, Any]:
        """快照。

        `weights` 为**生效权重**（兼容既有消费方）；`base_weights` + `temperature`
        为 A2 的意图层与算子层（续训后退火口径不漂移，旧快照按 `weights` 当 base 兜底）。
        """
        self._ensure_open()
        state = self._base_state()
        rust = self._ensure_mixer().state_dict()
        state.update({
            "weights": list(self.current_weights),
            "base_weights": list(self._base_weights),
            "temperature": self._temperature,
            "steps_yielded": int(rust["steps_yielded"]),
            "sub_stream_cursors": [list(c) for c in rust["sub_stream_cursors"]],
            "rng_word_pos": int(rust["rng_word_pos"]),
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._ensure_open()
        if self._active_session:
            raise RuntimeError("cannot load_state_dict during an active iteration session")
        self._require_schema(state)
        _check_fingerprint(state.get("manifest_fingerprint", {}), self._fingerprint)
        mixer = self._ensure_mixer()
        _warn_if_topology_changed(
            state.get("topology", {}),
            world_size=self._world_size, rank=self._rank,
            partition=self._partition,
            shard_order=self._shard_order,
        )
        # A2：优先新字段（base_weights + temperature）；旧快照回退把 weights 当 base
        base = state.get("base_weights")
        if base is None:
            base = list(state.get("weights") or self.current_weights)
        base = [float(w) for w in base]
        # F-02：checkpoint 里的权重同样必须过长度/有限性/非负校验（Rust load_state 不校验长度）
        self._validate_weights(base, source="load_state_dict")
        temperature = state.get("temperature", None)
        if temperature is not None:
            temperature = float(temperature)
            if temperature <= 0:
                raise ValueError(f"invalid temperature in snapshot (must be > 0): {state.get('temperature')!r}")
        self._base_weights = base
        self._temperature = temperature
        self._apply_zero_length_mask()
        mixer.load_state_dict(state, list(self.current_weights))
        # 确定性模式：恢复点落在本轮第 k 步 ⇒ 本次迭代只应再产出 N-k 步，
        # 否则轮边界被拉长（HF Trainer 的 LR scheduler 按 N 步排程，会整体错位）。
        # `steps_yielded` 是跨轮全局计数，`% N` 即"本轮已走过的步数"。
        if self._steps_per_epoch is not None:
            self._steps_before_session = int(state.get("steps_yielded", 0)) % self._steps_per_epoch
        else:
            self._steps_before_session = 0


# ============================================================
# 3. RandomAccessBatchReader：随机按 batch 访问（评估/抽查）
# ============================================================
class RandomAccessBatchReader(_EngineBase):
    """随机索引批读取器（`get_batch(idx)` 返回一个完整 batch）。

    - **无 torch 协议**：不是 `torch.utils.data.Dataset`，不能套 `DataLoader`；
    - `idx` = 本 rank 在**访问轮**（构造参数 `epoch`，默认 0）内的 batch 下标；
    - 无 state_dict（随机访问无状态）；
    - `options.shuffle` **有意忽略**：本入口无置换表（按 idx 直取，没有"顺序"语义），
      `LoaderOptions` 是共用值对象因而照单接收该字段；需要打乱请用 `BatchStream` /
      `MixBatchStream`。
    - `options.shard_order` 同理**有意忽略**（F6）：按 idx 直取没有"shard 访问
      顺序"语义，读序只由 idx 决定；需要控制顺序请用 `BatchStream`。
    """

    #: P1-1 差异声明：本入口**有意不消费 `shuffle`**（无置换表语义，见类 docstring），
    #: `_store_common_options` 据此跳过该字段拷贝。
    _consumes_shuffle = False
    #: F6 差异声明：本入口**有意不消费 `shard_order`**（按 idx 直取，无 shard
    #: 访问顺序语义，见类 docstring）。
    _consumes_shard_order = False

    def __init__(self, index_json_path, *,
                 batch_size: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 42,
                 partition: str = "round_robin",
                 num_canonical_nodes: Optional[int] = None,
                 predownload: Optional[int] = None,
                 epoch: int = 0,
                 options: Optional["LoaderOptions"] = None):
        opts = _resolve_options(
            "RandomAccessBatchReader", options, epoch=epoch,
            batch_size=batch_size, max_seq_len=max_seq_len,
            world_size=world_size, rank=rank, seed=seed,
            partition=partition, num_canonical_nodes=num_canonical_nodes,
            predownload=predownload,
        )
        # 注：`shuffle` 经类属性 `_consumes_shuffle = False` 声明有意不消费——
        # `_store_common_options` 不拷贝该字段（本入口无置换表，取值 True/False
        # 都不改变行为），`LoaderOptions` 照单接收但不报错。
        self._index_json_path = os.fspath(index_json_path)
        manifest = _load_manifest_cached(self._index_json_path)  # F-26：进程内缓存
        self._batch_size = int(opts.batch_size or manifest["global_stats"].get("batch_size") or 0)
        self._max_seq_len = int(opts.max_seq_len or manifest.get("max_seq_len") or 0)
        self._store_common_options(opts)  # P1-1 公共拷贝（seed/partition/ncn/predownload）
        self._format = _manifest_format(manifest)
        self._fingerprint = _fingerprint_of(self._index_json_path, manifest)
        self._world_size, self._rank = _detect_dist(opts.world_size, opts.rank)
        self._epoch = int(epoch)

        self._engine_obj = None
        self._engine_pid = -1
        self._worker_dims: Optional[tuple] = None
        self._next_expected: Optional[int] = None
        self._ensure_reader()
        self._len = self._query_epoch_total(self._epoch)
        _register(self)

    # ---- 引擎生命周期（fork 安全；模板方法见 _EngineBase）----
    def _make_engine(self) -> RustStreamReader:
        num_workers, worker_id = _detect_workers()
        self._worker_dims = (num_workers, worker_id)
        return RustStreamReader(
            self._index_json_path, self._batch_size, self._max_seq_len,
            self._world_size, self._rank, self._seed,
            num_workers, worker_id, False,           # Reader 无置换表
            partition=self._partition,
            num_canonical_nodes=self._num_canonical_nodes or 0,
            predownload=self._predownload or 0,
        )

    def _ensure_reader(self) -> RustStreamReader:
        """类型化别名：fork 安全的读头获取（实现见 `_EngineBase._engine`）。"""
        return self._engine

    def _on_engine_rebuilt(self, *, forked: bool) -> None:
        # 读头重建（含 fork）后顺序访问游标失效：下一次访问必须重新 seek
        self._next_expected = None

    def _query_epoch_total(self, epoch: int) -> int:
        """纯函数查询"访问轮"的轮长（A3：零副作用，参数与 `_make_engine` 一致）。"""
        num_workers, worker_id = _detect_workers()
        return int(_epoch_total_batches(
            self._index_json_path, int(epoch), self._world_size, self._rank,
            int(num_workers), int(worker_id), self._seed, self._partition,
            int(self._num_canonical_nodes or 0),
        ))

    def _set_epoch_checked(self, epoch: int) -> None:
        """切换"访问轮"：重建读头（`start_epoch` 语义随轮改变）并重算轮长。"""
        self._epoch = epoch
        self._next_expected = None
        self._dispose_engine()
        self._len = self._query_epoch_total(epoch)

    @property
    def format(self) -> str:
        return self._format

    def __len__(self) -> int:
        self._ensure_open()
        return self._len

    def get_batch(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()
        reader = self._ensure_reader()
        if idx < 0:
            idx += self._len
        if not (0 <= idx < self._len):
            raise IndexError(f"idx {idx} out of range [0, {self._len})")
        # 顺序访问 fast-path：连续 idx 不触发 seek
        if self._next_expected != idx:
            reader.seek_to_epoch_batch(self._epoch, int(idx))
        try:
            raw_dict = reader.next_batch_capsules()
        except KeyboardInterrupt as e:
            raise MegadataInterrupted("megadata map access interrupted") from e
        if raw_dict is None:
            raise IndexError(f"megadata returned None at idx {idx} (not enough data?)")
        # F-22：读成功后才提交顺序游标——错误路径不得提前推进，否则下一次访问
        # idx+1 时会跳过 seek，在未推进的读头上取到错位数据
        self._next_expected = int(idx) + 1
        return _convert_raw_batch(raw_dict, self._format)

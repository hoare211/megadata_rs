# python/megadata/dataloader.py
"""
megadata Python 接口协议 **v3**（完全重置；v1/v2 的迁移映射见仓库根 `MIGRATION.md`）。

协议要点：
- **D1 epoch-per-iter**：一次 `__iter__` = 恰好一个 epoch；`num_epochs` 已移除，
  轮数归训练脚本/框架。游标一律"下一个待产出位置"口径。
- **P1**：batch 为 tensor dict；`cu_seqlens` / `row_lens` 是 int32 tensor；
  **所有** loader 的 batch 都携带 `"format"` 键（"pt" / "sft"）。
- **P3**：轮边界来自 Rust 的 `at_eof` 信号（R1），Python 侧**不设产出计数护栏**。
- **P5**：state 只认 `schema == 3`，含 manifest 指纹与拓扑指纹。
- **P6**：`Eof → StopIteration`、`Fatal → OSError`、cancel → `MegadataInterrupted`。

协议分层（先看这张表，再看具体类）::

    你需要什么                              用什么
    ------------------------------------  ----------------------------------------
    单源流式、自己写训练循环               BatchStream             ← 引擎层：裸迭代器
    多源加权混采 / 课程学习 / 退火         MixBatchStream          ← 协议（刻意不是 torch
    按 batch 随机抽查（评估/调试）         RandomAccessBatchReader ← DataLoader 子类）
    交给 HF Trainer / Lightning            StreamingDataset
                                           + MegadataDataLoader    ← Tier-1：torch 适配

生命周期（全包统一，close 只有一种语义）：
- 所有持引擎对象（含 Tier-1）支持 `with`；`close()` 是**终态**（之后明确报错）；
- 只想释放"当前轮读头"请用 `release()`（Tier-1 非终态，下一轮 `__iter__` 自动重建）。

轮号（全包统一语义）：`epoch` = "下一次 `__iter__` 将产出的轮"，手动设置用 `set_epoch()`。
注意 break 语义两层不同：引擎层 break 后下轮**从本轮第 0 个 batch 整轮重放**（EOF 才推进轮号），
Tier-1 break 后 teardown 无条件推进（跳到下轮）——迁移时行为会翻转，详见使用指南 §5.3。
"""
# P2-11 拆包（纯移动、零行为变更）：原实现按职责拆为 4 个模块，本文件退化为兼容 facade。
#   `_manifest.py`  manifest 读取 / 进程内缓存 / 指纹（→ 仅依赖 _rust）
#   `_topology.py`  DDP / worker 拓扑探测与跨 rank 轮长判定（torch 仅函数内惰性导入）
#   `engine.py`     引擎层加载器（_EngineBase / BatchStream / MixBatchStream /
#                   RandomAccessBatchReader）+ LoaderOptions 与扁平 kwargs 冲突检测
#   `tier1.py`      torch 适配层（StreamingDataset / MixStreamingDataset / MegadataDataLoader）
# 依赖方向单向：tier1 → engine → {_manifest, _topology} → _options / _signals / _rust，无循环。
# 下方 re-export 保持 `from megadata.dataloader import X`（含全部私有符号）100% 兼容；
# `megadata.__init__` 的 _LAZY_ATTRS 仍指向本模块，`megadata.X` 惰性加载行为不变。

# F-36：信号/注册表已抽到 _signals（builder 只需 _register，不必拖入 torch）；
# install/uninstall 在此 re-export，保持 `from megadata.dataloader import ...` 向后兼容。
from megadata._signals import (  # noqa: F401
    MegadataInterrupted,
    _ACTIVE_INSTANCES,
    _register,
    install_signal_handlers,
    uninstall_signal_handlers,
)
from megadata._manifest import (  # noqa: F401
    _MANIFEST_CACHE,
    _MANIFEST_CACHE_MAX,
    _check_fingerprint,
    _check_one_fingerprint,
    _combined_manifest_fingerprint,
    _fingerprint_of,
    _load_manifest_cached,
    _manifest_format,
    _read_total_tokens,
    read_manifest,
)
from megadata._options import normalize_partition, normalize_shuffle  # noqa: F401
from megadata._rust import (  # noqa: F401
    RustStreamReader, RustStreamMixer,
    read_manifest_json,
    epoch_total_batches as _epoch_total_batches,
)
from megadata._topology import (  # noqa: F401
    _DDP_NATURAL_WARNED,
    _ENV_TOPOLOGY_WARNED,
    _ddp_world_size,
    _detect_dist,
    _detect_workers,
    _natural_mode_length_verdict,
    _warn_if_topology_changed,
)
from megadata.engine import (  # noqa: F401
    SCHEMA_VERSION,
    BatchStream,
    DatasetConfig,
    LoaderOptions,
    MixBatchStream,
    RandomAccessBatchReader,
    _EngineBase,
    _OPTION_DEFAULTS,
    _OPTION_NORMALIZERS,
    _convert_raw_batch,
    _derive_option_defaults,
    _resolve_options,
    _safe_from_dlpack,
    _unpack_arrow_zero_copy,
)
from megadata.tier1 import (  # noqa: F401
    MegadataDataLoader,
    MixStreamingDataset,
    StreamingDataset,
    _identity_collate,
)

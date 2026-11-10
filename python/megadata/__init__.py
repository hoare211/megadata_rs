# python/megadata/__init__.py
"""megadata：LLM 数据管线引擎（v3 协议）。

**接口速查**（先看这张表，再看具体类；详版见 `megadata.dataloader` 模块 docstring）::

    你需要什么                              用什么
    ------------------------------------  ----------------------------------------
    单源流式、自己写训练循环               BatchStream             ← 引擎层：裸迭代器
    多源加权混采 / 课程学习 / 退火         MixBatchStream          ← 协议（刻意不是 torch
    按 batch 随机抽查（评估/调试）         RandomAccessBatchReader ← DataLoader 子类）
    交给 HF Trainer / Lightning            StreamingDataset
                                           + MegadataDataLoader    ← Tier-1：torch 适配
    离线构建数据集                         DataBuilder + BuildConfig

生命周期（全包统一）：所有持引擎对象支持 `with`；`close()` 是**终态**；
只想释放当前轮读头用 `release()`（Tier-1 非终态，下一轮 `__iter__` 自动重建）。

batch 布局判定用 `is_pt_batch` / `is_sft_batch`（运行时守卫 + 静态类型收窄）。

导入设计（F-36）：
- `import megadata` 本身**不**导入 torch / pyarrow——所有重依赖符号按 PEP 562
  模块级 `__getattr__` 惰性加载（首次访问时才 import 对应子模块并缓存）。
  构建路径（`DataBuilder`）因此不再被拖入 torch 依赖链。
- 显式子模块导入（`import megadata.simulation` / `from megadata.simulation import ...`）
  照常工作。容量规划（`simulation`）是独立子模块、**不在本包顶层导出**（F-37 结论）；
  数据集自检同样按子模块使用：`from megadata.validate import validate` / CLI
  `python -m megadata.validate`（F-32）。

版本（F-19）：唯一来源 = 包元数据（`pyproject.toml`），未安装/元数据缺失时回退常量。
"""
# batch 类型与守卫：_types 无 torch/pyarrow 运行期依赖，可安全顶层导入（F-36 不破）。
# 保持 `from megadata import PTBatch` 顶层兼容（F401 在此为有意 re-export）。
from megadata._types import PTBatch, SFTBatch  # noqa: F401

try:
    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("megadata")
    except _PkgNotFound:
        __version__ = "0.2.0"
except ImportError:  # pragma: no cover - 极老 Python 兜底
    __version__ = "0.2.0"


# ------------------------------------------------------------------
# 惰性加载表：公开符号 -> (子模块, 属性名)
# ------------------------------------------------------------------
_LAZY_ATTRS = {
    # 构建
    "DataBuilder": ("megadata.builder", "DataBuilder"),
    # 加载（13 个核心符号 + F-31 的 MixStreamingDataset）
    "BatchStream": ("megadata.dataloader", "BatchStream"),
    "MixBatchStream": ("megadata.dataloader", "MixBatchStream"),
    "RandomAccessBatchReader": ("megadata.dataloader", "RandomAccessBatchReader"),
    "StreamingDataset": ("megadata.dataloader", "StreamingDataset"),
    "MixStreamingDataset": ("megadata.dataloader", "MixStreamingDataset"),
    "MegadataDataLoader": ("megadata.dataloader", "MegadataDataLoader"),
    "DatasetConfig": ("megadata.dataloader", "DatasetConfig"),
    # 配置值对象（#5 收敛）：加载参数 / 构建参数
    "LoaderOptions": ("megadata.dataloader", "LoaderOptions"),
    "BuildConfig": ("megadata.builder", "BuildConfig"),
    "Marker": ("megadata.builder", "Marker"),
    "read_manifest": ("megadata.dataloader", "read_manifest"),
    # batch 布局守卫（A5 类型接线）：运行时判定 + 静态类型收窄（TypeGuard）
    "is_pt_batch": ("megadata._types", "is_pt_batch"),
    "is_sft_batch": ("megadata._types", "is_sft_batch"),
    # 自检入口（F-32）是子模块（megadata.validate / from megadata.validate import validate），
    # 不在顶层导出同名函数——避免"megadata.validate 是模块还是函数"的歧义。
    # 中断（F-36：来自 _signals，不依赖 torch）
    "MegadataInterrupted": ("megadata._signals", "MegadataInterrupted"),
    "install_signal_handlers": ("megadata._signals", "install_signal_handlers"),
    "uninstall_signal_handlers": ("megadata._signals", "uninstall_signal_handlers"),
    # 分词器接入面
    "BatchTokenizer": ("megadata.tokenizer_adapter", "BatchTokenizer"),
    "HfTokenizerAdapter": ("megadata.tokenizer_adapter", "HfTokenizerAdapter"),
    # 多模态 vlm 构建面（P2）：modality encoder 参考实现
    "make_vlm_encode_callback": ("megadata.modality_adapter", "make_vlm_encode_callback"),
    "validate_encoded_sample": ("megadata.modality_adapter", "validate_encoded_sample"),
    # attention 适配面
    "AttentionBackend": ("megadata.mask_adapter", "AttentionBackend"),
    "normalize_cu_seqlens": ("megadata.mask_adapter", "normalize_cu_seqlens"),
    "valid_len": ("megadata.mask_adapter", "valid_len"),
    "cu_seqlens_to_doc_ids": ("megadata.mask_adapter", "cu_seqlens_to_doc_ids"),
    "cu_seqlens_to_seg_lens": ("megadata.mask_adapter", "cu_seqlens_to_seg_lens"),
    # P2-4：mask_adapter.__all__ 里有、顶层 _LAZY_ATTRS 漏了的导出面（保持两处一致）
    "sft_row_lens": ("megadata.mask_adapter", "sft_row_lens"),
    "to_flash_varlen": ("megadata.mask_adapter", "to_flash_varlen"),
    "to_flex_block_mask": ("megadata.mask_adapter", "to_flex_block_mask"),
    "to_xformers_block_diag": ("megadata.mask_adapter", "to_xformers_block_diag"),
    "flatten_and_unpad": ("megadata.mask_adapter", "flatten_and_unpad"),
    "repad_to_batch": ("megadata.mask_adapter", "repad_to_batch"),
    "build_attention": ("megadata.mask_adapter", "build_attention"),
}

#: 可直接属性访问的子模块名（`import megadata` 后 `megadata.simulation` 亦可）
_LAZY_MODULES = ("dataloader", "builder", "mask_adapter", "modality_adapter", "simulation",
                 "tokenizer_adapter", "validate", "_options", "_signals")

__all__ = ["PTBatch", "SFTBatch"] + sorted(_LAZY_ATTRS)


def __getattr__(name):  # PEP 562：模块属性未命中时按需加载
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        import importlib
        value = getattr(importlib.import_module(target[0]), target[1])
        globals()[name] = value  # 缓存，后续访问不再走 __getattr__
        return value
    if name in _LAZY_MODULES:
        import importlib
        return importlib.import_module(f"megadata.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()) | set(_LAZY_MODULES))

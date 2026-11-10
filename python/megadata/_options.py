# python/megadata/_options.py
"""共享的选项归一化（**无 torch / pyarrow 依赖**，供 dataloader / builder / simulation 共用）。

存在意义：
- F-11：shuffle 的"接受取值集合"此前散落在三处（`BatchStream` 4 种写法 / `MixBatchStream`
  无参数 / `SimulationConfig` 只认字符串且拼错静默当开启），收敛为唯一实现；
- P2 收敛：开关类参数（shuffle / partition / mode / compression）此前各写各的风格
  （bool+str 双口径 / 纯字符串 / 枚举），现在统一走 `normalize_choice`——
  **规范形式 = 字符串；布尔 / 枚举 = 等价简写；非法 = ValueError（列出可选值）**。
"""
from typing import Any, Mapping, Optional

#: shuffle 的规范取值集合（文档统一推荐字符串写法，布尔是等价简写）
SHUFFLE_VALUES = ("permutation", "none")
#: partition 的规范取值集合
PARTITION_VALUES = ("round_robin", "relaxed")
#: shard_order 的规范取值集合（F6：epoch 内 shard 访问顺序，与 shuffle 正交）
SHARD_ORDER_VALUES = ("random", "sequential")
#: 构建模式（vlm = 多模态 P2：图文混合，编码全在 modality_encoder 回调内自理）
MODE_VALUES = ("pt", "sft", "vlm")
#: 压缩算法（None 已移除，见 normalize_compression）
COMPRESSION_VALUES = ("auto", "zstd", "lz4", "none")


def normalize_choice(value: Any, *, name: str, spec: Mapping[Any, Any],
                     aliases: Optional[Mapping[Any, Any]] = None) -> Any:
    """通用开关归一化（全包唯一约定）。

    - `spec`：规范取值 → 归一化结果（键即合法字符串）；
    - `aliases`：等价简写 → 规范取值（如 `{True: "permutation", False: "none"}`）；
    - 字符串比较大小写 / 首尾空白不敏感；非法值报 `ValueError` 并列出全部可选值。
    """
    if aliases is not None:
        try:
            if value in aliases:
                value = aliases[value]
        except TypeError:
            # unhashable（如 list）：留给下面的 spec 查找统一报 "带可选值的 ValueError"
            pass
    if isinstance(value, str):
        value = value.strip().lower()
    try:
        return spec[value]
    except (KeyError, TypeError):
        raise ValueError(
            f"{name} must be one of {' / '.join(repr(k) for k in spec)}, got {value!r}"
        ) from None


def normalize_shuffle(value):
    """shuffle 归一化：``True/"permutation" → True``；``False/"none" → False``；其余报错。

    约定（v3）：字符串 ``"permutation"`` / ``"none"`` 为推荐写法，布尔 ``True`` / ``False``
    为等价简写；``"buffer"`` 已在 v3 移除（permutation 是其超集）。
    """
    try:
        return normalize_choice(
            value, name="shuffle",
            spec={"permutation": True, "none": False},
            aliases={True: "permutation", False: "none"},
        )
    except ValueError as e:
        raise ValueError(f"{e} (buffer mode was removed in v3; 'permutation' is its superset).") from None


def normalize_partition(value: Any) -> str:
    """partition 归一化：``"round_robin"`` / ``"relaxed"``（拼错在 Python 侧即报错）。"""
    return normalize_choice(
        value, name="partition", spec={v: v for v in PARTITION_VALUES},
    )


def normalize_shard_order(value: Any) -> str:
    """shard_order 归一化（F6）：``"random"`` / ``"sequential"``。

    语义：epoch 内**全局 shard 序**是否按 (seed, epoch) 打散——与 `shuffle`
    （shard 内 batch 级置换）正交。``"sequential"`` 下 RoundRobin 的全局序恒为
    ``0..n``（跨 epoch 稳定），用于逐 token 核对语料 / 与离线工具对齐顺序。
    """
    return normalize_choice(
        value, name="shard_order", spec={v: v for v in SHARD_ORDER_VALUES},
    )


def normalize_mode(value: Any) -> str:
    """构建模式归一化：``"pt"`` / ``"sft"``。"""
    return normalize_choice(value, name="mode", spec={v: v for v in MODE_VALUES})


def normalize_compression(value: Any) -> str:
    """compression 归一化（``"auto" | "zstd" | "lz4" | "none"``）。

    ``None`` 保留专属错误文案：v3 已移除 ``None``（与 ``"none"`` 的视觉陷阱），
    这是历史决策记录，不能用通用报错覆盖。
    """
    if value is None:
        raise ValueError(
            'compression=None was removed in v3 (visual trap of None vs "none"): '
            'pass "auto" (PT→zstd / SFT→uncompressed), "zstd", "lz4", or "none".'
        )
    return normalize_choice(
        value, name="compression", spec={v: v for v in COMPRESSION_VALUES},
    )

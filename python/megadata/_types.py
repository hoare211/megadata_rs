# python/megadata/_types.py
"""batch 类型与静态守卫（**无 torch / pyarrow 运行期依赖**，供 `__init__` 顶层导入）。

A5 类型接线：`PTBatch` / `SFTBatch` 此前是"挂名类型"（导出但无任何代码返回/标注），
现在由 loader 的实际 batch 形状 + 下面两个守卫接线：

```python
from megadata import is_sft_batch

if is_sft_batch(batch):          # 运行期判定 + 静态收窄为 SFTBatch
    loss = ce(model(batch["input_ids"]).logits, batch["labels"])
```

判定依据是 v3 起**所有** loader 都携带的 `"format"` 键（F-10：单源 / 同构多源 /
异构混采三种路径一致）。
"""
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypedDict

if TYPE_CHECKING:  # 仅类型检查期可见（运行期不 import torch）
    import torch
    from typing import TypeGuard


class PTBatch(TypedDict):
    """PT 批次（GreedyPacker 布局：input_ids + cu_seqlens，无 labels/position_ids）。"""
    input_ids: "torch.Tensor"
    cu_seqlens: "torch.Tensor"
    format: Literal["pt"]


class SFTBatch(TypedDict):
    """SFT 批次（DocumentAwarePacker 布局：行内 padding，labels 已做 next-token shift）。

    可能额外携带 `row_lens`（每行有效长度，A4 数据自描述列；旧数据集无该键）——
    该可选键未用 `NotRequired` 声明：`typing.NotRequired` 要 Python >= 3.11，
    本包 requires-python >= 3.8，不为类型标注引入 `typing_extensions` 依赖。
    """
    input_ids: "torch.Tensor"
    labels: "torch.Tensor"
    position_ids: "torch.Tensor"
    cu_seqlens: "torch.Tensor"
    format: Literal["sft"]


def is_pt_batch(batch: "Mapping[str, Any]") -> "TypeGuard[PTBatch]":
    """运行时守卫：`batch` 是 PT 布局（按 `format` 键判定；静态类型收窄为 `PTBatch`）。"""
    return isinstance(batch, Mapping) and batch.get("format") == "pt"


def is_sft_batch(batch: "Mapping[str, Any]") -> "TypeGuard[SFTBatch]":
    """运行时守卫：`batch` 是 SFT 布局（按 `format` 键判定；静态类型收窄为 `SFTBatch`）。"""
    return isinstance(batch, Mapping) and batch.get("format") == "sft"

# megadata/mask_adapter.py
"""Attention mask 适配层：把 megadata 的 ``cu_seqlens``（文档段边界）展开为
FlashAttention-2 varlen / FlexAttention / xformers 三种 backend 的 mask 表示。

====================================================================
设计前提（由项目数据格式决定，务必先读）
====================================================================
``cu_seqlens`` 描述 batch 内文档段边界：``[0, seg1_end, ..., valid_len]``，
``valid_len = cu_seqlens[-1]`` 是有效 token 数（不含 padding）。三个 backend 统一基于
"展平成一条长 packed 序列 + 按 cu_seqlens 分段"语义：段内 attend、段间隔离（块对角）。

但 **PT 与 SFT 的 padding 分布不同**，决定如何把二维 ``(batch_size, max_seq_len)``
还原成这条连续流：

- **PT (GreedyPacker)**：文档跨行连续拼接成一维流，padding 集中在流末尾，
  ``cu_seqlens`` 即一维流的物理偏移。:func:`flatten_and_unpad` 直接取
  ``flat[:valid_len]`` 即得连续流——**PT 全链路开箱即用**。
- **SFT (DocumentAwarePacker)**：文档不跨行，每行 ``[doc..., pad]``，padding 分散在
  各行末尾，``cu_seqlens`` 是文档长度的逻辑累计（不含 pad），并非物理偏移。
  :func:`flatten_and_unpad` / :func:`repad_to_batch` 传 ``packer="sft"`` 即按
  **行有效长度**（``row_lens``：新数据集构建期写入的 Arrow 列，旧数据集回退到
  :func:`sft_row_lens` 复刻装箱算法）抽取/还原连续流——文档边界与 ``cu_seqlens``
  一一对应，可直接接 varlen/Flex attention。SFT 通常每序列即一轮对话，
  多数场景也可按行独立 attention（不 flatten）。

三个 backend 对连续流的消费方式：
  - FA2 varlen：q/k/v 展平成 ``(valid_len, ...)``，``cu_seqlens`` 当 varlen 边界。
  - FlexAttention：``B=None`` 广播，``Q_LEN=KV_LEN`` pad 到 block_size 倍数，
    用逐 token ``document_id`` 构造 ``mask_mod``。
  - xformers：``BlockDiagonalMask.from_seqlens(seg_lens)``，视为单条 packed 序列。

核心转换函数（``to_*``）只依赖 ``cu_seqlens``、不接触 input_ids 物理布局，**假设
调用方已提供与 cu_seqlens 对齐的连续流**。对 PT 这天然成立。

**空 ``cu_seqlens``**（旧数据集无 ``.cu`` 文件）退化为单段 ``[0, valid_len]``，
等价标准 attention（不隔离文档）。此时若 ``valid_len`` 无法从 ``cu_seqlens[-1]``
推断（因为为空），调用方必须显式传 ``total_len``。

====================================================================
使用约束
====================================================================
采用本适配层后，模型前向的注意力输入是 **unpad 后的 1D 流 ``(valid_len, ...)``**
（无 batch 维；FlexAttention 用 ``B=None`` 广播，FA2 varlen 用 ``(total, nheads, dim)``），
而非原始 ``(batch_size, max_seq_len, ...)``。辅助函数
:func:`flatten_and_unpad` / :func:`repad_to_batch` 负责两者间的转换；
若模型要求显式 batch 维，对 1D 流做 ``unsqueeze(0)`` 即可。

示例（PT 数据 + FlexAttention，文档隔离 + 因果）::

    from megadata import BatchStream, to_flex_block_mask, flatten_and_unpad

    loader = BatchStream("out/pt/index.json", batch_size=8, max_seq_len=4096)
    for batch in loader:
        input_ids = batch["input_ids"]                       # (B, S)
        cu = batch["cu_seqlens"]                             # [0, s1, ..., valid]
        ids_1d = flatten_and_unpad(input_ids, cu)            # (valid,)
        block_mask = to_flex_block_mask(cu, causal=True)     # 跨层重用
        for layer in model.layers:
            x = layer(x, block_mask=block_mask)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "AttentionBackend",
    "normalize_cu_seqlens",
    "valid_len",
    "cu_seqlens_to_doc_ids",
    "cu_seqlens_to_seg_lens",
    "sft_row_lens",
    "to_flash_varlen",
    "to_flex_block_mask",
    "to_xformers_block_diag",
    "flatten_and_unpad",
    "repad_to_batch",
    "build_attention",
]

CuSeqlensLike = Union[Sequence[int], "torch.Tensor"]


# ====================================================================
# 1. 基础转换（纯函数，无 backend 依赖）
# ====================================================================
def normalize_cu_seqlens(cu_seqlens: CuSeqlensLike) -> torch.Tensor:
    """把 cu_seqlens 归一化为 1D ``int64`` tensor。

    - 接受 list / tuple / tensor。
    - 去除连续相等的边界点（防御性，项目构建期已去重）。
    - 保证首元素为 0（若缺失则补）。
    - 空输入返回空 tensor（由调用方按"整段"语义处理）。

    契约校验（公开 API 边界必检）：cu_seqlens 必须**非负且非递减**——
    畸形输入（如 ``[0, 5, 3]``）在下游会被静默吞掉（seg_lens 过滤掉负 diff，
    doc_ids 截断），最终产出错误的 mask 而不报错。
    """
    if cu_seqlens is None:
        return torch.empty(0, dtype=torch.long)
    t = torch.as_tensor(cu_seqlens, dtype=torch.long).flatten()
    if t.numel() == 0:
        return t
    if bool((t < 0).any()):
        raise ValueError(f"cu_seqlens must be non-negative, got {t.tolist()}")
    # 去重连续相等
    if t.numel() > 1:
        keep = torch.ones(t.numel(), dtype=torch.bool)
        keep[1:] = t[1:] != t[:-1]
        t = t[keep]
        # 去重后仍递减 = 真实递减段（如 [0, 5, 3]），必须拒绝
        if bool((t[1:] < t[:-1]).any()):
            raise ValueError(
                f"cu_seqlens must be non-decreasing (document boundaries are cumulative), "
                f"got {t.tolist()}"
            )
    # 保证首元素为 0
    if int(t[0].item()) != 0:
        t = torch.cat([torch.zeros(1, dtype=torch.long), t])
    return t


def valid_len(cu_seqlens: CuSeqlensLike, total_len: Optional[int] = None) -> int:
    """返回有效 token 数（= ``cu_seqlens[-1]``）。

    空 cu_seqlens 时回退到 ``total_len``；两者都缺失则报错。
    """
    cu = normalize_cu_seqlens(cu_seqlens)
    if cu.numel() > 0:
        return int(cu[-1].item())
    if total_len is not None:
        return int(total_len)
    raise ValueError(
        "cu_seqlens is empty and total_len was not provided: the valid length cannot be inferred. "
        "For old datasets (no .cu files), pass total_len=batch_size*max_seq_len explicitly."
    )


def cu_seqlens_to_seg_lens(cu_seqlens: CuSeqlensLike) -> List[int]:
    """cu_seqlens → 每段长度列表 ``[l1, l2, ...]``（过滤 0 长度段）。

    空 cu_seqlens 返回 ``[]``（调用方应按整段处理）。
    """
    cu = normalize_cu_seqlens(cu_seqlens)
    if cu.numel() <= 1:
        return []
    diffs = (cu[1:] - cu[:-1]).tolist()
    return [int(d) for d in diffs if d > 0]


def _sft_row_lens_from_seg_lens(seg_lens: Sequence[int], batch_size: int, max_seq_len: int) -> List[int]:
    """复刻 `DocumentAwarePacker` 的贪心行内装箱，还原每行有效长度（旧数据集回退路径）。

    逐边界与 Rust 实现对齐（`src/packing/document_aware.rs::pack_chunks`）：
    换行条件是 `>`（恰好填满不换行）、空文档已被 `seg_lens` 过滤、
    超长文档在 Rust 侧已截断为 `max_seq_len`（故 seg_len 恒 <= S）。
    """
    row_lens = [0] * batch_size
    r = 0
    for L in seg_lens:
        if L <= 0:
            continue
        if row_lens[r] + L > max_seq_len:
            r += 1
            if r >= batch_size:
                raise ValueError(
                    f"document segments are inconsistent with (batch_size={batch_size}, max_seq_len={max_seq_len}): "
                    "cu_seqlens does not match the physical layout, or the Rust packing algorithm has drifted"
                )
        row_lens[r] += L
    return row_lens


def sft_row_lens(
    cu_seqlens: CuSeqlensLike,
    batch_size: int,
    max_seq_len: int,
    *,
    row_lens: Optional[CuSeqlensLike] = None,
    labels: Optional["torch.Tensor"] = None,
) -> List[int]:
    """SFT (DocumentAwarePacker) 的**每行有效长度**（返回长度 = batch_size）。

    SFT 的 padding 分散在每行末尾（文档不跨行），`cu_seqlens` 是跨行逻辑累计，
    无法直接推出各行物理边界。两条数据来源（A4：数据自描述优先于运行期反推）：

    1. `row_lens` 给出时直接使用——新数据集在构建期写入了 Arrow `row_lens` 列，
       读取端传 `batch["row_lens"]` 即可（O(1)、无需复刻算法）；
    2. 否则按 `cu_seqlens` 复刻贪心装箱（旧数据集回退，见
       `_sft_row_lens_from_seg_lens`）。

    校验：
    - `sum(row_lens)` 必须等于 `cu_seqlens` 末值（有效 token 数）；
    - `labels` 给出时执行**行尾不变式**：`(B, S)` 行内布局下，行有效长度之后必须
      全是 -100。总数校验对"换行条件差一"类漂移不敏感（总数不变、行分布错），
      行尾校验恰好补上这一类，是重算路径的强校验。
    """
    B = int(batch_size)
    S = int(max_seq_len)
    if B <= 0 or S <= 0:
        raise ValueError(f"batch_size / max_seq_len must be positive integers, got ({batch_size}, {max_seq_len})")

    if row_lens is not None:
        rl = [int(v) for v in torch.as_tensor(row_lens).flatten().tolist()]
        if len(rl) != B:
            raise ValueError(
                f"row_lens length {len(rl)} does not match batch_size {B} (dataset and batch layout mismatch?)"
            )
        if any(v < 0 or v > S for v in rl):
            raise ValueError(f"row_lens contains out-of-range values (expected within [0, {S}]): {rl}")
    else:
        rl = _sft_row_lens_from_seg_lens(cu_seqlens_to_seg_lens(cu_seqlens), B, S)

    cu = normalize_cu_seqlens(cu_seqlens)
    if cu.numel() > 0:
        n_valid = int(cu[-1].item())
        total = sum(rl)
        if total != n_valid:
            raise ValueError(
                f"sum(row_lens)={total} does not match the cu_seqlens last value (valid token count) {n_valid}: "
                "row_lens / cu_seqlens are inconsistent with the batch physical layout (packing drift or snapshot mismatch?)"
            )

    if labels is not None:
        lab = torch.as_tensor(labels)
        if lab.dim() < 2 or int(lab.shape[0]) != B or int(lab.shape[1]) != S:
            raise ValueError(
                f"labels shape {tuple(lab.shape)} does not match (batch_size={B}, max_seq_len={S})"
            )
        for r, L in enumerate(rl):
            if 0 < L < S and not bool((lab[r, L:] == -100).all()):
                raise ValueError(
                    f"row_lens[{r}]={L} is inconsistent with the labels in-row layout (row tail is not -100 padding): "
                    "the replicated packing algorithm may have drifted from the Rust implementation, or row_lens/cu_seqlens do not match the batch"
                )
    return rl


def cu_seqlens_to_doc_ids(
    cu_seqlens: CuSeqlensLike,
    total_len: Optional[int] = None,
) -> torch.Tensor:
    """cu_seqlens → 逐 token 文档 id（长度 = valid_len，dtype int64）。

    ``doc_ids[i]`` = token ``i`` 所属的段索引。同段 token 共享同一 id，段间互不相同。
    空 cu_seqlens → 全 0（整段，等价标准 attention）。
    """
    cu = normalize_cu_seqlens(cu_seqlens)
    n = valid_len(cu, total_len)
    if cu.numel() == 0:
        # 整段：所有 token 同属段 0
        return torch.zeros(n, dtype=torch.long)

    seg_lens = cu_seqlens_to_seg_lens(cu)
    # 截断到实际有效长度（防御 total_len < sum(seg_lens) 的极端情况）
    seg_tensor = torch.tensor(seg_lens, dtype=torch.long)
    doc_ids = torch.repeat_interleave(
        torch.arange(len(seg_lens), dtype=torch.long), seg_tensor
    )
    if doc_ids.numel() > n:
        doc_ids = doc_ids[:n]
    elif doc_ids.numel() < n:
        # 段长度之和 < valid_len（不应发生），尾部补最后一个段 id
        pad_id = doc_ids[-1].item() if doc_ids.numel() > 0 else 0
        doc_ids = torch.cat(
            [doc_ids, torch.full((n - doc_ids.numel(),), pad_id, dtype=torch.long)]
        )
    return doc_ids


# ====================================================================
# 2. FlashAttention-2 varlen
# ====================================================================
_FLASH_ATTN_FUNC = None


def _import_flash_attn():
    """导入并**缓存** `flash_attn_varlen_func`（F-25：热路径每 batch 调用，不重复 import）。"""
    global _FLASH_ATTN_FUNC
    if _FLASH_ATTN_FUNC is not None:
        return _FLASH_ATTN_FUNC
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError:
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401  # type: ignore
        except ImportError as e:
            raise ImportError(
                "flash_attn not found. Install it with: pip install flash-attn --no-build-isolation"
            ) from e
    _FLASH_ATTN_FUNC = flash_attn_varlen_func
    return _FLASH_ATTN_FUNC


def to_flash_varlen(
    cu_seqlens: CuSeqlensLike,
    total_len: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """构造 FlashAttention-2 varlen 的输入。

    返回 ``(cu_seqlens_qk, max_seqlen)``，直接喂 ``flash_attn_varlen_func``：

        q, k, v = flatten_and_unpad(qkv, cu)        # (valid, nheads, dim)
        cu_qk, max_s = to_flash_varlen(cu)
        out = flash_attn_varlen_func(q, k, v, cu_qk, cu_qk, max_s, max_s, causal=True)

    - ``cu_seqlens_qk``: ``(num_segs + 1,)`` int32，cumulative 段边界，q/k 共用（自注意力）。
    - ``max_seqlen``: 最长段长度。
    - 空 cu_seqlens → 单段 ``[0, valid_len]``（等价标准 attention）。

    **纯转换，不要求安装 flash-attn**：返回物只是两个描述 varlen 边界的量，可直接喂
    自研 kernel。依赖校验在门面 `build_attention(..., "flash_varlen")` 处（那里才是
    "要用 flash" 的意图表达点）。
    """
    cu = normalize_cu_seqlens(cu_seqlens)
    if cu.numel() == 0:
        n = valid_len(cu, total_len)
        cu = torch.tensor([0, n], dtype=torch.long)
    # 边界校验：段边界不能越过 q 序列的有效长度，否则 flash_attn 内部才报错
    # （信息不直观）。这里给出明确诊断。
    if total_len is not None and int(cu[-1].item()) > int(total_len):
        raise ValueError(
            f"cu_seqlens last value ({int(cu[-1].item())}) exceeds total_len ({int(total_len)}): "
            "the document boundaries describe more tokens than the sequence provides"
        )
    seg_lens = cu_seqlens_to_seg_lens(cu)
    max_s = max(seg_lens) if seg_lens else 0
    return cu.to(torch.int32).contiguous(), int(max_s)


# ====================================================================
# 3. FlexAttention (PyTorch >= 2.5)
# ====================================================================
_CREATE_BLOCK_MASK = None


def _import_flex_attention():
    """导入并**缓存** `create_block_mask`（F-25：热路径每 batch 调用，不重复 import）。"""
    global _CREATE_BLOCK_MASK
    if _CREATE_BLOCK_MASK is not None:
        return _CREATE_BLOCK_MASK
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as e:
        raise ImportError(
            "FlexAttention requires PyTorch >= 2.5; the current torch version lacks "
            "torch.nn.attention.flex_attention."
        ) from e
    _CREATE_BLOCK_MASK = create_block_mask
    return _CREATE_BLOCK_MASK


def to_flex_block_mask(
    cu_seqlens: CuSeqlensLike,
    total_len: Optional[int] = None,
    causal: bool = True,
    block_size: int = 128,
    device: Optional[Union[str, "torch.device"]] = None,
    compile_mask: bool = True,
) -> Any:
    """构造 FlexAttention 的 ``BlockMask``（块对角 + 可选因果）。

    实现 PyTorch 官方推荐的"逐 token document_id + mask_mod"范式：
    从 cu_seqlens 派生 ``doc_ids``，``mask_mod`` 判断同文档（+ 因果），
    ``create_block_mask`` 编译为 BlockMask。**block_size 无关数据层**，运行期指定。

    - 序列长度会向上 pad 到 ``block_size`` 倍数（FlexAttention 硬性要求），
      pad 区 ``doc_id = -1``，仅与自身互 attend，不污染真实段。
    - ``device``：默认跟随**实际可用**的加速器（不可用则 CPU）。必须在这里显式
      解析而不是交给 torch —— `create_block_mask(device=None)` 会退到
      `torch.accelerator.current_accelerator()`（只看构建/枚举，不看驱动是否可用），
      在"CUDA 版 torch + 驱动过旧"的机器上直接抛驱动错误；而且它与本函数把
      ``doc_ids`` 放在 CPU 的行为**不一致**，GPU 机器上会造出"掩码在 GPU、doc_ids
      在 CPU"的坏 BlockMask。
    - ``causal=True``：段内因果（标准 LM）；``causal=False``：段内双向。
    - ``compile_mask=True``：传 ``_compile=True``，PyTorch 官方建议可降低峰值内存与
      运行时间约一个数量级。

    用法（官方"模式 B"：forward 开头算一次，跨层重用）::

        block_mask = to_flex_block_mask(cu, causal=True)
        for layer in model.layers:
            x = layer(x, block_mask=block_mask)   # 所有层复用同一 BlockMask
    """
    create_block_mask = _import_flex_attention()
    if block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size}")

    if device is None:
        # 显式解析：torch 的 None → accelerator 探测不检查驱动是否真的可用，
        # 且会让掩码与 doc_ids 分居两个设备（见 docstring）。
        acc = getattr(torch, "accelerator", None)
        try:
            device = acc.current_accelerator() if (acc and acc.is_available()) else "cpu"
        except Exception:
            device = "cpu"

    cu = normalize_cu_seqlens(cu_seqlens)
    n = valid_len(cu, total_len)
    doc_ids = cu_seqlens_to_doc_ids(cu, n).to(device)

    # pad 到 block_size 倍数（FlexAttention 要求 Q_LEN/KV_LEN 是 block_size 的倍数）
    pad = (-n) % block_size
    if pad:
        doc_ids = F.pad(doc_ids, (0, pad), value=-1)
    total_padded = n + pad

    def _doc_causal(b, h, q_idx, kv_idx):
        # 同文档 + 因果。pad 区 doc_id=-1 仅与自身（-1）互 attend，不污染真实段。
        return (doc_ids[q_idx] == doc_ids[kv_idx]) & (q_idx >= kv_idx)

    def _doc_full(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]

    mask_mod = _doc_causal if causal else _doc_full

    # 不同 torch 版本 create_block_mask 签名略有差异，兼容 _compile / BLOCK_SIZE 关键字
    try:
        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=total_padded,
            KV_LEN=total_padded,
            device=device,
            BLOCK_SIZE=block_size,
            _compile=compile_mask,
        )
    except TypeError:
        # 旧版本可能不支持 _compile 或 BLOCK_SIZE 关键字
        try:
            return create_block_mask(
                mask_mod, None, None, total_padded, total_padded,
                device=device, BLOCK_SIZE=block_size,
            )
        except TypeError:
            return create_block_mask(mask_mod, None, None, total_padded, total_padded)


# ====================================================================
# 4. xformers
# ====================================================================
_XFORMERS_MASKS = None


def _import_xformers():
    """导入并**缓存** xformers 的两种 mask 类（F-25：热路径每 batch 调用，不重复 import）。"""
    global _XFORMERS_MASKS
    if _XFORMERS_MASKS is not None:
        return _XFORMERS_MASKS
    try:
        from xformers.ops import (  # noqa: F401
            BlockDiagonalMask,
            BlockDiagonalCausalMask,
        )
    except ImportError as e:
        raise ImportError(
            "xformers not found. Install it with: pip install xformers"
        ) from e
    _XFORMERS_MASKS = (BlockDiagonalMask, BlockDiagonalCausalMask)
    return _XFORMERS_MASKS


def to_xformers_block_diag(
    cu_seqlens: CuSeqlensLike,
    total_len: Optional[int] = None,
    causal: bool = True,
) -> Any:
    """构造 xformers 的块对角 mask。

    - ``causal=True`` → ``BlockDiagonalCausalMask``（段内因果，段间隔离）。
    - ``causal=False`` → ``BlockDiagonalMask``（段内双向，段间隔离）。

    用法::

        mask = to_xformers_block_diag(cu, causal=True)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)

    注：xformers 的 ``BlockDiagonalMask`` 同样把整条 packed 序列视为单 batch，
    ``from_seqlens(seg_lens)`` 接受段长度列表。
    """
    BlockDiagonalMask, BlockDiagonalCausalMask = _import_xformers()
    cu = normalize_cu_seqlens(cu_seqlens)
    seg_lens = cu_seqlens_to_seg_lens(cu)
    if not seg_lens:
        # 空 cu → 整段单块
        n = valid_len(cu, total_len)
        seg_lens = [n]
    cls = BlockDiagonalCausalMask if causal else BlockDiagonalMask
    return cls.from_seqlens(seg_lens)


# ====================================================================
# 5. unpad / repad 辅助（与 backend 无关）
# ====================================================================
def flatten_and_unpad(
    batch_tensor: "torch.Tensor",
    cu_seqlens: CuSeqlensLike,
    total_len: Optional[int] = None,
    packer: Optional[str] = None,
    *,
    row_lens: Optional[CuSeqlensLike] = None,
) -> "torch.Tensor":
    """把 ``(batch_size, max_seq_len, ...)`` 还原为连续文档流 → ``(valid_len, ...)``。

    两种布局（由 `packer` 选择，建议传 ``batch.get("format")`` 自动选择）：

    - **PT (GreedyPacker)**：文档跨行连续拼接成一维流，padding 集中在流末尾，
      ``flat[:valid_len]`` 即连续文档流；
    - **SFT (DocumentAwarePacker)**（``packer="sft"``）：文档不跨行、padding 分散在
      每行末尾。按行有效长度切片拼接得到连续流，文档段边界与 ``cu_seqlens``
      一一对应（可直接接 varlen / Flex attention）：

          ids_1d = flatten_and_unpad(input_ids, cu, packer="sft")
          block_mask = to_flex_block_mask(cu, causal=True)

    ``row_lens``（可选）：SFT 数据集构建期写入的 Arrow 列
    （``batch["row_lens"]``，每行有效长度，A4 数据自描述）。给出时 O(1) 直读；
    省略时回退为复刻 Rust 装箱算法（旧数据集，见 :func:`sft_row_lens`）。

    对 ``input_ids``、``hidden states``、``q/k/v`` 等任意后续维均适用::

        ids_1d = flatten_and_unpad(input_ids, cu, packer=batch.get("format"),
                                   row_lens=batch.get("row_lens"))   # (valid,)
        q_1d   = flatten_and_unpad(q, cu, packer=batch.get("format"),
                                   row_lens=batch.get("row_lens"))   # (valid, nheads, dim)
    """
    p = None
    if packer is not None:
        p = str(packer).strip().lower()
        if p not in ("pt", "sft"):
            raise ValueError(f'packer must be "pt" / "sft" / None, got {packer!r}')
    if batch_tensor.dim() < 2:
        raise ValueError(
            f"batch_tensor requires at least 2 dims (batch_size, max_seq_len, ...), "
            f"got {batch_tensor.dim()} dims"
        )
    n = valid_len(cu_seqlens, total_len)

    if p == "sft":
        B, S = int(batch_tensor.shape[0]), int(batch_tensor.shape[1])
        rl = sft_row_lens(cu_seqlens, B, S, row_lens=row_lens)
        if not any(rl):
            return batch_tensor.reshape(-1, *batch_tensor.shape[2:])[:0].contiguous()
        out = torch.cat([batch_tensor[r, :L] for r, L in enumerate(rl) if L > 0], dim=0)
        if int(out.shape[0]) != n:   # 自校验：装箱算法一旦漂移立即响亮失败
            raise ValueError(
                f"SFT unpad produced {out.shape[0]} tokens, inconsistent with valid_len {n}: "
                "row_lens / cu_seqlens are inconsistent with the batch physical layout"
            )
        return out.contiguous()

    # PT（含 packer=None 的默认路径）：row-major 展平，padding 集中在流末尾
    flat = batch_tensor.reshape(-1, *batch_tensor.shape[2:])
    return flat[:n].contiguous()


def repad_to_batch(
    flat_tensor: "torch.Tensor",
    cu_seqlens: CuSeqlensLike,
    batch_size: int,
    max_seq_len: int,
    pad_value: float = 0.0,
    packer: Optional[str] = None,
    *,
    row_lens: Optional[CuSeqlensLike] = None,
) -> "torch.Tensor":
    """把 ``(valid_len, ...)`` 还原为 ``(batch_size, max_seq_len, ...)``，补齐区填 ``pad_value``。

    :func:`flatten_and_unpad` 的逆操作——**布局必须对称**：SFT 的 flatten 必须配
    ``packer="sft"`` 的 repad，否则 logits 与 ``(B, S)`` 行内布局的 labels 静默错位：

    - **PT**：有效 token 连续铺在前 n 个位置、尾部集中 pad；
    - **SFT**（``packer="sft"``）：按行有效长度把 1D 流拆回各行、行尾 pad
      （与 SFT labels 的行内布局对齐）。

        logits_1d = model(ids_1d)                        # (valid, vocab)
        logits    = repad_to_batch(logits_1d, cu, B, S, packer=fmt,
                                   row_lens=batch.get("row_lens"))   # (B, S, vocab)
        loss = loss_fn(logits, labels)                   # labels 同样 (B, S)
    """
    p = None
    if packer is not None:
        p = str(packer).strip().lower()
        if p not in ("pt", "sft"):
            raise ValueError(f'packer must be "pt" / "sft" / None, got {packer!r}')
    n = valid_len(cu_seqlens)
    capacity = batch_size * max_seq_len
    if flat_tensor.shape[0] < n:
        raise ValueError(
            f"flat_tensor first dim {flat_tensor.shape[0]} is smaller than the valid length {n}"
        )
    out = torch.full(
        (capacity, *flat_tensor.shape[1:]),
        pad_value,
        dtype=flat_tensor.dtype,
        device=flat_tensor.device,
    )
    if p == "sft":
        rl = sft_row_lens(cu_seqlens, batch_size, max_seq_len, row_lens=row_lens)
        pos = 0
        for r, L in enumerate(rl):
            if L > 0:
                out[r * max_seq_len:r * max_seq_len + L] = flat_tensor[pos:pos + L]
                pos += L
    else:
        out[:n] = flat_tensor[:n]
    return out.reshape(batch_size, max_seq_len, *flat_tensor.shape[1:]).contiguous()


# ====================================================================
# 6. 统一入口
# ====================================================================
class AttentionBackend(Enum):
    """支持的 attention backend。"""

    FLASH_VARLEN = "flash_varlen"
    FLEX = "flex"
    XFORMERS = "xformers"


def build_attention(
    cu_seqlens: CuSeqlensLike,
    backend: Union[AttentionBackend, str],
    total_len: Optional[int] = None,
    causal: bool = True,
    **kwargs: Any,
) -> Any:
    """按 backend 分发，构造对应的 mask / varlen 输入。

    参数:
        cu_seqlens: 文档段边界（list 或 tensor）。
        backend: ``AttentionBackend`` 枚举或字符串
            (``"flash_varlen"`` / ``"flex"`` / ``"xformers"``)。
        total_len: 有效长度；空 cu_seqlens 时必填。
        causal: 是否段内因果（默认 True，标准 LM）。
        **kwargs: 透传给具体 backend（如 FlexAttention 的 ``block_size``/``device``）。

    返回:
        - ``flash_varlen`` → ``(cu_seqlens_qk_int32, max_seqlen)``
        - ``flex`` → ``BlockMask``
        - ``xformers`` → ``BlockDiagonalCausalMask`` / ``BlockDiagonalMask``
    """
    if isinstance(backend, str):
        try:
            backend = AttentionBackend(backend.lower())
        except ValueError:
            raise ValueError(
                f"unknown backend: {backend!r}, valid options: "
                f"{[b.value for b in AttentionBackend]}"
            )

    if backend is AttentionBackend.FLASH_VARLEN:
        # F-13：不再静默忽略 causal——flash_varlen 的返回物（cu_seqlens/max_seqlen）不含
        # 因果信息，causal 由调用方在 flash_attn_varlen_func 调用时指定；传 False 必须响亮拒绝。
        if not causal:
            raise ValueError(
                "flash_varlen backend does not support causal=False: this entry only builds "
                "(cu_seqlens, max_seqlen), which carries no causality (causality is specified by the caller "
                "at flash_attn_varlen_func(..., causal=)). For bidirectional attention within segments, use "
                "'flex' or 'xformers' (both support causal=False)."
            )
        # F-13 补充：依赖校验只在门面做（`to_flash_varlen` 本体是纯转换，
        # 不强制用户为"喂自研 kernel"安装 flash-attn）。
        _import_flash_attn()
        return to_flash_varlen(cu_seqlens, total_len=total_len)
    elif backend is AttentionBackend.FLEX:
        return to_flex_block_mask(
            cu_seqlens, total_len=total_len, causal=causal, **kwargs
        )
    elif backend is AttentionBackend.XFORMERS:
        # kwargs 目前只被 Flex 分支消费；xformers 分支显式拒绝，避免用户以为
        # 参数生效了却静默被丢掉（后续 xformers 需要参数时在此显式开放）。
        if kwargs:
            raise ValueError(
                f"xformers backend does not accept extra keyword arguments, got "
                f"{sorted(kwargs)} (supported args: cu_seqlens, backend, total_len, causal). "
                "Use the 'flex' backend if you need options like block_size/device."
            )
        return to_xformers_block_diag(
            cu_seqlens, total_len=total_len, causal=causal
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled backend: {backend}")

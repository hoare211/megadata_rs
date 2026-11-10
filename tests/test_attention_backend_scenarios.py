"""mask_adapter（注意力后端适配层）全场景测试。

重点不是"函数能跑"，而是**语义正确**：段内 attend / 段间隔离 / 因果性 / padding
不污染真实段 —— 用真实 attention 输出与手工参考实现逐位对比。
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "python"))

import megadata.mask_adapter as M  # noqa: E402


def reference_attention(cu, valid, block_size, causal, q, k, v):
    """按 mask_mod 的字面语义手算参考输出（含 padding 的 -1 语义）。"""
    doc_ids = M.cu_seqlens_to_doc_ids(cu, valid)
    pad = (-valid) % block_size
    if pad:
        doc_ids = torch.nn.functional.pad(doc_ids, (0, pad), value=-1)
    n = doc_ids.numel()
    idx = torch.arange(n)
    allow = doc_ids[:, None] == doc_ids[None, :]
    if causal:
        allow = allow & (idx[:, None] >= idx[None, :])
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allow, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


# ==============================================================================
# 1. FlexAttention：语义正确性（段隔离 + 因果 + padding 不污染）
# ==============================================================================
CU = [0, 3, 5, 11]          # 三段：3 / 2 / 6
BLOCK = 8


def _flex():
    return pytest.importorskip("torch.nn.attention.flex_attention",
                               reason="需要 torch >= 2.5 的 FlexAttention")


def test_flex_matches_reference_causal():
    _flex()
    from torch.nn.attention.flex_attention import flex_attention

    valid = CU[-1]
    mask = M.to_flex_block_mask(CU, causal=True, block_size=BLOCK, compile_mask=False)
    n = mask.shape[2]
    torch.manual_seed(0)
    q = torch.randn(1, 1, n, 8)
    k = torch.randn(1, 1, n, 8)
    v = torch.randn(1, 1, n, 8)

    got = flex_attention(q, k, v, block_mask=mask)
    ref = reference_attention(CU, valid, BLOCK, True, q, k, v)
    assert torch.allclose(got[..., :valid, :], ref[..., :valid, :], atol=2e-5), \
        "FlexAttention 输出与参考实现不一致（段隔离/因果/padding 语义有误）"


def test_flex_matches_reference_non_causal():
    _flex()
    from torch.nn.attention.flex_attention import flex_attention

    valid = CU[-1]
    mask = M.to_flex_block_mask(CU, causal=False, block_size=BLOCK, compile_mask=False)
    n = mask.shape[2]
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1, n, 8) for _ in range(3))
    got = flex_attention(q, k, v, block_mask=mask)
    ref = reference_attention(CU, valid, BLOCK, False, q, k, v)
    assert torch.allclose(got[..., :valid, :], ref[..., :valid, :], atol=2e-5)


def test_flex_segments_are_isolated():
    """段间隔离的直接断言：改动第二段的 token，不应影响第一段的输出。"""
    _flex()
    from torch.nn.attention.flex_attention import flex_attention

    mask = M.to_flex_block_mask(CU, causal=True, block_size=BLOCK, compile_mask=False)
    n = mask.shape[2]
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 1, n, 8) for _ in range(3))
    base = flex_attention(q, k, v, block_mask=mask)

    k2, v2 = k.clone(), v.clone()
    k2[..., 5:11, :] = torch.randn_like(k2[..., 5:11, :])   # 第 2 段（[5,11)）
    v2[..., 5:11, :] = torch.randn_like(v2[..., 5:11, :])
    perturbed = flex_attention(q, k2, v2, block_mask=mask)

    assert torch.allclose(base[..., :5, :], perturbed[..., :5, :], atol=1e-6), \
        "第 1 段的输出受第 2 段影响：段间未隔离（跨文档 attend）"
    assert not torch.allclose(base[..., 5:11, :], perturbed[..., 5:11, :]), \
        "第 2 段自身改了却没变化：断言本身失效"


def test_flex_block_size_padding_contract():
    _flex()
    valid = CU[-1]
    for bs in (8, 16, 32):
        mask = M.to_flex_block_mask(CU, causal=True, block_size=bs, compile_mask=False)
        assert mask.shape[2] == mask.shape[3]
        assert mask.shape[2] % bs == 0
        assert mask.shape[2] >= valid


def test_flex_invalid_block_size():
    _flex()
    with pytest.raises(ValueError, match="block_size"):
        M.to_flex_block_mask(CU, causal=True, block_size=0, compile_mask=False)


def test_flex_empty_cu_requires_total_len():
    _flex()
    with pytest.raises(ValueError, match="total_len"):
        M.to_flex_block_mask([], causal=True, block_size=8, compile_mask=False)
    mask = M.to_flex_block_mask([], total_len=10, causal=True, block_size=8,
                                compile_mask=False)
    assert mask.shape[2] == 16


# ==============================================================================
# 2. 依赖缺失时的报错要可操作
# ==============================================================================
def test_flash_varlen_pure_conversion_without_flash_attn():
    """`to_flash_varlen` 是纯转换（喂自研 kernel 的场景不强制装 flash-attn）。"""
    cu_i32, max_s = M.to_flash_varlen(CU)
    assert cu_i32.dtype == torch.int32
    assert cu_i32.tolist() == CU
    assert max_s == max(b - a for a, b in zip(CU, CU[1:]))


def test_build_attention_flash_varlen_reports_install_hint():
    """依赖校验在门面 `build_attention` 处（那里才是"要用 flash"的意图表达点）。"""
    try:
        import flash_attn  # noqa: F401
        pytest.skip("本环境装了 flash_attn，跳过缺失路径")
    except ImportError:
        pass
    with pytest.raises(ImportError) as exc:
        M.build_attention(CU, "flash_varlen")
    assert "flash-attn" in str(exc.value), f"报错未给出安装指引：{exc.value}"


def test_xformers_reports_install_hint():
    try:
        import xformers  # noqa: F401
        pytest.skip("本环境装了 xformers，跳过缺失路径")
    except ImportError:
        pass
    with pytest.raises(ImportError) as exc:
        M.to_xformers_block_diag(CU)
    assert "xformers" in str(exc.value)


def test_build_attention_dispatch():
    _flex()
    assert M.build_attention(CU, "flex", block_size=8, compile_mask=False) is not None
    assert M.build_attention(CU, M.AttentionBackend.FLEX, block_size=8,
                             compile_mask=False) is not None
    with pytest.raises(ValueError, match="unknown backend"):
        M.build_attention(CU, "no_such_backend")
    # 大小写不敏感（文档承诺）
    assert M.build_attention(CU, "FLEX", block_size=8, compile_mask=False) is not None


# ==============================================================================
# 3. flatten_and_unpad / repad_to_batch（含 packer 防护）
# ==============================================================================
def test_flatten_and_unpad_pt_semantics():
    batch = torch.arange(2 * 8).reshape(2, 8).clone()
    flat = M.flatten_and_unpad(batch, [0, 3, 5])
    assert flat.tolist() == [0, 1, 2, 3, 4]


def test_flatten_and_unpad_sft_semantics():
    """SFT 行内布局：按行有效长度切片还原连续流（旧实现直接拒绝，现由 row_lens / 复刻装箱支持）。

    `packer="pt"` 的 flat[:n] 会混入行间 padding —— 这正是旧实现拒绝 SFT 的原因；
    `packer="sft"` 按 `row_lens`（列直读或复刻装箱）逐行切片，得到真正的连续文档流。
    """
    batch = torch.zeros(2, 8, dtype=torch.long)
    batch[0, :3] = torch.tensor([0, 1, 2])
    batch[1, :6] = torch.tensor([3, 4, 5, 6, 7, 8])
    cu = [0, 3, 9]                       # 跨行逻辑累计：行0 有效 3 + 行1 有效 6
    # PT 语义（取前 valid_len 个）：行间 padding 混入流中
    assert M.flatten_and_unpad(batch, cu, packer="pt").tolist() == [0, 1, 2, 0, 0, 0, 0, 0, 3]
    # SFT 语义：严格按行有效长度拼接
    flat = M.flatten_and_unpad(batch, cu, packer="sft")
    assert flat.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # row_lens 列直读（新数据集自描述列）与复刻装箱结果一致
    flat2 = M.flatten_and_unpad(batch, cu, packer="sft", row_lens=[3, 6])
    assert torch.equal(flat, flat2)
    # repad 布局必须对称（SFT flatten 配 SFT repad），否则与 (B, S) 行内 labels 静默错位
    back = M.repad_to_batch(flat, cu, batch_size=2, max_seq_len=8, pad_value=9, packer="sft")
    expect = batch.clone()
    expect[0, 3:] = 9
    expect[1, 6:] = 9
    assert torch.equal(back, expect)
    # 非法 packer 仍响亮拒绝
    with pytest.raises(ValueError, match="packer"):
        M.flatten_and_unpad(batch, cu, packer="magic")


def test_flatten_and_unpad_requires_2d():
    with pytest.raises(ValueError, match="at least 2 dims"):
        M.flatten_and_unpad(torch.arange(8), [0, 3, 5])


def test_repad_roundtrip():
    batch = torch.arange(2 * 8).reshape(2, 8).clone()
    cu = [0, 3, 5]
    flat = M.flatten_and_unpad(batch, cu).float()
    back = M.repad_to_batch(flat, cu, batch_size=2, max_seq_len=8, pad_value=-1.0)
    assert back.reshape(2, 8).tolist() == [[0, 1, 2, 3, 4, -1, -1, -1],
                                           [-1, -1, -1, -1, -1, -1, -1, -1]]


def test_repad_rejects_short_tensor():
    with pytest.raises(ValueError, match="smaller than the valid length"):
        M.repad_to_batch(torch.zeros(2, 3), [0, 3, 5], batch_size=1, max_seq_len=8)


# ==============================================================================
# 4. cu_seqlens 基础换算的边界
# ==============================================================================
@pytest.mark.parametrize("raw,want", [
    ([0, 3, 3, 5, 11], [0, 3, 5, 11]),     # 去重相邻相等
    ([3, 5, 11], [0, 3, 5, 11]),           # 补前导 0
    ([], []),
    (None, []),
])
def test_normalize_cu_seqlens(raw, want):
    assert M.normalize_cu_seqlens(raw).tolist() == want


def test_valid_len_and_seg_lens():
    assert M.valid_len([0, 3, 5, 11]) == 11
    assert M.valid_len([], total_len=7) == 7
    with pytest.raises(ValueError, match="total_len"):
        M.valid_len([])
    assert M.cu_seqlens_to_seg_lens([0, 3, 5, 11]) == [3, 2, 6]
    assert M.cu_seqlens_to_seg_lens([]) == []


def test_doc_ids_contract():
    doc_ids = M.cu_seqlens_to_doc_ids([0, 3, 5, 11])
    assert doc_ids.tolist() == [0, 0, 0, 1, 1] + [2] * 6
    assert doc_ids.dtype == torch.int64
    # 空 cu → 整段（全 0），等价标准 attention
    assert M.cu_seqlens_to_doc_ids([], total_len=4).tolist() == [0, 0, 0, 0]

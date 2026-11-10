"""文档级旁路列（doc_ids / loss_weights）端到端：构建 → 落盘 → 读取 → DLPack 透出。

覆盖范围：
- 构建期 `BuildConfig(extra_columns=True)` ⇒ 写 `.col_doc_ids` / `.col_loss_weights`
- 读取端按 manifest schema key 透出 DLPack capsule（i32 / float16）
- 关闭开关时：无列文件、batch dict 不出现这两个 key（零额外开销）
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from _scenario_kit import (
    MAX_SEQ_LEN,
    build_dataset,
    pt_records,
    tokenizer_path,
)


def _require_torch() -> None:
    if not hasattr(torch, "from_dlpack"):
        pytest.skip("torch.from_dlpack unavailable")


def test_extra_columns_roundtrip_via_dlpack(tmp_path):
    """开启列扩展：batch 里必须拿到 doc_ids(int32) 与 loss_weights(float16)。"""
    _require_torch()
    out = str(tmp_path / "cols")
    index = build_dataset(
        out, pt_records(40), "pt",
        max_seq_len=MAX_SEQ_LEN, batch_size=4, batches_per_shard=4,
        extra_columns=True,
    )

    # 1. 落盘产物：列文件存在，manifest schema 有声明
    ds_dir = os.path.dirname(index)
    cols = sorted(f for f in os.listdir(ds_dir) if ".col_" in f)
    assert cols, f"expected .col_* files, got {os.listdir(ds_dir)}"
    with open(index, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["schema"].get("doc_ids") == "list<int32>"
    assert manifest["schema"].get("loss_weights") == "list<f16>"

    # 2. 读取端透出
    from megadata import BatchStream

    loader = BatchStream(
        index_json_path=index, batch_size=4, max_seq_len=MAX_SEQ_LEN,
        world_size=1, rank=0, shuffle=False,
    )
    batch = next(iter(loader))
    assert "doc_ids" in batch, f"batch keys: {sorted(batch)}"
    assert "loss_weights" in batch, f"batch keys: {sorted(batch)}"

    doc_ids = torch.from_dlpack(batch["doc_ids"])
    weights = torch.from_dlpack(batch["loss_weights"])
    assert doc_ids.dtype == torch.int32
    assert weights.dtype == torch.float16
    assert doc_ids.shape == (4, MAX_SEQ_LEN)
    assert weights.shape == (4, MAX_SEQ_LEN)

    # 3. 契约：padding 位置 doc_ids = -1、权重 = 0；有效位置 doc_ids >= 0、权重 = 1
    cu = batch.get("cu_seqlens")
    valid = int(cu[-1]) if cu is not None and len(cu) else doc_ids.numel()
    flat = doc_ids.reshape(-1)
    wflat = weights.reshape(-1)
    assert bool((flat[:valid] >= 0).all()), "有效 token 必须归属某个文档"
    if valid < flat.numel():
        assert bool((flat[valid:] == -1).all()), "pad 位置 doc_ids 必须是 -1"
        assert bool((wflat[valid:] == 0).all()), "pad 位置权重必须是 0"
    assert bool((wflat[:valid] == 1).all()), "未配置加权时有效 token 权重默认 1.0"


def test_without_extra_columns_has_no_column_files(tmp_path):
    """关闭列扩展：不产出列文件、batch dict 不含这两列（向后兼容、零开销）。"""
    _require_torch()
    out = str(tmp_path / "plain")
    index = build_dataset(
        out, pt_records(20), "pt",
        max_seq_len=MAX_SEQ_LEN, batch_size=4, batches_per_shard=4,
    )
    ds_dir = os.path.dirname(index)
    assert not [f for f in os.listdir(ds_dir) if ".col_" in f], "未开启时不该有 .col_* 文件"

    from megadata import BatchStream

    loader = BatchStream(
        index_json_path=index, batch_size=4, max_seq_len=MAX_SEQ_LEN,
        world_size=1, rank=0, shuffle=False,
    )
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "doc_ids" not in batch and "loss_weights" not in batch


def test_tokenizer_available() -> None:
    """环境自检（分词器缺失会 skip）。"""
    assert tokenizer_path()

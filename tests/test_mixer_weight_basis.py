"""Mixer weight basis（锚点 2 / D3 / R9）口径测试。

覆盖 handoff §4.4 的四条验收：
  1. 纯文本数据集不告警（不刷屏）；
  2. 多模态数据集不显式配 basis → WARN 一次；显式配 → 不告警；
  3. 旧 manifest（无 stats_version）：text_tokens basis == total_tokens basis
     （_effective_modality_tokens 解释函数生效）；
  4. 三种 basis 数值各自正确（构造已知统计的 manifest 断言权重比）。

数据集为**手工 manifest + 假 .bin**（构造期只需要 shard 可打开，不迭代数据），
不依赖分词器（对齐 P0-T7 做法）。
"""
import json
import os
import warnings

import pytest

from megadata import MixBatchStream

BS, MSL = 4, 8


def _make_manifest(dir_path: str, nbatches: int, **stats) -> str:
    """最小合法 Memmap manifest + 可打开的假 .bin（全零字节，不迭代）。"""
    os.makedirs(dir_path, exist_ok=True)
    manifest = {
        "version": "2.0",
        "format": "Memmap",
        "max_seq_len": MSL,
        "global_stats": {
            "batch_size": BS, "total_batches": nbatches, "total_shards": 1,
            **stats,
        },
        "schema": {"input_ids": "int32", "labels": "int32", "position_ids": "int32",
                   "cu_seqlens": "list<int32>"},
        "shards": [{"shard_id": 0, "filename": "shard_00000.bin",
                    "num_batches": nbatches, "size_bytes": nbatches * BS * MSL * 4,
                    "hashes": {}}],
    }
    index = os.path.join(dir_path, "index.json")
    with open(index, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(dir_path, "shard_00000.bin"), "wb") as f:
        f.write(b"\x00" * (nbatches * BS * MSL * 4))
    return index


def _build(paths, weight_basis=None):
    """两源混采 loader（不迭代数据，只断言构造期口径）。"""
    configs = [{"path": p} for p in paths]
    kwargs = {} if weight_basis is None else {"weight_basis": weight_basis}
    return MixBatchStream(configs=configs, batch_size=BS, max_seq_len=MSL, **kwargs)


def _pure_text_stats(tokens, nbatches):
    return {"total_tokens": tokens, "total_batches": nbatches}


def _mm_stats(tokens, text, image, nbatches):
    return {"total_tokens": tokens, "stats_version": 2,
            "total_text_tokens": text, "total_image_tokens": image,
            "total_batches": nbatches}


def test_pure_text_default_basis_does_not_warn(tmp_path):
    """验收 1：纯文本数据集不显式配 basis → 不告警（不刷屏）。"""
    a = _make_manifest(str(tmp_path / "a"), 4, **_pure_text_stats(1000, 4))
    b = _make_manifest(str(tmp_path / "b"), 4, **_pure_text_stats(1000, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader = _build([a, b])
    assert len(loader._base_weights) == 2
    assert loader._weight_basis == "total_tokens"
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == [], f"纯文本数据集不应告警: {[str(w.message) for w in user_warnings]}"


def test_multimodal_default_basis_warns_once_explicit_does_not(tmp_path):
    """验收 2：多模态数据集不显式配 basis → WARN 一次；显式配（哪怕与默认相同）→ 不告警。"""
    a = _make_manifest(str(tmp_path / "a"), 4, **_mm_stats(1000, 600, 400, 4))
    b = _make_manifest(str(tmp_path / "b"), 4, **_mm_stats(2000, 2000, 0, 4))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build([a, b])  # 不显式配 basis
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)
                     and "weight_basis" in str(w.message)]
    assert len(user_warnings) == 1, f"应恰好 WARN 一次: {[str(w.message) for w in user_warnings]}"

    for explicit in ("total_tokens", "text_tokens", "samples"):
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            loader = _build([a, b], weight_basis=explicit)
        user_warnings2 = [w for w in caught2 if issubclass(w.category, UserWarning)
                          and "weight_basis" in str(w.message)]
        assert user_warnings2 == [], f"显式 basis={explicit} 不应告警"
        assert loader._weight_basis == explicit


def test_invalid_basis_rejected(tmp_path):
    """口径取值白名单：未知 basis 报错（防拼写错误静默落回默认口径）。"""
    a = _make_manifest(str(tmp_path / "a"), 4, **_pure_text_stats(1000, 4))
    b = _make_manifest(str(tmp_path / "b"), 4, **_pure_text_stats(1000, 4))
    with pytest.raises(ValueError, match="weight_basis"):
        _build([a, b], weight_basis="tokns")


def test_old_manifest_text_basis_equals_total_basis(tmp_path):
    """验收 3：旧 manifest（无 stats_version）→ text_tokens basis 结果
    == total_tokens basis 结果（解释函数 text ≡ total 生效）。"""
    a = _make_manifest(str(tmp_path / "a"), 4, **_pure_text_stats(1000, 4))
    b = _make_manifest(str(tmp_path / "b"), 4, **_pure_text_stats(3000, 4))
    total = _build([a, b], weight_basis="total_tokens")
    text = _build([a, b], weight_basis="text_tokens")
    assert text._base_weights == pytest.approx(total._base_weights)
    assert text._base_weights == pytest.approx([0.25, 0.75])


def test_three_bases_numeric(tmp_path):
    """验收 4：三种 basis 数值各自正确（构造已知统计断言权重比）。

    A：total=1000（text 600 + image 400），samples = 5×4 = 20
    B：total=2000（text 2000），samples = 20×4 = 80
      - total_tokens → [1000/3000, 2000/3000] = [1/3, 2/3]
      - text_tokens  → [600/2600, 2000/2600]
      - samples      → [20/100, 80/100] = [0.2, 0.8]
    三者两两互异（口径污染可被数值区分）。
    """
    a = _make_manifest(str(tmp_path / "a"), 5, **_mm_stats(1000, 600, 400, 5))
    b = _make_manifest(str(tmp_path / "b"), 20, **_mm_stats(2000, 2000, 0, 20))

    total = _build([a, b], weight_basis="total_tokens")._base_weights
    text = _build([a, b], weight_basis="text_tokens")._base_weights
    samples = _build([a, b], weight_basis="samples")._base_weights

    assert total == pytest.approx([1 / 3, 2 / 3])
    assert text == pytest.approx([600 / 2600, 2000 / 2600])
    assert samples == pytest.approx([0.2, 0.8])
    # 口径互异断言：三者两两不同（防"换了 basis 数值没变"的口径污染回归）
    assert total != pytest.approx(text)
    assert total != pytest.approx(samples)
    assert text != pytest.approx(samples)


def test_zero_count_basis_fails_fast(tmp_path):
    """某子流在该口径下基数为 0（如 text_tokens=0 的纯图像流）→ 报错而非静默。"""
    a = _make_manifest(str(tmp_path / "a"), 4, **_mm_stats(1000, 600, 400, 4))
    # B：text_tokens = 0（visual-only 流）
    b = _make_manifest(str(tmp_path / "b"), 4,
                       **{"total_tokens": 500, "stats_version": 2,
                          "total_text_tokens": 0, "total_image_tokens": 500,
                          "total_batches": 4})
    with pytest.raises(ValueError, match="text_tokens"):
        _build([a, b], weight_basis="text_tokens")

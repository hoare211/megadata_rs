"""多模态 P0 读取侧集成测试：`debug_read_samples` 驱动 `read_samples` 全链路。

覆盖 P0 读取契约（方案 §3.1 Sample 契约 + D5 chunk 双约束 + 失败域）：
- 三种图像引用形态（image_b64 / image_path / image_url）与 data: URI
- SkipAndCount 按错误分类计数，错误样本占位不丢 id
- FailFast 显式中止（fail_fast_triggered / aborted / abort_msg）
- chunk 双约束（max_bytes 字节预算 / max_samples 条数上限）
- sample_id 全链无漏无重（含 SkipAndCount 占位、跨输入文件）

不经分词器、不产 shard——只验证读取侧；生产 vlm 构建管线 P2 才接通。
"""

from __future__ import annotations

import base64

import pytest

from _scenario_kit import write_jsonl
from megadata._rust import debug_read_samples

# 最小 PNG：8 字节魔数 + 少量载荷，够 `sniff_image_codec` 识别（缺省 media_type 时）
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_1K = PNG_MAGIC + b"\x00" * 1024
PNG_256K = PNG_MAGIC + b"\x00" * (256 * 1024)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _assert_ids_contiguous(summary: dict) -> None:
    """全链不重复不丢失：id 序列恰为 0..allocated，且 sink 收到的样本数一致。"""
    n = summary["allocated"]
    assert summary["id_check"] == "ok", summary["id_check"]
    assert summary["sample_ids"] == list(range(n))
    assert summary["samples_seen"] == n


def test_text_and_three_reference_forms_and_id_continuity(tmp_path):
    """纯文本 + 三种引用形态 + data: URI：全部正常读取，id 连续，payload 字节如实计入。"""
    img_file = tmp_path / "img.png"
    img_file.write_bytes(PNG_1K)
    records = [
        {"text": "纯文本样本，无图像引用"},
        # b64 内联：不传 image_media_type → 魔数嗅探
        {"text": "b64 引用", "image_b64": _b64(PNG_1K)},
        # data URI 内联：media_type 从 URI 取
        {"text": "data uri", "image_url": f"data:image/png;base64,{_b64(PNG_1K)}"},
        # 本地路径引用
        {"text": "path 引用", "image_path": str(img_file)},
        # URL 引用 + download_url_media=False：仅记录引用，不下载、不报错（payload=0）
        {"text": "url 引用", "image_url": "http://127.0.0.1:1/unreachable.png"},
    ]
    path = write_jsonl(str(tmp_path / "forms.jsonl"), records)

    s = debug_read_samples([path], {"download_url_media": False})

    assert not s["aborted"], s["abort_msg"]
    assert s["skipped_by_class"] == {}
    assert s["fail_fast_triggered"] == 0
    _assert_ids_contiguous(s)
    assert s["docs_read"] == 5 and s["bytes_read"] > 0
    # 默认 ChunkLimit（4096 条 / 64MiB）下 5 条小样本必为单 chunk
    assert s["chunk_count"] == 1 and s["chunk_sizes"] == [5]
    # payload 字节：b64 + data uri + path 各 1KiB；文本与 url 引用为 0
    assert s["chunk_byte_sizes"] == [3 * len(PNG_1K)]


def test_skip_and_count_classifies_and_keeps_ids(tmp_path):
    """SkipAndCount：坏 path → source_not_found、坏 b64 → source_corrupt；占位样本占 id 不丢号。"""
    records = [
        {"text": "ok-0"},
        {"text": "bad path", "image_path": str(tmp_path / "missing.png")},
        {"text": "ok-2"},
        {"text": "bad b64", "image_b64": "!!!not-base64!!!"},
        {"text": "ok-4"},
    ]
    path = write_jsonl(str(tmp_path / "mixed.jsonl"), records)

    s = debug_read_samples([path])

    assert not s["aborted"], s["abort_msg"]
    assert s["skipped_by_class"] == {"source_not_found": 1, "source_corrupt": 1}
    assert s["fail_fast_triggered"] == 0
    # 被跳过的样本同样占 id（行号 ↔ id 稳定对应），全链无漏无重
    _assert_ids_contiguous(s)
    assert s["docs_read"] == 5


def test_fail_fast_aborts_and_reports(tmp_path):
    """FailFast：违规样本处显式中止，已推送前缀连续，samples_seen < allocated。"""
    records = [
        {"text": "ok-0"},
        {"text": "boom", "image_path": str(tmp_path / "missing.png")},
        {"text": "never-reached"},
    ]
    path = write_jsonl(str(tmp_path / "ff.jsonl"), records)

    s = debug_read_samples([path], {"source_not_found": "fail"})

    assert s["aborted"] is True
    assert s["abort_msg"] is not None
    assert "FailFast" in s["abort_msg"] and "source_not_found" in s["abort_msg"]
    assert s["fail_fast_triggered"] == 1
    assert s["allocated"] == 2  # 中止处（含违规行自身）已分配；其后样本不再读取
    assert s["samples_seen"] < s["allocated"]  # 前缀推送，尾部未推送
    assert s["id_check"] == "ok"
    assert s["sample_ids"] == list(range(s["samples_seen"]))


def test_chunk_byte_budget_splits_and_respects_budget(tmp_path):
    """max_bytes 字节预算：多 chunk 切分，每 chunk payload 和不超预算，payload 守恒。

    预算语义 = 行长 + 内联解码字节（byte_cost_of_line）；本用例单样本
    cost（≈612KB）< 预算（2MiB），故每 chunk payload 和（4×256KiB=1MiB）≤ 预算。
    """
    budget = 2 * 1024 * 1024
    records = [
        {"text": f"sample-{i}", "image_b64": _b64(PNG_256K)} for i in range(12)
    ]
    path = write_jsonl(str(tmp_path / "big.jsonl"), records)

    s = debug_read_samples([path], {"max_bytes": budget})

    assert not s["aborted"], s["abort_msg"]
    assert s["chunk_count"] > 1
    assert all(b <= budget for b in s["chunk_byte_sizes"]), s["chunk_byte_sizes"]
    assert sum(s["chunk_sizes"]) == s["allocated"] == 12
    assert sum(s["chunk_byte_sizes"]) == 12 * len(PNG_256K)
    _assert_ids_contiguous(s)


def test_chunk_max_samples_limit(tmp_path):
    """max_samples 条数上限：按条数切 chunk，尾部允许非满批。"""
    records = [{"text": f"s{i}"} for i in range(10)]
    path = write_jsonl(str(tmp_path / "many.jsonl"), records)

    s = debug_read_samples([path], {"max_samples": 3})

    assert not s["aborted"], s["abort_msg"]
    assert s["chunk_sizes"] == [3, 3, 3, 1]
    assert s["chunk_count"] == 4
    _assert_ids_contiguous(s)


def test_multiple_input_files_share_id_allocator(tmp_path):
    """多输入文件串行读取：sample_id 跨文件连续（共享分配器），无漏无重。"""
    p1 = write_jsonl(str(tmp_path / "a.jsonl"), [{"text": "a0"}, {"text": "a1"}])
    p2 = write_jsonl(
        str(tmp_path / "b.jsonl"),
        [{"text": "b0", "image_b64": _b64(PNG_1K)}, {"text": "b1"}, {"text": "b2"}],
    )

    s = debug_read_samples([p1, p2])

    assert not s["aborted"], s["abort_msg"]
    assert s["allocated"] == 5 and s["docs_read"] == 5
    _assert_ids_contiguous(s)


def test_blank_lines_and_empty_input(tmp_path):
    """空行不占 id；空文件产出全零统计、零 chunk（不报错）。"""
    body = '{"text":"x"}\n\n   \n{"text":"y"}\n'
    p_nonblank = tmp_path / "blank.jsonl"
    p_nonblank.write_text(body, encoding="utf-8")
    p_empty = write_jsonl(str(tmp_path / "empty.jsonl"), [])

    s1 = debug_read_samples([str(p_nonblank)])
    assert s1["allocated"] == 2 and s1["docs_read"] == 2  # 空行不计 docs、不占 id
    _assert_ids_contiguous(s1)

    s2 = debug_read_samples([p_empty])
    assert s2["allocated"] == 0 and s2["docs_read"] == 0
    assert s2["chunk_count"] == 0 and s2["sample_ids"] == []
    assert s2["id_check"] == "ok" and not s2["aborted"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

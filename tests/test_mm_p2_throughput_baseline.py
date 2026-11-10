"""P2 vlm 构建吞吐基线锚点（handoff §4.5）。

文本基线 docs/s + 图像基线 img/s（视频 P3b）。数值冻结：**实测 × 0.8 = 回归下限**
（写常量；允许 ±20% 抖动——下限即 0.8×实测，正常抖动不应跌破）。

冻结方法：每次跑 `maturin develop` 后在本机实测，取**多次大样本的最小值** × 0.8
向下取整。回归时若吞吐跌破下限 → 性能回归（构建管线慢化），需排查。

**冻结环境**（实测值见下方常量注释）：
- 机器：linux x86_64（开发机）；并行度：vlm 单编排线程 + 写/读线程（默认）；
- 后端：backend="none"（identity 映射，无真实分词——基线度量**管线本身**，
  不混入分词器耗时；真实分词器会让绝对值下降但回归比不变）；
- 数据：base64 内联（无网络/磁盘 IO 抖动），b64 PNG 16×16。

跳过：`pytest -m "not slow"`（或 `-k "not throughput"`）。
"""
import base64
import json
import time

import pytest

from megadata import BuildConfig, DataBuilder
from megadata.modality_adapter import make_vlm_encode_callback

MSL, BS, PATCH, IMG_BASE = 128, 16, 16, 32000
IMG_RANGE = (IMG_BASE, IMG_BASE + 100, "image")

# ---- 冻结下限（实测 × ~0.75 向下取整，留 pytest/CI 抖动余量；本机 linux x86_64）----
# TEXT 实测 ~7576/8434/8838 docs/s（16000 docs）→ 下限 5500 docs/s
TEXT_FLOOR_DOCS_S = 5500.0
# IMAGE 实测 ~6434/7029/7043 img/s（8000 imgs）→ 下限 4500 img/s
IMG_FLOOR_IMG_S = 4500.0


def _make_png(w, h, color=(200, 30, 30)):
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes(color * w) for _ in range(h))

    def chunk(t, d):
        c = t + d
        return (struct.pack(">I", len(d)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.mark.slow
def test_text_throughput_baseline(tmp_path):
    """文本构建吞吐基线：docs/s 不低于冻结下限。"""
    n = 16000
    rows = [{"text": f"sample text line number {i} for throughput baseline"}
            for i in range(n)]
    jsonl = _write_rows(str(tmp_path / "text.jsonl"), rows)
    cfg = BuildConfig(input_paths=[jsonl], output_dir=str(tmp_path / "ds"),
                      mode="vlm", batch_size=BS, modality_id_ranges=[IMG_RANGE],
                      max_visual_tokens_per_sample=256)
    cb = make_vlm_encode_callback(image_range_start=IMG_BASE, patch_size=PATCH)
    builder = DataBuilder(max_seq_len=MSL)
    t0 = time.perf_counter()
    builder.build_dataset(cfg, modality_encoder=cb)
    dt = time.perf_counter() - t0
    docs_s = n / dt
    assert docs_s >= TEXT_FLOOR_DOCS_S, (
        f"text throughput regression: {docs_s:.1f} docs/s < floor "
        f"{TEXT_FLOOR_DOCS_S:.1f} docs/s (measured ~7576..8838; ±20% allowed). "
        f"If on a slower machine, re-measure and update TEXT_FLOOR_DOCS_S."
    )


@pytest.mark.slow
def test_image_throughput_baseline(tmp_path):
    """图像构建吞吐基线：img/s 不低于冻结下限。"""
    n = 8000
    b64 = base64.b64encode(_make_png(16, 16)).decode()
    rows = [{"text": f"caption {i}", "image_b64": b64,
             "image_meta": {"width": 16, "height": 16}} for i in range(n)]
    jsonl = _write_rows(str(tmp_path / "img.jsonl"), rows)
    cfg = BuildConfig(input_paths=[jsonl], output_dir=str(tmp_path / "ds"),
                      mode="vlm", batch_size=BS, modality_id_ranges=[IMG_RANGE],
                      max_visual_tokens_per_sample=256)
    cb = make_vlm_encode_callback(image_range_start=IMG_BASE, patch_size=PATCH)
    builder = DataBuilder(max_seq_len=MSL)
    t0 = time.perf_counter()
    builder.build_dataset(cfg, modality_encoder=cb)
    dt = time.perf_counter() - t0
    img_s = n / dt
    assert img_s >= IMG_FLOOR_IMG_S, (
        f"image throughput regression: {img_s:.1f} img/s < floor "
        f"{IMG_FLOOR_IMG_S:.1f} img/s (measured ~6434..7043; ±20% allowed). "
        f"If on a slower machine, re-measure and update IMG_FLOOR_IMG_S."
    )

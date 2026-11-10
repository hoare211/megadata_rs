"""P3b video e2e（`tests/test_mm_p3b_video.py`，P3b 方案 §11 锚点）。

**不依赖分词器 / 网络 / PIL / 真实视频解码器**：参考回调 + 合成帧 stub
（``frame_fn`` 确定性生成），视频字节用伪 mp4 载荷内联。

覆盖锚点：
- A4 确定性：同参数两次构建，``col_vid_rec``（含 start_ms）+ ``col_vid_blob``
  全文件逐字节相同。
- A5 定宽性：Vlm 流行满足 ``bs × msl`` 恒定。
- A8 blob 生命周期：videos 条目**不带 bytes**（blob_ref 按需读）；经
  ``read_video_blob`` 读回的字节 == 写入字节；video blob 不进预取窗口
  （小 prefetch_bytes_limit 下构建/读取不卡）。
- 守恒：``Σ mm_cu.vid_cnt == col_vid_rec 记录数``；
  ``Σ vid_rec.len(VALID) == col_vid_blob 长度``；混合 image+video 样本
  ``total_video_tokens/clips`` 非 0 且 ``total_tokens == Σ 模态``。
- INV-2 负例：video 段缺 timestamp → fail-fast 含 sid。
- 采样边界：``video_n_frames > n_frames_total`` → fail-fast（不静默重复帧）。
- 按需读上界：``video_ondemand_bytes_limit`` 小于单 clip → WARN + 放行
  （活性保证）。
- 兼容：无 video 的数据集不产 col_vid_* 文件、不写 vid schema key
  （图像路径产物逐字节不变的回归面）。
"""
import base64
import hashlib
import json
import os
import struct

import pytest
import torch

from megadata import BuildConfig, DataBuilder
from megadata.engine import RustStreamReader, read_video_blob, _safe_from_dlpack
from megadata.modality_adapter import make_vlm_encode_callback

IMG_BASE = 32000
IMG_RANGE = (IMG_BASE, IMG_BASE + 100, "image")
VID_BASE = 34000
VID_RANGE = (VID_BASE, VID_BASE + 4096, "video")

# 采样/网格参数（与 align_fingerprint.video 对应）
VFRAMES = 16           # video_n_frames
VRESIZE = (64, 64)     # video_resize（patch=16 → 4×4=16 token/帧）
PATCH = 16
TOKENS_PER_CLIP = (VRESIZE[0] // PATCH) * (VRESIZE[1] // PATCH) * VFRAMES  # 256


def make_video_payload(n, seed=0x00):
    """伪 mp4 载荷（魔数 + 确定性填充；回调只透传，不解码）。"""
    return b"\x00\x00\x00\x18ftypmp42" + bytes((i * 3 + seed) % 256 for i in range(max(0, n - 12)))


def frame_fn(payload, indices):
    """合成帧 stub（P3b §6.3：参考实现不绑定解码器；确定性、无 PIL）。"""
    return [bytes(((i * 7 + 11) % 256) for _ in range(64)) for i in indices]


def video_row(*, ts, dur, fps=30.0, n_frames_total=900, text=None, payload_len=512):
    """video-only JSONL 行（video_b64 内联 + timestamp + video_meta）。"""
    row = {
        "video_b64": base64.b64encode(make_video_payload(payload_len)).decode(),
        "video_media_type": "video/mp4",
        "video_meta": {"fps": fps, "n_frames_total": n_frames_total, "duration_sec": dur},
    }
    if ts is not None:
        row["timestamp"] = ts
    if text:
        row["text"] = text
    return row


def image_row(*, w=16, h=16, text=None):
    row = {"image_b64": base64.b64encode(bytes((i * 13) % 256 for i in range(64))).decode(),
           "image_meta": {"width": w, "height": h}}
    if text:
        row["text"] = text
    return row


def mixed_row(*, ts, dur, text=None):
    """图+视同行（交错 [Image, Video]；text 挂样本级 caption）。"""
    row = {
        "image_b64": base64.b64encode(bytes((i * 5) % 256 for i in range(48))).decode(),
        "image_meta": {"width": 16, "height": 16},
        "video_b64": base64.b64encode(make_video_payload(300)).decode(),
        "video_media_type": "video/mp4",
        "video_meta": {"fps": 30.0, "n_frames_total": 900, "duration_sec": dur},
        "timestamp": ts,
    }
    if text:
        row["text"] = text
    return row


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def video_cfg(jsonl, out_dir, *, bs=2, msl=512, msl_visual=None, **kw):
    return BuildConfig(
        input_paths=[jsonl], output_dir=str(out_dir), mode="vlm",
        batch_size=bs, modality_id_ranges=[IMG_RANGE, VID_RANGE],
        max_visual_tokens_per_sample=msl_visual,
        video_n_frames=VFRAMES, video_resize=VRESIZE,
        **kw,
    )


def build_video(cfg, msl, cb=None):
    if cb is None:
        cb = make_vlm_encode_callback(
            image_range_start=IMG_BASE, video_range_start=VID_BASE,
            video_n_frames=VFRAMES, video_resize=VRESIZE, frame_fn=frame_fn,
        )
    return DataBuilder(max_seq_len=msl).build_dataset(cfg, modality_encoder=cb)


def manifest_of(out_dir):
    with open(os.path.join(str(out_dir), "index.json")) as f:
        return json.load(f)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vid_files(out_dir, manifest):
    """每个 Vlm shard 的 col_vid_* 解析结果（rec 40B / mm_cu 16B / blob）。"""
    out = []
    for sm in manifest["shards"]:
        stem = sm["filename"][: -len(".bin")]
        d = str(out_dir)
        rec_data = read_bytes(f"{d}/{stem}.col_vid_rec")
        recs = []
        for i in range(len(rec_data) // 40):
            # 40B：offset u64, len u32, n_frames u16, w u16, h u16, flags u8,
            # rsv u8, sample_id u64, seg_order u32, fps_milli u32, start_ms u32
            off, ln, nf, w, h, flags, rsv, sid, seg, fps, start = struct.unpack_from(
                "<QIHHHBBQIII", rec_data, i * 40)
            recs.append({"offset": off, "len": ln, "n_frames": nf, "w": w, "h": h,
                         "flags": flags, "sample_id": sid, "seg_order": seg,
                         "fps_milli": fps, "start_ms": start})
        mm_cu = read_bytes(f"{d}/{stem}.mm_cu")
        counts = [struct.unpack_from("<IIII", mm_cu, i * 16) for i in range(len(mm_cu) // 16)]
        out.append({"stem": stem, "meta": sm, "recs": recs, "cu": counts,
                    "blob": read_bytes(f"{d}/{stem}.col_vid_blob")})
    return out


# ============================================================
# A4：确定性（两次构建逐字节相同）
# ============================================================
def test_a4_determinism_byte_identical(tmp_path):
    rows = [video_row(ts=1.5, dur=4.0), mixed_row(ts=0.25, dur=2.0, text="m"),
            video_row(ts=0.0, dur=1.0)]
    jsonl = write_jsonl(tmp_path / "v.jsonl", rows)
    outs = []
    for i in range(2):
        out = tmp_path / f"out_a4_{i}"
        build_video(video_cfg(jsonl, out), msl=512)
        outs.append(out)
    ma, mb = manifest_of(outs[0]), manifest_of(outs[1])
    assert [s["hashes"] for s in ma["shards"]] == [s["hashes"] for s in mb["shards"]], \
        "同参数两次构建 Vlm shard 哈希必须一致（含 col_vid_*）"
    for sa, sb in zip(vid_files(outs[0], ma), vid_files(outs[1], mb)):
        assert sa["recs"] == sb["recs"], "col_vid_rec 内容（含 start_ms）逐字节相同"
        assert sa["blob"] == sb["blob"], "col_vid_blob 逐字节相同"
    # align_fingerprint.video 落盘且参数正确
    fp = ma["align_fingerprint"]["video"]
    assert fp == {"n_frames": VFRAMES, "fps": 30.0, "resize": list(VRESIZE), "seed": 0}


# ============================================================
# A5 + 守恒 + A8：video-only 数据集
# ============================================================
def test_a5_conservation_and_ondemand_read(tmp_path):
    rows = [video_row(ts=1.5, dur=4.0), video_row(ts=0.0, dur=2.0)]
    jsonl = write_jsonl(tmp_path / "v.jsonl", rows)
    out = tmp_path / "out_a5"
    build_video(video_cfg(jsonl, out, bs=2, msl=512, msl_visual=512), msl=512)
    man = manifest_of(out)
    gs = man["global_stats"]

    # ---- 守恒 ----
    shards = vid_files(out, man)
    recs = [r for s in shards for r in s["recs"]]
    assert recs, "video stream must produce recs"
    assert sum(c[3] for s in shards for c in s["cu"]) == len(recs), "Σ mm_cu.vid_cnt == rec 数"
    assert sum(r["len"] for r in recs if r["flags"] == 0) == sum(len(s["blob"]) for s in shards)
    for s in shards:
        for c in s["cu"]:
            assert c[0] == 3 and c[2] == 0, "n_mod=3 / aud_cnt=0"
    assert gs["total_video_clips"] == sum(1 for r in recs if r["flags"] == 0)
    assert gs["total_video_tokens"] == TOKENS_PER_CLIP * 2, "每 clip 固定 token 数（D7 定宽）"
    assert gs["total_video_seconds"] == pytest.approx(6.0), "源时长口径（Σ duration_sec）"
    assert gs["total_tokens"] == gs["total_video_tokens"] + gs.get("total_text_tokens", 0) \
        + gs.get("total_image_tokens", 0), "total_tokens == Σ 模态"
    assert man["schema"]["vid_blob"] == "bytes" and man["schema"]["vid_rec"] == "vidrec40"
    for s in shards:
        assert s["meta"]["hashes"]["sha256:col_vid_blob"] == sha256_file(
            os.path.join(str(out), f"{s['stem']}.col_vid_blob"))
        assert s["meta"]["hashes"]["sha256:col_vid_rec"] == sha256_file(
            os.path.join(str(out), f"{s['stem']}.col_vid_rec"))

    # ---- A5 定宽性 + A8 按需读 ----
    reader = RustStreamReader(os.path.join(str(out), "index.json"), 2, 512, 1, 0, 42)
    n_batches = 0
    while True:
        d = reader.next_batch_capsules()
        if d is None:
            break
        n_batches += 1
        ids = _safe_from_dlpack(d["input_ids"])
        assert ids.shape == (2, 512), f"定宽行 bs×msl 恒定: {ids.shape}"
        videos = d.get("videos")
        assert videos, "含 video 的 batch 必须透出 videos"
        for e in videos:
            assert "bytes" not in e and "blob_ref" in e, \
                "videos 条目必须字节延迟（blob_ref 按需读，不进预取窗口）"
            # start_ms 由 encoder 计算（round(timestamp×1000)）
            assert e["start_ms"] in (0, 1500)
            blob = read_video_blob(e)
            assert isinstance(blob, torch.Tensor) and blob.dtype == torch.uint8
            # 读回字节 == 该样本的伪 mp4 载荷
            expect = make_video_payload(512)
            assert blob.numpy().tobytes() == expect, "blob 字节必须与写入一致"
    assert n_batches >= 1


# ============================================================
# A8：video blob 不进预取窗口（小 prefetch 预算下不卡）
# ============================================================
def test_a8_video_blob_not_in_prefetch_window(tmp_path):
    # 大 video blob（~256KB）+ 极小 prefetch_bytes_limit：若 video blob 计入
    # 预取预算，单 batch 账单超限会走 WARN 放行（仍能出）——这里断言读取全程
    # 无致命错误且 videos 条目可用（结构上证明 blob 未进预取路径）。
    rows = [video_row(ts=0.0, dur=2.0, payload_len=256 * 1024)]
    jsonl = write_jsonl(tmp_path / "v.jsonl", rows)
    out = tmp_path / "out_a8"
    build_video(video_cfg(jsonl, out, bs=1, msl=512, msl_visual=512), msl=512)
    reader = RustStreamReader(os.path.join(str(out), "index.json"), 1, 512, 1, 0, 42,
                              prefetch_bytes_limit=64 * 1024)
    d = reader.next_batch_capsules()
    assert d is not None
    videos = d["videos"]
    assert len(videos) == 1
    blob = read_video_blob(videos[0])
    assert blob.numel() == 256 * 1024


# ============================================================
# 混合样本：image + video + text 各自守恒
# ============================================================
def test_mixed_image_video_text_two_modalities(tmp_path):
    rows = [
        {"text": "plain text row"},              # sid 0：纯文本
        image_row(text="an image"),               # sid 1：单图
        video_row(ts=0.5, dur=2.0),               # sid 2：纯视频
        mixed_row(ts=0.25, dur=1.0, text="mix"),  # sid 3：图+视
    ]
    jsonl = write_jsonl(tmp_path / "mix.jsonl", rows)
    out = tmp_path / "out_mix"
    build_video(video_cfg(jsonl, out, bs=2, msl=512, msl_visual=512), msl=512)
    man = manifest_of(out)
    gs = man["global_stats"]
    # 模态统计分流：video 不计 image；image 存在；text 存在
    assert gs["total_video_clips"] == 2, "sid 2 + sid 3 各 1 clip"
    assert gs["total_video_tokens"] == TOKENS_PER_CLIP * 2
    assert gs["total_images"] >= 2, "sid 1 + sid 3 各 1 图"
    assert gs["total_image_tokens"] > 0
    assert gs["total_text_tokens"] > 0
    assert gs["total_tokens"] == (gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)
                                  + gs.get("total_video_tokens", 0))
    # rec 的 sample_id 覆盖两个 video 样本
    recs = [r for s in vid_files(out, man) for r in s["recs"]]
    assert {r["sample_id"] for r in recs} == {2, 3}
    # 混合样本的 video rec：start_ms = round(0.25×1000) = 250
    sid3 = [r for r in recs if r["sample_id"] == 3]
    assert sid3 and sid3[0]["start_ms"] == 250
    assert sid3[0]["n_frames"] == VFRAMES and sid3[0]["fps_milli"] == 30000


# ============================================================
# INV-2 负例：timestamp 缺失 → fail-fast 含 sid
# ============================================================
def test_negative_missing_timestamp_fail_fast(tmp_path):
    rows = [video_row(ts=None, dur=2.0)]
    jsonl = write_jsonl(tmp_path / "bad.jsonl", rows)
    out = tmp_path / "out_bad"
    with pytest.raises(Exception, match="no timestamp"):
        build_video(video_cfg(jsonl, out), msl=512)


# ============================================================
# 采样边界：video_n_frames > n_frames_total → fail-fast（不静默重复帧）
# ============================================================
def test_negative_n_frames_exceeds_total_fail_fast(tmp_path):
    rows = [video_row(ts=0.0, dur=2.0, n_frames_total=8)]
    jsonl = write_jsonl(tmp_path / "few.jsonl", rows)
    out = tmp_path / "out_few"
    with pytest.raises(Exception, match="exceeds total frames"):
        build_video(video_cfg(jsonl, out), msl=512)


# ============================================================
# fps 缺失 → WARN + 兜底 fps + 计数入 manifest
# ============================================================
def test_fps_default_counted_in_manifest(tmp_path):
    row = video_row(ts=0.0, dur=2.0)
    del row["video_meta"]["fps"]
    jsonl = write_jsonl(tmp_path / "nofps.jsonl", [row])
    out = tmp_path / "out_nofps"
    build_video(video_cfg(jsonl, out), msl=512)
    man = manifest_of(out)
    assert man["global_stats"]["video_fps_defaulted_clips"] == 1
    recs = [r for s in vid_files(out, man) for r in s["recs"]]
    assert recs[0]["fps_milli"] == 30000, "缺 fps → 兜底 30.0 → fps_milli=30000"
    assert man["align_fingerprint"]["video"]["fps"] == 30.0


# ============================================================
# 按需读上界：单 clip 超 video_ondemand_bytes_limit → WARN + 放行
# ============================================================
def test_ondemand_limit_single_clip_warn_and_proceed(tmp_path):
    rows = [video_row(ts=0.0, dur=2.0, payload_len=64 * 1024)]
    jsonl = write_jsonl(tmp_path / "v.jsonl", rows)
    out = tmp_path / "out_lim"
    build_video(video_cfg(jsonl, out, bs=1, msl=512, msl_visual=512), msl=512)
    # 单 clip 64KB > limit 16KB → WARN 一次 + 放行（活性保证）
    reader = RustStreamReader(os.path.join(str(out), "index.json"), 1, 512, 1, 0, 42,
                              video_ondemand_bytes_limit=16 * 1024)
    d = reader.next_batch_capsules()
    assert d is not None
    blob = read_video_blob(d["videos"][0])
    assert blob.numel() == 64 * 1024


# ============================================================
# 兼容：无 video 的数据集不产 col_vid_* 文件、不写 vid schema key
# ============================================================
def test_image_only_dataset_has_no_video_sidecar(tmp_path):
    rows = [{"text": "t0"}, image_row(text="c"), {"text": "t1"}]
    jsonl = write_jsonl(tmp_path / "img.jsonl", rows)
    out = tmp_path / "out_img"
    build_video(video_cfg(jsonl, out), msl=256)
    man = manifest_of(out)
    assert "vid_blob" not in man["schema"] and "vid_rec" not in man["schema"]
    for sm in man["shards"]:
        stem = sm["filename"][: -len(".bin")]
        assert not os.path.exists(os.path.join(str(out), f"{stem}.col_vid_blob"))
        assert not os.path.exists(os.path.join(str(out), f"{stem}.col_vid_rec"))
    assert man["global_stats"].get("total_video_clips", 0) == 0
    # 读回：无 videos 键（旧 trainer 零影响）
    reader = RustStreamReader(os.path.join(str(out), "index.json"), 2, 256, 1, 0, 42)
    d = reader.next_batch_capsules()
    assert d is not None
    assert "videos" not in d


# ============================================================
# P1-2（20261031 评审）：多 shard 预取后读旧 batch 的 video blob
# ============================================================
def test_p1_2_multishard_delayed_video_blob_read(tmp_path):
    """预取跨 shard 后，已交付 batch 的 blob 句柄不得被提前注销。

    旧实现切 shard 立即 `unregister(store_id)`：`batches_per_shard=1` + 6 行
    ⇒ 6 个 video shard，取走第一个 batch、等预取线程切走后再 `read_video_blob`
    得到 `NotFound`（训练在正常时序下中断）。修复后 batch 持有 RAII pin，
    切 shard 只标记 retired，最后一个 pin 归还才关闭句柄。
    """
    import time

    rows = [video_row(ts=0.0, dur=2.0) for _ in range(6)]
    jsonl = write_jsonl(tmp_path / "p12.jsonl", rows)
    out = tmp_path / "out_p12"
    build_video(
        video_cfg(jsonl, out, bs=1, msl=512, msl_visual=512, batches_per_shard=1),
        msl=512,
    )
    man = manifest_of(out)
    assert len(man["shards"]) >= 2, "每个 shard 1 batch ⇒ 多 shard 数据集"

    reader = RustStreamReader(os.path.join(str(out), "index.json"), 1, 512, 1, 0, 42,
                              prefetch_bytes_limit=0)
    first = reader.next_batch_capsules()
    assert first is not None and first["videos"], "首个 batch 必须带 video 条目"
    # 让预取线程越过若干 shard（旧实现在此期间注销旧 store_id）
    time.sleep(0.5)
    blob = read_video_blob(first["videos"][0])
    assert blob.numpy().tobytes() == make_video_payload(512), (
        "P1-2: 已交付 batch 的 video blob 在预取跨 shard 后必须仍可读")

"""P3c 对齐落地 e2e（`tests/test_mm_p3c_align.py`，P3c 方案 §9 锚点）。

**不依赖分词器 / 网络 / PIL / 真实视频解码器**：参考回调 + 合成帧 stub，
音视频字节用伪载荷内联；text+AV+timestamp 行 → Text 段打头的交错样本
（三模态对齐轴输入，source P3c 扩展）。

覆盖锚点：
- A2 可追溯链：`av_align` 条目 `sid → {video, audio, text} → window` 全链可还原；
  text 侧经 ts token 定位 window。
- A3 window 计算：`window_index == floor(start_ms / W_ms)` 单条 rec 自足可算
  （含非整倍起点 2500ms → w1）。
- A6 兼容：① 不请求 align 的 batch dict 无 `av_align` 键；② 不启用对齐的
  构建产物无 `col_row_sids` 文件、无 `row_sids` schema key（与 P2 期一致）。
- A7 负面：align=True 但数据集未启用对齐（纯图像）→ 构造期 fail-fast。
- 守恒：所有有 start_ms 的 rec 出现在某条目中（无遗漏）。
- 确定：同参数两次构建 av_align 内容一致。
"""
import base64
import json
import os

import pytest

from megadata import BuildConfig, DataBuilder
from megadata.engine import RustStreamReader, _safe_from_dlpack
from megadata.modality_adapter import make_vlm_encode_callback

IMG_BASE = 32000
IMG_RANGE = (IMG_BASE, IMG_BASE + 100, "image")
AUD_BASE = 33000
AUD_RANGE = (AUD_BASE, AUD_BASE + 1000, "audio")
VID_BASE = 35000
VID_RANGE = (VID_BASE, VID_BASE + 4096, "video")

RATE = 50
VFRAMES = 16
VRESIZE = (64, 64)
TOKENS_PER_CLIP = (VRESIZE[0] // 16) * (VRESIZE[1] // 16) * VFRAMES  # 256


def make_audio_payload(n, seed=0x66):
    return b"fLaC" + bytes((i * 7 + seed) % 256 for i in range(max(0, n - 4)))


def make_video_payload(n, seed=0x00):
    return b"\x00\x00\x00\x18ftypmp42" + bytes((i * 3 + seed) % 256 for i in range(max(0, n - 12)))


def frame_fn(payload, indices):
    return [bytes(((i * 7 + 11) % 256) for _ in range(64)) for i in indices]


def audio_row(*, ts, dur, text=None, payload_len=256):
    """audio 行：text+timestamp → [Text(0,ts), Audio(1,ts)]（P3c 输入形态）。"""
    row = {
        "audio_b64": base64.b64encode(make_audio_payload(payload_len)).decode(),
        "audio_media_type": "audio/flac",
        "audio_meta": {"sample_rate": 16000, "channels": 1, "duration_sec": dur},
        "timestamp": ts,
    }
    if text:
        row["text"] = text
    return row


def video_row(*, ts, dur, text=None, payload_len=512):
    row = {
        "video_b64": base64.b64encode(make_video_payload(payload_len)).decode(),
        "video_media_type": "video/mp4",
        "video_meta": {"fps": 30.0, "n_frames_total": 900, "duration_sec": dur},
        "timestamp": ts,
    }
    if text:
        row["text"] = text
    return row


def image_row(*, w=16, h=16, text=None):
    row = {"image_b64": base64.b64encode(bytes((i * 13) % 256 for i in range(64))).decode(),
           "image_meta": {"width": w, "height": h}}
    if text:
        row["text"] = text
    return row


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def align_cfg(jsonl, out_dir, *, bs=2, msl=512, msl_visual=None, **kw):
    return BuildConfig(
        input_paths=[jsonl], output_dir=str(out_dir), mode="vlm",
        batch_size=bs, modality_id_ranges=[IMG_RANGE, AUD_RANGE, VID_RANGE],
        max_visual_tokens_per_sample=msl_visual,
        w_ms=2000, audio_window_token_rate=RATE,
        video_n_frames=VFRAMES, video_resize=VRESIZE,
        **kw,
    )


def build_align(cfg, msl, cb=None):
    if cb is None:
        cb = make_vlm_encode_callback(
            image_range_start=IMG_BASE, audio_range_start=AUD_BASE, audio_token_rate=RATE,
            video_range_start=VID_BASE, video_n_frames=VFRAMES, video_resize=VRESIZE,
            frame_fn=frame_fn,
        )
    return DataBuilder(max_seq_len=msl).build_dataset(cfg, modality_encoder=cb)


def manifest_of(out_dir):
    with open(os.path.join(str(out_dir), "index.json")) as f:
        return json.load(f)


def iter_align_batches(out_dir, bs, msl, **kw):
    reader = RustStreamReader(os.path.join(str(out_dir), "index.json"), bs, msl, 1, 0, 42, **kw)
    while True:
        d = reader.next_batch_capsules()
        if d is None:
            return
        yield d


# ============================================================
# A2/A3：三模态样本 → av_align 全链可还原
# ============================================================
def test_av_align_three_modalities(tmp_path):
    rows = [
        # sid 0：text(ts=0.5s→w0) + audio(ts=0.5s→w0)
        audio_row(ts=0.5, dur=2.0, text="zero five"),
        # sid 1：text(ts=2.5s→w1) + video(ts=2.5s→w1)
        video_row(ts=2.5, dur=2.0, text="two five"),
        # sid 2：纯文本（无 A/V 轨 → 不进对齐视图）
        {"text": "plain text"},
        # sid 3：text(ts=1.0s→w0) + image + video(ts=1.0s→w0)
        {
            "text": "mixed", "timestamp": 1.0,
            "image_b64": base64.b64encode(bytes((i * 5) % 256 for i in range(48))).decode(),
            "image_meta": {"width": 16, "height": 16},
            "video_b64": base64.b64encode(make_video_payload(300)).decode(),
            "video_media_type": "video/mp4",
            "video_meta": {"fps": 30.0, "n_frames_total": 900, "duration_sec": 1.0},
        },
    ]
    jsonl = write_jsonl(tmp_path / "mix.jsonl", rows)
    out = tmp_path / "out_align"
    build_align(align_cfg(jsonl, out, bs=2, msl=512, msl_visual=512), msl=512)
    man = manifest_of(out)

    # 指纹全集（DP5）
    fp = man["align_fingerprint"]
    assert fp["W_ms"] == 2000 and fp["granularity"] == "G2"
    assert fp["text_in_axis"] is True
    assert fp["ts_token"]["base_id"] >= 35000 + 4096, "ts 区间在所有模态区间之后"
    assert fp["ts_token"]["range_size"] == 2048
    assert fp["ts_token"]["encoding"] == "contiguous"
    assert fp["audio"]["window_token_rate"] == RATE
    assert fp["video"]["n_frames"] == VFRAMES
    assert any(r["kind"] == "audio" for r in fp["modality_id_ranges"])
    # row_sids 列落盘 + schema key
    assert man["schema"]["row_sids"] == "list<uint64>"
    for sm in man["shards"]:
        assert "sha256:col_row_sids" in sm["hashes"]
        assert os.path.exists(os.path.join(
            str(out), sm["filename"][: -len(".bin")] + ".col_row_sids"))
    # audio 流存在 ⇒ align_aud_range 落盘
    assert man["audio_shards"], "audio 流必须有 shard"
    for sm in man["shards"]:
        assert sm.get("align_aud_range") is not None, "Vlm shard 必须带音频范围索引"

    # ---- 读回 av_align ----
    entries = []
    for d in iter_align_batches(out, 2, 512, align=True):
        got = d.get("av_align")
        assert got, "align=True 必须透出 av_align"
        # P1-1 回归：text_token_range 内不得含 image/audio/video placeholder
        ids = _safe_from_dlpack(d["input_ids"]).reshape(-1).tolist()
        for e in got:
            for (s, t) in e["text_token_range"]:
                span = ids[s:t]
                visual = [x for x in span
                          if IMG_RANGE[0] <= x < IMG_RANGE[1]
                          or AUD_RANGE[0] <= x < AUD_RANGE[1]
                          or VID_RANGE[0] <= x < VID_RANGE[1]]
                assert not visual, (
                    f"P1-1: text_token_range {(s, t)} 含视觉 placeholder {visual[:8]} "
                    f"(sample {e['sample_id']}, window {e['window_index']})")
        entries.extend(got)
    assert entries, "必须有对齐条目"

    by_key = {(e["sample_id"], e["window_index"]): e for e in entries}
    # sid 0：w0 三侧齐全（text 区间 + audio ref；无 video）
    e0 = by_key[(0, 0)]
    assert e0["video_rec_range"] == []
    assert len(e0["audio_rec_range"]) >= 1
    assert e0["text_token_range"], "text_token_range 为区间列表（空 = 无文本）"
    s, t = e0["text_token_range"][0]
    # sid 1：w1 video 侧齐全 + text 区间；audio 空
    e1 = by_key[(1, 1)]
    assert len(e1["video_rec_range"]) == 1
    assert e1["audio_rec_range"] == []
    assert e1["text_token_range"]
    # sid 2 纯文本：不进对齐视图（R4 硬规则——无 ts token）
    assert not any(e["sample_id"] == 2 for e in entries), "text-only 样本不得进 av_align"
    # sid 3：w0 video + text（图+视同行）
    e3 = by_key[(3, 0)]
    assert len(e3["video_rec_range"]) == 1
    assert e3["text_token_range"]
    # window 计算正确性：floor(start_ms / W_ms)——2500ms → w1（非整倍起点）
    assert (1, 1) in by_key and (0, 0) in by_key and (3, 0) in by_key


# ============================================================
# A6 兼容：不请求 align → 无 av_align 键；未启用对齐的构建无 row_sids
# ============================================================
def test_compat_no_align_key_and_no_row_sids(tmp_path):
    rows = [{"text": "t0"}, image_row(text="c"), {"text": "t1"}]
    jsonl = write_jsonl(tmp_path / "img.jsonl", rows)
    out = tmp_path / "out_img"
    build_align(align_cfg(jsonl, out), msl=256)  # 声明了 audio/video 区间但无内容
    man = manifest_of(out)
    # 声明了区间 ⇒ 指纹落盘（对齐启用），但无 audio 内容 ⇒ 无 audio_shards
    assert man["align_fingerprint"] is not None
    assert man.get("audio_shards", []) == []
    # 纯图像构建（不声明 AV 区间）⇒ 无指纹、无 row_sids
    rows2 = [{"text": "t0"}, image_row(text="c")]
    jsonl2 = write_jsonl(tmp_path / "img2.jsonl", rows2)
    out2 = tmp_path / "out_img2"
    cfg2 = BuildConfig(
        input_paths=[jsonl2], output_dir=str(out2), mode="vlm",
        batch_size=2, modality_id_ranges=[IMG_RANGE], max_visual_tokens_per_sample=256,
    )
    DataBuilder(max_seq_len=256).build_dataset(
        cfg2, modality_encoder=make_vlm_encode_callback(image_range_start=IMG_BASE))
    man2 = manifest_of(out2)
    assert man2.get("align_fingerprint") is None, "不声明 AV 区间 ⇒ 对齐未启用"
    assert "row_sids" not in man2["schema"]
    for sm in man2["shards"]:
        assert not os.path.exists(os.path.join(
            str(out2), sm["filename"][: -len(".bin")] + ".col_row_sids"))
    # 读回：无 videos/av_align 键
    for d in iter_align_batches(out2, 2, 256):
        assert "av_align" not in d
        assert "videos" not in d


def test_compat_align_not_requested_no_key(tmp_path):
    rows = [audio_row(ts=0.5, dur=2.0, text="a")]
    jsonl = write_jsonl(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out_a"
    build_align(align_cfg(jsonl, out, bs=1, msl=512, msl_visual=512), msl=512)
    # 不请求 align：batch dict 无 av_align 键（旧 trainer 零影响）
    for d in iter_align_batches(out, 1, 512):
        assert "av_align" not in d


# ============================================================
# A7 负面：align=True 但数据集未启用对齐 → 构造期 fail-fast
# ============================================================
def test_negative_align_requested_but_not_enabled(tmp_path):
    rows = [{"text": "t0"}, image_row(text="c")]
    jsonl = write_jsonl(tmp_path / "img.jsonl", rows)
    out = tmp_path / "out_img"
    cfg = BuildConfig(
        input_paths=[jsonl], output_dir=str(out), mode="vlm",
        batch_size=2, modality_id_ranges=[IMG_RANGE], max_visual_tokens_per_sample=256,
    )
    DataBuilder(max_seq_len=256).build_dataset(
        cfg, modality_encoder=make_vlm_encode_callback(image_range_start=IMG_BASE))
    with pytest.raises(Exception, match="no align_fingerprint"):
        RustStreamReader(os.path.join(str(out), "index.json"), 2, 256, 1, 0, 42, align=True)


# ============================================================
# 确定：同参数两次构建 av_align 一致
# ============================================================
def test_determinism_two_builds(tmp_path):
    rows = [audio_row(ts=0.5, dur=2.0, text="a"), video_row(ts=2.5, dur=2.0, text="b")]
    jsonl = write_jsonl(tmp_path / "mix.jsonl", rows)
    outs = []
    for i in range(2):
        out = tmp_path / f"out_det_{i}"
        build_align(align_cfg(jsonl, out, bs=2, msl=512, msl_visual=512), msl=512)
        outs.append(out)
    ma, mb = manifest_of(outs[0]), manifest_of(outs[1])
    assert [s["hashes"] for s in ma["shards"]] == [s["hashes"] for s in mb["shards"]]
    ea = [e for d in iter_align_batches(outs[0], 2, 512, align=True) for e in d["av_align"]]
    eb = [e for d in iter_align_batches(outs[1], 2, 512, align=True) for e in d["av_align"]]
    assert ea == eb, "同参数两次构建 av_align 必须一致"


# ============================================================
# P3c-5 锚点：桶边界——W_ms=2000 整倍值落下一桶（floor 语义）
# ============================================================
def test_bucket_boundary_exact_w_ms_multiples(tmp_path):
    """window_index == floor(start_ms / W_ms)：ts 恰为 W_ms(2000ms) 整倍时进下一桶。

    ts=2.0s→w1（不是 w0）、ts=4.0s→w2（不是 w1）——精确整倍边界归属后一窗，
    与 A3 的非整倍起点（2500ms→w1）互补，锁定 floor 而非 round/ceil。

    模态混合（受 INV-2 约束所迫）：audio 单窗要求 `start_ms ≤ duration_sec×1000`
    且 `end_ms` 不跨下一桶——w1 用 audio(ts=2.0,dur=2.0)（start_ms=2000、
    end_ms=4000 ⇒ k_min=k_max=1 恰单窗）；w0 用 audio(ts=0.5,dur=1.0)。而 w2 无法
    用 audio 构造单窗（INV-2 要 dur≥4 ⇒ end_ms=8000 跨 w2/w3），故 w2 用 video
    （rec 不按 duration 切窗、T 由帧率推导，ts=4.0 单窗 w2）——顺带验证 audio/video
    两侧 floor 边界语义一致。
    """
    rows = [
        audio_row(ts=2.0, dur=2.0, text="boundary two"),   # audio 2000ms → w1（单窗）
        video_row(ts=4.0, dur=2.0, text="boundary four"),  # video 4000ms → w2
        audio_row(ts=0.5, dur=1.0, text="below one"),      # audio 500ms  → w0
    ]
    jsonl = write_jsonl(tmp_path / "boundary.jsonl", rows)
    out = tmp_path / "out_boundary"
    build_align(align_cfg(jsonl, out, bs=1, msl=512, msl_visual=512), msl=512)
    entries = [e for d in iter_align_batches(out, 1, 512, align=True) for e in d["av_align"]]
    # sample_id 按输入序：0=audio ts2.0 / 1=video ts4.0 / 2=audio ts0.5；均单窗
    wins = {}
    for e in entries:
        wins.setdefault(e["sample_id"], set()).add(e["window_index"])
    assert wins.get(0) == {1}, "audio ts=2000ms 整倍 → window_index=1（floor 边界，不落 w0）"
    assert wins.get(1) == {2}, "video ts=4000ms 整倍 → window_index=2"
    assert wins.get(2) == {0}, "audio ts=500ms → window_index=0"


# ============================================================
# P3c-5 锚点：多 shard + 跨 shard join + text-only 负面注入
# ============================================================
def test_multishard_cross_shard_audio_join(tmp_path):
    """`batches_per_shard=1` 强制多 shard；audio 样本分散在不同 Vlm shard，其 rec
    落在独立的 audio 流 shard 上——读回 join 必须跨 shard 覆盖全部 audio rec（不漏），
    且注入的 text-only 样本不得进入 av_align（负面注入）。

    这是 §2 屏障 / VR-2 `align_aud_range` 覆盖路径的 e2e 锚点（原评审 P3c-5：
    此前多 shard 屏障路径零 e2e 覆盖）。
    """
    # audio 受 INV-2 `start_ms ≤ duration_sec×1000` 约束 ⇒ 三处 audio 均 ts≤dur
    # （跨 shard join 覆盖不依赖 window 差异，sid 已唯一区分；统一 w0 即可）。
    rows = [
        audio_row(ts=0.5, dur=1.0, text="a0"),
        video_row(ts=1.5, dur=1.0, text="v1"),
        audio_row(ts=0.5, dur=1.0, text="a2"),
        {"text": "plain text only, no av track"},   # text-only 负面注入（sid 3）
        audio_row(ts=0.5, dur=1.0, text="a4"),
    ]
    jsonl = write_jsonl(tmp_path / "ms.jsonl", rows)
    out = tmp_path / "out_ms"
    build_align(align_cfg(jsonl, out, bs=1, msl=512, msl_visual=512,
                          batches_per_shard=1), msl=512)
    man = manifest_of(out)
    assert len(man["shards"]) > 1, "batches_per_shard=1 + 5 样本必须产出多 shard"
    assert man["audio_shards"], "有 audio 样本 ⇒ 必须有 audio 流 shard"

    entries = [e for d in iter_align_batches(out, 1, 512, align=True) for e in d["av_align"]]
    # 跨 shard join：3 个 audio 样本（输入序 sid 0/2/4）的 rec 都必须 join 到
    audio_sids = {e["sample_id"] for e in entries if e["audio_rec_range"]}
    assert audio_sids == {0, 2, 4}, (
        f"跨 shard audio join 漏覆盖：期望 sid {{0,2,4}}，实得 {audio_sids}"
    )
    # video 样本（sid 1）video 侧 join 到
    assert any(e["sample_id"] == 1 and e["video_rec_range"] for e in entries), \
        "video 样本 sid 1 的 video rec 须 join 到"
    # text-only 负面注入：sid 3 纯文本（无 AV 轨）不得进 av_align
    assert not any(e["sample_id"] == 3 for e in entries), "text-only 样本不得进 av_align"

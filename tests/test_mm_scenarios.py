"""使用场景矩阵 e2e（tests/test_mm_scenarios.py）。

20 个用户视角场景：PT/SFT 旧路径回归、VLM 单模态/混合模态、长音频切窗、
对齐读回、预算与负面契约、失败清理、确定性。资源约束：全部小数据
（<=8 行、载荷 <=512B、msl <=512），不跑吞吐基线。

S01 PT 纯文本            S11 四模态混合数据集
S02 SFT ChatML 对话      S12 长音频多窗切分
S03 pt 误传 vlm 参数      S13 align=True 三链读回
S04 VLM 纯文本            S14 align 未请求无键
S05 VLM 单图+caption      S15 视觉预算 TruncateAndCount
S06 VLM 多图              S16 视觉预算 HardFail
S07 VLM 纯音频            S17 缺 audio rate fail-fast
S08 VLM 纯视频            S18 缺 timestamp fail-fast
S09 图+音交错样本          S19 fail-fast 无半成品残留
S10 图+视交错样本          S20 两次构建字节一致
"""
import base64
import json
import os
import struct

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
W_MS = 2000

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")

# ChatML 模板（用字符拼接构造，避免源文件出现模板括号序列）
_LB, _RB, _PC = "{", "}", "%"
CHAT_TEMPLATE = (
    _LB + _PC + " for msg in messages " + _PC + _RB
    + _LB + _LB + " msg.role " + _RB + _RB + "\n"
    + _LB + _LB + " msg.content " + _RB + _RB
    + _LB + _PC + " endfor " + _PC + _RB
)


def make_audio_payload(n, seed=0x66):
    return b"fLaC" + bytes((i * 7 + seed) % 256 for i in range(max(0, n - 4)))


def make_video_payload(n, seed=0x00):
    return b"\x00\x00\x00\x18ftypmp42" + bytes((i * 3 + seed) % 256 for i in range(max(0, n - 12)))


def frame_fn(payload, indices):
    return [bytes(((i * 7 + 11) % 256) for _ in range(64)) for i in indices]


def b64(data):
    return base64.b64encode(data).decode()


def audio_row(ts, dur, text=None, payload_len=256):
    row = {
        "audio_b64": b64(make_audio_payload(payload_len)),
        "audio_media_type": "audio/flac",
        "audio_meta": {"sample_rate": 16000, "channels": 1, "duration_sec": dur},
        "timestamp": ts,
    }
    if text:
        row["text"] = text
    return row


def video_row(ts, dur, text=None, payload_len=384):
    row = {
        "video_b64": b64(make_video_payload(payload_len)),
        "video_media_type": "video/mp4",
        "video_meta": {"fps": 30.0, "n_frames_total": 900, "duration_sec": dur},
        "timestamp": ts,
    }
    if text:
        row["text"] = text
    return row


def image_row(w=16, h=16, text=None):
    row = {"image_b64": b64(bytes((i * 13) % 256 for i in range(64))),
           "image_meta": {"width": w, "height": h}}
    if text:
        row["text"] = text
    return row


def multi_image_row(w=16, h=16, text=None):
    row = {"image_b64": [b64(bytes((i * 13) % 256 for i in range(64))) for _ in range(3)],
           "image_meta": {"width": w, "height": h}}
    if text:
        row["text"] = text
    return row


def img_audio_row(ts, dur, text=None):
    row = {
        "image_b64": b64(bytes((i * 5) % 256 for i in range(48))),
        "image_meta": {"width": 16, "height": 16},
        "audio_b64": b64(make_audio_payload(128)),
        "audio_media_type": "audio/flac",
        "audio_meta": {"sample_rate": 16000, "channels": 1, "duration_sec": dur},
        "timestamp": ts,
    }
    if text:
        row["text"] = text
    return row


def img_video_row(ts, dur, text=None):
    row = {
        "image_b64": b64(bytes((i * 5) % 256 for i in range(48))),
        "image_meta": {"width": 16, "height": 16},
        "video_b64": b64(make_video_payload(300)),
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


def vlm_cb():
    return make_vlm_encode_callback(
        image_range_start=IMG_BASE, audio_range_start=AUD_BASE, audio_token_rate=RATE,
        video_range_start=VID_BASE, video_n_frames=VFRAMES, video_resize=VRESIZE,
        frame_fn=frame_fn,
    )


def vlm_cfg(jsonl, out_dir, bs=2, msl_visual=None, id_ranges=None, **kw):
    # kw 可覆盖 w_ms / audio_window_token_rate / video_n_frames / video_resize 等
    # （setdefault 语义：调用方显式传入的值优先，避免重复传参 TypeError）
    # id_ranges：默认声明三模态区间（构建期启用对齐）；传 [IMG_RANGE] 可构造
    # "不启用对齐"的产物（P3c：align_enabled = 声明了 audio/video 专属区间）。
    return BuildConfig(
        input_paths=[jsonl], output_dir=str(out_dir), mode="vlm",
        batch_size=bs,
        modality_id_ranges=id_ranges or [IMG_RANGE, AUD_RANGE, VID_RANGE],
        max_visual_tokens_per_sample=msl_visual,
        w_ms=kw.pop("w_ms", W_MS),
        audio_window_token_rate=kw.pop("audio_window_token_rate", RATE),
        video_n_frames=kw.pop("video_n_frames", VFRAMES),
        video_resize=kw.pop("video_resize", VRESIZE),
        **kw,
    )


def build(cfg, msl, cb=None):
    return DataBuilder(max_seq_len=msl).build_dataset(cfg, modality_encoder=cb or vlm_cb())


def manifest_of(out_dir):
    with open(os.path.join(str(out_dir), "index.json")) as f:
        return json.load(f)


def files_in(out_dir):
    return sorted(os.listdir(str(out_dir)))


def read_vlm_batches(out_dir, bs, msl, **kw):
    reader = RustStreamReader(os.path.join(str(out_dir), "index.json"), bs, msl, 1, 0, 42, **kw)
    while True:
        d = reader.next_batch_capsules()
        if d is None:
            return
        yield d


def parse_aud_cu(data):
    assert len(data) % 16 == 0
    return [struct.unpack_from("<IIQ", data, i * 16) for i in range(len(data) // 16)]


def parse_aud_rec(data):
    assert len(data) % 40 == 0
    return [struct.unpack_from("<QIIIHBBQII", data, i * 40) for i in range(len(data) // 40)]


def parse_cu_file(data):
    """Memmap .cu：length-prefixed `[u32 count][i32*count]` per batch。"""
    batches, pos = [], 0
    while pos < len(data):
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4
        cu = list(struct.unpack_from(f"<{n}i", data, pos))
        pos += 4 * n
        batches.append(cu)
    return batches


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ============================================================
# S01-S02：旧路径回归（PT 纯文本 / SFT ChatML 对话）
# ============================================================
def test_s01_pt_plain_text_legacy_path(tmp_path):
    if not TOKENIZER_PATH or not os.path.exists(TOKENIZER_PATH):
        pytest.skip("TOKENIZER_PATH not available")
    jsonl = write_jsonl(tmp_path / "pt.jsonl",
                        [{"text": "plain document number %d" % i} for i in range(6)])
    out = tmp_path / "out_pt"
    cfg = BuildConfig(input_paths=[jsonl], output_dir=str(out), mode="pt", batch_size=2)
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=128).build_dataset(cfg)
    man = manifest_of(out)
    assert man["shards"] and not man.get("audio_shards")
    assert "align_fingerprint" not in man
    for k in ("aud_blob", "aud_rec", "aud_cu", "vid_blob", "vid_rec", "row_sids"):
        assert k not in man["schema"], "PT 产物不得出现多模态 schema 键: %s" % k
    assert all(f.startswith("shard_") for f in files_in(out) if f != "index.json")
    batches = list(read_vlm_batches(out, 2, 128))
    assert batches and batches[0]["input_ids"] is not None


def test_s02_sft_chatml_legacy_path(tmp_path):
    if not TOKENIZER_PATH or not os.path.exists(TOKENIZER_PATH):
        pytest.skip("TOKENIZER_PATH not available")
    rows = [{"messages": [{"role": "user", "content": "q%d" % i},
                          {"role": "assistant", "content": "a%d" % i}]} for i in range(6)]
    jsonl = write_jsonl(tmp_path / "sft.jsonl", rows)
    out = tmp_path / "out_sft"
    # template / turn 标记属 BuildConfig 配置面（build_dataset(cfg, ...) 已拒绝
    # config + legacy kwargs 混传）；turn 标记 id 交由分词器自动推导。
    cfg = BuildConfig(input_paths=[jsonl], output_dir=str(out), mode="sft",
                      batch_size=2, template_str=CHAT_TEMPLATE)
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=128).build_dataset(cfg)
    man = manifest_of(out)
    assert man["shards"]
    assert "align_fingerprint" not in man
    batches = list(read_vlm_batches(out, 2, 128))
    # SFT 产物走 DocumentAwarePacker → Arrow 格式，RustStreamReader 原样透出
    # arrow_batch（input_ids capsule 仅 PT/Vlm 的 Memmap 分支有）。
    assert batches and "arrow_batch" in batches[0]


def test_s03_pt_with_vlm_args_loud_fail(tmp_path):
    jsonl = write_jsonl(tmp_path / "pt.jsonl", [{"text": "x"}])
    with pytest.raises(ValueError, match="vlm-only"):
        BuildConfig(input_paths=[jsonl], output_dir=str(tmp_path / "o"), mode="pt",
                    w_ms=4000)


# ============================================================
# S04-S08：VLM 单模态场景
# ============================================================
def test_s04_vlm_text_only(tmp_path):
    jsonl = write_jsonl(tmp_path / "t.jsonl", [{"text": "t%d" % i} for i in range(4)])
    out = tmp_path / "out_t"
    # 纯文本构建不声明 audio/video 区间 → 构建期不启用对齐（P3c：无
    # align_fingerprint、无 col_row_sids），产物与 P2 期逐字节兼容。
    build(vlm_cfg(jsonl, out, id_ranges=[IMG_RANGE]), msl=256)
    man = manifest_of(out)
    # audio_shards 空时 serde skip_serializing_if 省略键（无 audio 流 = 无此键）。
    assert man["shards"] and not man.get("audio_shards")
    assert "align_fingerprint" not in man, "纯文本 vlm 构建不应有 audio 指纹"
    assert "aud_blob" not in man["schema"] and "vid_rec" not in man["schema"]
    assert not any(f.endswith((".aud_bin", ".col_aud_blob", ".col_vid_rec")) for f in files_in(out))
    ids = [t for d in read_vlm_batches(out, 2, 256)
           for t in _safe_from_dlpack(d["input_ids"]).reshape(-1).tolist()]
    assert ids and all(t < IMG_BASE for t in ids), "纯文本样本不得出现模态 placeholder"


def test_s05_vlm_single_image_caption(tmp_path):
    jsonl = write_jsonl(tmp_path / "i.jsonl", [image_row(text="a red square")])
    out = tmp_path / "out_i"
    build(vlm_cfg(jsonl, out, msl_visual=256), msl=256)
    man = manifest_of(out)
    assert man["schema"]["img_blob"] == "bytes" and man["schema"]["mm_cu"] == "mmcu"
    assert man["shards"][0]["hashes"].get("sha256:col_img_rec")
    ids = [t for d in read_vlm_batches(out, 2, 256)
           for t in _safe_from_dlpack(d["input_ids"]).reshape(-1).tolist()]
    assert any(IMG_BASE <= t < IMG_BASE + 100 for t in ids), "单图样本必须有 image placeholder"
    assert not man.get("audio_shards")


def test_s06_vlm_multi_image(tmp_path):
    # 一期契约：多图（image_b64 数组）与伴随 caption 的组合显式不支持
    # （source.rs "multi-image ... with text caption is not supported yet"），
    # 多图本身走交错样本正常落 3 条 img_rec。
    jsonl = write_jsonl(tmp_path / "mi.jsonl", [multi_image_row()])
    out = tmp_path / "out_mi"
    build(vlm_cfg(jsonl, out, msl_visual=512), msl=512)
    man = manifest_of(out)
    stem = man["shards"][0]["filename"][: -len(".bin")]
    rec = read_bytes(os.path.join(str(out), stem + ".col_img_rec"))
    assert len(rec) % 32 == 0 and len(rec) // 32 == 3, "三图应落 3 条 img_rec（实测 %d）" % (len(rec) // 32)


def test_s07_vlm_audio_only(tmp_path):
    rows = [audio_row(1.5, 4.0), audio_row(0.0, 2.0)]
    jsonl = write_jsonl(tmp_path / "a.jsonl", rows)
    out = tmp_path / "out_a"
    build(vlm_cfg(jsonl, out), msl=256)
    man = manifest_of(out)
    assert man["audio_shards"], "纯音频构建必须有 audio 流 shard"
    assert man["schema"]["aud_rec"] == "audrec40"
    gs = man["global_stats"]
    assert gs["total_audio_tokens"] > 0 and gs["total_audio_clips"] > 0
    assert gs["total_tokens"] == (gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)
                                  + gs.get("total_audio_tokens", 0) + gs.get("total_video_tokens", 0))
    stem = man["audio_shards"][0]["filename"][: -len(".aud_bin")]
    cu = parse_aud_cu(read_bytes(os.path.join(str(out), stem + ".col_aud_cu")))
    recs = parse_aud_rec(read_bytes(os.path.join(str(out), stem + ".col_aud_rec")))
    assert sum(c[1] for c in cu) == len(recs), "Σ aud_cnt == rec 数"
    blob = read_bytes(os.path.join(str(out), stem + ".col_aud_blob"))
    assert sum(r[1] for r in recs if r[5] == 0) == len(blob), "VALID 过滤求和 == blob 长度"


def test_s08_vlm_video_only(tmp_path):
    jsonl = write_jsonl(tmp_path / "v.jsonl", [video_row(0.0, 2.0)])
    out = tmp_path / "out_v"
    build(vlm_cfg(jsonl, out, msl_visual=512), msl=512)
    man = manifest_of(out)
    assert man["schema"]["vid_rec"] == "vidrec40"
    stem = man["shards"][0]["filename"][: -len(".bin")]
    vid = read_bytes(os.path.join(str(out), stem + ".col_vid_rec"))
    assert len(vid) == 40, "单视频应落 1 条 vid_rec"
    mm = read_bytes(os.path.join(str(out), stem + ".mm_cu"))
    n_mod, img_cnt, aud_cnt, vid_cnt = struct.unpack_from("<IIII", mm, 0)
    assert vid_cnt == 1 and aud_cnt == 0
    assert man["global_stats"]["total_video_clips"] == 1
    assert not man.get("audio_shards")


# ============================================================
# S09-S11：混合模态场景（跨功能组合，本文件核心增量）
# ============================================================
def test_s09_image_plus_audio_interleaved(tmp_path):
    jsonl = write_jsonl(tmp_path / "ia.jsonl", [img_audio_row(0.25, 2.0, text="mixed")])
    out = tmp_path / "out_ia"
    build(vlm_cfg(jsonl, out, msl_visual=512), msl=512)
    man = manifest_of(out)
    assert man["shards"] and man["audio_shards"], "图+音样本必须同时产两流"
    stem = man["shards"][0]["filename"][: -len(".bin")]
    mm = read_bytes(os.path.join(str(out), stem + ".mm_cu"))
    for i in range(len(mm) // 16):
        n_mod, img_cnt, aud_cnt, vid_cnt = struct.unpack_from("<IIII", mm, i * 16)
        assert aud_cnt == 0, "Vlm 流 mm_cu.aud_cnt 恒 0"
    assert any(f.endswith(".col_img_rec") for f in files_in(out))
    ids = [t for d in read_vlm_batches(out, 2, 512)
           for t in _safe_from_dlpack(d["input_ids"]).reshape(-1).tolist()]
    assert any(IMG_BASE <= t < IMG_BASE + 100 for t in ids), "Vlm 流保留 image token"
    assert not any(AUD_BASE <= t < AUD_BASE + 1000 for t in ids), "audio token 不得进 Vlm 流"
    a_stem = man["audio_shards"][0]["filename"][: -len(".aud_bin")]
    recs = parse_aud_rec(read_bytes(os.path.join(str(out), a_stem + ".col_aud_rec")))
    assert recs and recs[0][7] == 0 and recs[0][9] == 250, "audio rec 覆盖 sid 0 / start_ms 250"


def test_s10_image_plus_video_interleaved(tmp_path):
    jsonl = write_jsonl(tmp_path / "iv.jsonl", [img_video_row(0.0, 2.0, text="av")])
    out = tmp_path / "out_iv"
    build(vlm_cfg(jsonl, out, msl_visual=512), msl=512)
    man = manifest_of(out)
    stem = man["shards"][0]["filename"][: -len(".bin")]
    mm = read_bytes(os.path.join(str(out), stem + ".mm_cu"))
    n_mod, img_cnt, aud_cnt, vid_cnt = struct.unpack_from("<IIII", mm, 0)
    assert img_cnt == 1 and vid_cnt == 1, "同 batch 应各 1 条 img/vid rec（实测 img=%d vid=%d）" % (img_cnt, vid_cnt)
    assert aud_cnt == 0
    assert len(read_bytes(os.path.join(str(out), stem + ".col_vid_rec"))) == 40
    assert not man.get("audio_shards")


def test_s11_four_modality_mixed_dataset(tmp_path):
    rows = [
        {"text": "plain"},
        image_row(text="pic"),
        audio_row(1.0, 2.0),
        video_row(0.0, 2.0),
        img_audio_row(0.0, 1.0),
        img_video_row(2.5, 1.0),
    ]
    jsonl = write_jsonl(tmp_path / "all.jsonl", rows)
    out = tmp_path / "out_all"
    build(vlm_cfg(jsonl, out, msl_visual=512), msl=512)
    man = manifest_of(out)
    gs = man["global_stats"]
    assert gs["stats_version"] == 2
    assert gs["total_tokens"] == (gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)
                                  + gs.get("total_audio_tokens", 0) + gs.get("total_video_tokens", 0))
    assert gs.get("total_image_tokens", 0) > 0 and gs["total_audio_tokens"] > 0
    assert gs.get("total_video_tokens", 0) > 0
    assert man["shards"] and man["audio_shards"]
    assert man["align_fingerprint"]["W_ms"] == W_MS
    assert "audio" in man["align_fingerprint"] and "video" in man["align_fingerprint"]
    for key in ("img_rec", "vid_rec", "aud_rec"):
        assert key in man["schema"]
    names = files_in(out)
    assert any(n.endswith(".col_vid_rec") for n in names)
    assert any(n.endswith(".col_aud_rec") for n in names)
    assert any(n.endswith(".col_img_rec") for n in names)
    # 三流文件族完整性：Vlm 主流 + 三模态 sidecar + audio 流五件套
    v_stem = man["shards"][0]["filename"][: -len(".bin")]
    for suffix in (".bin", ".idx", ".cu", ".mm_cu",
                   ".col_img_blob", ".col_img_rec",
                   ".col_vid_blob", ".col_vid_rec", ".col_row_sids"):
        assert os.path.exists(os.path.join(str(out), v_stem + suffix)), "Vlm 流文件缺: " + suffix
    a_stem = man["audio_shards"][0]["filename"][: -len(".aud_bin")]
    for suffix in (".aud_bin", ".aud_idx", ".aud_cu", ".aud_doc_ids",
                   ".aud_loss_weights", ".col_aud_blob", ".col_aud_rec", ".col_aud_cu"):
        assert os.path.exists(os.path.join(str(out), a_stem + suffix)), "audio 流文件缺: " + suffix


def test_s12_long_audio_window_split(tmp_path):
    jsonl = write_jsonl(tmp_path / "long.jsonl", [audio_row(1.5, 4.0)])
    out = tmp_path / "out_long"
    build(vlm_cfg(jsonl, out), msl=512)
    man = manifest_of(out)
    a_stem = man["audio_shards"][0]["filename"][: -len(".aud_bin")]
    recs = parse_aud_rec(read_bytes(os.path.join(str(out), a_stem + ".col_aud_rec")))
    assert [(r[9], r[3]) for r in recs] == [(1500, 500), (2000, 2000), (4000, 1500)]
    assert [r[5] for r in recs] == [0, 4, 4], "首窗 VALID + 两尾窗引用"
    first = (recs[0][0], recs[0][1])
    for r in recs[1:]:
        assert (r[0], r[1]) == first, "尾窗 (offset,len) 必须复用首窗"
    # token 守恒：audio 流各 batch 有效 token（.aud_cu 末值）之和 == manifest 口径；
    # 每个窗口 doc 由 GreedyPacker PT 语义追加 1 个 eos → 200 + 3 = 203，
    # 去 eos 的原始 token 序列 = 25 + 100 + 75 = 200（与 P3a A9 锚点同口径）。
    cu_batches = parse_cu_file(read_bytes(os.path.join(str(out), a_stem + ".aud_cu")))
    total = sum(cu[-1] for cu in cu_batches)
    assert total == man["global_stats"]["total_audio_tokens"]
    assert total - len(recs) == 200


# ============================================================
# S13-S14：对齐场景
# ============================================================
def test_s13_align_three_modalities_readback(tmp_path):
    rows = [
        audio_row(0.5, 2.0, text="zero five"),
        video_row(2.5, 2.0, text="two five"),
        {"text": "plain text"},
        img_video_row(1.0, 1.0, text="mixed"),
    ]
    jsonl = write_jsonl(tmp_path / "align.jsonl", rows)
    out = tmp_path / "out_align"
    build(vlm_cfg(jsonl, out, bs=2, msl_visual=512), msl=512)
    man = manifest_of(out)
    fp = man["align_fingerprint"]
    assert fp["text_in_axis"] is True
    assert fp["ts_token"]["encoding"] == "contiguous"
    assert man["schema"]["row_sids"] == "list<uint64>"
    for sm in man["shards"]:
        assert sm.get("align_aud_range") is not None
        assert "sha256:col_row_sids" in sm["hashes"]
    entries = []
    for d in read_vlm_batches(out, 2, 512, align=True):
        got = d.get("av_align")
        assert got, "align=True 必须透出 av_align"
        entries.extend(got)
    by_key = {(e["sample_id"], e["window_index"]): e for e in entries}
    assert (0, 0) in by_key and (1, 1) in by_key and (3, 0) in by_key, \
        "window_index == floor(start_ms / W_ms)（含非整倍 2500ms->w1）"
    e0 = by_key[(0, 0)]
    assert e0["text_token_range"] and e0["audio_rec_range"]
    e1 = by_key[(1, 1)]
    assert e1["video_rec_range"] and e1["text_token_range"]
    assert not any(e["sample_id"] == 2 for e in entries), "text-only 样本不得进对齐视图"


def test_s14_align_not_requested_no_key(tmp_path):
    jsonl = write_jsonl(tmp_path / "t.jsonl", [{"text": "a"}, image_row(text="b")])
    out = tmp_path / "out_noalign"
    # 只声明 image 区间 → 构建期不启用对齐（P3c：align_enabled = 声明了
    # audio/video 专属区间）；启用对齐才落 col_row_sids，读取端 align=False
    # 也不得透出 av_align。
    build(vlm_cfg(jsonl, out, id_ranges=[IMG_RANGE]), msl=256)
    for d in read_vlm_batches(out, 2, 256):
        assert "av_align" not in d, "未请求 align 时 batch dict 不得有 av_align 键"
    assert not any(f.endswith(".col_row_sids") for f in files_in(out))


# ============================================================
# S15-S20：行为契约（预算 / 负面 / 清理 / 确定性）
# ============================================================
def test_s15_visual_budget_truncate_and_count(tmp_path, capfd):
    jsonl = write_jsonl(tmp_path / "big.jsonl", [image_row(w=256, h=256, text="big")])
    out = tmp_path / "out_trunc"
    build(vlm_cfg(jsonl, out, msl_visual=64), msl=512)
    # Rust tracing 直接写 OS fd（绕过 sys.stdout）→ capfd 做 fd 级捕获。
    captured = capfd.readouterr()
    man = manifest_of(out)
    gs = man["global_stats"]
    # 256×256 图 = 256 visual token > 预算 64：整段丢弃，manifest 统计口径为
    # post-drop（与 P2 packer 单测一致）；截断计数经构建期 WARN 透出。
    assert gs.get("total_images", 0) == 0, "超预算图像段必须被丢弃"
    assert gs.get("total_image_tokens", 0) == 0
    assert "image segment(s) dropped by visual budget" in captured.out + captured.err
    assert man["shards"]


def test_s16_visual_budget_hard_fail(tmp_path):
    jsonl = write_jsonl(tmp_path / "big.jsonl", [image_row(w=256, h=256, text="big")])
    out = tmp_path / "out_hard"
    cfg = vlm_cfg(jsonl, out, msl_visual=64, visual_budget_hard_fail=True)
    with pytest.raises(Exception, match="visual"):
        build(cfg, msl=512)


def test_s17_audio_without_rate_fail_fast(tmp_path):
    jsonl = write_jsonl(tmp_path / "a.jsonl", [audio_row(0.0, 2.0)])
    out = tmp_path / "out_norate"
    cfg = vlm_cfg(jsonl, out, audio_window_token_rate=None)
    with pytest.raises(Exception, match="alignment config missing"):
        build(cfg, msl=256)


def test_s18_audio_without_timestamp_fail_fast(tmp_path):
    row = {"audio_b64": b64(make_audio_payload(64)), "audio_media_type": "audio/flac",
           "audio_meta": {"sample_rate": 16000, "channels": 1, "duration_sec": 2.0}}
    jsonl = write_jsonl(tmp_path / "nots.jsonl", [row])
    out = tmp_path / "out_nots"
    with pytest.raises(Exception, match="no timestamp"):
        build(vlm_cfg(jsonl, out), msl=256)


def test_s19_fail_fast_leaves_no_partial_files(tmp_path):
    row = {"audio_b64": b64(make_audio_payload(64)), "audio_media_type": "audio/flac",
           "audio_meta": {"channels": 1, "duration_sec": 2.0}, "timestamp": 0.0}
    jsonl = write_jsonl(tmp_path / "bad.jsonl", [row])
    out = tmp_path / "out_bad"
    with pytest.raises(Exception, match="no sample_rate"):
        build(vlm_cfg(jsonl, out), msl=256)
    assert not os.path.exists(os.path.join(str(out), "index.json")), "失败不得落 index.json"
    leftovers = [f for f in files_in(out) if f != "index.json"] if os.path.isdir(out) else []
    assert not leftovers, "失败路径不得留半成品文件: %s" % leftovers


def _file_tree(root):
    """递归收集目录全部文件（相对路径 → 字节），供逐文件确定性比对。"""
    tree = {}
    for dirpath, _, names in os.walk(str(root)):
        for name in names:
            path = os.path.join(dirpath, name)
            tree[os.path.relpath(path, str(root))] = read_bytes(path)
    return tree


def test_s20_two_builds_byte_identical(tmp_path):
    rows = [{"text": "plain"}, image_row(text="pic"), audio_row(1.5, 4.0), video_row(0.0, 2.0)]
    jsonl = write_jsonl(tmp_path / "det.jsonl", rows)
    outs = []
    for i in range(2):
        out = tmp_path / ("out_det%d" % i)
        build(vlm_cfg(jsonl, out, msl_visual=512), msl=512)
        outs.append(out)
    man_a, man_b = manifest_of(outs[0]), manifest_of(outs[1])
    assert [(s["filename"], s["hashes"]) for s in man_a["shards"]] == \
           [(s["filename"], s["hashes"]) for s in man_b["shards"]], "Vlm 流 shard 哈希必须一致"
    assert [(s["filename"], s["hashes"]) for s in man_a["audio_shards"]] == \
           [(s["filename"], s["hashes"]) for s in man_b["audio_shards"]], "audio 流 shard 哈希必须一致"
    # 逐文件比对（计划口径：两次构建全部产物 sha256 一致；逐字节等价且更强）：
    # 覆盖 Vlm/audio 主流五件套、三模态 sidecar、col_row_sids 与 index.json。
    tree_a, tree_b = _file_tree(outs[0]), _file_tree(outs[1])
    assert sorted(tree_a) == sorted(tree_b), "两次构建产物文件集合必须一致"
    for name in sorted(tree_a):
        assert tree_a[name] == tree_b[name], "两次构建产物必须逐字节相同: %s" % name

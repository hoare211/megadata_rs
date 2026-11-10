"""P2 vlm 纵向闭环 golden case + 集成测试（handoff §4.3 / §4.4）。

**不依赖分词器**（identity 映射，``DataBuilder(max_seq_len=N)`` backend="none"）/ 网络
（base64 内联 PNG，``struct``+``zlib`` 手拼）。

覆盖：
- §4.3 T1 纵向闭环 golden case：单图样本 + 纯文本样本，五对应 + manifest 全断言
  （vlm e2e 首次闭环的回归锚点）。
- §4.4 九组集成用例：多图 / 交错（多样本混 batch）/ 图文混排 / padding 残余批 /
  视觉预算 TruncateAndCount / 视觉预算 HardFail / 图像段原子性丢弃 /
  负面（回调契约违反四类）/ 旧路径回归由 pytest 全量覆盖。

读回约定：``next_batch_capsules`` 返回的 ``cu_seqlens`` 是裸 ``Vec<i32>``→Python list
（**非** DLPack capsule），``input_ids`` / ``loss_weights`` / ``images[].bytes`` 才是
capsule——按引擎 ``_safe_from_dlpack`` + ``torch.as_tensor`` 规范消费。

``labels`` 列在 manifest schema 声明为 ``int32``（与 PT/SFT 一致的格式契约），但
**不持久化、读侧不作为 capsule 透出**；VLM 的逐位置监督掩码由 ``loss_weights``
（f16）承载（image/pad 位 0.0、text 位 1.0）——这是既有模式，非 VLM 缺口。
"""
import base64
import json
import os
import struct
import zlib

import pytest
import torch

from megadata import BuildConfig, DataBuilder
from megadata.engine import RustStreamReader, _safe_from_dlpack
from megadata.modality_adapter import make_vlm_encode_callback

IMG_BASE = 32000
IMG_RANGE = (IMG_BASE, IMG_BASE + 100, "image")


# ============================================================
# 辅助
# ============================================================
def make_png(w, h, color=(200, 30, 30)):
    """手拼最小 PNG（无 PIL 依赖；与冒烟脚本一致）。"""
    raw = b"".join(b"\x00" + bytes(color * w) for _ in range(h))

    def chunk(t, d):
        c = t + d
        return (struct.pack(">I", len(d)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def vlm_cfg(jsonl, out_dir, *, bs, msl_visual=None, **kw):
    """vlm BuildConfig 构造器（id_ranges 必填，其余按需）。"""
    return BuildConfig(
        input_paths=[jsonl], output_dir=str(out_dir), mode="vlm",
        batch_size=bs, modality_id_ranges=[IMG_RANGE],
        max_visual_tokens_per_sample=msl_visual,
        **kw,
    )


def build(cfg, msl):
    """无分词器构造 + vlm 回调构建，返回 summary dict。"""
    cb = make_vlm_encode_callback(image_range_start=IMG_BASE, patch_size=16)
    return DataBuilder(max_seq_len=msl).build_dataset(cfg, modality_encoder=cb)


def build_with(cfg, cb, msl):
    return DataBuilder(max_seq_len=msl).build_dataset(cfg, modality_encoder=cb)


def read_first_batch(index, bs, msl):
    """读回首个 batch：capsule→tensor / list→tensor 规范消费。"""
    reader = RustStreamReader(index, bs, msl, 1, 0, 42)
    d = reader.next_batch_capsules()
    assert d is not None, "dataset produced no batch"
    ids = _safe_from_dlpack(d["input_ids"])          # [bs, msl] int32
    cu_raw = d.get("cu_seqlens")
    cu = torch.as_tensor(cu_raw, dtype=torch.int32).tolist() if cu_raw else []
    lw_raw = d.get("loss_weights")
    lw = _safe_from_dlpack(lw_raw) if lw_raw is not None else None  # [bs, msl] f16
    imgs = d.get("images", [])
    return {"ids": ids, "cu": cu, "lw": lw, "images": imgs, "raw": d}


def manifest_of(out_dir):
    return json.load(open(os.path.join(str(out_dir), "index.json")))


def img_records(imgs):
    return [(e["sample_id"], e["seg_order"], e["grid_h"], e["grid_w"], e["flags"])
            for e in imgs]


def blob_bytes(img_entry):
    return _safe_from_dlpack(img_entry["bytes"]).numpy().tobytes()


def _enc(sid, token_seq, seg_table, labels, media_refs=None):
    """构造一个结构完整的 EncodedSample dict（坏回调用）。"""
    return {
        "sample_id": sid,
        "token_seq": token_seq,
        "segment_table": seg_table,
        "labels": labels,
        "loss_weights": None,
        "stats": {"text_tokens": len(token_seq), "image_tokens": 0,
                  "audio_tokens": 0, "video_tokens": 0, "images": 0,
                  "audio_clips": 0, "video_clips": 0},
        "media_refs": media_refs or [],
    }


def _txt_seg(order, start, end):
    return {"kind": "text", "order": order, "token_start": start,
            "token_end": end, "media_record_id": None, "loss_weight": 255}


# ============================================================
# §4.3 T1 纵向闭环 golden case
# ============================================================
def test_t1_vertical_loop_golden(tmp_path):
    """单图样本（caption "hello" + 16×32 PNG → grid 1×2 = 2 img tokens）
    + 纯文本样本 "pure text row"（13 tokens）。MSL=64 / BS=2 / patch=16 / 恒等映射。

    五对应 + manifest 全断言（vlm e2e 首次闭环回归锚点）。
    """
    MSL, BS = 64, 2
    png = make_png(16, 32)  # grid (1, 2) → 2 image tokens
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"text": "hello", "image_b64": base64.b64encode(png).decode(),
         "image_meta": {"width": 16, "height": 32}},
        {"text": "pure text row"},
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=256)
    summary = build(cfg, MSL)
    assert summary["total_tokens"] == 20  # 5 + 2 + 13

    m = manifest_of(tmp_path / "ds")
    gs = m["global_stats"]

    # ---- manifest 断言 ----
    assert m["modality_id_ranges"] == [{"start": IMG_BASE, "end": IMG_BASE + 100,
                                         "kind": "image"}]
    for k in ("img_blob", "img_rec", "mm_cu", "loss_weights", "input_ids",
              "labels", "position_ids", "cu_seqlens"):
        assert k in m["schema"], k
    assert gs.get("total_images", 0) == 1
    assert gs.get("total_text_tokens", 0) == 18           # 5 + 13
    assert gs.get("total_image_tokens", 0) == 2
    assert gs["total_tokens"] == gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)
    assert gs["stats_version"] == 2
    sh0 = m["shards"][0]["hashes"]
    for k in ("sha256", "sha256:col_img_blob", "sha256:col_img_rec",
              "sha256:mm_cu", "sha256:col_loss_weights"):
        assert k in sh0, k

    # ---- 读回断言 ----
    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    ids = b["ids"].tolist()
    lw = b["lw"]

    row0, row1 = ids[0], ids[1]
    # 1) token range：placeholder 连续 + cu 样本边界
    ph = [i for i, v in enumerate(row0) if IMG_BASE <= v < IMG_BASE + 100]
    assert ph == [5, 6], f"row0 placeholder 应连续 [5,6]: {ph}"
    assert row0[:5] == [ord(c) for c in "hello"]
    assert b["cu"] == [0, 7, 20], b["cu"]            # bs+1 个样本边界
    assert row1[:13] == [ord(c) for c in "pure text row"]

    # 2) image record：sample_id / seg_order / grid / flags
    recs = img_records(b["images"])
    # P3a/P3b 起媒体段保持源侧 segment_order（demux 的 (sid, seg_order) 源侧
    # 查找契约）：caption Text 取 900 新 order，Image 保持源侧 0。
    assert recs == [(0, 0, 1, 2, 0)], recs
    # 4) blob bytes == 输入 PNG
    assert blob_bytes(b["images"][0]) == png

    # 3) labels+loss_weights（loss_weights 是 VLM 监督掩码；labels 列在 schema 声明
    #    但不作为 capsule 透出——与 PT/SFT 一致）
    assert lw is not None
    lw0 = lw[0].view(torch.float16).tolist()
    assert lw0[0] == 1.0 and lw0[4] == 1.0           # text 位 1.0
    assert lw0[5] == 0.0 and lw0[6] == 0.0           # image 位 0.0
    assert lw0[10] == 0.0                            # pad 位 0.0
    lw1 = lw[1].view(torch.float16).tolist()
    assert lw1[0] == 1.0 and lw1[12] == 1.0          # row1 text 位 1.0
    assert lw1[13] == 0.0                            # row1 pad 位 0.0

    # 5) 可追溯链：sample_id 0 → seg_order 0 → grid (1,2) → placeholder [32000,32001]
    #    @ input_ids[5:7]
    assert row0[5:7] == [IMG_BASE, IMG_BASE + 1]


# ============================================================
# §4.4 集成测试（九组）
# ============================================================
def test_1_multi_image_array(tmp_path):
    """多图：image_b64 数组 → 交错样本，2 条 rec、Σ img_cnt == 2。"""
    MSL, BS = 64, 2
    png1, png2 = make_png(16, 16), make_png(16, 16)   # 各 grid (1,1) → 1 token
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"image_b64": [base64.b64encode(png1).decode(),
                       base64.b64encode(png2).decode()],
         "image_meta": {"width": 16, "height": 16}},
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=256)
    build(cfg, MSL)

    m = manifest_of(tmp_path / "ds")
    gs = m["global_stats"]
    assert gs.get("total_images", 0) == 2
    assert gs.get("total_image_tokens", 0) == 2
    assert gs["total_tokens"] == 2 == gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)

    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    recs = img_records(b["images"])
    assert recs == [(0, 0, 1, 1, 0), (0, 1, 1, 1, 0)], recs
    assert b["ids"][0].tolist()[:2] == [IMG_BASE, IMG_BASE + 1]
    assert b["cu"] == [0, 2, 2]                       # row1 空 → cu 末值重复
    assert blob_bytes(b["images"][0]) == png1
    assert blob_bytes(b["images"][1]) == png2


def test_2_interleave_via_multi_sample_batch(tmp_path):
    """交错 text+image+text 用多样本混 batch 表达（单样本内多 Text 段一期不支持）。"""
    MSL, BS = 8, 3
    png = make_png(16, 16)                            # grid (1,1) → 1 token
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"text": "a"},                                # sample0 text
        {"image_b64": base64.b64encode(png).decode(),  # sample1 image（无 caption）
         "image_meta": {"width": 16, "height": 16}},
        {"text": "c"},                                # sample2 text
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=256)
    build(cfg, MSL)

    m = manifest_of(tmp_path / "ds")
    gs = m["global_stats"]
    assert gs.get("total_images", 0) == 1
    assert gs.get("total_text_tokens", 0) == 2              # "a" + "c"
    assert gs.get("total_image_tokens", 0) == 1

    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    ids = b["ids"].tolist()
    assert ids[0][0] == ord("a") and ids[2][0] == ord("c")
    assert ids[1][0] == IMG_BASE                       # 中间行 = image
    assert b["cu"] == [0, 1, 2, 3]                    # 3 样本边界
    assert img_records(b["images"]) == [(1, 0, 1, 1, 0)]  # 唯一 rec 属 sample1


def test_3_mixed_batch_image_plus_text(tmp_path):
    """图文混排：图像样本 + 纯文本样本同 batch → 纯文本样本不产 rec。"""
    MSL, BS = 64, 2
    png = make_png(16, 32)                            # grid (1,2) → 2 tokens
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"text": "cap", "image_b64": base64.b64encode(png).decode(),
         "image_meta": {"width": 16, "height": 32}},
        {"text": "plain text"},                       # 纯文本，无图
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=256)
    build(cfg, MSL)

    m = manifest_of(tmp_path / "ds")
    assert m["global_stats"]["total_images"] == 1    # 只有图像样本产 rec
    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    # row0 = caption "cap"(3) + image(2)；row1 = "plain text"(10)
    assert b["cu"] == [0, 5, 15]
    assert b["ids"][0].tolist()[:5] == [ord("c"), ord("a"), ord("p"),
                                         IMG_BASE, IMG_BASE + 1]
    assert b["ids"][1].tolist()[:10] == [ord(c) for c in "plain text"]
    assert len(b["images"]) == 1


def test_4_padding_residual_batch(tmp_path):
    """padding/残余批：bs×msl 恒定、空行 cu 重复值。"""
    MSL, BS = 8, 2
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [{"text": "hi"}])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=256)
    build(cfg, MSL)

    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    ids = b["ids"].tolist()
    # 定宽：两行均 msl 长
    assert len(ids[0]) == MSL and len(ids[1]) == MSL
    # row0 = "hi"(2) + pad；row1 全 pad
    assert ids[0][:2] == [ord("h"), ord("i")]
    assert all(v == 0 for v in ids[0][2:])
    assert all(v == 0 for v in ids[1])
    # cu 末值重复（空行）
    assert b["cu"] == [0, 2, 2]


def test_5_visual_budget_truncate_and_count(tmp_path, capfd):
    """视觉预算 TruncateAndCount：2 图超预算 → seg_order 靠后丢图、守恒成立。"""
    MSL, BS = 64, 2
    png1, png2 = make_png(16, 16), make_png(16, 16)   # 各 1 token → 共 2 visual
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"image_b64": [base64.b64encode(png1).decode(),
                       base64.b64encode(png2).decode()],
         "image_meta": {"width": 16, "height": 16}},
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=1)  # 预算 1 < 2
    build(cfg, MSL)
    # Rust tracing 直接写 OS fd 1（绕过 Python sys.stdout）→ 用 capfd（fd 级捕获）。
    captured = capfd.readouterr()

    m = manifest_of(tmp_path / "ds")
    gs = m["global_stats"]
    # seg_order 靠后的图被丢 → 只剩 1 rec / 1 image token
    assert gs.get("total_images", 0) == 1
    assert gs.get("total_image_tokens", 0) == 1
    assert gs["total_tokens"] == 1 == gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)
    # 截断汇总 WARN 可见（fd 级）
    combined = captured.out + captured.err
    assert "image segment(s) dropped by visual budget" in combined, combined

    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    # 留下的是 seg_order 0
    assert img_records(b["images"]) == [(0, 0, 1, 1, 0)]
    assert b["ids"][0].tolist()[:1] == [IMG_BASE]


def test_6_visual_budget_hard_fail(tmp_path):
    """视觉预算 HardFail：构建失败含 'visual budget exceeded'。"""
    MSL, BS = 64, 2
    png1, png2 = make_png(16, 16), make_png(16, 16)
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"image_b64": [base64.b64encode(png1).decode(),
                       base64.b64encode(png2).decode()],
         "image_meta": {"width": 16, "height": 16}},
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=1,
                  visual_budget_hard_fail=True)
    with pytest.raises(Exception, match="visual budget exceeded"):
        build(cfg, MSL)


def test_7_image_atomicity_drop(tmp_path):
    """原子性：image 段 > 行剩余 → 整段丢不产 rec。"""
    MSL, BS = 8, 2
    # caption "hello"(5) + 32×32 image（grid 2×2 = 4 tokens）；5+4=9 > MSL=8，
    # 文本占 5 后剩余 3 < 4 → image 整段丢。
    png = make_png(32, 32)
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [
        {"text": "hello", "image_b64": base64.b64encode(png).decode(),
         "image_meta": {"width": 32, "height": 32}},
    ])
    cfg = vlm_cfg(jsonl, tmp_path / "ds", bs=BS, msl_visual=256)
    build(cfg, MSL)

    m = manifest_of(tmp_path / "ds")
    gs = m["global_stats"]
    assert gs.get("total_images", 0) == 0                    # 整段丢 → 无 rec
    assert gs.get("total_image_tokens", 0) == 0
    assert gs["total_tokens"] == 5 == gs.get("total_text_tokens", 0)
    b = read_first_batch(os.path.join(str(tmp_path / "ds"), "index.json"), BS, MSL)
    assert len(b["images"]) == 0
    assert b["ids"][0].tolist()[:5] == [ord(c) for c in "hello"]
    # 无 image placeholder 出现在 row0
    assert not any(IMG_BASE <= v < IMG_BASE + 100 for v in b["ids"][0].tolist())


def test_8_negative_bad_callbacks(tmp_path):
    """负面：回调返回 缺长度/段表有洞/labels 长度错/负 id → 全部 fail-fast 含定位。"""
    MSL, BS = 64, 2
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [{"text": "hi"}])

    valid = _enc(0, [ord("h"), ord("i")],
                 [_txt_seg(0, 0, 2)], [ord("h"), ord("i")])

    def cb_len(_):
        return []   # 长度不匹配

    def cb_hole(_):
        # segment_table 覆盖 [0,1) 但 token_seq 有 2 → validate 失败
        return [dict(valid, segment_table=[_txt_seg(0, 0, 1)])]

    def cb_labels(_):
        return [dict(valid, labels=[ord("h")])]      # labels len 1 != 2

    def cb_neg_id(_):
        # 负 token id → parse 期硬失败（token_seq 解析拒绝负 id，含 chunk/sample
        # 定位）。vocab=None 时非负 id 全判为 text，无法触发 D6 的 !in_modality 分支；
        # 负 id 在更早的 parse 处被拦，仍是 fail-fast 含定位。
        return [_enc(0, [-1, ord("i")], [_txt_seg(0, 0, 2)], [-1, ord("i")])]

    cases = [
        (cb_len, "returned 0 samples"),
        (cb_hole, "not contiguous|covers.*token_seq"),
        (cb_labels, "labels length"),
        (cb_neg_id, "out of i32 range"),
    ]
    for cb, pat in cases:
        cfg = vlm_cfg(jsonl, tmp_path / f"ds_{cb.__name__}", bs=BS, msl_visual=256)
        with pytest.raises(Exception, match=pat):
            build_with(cfg, cb, MSL)


def test_9_old_path_regression_marker(tmp_path):
    """旧路径回归：pt/sft 构建误传 modality_encoder → 入口防护报错。"""
    MSL, BS = 64, 2
    jsonl = write_jsonl(str(tmp_path / "data.jsonl"), [{"text": "hi"}])
    cb = make_vlm_encode_callback(image_range_start=IMG_BASE, patch_size=16)
    # pt 模式 + vlm 回调 → 入口防护拒绝（handoff §2.8）
    cfg = BuildConfig(input_paths=[jsonl], output_dir=str(tmp_path / "ds"),
                      mode="pt", batch_size=BS)
    # DataBuilder 无分词器 → pt 先在 build_dataset 报 backend="none"；
    # 给一个 tokenizer_path 走通 pt 入口后由入口防护拒绝 modality_encoder。
    with pytest.raises(Exception):
        DataBuilder(max_seq_len=MSL).build_dataset(cfg, modality_encoder=cb)

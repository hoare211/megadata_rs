"""P3a audio 流 e2e（`tests/test_mm_p3a_audio.py`，P3a 方案 §9 锚点）。

**不依赖分词器 / 网络 / PIL**：自定义 vlm 回调（identity 文本映射 + 合成音频
token），音频字节用伪 FLAC 载荷内联（demux 只透传压缩字节，不解码）。

覆盖锚点：
- A1 守恒：``Σ col_aud_cu.aud_cnt == col_aud_rec 记录数``；
  ``Σ aud_rec.len(flags==VALID) == col_aud_blob 长度``；尾窗 (offset,len) 复用
  同组首窗；三文件 sha256 == manifest hashes（增量哈希铁律）。
- A4 确定性：同参数两次构建，audio 主文件五件套 + col_aud_* 三件套逐字节相同。
- A5 变长性：audio 流"全数据集至多 1 个非满批"（全数据集口径）。
- A7 负面：timestamp / sample_rate / channels 缺失、start_ms 超界、超长窗 →
  fail-fast 含定位；duration_sec 缺失 → WARN + token 隐含时长回退（构建成功）。
- A8 混合样本：text/image/audio/图+音 混数据集 → Vlm 流与 audio 流各自守恒；
  Vlm 流 ``mm_cu.aud_cnt`` 恒 0；audio placeholder 不出现在 Vlm 流 input_ids；
  ``total_tokens == Σ 模态``。
- A9 切窗无重无丢：W=2000、段 [1500,5500) → 子段 25/100/75 token 首尾相接。
- A10 rec↔batch 归属：每条 rec 有唯一起始 batch（其 doc 首段所属），可由
  col_aud_cu 前缀和定位；**容忍 doc 跨 batch 切分**——延续段（同 uid）经 claimed
  去重不重复产 rec（小于 capacity 的 doc 在批次剩余空间不足时同样被切）。

契约注记（demux 的 audio 源查找）：pipeline 在 encode 前按**源侧**
``(sample_id, segment_order)`` 暂存 audio 字节/元数据；modality encoder 必须
**保持音频段的 segment_order 不变**（caption 前置的 Text 段取不冲突的新 order，
不得顺移既有媒体段 order——否则 demux 查找 miss 即 fail-fast）。
"""
import base64
import hashlib
import json
import os
import struct

import pytest

from megadata import BuildConfig, DataBuilder
from megadata.engine import RustStreamReader, _safe_from_dlpack

IMG_BASE = 32000
IMG_RANGE = (IMG_BASE, IMG_BASE + 100, "image")
AUD_BASE = 33000
AUD_RANGE = (AUD_BASE, AUD_BASE + 2000, "audio")

RATE = 50       # audio_window_token_rate（tok/s）
W_MS = 2000     # 对齐窗长（ms）
PATCH = 16


# ============================================================
# 辅助：载荷 / 行 / 回调
# ============================================================
def make_audio_payload(n, seed=0x66):
    """伪 FLAC 载荷（魔数 + 确定性填充；demux 只透传，不解析内容）。"""
    return b"fLaC" + bytes((i * 7 + seed) % 256 for i in range(max(0, n - 4)))


def audio_row(*, ts, dur=None, sr=16000, ch=1, payload_len=256, text=None):
    """audio-only JSONL 行（audio_b64 内联 + timestamp + audio_meta）。"""
    row = {
        "audio_b64": base64.b64encode(make_audio_payload(payload_len)).decode(),
        "audio_media_type": "audio/flac",
        "audio_meta": {"sample_rate": sr, "channels": ch},
    }
    if dur is not None:
        row["audio_meta"]["duration_sec"] = dur
    if ts is not None:
        row["timestamp"] = ts
    if text:
        row["text"] = text
    return row


def image_row(*, w=16, h=32, text=None):
    """单图 JSONL 行（内联 PNG 由回调侧合成，字节内容无关）。"""
    row = {"image_b64": base64.b64encode(bytes((i * 13) % 256 for i in range(64))).decode(),
           "image_meta": {"width": w, "height": h}}
    if text:
        row["text"] = text
    return row


def mixed_row(*, ts, dur, w=16, h=16, text=None, payload_len=128):
    """图+音同行（交错样本 [Image, Audio]；text 挂样本级 caption）。"""
    row = {
        "image_b64": base64.b64encode(bytes((i * 5) % 256 for i in range(48))).decode(),
        "image_meta": {"width": w, "height": h},
        "audio_b64": base64.b64encode(make_audio_payload(payload_len)).decode(),
        "audio_media_type": "audio/flac",
        "audio_meta": {"sample_rate": 16000, "channels": 1, "duration_sec": dur},
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


def make_audio_cb(*, audio_token_counts=None, rate=RATE, default_audio_tokens=100):
    """P3a 测试用 modality encoder 回调（自包含，不依赖参考 adapter）。

    - text 段：identity 映射（ord(c)）；
    - image 段：ceil(w/patch)×ceil(h/patch) 个 placeholder（id 取 IMG_BASE），
      media_refs 携字节（Vlm 流 sidecar）；
    - audio 段：``round(duration_sec × rate)`` 个 placeholder（id 取 AUD_BASE，
      缺 duration 时用 ``default_audio_tokens``）；**保持源侧 segment_order**
      （caption 前置的 Text 段取 ``order = 900 + k`` 不冲突新值）。
    ``audio_token_counts``：{sample_id: n} 显式覆盖 token 数（负面用例）。
    """
    overrides = audio_token_counts or {}

    def cb(samples):
        out = []
        for s in samples:
            sid = s["sample_id"]
            meta = s.get("media_meta") or {}
            segs_in = list(s.get("segments") or [])
            caption = s.get("text")
            # caption 前置为 Text 段（order 取不冲突新值，媒体段 order 原样保持）
            prefix = []
            if caption and not (segs_in and segs_in[0].get("kind") == "text"):
                prefix = [{"kind": "text", "order": 900, "text": caption}]
            seq, table, labels, refs = [], [], [], []

            def push_text(t, order):
                ids = [ord(c) for c in t]
                start = len(seq)
                seq.extend(ids)
                labels.extend(ids)
                table.append({"kind": "text", "order": order, "token_start": start,
                              "token_end": start + len(ids), "media_record_id": None,
                              "loss_weight": 255})

            for g in prefix:
                push_text(g["text"], g["order"])
            for g in segs_in:
                kind, order = g["kind"], g["order"]
                start = len(seq)
                if kind == "text":
                    push_text(g.get("text") or "", order)
                elif kind == "image":
                    w = int(meta.get("width") or 16)
                    h = int(meta.get("height") or 16)
                    gh, gw = -(-w // PATCH), -(-h // PATCH)
                    n = gh * gw
                    seq.extend(IMG_BASE + (i % 100) for i in range(n))
                    labels.extend(-100 for _ in range(n))
                    table.append({"kind": "image", "order": order, "token_start": start,
                                  "token_end": start + n, "media_record_id": None,
                                  "loss_weight": 0})
                    refs.append({"kind": "image", "seg_order": order,
                                 "bytes": g.get("bytes"), "grid_h": gh, "grid_w": gw,
                                 "flags": 0})
                elif kind == "audio":
                    dur = meta.get("duration_sec")
                    n = overrides.get(sid)
                    if n is None:
                        n = int(round(dur * rate)) if dur else default_audio_tokens
                    seq.extend(AUD_BASE + (i % 2000) for i in range(n))
                    labels.extend(-100 for _ in range(n))
                    table.append({"kind": "audio", "order": order, "token_start": start,
                                  "token_end": start + n, "media_record_id": None,
                                  "loss_weight": 0})
                else:
                    raise AssertionError(f"unexpected segment kind {kind!r}")
            out.append({
                "sample_id": sid,
                "token_seq": seq,
                "segment_table": table,
                "labels": labels,
                "loss_weights": None,
                "stats": {"text_tokens": sum(1 for e in table if e["kind"] == "text"),
                          "image_tokens": sum(e["token_end"] - e["token_start"]
                                              for e in table if e["kind"] == "image"),
                          "audio_tokens": sum(e["token_end"] - e["token_start"]
                                              for e in table if e["kind"] == "audio"),
                          "video_tokens": 0,
                          "images": sum(1 for e in table if e["kind"] == "image"),
                          "audio_clips": sum(1 for e in table if e["kind"] == "audio"),
                          "video_clips": 0},
                "media_refs": refs,
            })
        return out

    cb.__doc__ = "P3a test modality encoder (identity text / synthetic audio tokens)"
    return cb


def audio_cfg(jsonl, out_dir, *, bs=2, msl=256, rate=RATE, w_ms=W_MS, **kw):
    """vlm BuildConfig（image+audio 双区间；audio 流参数按需）。"""
    return BuildConfig(
        input_paths=[jsonl], output_dir=str(out_dir), mode="vlm",
        batch_size=bs, modality_id_ranges=[IMG_RANGE, AUD_RANGE],
        w_ms=w_ms, audio_window_token_rate=rate, **kw,
    )


def build_audio(cfg, msl, cb=None, rate=RATE):
    """构建（回调 rate 必须与配置 audio_window_token_rate 一致——回调据此决定
    audio token 数，pipeline 据此切窗；不一致会触发超长窗 fail-fast）。"""
    return DataBuilder(max_seq_len=msl).build_dataset(
        cfg, modality_encoder=cb or make_audio_cb(rate=rate))


def manifest_of(out_dir):
    with open(os.path.join(str(out_dir), "index.json")) as f:
        return json.load(f)


# ============================================================
# 辅助：audio 流文件解析（ashard_ 文件族）
# ============================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_aud_cu(data):
    """col_aud_cu：16B/batch = n_rec u32 + aud_cnt u32 + rsv u64（格式冻结 §2.5）。"""
    assert len(data) % 16 == 0, f"col_aud_cu length {len(data)} not a multiple of 16"
    out = []
    for i in range(len(data) // 16):
        n_rec, aud_cnt, rsv = struct.unpack_from("<IIQ", data, i * 16)
        out.append({"n_rec": n_rec, "aud_cnt": aud_cnt, "rsv": rsv})
    return out


def parse_aud_rec(data):
    """col_aud_rec：40B/条（格式冻结 §2.2）。"""
    assert len(data) % 40 == 0, f"col_aud_rec length {len(data)} not a multiple of 40"
    out = []
    for i in range(len(data) // 40):
        off, ln, sr, dur, ch, flags, rsv, sid, seg, start = struct.unpack_from(
            "<QIIIHBBQII", data, i * 40)
        out.append({"offset": off, "len": ln, "sr": sr, "duration_ms": dur,
                    "channels": ch, "flags": flags, "rsv": rsv, "sample_id": sid,
                    "seg_order": seg, "start_ms": start})
    return out


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


def audio_files(out_dir, manifest):
    """每个 audio shard 的文件路径 + 解析结果（主文件 + sidecar 三件套）。"""
    shards = []
    for sm in manifest["audio_shards"]:
        stem = sm["filename"][: -len(".aud_bin")]
        d = str(out_dir)
        shards.append({
            "meta": sm,
            "stem": stem,
            "aud_cu": parse_cu_file(read_bytes(f"{d}/{stem}.aud_cu")),
            "doc_ids": list(struct.unpack(f"<{len(read_bytes(f'{d}/{stem}.aud_doc_ids'))//4}i",
                                          read_bytes(f"{d}/{stem}.aud_doc_ids"))),
            "aud_cu_counts": parse_aud_cu(read_bytes(f"{d}/{stem}.col_aud_cu")),
            "aud_rec": parse_aud_rec(read_bytes(f"{d}/{stem}.col_aud_rec")),
            "blob": read_bytes(f"{d}/{stem}.col_aud_blob"),
        })
    return shards


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def valid_tokens_per_batch(aud_cu_batches, capacity):
    """每 batch 的有效 token 数 = cu_seqlens 末值（不含 eos padding）。"""
    return [cu[-1] for cu in aud_cu_batches]


# ============================================================
# A1：守恒（aud_cu 计数 / VALID 过滤求和 / 尾窗引用 / 增量哈希）
# ============================================================
def test_a1_conservation_and_incremental_hashes(tmp_path):
    rows = [
        audio_row(ts=1.5, dur=4.0),                      # sid 0：[1500,5500) → 3 窗
        audio_row(ts=0.0, dur=2.0),                      # sid 1：单窗
        audio_row(ts=0.5, dur=6.0),                      # sid 2：[500,6500) → 4 窗
    ]
    jsonl = write_jsonl(tmp_path / "aud.jsonl", rows)
    out = tmp_path / "out_a1"
    build_audio(audio_cfg(jsonl, out), msl=256)
    man = manifest_of(out)
    assert man["audio_shards"], "audio stream must produce shards"

    shards = audio_files(out, man)
    recs = [r for s in shards for r in s["aud_rec"]]
    cu_counts = [c for s in shards for c in s["aud_cu_counts"]]
    blob = b"".join(s["blob"] for s in shards)

    # Σ col_aud_cu.aud_cnt == col_aud_rec 记录数（§9.7）
    assert sum(c["aud_cnt"] for c in cu_counts) == len(recs)
    for c in cu_counts:
        assert c["n_rec"] == 1 and c["rsv"] == 0
    # 边界切分回归锚点：本数据集 608 token > capacity 512，GreedyPacker 必然在
    # batch 边界切分某个窗口 doc（PT 语义）——配对仍守恒（rec 归首段 batch）
    assert len(cu_counts) == 2, "608 token / capacity 512 ⇒ 2 个 audio batch（含边界切分）"
    # Σ aud_rec.len(flags==VALID) == blob 长度（§9.8；尾窗 len 复用首窗，不计入）
    valid_sum = sum(r["len"] for r in recs if r["flags"] == 0)
    assert valid_sum == len(blob), "VALID 过滤求和必须等于 blob 长度"
    # 尾窗引用一致性：同 (sample_id, seg_order) 组内尾窗 (offset,len) == 首窗
    first = {}
    for r in recs:
        key = (r["sample_id"], r["seg_order"])
        if r["flags"] == 0:
            first[key] = (r["offset"], r["len"])
        else:
            assert r["flags"] == 4, f"audio flags must be 0/4, got {r['flags']}"
            assert (r["offset"], r["len"]) == first[key], "尾窗必须复用同组首窗 (offset,len)"
    # flags=0 计数 == total_audio_clips（尾窗不计）
    assert sum(1 for r in recs if r["flags"] == 0) == man["global_stats"]["total_audio_clips"]
    # 铁律：三文件 sha256 == manifest hashes
    for s in shards:
        for suffix, key in (("col_aud_blob", "sha256:col_aud_blob"),
                            ("col_aud_rec", "sha256:col_aud_rec"),
                            ("col_aud_cu", "sha256:col_aud_cu")):
            assert s["meta"]["hashes"][key] == sha256_file(
                os.path.join(str(out), f"{s['stem']}.{suffix}")), f"{suffix} 增量哈希 != 回读"
        for suffix, key in (("aud_bin", "sha256:aud_bin"), ("aud_idx", "sha256:aud_idx"),
                            ("aud_cu", "sha256:aud_cu"), ("aud_doc_ids", "sha256:aud_doc_ids"),
                            ("aud_loss_weights", "sha256:aud_loss_weights")):
            assert s["meta"]["hashes"][key] == sha256_file(
                os.path.join(str(out), f"{s['stem']}.{suffix}")), f"{suffix} 增量哈希 != 回读"


# ============================================================
# A4：确定性（两次构建逐字节相同）
# ============================================================
def test_a4_determinism_byte_identical(tmp_path):
    rows = [audio_row(ts=1.5, dur=4.0), audio_row(ts=0.0, dur=2.0), mixed_row(ts=0.25, dur=3.0)]
    jsonl = write_jsonl(tmp_path / "aud.jsonl", rows)
    outs = []
    for i in range(2):
        out = tmp_path / f"out_a4_{i}"
        build_audio(audio_cfg(jsonl, out), msl=256)
        outs.append(out)
    man_a, man_b = manifest_of(outs[0]), manifest_of(outs[1])
    assert [s["filename"] for s in man_a["audio_shards"]] == [s["filename"] for s in man_b["audio_shards"]]
    for sa, sb in zip(man_a["audio_shards"], man_b["audio_shards"]):
        assert sa["hashes"] == sb["hashes"], "同参数两次构建 audio shard 哈希必须一致"
        for suffix in ("aud_bin", "aud_idx", "aud_cu", "aud_doc_ids", "aud_loss_weights",
                       "col_aud_blob", "col_aud_rec", "col_aud_cu"):
            stem = sa["filename"][: -len(".aud_bin")]
            pa = os.path.join(str(outs[0]), f"{stem}.{suffix}")
            pb = os.path.join(str(outs[1]), f"{stem}.{suffix}")
            assert read_bytes(pa) == read_bytes(pb), f"{suffix} 必须逐字节相同"


# ============================================================
# A5：变长性（全数据集至多 1 个非满批）
# ============================================================
def test_a5_at_most_one_non_full_batch(tmp_path):
    # audio msl=64 / bs=2 → capacity 128；rate=50 → 每窗 ≤ 100 token + eos
    rows = [audio_row(ts=0.0, dur=float(2 + i)) for i in range(6)]
    jsonl = write_jsonl(tmp_path / "aud.jsonl", rows)
    out = tmp_path / "out_a5"
    build_audio(audio_cfg(jsonl, out, bs=2, audio_max_seq_len=64, audio_batch_size=2), msl=256)
    man = manifest_of(out)
    shards = audio_files(out, man)
    capacity = 64 * 2
    valids = [v for s in shards for v in valid_tokens_per_batch(s["aud_cu"], capacity)]
    assert valids, "audio stream must have batches"
    non_full = [v for v in valids if v < capacity]
    assert len(non_full) <= 1, f"全数据集至多 1 个非满批，实测 {len(non_full)}: {valids}"
    # 除末批外全部满载（GreedyPacker push 阶段只出满批）
    assert all(v == capacity for v in valids[:-1]), f"非末批必须满载: {valids}"


# ============================================================
# A7：负面 fail-fast（timestamp / sr / channels 缺失、start_ms 超界、超长窗）
# ============================================================
@pytest.mark.parametrize("row,match", [
    # timestamp 缺失 → INV-2 硬校验
    (audio_row(ts=None, dur=2.0), "no timestamp"),
    # sample_rate 缺失（N-1 定稿裁定）
    ({"audio_b64": base64.b64encode(make_audio_payload(64)).decode(),
      "audio_meta": {"channels": 1, "duration_sec": 2.0}, "timestamp": 0.0}, "no sample_rate"),
    # channels 缺失
    ({"audio_b64": base64.b64encode(make_audio_payload(64)).decode(),
      "audio_meta": {"sample_rate": 16000, "duration_sec": 2.0}, "timestamp": 0.0}, "no channels"),
    # start_ms 超界（timestamp=5.0 但 duration=1.0 ⇒ 5000 > T×1000=1000）
    (audio_row(ts=5.0, dur=1.0), "INV-2 hard"),
])
def test_a7_negative_fail_fast(tmp_path, row, match):
    jsonl = write_jsonl(tmp_path / "bad.jsonl", [row])
    out = tmp_path / "out_a7"
    with pytest.raises(Exception, match=match):
        build_audio(audio_cfg(jsonl, out), msl=256)


def test_a7_oversized_window_fail_fast(tmp_path):
    # capacity=128（msl=128/bs=1）：准入 100+1=101 ≤ 128 通过；但 encoder 对
    # 1.0s 的段产 200 token（> 127）→ 单窗超限 fail-fast
    jsonl = write_jsonl(tmp_path / "big.jsonl", [audio_row(ts=0.0, dur=1.0)])
    out = tmp_path / "out_a7_big"
    cb = make_audio_cb(audio_token_counts={0: 200})
    with pytest.raises(Exception, match="oversized audio window"):
        build_audio(audio_cfg(jsonl, out, audio_max_seq_len=128, audio_batch_size=1), msl=256, cb=cb)


def test_a7_admission_config_fail_fast(tmp_path):
    # 准入条件：W×rate/1000 + 1 > capacity → 配置错误 fail-fast
    jsonl = write_jsonl(tmp_path / "adm.jsonl", [audio_row(ts=0.0, dur=2.0)])
    out = tmp_path / "out_a7_adm"
    with pytest.raises(Exception, match="exceeds stream capacity"):
        build_audio(audio_cfg(jsonl, out, audio_max_seq_len=8, audio_batch_size=1), msl=256)


def test_a7_duration_missing_falls_back_to_token_implied(tmp_path):
    # duration_sec 缺失 → token 隐含时长回退（100 token @50/s = 2000ms = 1 窗）
    jsonl = write_jsonl(tmp_path / "nodur.jsonl", [audio_row(ts=0.0, dur=None)])
    out = tmp_path / "out_a7_nodur"
    build_audio(audio_cfg(jsonl, out), msl=256)  # 不得报错
    man = manifest_of(out)
    shards = audio_files(out, man)
    recs = [r for s in shards for r in s["aud_rec"]]
    assert len(recs) == 1 and recs[0]["start_ms"] == 0 and recs[0]["duration_ms"] == 2000
    # token 守恒：单 doc = 100 token + eos
    cu = [cu for s in shards for cu in s["aud_cu"]]
    doc_lens = [cu[i + 1] - cu[i] for cu in cu for i in range(len(cu) - 1)]
    # eos 每 doc 一次（切分段不重复计）：Σ 段长 − doc 数 == 源 token 数
    assert sum(doc_lens) - len(recs) == 100, "token 隐含时长回退后 token 守恒"


# ============================================================
# A8：混合样本（text/image/audio/图+音）两流各自守恒
# ============================================================
def test_a8_mixed_dataset_two_streams(tmp_path):
    rows = [
        {"text": "plain text row"},                     # sid 0：纯文本 → Vlm 流
        image_row(text="an image", w=16, h=32),         # sid 1：单图+caption → Vlm 流
        audio_row(ts=1.5, dur=4.0),                     # sid 2：纯音频 → audio 流
        mixed_row(ts=0.25, dur=2.0, text="mixed"),      # sid 3：图+音 → 两流
        audio_row(ts=0.0, dur=1.0),                     # sid 4：纯音频 → audio 流
    ]
    jsonl = write_jsonl(tmp_path / "mix.jsonl", rows)
    out = tmp_path / "out_a8"
    build_audio(audio_cfg(jsonl, out), msl=256)
    man = manifest_of(out)
    gs = man["global_stats"]

    # ---- manifest：两流统计 + 守恒 ----
    assert gs["stats_version"] == 2
    # 为 0 的模态字段序列化省略（skip_serializing_if）——按 0 取
    assert gs["total_tokens"] == (gs.get("total_text_tokens", 0) + gs.get("total_image_tokens", 0)
                                  + gs.get("total_audio_tokens", 0) + gs.get("total_video_tokens", 0))
    assert gs["total_audio_tokens"] > 0 and gs["total_audio_clips"] > 0
    assert gs["total_image_tokens"] > 0 and gs["total_images"] > 0
    assert gs["total_text_tokens"] > 0
    # audio 统计口径：clips = VALID rec 数；tokens = audio 流有效 token
    shards = audio_files(out, man)
    recs = [r for s in shards for r in s["aud_rec"]]
    assert gs["total_audio_clips"] == sum(1 for r in recs if r["flags"] == 0)
    # align_fingerprint 最小集（P3a §6.3）
    fp = man["align_fingerprint"]
    assert fp["W_ms"] == W_MS and fp["granularity"] == "G2"
    assert fp["audio"]["window_token_rate"] == RATE
    assert "audio/flac" in fp["audio"]["codec"]
    # schema 键（格式冻结 §5）
    for k, v in (("aud_blob", "bytes"), ("aud_rec", "audrec40"), ("aud_cu", "audcu"),
                 ("aud_doc_ids", "list<int32>"), ("aud_loss_weights", "list<f16>")):
        assert man["schema"][k] == v

    # ---- Vlm 流：mm_cu.aud_cnt 恒 0；audio placeholder 不进 input_ids ----
    for sm in man["shards"]:
        stem = sm["filename"][: -len(".bin")]
        mm_cu = read_bytes(os.path.join(str(out), f"{stem}.mm_cu"))
        assert len(mm_cu) % 16 == 0
        for i in range(len(mm_cu) // 16):
            n_mod, img_cnt, aud_cnt, vid_cnt = struct.unpack_from("<IIII", mm_cu, i * 16)
            assert n_mod == 3
            assert aud_cnt == 0, "Vlm 流 mm_cu.aud_cnt 恒 0（audio 走独立流）"
            assert vid_cnt == 0
        assert img_cnt >= 0
    reader = RustStreamReader(os.path.join(str(out), "index.json"), 2, 256, 1, 0, 42)
    seen_img = seen_txt = 0
    while True:
        d = reader.next_batch_capsules()
        if d is None:
            break
        ids = _safe_from_dlpack(d["input_ids"]).reshape(-1).tolist()
        for t in ids:
            assert not (AUD_BASE <= t < AUD_BASE + 2000), \
                f"audio placeholder {t} leaked into the Vlm stream (demux must remove audio)"
            if IMG_BASE <= t < IMG_BASE + 100:
                seen_img += 1
            elif t < 32000:
                seen_txt += 1
    assert seen_img > 0 and seen_txt > 0, "Vlm 流应同时含 image 与 text token"

    # ---- audio 流：rec 的 sample_id 覆盖两个纯音频样本 + 混合样本 ----
    audio_sids = {r["sample_id"] for r in recs}
    assert audio_sids == {2, 3, 4}, f"audio rec 应覆盖 sid 2/3/4，实测 {audio_sids}"
    # 图+音样本（sid 3）的 audio 段：start_ms = round(0.25×1000) = 250
    sid3 = [r for r in recs if r["sample_id"] == 3]
    assert sid3 and sid3[0]["start_ms"] == 250


# ============================================================
# A9：切窗无重无丢（样本时间轴锚定反例：W=2000、段 [1500,5500)）
# ============================================================
def test_a9_window_split_no_loss_no_dup(tmp_path):
    jsonl = write_jsonl(tmp_path / "w.jsonl", [audio_row(ts=1.5, dur=4.0)])
    out = tmp_path / "out_a9"
    # capacity=1024（msl=256/bs=4）⇒ 3 个 doc 同批（无边界切分干扰）
    build_audio(audio_cfg(jsonl, out, audio_max_seq_len=256, audio_batch_size=4), msl=256)
    man = manifest_of(out)
    shards = audio_files(out, man)
    recs = [r for s in shards for r in s["aud_rec"]]
    # 3 个子段：[1500,2000)/[2000,4000)/[4000,5500)，各属 w0/w1/w2
    assert [(r["start_ms"], r["duration_ms"]) for r in recs] == [
        (1500, 500), (2000, 2000), (4000, 1500)]
    assert [r["start_ms"] // W_MS for r in recs] == [0, 1, 2], "每子段恒属单窗"
    assert [r["flags"] for r in recs] == [0, 4, 4], "首窗 VALID + 两尾窗引用"
    # token 区间首尾相接：25/100/75 = 200（round(4.0s × 50/s)）
    cu = [cu for s in shards for cu in s["aud_cu"]]
    doc_lens = [cu[i + 1] - cu[i] for cu in cu for i in range(len(cu) - 1)]
    assert [d - 1 for d in doc_lens] == [25, 100, 75], "doc token 数（去 eos）"
    assert sum(doc_lens) - len(recs) == 200, "拼接全部子段 == 原 audio token 序列"


# ============================================================
# A10：rec↔batch 归属（col_aud_cu 前缀和定位；切分容忍）
# ============================================================
def test_a10_rec_batch_attribution(tmp_path):
    # rate=10 → 每窗 20 token + eos = 21；capacity=32（msl=32/bs=1）。
    # 注：GreedyPacker PT 语义会在 batch 边界切分 doc（"一行不动"下无法避免），
    # 小于 capacity 的 doc 落在剩余空间不足时同样被切——rec 归属裁定为"doc 首段
    # 所在 batch"（P3a §5.4 修订）：每条 rec 有唯一起始 batch，前缀和寻址严格成立。
    jsonl = write_jsonl(tmp_path / "long.jsonl", [audio_row(ts=0.0, dur=20.0)])
    out = tmp_path / "out_a10"
    build_audio(audio_cfg(jsonl, out, rate=10, audio_max_seq_len=32, audio_batch_size=1),
                msl=256, rate=10)
    man = manifest_of(out)
    shards = audio_files(out, man)
    cu_counts = [c for s in shards for c in s["aud_cu_counts"]]
    recs = [r for s in shards for r in s["aud_rec"]]
    assert len(recs) == 10, "10 个窗口 ⇒ 10 条 rec（每窗一条）"
    # 守恒：Σ aud_cu.aud_cnt == rec 数（每条 rec 恰出现一次——切分不重复产 rec）
    assert sum(c["aud_cnt"] for c in cu_counts) == len(recs)
    # 前缀和定位：rec 按 doc 序归批（batch_of_rec 单调不减），且无漏无重
    batch_of_rec, prefix = [], 0
    for c in cu_counts:
        batch_of_rec.extend([prefix] * c["aud_cnt"])
        prefix += 1
    assert len(batch_of_rec) == 10
    assert batch_of_rec == sorted(batch_of_rec), "rec 必须按 doc 序归批（前缀和定位）"
    # 每窗 start_ms = k × 2000（样本时间轴锚定，段起点 0）
    assert [r["start_ms"] for r in recs] == [k * 2000 for k in range(10)]
    # doc token 守恒：Σ 各 doc（去 eos）== 200（10 窗 × 20 token）
    doc_lens = [cu[i + 1] - cu[i] for s in shards for cu in s["aud_cu"]
                for i in range(len(cu) - 1)]
    # 切分会增加段数（> doc 数）；eos 每 doc 一次 ⇒ Σ 段长 − doc 数 == 源 token 数
    assert sum(doc_lens) - len(recs) == 200


# ============================================================
# P2-1（20261031 评审）：长音频窗口组跨 shard 原子性
# ============================================================
def test_p2_1_window_group_stays_in_one_shard(tmp_path):
    """`audio_batches_per_shard=1` + 单段多窗口：同段首尾窗必须在同一 shard。

    旧实现按 batch 数立即 rollover：每个窗口一个 batch ⇒ 每个 shard 一个窗口，
    新 shard 的 `AudioSidecarWriter.seg_ref` 为空，尾窗报
    `has no VALID record in its group`，合法长音频构建失败（P2-1）。
    修复后 rollover 延后到窗口组收尾，构建成功且全部 rec 落在一个 shard。
    """
    # rate=50、W=2000ms：段 [500,6500) → 3 个窗口（各 100 token）
    jsonl = write_jsonl(tmp_path / "p21.jsonl", [audio_row(ts=0.5, dur=6.0)])
    out = tmp_path / "out_p21"
    build_audio(
        audio_cfg(jsonl, out, bs=1, audio_max_seq_len=256,
                  audio_batch_size=1, audio_batches_per_shard=1),
        msl=256,
    )
    man = manifest_of(out)
    assert len(man["audio_shards"]) == 1, (
        "P2-1: 同段窗口组不得跨 shard（rollover 必须延后到组收尾）")
    s = audio_files(out, man)[0]
    # [500,6500) 跨 4 个对齐窗（w0 部分 + w1/w2 完整 + w3 部分），每窗一条 rec
    recs = s["aud_rec"]
    assert len(recs) == 4, f"4 个窗口 ⇒ 4 条 rec，got {len(recs)}"
    assert len(s["aud_cu_counts"]) >= 2, "4 窗 token 超过单批容量 ⇒ 至少 2 个 audio batch"
    assert [r["flags"] for r in recs] == [0, 4, 4, 4], "首窗 VALID + 三尾窗引用"
    first = (recs[0]["offset"], recs[0]["len"])
    assert all((r["offset"], r["len"]) == first for r in recs[1:]), (
        "尾窗 (offset,len) 必须复用同 shard 内首窗（P2-1 跨 shard 即失效）")

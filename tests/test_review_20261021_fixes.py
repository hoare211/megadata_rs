# -*- coding: utf-8 -*-
"""20261021 代码评审修复回归：P3c 对齐屏障死锁（P3c-1 / P3a-1，阻断级）。

背景（评审 `20261021-多模态P3a-P3b-P3c-P4-代码评审.md` §P3c-1）
============================================================
`align_aud_range` 由 Vlm 写线程在 **shard 收尾**时快照 audio 写游标。旧实现用
**sid 水位**（`aud_max_sid_written >= shard_max_sid`）等待 audio 写线程，在三线程
（compute / audio writer / Vlm writer）之间构成环等：

- 纯 video / 尾部非 audio 构建中，收尾 shard 的 `shard_max_sid` 由**无 audio 的
  样本**抬升；
- audio 写线程阻塞在 `recv`（没有更新的 audio batch ⇒ sid 不前进、done 也不置位）；
- Vlm 写线程 spin 期间不消费 `tx_packed`，compute 被 `bounded(100)` 反压卡住
  ⇒ audio 通道永不关闭 ⇒ 死锁。

现修法（P3c-1 屏障，评审已核实通过）= **batch 水位**：compute 单线程维护
`audio_batches_sent`，随 Vlm batch 携带；audio 写线程每写完一批 `aud_batches_written
+= 1`；shard 收尾等 `aud_batches_written >= shard_max_aud_batches || aud_stream_done`。

**VR-2 设计修正（20261022 复审）**：`align_aud_range` 已从"逐 shard 游标快照"改为
"audio 流整体收尾后据 per-batch span + 每 shard 样本 id 区间二分构造性导出"
（`finalize_align_aud_ranges`）。逐 shard 水位屏障随之移除——Vlm 写线程不再在主循环
内等待 audio，只在收尾后等一次 `aud_stream_done`，死锁面进一步收窄。本文件三条死锁
回归继续作为"构建不挂死"的锚点（现机制下更不可能死锁）。

本文件锁死三个曾经死锁的形态（评审 §10.3 P0 三条）
==================================================
1. 纯 video 多 shard：只声明 video id 区间即触发 `align_enabled`，全为无 audio 样本；
2. 反压：Vlm batch 数 > `tx_packed` 容量 100（每 batch 一 shard，旧码逐 shard 等屏障）；
3. 尾部非 audio：audio 样本之后跟 video/text，收尾 shard 的最大样本无 audio rec。

工程约束
========
- **所有构建都在子进程里跑并带硬超时**：死锁回归若在同一进程内表现为永久挂起，
  CI 只会挂死而不是失败（评审明确要求超时断言）。超时 → terminate/kill + fail。
- 另断言"无漏覆盖"：每个 Vlm shard 的 `align_aud_range` 必须覆盖本 shard 样本的
  全部 audio rec（`av_align.audio_rec_range` 的 `aud_batch` 落在区间内）。VR-2 后这是
  **构造性保证**（区间 = 本 shard 样本 rec 所在的最小/最大 batch），不再是弱一致
  快照；无 audio rec 的 shard 精确判空为 None。
- 不依赖分词器 / 网络 / PIL / 真实音视频解码器（adapter 恒等映射 + 伪载荷）。
"""
from __future__ import annotations

import base64
import json
import multiprocessing
import os
import struct

import pytest

from megadata.engine import RustStreamReader

# 死锁回归的硬超时（秒）：正常构建 < 10s，留足余量后仍能在 CI 内失败而非挂死
DEADLOCK_TIMEOUT = 120

IMG_BASE = 32000
IMG_RANGE = (IMG_BASE, IMG_BASE + 100, "image")
AUD_BASE = 33000
AUD_RANGE = (AUD_BASE, AUD_BASE + 1000, "audio")
VID_BASE = 35000
VID_RANGE = (VID_BASE, VID_BASE + 4096, "video")

RATE = 50          # audio_window_token_rate（tokens/s）
W_MS = 2000
VFRAMES = 4
VRESIZE = (32, 32)
TOKENS_PER_CLIP = (VRESIZE[0] // 16) * (VRESIZE[1] // 16) * VFRAMES  # 16

PATCH = 16
NO_SAMPLE_SID = (1 << 64) - 1


# ===========================================================================
# 载荷构造（零解码：伪 FLAC / mp4 头 + 定长字节）
# ===========================================================================
def make_audio_payload(n, seed=0x66):
    return b"fLaC" + bytes((i * 7 + seed) % 256 for i in range(max(0, n - 4)))


def make_video_payload(n, seed=0x00):
    return b"\x00\x00\x00\x18ftypmp42" + bytes(
        (i * 3 + seed) % 256 for i in range(max(0, n - 12))
    )


def frame_fn(payload, indices):
    return [bytes(((i * 7 + 11) % 256) for _ in range(64)) for i in indices]


def audio_row(*, ts, dur=2.0, text=None, payload_len=256):
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


def video_row(*, ts, dur=1.0, text=None, payload_len=512):
    row = {
        "video_b64": base64.b64encode(make_video_payload(payload_len)).decode(),
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


# ===========================================================================
# 子进程构建 + 硬超时
# ===========================================================================
def _child_build(spec):
    """子进程入口：只做一次 build_dataset（异常 → 非 0 退出码 + traceback 到 stderr）。"""
    from megadata import BuildConfig, DataBuilder
    from megadata.modality_adapter import make_vlm_encode_callback

    cb = make_vlm_encode_callback(
        image_range_start=spec["img_base"],
        audio_range_start=spec["aud_base"] if spec["with_audio"] else None,
        audio_token_rate=spec["rate"],
        video_range_start=spec["vid_base"],
        video_n_frames=spec["vframes"],
        video_resize=tuple(spec["vresize"]),
        frame_fn=frame_fn,
    )
    cfg = BuildConfig(
        input_paths=[spec["jsonl"]],
        output_dir=spec["out"],
        mode="vlm",
        batch_size=spec["bs"],
        batches_per_shard=spec["bps"],
        modality_id_ranges=[tuple(r) for r in spec["ranges"]],
        max_visual_tokens_per_sample=spec["msl_visual"],
        w_ms=spec["w_ms"],
        audio_window_token_rate=spec["rate"],
        audio_batch_size=spec.get("audio_bs"),
        audio_max_seq_len=spec.get("audio_msl"),
        video_n_frames=spec["vframes"],
        video_resize=tuple(spec["vresize"]),
    )
    DataBuilder(max_seq_len=spec["msl"]).build_dataset(cfg, modality_encoder=cb)


def build_or_deadlock_fail(spec, label):
    """在 spawn 子进程里跑构建；超时即判定屏障死锁（终止子进程后 fail）。

    用 ``spawn`` 而非 ``fork``：父 pytest 进程已加载 torch/numpy（多线程），
    ``fork`` 会触发 DeprecationWarning 且可能在子进程内死锁（评审要求死锁回归
    表现为失败而非挂死，子进程自身不能因 fork 抢锁而误伤）。
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_child_build, args=(spec,))
    proc.start()
    proc.join(DEADLOCK_TIMEOUT)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        pytest.fail(
            f"{label}: 构建超过 {DEADLOCK_TIMEOUT}s 未完成 —— P3c 对齐屏障死锁回归"
            "（compute / audio writer / Vlm writer 环等）。已强制终止子进程。"
        )
    assert proc.exitcode == 0, f"{label}: 子进程构建失败（exitcode={proc.exitcode}），见上方 stderr"


def spec_of(tmp_path, rows, name, *, ranges, bs=1, bps=1, msl=256, msl_visual=None,
            with_audio=True, audio_bs=None, audio_msl=None):
    jsonl = write_jsonl(tmp_path / f"{name}.jsonl", rows)
    out = tmp_path / name
    return {
        "jsonl": jsonl,
        "out": str(out),
        "bs": bs,
        "bps": bps,
        "msl": msl,
        "msl_visual": msl if msl_visual is None else msl_visual,
        "ranges": [list(r) for r in ranges],
        "w_ms": W_MS,
        "rate": RATE,
        "vframes": VFRAMES,
        "vresize": list(VRESIZE),
        "img_base": IMG_BASE,
        "aud_base": AUD_BASE,
        "vid_base": VID_BASE,
        "with_audio": with_audio,
        # VR-2：audio 流独立装箱参数（None = 同 Vlm 流），用于构造确定性 flush 批
        "audio_bs": audio_bs,
        "audio_msl": audio_msl,
        "out_dir": str(out),
    }


# ===========================================================================
# 读回辅助：manifest / av_align / col_row_sids / audio 坐标换算
# ===========================================================================
def manifest_of(out_dir):
    with open(os.path.join(out_dir, "index.json")) as f:
        return json.load(f)


def iter_align_batches(out_dir, bs, msl):
    reader = RustStreamReader(os.path.join(out_dir, "index.json"), bs, msl, 1, 0, 42,
                              align=True)
    while True:
        d = reader.next_batch_capsules()
        if d is None:
            return
        yield d


def read_row_sids(out_dir, shard_meta):
    """读 `.col_row_sids`（u64 LE 每行一个 sid）。文件缺失 ⇒ 断言失败。"""
    path = os.path.join(out_dir, shard_meta["filename"][: -len(".bin")] + ".col_row_sids")
    assert os.path.exists(path), f"对齐启用必须落 col_row_sids，缺失：{path}"
    raw = open(path, "rb").read()
    assert len(raw) % 8 == 0
    return [struct.unpack_from("<Q", raw, i)[0] for i in range(0, len(raw), 8)]


def global_aud_batch(man, aud_shard, aud_batch_in_shard):
    """audio 流 (shard, batch_in_shard) → 全局 batch 序（含 `aud_batches_written` 语义）。"""
    total = 0
    for i, sm in enumerate(man.get("audio_shards") or []):
        if i == aud_shard:
            assert 0 <= aud_batch_in_shard < sm["num_batches"], (
                f"audio 坐标越界：shard {aud_shard} batch {aud_batch_in_shard} "
                f">= num_batches {sm['num_batches']}"
            )
            return total + aud_batch_in_shard
        total += sm["num_batches"]
    raise AssertionError(f"audio shard {aud_shard} 不存在（audio_shards={man.get('audio_shards')}）")


def assert_no_audio_leak(man, out_dir, entries):
    """每个 Vlm shard 的 `align_aud_range` 必须覆盖本 shard 样本的全部 audio rec。"""
    by_sid = {}
    for e in entries:
        by_sid.setdefault(e["sample_id"], []).extend(e["audio_rec_range"])

    checked_shards = 0
    for shard in man["shards"]:
        sids = {s for s in read_row_sids(out_dir, shard) if s != NO_SAMPLE_SID}
        rng = shard.get("align_aud_range")
        if not sids or rng is None:
            continue
        lo = global_aud_batch(man, *rng["first"])
        hi = global_aud_batch(man, *rng["last"])
        assert lo <= hi
        checked_shards += 1
        for sid in sids:
            for aud_batch, _idx in by_sid.get(sid, []):
                assert lo <= aud_batch <= hi, (
                    f"P3c 屏障漏覆盖：shard {shard['shard_id']} 的样本 sid={sid} 的 audio rec "
                    f"aud_batch={aud_batch} 不在 align_aud_range=[{lo},{hi}] 内"
                )
    assert checked_shards > 0, "没有任何 shard 带可校验的 align_aud_range"


# ===========================================================================
# 1. 纯 video 多 shard：无 audio 内容 + 每 batch 一个 shard
#    旧实现：shard_max_sid 由 video 样本抬升、audio 写线程 recv 阻塞 ⇒ 死锁
# ===========================================================================
def test_pure_video_multishard_no_deadlock(tmp_path):
    rows = [
        video_row(ts=0.5, text="v zero"),
        video_row(ts=2.5, text="v one"),
        video_row(ts=5.0, text="v two"),
        video_row(ts=7.5, text="v three"),
    ]
    spec = spec_of(tmp_path, rows, "pure_video", ranges=[VID_RANGE],
                   bs=1, bps=1, msl=128, with_audio=False)
    build_or_deadlock_fail(spec, "纯 video 多 shard")

    man = manifest_of(spec["out"])
    assert man["align_fingerprint"] is not None, "只声明 video 区间也应触发 align_enabled"
    assert man.get("audio_shards", []) == [], "纯 video 数据集不得有 audio 流"
    assert len(man["shards"]) >= 4, f"batches_per_shard=1 ⇒ 每样本一 shard，got {len(man['shards'])}"
    for shard in man["shards"]:
        assert shard.get("align_aud_range") is None, (
            "无任何 audio 写入时游标未推进，align_aud_range 必须为 None（不得编造区间）"
        )
        read_row_sids(spec["out"], shard)  # 文件存在且是 u64 定宽
    assert man["schema"]["row_sids"] == "list<uint64>"

    entries = [e for d in iter_align_batches(spec["out"], 1, 128) for e in d["av_align"]]
    assert {e["sample_id"] for e in entries} == {0, 1, 2, 3}, "每个 video 样本都须进对齐视图"
    assert all(e["audio_rec_range"] == [] for e in entries)
    assert all(e["video_rec_range"] for e in entries)


# ===========================================================================
# 2. 反压：Vlm batch 数 > tx_packed bounded(100) + 每 batch 一个 shard
#    旧实现：Vlm 写线程 spin 时不消费 tx_packed ⇒ compute 被反压卡死 ⇒ 死锁
# ===========================================================================
def test_backpressure_over_channel_capacity_no_deadlock(tmp_path):
    n_pairs = 55  # 110 个 batch > bounded(100)
    rows = []
    for i in range(n_pairs):
        # audio ts=0.0 + dur=2.0 ⇒ 单窗 100 tok；audio msl=128 ⇒ 每样本独占一 audio
        # batch（100 > 64 ⇒ 2 doc 装不下一批），避免 GreedyPacker 跨样本交错把后续
        # 样本的 rec 塞进早期 batch（那是另一类"first=prev_last 非超集"覆盖洞，
        # 不在本 P3c-1 屏障死锁回归范围内）。
        rows.append(audio_row(ts=0.0, text=f"a{i}"))
        rows.append(video_row(ts=0.5, text=f"v{i}"))
    spec = spec_of(tmp_path, rows, "backpressure", ranges=[AUD_RANGE, VID_RANGE],
                   bs=1, bps=1, msl=128)
    build_or_deadlock_fail(spec, "反压 110 batch")

    man = manifest_of(spec["out"])
    assert len(man["shards"]) >= 100, (
        f"该用例要求 batch 数超过 tx_packed 容量 100，实际 shard 数 {len(man['shards'])}"
    )
    entries = [e for d in iter_align_batches(spec["out"], 1, 128) for e in d["av_align"]]
    assert len(entries) >= n_pairs, "每个带 timestamp 的样本至少一条对齐条目"
    # 注：本用例**不做** "每个 audio rec 都被 align_aud_range 覆盖" 的断言——
    # 那是一处与本次屏障修复正交的已知缺口，见文件末尾
    # `test_known_gap_align_aud_range_first_skips_earlier_aud_batches`。


# ===========================================================================
# 3. 尾部非 audio：audio 样本之后跟 video / text，收尾 shard 的最大样本无 audio rec
#    旧实现：收尾 shard `shard_max_sid` 由 text/video 样本抬升 ⇒ 死锁；
#    修法后还要求 align_aud_range 不漏掉前面已 send 的 audio rec
# ===========================================================================
def test_tail_non_audio_shard_no_deadlock(tmp_path):
    rows = [
        # 3 个 audio 样本（ts=0.0 + dur=2.0 ⇒ 单窗 100 tok，独占一 audio batch），
        # 其后接 video / text —— 收尾 shard 的最大样本无 audio rec
        audio_row(ts=0.0, text="a zero"),
        audio_row(ts=0.0, text="a one"),
        audio_row(ts=0.0, text="a two"),
        video_row(ts=0.5, text="v tail"),
        video_row(ts=0.5, text="v tail two"),
        {"text": "plain tail"},
    ]
    spec = spec_of(tmp_path, rows, "tail_non_audio", ranges=[AUD_RANGE, VID_RANGE],
                   bs=1, bps=2, msl=128)
    build_or_deadlock_fail(spec, "尾部非 audio")

    man = manifest_of(spec["out"])
    assert man["audio_shards"], "audio 样本必须产出音频流"
    # VR-2 设计修正后：align_aud_range = 本 shard 样本 rec 所在的最小/最大 audio batch。
    # 最后一个 Vlm shard 含 video/text（无 audio rec）⇒ 精确判空为 None（旧"游标
    # 快照"会给它一个覆盖不到任何本 shard rec 的超集范围，语义不准）。
    last = man["shards"][-1]
    assert last.get("align_aud_range") is None, (
        "尾部纯 video/text shard 无 audio rec，align_aud_range 必须为 None（不再编造区间）"
    )
    entries = [e for d in iter_align_batches(spec["out"], 1, 128) for e in d["av_align"]]
    assert 5 not in {e["sample_id"] for e in entries}, "纯文本样本不得进对齐视图"
    # VR-2：含 audio 样本的前两个 shard 必须有精确范围且不漏覆盖（deferred 导出）
    assert_no_audio_leak(man, spec["out"], entries)


# ===========================================================================
# VR-2 / §10.6(3) 设计修正回归（原 xfail 缺口，现已修复 → 硬断言）：
# 旧 `align_aud_range.first = 上一 Vlm shard 的快照游标`，audio 写线程天然领先
# Vlm 写线程 ⇒ 本 shard 样本的 rec 可能落在 `first` **之前**的 audio batch，
# join 读不到（对齐视图丢该样本 audio 侧）。改为"per-shard 样本 rec 批次区间"
# （span + sid 区间 + 二分）后，first/last 精确等于本 shard 样本 rec 所在的最小/
# 最大 audio batch，两个同族缺口（first 非超集 + VR-2 last 漏尾部 flush）一并消解。
# ===========================================================================
def test_align_aud_range_covers_all_shard_audio_recs(tmp_path):
    rows = [
        audio_row(ts=0.0, text="a zero"),
        audio_row(ts=0.0, text="a one"),
        audio_row(ts=0.0, text="a two"),
        video_row(ts=0.5, text="v tail"),
        video_row(ts=0.5, text="v tail two"),
        {"text": "plain tail"},
    ]
    spec = spec_of(tmp_path, rows, "coverage_gap", ranges=[AUD_RANGE, VID_RANGE],
                   bs=1, bps=2, msl=128)
    build_or_deadlock_fail(spec, "覆盖缺口用例")

    man = manifest_of(spec["out"])
    entries = [e for d in iter_align_batches(spec["out"], 1, 128) for e in d["av_align"]]
    audio_sids = {e["sample_id"] for e in entries if e["audio_rec_range"]}
    assert audio_sids == {0, 1, 2}, f"audio 样本须全部进对齐视图，got {audio_sids}"
    assert_no_audio_leak(man, spec["out"], entries)


# ===========================================================================
# VR-2 确定性回归（复审建议构造：尾部 audio 样本 + batches_per_shard=1 +
# audio doc 不满批）：末 Vlm shard 由**滚动**触发收尾（bps=1 ⇒ 每 shard 在循环
# 内收尾），而其样本的 audio rec 落在 `audio_packer.finish()` 冲出的**尾部 flush 批**
# （全局序在所有 Vlm batch 之后）。旧"游标快照 + is_final"对此不设防：末 shard
# 只等到水位（此时 flush 批尚未写出）⇒ align_aud_range 漏覆盖尾部 audio。
# 改为 deferred 精确导出后：等 aud_stream_done（spans 已含 flush 批）再二分 ⇒ 精确覆盖。
#
# 构造：audio_max_seq_len=512 / audio_batch_size=1 ⇒ audio 容量 512 > 3 窗总 token
# （每窗 dur=2.0×rate=50 ⇒ 100 tok），3 个 audio doc 全为残余 ⇒ 唯一 audio 批由
# finish() 冲出（flush），且携带全部 3 个样本的 rec；bps=1 ⇒ 3 个 Vlm shard 均滚动收尾。
# ===========================================================================
def test_vr2_tail_flush_audio_covered_by_rolling_last_shard(tmp_path):
    rows = [
        audio_row(ts=0.0, text="a zero"),
        audio_row(ts=0.0, text="a one"),
        audio_row(ts=0.0, text="a two"),
    ]
    spec = spec_of(tmp_path, rows, "vr2_tail_flush", ranges=[AUD_RANGE],
                   bs=1, bps=1, msl=128, audio_bs=1, audio_msl=512)
    build_or_deadlock_fail(spec, "VR-2 尾部 flush audio")

    man = manifest_of(spec["out"])
    assert man["audio_shards"], "audio 样本必须产出音频流"
    # bps=1 ⇒ 每样本一 shard，全部由滚动触发收尾（含最后一个）
    assert len(man["shards"]) == 3, f"bps=1 ⇒ 3 shard，got {len(man['shards'])}"
    # VR-2 核心：尾部 flush audio 批必须被每个含 audio 样本的 shard 覆盖（旧码会全置 None）
    for shard in man["shards"]:
        assert shard.get("align_aud_range") is not None, (
            f"含 audio 样本的 shard {shard['shard_id']} 漏落 align_aud_range（尾部 flush 未覆盖）"
        )
    entries = [e for d in iter_align_batches(spec["out"], 1, 128) for e in d["av_align"]]
    assert {e["sample_id"] for e in entries if e["audio_rec_range"]} == {0, 1, 2}, (
        "3 个 audio 样本的 rec 都须能经 align_aud_range join 到（尾部 flush 批不漏）"
    )
    assert_no_audio_leak(man, spec["out"], entries)


# ===========================================================================
# P4-1（评审）：manifest `source_formats` / `source_field_map` 的真实链路透出
#
# 原 `source_formats_and_field_map_summary` 只是把 `build_dataset` 内的去重/摘要
# 逻辑复制进测试里（删掉生产代码它依然绿），不构成覆盖。下面这条走真实构建 →
# 真实 `index.json`，断言两个字段的实际内容。
# ===========================================================================
def test_p4_manifest_source_formats_and_field_map_end_to_end(tmp_path):
    rows = [{"text": "t0"}, {"text": "t1"}]
    jsonl = write_jsonl(tmp_path / "p4.jsonl", rows)
    out = tmp_path / "p4_out"
    from megadata import BuildConfig, DataBuilder
    from megadata.modality_adapter import make_vlm_encode_callback

    cfg = BuildConfig(
        input_paths=[jsonl],
        output_dir=str(out),
        mode="vlm",
        batch_size=2,
        modality_id_ranges=[IMG_RANGE],
        max_visual_tokens_per_sample=128,
        field_map={
            "image_fields": ["my_img"],
            "timestamp_field": "ts",
            "projection": ["my_img"],
        },
    )
    DataBuilder(max_seq_len=128).build_dataset(
        cfg, modality_encoder=make_vlm_encode_callback(image_range_start=IMG_BASE)
    )
    man = manifest_of(str(out))

    # 源格式：jsonl 输入去重 + 排序后可复现
    assert man["source_formats"] == ["jsonl"]
    # 字段映射摘要：覆盖项生效（含 timestamp_field 单元素列表形态）
    fm = man["source_field_map"]
    assert fm is not None, "vlm 构建必须透出 source_field_map"
    assert fm["image_fields"] == ["my_img"]
    assert fm["timestamp_field"] == ["ts"]
    # 未覆盖的键保留默认（P4-2：tar 扩展名 + parquet 列名共用单列表）
    assert "txt" in fm["text_fields"]
    assert "audio_bytes" in fm["audio_fields"]
    assert "video_bytes" in fm["video_fields"]
    assert "metadata" in fm["meta_fields"]

    # pt/sft 不落这两个字段（manifest byte 级兼容的反向确认由 test_mm_* 覆盖，
    # 这里只核对 vlm 未因 field_map 污染其它 key）
    assert "align_fingerprint" not in man or man["align_fingerprint"] is None


# ===========================================================================
# P4-1（评审）：pt/sft 传 field_map → 构建期 loud-fail（不得静默忽略）
# ===========================================================================
def test_p4_field_map_rejected_for_pt_and_sft(tmp_path):
    from megadata import BuildConfig

    jsonl = write_jsonl(tmp_path / "p4pt.jsonl", [{"text": "t0"}])
    for mode in ("pt", "sft"):
        with pytest.raises(ValueError, match="vlm-only"):
            BuildConfig(
                input_paths=[jsonl],
                output_dir=str(tmp_path / f"out_{mode}"),
                mode=mode,
                batch_size=2,
                field_map={"image_fields": ["x"]},
            )

"""多模态 P1 sidecar 集成测试（P1-T7，sidecar 格式冻结 §9 清单全落地）。

**独立实现原则**：sidecar 三文件用 Python ``struct.pack`` 按字节布局手工拼装
（**不依赖 Rust writer 绑定**）——这同时是对格式契约"可被独立实现"的交叉验证；
Rust writer→reader roundtrip 与原子性/恢复语义
（§6：单次调用内 blob+rec+mm_cu 三者原子成对更新、半成品 shard 收尾失败
整体作废重写）已由 ``builder::sidecar`` / ``runtime::mm_sidecar`` 单测覆盖，
本文件在读取端补充"半成品不静默"的集成验证。

守恒清单（§9）：1) Σ img_rec.len(VALID) == blob 长度；2) Σ mm_cu.img_cnt ==
rec 记录数；3) 多 shard 下 total_images == 各 shard rec 数（VALID+EMPTY）；
4) total_tokens == Σ 模态（stats_version=2 构造端断言 + 旧 manifest 解释）；
5) 旧数据集读取输出逐字节不变；6) 负面路径全 fail-fast 且可定位。

不依赖分词器（不 import DataBuilder / 不触碰 TOKENIZER_PATH）。
"""
import hashlib
import json
import os
import struct

import pytest
import torch

from megadata._manifest import _effective_modality_tokens
from megadata.engine import RustStreamReader

# ---- 格式冻结 §2 布局（Python 独立实现）----
IMG_REC_FMT = "<QIHHBBQI"          # offset u64, len u32, grid_h u16, grid_w u16,
                                   # flags u8, rsv u8, sample_id u64, seg_order u32
IMG_REC_SIZE = 32                  # 30B 字段 + 尾部 rsv2 u16（§2.1 v1.1 修订）
MM_CU_FMT = "<IIII"                # n_mod u32, img_cnt u32, aud_cnt u32, vid_cnt u32
FLAG_VALID, FLAG_NO_IMAGE, FLAG_EMPTY_IMAGE, FLAG_ERRORED = 0, 1, 2, 3
SCHEMA_MM_KEYS = {"img_blob": "bytes", "img_rec": "imgrec32", "mm_cu": "mmcu"}


def pack_img_rec(offset, length, *, grid=1, flags=FLAG_VALID, sid=0, seg=0):
    """§2.1：32 字节定宽（字段 30B + rsv2 2B，全部小端、手工序列化）。"""
    return struct.pack(IMG_REC_FMT, offset, length, grid, grid, flags, 0, sid, seg) + b"\x00\x00"


def pack_mm_cu(img_cnt, *, n_mod=3, aud=0, vid=0):
    """§2.4：16 字节/batch，n_mod 恒 3（length-prefix）。"""
    return struct.pack(MM_CU_FMT, n_mod, img_cnt, aud, vid)


def fake_bin_bytes(n):
    """主 .bin 的可预测字节（i32 LE 流的来源；兼容测试逐字节断言两端共用）。"""
    return (bytes(range(256)) * (n // 256 + 1))[:n]


def build_mm_dataset(root, shards, *, bs=2, msl=8, with_mm=True, stats_version=2,
                     total_tokens=None):
    """构造多 shard 多模态数据集。

    ``shards``: list[ batch 列表 ]，每个 batch 是图像规格列表
    ``(payload_bytes | None, flags, sid, seg)``——``payload=None`` 的占位记录
    不写 blob 字节（§3：NO_IMAGE/EMPTY_IMAGE/ERRORED）。

    manifest 的 global_stats 按真实统计落盘（total_images = VALID+EMPTY 数、
    total_tokens == Σ 模态，守恒 4 的构造端满足）。返回 (index_path, manifest)。
    """
    os.makedirs(root, exist_ok=True)
    all_image_tokens = 0
    all_images = 0
    text_tokens = 0
    for s, batches in enumerate(shards):
        blob = bytearray()
        recs = bytearray()
        cu = bytearray()
        for imgs in batches:
            cu += pack_mm_cu(len(imgs))
            for payload, flags, sid, seg in imgs:
                if payload is not None and flags == FLAG_VALID:
                    off = len(blob)
                    blob += payload
                    recs += pack_img_rec(off, len(payload), flags=flags, sid=sid, seg=seg)
                    all_image_tokens += 4 * len(payload)  # grid 1x1：patch 计数代理
                    all_images += 1
                elif payload is not None:
                    # EMPTY_IMAGE：len==0 但计入 total_images（§8）
                    off = len(blob)
                    recs += pack_img_rec(off, 0, flags=flags, sid=sid, seg=seg)
                    all_images += 1
                else:
                    recs += pack_img_rec(len(blob), 0, flags=flags, sid=sid, seg=seg)
        stem = f"shard_{s:05d}"
        (root / f"{stem}.col_img_blob").write_bytes(blob)
        (root / f"{stem}.col_img_rec").write_bytes(recs)
        (root / f"{stem}.mm_cu").write_bytes(cu)
        # 主 .bin：可预测的 i32 LE 流（兼容测试的逐字节断言用）
        (root / f"{stem}.bin").write_bytes(fake_bin_bytes(len(batches) * bs * msl * 4))
    nbatches = [len(b) for b in shards]
    manifest = {
        "version": "2.0",
        "format": "Memmap",
        "max_seq_len": msl,
        "global_stats": {
            "batch_size": bs,
            "total_batches": sum(nbatches),
            "total_shards": len(shards),
            "total_tokens": total_tokens if total_tokens is not None
            else text_tokens + all_image_tokens,
            "total_images": all_images,
        },
        "schema": {"input_ids": "int32", "labels": "int32", "position_ids": "int32",
                   "cu_seqlens": "list<int32>"},
        "shards": [
            {"shard_id": s, "filename": f"shard_{s:05d}.bin",
             "num_batches": nb, "size_bytes": nb * bs * msl * 4, "hashes": {}}
            for s, nb in enumerate(nbatches)
        ],
    }
    if with_mm:
        manifest["schema"].update(SCHEMA_MM_KEYS)
        for sh in manifest["shards"]:
            stem = sh["filename"][: -len(".bin")]
            for key, ext in [("sha256:col_img_blob", "col_img_blob"),
                             ("sha256:col_img_rec", "col_img_rec"),
                             ("sha256:mm_cu", "mm_cu")]:
                p = root / f"{stem}.{ext}"
                sh["hashes"][key] = hashlib.sha256(p.read_bytes()).hexdigest()
    if stats_version >= 2:
        gs = manifest["global_stats"]
        gs["stats_version"] = 2
        gs["total_text_tokens"] = gs["total_tokens"] - all_image_tokens
        gs["total_image_tokens"] = all_image_tokens
    index = root / "index.json"
    index.write_text(json.dumps(manifest), encoding="utf-8")
    return str(index), manifest


def make_reader(index, *, shard_order="sequential"):
    return RustStreamReader(index, _BS, _MSL, 1, 0, 42, 0, 0, False,
                            "round_robin", shard_order)


def iter_batches(reader):
    """迭代本轮全部 batch（Eof → None 结束）。"""
    while True:
        d = reader.next_batch_capsules()
        if d is None:
            return
        yield d


_BS, _MSL = 2, 8


@pytest.fixture()
def mm_dataset(tmp_path):
    """单 shard 3 batch：b0 两张图 / b1 纯文本 / b2 VALID+EMPTY+NO_IMAGE 混合。"""
    shards = [[
        [(bytes([10] * 8), FLAG_VALID, 0, 0), (bytes([11] * 4), FLAG_VALID, 0, 1)],
        [],
        [(bytes([12] * 6), FLAG_VALID, 2, 0), (None, FLAG_EMPTY_IMAGE, 2, 1),
         (None, FLAG_NO_IMAGE, 2, 2)],
    ]]
    return build_mm_dataset(tmp_path / "ds", shards)


# ==================== 守恒不变式（§9.1–9.3）====================

def entry_bytes(entry):
    """images 条目的字节（DLPack capsule → torch tensor）。"""
    return torch.from_dlpack(entry["bytes"])


def test_conservation_1_valid_len_equals_blob_length(mm_dataset):
    """守恒 1：Σ img_rec.len(VALID) == blob 文件长度（尾部无孤儿字节）。"""
    index, manifest = mm_dataset
    root = os.path.dirname(index)
    blob = (root + "/shard_00000.col_img_blob")
    recs_raw = open(root + "/shard_00000.col_img_rec", "rb").read()
    total_valid = sum(
        struct.unpack_from("<I", r, 8)[0]
        for r in (recs_raw[i:i + IMG_REC_SIZE] for i in range(0, len(recs_raw), IMG_REC_SIZE))
        if r[16] == FLAG_VALID
    )
    assert total_valid == os.path.getsize(blob), "Σ VALID len 必须等于 blob 长度"
    # 端到端：BatchStream 读出的图像字节总和一致（零拷贝 capsule 数据完整）
    reader = make_reader(index)
    got = sum(entry_bytes(e).numel()
              for d in iter_batches(reader) for e in d["images"]
              if e["flags"] == FLAG_VALID)
    assert got == total_valid


def test_conservation_2_cu_counts_equal_rec_count(mm_dataset):
    """守恒 2：Σ mm_cu.img_cnt == rec 记录数 == 各 batch images 条数和。"""
    index, _ = mm_dataset
    root = os.path.dirname(index)
    cu_raw = open(root + "/shard_00000.mm_cu", "rb").read()
    cu_counts = [struct.unpack_from("<I", cu_raw, i * 16 + 4)[0]
                 for i in range(len(cu_raw) // 16)]
    rec_count = os.path.getsize(root + "/shard_00000.col_img_rec") // IMG_REC_SIZE
    assert sum(cu_counts) == rec_count
    reader = make_reader(index)
    per_batch = [len(d["images"]) for d in iter_batches(reader)]
    assert per_batch == [2, 0, 3], "各 batch images 条数与写入一致"
    assert sum(per_batch) == rec_count == sum(cu_counts)


def test_conservation_3_total_images_multi_shard(tmp_path):
    """守恒 3：多 shard 构造下 total_images 与各 shard rec 数（VALID+EMPTY）一致，
    且 BatchStream 读取端不依赖 shard 顺序地看到同样多的非 NO_IMAGE 条目。"""
    shards = [
        [[(bytes([1] * 4), FLAG_VALID, 0, 0), (None, FLAG_NO_IMAGE, 0, 1)],
         [(bytes([2] * 6), FLAG_VALID, 1, 0)]],
        [[(b"", FLAG_EMPTY_IMAGE, 2, 0)],
         [(bytes([3] * 2), FLAG_VALID, 3, 0), (bytes([4] * 2), FLAG_VALID, 3, 1)],
         []],
    ]
    index, manifest = build_mm_dataset(tmp_path / "ds2", shards)
    assert manifest["global_stats"]["total_images"] == 5  # 4 VALID + 1 EMPTY
    reader = make_reader(index)
    seen = [e for d in iter_batches(reader) for e in d["images"]]
    assert len(seen) == 6  # 全部 rec（含 NO_IMAGE 占位；空 batch 不产生 rec）
    assert sum(1 for e in seen if e["flags"] != FLAG_NO_IMAGE) == 5
    # mm_cu 求和（文件算术，独立于读取端；Σ 全部条目的 img_cnt）
    total_rec = sum(
        os.path.getsize(os.path.join(os.path.dirname(index), f"shard_{s:05d}.col_img_rec"))
        // IMG_REC_SIZE for s in range(2))
    assert total_rec == 6
    cu_sum = 0
    for s in range(2):
        cu_raw = open(os.path.join(os.path.dirname(index), f"shard_{s:05d}.mm_cu"), "rb").read()
        cu_sum += sum(struct.unpack_from("<I", cu_raw, i * 16 + 4)[0]
                      for i in range(len(cu_raw) // 16))
    assert cu_sum == 6


def test_conservation_4_tokens_split_and_legacy(tmp_path):
    """守恒 4：stats_version=2 时 total_tokens == Σ 模态（构造端断言 + 读取解释）；
    旧 manifest（无 stats_version）解释为 text ≡ total_tokens、其余模态 0。"""
    # v2 路径：构造端保证守恒
    index, manifest = build_mm_dataset(tmp_path / "v2", [[[(bytes([1] * 8), FLAG_VALID, 0, 0)]]])
    gs = manifest["global_stats"]
    assert gs["stats_version"] == 2
    assert gs["total_tokens"] == (gs["total_text_tokens"] + gs["total_image_tokens"]
                                  + gs.get("total_audio_tokens", 0) + gs.get("total_video_tokens", 0))
    # 旧 manifest 路径：解释函数 text ≡ total、其余 0（读不改写）
    legacy = build_mm_dataset(tmp_path / "legacy", [[[(bytes([1] * 8), FLAG_VALID, 0, 0)]]],
                              stats_version=0)[1]["global_stats"]
    assert "stats_version" not in legacy
    text, image, audio, video = _effective_modality_tokens(legacy)
    assert text == legacy["total_tokens"]
    assert (image, audio, video) == (0, 0, 0)


# ==================== 兼容（§9.5）====================

def test_legacy_dataset_output_unchanged_no_images_key(tmp_path):
    """兼容：旧数据集（schema 无 mm key）→ batch dict 不含 images 键，
    input_ids 的 DLPack 数据与写入字节逐字节一致。"""
    index, _ = build_mm_dataset(tmp_path / "plain", [[[], [], []]], with_mm=False)
    reader = make_reader(index)
    bs, msl = _BS, _MSL
    for b, d in enumerate(iter_batches(reader)):
        assert "images" not in d, f"旧数据集不得透出 images 键（batch {b}）"
        t = torch.from_dlpack(d["input_ids"])
        expected = fake_bin_bytes(3 * bs * msl * 4)[(b * bs * msl * 4): ((b + 1) * bs * msl * 4)]
        assert t.numpy().tobytes() == expected, f"batch {b} input_ids 逐字节一致"


# ==================== 负面（§9.6：fail-fast 且可定位）====================

def test_negative_blob_truncated_fail_fast(tmp_path):
    """blob 截断 → 读 batch fail-fast，错误信息可定位（含文件/batch 上下文）。"""
    index, _ = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 16), FLAG_VALID, 0, 0)]]])
    p = os.path.join(str(tmp_path), "ds", "shard_00000.col_img_blob")
    data = open(p, "rb").read()
    open(p, "wb").write(data[:-1])  # 截 1 字节
    reader = make_reader(index)
    with pytest.raises(OSError, match="col_img_blob"):
        reader.next_batch_capsules()


def test_negative_rec_truncated_fail_fast(tmp_path):
    """rec 截断 → 记录区间越界 UnexpectedEof（Fatal → OSError）。"""
    index, _ = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 16), FLAG_VALID, 0, 0),
                                                   (bytes([8] * 16), FLAG_VALID, 0, 1)]]])
    p = os.path.join(str(tmp_path), "ds", "shard_00000.col_img_rec")
    data = open(p, "rb").read()
    open(p, "wb").write(data[:IMG_REC_SIZE + 8])  # 第 2 条记录截半
    reader = make_reader(index)
    with pytest.raises(OSError, match="col_img_rec"):
        reader.next_batch_capsules()


def test_negative_mm_cu_truncated_fail_fast(tmp_path):
    """mm_cu 截断（0 batch）→ 读 batch 0 fail-fast，不静默当 EOF / 空图。"""
    index, _ = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 4), FLAG_VALID, 0, 0)]]])
    p = os.path.join(str(tmp_path), "ds", "shard_00000.mm_cu")
    open(p, "wb").write(b"")
    reader = make_reader(index)
    with pytest.raises(OSError, match="mm_cu"):
        reader.next_batch_capsules()


def test_negative_tampered_n_mod(tmp_path):
    """篡改 n_mod（3→5）→ 前向不兼容检测，错误信息含 n_mod 值。"""
    index, _ = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 4), FLAG_VALID, 0, 0)]]])
    p = os.path.join(str(tmp_path), "ds", "shard_00000.mm_cu")
    raw = bytearray(open(p, "rb").read())
    struct.pack_into("<I", raw, 0, 5)
    open(p, "wb").write(bytes(raw))
    reader = make_reader(index)
    with pytest.raises(OSError, match=r"n_mod=5"):
        reader.next_batch_capsules()


def test_negative_unknown_flags(tmp_path):
    """未知 flags（9）→ fail-fast（格式前向不兼容检测），错误信息含 flags 值。"""
    index, _ = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 4), FLAG_VALID, 0, 0)]]])
    p = os.path.join(str(tmp_path), "ds", "shard_00000.col_img_rec")
    raw = bytearray(open(p, "rb").read())
    raw[16] = 9
    open(p, "wb").write(bytes(raw))
    reader = make_reader(index)
    with pytest.raises(OSError, match="flags 9"):
        reader.next_batch_capsules()


def test_negative_partial_schema_keys(tmp_path):
    """缺 key 组合（只有 img_rec）→ manifest 自相矛盾，构造期 fail-fast 含 inconsistent。"""
    index, manifest = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 4), FLAG_VALID, 0, 0)]]])
    manifest["schema"] = {k: v for k, v in manifest["schema"].items() if k != "img_blob"}
    manifest["shards"][0]["hashes"].pop("sha256:col_img_blob")
    (tmp_path / "ds" / "index.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OSError, match="inconsistent"):
        make_reader(index)


def test_negative_missing_sidecar_file(tmp_path):
    """manifest 有 key + hash 但文件被删 → fail-fast（不静默降级）。"""
    index, _ = build_mm_dataset(tmp_path / "ds", [[[(bytes([9] * 4), FLAG_VALID, 0, 0)]]])
    os.remove(os.path.join(str(tmp_path), "ds", "shard_00000.mm_cu"))
    with pytest.raises(OSError, match="mm_cu"):
        make_reader(index)


# ==================== 恢复语义（§9.7 / 格式冻结 §6）====================

def test_recovery_partial_shard_not_silent(tmp_path):
    """恢复（§6）读取端语义：mm_cu 只覆盖部分 batch（半成品 shard）时，
    读取超出部分必须 fail-fast（含 batch 定位），绝不静默当 EOF / 空图。
    写入端原子性（blob+rec+mm_cu 单次调用成对更新、半成品整体作废重写）
    由 runtime::mm_sidecar 与 builder::sidecar 的 Rust 单测覆盖
    （test_roundtrip_batch_records_and_blob_bytes / write_dataset 布局）。"""
    # 构造 2 batch 数据集，但 mm_cu 只写第 1 条（模拟收尾中断的半成品）
    index, _ = build_mm_dataset(tmp_path / "ds", [
        [[(bytes([1] * 4), FLAG_VALID, 0, 0)], [(bytes([2] * 4), FLAG_VALID, 1, 0)]],
    ])
    p = os.path.join(str(tmp_path), "ds", "shard_00000.mm_cu")
    open(p, "wb").write(pack_mm_cu(1))  # 只剩 batch 0 条目
    reader = make_reader(index)
    d = reader.next_batch_capsules()   # batch 0 正常
    assert d is not None and len(d["images"]) == 1
    with pytest.raises(OSError, match="batch 1"):
        reader.next_batch_capsules()   # batch 1 无 mm_cu 条目 → fail-fast

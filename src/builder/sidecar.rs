//! 多模态 sidecar 写入（P1-T2，格式冻结基线 v1.4）。
//!
//! 字节布局契约以本模块文档 + `tests/test_mm_p1_sidecar.py` 的独立实现断言为准，
//! 本模块实现图像模态（P1）+ 音频模态（P3a：`AudioSidecarWriter`，aud/vid 布局
//! 一次到位冻结；vid 布局已冻结、P3b 才落地，mm_cu 的 vid 计数位恒 0）。
//!
//! 图像三文件：`col_img_blob`（变长字节流）/ `col_img_rec`（32 字节定宽记录）/
//! `mm_cu`（16 字节/batch 边界索引，仅服务 Vlm 流）。
//! 音频三文件（P3a，`ashard_` 独立 shard 序列）：`col_aud_blob` /
//! `col_aud_rec`（40 字节定宽记录）/ `col_aud_cu`（16 字节/batch 边界索引，
//! 服务 audio 流——格式冻结 §2.4 v1.2 语义限定：`mm_cu` 单序列不足以索引
//! R1 路由下 audio/video 两条独立 batch 序列）。
//! **每个写入分支配对 `hasher.update`**（项目铁律，漏配会使 finish 哈希 !=
//! 落盘回读哈希且静默）。
//!
//! P1/P3a 无生产调用方（vlm 管线 P2 接通、audio 流 P3a-T2 接通，见 handoff §3.4）；
//! 由单测与集成测试直接驱动。调用顺序契约：`append_mm_batch` 与
//! `MemmapWriter::append_batch` 同 batch 顺序配对；`append_audio_batch` 与
//! audio 流 `GreedyPacker` 出批同序配对（P3a §5.4：rec↔batch 配对载体 =
//! `AudioPackedBatch.audio_records`）。

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::packing::document_aware::PackedBatch;

/// `col_img_rec` 定宽记录字节数（格式冻结 §2.1）。
pub(crate) const IMG_REC_SIZE: usize = 32;
/// `col_vid_rec` 定宽记录字节数（格式冻结 §2.3：40B，含 start_ms）。
pub(crate) const VID_REC_SIZE: usize = 40;

/// `mm_cu` 每 batch 字节数（格式冻结 §2.4：n_mod + 3 计数）。
pub(crate) const MM_CU_BATCH_SIZE: usize = 16;
/// `mm_cu` 的 n_mod 恒 3（img/aud/vid；加模态时递增，读取端按它判宽度）。
const MM_CU_N_MOD: u32 = 3;

/// `col_img_rec` 记录 flags（格式冻结 §3；高位保留，未知值读取端 fail-fast）。
pub mod img_flags {
    /// 有图、blob 有字节（len>0）
    pub const VALID: u8 = 0;
    /// 该位置无图（纯文本样本占位 / padding row）
    #[allow(dead_code)]
    pub const NO_IMAGE: u8 = 1;
    /// 图存在但字节为空（len==0）
    #[allow(dead_code)]
    pub const EMPTY_IMAGE: u8 = 2;
    /// 构建期错误占位（SkipAndCount 跳过的样本）
    pub const ERRORED: u8 = 3;

    /// 已冻结的合法值集合（0..=3）；未知值 = 前向不兼容。
    pub fn is_known(f: u8) -> bool {
        f <= ERRORED
    }
}

/// `col_vid_rec` 记录 flags（P3b；数值与语义沿用格式冻结 §3 的 0..=3，
/// **不新增值**——video 走 Vlm 流，flags 值域与 img 共用同一冻结区间；
/// 值 4 `AUDIO_TAIL_WINDOW` 是 audio 流专用，video 不使用）。
pub mod vid_flags {
    /// 有视频、blob 有字节（len>0）
    pub const VALID: u8 = 0;
    /// 该位置无视频段（纯文本样本占位 / padding row）
    #[allow(dead_code)]
    pub const NO_VIDEO: u8 = 1;
    /// 视频存在但字节为空（len==0）
    #[allow(dead_code)]
    pub const EMPTY_VIDEO: u8 = 2;
    /// 构建期错误占位（SkipAndCount 跳过的样本）
    pub const ERRORED: u8 = 3;

    /// 已冻结的合法值集合（0..=3）；未知值 = 前向不兼容。
    pub fn is_known(f: u8) -> bool {
        f <= ERRORED
    }
}

/// `append_mm_batch` 的图像记录输入（offset 由 writer 分配，调用方不可指定）。
pub struct ImageRecordInput<'a> {
    /// `Some(非空)` = VALID 载荷（写入 blob）；`None`/空 = 占位记录
    /// （NO_IMAGE/EMPTY_IMAGE/ERRORED，不写 blob 字节）。
    pub bytes: Option<&'a [u8]>,
    pub grid_h: u16,
    pub grid_w: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
}

/// `col_img_rec` 的落盘形态（offset 已分配；读取端解码亦用此结构，见
/// `runtime::mm_sidecar`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRecord {
    pub offset: u64,
    pub len: u32,
    pub grid_h: u16,
    pub grid_w: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
}

/// `col_img_rec` 32 字节 LE 编码（手工字段序列化，无平台/对齐依赖）。
pub(crate) fn encode_img_rec(rec: &ImageRecord) -> [u8; IMG_REC_SIZE] {
    let mut b = [0u8; IMG_REC_SIZE];
    b[0..8].copy_from_slice(&rec.offset.to_le_bytes());
    b[8..12].copy_from_slice(&rec.len.to_le_bytes());
    b[12..14].copy_from_slice(&rec.grid_h.to_le_bytes());
    b[14..16].copy_from_slice(&rec.grid_w.to_le_bytes());
    b[16] = rec.flags;
    b[17] = 0; // rsv：写 0
    b[18..26].copy_from_slice(&rec.sample_id.to_le_bytes());
    b[26..30].copy_from_slice(&rec.seg_order.to_le_bytes());
    b[30..32].copy_from_slice(&[0, 0]); // rsv2
    b
}

/// `col_img_rec` 解码（读取端用；长度不足由调用方先行校验）。
pub(crate) fn decode_img_rec(b: &[u8]) -> ImageRecord {
    debug_assert_eq!(b.len(), IMG_REC_SIZE);
    ImageRecord {
        offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
        len: u32::from_le_bytes(b[8..12].try_into().unwrap()),
        grid_h: u16::from_le_bytes(b[12..14].try_into().unwrap()),
        grid_w: u16::from_le_bytes(b[14..16].try_into().unwrap()),
        flags: b[16],
        sample_id: u64::from_le_bytes(b[18..26].try_into().unwrap()),
        seg_order: u32::from_le_bytes(b[26..30].try_into().unwrap()),
    }
}

/// 多模态 sidecar 写入器（图像模态，P1）。
///
/// 恢复语义（格式冻结 §6）：append-only、shard 为原子单元——收尾失败时
/// 半成品 shard 整体作废重写，与 PT shard 恢复语义一致。
pub struct MmSidecarWriter {
    img_blob: BufWriter<File>,
    img_rec: BufWriter<File>,
    mm_cu: BufWriter<File>,
    blob_hasher: Sha256,
    rec_hasher: Sha256,
    cu_hasher: Sha256,
    /// 下一条记录的 blob 偏移（= 已写字节数；append-only 单调）。
    blob_offset: u64,
    /// 已写入的 rec 记录数（校验 + P1-T7 守恒锚点）。
    rec_count: u64,
}

impl MmSidecarWriter {
    /// 在 `dir` 下为 `stem`（如 `shard_00000`，不含扩展名）创建图像 sidecar 三文件。
    pub fn create(dir: &Path, stem: &str) -> std::io::Result<Self> {
        let open = |ext: &str| -> std::io::Result<BufWriter<File>> {
            Ok(BufWriter::with_capacity(
                1024 * 1024,
                File::create(dir.join(format!("{stem}.{ext}")))?,
            ))
        };
        Ok(Self {
            img_blob: open("col_img_blob")?,
            img_rec: open("col_img_rec")?,
            mm_cu: open("mm_cu")?,
            blob_hasher: Sha256::new(),
            rec_hasher: Sha256::new(),
            cu_hasher: Sha256::new(),
            blob_offset: 0,
            rec_count: 0,
        })
    }

    /// 追加一个 batch 的图像记录（与 `MemmapWriter::append_batch` 同序配对调用）。
    ///
    /// `vid_cnt`：同 batch 的 video 记录数（P3b 起由 `VideoSidecarWriter` 平行
    /// 落盘，计数经 `mm_cu` 第三位透出；格式冻结 §2.4）。`vid_cnt = 0` 时
    /// `mm_cu` 字节输出与 P2 期逐字节一致（向后兼容）。
    ///
    /// 单次调用内 blob / rec / mm_cu 三者成对更新——调用方无法造成
    /// "blob 追加了而 rec 没写"的中间态（格式冻结 §6 幂等契约）。
    pub fn append_mm_batch(
        &mut self,
        images: &[ImageRecordInput<'_>],
        vid_cnt: u32,
    ) -> std::io::Result<()> {
        // 先做输入契约校验，再写任何字节（避免半批写入后报错）。
        for img in images {
            let has_bytes = img.bytes.is_some_and(|b| !b.is_empty());
            if img.flags == img_flags::VALID && !has_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "VALID image record (sample_id={}, seg_order={}) must carry non-empty bytes",
                        img.sample_id, img.seg_order
                    ),
                ));
            }
            if img.flags != img_flags::VALID && has_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "placeholder image record (flags={}, sample_id={}, seg_order={}) must not carry blob bytes",
                        img.flags, img.sample_id, img.seg_order
                    ),
                ));
            }
            if !img_flags::is_known(img.flags) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "unknown image record flags {} (sample_id={}): format v1 only defines 0..=3",
                        img.flags, img.sample_id
                    ),
                ));
            }
        }

        for img in images {
            let payload_len = match img.bytes {
                Some(b) if !b.is_empty() => {
                    // P2-8（评审修复）：blob rec 的 len 是 u32；单个 payload 超
                    // u32::MAX 时旧 `b.len() as u32` 截断，rec.len 与实际字节不符、
                    // 下一条 offset 错位。写前 checked conversion，超限硬失败。
                    let len = u32::try_from(b.len()).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "image payload (sample_id={}, seg_order={}) is {} bytes, which \
                                 exceeds the u32 blob record length limit; split the media payload",
                                img.sample_id,
                                img.seg_order,
                                b.len()
                            ),
                        )
                    })?;
                    // ★ 每个写入分支配对 hasher.update（项目铁律）
                    self.img_blob.write_all(b)?;
                    self.blob_hasher.update(b);
                    len
                }
                _ => 0, // 占位记录：不写 blob 字节
            };
            let rec = ImageRecord {
                offset: self.blob_offset,
                len: payload_len,
                grid_h: img.grid_h,
                grid_w: img.grid_w,
                flags: img.flags,
                sample_id: img.sample_id,
                seg_order: img.seg_order,
            };
            self.blob_offset += payload_len as u64;
            let rec_bytes = encode_img_rec(&rec);
            self.img_rec.write_all(&rec_bytes)?;
            self.rec_hasher.update(rec_bytes);
            self.rec_count += 1;
        }

        // mm_cu：[n_mod=3][img][aud=0][vid]（aud 恒 0——audio 走独立流；
        // vid 由 P3b 填充，格式冻结 §2.4 v1.2 语义限定）
        let mut cu = [0u8; MM_CU_BATCH_SIZE];
        cu[0..4].copy_from_slice(&MM_CU_N_MOD.to_le_bytes());
        cu[4..8].copy_from_slice(&(images.len() as u32).to_le_bytes());
        cu[12..16].copy_from_slice(&vid_cnt.to_le_bytes());
        self.mm_cu.write_all(&cu)?;
        self.cu_hasher.update(cu);
        Ok(())
    }

    /// 收尾：flush + finalize 三文件 sha256，返回 manifest `hashes` 键值对
    /// （`sha256:col_img_blob` 等，格式冻结 §5）。
    pub fn finish(&mut self) -> std::io::Result<BTreeMap<String, String>> {
        self.img_blob.flush()?;
        self.img_rec.flush()?;
        self.mm_cu.flush()?;
        let mut hashes = BTreeMap::new();
        hashes.insert(
            "sha256:col_img_blob".to_string(),
            hex::encode(self.blob_hasher.finalize_reset()),
        );
        hashes.insert(
            "sha256:col_img_rec".to_string(),
            hex::encode(self.rec_hasher.finalize_reset()),
        );
        hashes.insert(
            "sha256:mm_cu".to_string(),
            hex::encode(self.cu_hasher.finalize_reset()),
        );
        Ok(hashes)
    }

    /// 已写入的 rec 记录数（测试与 P2 pipeline 守恒断言用）。
    #[allow(dead_code)]
    pub fn record_count(&self) -> u64 {
        self.rec_count
    }
}

// ============================================================================
// P3a：audio 流 sidecar（col_aud_blob / col_aud_rec / col_aud_cu）
// ============================================================================

/// `col_aud_rec` 定宽记录字节数（格式冻结 §2.2）。
pub(crate) const AUD_REC_SIZE: usize = 40;
/// `col_aud_cu` 每 batch 字节数（格式冻结 §2.5：n_rec + aud_cnt + rsv）。
pub(crate) const AUD_CU_BATCH_SIZE: usize = 16;
/// `col_aud_cu` 的 n_rec 恒 1（length-prefix；当前仅 audio 单模态，预留扩展）。
const AUD_CU_N_REC: u32 = 1;

/// `col_aud_rec` 记录 flags（格式冻结 §3 v1.3）。
///
/// audio 流**仅产 0 / 4 两值**：图像值 1–3（NO_IMAGE/EMPTY_IMAGE/ERRORED）不用
/// 于 audio——空 payload / 零 token 的 audio 段在 build 侧 SkipAndCount（源数据
/// 质量分类），不产 rec（格式冻结 §3 v1.3 注）。
pub mod aud_flags {
    /// 首窗：有音频段、blob 携带**整段压缩字节**（len>0）。
    pub const VALID: u8 = 0;
    /// 长音频切窗**尾窗**：窗口 token 在 audio 流主文件；原始压缩字节在**同组
    /// 首窗** VALID rec 中，本 rec 的 `offset/len` **复用首窗值**（自足定位，
    /// 读取端无需跨 rec/跨 shard 二次查找），**不重复写 blob 字节**。
    /// 组键 = `(sample_id, seg_order)`（seg_order 继承原段，格式冻结 §3 v1.3）。
    pub const AUDIO_TAIL_WINDOW: u8 = 4;

    /// audio 流合法值集合：仅 0 / 4（与 img_flags 0..=3 各自独立——
    /// 图像记录不经 audio writer，audio 记录不经图像 writer；读取端未知值
    /// fail-fast 的格式前向不兼容检测由各自 reader 执行）。
    pub fn is_known(f: u8) -> bool {
        f == VALID || f == AUDIO_TAIL_WINDOW
    }
}

/// `append_audio_batch` 的音频记录输入（offset 由 writer 内部分配/引用，
/// 调用方不可指定——消除 offset 错位错误类，同图像先例）。
pub struct AudioRecordInput<'a> {
    /// `Some(非空)` = VALID（首窗，写入整段压缩字节）；`None` =
    /// `AUDIO_TAIL_WINDOW`（尾窗，字节引用同组首窗 rec）。
    pub bytes: Option<&'a [u8]>,
    /// 0 = VALID / 4 = AUDIO_TAIL_WINDOW（见 [`aud_flags`]）。
    pub flags: u8,
    pub sample_rate: u32,
    /// **子段**时长（毫秒；`sub_end - sub_start`，非原段时长，P3a §4.1）。
    pub duration_ms: u32,
    pub channels: u16,
    pub sample_id: u64,
    /// **继承原段** seg_order（同段所有子窗同值——尾窗字节引用组键）。
    pub seg_order: u32,
    /// **子段**真实起点（毫秒，相对样本起点；`max(start_ms, k·W_ms)`，P3a §4.1）。
    pub start_ms: u32,
}

/// `col_aud_rec` 的落盘形态（offset/len 已分配或引用；读取端解码亦用此结构，
/// P3c `runtime::mm_sidecar`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioRecord {
    pub offset: u64,
    pub len: u32,
    pub sample_rate: u32,
    pub duration_ms: u32,
    pub channels: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
    pub start_ms: u32,
}

/// `col_aud_rec` 40 字节 LE 编码（手工字段序列化，无平台/对齐依赖；
/// 字段偏移与格式冻结 §2.2 逐字段一致）。
pub(crate) fn encode_aud_rec(rec: &AudioRecord) -> [u8; AUD_REC_SIZE] {
    let mut b = [0u8; AUD_REC_SIZE];
    b[0..8].copy_from_slice(&rec.offset.to_le_bytes());
    b[8..12].copy_from_slice(&rec.len.to_le_bytes());
    b[12..16].copy_from_slice(&rec.sample_rate.to_le_bytes());
    b[16..20].copy_from_slice(&rec.duration_ms.to_le_bytes());
    b[20..22].copy_from_slice(&rec.channels.to_le_bytes());
    b[22] = rec.flags;
    b[23] = 0; // rsv：写 0
    b[24..32].copy_from_slice(&rec.sample_id.to_le_bytes());
    b[32..36].copy_from_slice(&rec.seg_order.to_le_bytes());
    b[36..40].copy_from_slice(&rec.start_ms.to_le_bytes());
    b
}

/// audio 流 sidecar 记录的 owned 形态（跨 flume 通道传输；落盘时由 pipeline 转
/// `AudioRecordInput<'a>` 借用给 `AudioSidecarWriter`，镜像
/// `ImageRecordInputOwned` 先例）。
#[derive(Debug, Clone)]
pub struct AudioRecordInputOwned {
    /// `Some(非空)` = VALID（首窗，整段压缩字节）；`None` = AUDIO_TAIL_WINDOW。
    pub bytes: Option<Vec<u8>>,
    pub flags: u8,
    pub sample_rate: u32,
    pub duration_ms: u32,
    pub channels: u16,
    pub sample_id: u64,
    pub seg_order: u32,
    pub start_ms: u32,
}

/// audio 流通道元素（镜像 `VlmPackedBatch`，P3a §5.4）：主 batch + 配对 rec 载体。
///
/// rec↔batch 配对机制：demux 侧维护 `pending: HashMap<doc_uid, AudioRecordInputOwned>`
/// （窗口 doc → 其 rec），`push_docs`/`finish` 返回的每个 `PackedBatch` 按
/// `cu_seqlens` 段边界读各段首 token 的 `doc_ids` 值（= uid），按段序从
/// `pending` 取出对应 rec 组装本结构；uid 缺失 → fail-fast（doc 跨 batch 或
/// 配对 bug——超长窗拦截的构造性兜底）。
#[derive(Debug)]
pub struct AudioPackedBatch {
    /// 主 batch（input_ids/cu_seqlens/doc_ids/loss_weights；audio 流 Memmap 五件套）。
    pub batch: PackedBatch,
    /// 该 batch 的音频 sidecar 记录（顺序 = batch 内 doc 顺序）。
    pub audio_records: Vec<AudioRecordInputOwned>,
}

/// 多模态 sidecar 写入器（**音频模态**，P3a；`ashard_XXXXX` 独立 shard 序列）。
///
/// 恢复语义（格式冻结 §6）：append-only、shard 为原子单元——收尾失败时半成品
/// shard 整体作废重写（pipeline `abort_current_shard` 删全部 8 个文件）。
pub struct AudioSidecarWriter {
    aud_blob: BufWriter<File>,
    aud_rec: BufWriter<File>,
    aud_cu: BufWriter<File>,
    blob_hasher: Sha256,
    rec_hasher: Sha256,
    cu_hasher: Sha256,
    /// 下一条 VALID 记录的 blob 偏移（= 已写字节数；append-only 单调）。
    blob_offset: u64,
    /// 已写入的 rec 记录数（含尾窗；守恒断言用）。
    rec_count: u64,
    /// 尾窗字节引用表：`(sample_id, seg_order) → (offset, len)`。
    ///
    /// 只需容纳"当前样本"的段——demux 按（样本 → 段 → 窗）字典序构造 doc 列表、
    /// GreedyPacker 按列表序装箱（P3a §5.4 发射序保证）⇒ rec 到达写线程时同一样本
    /// 的段连续、同段的窗口连续（首窗先于尾窗）。`sample_id` 更替即清表
    /// （防跨样本键碰撞的悬空引用；清表后查不到 = 发射序被破坏，fail-fast）。
    seg_ref: HashMap<(u64, u32), (u64, u32)>,
    /// 当前样本 id（`None` = 尚未写入任何记录；用于触发清表）。
    current_sid: Option<u64>,
}

impl AudioSidecarWriter {
    /// 在 `dir` 下为 `stem`（如 `ashard_00000`，不含扩展名）创建 audio sidecar
    /// 三文件（格式冻结 §1 v1.3：`col_aud_blob` / `col_aud_rec` / `col_aud_cu`）。
    pub fn create(dir: &Path, stem: &str) -> std::io::Result<Self> {
        // 后缀字面量与格式冻结 §1 文件命名逐字一致（锚点机器校验同源）。
        let open = |suffix: &str| -> std::io::Result<BufWriter<File>> {
            Ok(BufWriter::with_capacity(
                1024 * 1024,
                File::create(dir.join(format!("{stem}{suffix}")))?,
            ))
        };
        Ok(Self {
            aud_blob: open(".col_aud_blob")?,
            aud_rec: open(".col_aud_rec")?,
            aud_cu: open(".col_aud_cu")?,
            blob_hasher: Sha256::new(),
            rec_hasher: Sha256::new(),
            cu_hasher: Sha256::new(),
            blob_offset: 0,
            rec_count: 0,
            seg_ref: HashMap::new(),
            current_sid: None,
        })
    }

    /// 追加一个 audio batch 的记录（与 audio 流 GreedyPacker 出批**同序配对**
    /// 调用；载体 = [`AudioPackedBatch`]）。
    ///
    /// 单次调用内 blob / rec / cu 三者成对更新——调用方无法造成"blob 追加了而
    /// rec 没写"的中间态（格式冻结 §6 幂等契约）。输入契约校验先于任何写入
    /// （半批原子，同 `MmSidecarWriter` 先例）。
    ///
    /// flags 语义（格式冻结 §3 v1.3）：
    /// - `VALID`（首窗）：整段压缩字节 append 到 blob 一次，登记
    ///   `seg_ref[(sample_id, seg_order)] = (offset, len)`；
    /// - `AUDIO_TAIL_WINDOW`（尾窗）：**不写 blob 字节**，查 `seg_ref` 得
    ///   `(offset, len)`（= 首窗值，自足定位）；查不到 → fail-fast。
    pub fn append_audio_batch(&mut self, records: &[AudioRecordInput<'_>]) -> std::io::Result<()> {
        // 先做输入契约校验，再写任何字节（避免半批写入后报错）。
        for r in records {
            let has_bytes = r.bytes.is_some_and(|b| !b.is_empty());
            if r.flags == aud_flags::VALID && !has_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "VALID audio record (sample_id={}, seg_order={}) must carry non-empty compressed bytes \
                         (empty payload / zero-token segments are SkipAndCount upstream, no rec emitted)",
                        r.sample_id, r.seg_order
                    ),
                ));
            }
            if r.flags == aud_flags::AUDIO_TAIL_WINDOW && has_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "AUDIO_TAIL_WINDOW record (sample_id={}, seg_order={}) must not carry blob bytes \
                         (bytes are referenced from the group's VALID record via (offset, len))",
                        r.sample_id, r.seg_order
                    ),
                ));
            }
            if !aud_flags::is_known(r.flags) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "unknown audio record flags {} (sample_id={}): the audio stream only defines \
                         0 (VALID) / 4 (AUDIO_TAIL_WINDOW); image values 1-3 are not used for audio",
                        r.flags, r.sample_id
                    ),
                ));
            }
        }

        for r in records {
            // sample 切换 → 清尾窗引用表（发射序连续性见字段文档）。
            if self.current_sid != Some(r.sample_id) {
                self.seg_ref.clear();
                self.current_sid = Some(r.sample_id);
            }
            let (offset, len) = match r.flags {
                aud_flags::VALID => {
                    let b = r.bytes.expect("VALID validated: non-empty bytes present");
                    // P2-8：u32 rec.len 溢出防护（同 image 分支）。
                    let len = u32::try_from(b.len()).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "audio payload (sample_id={}, seg_order={}) is {} bytes, which \
                                 exceeds the u32 blob record length limit; split the audio segment",
                                r.sample_id,
                                r.seg_order,
                                b.len()
                            ),
                        )
                    })?;
                    let offset = self.blob_offset;
                    // ★ 每个写入分支配对 hasher.update（项目铁律）
                    self.aud_blob.write_all(b)?;
                    self.blob_hasher.update(b);
                    self.blob_offset += len as u64;
                    // 登记尾窗字节引用（同段所有子窗复用首窗 (offset, len)）
                    self.seg_ref
                        .insert((r.sample_id, r.seg_order), (offset, len));
                    (offset, len)
                }
                aud_flags::AUDIO_TAIL_WINDOW => {
                    // 查同组首窗引用；查不到 = 发射序被破坏（构造性检测，fail-fast）
                    *self.seg_ref.get(&(r.sample_id, r.seg_order)).ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "AUDIO_TAIL_WINDOW record (sample_id={}, seg_order={}) has no VALID record \
                                 in its group (emission order violated: windows of one segment must be emitted \
                                 contiguously with the first window before tail windows)",
                                r.sample_id, r.seg_order
                            ),
                        )
                    })?
                }
                _ => unreachable!("flags validated above (aud_flags::is_known)"),
            };
            let rec = AudioRecord {
                offset,
                len,
                sample_rate: r.sample_rate,
                duration_ms: r.duration_ms,
                channels: r.channels,
                flags: r.flags,
                sample_id: r.sample_id,
                seg_order: r.seg_order,
                start_ms: r.start_ms,
            };
            let rec_bytes = encode_aud_rec(&rec);
            self.aud_rec.write_all(&rec_bytes)?;
            self.rec_hasher.update(rec_bytes);
            self.rec_count += 1;
        }

        // col_aud_cu：[n_rec=1][aud_cnt][rsv u64=0]（格式冻结 §2.5，16B 定宽）
        let mut cu = [0u8; AUD_CU_BATCH_SIZE];
        cu[0..4].copy_from_slice(&AUD_CU_N_REC.to_le_bytes());
        cu[4..8].copy_from_slice(&(records.len() as u32).to_le_bytes());
        self.aud_cu.write_all(&cu)?;
        self.cu_hasher.update(cu);
        Ok(())
    }

    /// P2-1（评审修复）：该 `(sample_id, seg_order)` 的首窗 VALID 是否已写入
    /// **本 writer 当前 shard**。
    ///
    /// shard rollover 前据此判断"下一 batch 是否仍在续写本 shard 内未收尾的
    /// 窗口组"：是则延后切 shard。旧实现直接按 batch 数切 shard，新 shard 新
    /// writer 的 `seg_ref` 为空——同段首窗/尾窗分属相邻 shard 时尾窗查不到
    /// 首窗，合法长音频构建失败（P2-1）。
    pub fn has_valid_group(&self, sample_id: u64, seg_order: u32) -> bool {
        self.seg_ref.contains_key(&(sample_id, seg_order))
    }

    /// 收尾：flush + finalize 三文件 sha256，返回 manifest `hashes` 键值对
    /// （`sha256:col_aud_blob` 等，格式冻结 §5 v1.3）。
    pub fn finish(&mut self) -> std::io::Result<BTreeMap<String, String>> {
        self.aud_blob.flush()?;
        self.aud_rec.flush()?;
        self.aud_cu.flush()?;
        let mut hashes = BTreeMap::new();
        hashes.insert(
            "sha256:col_aud_blob".to_string(),
            hex::encode(self.blob_hasher.finalize_reset()),
        );
        hashes.insert(
            "sha256:col_aud_rec".to_string(),
            hex::encode(self.rec_hasher.finalize_reset()),
        );
        hashes.insert(
            "sha256:col_aud_cu".to_string(),
            hex::encode(self.cu_hasher.finalize_reset()),
        );
        Ok(hashes)
    }

    /// 已写入的 rec 记录数（含尾窗；测试与 pipeline 守恒断言用）。
    #[allow(dead_code)]
    pub fn record_count(&self) -> u64 {
        self.rec_count
    }
}

// ============================================================================
// 多模态 video sidecar 写入（P3b；video 走 Vlm 流，per-batch 计数经 mm_cu.vid_cnt）
// ============================================================================

/// `append_vid_batch` 的 video 记录输入（offset 由 writer 分配，调用方不可指定
/// ——消除 offset 错位错误类，同图像先例）。
pub struct VideoRecordInput<'a> {
    /// `Some(非空)` = VALID 载荷（写入 blob）；`None`/空 = 占位记录
    /// （NO_VIDEO/EMPTY_VIDEO/ERRORED，不写 blob 字节）。
    pub bytes: Option<&'a [u8]>,
    /// 采样帧数（= 构建配置 video_n_frames；非源总帧数，P3b §4.4 口径）
    pub n_frames: u16,
    /// resize 后目标分辨率宽（以指纹 resize 参数为准，rec 透传，P3b §6.2）
    pub width: u16,
    /// resize 后目标分辨率高
    pub height: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
    /// fps × 1000 取整（显式公式 `(fps_f32 × 1000.0).round() as u32`）
    pub fps_milli: u32,
    /// 段起点（相对样本起点，整数毫秒；终稿 DP4(a′)，INV-2 保证非缺）
    pub start_ms: u32,
}

/// `col_vid_rec` 的落盘形态（offset 已分配；`duration_ms` 仅统计口径用，
/// **不落 rec**——由调用方在 pipeline 侧累计 `total_video_seconds`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRecord {
    pub offset: u64,
    pub len: u32,
    pub n_frames: u16,
    pub width: u16,
    pub height: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
    pub fps_milli: u32,
    pub start_ms: u32,
}

/// `col_vid_rec` 40 字节 LE 编码（字段偏移与格式冻结 §2.3 v1.2 逐字段一致）。
pub(crate) fn encode_vid_rec(rec: &VideoRecord) -> [u8; VID_REC_SIZE] {
    let mut b = [0u8; VID_REC_SIZE];
    b[0..8].copy_from_slice(&rec.offset.to_le_bytes());
    b[8..12].copy_from_slice(&rec.len.to_le_bytes());
    b[12..14].copy_from_slice(&rec.n_frames.to_le_bytes());
    b[14..16].copy_from_slice(&rec.width.to_le_bytes());
    b[16..18].copy_from_slice(&rec.height.to_le_bytes());
    b[18] = rec.flags;
    b[19] = 0; // rsv：写 0
    b[20..28].copy_from_slice(&rec.sample_id.to_le_bytes());
    b[28..32].copy_from_slice(&rec.seg_order.to_le_bytes());
    b[32..36].copy_from_slice(&rec.fps_milli.to_le_bytes());
    b[36..40].copy_from_slice(&rec.start_ms.to_le_bytes());
    b
}

/// `col_vid_rec` 解码（读取端用；长度不足由调用方先行校验）。
pub(crate) fn decode_vid_rec(b: &[u8]) -> VideoRecord {
    debug_assert_eq!(b.len(), VID_REC_SIZE);
    VideoRecord {
        offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
        len: u32::from_le_bytes(b[8..12].try_into().unwrap()),
        n_frames: u16::from_le_bytes(b[12..14].try_into().unwrap()),
        width: u16::from_le_bytes(b[14..16].try_into().unwrap()),
        height: u16::from_le_bytes(b[16..18].try_into().unwrap()),
        flags: b[18],
        sample_id: u64::from_le_bytes(b[20..28].try_into().unwrap()),
        seg_order: u32::from_le_bytes(b[28..32].try_into().unwrap()),
        fps_milli: u32::from_le_bytes(b[32..36].try_into().unwrap()),
        start_ms: u32::from_le_bytes(b[36..40].try_into().unwrap()),
    }
}

/// 多模态 video sidecar 写入器（P3b；独立结构体，与 `AudioSidecarWriter`
/// 同模式——不扩展 `MmSidecarWriter`：`append_mm_batch` 与 Vlm 流图像 batch
/// 同序配对，video rec 是同 batch 的另一份记录，经 `mm_cu.vid_cnt` 计数）。
///
/// video 走 Vlm 流（定宽行），**不需要独立 `.cu` 边界索引**——per-batch 计数
/// 由已冻结的 `mm_cu` 第三计数位承载（`append_mm_batch` 的 `vid_cnt` 参数）。
pub struct VideoSidecarWriter {
    vid_blob: BufWriter<File>,
    vid_rec: BufWriter<File>,
    blob_hasher: Sha256,
    rec_hasher: Sha256,
    /// 下一条记录的 blob 偏移（= 已写字节数；append-only 单调）。
    blob_offset: u64,
    /// 已写入的 rec 记录数（守恒断言用）。
    rec_count: u64,
}

impl VideoSidecarWriter {
    /// 在 `dir` 下为 `stem`（如 `shard_00000`，不含扩展名）创建 video sidecar 两文件。
    pub fn create(dir: &Path, stem: &str) -> std::io::Result<Self> {
        let open = |ext: &str| -> std::io::Result<BufWriter<File>> {
            Ok(BufWriter::with_capacity(
                1024 * 1024,
                File::create(dir.join(format!("{stem}.{ext}")))?,
            ))
        };
        Ok(Self {
            vid_blob: open("col_vid_blob")?,
            vid_rec: open("col_vid_rec")?,
            blob_hasher: Sha256::new(),
            rec_hasher: Sha256::new(),
            blob_offset: 0,
            rec_count: 0,
        })
    }

    /// 追加一个 batch 的 video 记录（与 `MmSidecarWriter::append_mm_batch` 同 batch
    /// 配对调用；blob/rec 单次调用内成对更新——半批原子，输入校验先于任何写入）。
    pub fn append_vid_batch(&mut self, records: &[VideoRecordInput<'_>]) -> std::io::Result<()> {
        // 先做输入契约校验，再写任何字节（参照 sidecar.rs 图像先例）。
        for v in records {
            let has_bytes = v.bytes.is_some_and(|b| !b.is_empty());
            if v.flags == vid_flags::VALID && !has_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "VALID video record (sample_id={}, seg_order={}) must carry non-empty bytes",
                        v.sample_id, v.seg_order
                    ),
                ));
            }
            if v.flags != vid_flags::VALID && has_bytes {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "placeholder video record (flags={}, sample_id={}, seg_order={}) must not carry blob bytes",
                        v.flags, v.sample_id, v.seg_order
                    ),
                ));
            }
            if !vid_flags::is_known(v.flags) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "unknown video record flags {} (sample_id={}): video flags are 0..=3 (value 4 is audio-only)",
                        v.flags, v.sample_id
                    ),
                ));
            }
        }

        for v in records {
            let payload_len = match v.bytes {
                Some(b) if !b.is_empty() => {
                    // P2-8：u32 rec.len 溢出防护（同 image/audio 分支）。
                    let len = u32::try_from(b.len()).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "video payload (sample_id={}, seg_order={}) is {} bytes, which \
                                 exceeds the u32 blob record length limit; split the video segment",
                                v.sample_id,
                                v.seg_order,
                                b.len()
                            ),
                        )
                    })?;
                    // ★ 每个写入分支配对 hasher.update（项目铁律）
                    self.vid_blob.write_all(b)?;
                    self.blob_hasher.update(b);
                    len
                }
                _ => 0, // 占位记录：不写 blob 字节
            };
            let rec = VideoRecord {
                offset: self.blob_offset,
                len: payload_len,
                n_frames: v.n_frames,
                width: v.width,
                height: v.height,
                flags: v.flags,
                sample_id: v.sample_id,
                seg_order: v.seg_order,
                fps_milli: v.fps_milli,
                start_ms: v.start_ms,
            };
            self.blob_offset += payload_len as u64;
            let rec_bytes = encode_vid_rec(&rec);
            self.vid_rec.write_all(&rec_bytes)?;
            self.rec_hasher.update(rec_bytes);
            self.rec_count += 1;
        }
        Ok(())
    }

    /// 收尾：flush + finalize 两文件 sha256，返回 manifest `hashes` 键值对
    /// （`sha256:col_vid_blob` / `sha256:col_vid_rec`，格式冻结 §5 v1.4）。
    pub fn finish(&mut self) -> std::io::Result<BTreeMap<String, String>> {
        self.vid_blob.flush()?;
        self.vid_rec.flush()?;
        let mut hashes = BTreeMap::new();
        hashes.insert(
            "sha256:col_vid_blob".to_string(),
            hex::encode(self.blob_hasher.finalize_reset()),
        );
        hashes.insert(
            "sha256:col_vid_rec".to_string(),
            hex::encode(self.rec_hasher.finalize_reset()),
        );
        Ok(hashes)
    }

    /// 已写入的 rec 记录数（守恒断言用）。
    #[allow(dead_code)]
    pub fn record_count(&self) -> u64 {
        self.rec_count
    }
}

/// `col_aud_rec` 解码（读取端用；长度不足由调用方先行校验）。
/// P3a 期测试先行消费；P3c reader 的对齐 join 复用（同 `decode_img_rec` 先例）。
pub(crate) fn decode_aud_rec(b: &[u8]) -> AudioRecord {
    debug_assert_eq!(b.len(), AUD_REC_SIZE);
    AudioRecord {
        offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
        len: u32::from_le_bytes(b[8..12].try_into().unwrap()),
        sample_rate: u32::from_le_bytes(b[12..16].try_into().unwrap()),
        duration_ms: u32::from_le_bytes(b[16..20].try_into().unwrap()),
        channels: u16::from_le_bytes(b[20..22].try_into().unwrap()),
        flags: b[22],
        sample_id: u64::from_le_bytes(b[24..32].try_into().unwrap()),
        seg_order: u32::from_le_bytes(b[32..36].try_into().unwrap()),
        start_ms: u32::from_le_bytes(b[36..40].try_into().unwrap()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::sha256_file_hex;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("megadata_mm_sidecar_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn valid<'a>(bytes: &'a [u8], sample_id: u64, seg_order: u32) -> ImageRecordInput<'a> {
        ImageRecordInput {
            bytes: Some(bytes),
            grid_h: 4,
            grid_w: 4,
            flags: img_flags::VALID,
            sample_id,
            seg_order,
        }
    }

    fn placeholder(flags: u8, sample_id: u64, seg_order: u32) -> ImageRecordInput<'static> {
        ImageRecordInput {
            bytes: None,
            grid_h: 0,
            grid_w: 0,
            flags,
            sample_id,
            seg_order,
        }
    }

    /// 项目铁律：finish 增量哈希必须等于落盘文件回读 sha256（三文件逐一验证）。
    /// 历史教训：漏配一个 hasher.update 不会被"只断言字节"的测试暴露。
    #[test]
    fn test_sidecar_incremental_hashes_match_readback() {
        let dir = tmp_dir("hash");
        let mut w = MmSidecarWriter::create(&dir, "s").unwrap();
        let img1 = vec![1u8; 100];
        let img2 = vec![2u8; 50];
        w.append_mm_batch(&[valid(&img1, 0, 0)], 0).unwrap();
        w.append_mm_batch(
            &[valid(&img2, 1, 0), placeholder(img_flags::NO_IMAGE, 2, 0)],
            0,
        )
        .unwrap();
        w.append_mm_batch(&[], 0).unwrap(); // 空 batch（纯文本 batch）：mm_cu 记 0
        let hashes = w.finish().unwrap();

        assert_eq!(
            hashes.get("sha256:col_img_blob").unwrap(),
            &sha256_file_hex(&dir.join("s.col_img_blob")).unwrap()
        );
        assert_eq!(
            hashes.get("sha256:col_img_rec").unwrap(),
            &sha256_file_hex(&dir.join("s.col_img_rec")).unwrap()
        );
        assert_eq!(
            hashes.get("sha256:mm_cu").unwrap(),
            &sha256_file_hex(&dir.join("s.mm_cu")).unwrap()
        );
    }

    /// 守恒锚点（格式冻结 §9.1/9.2）：offset 连续无洞、Σ len == blob 长度、
    /// mm_cu 求和 == 记录数、每 batch 计数正确。
    #[test]
    fn test_offset_conservation_and_mm_cu_counts() {
        let dir = tmp_dir("conserv");
        let mut w = MmSidecarWriter::create(&dir, "s").unwrap();
        let img1 = vec![7u8; 64];
        let img2 = vec![9u8; 32];
        let img3 = vec![0u8; 16];
        w.append_mm_batch(&[valid(&img1, 0, 0), valid(&img2, 0, 1)], 0)
            .unwrap(); // 双图样本
        w.append_mm_batch(&[valid(&img3, 5, 0)], 0).unwrap();
        w.append_mm_batch(&[placeholder(img_flags::ERRORED, 9, 0)], 0)
            .unwrap();
        w.finish().unwrap();

        // rec 文件：4 条 × 32B
        let rec_bytes = std::fs::read(dir.join("s.col_img_rec")).unwrap();
        assert_eq!(rec_bytes.len(), 4 * IMG_REC_SIZE);
        let recs: Vec<ImageRecord> = (0..4)
            .map(|i| decode_img_rec(&rec_bytes[i * IMG_REC_SIZE..(i + 1) * IMG_REC_SIZE]))
            .collect();
        // offset 连续 + Σlen == blob 长度（含 ERRORED len=0 不产字节）
        let mut expect_off = 0u64;
        for r in &recs {
            assert_eq!(r.offset, expect_off);
            expect_off += r.len as u64;
        }
        let blob_len = std::fs::metadata(dir.join("s.col_img_blob")).unwrap().len();
        assert_eq!(expect_off, blob_len);
        assert_eq!(blob_len, (64 + 32 + 16) as u64);
        assert_eq!(recs[3].len, 0); // ERRORED 占位无字节
        assert_eq!(recs[0].sample_id, 0);
        assert_eq!(recs[1].seg_order, 1);

        // mm_cu：每 batch 16B；计数与 n_mod 校验
        let cu = std::fs::read(dir.join("s.mm_cu")).unwrap();
        assert_eq!(cu.len(), 3 * MM_CU_BATCH_SIZE);
        let counts: Vec<u32> = (0..3)
            .map(|k| {
                let base = k * MM_CU_BATCH_SIZE;
                assert_eq!(
                    u32::from_le_bytes(cu[base..base + 4].try_into().unwrap()),
                    3,
                    "n_mod 恒 3"
                );
                u32::from_le_bytes(cu[base + 4..base + 8].try_into().unwrap())
            })
            .collect();
        assert_eq!(counts, vec![2, 1, 1]);
        assert_eq!(counts.iter().sum::<u32>(), 4);

        // §9.10（v1.2）：audio 走独立 col_aud_cu，Vlm 流 mm_cu.aud_cnt 恒 0
        for k in 0..3 {
            let base = k * MM_CU_BATCH_SIZE;
            assert_eq!(
                u32::from_le_bytes(cu[base + 8..base + 12].try_into().unwrap()),
                0,
                "mm_cu.aud_cnt 在 Vlm 流恒 0"
            );
        }
    }

    /// API 契约（格式冻结 §7）：VALID 必须带字节；占位不得带字节；未知 flags 拒绝。
    /// 且输入校验先于写入——首个坏记录报错时整批未写。
    #[test]
    fn test_input_contract_enforced_before_write() {
        let dir = tmp_dir("contract");
        let mut w = MmSidecarWriter::create(&dir, "s").unwrap();
        let good = vec![1u8; 8];

        assert!(
            w.append_mm_batch(
                &[ImageRecordInput {
                    bytes: None,
                    grid_h: 1,
                    grid_w: 1,
                    flags: img_flags::VALID,
                    sample_id: 0,
                    seg_order: 0,
                }],
                0
            )
            .is_err()
        );
        assert!(
            w.append_mm_batch(
                &[ImageRecordInput {
                    bytes: Some(&good),
                    grid_h: 0,
                    grid_w: 0,
                    flags: img_flags::NO_IMAGE,
                    sample_id: 0,
                    seg_order: 0,
                }],
                0
            )
            .is_err()
        );
        assert!(
            w.append_mm_batch(
                &[ImageRecordInput {
                    bytes: None,
                    grid_h: 0,
                    grid_w: 0,
                    flags: 9,
                    sample_id: 0,
                    seg_order: 0,
                }],
                0
            )
            .is_err()
        );
        // 整批原子：前面三条非法输入均未落任何字节
        assert_eq!(std::fs::read(dir.join("s.col_img_rec")).unwrap().len(), 0);
        assert_eq!(std::fs::read(dir.join("s.mm_cu")).unwrap().len(), 0);
        assert_eq!(w.record_count(), 0);

        // 合法批仍可正常写（错误不毒化 writer）
        w.append_mm_batch(&[valid(&good, 0, 0)], 0).unwrap();
        assert_eq!(w.record_count(), 1);
        w.finish().unwrap();
    }

    /// 空批（无图 batch）也写 mm_cu 计数 0——边界索引完整覆盖每个 batch，
    /// 这是 reader "batch k 有多少图"纯算术寻址的前提。
    #[test]
    fn test_empty_batch_writes_zero_count() {
        let dir = tmp_dir("emptybatch");
        let mut w = MmSidecarWriter::create(&dir, "s").unwrap();
        w.append_mm_batch(&[], 0).unwrap();
        w.finish().unwrap();
        let cu = std::fs::read(dir.join("s.mm_cu")).unwrap();
        assert_eq!(cu.len(), MM_CU_BATCH_SIZE);
        assert_eq!(u32::from_le_bytes(cu[4..8].try_into().unwrap()), 0);
    }

    // ------------------------------------------------------------------
    // P3a：AudioSidecarWriter 单测（哈希 / 守恒 / 尾窗引用 / 契约）
    // ------------------------------------------------------------------

    fn aud_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "megadata_mm_sidecar_aud_{}_{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// 首窗 VALID 记录输入（携带整段压缩字节）。
    fn aud_valid<'a>(
        bytes: &'a [u8],
        sample_id: u64,
        seg_order: u32,
        start_ms: u32,
        duration_ms: u32,
    ) -> AudioRecordInput<'a> {
        AudioRecordInput {
            bytes: Some(bytes),
            flags: aud_flags::VALID,
            sample_rate: 16000,
            duration_ms,
            channels: 1,
            sample_id,
            seg_order,
            start_ms,
        }
    }

    /// 尾窗 AUDIO_TAIL_WINDOW 记录输入（不携带字节；引用同组首窗）。
    fn aud_tail(
        sample_id: u64,
        seg_order: u32,
        start_ms: u32,
        duration_ms: u32,
    ) -> AudioRecordInput<'static> {
        AudioRecordInput {
            bytes: None,
            flags: aud_flags::AUDIO_TAIL_WINDOW,
            sample_rate: 16000,
            duration_ms,
            channels: 1,
            sample_id,
            seg_order,
            start_ms,
        }
    }

    /// 项目铁律：finish 增量哈希必须等于落盘文件回读 sha256（三文件逐一验证）。
    /// 历史教训：漏配一个 hasher.update 不会被"只断言字节"的测试暴露。
    #[test]
    fn test_audio_sidecar_incremental_hashes_match_readback() {
        let dir = aud_dir("hash");
        let mut w = AudioSidecarWriter::create(&dir, "ashard_00000").unwrap();
        let clip1 = vec![0xABu8; 200];
        let clip2 = vec![0xCDu8; 120];
        // batch1：样本 0 首窗 VALID + 同段尾窗引用
        w.append_audio_batch(&[aud_valid(&clip1, 0, 0, 0, 2000), aud_tail(0, 0, 2000, 2000)])
            .unwrap();
        // batch2：样本 1 首窗 VALID
        w.append_audio_batch(&[aud_valid(&clip2, 1, 0, 0, 1200)])
            .unwrap();
        let hashes = w.finish().unwrap();

        assert_eq!(
            hashes.get("sha256:col_aud_blob").unwrap(),
            &sha256_file_hex(&dir.join("ashard_00000.col_aud_blob")).unwrap()
        );
        assert_eq!(
            hashes.get("sha256:col_aud_rec").unwrap(),
            &sha256_file_hex(&dir.join("ashard_00000.col_aud_rec")).unwrap()
        );
        assert_eq!(
            hashes.get("sha256:col_aud_cu").unwrap(),
            &sha256_file_hex(&dir.join("ashard_00000.col_aud_cu")).unwrap()
        );
    }

    /// 守恒锚点（格式冻结 §9.7 + §2.5 寻址公式）：offset 连续无洞、
    /// `Σ col_aud_cu.aud_cnt == col_aud_rec 记录数`、每 batch 计数正确、
    /// `n_rec` 恒 1、rsv 为 0。
    #[test]
    fn test_audio_offset_conservation_and_aud_cu_counts() {
        let dir = aud_dir("conserv");
        let mut w = AudioSidecarWriter::create(&dir, "ashard_00000").unwrap();
        let c1 = vec![7u8; 64];
        let c2 = vec![9u8; 32];
        // batch1：样本 0 双段各一条首窗（多段样本 seg_ref 分组键互不干扰）
        w.append_audio_batch(&[
            aud_valid(&c1, 0, 0, 0, 1000),
            aud_valid(&c2, 0, 1, 1000, 500),
        ])
        .unwrap();
        // batch2：样本 1 首窗 + 同段尾窗（尾窗也占一条 rec、计入计数）
        w.append_audio_batch(&[aud_valid(&c1, 1, 0, 0, 2000), aud_tail(1, 0, 2000, 1000)])
            .unwrap();
        assert_eq!(w.record_count(), 4);
        w.finish().unwrap();

        // rec 文件：4 条 × 40B；offset 连续 + VALID 段 Σ len == blob 长度
        let rec_bytes = std::fs::read(dir.join("ashard_00000.col_aud_rec")).unwrap();
        assert_eq!(rec_bytes.len(), 4 * AUD_REC_SIZE);
        let recs: Vec<AudioRecord> = (0..4)
            .map(|i| decode_aud_rec(&rec_bytes[i * AUD_REC_SIZE..(i + 1) * AUD_REC_SIZE]))
            .collect();
        let mut expect_off = 0u64;
        let mut valid_len_sum = 0u64;
        let mut last_valid: Option<(u64, u32)> = None; // 同组首窗 (offset, len)
        for r in &recs {
            if r.flags == aud_flags::VALID {
                assert_eq!(r.offset, expect_off, "VALID offset 连续无洞");
                expect_off += r.len as u64;
                valid_len_sum += r.len as u64;
                last_valid = Some((r.offset, r.len));
            } else {
                // 尾窗：offset/len 复用同组首窗（不推进 blob 偏移）
                assert_eq!(
                    (r.offset, r.len),
                    last_valid.expect("尾窗前必有同组首窗"),
                    "尾窗引用同组首窗"
                );
            }
        }
        let blob_len = std::fs::metadata(dir.join("ashard_00000.col_aud_blob"))
            .unwrap()
            .len();
        assert_eq!(
            blob_len,
            (64 + 32 + 64) as u64,
            "只有 VALID 首窗写 blob 字节"
        );
        assert_eq!(valid_len_sum, blob_len, "Σ VALID len == blob 长度（§9.8）");

        // aud_cu：每 batch 16B；计数与 n_rec/rsv 校验；Σ == rec 总数
        let cu = std::fs::read(dir.join("ashard_00000.col_aud_cu")).unwrap();
        assert_eq!(cu.len(), 2 * AUD_CU_BATCH_SIZE);
        let counts: Vec<u32> = (0..2)
            .map(|k| {
                let base = k * AUD_CU_BATCH_SIZE;
                assert_eq!(
                    u32::from_le_bytes(cu[base..base + 4].try_into().unwrap()),
                    1,
                    "n_rec 恒 1"
                );
                assert_eq!(
                    u64::from_le_bytes(cu[base + 8..base + 16].try_into().unwrap()),
                    0,
                    "rsv u64 保留位写 0"
                );
                u32::from_le_bytes(cu[base + 4..base + 8].try_into().unwrap())
            })
            .collect();
        assert_eq!(counts, vec![2, 2]);
        assert_eq!(
            counts.iter().sum::<u32>(),
            4,
            "Σ aud_cu.aud_cnt == rec 记录数"
        );
    }

    /// 尾窗字节引用（P3a §4.4 裁定 b）：尾窗 rec 的 (offset, len) **复用同组
    /// 首窗** VALID rec 的值（自足定位，不重复写 blob）；引用悬空 → fail-fast。
    #[test]
    fn test_audio_tail_window_reference_and_missing_ref_fails() {
        let dir = aud_dir("tailref");
        let mut w = AudioSidecarWriter::create(&dir, "ashard_00000").unwrap();
        let clip = vec![0x5Au8; 300];
        // 样本 0 段 0：首窗 VALID + 两个尾窗（模拟 [1500,5500) W=2000 → 3 子段）
        w.append_audio_batch(&[
            aud_valid(&clip, 0, 0, 1500, 500),
            aud_tail(0, 0, 2000, 2000),
            aud_tail(0, 0, 4000, 1500),
        ])
        .unwrap();
        // 样本 1 段 0：首窗 + 尾窗（跨 batch 的尾窗引用：seg_ref 跨调用存活）
        let clip2 = vec![0x33u8; 100];
        w.append_audio_batch(&[aud_valid(&clip2, 1, 0, 0, 1000), aud_tail(1, 0, 2000, 500)])
            .unwrap();
        w.finish().unwrap();

        let rec_bytes = std::fs::read(dir.join("ashard_00000.col_aud_rec")).unwrap();
        let recs: Vec<AudioRecord> = (0..5)
            .map(|i| decode_aud_rec(&rec_bytes[i * AUD_REC_SIZE..(i + 1) * AUD_REC_SIZE]))
            .collect();
        // 组 0：首窗 (0, 300)；尾窗复用 (0, 300)，len 非 0（自足定位语义）
        assert_eq!(recs[0].flags, aud_flags::VALID);
        assert_eq!((recs[0].offset, recs[0].len), (0, 300));
        for t in &recs[1..3] {
            assert_eq!(t.flags, aud_flags::AUDIO_TAIL_WINDOW);
            assert_eq!((t.offset, t.len), (0, 300), "尾窗 (offset,len) 复用首窗值");
            assert_eq!(t.seg_order, 0, "seg_order 继承原段（组键）");
        }
        // start_ms 为子段真实起点（样本时间轴锚定，P3a §4.1）
        assert_eq!(recs[0].start_ms, 1500);
        assert_eq!(recs[1].start_ms, 2000);
        assert_eq!(recs[2].start_ms, 4000);
        // 组 1：跨 batch 引用同样成立（首窗 offset=300）
        assert_eq!((recs[3].offset, recs[3].len), (300, 100));
        assert_eq!((recs[4].offset, recs[4].len), (300, 100));
        // duration_ms 为子段时长
        assert_eq!(recs[0].duration_ms, 500);
        assert_eq!(recs[1].duration_ms, 2000);
        // blob 零膨胀：只有两个首窗的字节
        assert_eq!(
            std::fs::metadata(dir.join("ashard_00000.col_aud_blob"))
                .unwrap()
                .len(),
            400
        );

        // 引用悬空（首窗从未发射）→ fail-fast（发射序被破坏的构造性检测）
        let mut w2 = AudioSidecarWriter::create(&dir, "ashard_00001").unwrap();
        let e = w2
            .append_audio_batch(&[aud_tail(42, 7, 2000, 1000)])
            .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().contains("has no VALID record"), "{e}");
        // 整批未写（校验/查表失败先于任何写入）
        assert_eq!(
            std::fs::read(dir.join("ashard_00001.col_aud_rec"))
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            std::fs::read(dir.join("ashard_00001.col_aud_cu"))
                .unwrap()
                .len(),
            0
        );
    }

    /// API 契约（P3a §5.2）：VALID 必须带字节；尾窗不得带字节；未知 flags 拒绝；
    /// 且输入校验先于写入——首个坏记录报错时整批未写。
    #[test]
    fn test_audio_input_contract_enforced_before_write() {
        let dir = aud_dir("contract");
        let mut w = AudioSidecarWriter::create(&dir, "ashard_00000").unwrap();
        let good = vec![1u8; 8];

        // VALID 无字节 → 拒
        assert!(
            w.append_audio_batch(&[AudioRecordInput {
                bytes: None,
                flags: aud_flags::VALID,
                sample_rate: 16000,
                duration_ms: 10,
                channels: 1,
                sample_id: 0,
                seg_order: 0,
                start_ms: 0,
            }])
            .is_err()
        );
        // 尾窗带字节 → 拒
        assert!(
            w.append_audio_batch(&[AudioRecordInput {
                bytes: Some(&good),
                flags: aud_flags::AUDIO_TAIL_WINDOW,
                sample_rate: 16000,
                duration_ms: 10,
                channels: 1,
                sample_id: 0,
                seg_order: 0,
                start_ms: 0,
            }])
            .is_err()
        );
        // 未知 flags（图像值 1 不得用于 audio）→ 拒
        assert!(
            w.append_audio_batch(&[AudioRecordInput {
                bytes: None,
                flags: 1,
                sample_rate: 16000,
                duration_ms: 10,
                channels: 1,
                sample_id: 0,
                seg_order: 0,
                start_ms: 0,
            }])
            .is_err()
        );
        // 整批原子：三条非法输入均未落任何字节
        assert_eq!(
            std::fs::read(dir.join("ashard_00000.col_aud_rec"))
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            std::fs::read(dir.join("ashard_00000.col_aud_cu"))
                .unwrap()
                .len(),
            0
        );
        assert_eq!(w.record_count(), 0);

        // 合法批仍可正常写（错误不毒化 writer）
        w.append_audio_batch(&[aud_valid(&good, 0, 0, 0, 10)])
            .unwrap();
        assert_eq!(w.record_count(), 1);
        w.finish().unwrap();
    }

    // ------------------------------------------------------------------
    // P3b：video sidecar（col_vid_blob/rec + mm_cu.vid_cnt）
    // ------------------------------------------------------------------

    fn vid_valid<'a>(
        bytes: &'a [u8],
        sample_id: u64,
        seg_order: u32,
        n_frames: u16,
        fps_milli: u32,
        start_ms: u32,
    ) -> VideoRecordInput<'a> {
        VideoRecordInput {
            bytes: Some(bytes),
            n_frames,
            width: 336,
            height: 336,
            flags: vid_flags::VALID,
            sample_id,
            seg_order,
            fps_milli,
            start_ms,
        }
    }

    /// 守恒（P3b §11）：offset 连续无洞、Σ len(VALID) == blob 长度、rec 定宽 40B、
    /// encode/decode roundtrip（含 start_ms/fps_milli）。
    #[test]
    fn test_video_offset_conservation_and_roundtrip() {
        let dir = tmp_dir("vid_conserv");
        let mut w = VideoSidecarWriter::create(&dir, "shard_00000").unwrap();
        let v1 = vec![3u8; 1024];
        let v2 = vec![4u8; 2048];
        w.append_vid_batch(&[vid_valid(&v1, 7, 0, 16, 30000, 250)])
            .unwrap();
        w.append_vid_batch(&[
            vid_valid(&v2, 7, 1, 8, 29970, 4000),
            VideoRecordInput {
                bytes: None,
                n_frames: 0,
                width: 0,
                height: 0,
                flags: vid_flags::NO_VIDEO,
                sample_id: 8,
                seg_order: 0,
                fps_milli: 0,
                start_ms: 0,
            },
        ])
        .unwrap();
        let hashes = w.finish().unwrap();

        let rec_bytes = std::fs::read(dir.join("shard_00000.col_vid_rec")).unwrap();
        assert_eq!(rec_bytes.len(), 3 * VID_REC_SIZE);
        let recs: Vec<VideoRecord> = (0..3)
            .map(|i| decode_vid_rec(&rec_bytes[i * VID_REC_SIZE..(i + 1) * VID_REC_SIZE]))
            .collect();
        let mut expect_off = 0u64;
        for r in &recs {
            assert_eq!(r.offset, expect_off);
            expect_off += r.len as u64;
        }
        let blob_len = std::fs::metadata(dir.join("shard_00000.col_vid_blob"))
            .unwrap()
            .len();
        assert_eq!(expect_off, blob_len);
        assert_eq!(blob_len, (1024 + 2048) as u64, "占位记录不产字节");
        assert_eq!(recs[2].len, 0);
        // 字段 roundtrip（offset 由 writer 分配，其余逐字段）
        assert_eq!(recs[0].n_frames, 16);
        assert_eq!(recs[0].fps_milli, 30000);
        assert_eq!(recs[0].start_ms, 250);
        assert_eq!(recs[0].sample_id, 7);
        assert_eq!(recs[1].start_ms, 4000);
        assert_eq!(recs[1].fps_milli, 29970);
        // 增量哈希 == 回读（铁律）
        assert_eq!(
            hashes.get("sha256:col_vid_blob").unwrap(),
            &sha256_file_hex(&dir.join("shard_00000.col_vid_blob")).unwrap()
        );
        assert_eq!(
            hashes.get("sha256:col_vid_rec").unwrap(),
            &sha256_file_hex(&dir.join("shard_00000.col_vid_rec")).unwrap()
        );
        assert_eq!(w.record_count(), 3);
    }

    /// mm_cu.vid_cnt 填充（P3b §3）：第三计数位透出同 batch video 记录数；
    /// vid_cnt=0 时 mm_cu 字节与 P2 期逐字节一致（向后兼容）。
    #[test]
    fn test_mm_cu_vid_cnt_and_zero_compat() {
        let dir = tmp_dir("vid_cu");
        let mut w = MmSidecarWriter::create(&dir, "s").unwrap();
        let img = vec![5u8; 32];
        w.append_mm_batch(&[valid(&img, 0, 0)], 0).unwrap();
        w.append_mm_batch(&[valid(&img, 0, 0)], 2).unwrap(); // 同 batch 2 条 video rec
        w.append_mm_batch(&[], 1).unwrap();
        w.finish().unwrap();

        let cu = std::fs::read(dir.join("s.mm_cu")).unwrap();
        assert_eq!(cu.len(), 3 * MM_CU_BATCH_SIZE);
        for (k, expect_vid) in [0u32, 2, 1].iter().enumerate() {
            let base = k * MM_CU_BATCH_SIZE;
            assert_eq!(
                u32::from_le_bytes(cu[base + 12..base + 16].try_into().unwrap()),
                *expect_vid
            );
        }
        // vid_cnt=0 的 batch：mm_cu 条目与旧版（全 0 缓冲）逐字节一致
        let dir2 = tmp_dir("vid_cu_zero");
        let mut w2 = MmSidecarWriter::create(&dir2, "s").unwrap();
        w2.append_mm_batch(&[valid(&img, 0, 0)], 0).unwrap();
        w2.finish().unwrap();
        let cu2 = std::fs::read(dir2.join("s.mm_cu")).unwrap();
        assert_eq!(&cu[0..MM_CU_BATCH_SIZE], &cu2[0..MM_CU_BATCH_SIZE]);
    }

    /// API 契约（P3b §5.2）：VALID 必须带字节；占位不得带字节；未知 flags
    /// （含 audio 专用值 4）拒绝；输入校验先于写入。
    #[test]
    fn test_video_input_contract_enforced_before_write() {
        let dir = tmp_dir("vid_contract");
        let mut w = VideoSidecarWriter::create(&dir, "shard_00000").unwrap();
        let good = vec![9u8; 64];

        assert!(
            w.append_vid_batch(&[VideoRecordInput {
                bytes: None,
                n_frames: 16,
                width: 336,
                height: 336,
                flags: vid_flags::VALID,
                sample_id: 0,
                seg_order: 0,
                fps_milli: 30000,
                start_ms: 0,
            }])
            .is_err()
        );
        assert!(
            w.append_vid_batch(&[VideoRecordInput {
                bytes: Some(&good),
                n_frames: 16,
                width: 336,
                height: 336,
                flags: vid_flags::NO_VIDEO,
                sample_id: 0,
                seg_order: 0,
                fps_milli: 30000,
                start_ms: 0,
            }])
            .is_err()
        );
        // audio 专用值 4 不得用于 video
        assert!(
            w.append_vid_batch(&[VideoRecordInput {
                bytes: None,
                n_frames: 16,
                width: 336,
                height: 336,
                flags: 4,
                sample_id: 0,
                seg_order: 0,
                fps_milli: 30000,
                start_ms: 0,
            }])
            .is_err()
        );
        assert_eq!(
            std::fs::read(dir.join("shard_00000.col_vid_rec"))
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            std::fs::read(dir.join("shard_00000.col_vid_blob"))
                .unwrap()
                .len(),
            0
        );
        assert_eq!(w.record_count(), 0);

        w.append_vid_batch(&[vid_valid(&good, 0, 0, 16, 30000, 0)])
            .unwrap();
        assert_eq!(w.record_count(), 1);
        w.finish().unwrap();
    }
}

//! 多模态 sidecar 读取核心（P1-T4，格式冻结基线 v1）。
//!
//! 字节布局契约以 `builder::sidecar` 的写入实现 + `tests/test_mm_p1_sidecar.py` 的手工拼装为准。
//! 本模块只做**寻址 + 结构校验 + 字节读取**；错误语义 = fail-fast
//! （缺失/长度不符/越界/未知 flags → 带上下文的 io::Error，**不静默返回空图**）。
//!
//! 分工边界（P1）：本结构只接收已打开的 `File` 句柄——云端物化/缓存/checksum
//! 校验走 `stream.rs` 的 `open_cached` 接缝（与 `.col_doc_ids` 同模式），
//! DLPack 生命周期与预取预算（P1-T5/T6）在 stream/ffi 层落，本模块不掺和。

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::builder::sidecar::{
    IMG_REC_SIZE, ImageRecord, MM_CU_BATCH_SIZE, VID_REC_SIZE, VideoRecord, decode_img_rec,
    decode_vid_rec, img_flags, vid_flags,
};

/// batch 内共享的 blob 字节形态（P1-T6）。
///
/// - `Owned`：独立缓冲（单图路径 / 兼容；`read_batch_entries` 直读路径）。
/// - `Shared`：**整 batch 的图像字节一次读入**的共享缓冲 view（stream.rs 主
///   路径）——Arc 强引用保活 = D8"驱逐保护"堆形态（tensor 存活期 buffer 不
///   回收不复用）；多 view 单次释放（最后一个引用 Drop 时
///   `SharedBlobBuffer::release_count` +1）；`Arc::clone` 只增引用计数，
///   全链路无深拷贝（`copy_count` 恒 0，D8 零额外拷贝锚点）。
#[derive(Clone)]
pub enum BlobBytes {
    Owned(Vec<u8>),
    Shared(
        std::sync::Arc<crate::runtime::ffi_dlpack::SharedBlobBuffer>,
        u64,
        u32,
    ),
}

impl BlobBytes {
    /// 字节内容（只读借用；`Shared` 形态零拷贝切片）。
    pub fn as_slice(&self) -> &[u8] {
        match self {
            BlobBytes::Owned(v) => v.as_slice(),
            BlobBytes::Shared(buf, off, len) => {
                let (off, len) = (*off as usize, *len as usize);
                &buf.data[off..off + len]
            }
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        match self {
            BlobBytes::Owned(v) => v.len(),
            BlobBytes::Shared(_, _, len) => *len as usize,
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PartialEq for BlobBytes {
    /// 逐字节相等（跨形态比较：Shared view 与 Owned 值相等即相等）。
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for BlobBytes {}

impl std::fmt::Debug for BlobBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlobBytes::Owned(v) => f.debug_tuple("Owned").field(&v.len()).finish(),
            BlobBytes::Shared(_, off, len) => {
                f.debug_tuple("Shared").field(off).field(len).finish()
            }
        }
    }
}

/// 单条图像的 batch 透出形态（P1-T4b：`bytes` + 元数据；P1-T6 起 `bytes` 为
/// [`BlobBytes`]——绑定层转 DLPack capsule（torch 可 `from_dlpack`）时保留
/// 元数据键）。占位记录（NO_IMAGE/EMPTY_IMAGE/ERRORED，len=0）的 `bytes`
/// 为空 view——语义与 `read_blob` 一致，消费端按 `flags` 判别。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmImageEntry {
    pub bytes: BlobBytes,
    pub sample_id: u64,
    pub seg_order: u32,
    pub grid_h: u16,
    pub grid_w: u16,
    pub flags: u8,
}

/// 单条 video 的 batch 透出形态（P3b §7.1：**字节延迟**——预取只带 rec 元数据，
/// blob 字节经 `blob_ref` 由 collate 时点按需读，不进预取窗口）。
///
/// `blob_ref = (store_id, offset, len)`：`store_id` 是
/// [`crate::runtime::video_blob::VideoBlobStore`] 的句柄 id（纯数据，跨线程/
/// 跨 FFI 无所有权问题）；`offset/len` 直接可用（自足定位，读取端无需跨 rec
/// 二次查找——尾窗引用语义不适用于 video：video 每条 rec 自带完整字节区间）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmVideoEntry {
    pub sample_id: u64,
    pub seg_order: u32,
    pub n_frames: u16,
    pub width: u16,
    pub height: u16,
    pub flags: u8,
    pub fps_milli: u32,
    /// 段起点（相对样本起点，整数毫秒；P3c join 键 `window_index` 的真源）
    pub start_ms: u32,
    pub blob_ref: (u64, u64, u32),
}

/// 图像 sidecar 读取器（blob / rec / mm_cu 三句柄）。
///
/// `mm_cu` 前缀和游标缓存：顺序迭代（训练主路径）下 O(1) 摊销——每次
/// `read_batch_images` 记录 `(batch_idx, 累计 img_cnt)`，下一次顺序请求
/// 从游标续算，不重扫前缀；随机访问回退为一次性读 `[0..k)` 前缀块求和。
pub struct MmSidecarReader {
    img_blob: File,
    img_rec: File,
    mm_cu: File,
    blob_len: u64,
    rec_len: u64,
    cu_len: u64,
    /// 前缀和游标：`(已消费的 batch 数, 累计 img_cnt)`。
    cu_cursor: (u64, u32),
}

impl MmSidecarReader {
    /// 从已打开的三文件句柄构造（生产接缝：句柄由 `stream.rs` 经 `open_cached`
    /// 物化并完成 checksum 校验后传入）。
    pub fn from_files(img_blob: File, img_rec: File, mm_cu: File) -> std::io::Result<Self> {
        let blob_len = img_blob.metadata()?.len();
        let rec_len = img_rec.metadata()?.len();
        let cu_len = mm_cu.metadata()?.len();
        Ok(Self {
            img_blob,
            img_rec,
            mm_cu,
            blob_len,
            rec_len,
            cu_len,
            cu_cursor: (0, 0),
        })
    }

    /// 本地路径便捷构造（测试与本地数据集直读）。
    #[allow(dead_code)]
    pub fn open_local(dir: &Path, stem: &str) -> std::io::Result<Self> {
        Self::from_files(
            File::open(dir.join(format!("{stem}.col_img_blob")))?,
            File::open(dir.join(format!("{stem}.col_img_rec")))?,
            File::open(dir.join(format!("{stem}.mm_cu")))?,
        )
    }

    /// 该数据集 sidecar 覆盖的 batch 数（mm_cu 条目数；空文件 = 0）。
    pub fn batch_count(&self) -> u64 {
        self.cu_len / MM_CU_BATCH_SIZE as u64
    }

    /// 读第 `in_shard` 个 batch 的全部图像记录（按写入序）。
    ///
    /// 结构校验（全部 fail-fast，见格式冻结 §4/§5）：
    /// - `in_shard` 越界（mm_cu 不足）→ `UnexpectedEof`
    /// - `n_mod != 3` → `InvalidData`（未来格式，前向不兼容检测）
    /// - 记录区间超出 rec 文件 → `UnexpectedEof`
    /// - 记录 `offset+len` 超出 blob 文件 / 未知 flags → `InvalidData`
    pub fn read_batch_images(&mut self, in_shard: u64) -> std::io::Result<Vec<ImageRecord>> {
        let count = self.read_mm_cu_counts(in_shard)?.0;

        // 记录区间 = [prefix, prefix+count) × 32 字节定宽寻址
        let prefix = self.img_prefix_upto(in_shard)? as u64;
        let start = prefix
            .checked_mul(IMG_REC_SIZE as u64)
            .ok_or_else(|| corrupt("record offset overflow (rec prefix too large)"))?;
        let end = start + count as u64 * IMG_REC_SIZE as u64;
        if end > self.rec_len {
            return Err(truncated(&format!(
                "col_img_rec has {} bytes, but batch {in_shard} needs records [{prefix}, {}) \
                 ({} bytes) — file truncated or manifest inconsistent",
                self.rec_len,
                prefix + count as u64,
                end - start
            )));
        }

        let mut buf = vec![0u8; count as usize * IMG_REC_SIZE];
        self.img_rec.seek(SeekFrom::Start(start))?;
        self.img_rec.read_exact(&mut buf)?;
        let (chunks, _) = buf.as_chunks::<IMG_REC_SIZE>();
        let recs: Vec<ImageRecord> = chunks.iter().map(|c| decode_img_rec(c)).collect();
        for r in &recs {
            if !img_flags::is_known(r.flags) {
                return Err(corrupt(&format!(
                    "unknown image record flags {} (sample_id={}): format v1 only defines 0..=3",
                    r.flags, r.sample_id
                )));
            }
            // 越界即格式损坏（写入端保证 offset 连续 + Σlen == blob 长度）
            if r.offset.saturating_add(r.len as u64) > self.blob_len {
                return Err(corrupt(&format!(
                    "image record (sample_id={}, seg_order={}) points to [{}, {}) beyond \
                     col_img_blob length {} — blob/rec inconsistent",
                    r.sample_id,
                    r.seg_order,
                    r.offset,
                    r.offset + r.len as u64,
                    self.blob_len
                )));
            }
        }
        Ok(recs)
    }

    /// 读第 `in_shard` 个 batch 的全部图像（记录校验 + blob 字节一步到位，
    /// 按 `read_batch_images` 写入序）。语义 = `read_batch_images`（全部结构
    /// 校验 fail-fast）+ 逐条 `read_blob`，供测试与本地直读（Owned 形态；
    /// stream.rs 主路径走 `read_blob_range` 的共享缓冲）。
    #[allow(dead_code)]
    pub fn read_batch_entries(&mut self, in_shard: u64) -> std::io::Result<Vec<MmImageEntry>> {
        let recs = self.read_batch_images(in_shard)?;
        let mut out = Vec::with_capacity(recs.len());
        for rec in &recs {
            out.push(MmImageEntry {
                bytes: BlobBytes::Owned(self.read_blob(rec)?),
                sample_id: rec.sample_id,
                seg_order: rec.seg_order,
                grid_h: rec.grid_h,
                grid_w: rec.grid_w,
                flags: rec.flags,
            });
        }
        Ok(out)
    }

    /// 读 blob 文件 `[offset, offset+len)` 区间（一次连续读入；P1-T6 整 batch
    /// 共享缓冲的底层原语）。越界 fail-fast（与 `read_blob` 同语义）。
    pub fn read_blob_range(&mut self, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
        if offset.saturating_add(len) > self.blob_len {
            return Err(corrupt(&format!(
                "blob range [{offset}, {}) beyond col_img_blob length {} — blob/rec inconsistent",
                offset + len,
                self.blob_len
            )));
        }
        let mut buf = vec![0u8; len as usize];
        if buf.is_empty() {
            return Ok(buf);
        }
        self.img_blob.seek(SeekFrom::Start(offset))?;
        self.img_blob.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// 一个 batch 的 blob 连续跨度 `[min(VALID offset), max(offset+len))`。
    ///
    /// 写入端保证 offset 单调连续（append-only，占位记录不写字节、offset 指向
    /// 当前位置），因此 VALID 记录的字节区间连续无洞——整 batch 一次
    /// `read_blob_range(span)` 读入 = `Σ VALID len`（P1-T5 账本口径一致）。
    /// 全占位（无 VALID）返回 `None`。
    pub fn batch_blob_span(recs: &[ImageRecord]) -> Option<(u64, u64)> {
        let mut span: Option<(u64, u64)> = None;
        for r in recs {
            if r.len == 0 {
                continue;
            }
            let (s, e) = (r.offset, r.offset + r.len as u64);
            span = Some(match span {
                None => (s, e),
                Some((ms, me)) => (ms.min(s), me.max(e)),
            });
        }
        span
    }

    /// 读一条记录的 blob 字节（`read_batch_images` 校验过边界后按需调用；
    /// 占位记录 len=0 → 空 Vec）。
    #[allow(dead_code)]
    pub fn read_blob(&mut self, rec: &ImageRecord) -> std::io::Result<Vec<u8>> {
        self.read_blob_range(rec.offset, rec.len as u64)
    }

    /// 读第 `in_shard` 条 `mm_cu`，返回 `(img_cnt, aud_cnt, vid_cnt)`。
    fn read_mm_cu_counts(&mut self, in_shard: u64) -> std::io::Result<(u32, u32, u32)> {
        let offset = in_shard
            .checked_mul(MM_CU_BATCH_SIZE as u64)
            .ok_or_else(|| corrupt("mm_cu seek offset overflow"))?;
        if offset + MM_CU_BATCH_SIZE as u64 > self.cu_len {
            return Err(truncated(&format!(
                "mm_cu covers only {} batches, but batch {in_shard} was requested",
                self.batch_count()
            )));
        }
        let mut b = [0u8; MM_CU_BATCH_SIZE];
        self.mm_cu.seek(SeekFrom::Start(offset))?;
        self.mm_cu.read_exact(&mut b)?;
        let n_mod = u32::from_le_bytes(b[0..4].try_into().unwrap());
        if n_mod != 3 {
            return Err(corrupt(&format!(
                "mm_cu entry {in_shard} declares n_mod={n_mod}, but format v1 defines 3 — \
                 dataset written by a future/incompatible version?"
            )));
        }
        Ok((
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
            u32::from_le_bytes(b[8..12].try_into().unwrap()),
            u32::from_le_bytes(b[12..16].try_into().unwrap()),
        ))
    }

    /// `Σ_{j<k} img_cnt[j]`：顺序访问从游标续算（O(1) 摊销），随机访问
    /// 一次读 `[0..k)` 前缀块求和（O(k) 单次，16 字节/条，成本可忽略）。
    fn img_prefix_upto(&mut self, k: u64) -> std::io::Result<u32> {
        if k == 0 {
            self.cu_cursor = (0, 0);
            return Ok(0);
        }
        let (done, acc) = self.cu_cursor;
        let (start, mut sum) = if done <= k && done > 0 {
            (done, acc)
        } else {
            (0, 0)
        };
        // 从 start 续读 (k - start) 条
        self.mm_cu
            .seek(SeekFrom::Start(start * MM_CU_BATCH_SIZE as u64))?;
        let mut buf = vec![0u8; (k - start) as usize * MM_CU_BATCH_SIZE];
        self.mm_cu.read_exact(&mut buf)?;
        let (chunks, _) = buf.as_chunks::<MM_CU_BATCH_SIZE>();
        for chunk in chunks {
            sum = sum.saturating_add(u32::from_le_bytes(chunk[4..8].try_into().unwrap()));
        }
        self.cu_cursor = (k, sum);
        Ok(sum)
    }

    /// `mm_cu` 句柄的共享克隆（P3b：`VideoSidecarReader` 需要同文件的独立读
    /// 游标读 `vid_cnt`——`try_clone` 共享 fd，不重复打开）。
    pub fn mm_cu_try_clone(&self) -> std::io::Result<File> {
        self.mm_cu.try_clone()
    }
}

/// video sidecar 读取器（P3b §7.5：`col_vid_rec` 定宽 + `mm_cu` 共享计数）。
///
/// **不持有 `col_vid_blob` 读句柄**：blob 字节由 collate 时点经
/// [`crate::runtime::video_blob::VideoBlobStore`] 按需读（句柄注册表持有，
/// 本 reader 只带 `blob_len` 做边界校验）——避免双句柄重复打开。
pub struct VideoSidecarReader {
    vid_rec: File,
    rec_len: u64,
    /// `mm_cu` 共享句柄（独立读游标；与 `MmSidecarReader` 的互不干扰）
    mm_cu: File,
    cu_len: u64,
    /// `col_vid_blob` 长度（仅校验用）
    vid_blob_len: u64,
    /// 前缀和游标：`(已消费的 batch 数, 累计 vid_cnt)`（同 img 先例）
    cu_cursor: (u64, u32),
}

impl VideoSidecarReader {
    /// `vid_rec` + `mm_cu` 共享句柄 + `vid_blob_len`（blob 句柄归
    /// `VideoBlobStore`，这里只带长度）。
    pub fn from_files(vid_rec: File, mm_cu: File, vid_blob_len: u64) -> std::io::Result<Self> {
        let rec_len = vid_rec.metadata()?.len();
        let cu_len = mm_cu.metadata()?.len();
        Ok(Self {
            vid_rec,
            rec_len,
            mm_cu,
            cu_len,
            vid_blob_len,
            cu_cursor: (0, 0),
        })
    }

    /// 该 sidecar 覆盖的 batch 数（mm_cu 条目数）。
    pub fn batch_count(&self) -> u64 {
        self.cu_len / MM_CU_BATCH_SIZE as u64
    }

    /// 读第 `in_shard` 个 batch 的全部 video 记录（40B 定宽寻址；vid_cnt 取
    /// 自共享 `mm_cu` 第三计数位——格式冻结 §2.4）。
    ///
    /// 结构校验（全部 fail-fast，同 `read_batch_images` 语义）：
    /// - `in_shard` 越界 / `n_mod != 3` → 截断/损坏错误
    /// - 记录区间超出 rec 文件 → `UnexpectedEof`
    /// - 记录 `offset+len` 超出 `col_vid_blob` 长度 / 未知 flags → `InvalidData`
    pub fn read_batch_videos(&mut self, in_shard: u64) -> std::io::Result<Vec<VideoRecord>> {
        let count = self.read_mm_cu_counts(in_shard)?.2;
        let prefix = self.vid_prefix_upto(in_shard)? as u64;
        let start = prefix
            .checked_mul(VID_REC_SIZE as u64)
            .ok_or_else(|| corrupt("video record offset overflow (rec prefix too large)"))?;
        let end = start + count as u64 * VID_REC_SIZE as u64;
        if end > self.rec_len {
            return Err(truncated(&format!(
                "col_vid_rec has {} bytes, but batch {in_shard} needs records [{prefix}, {}) \
                 ({} bytes) — file truncated or manifest inconsistent",
                self.rec_len,
                prefix + count as u64,
                end - start
            )));
        }
        let mut buf = vec![0u8; count as usize * VID_REC_SIZE];
        self.vid_rec.seek(SeekFrom::Start(start))?;
        self.vid_rec.read_exact(&mut buf)?;
        let (chunks, _) = buf.as_chunks::<VID_REC_SIZE>();
        let recs: Vec<VideoRecord> = chunks.iter().map(|c| decode_vid_rec(c)).collect();
        for r in &recs {
            // video flags 值域 0..=3（值 4 是 audio 专用，video 不得使用）
            if !vid_flags::is_known(r.flags) {
                return Err(corrupt(&format!(
                    "unknown video record flags {} (sample_id={}): video flags are 0..=3 \
                     (value 4 is audio-only)",
                    r.flags, r.sample_id
                )));
            }
            if r.offset.saturating_add(r.len as u64) > self.vid_blob_len {
                return Err(corrupt(&format!(
                    "video record (sample_id={}, seg_order={}) points to [{}, {}) beyond \
                     col_vid_blob length {} — blob/rec inconsistent",
                    r.sample_id,
                    r.seg_order,
                    r.offset,
                    r.offset + r.len as u64,
                    self.vid_blob_len
                )));
            }
        }
        Ok(recs)
    }

    /// 读第 `in_shard` 条 `mm_cu`（随机访问：seek + 读 16B；与 img 侧各自持有
    /// 独立游标，互不干扰）。
    fn read_mm_cu_counts(&mut self, in_shard: u64) -> std::io::Result<(u32, u32, u32)> {
        let offset = in_shard
            .checked_mul(MM_CU_BATCH_SIZE as u64)
            .ok_or_else(|| corrupt("mm_cu seek offset overflow"))?;
        if offset + MM_CU_BATCH_SIZE as u64 > self.cu_len {
            return Err(truncated(&format!(
                "mm_cu covers only {} batches, but batch {in_shard} was requested",
                self.batch_count()
            )));
        }
        let mut b = [0u8; MM_CU_BATCH_SIZE];
        self.mm_cu.seek(SeekFrom::Start(offset))?;
        self.mm_cu.read_exact(&mut b)?;
        let n_mod = u32::from_le_bytes(b[0..4].try_into().unwrap());
        if n_mod != 3 {
            return Err(corrupt(&format!(
                "mm_cu entry {in_shard} declares n_mod={n_mod}, but format v1 defines 3 — \
                 dataset written by a future/incompatible version?"
            )));
        }
        Ok((
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
            u32::from_le_bytes(b[8..12].try_into().unwrap()),
            u32::from_le_bytes(b[12..16].try_into().unwrap()),
        ))
    }

    /// `Σ_{j<k} vid_cnt[j]`（同 `img_prefix_upto` 的游标/随机两路语义）。
    fn vid_prefix_upto(&mut self, k: u64) -> std::io::Result<u32> {
        if k == 0 {
            self.cu_cursor = (0, 0);
            return Ok(0);
        }
        let (done, acc) = self.cu_cursor;
        let (start, mut sum) = if done <= k && done > 0 {
            (done, acc)
        } else {
            (0, 0)
        };
        self.mm_cu
            .seek(SeekFrom::Start(start * MM_CU_BATCH_SIZE as u64))?;
        let mut buf = vec![0u8; (k - start) as usize * MM_CU_BATCH_SIZE];
        self.mm_cu.read_exact(&mut buf)?;
        let (chunks, _) = buf.as_chunks::<MM_CU_BATCH_SIZE>();
        for chunk in chunks {
            sum = sum.saturating_add(u32::from_le_bytes(chunk[12..16].try_into().unwrap()));
        }
        self.cu_cursor = (k, sum);
        Ok(sum)
    }
}

fn corrupt(msg: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("mm sidecar corrupt: {msg}"),
    )
}

fn truncated(msg: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        format!("mm sidecar truncated: {msg}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::sidecar::{
        ImageRecordInput, MmSidecarWriter, VideoRecordInput, VideoSidecarWriter,
    };

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("megadata_mm_reader_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_dataset(dir: &Path) -> Vec<Vec<u8>> {
        let mut w = MmSidecarWriter::create(dir, "s").unwrap();
        let imgs: Vec<Vec<u8>> = vec![vec![1u8; 10], vec![2u8; 20], vec![3u8; 5]];
        w.append_mm_batch(
            &[ImageRecordInput {
                bytes: Some(&imgs[0]),
                grid_h: 2,
                grid_w: 2,
                flags: img_flags::VALID,
                sample_id: 0,
                seg_order: 0,
            }],
            0,
        )
        .unwrap();
        w.append_mm_batch(&[], 0).unwrap(); // 纯文本 batch：0 图
        w.append_mm_batch(
            &[
                ImageRecordInput {
                    bytes: Some(&imgs[1]),
                    grid_h: 3,
                    grid_w: 3,
                    flags: img_flags::VALID,
                    sample_id: 7,
                    seg_order: 0,
                },
                ImageRecordInput {
                    bytes: None,
                    grid_h: 0,
                    grid_w: 0,
                    flags: img_flags::NO_IMAGE,
                    sample_id: 8,
                    seg_order: 0,
                },
            ],
            0,
        )
        .unwrap();
        w.append_mm_batch(
            &[ImageRecordInput {
                bytes: Some(&imgs[2]),
                grid_h: 1,
                grid_w: 1,
                flags: img_flags::VALID,
                sample_id: 9,
                seg_order: 2,
            }],
            0,
        )
        .unwrap();
        w.finish().unwrap();
        imgs
    }

    /// P3b：图 + 视混合数据集（mm_cu vid_cnt 非零；col_vid_* 与 col_img_* 同 shard）。
    fn write_video_dataset(dir: &Path) -> Vec<Vec<u8>> {
        let mut mm = MmSidecarWriter::create(dir, "s").unwrap();
        let mut vid = VideoSidecarWriter::create(dir, "s").unwrap();
        let clips: Vec<Vec<u8>> = vec![vec![0xAAu8; 300], vec![0xBBu8; 150]];
        // batch 0：1 图 + 1 视频
        mm.append_mm_batch(
            &[ImageRecordInput {
                bytes: Some(&[1u8; 10]),
                grid_h: 1,
                grid_w: 1,
                flags: img_flags::VALID,
                sample_id: 0,
                seg_order: 0,
            }],
            1,
        )
        .unwrap();
        vid.append_vid_batch(&[VideoRecordInput {
            bytes: Some(&clips[0]),
            n_frames: 16,
            width: 336,
            height: 336,
            flags: vid_flags::VALID,
            sample_id: 0,
            seg_order: 1,
            fps_milli: 30000,
            start_ms: 250,
        }])
        .unwrap();
        // batch 1：纯文本（0 图 0 视频）
        mm.append_mm_batch(&[], 0).unwrap();
        // batch 2：1 视频（无图）
        mm.append_mm_batch(&[], 1).unwrap();
        vid.append_vid_batch(&[VideoRecordInput {
            bytes: Some(&clips[1]),
            n_frames: 8,
            width: 224,
            height: 224,
            flags: vid_flags::VALID,
            sample_id: 5,
            seg_order: 0,
            fps_milli: 29970,
            start_ms: 4000,
        }])
        .unwrap();
        mm.finish().unwrap();
        vid.finish().unwrap();
        clips
    }

    /// P3b：`read_batch_videos` 定宽寻址 + 字段 roundtrip + 守恒
    /// （Σ vid_cnt == rec 数；Σ len == blob 长度）。
    // chunks_exact 保持：`as_chunks` 需 Rust 1.88，本 crate 的 MSRV 是 1.85。
    #[allow(clippy::chunks_exact_to_as_chunks)]
    #[test]
    fn test_read_batch_videos_roundtrip_and_conservation() {
        let dir = tmp_dir("vid");
        let clips = write_video_dataset(&dir);
        let blob_len = std::fs::metadata(dir.join("s.col_vid_blob")).unwrap().len();
        // mm_cu 共享句柄（try_clone 语义）+ rec + blob 长度
        let mm_cu = File::open(dir.join("s.mm_cu")).unwrap();
        let mut r = VideoSidecarReader::from_files(
            File::open(dir.join("s.col_vid_rec")).unwrap(),
            mm_cu,
            blob_len,
        )
        .unwrap();
        assert_eq!(r.batch_count(), 3);

        let b0 = r.read_batch_videos(0).unwrap();
        assert_eq!(b0.len(), 1);
        assert_eq!(b0[0].sample_id, 0);
        assert_eq!(b0[0].seg_order, 1);
        assert_eq!(b0[0].n_frames, 16);
        assert_eq!(b0[0].width, 336);
        assert_eq!(b0[0].fps_milli, 30000);
        assert_eq!(b0[0].start_ms, 250, "start_ms roundtrip（P3c join 键真源）");
        assert_eq!(b0[0].offset, 0);
        assert_eq!(b0[0].len as usize, clips[0].len());

        assert!(
            r.read_batch_videos(1).unwrap().is_empty(),
            "纯文本 batch 无 video rec"
        );

        let b2 = r.read_batch_videos(2).unwrap();
        assert_eq!(b2.len(), 1);
        assert_eq!(b2[0].sample_id, 5);
        assert_eq!(b2[0].start_ms, 4000);
        assert_eq!(
            b2[0].offset as usize,
            clips[0].len(),
            "offset 连续（append-only）"
        );

        // 守恒：Σ rec len == blob 长度；Σ mm_cu.vid_cnt == rec 数
        let total_len: u64 = [b0[0].len as u64, b2[0].len as u64].iter().sum();
        assert_eq!(total_len, blob_len);
        let cu = std::fs::read(dir.join("s.mm_cu")).unwrap();
        let vid_sum: u32 = cu
            .chunks_exact(MM_CU_BATCH_SIZE)
            .map(|c| u32::from_le_bytes(c[12..16].try_into().unwrap()))
            .sum();
        assert_eq!(vid_sum, 2);

        // 随机访问回跳（游标正确性）：batch 0 再读一次结果一致
        let again = r.read_batch_videos(0).unwrap();
        assert_eq!(again, b0);
    }

    /// P3b：rec 越界 / 未知 flags → fail-fast（不静默）。
    #[test]
    fn test_read_batch_videos_fail_fast_on_corruption() {
        let dir = tmp_dir("vid_bad");
        write_video_dataset(&dir);
        // 篡改 rec 的 offset（越界）
        let rec_path = dir.join("s.col_vid_rec");
        let mut bytes = std::fs::read(&rec_path).unwrap();
        bytes[0..8].copy_from_slice(&9999u64.to_le_bytes());
        std::fs::write(&rec_path, &bytes).unwrap();
        let mm_cu = File::open(dir.join("s.mm_cu")).unwrap();
        let mut r = VideoSidecarReader::from_files(
            File::open(dir.join("s.col_vid_rec")).unwrap(),
            mm_cu,
            300 + 150,
        )
        .unwrap();
        let e = r.read_batch_videos(0).unwrap_err();
        assert!(e.to_string().contains("beyond col_vid_blob length"), "{e}");
    }

    #[test]
    fn test_roundtrip_batch_records_and_blob_bytes() {
        let dir = tmp_dir("roundtrip");
        let imgs = write_dataset(&dir);
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        assert_eq!(r.batch_count(), 4);

        let b0 = r.read_batch_images(0).unwrap();
        assert_eq!(b0.len(), 1);
        assert_eq!(r.read_blob(&b0[0]).unwrap(), imgs[0]);

        let b1 = r.read_batch_images(1).unwrap(); // 空 batch
        assert!(b1.is_empty());

        let b2 = r.read_batch_images(2).unwrap();
        assert_eq!(b2.len(), 2);
        assert_eq!(b2[0].sample_id, 7);
        assert_eq!(r.read_blob(&b2[0]).unwrap(), imgs[1]);
        assert_eq!(b2[1].flags, img_flags::NO_IMAGE);
        assert!(r.read_blob(&b2[1]).unwrap().is_empty()); // 占位无字节

        let b3 = r.read_batch_images(3).unwrap();
        assert_eq!(r.read_blob(&b3[0]).unwrap(), imgs[2]);
        assert_eq!(b3[0].seg_order, 2);
    }

    #[test]
    fn test_random_access_after_sequential() {
        // 随机寻址（置换 shuffle 场景）：前缀和缓存正确性——顺序后跳回、
        // 直接随机访问结果一致。
        let dir = tmp_dir("random");
        write_dataset(&dir);
        let mut seq = MmSidecarReader::open_local(&dir, "s").unwrap();
        let mut rnd = MmSidecarReader::open_local(&dir, "s").unwrap();

        for k in 0..4 {
            let a = seq.read_batch_images(k).unwrap();
            let b = rnd.read_batch_images(k).unwrap();
            assert_eq!(a, b);
        }
        // 跳回已读位置（游标不能给出过期前缀）
        let back = seq.read_batch_images(1).unwrap();
        assert!(back.is_empty());
        let back2 = rnd.read_batch_images(3).unwrap();
        assert_eq!(back2.len(), 1);
    }

    #[test]
    fn test_read_batch_entries_assembles_bytes_and_metadata() {
        let dir = tmp_dir("entries");
        let imgs = write_dataset(&dir);
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        let e0 = r.read_batch_entries(0).unwrap();
        assert_eq!(e0.len(), 1);
        assert_eq!(e0[0].bytes.as_slice(), imgs[0].as_slice());
        assert_eq!(e0[0].sample_id, 0);
        assert_eq!(e0[0].flags, img_flags::VALID);
        // 纯文本 batch：空 entries（不是错误）
        assert!(r.read_batch_entries(1).unwrap().is_empty());
        // batch 2：VALID + NO_IMAGE 占位（bytes 为空）
        let e2 = r.read_batch_entries(2).unwrap();
        assert_eq!(e2.len(), 2);
        assert_eq!(e2[0].bytes.as_slice(), imgs[1].as_slice());
        assert_eq!(e2[1].flags, img_flags::NO_IMAGE);
        assert!(e2[1].bytes.is_empty());
        assert_eq!(e2[1].sample_id, 8);
    }

    #[test]
    fn test_batch_blob_span_and_range_read() {
        let dir = tmp_dir("span");
        let imgs = write_dataset(&dir);
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        let recs2 = r.read_batch_images(2).unwrap();
        // batch 2：VALID(20B) + NO_IMAGE 占位 → 跨度 = VALID 的 [offset, offset+20)
        let span = MmSidecarReader::batch_blob_span(&recs2).unwrap();
        assert_eq!(
            span.1 - span.0,
            imgs[1].len() as u64,
            "Σ VALID len == span（账本口径一致）"
        );
        let data = r.read_blob_range(span.0, span.1 - span.0).unwrap();
        assert_eq!(data, imgs[1]);
        // 全占位 batch（b1 空 recs / 全 len=0）→ None
        assert!(MmSidecarReader::batch_blob_span(&[]).is_none());
        let empty = vec![ImageRecord {
            offset: 5,
            len: 0,
            grid_h: 0,
            grid_w: 0,
            flags: img_flags::NO_IMAGE,
            sample_id: 0,
            seg_order: 0,
        }];
        assert!(MmSidecarReader::batch_blob_span(&empty).is_none());
        // 越界 range fail-fast
        let err = r.read_blob_range(0, u64::MAX / 2).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_batch_beyond_mm_cu_fails_fast() {
        let dir = tmp_dir("beyond");
        write_dataset(&dir);
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        let err = r.read_batch_images(4).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        assert!(err.to_string().contains("only 4 batches"), "{err}");
    }

    #[test]
    fn test_tampered_n_mod_rejected() {
        let dir = tmp_dir("nmod");
        write_dataset(&dir);
        let p = dir.join("s.mm_cu");
        let mut cu = std::fs::read(&p).unwrap();
        cu[0..4].copy_from_slice(&5u32.to_le_bytes()); // n_mod: 3 → 5
        std::fs::write(&p, cu).unwrap();
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        let err = r.read_batch_images(0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("n_mod=5"), "{err}");
    }

    #[test]
    fn test_tampered_rec_truncated_and_blob_bounds_rejected() {
        let dir = tmp_dir("tamper");
        write_dataset(&dir);

        // rec 文件截半 → 记录区间越界（batch 0 只要 1 条 32B，仍在截半后的
        // 64B 内；batch 2 要 [32,96) 越界）
        let rec_p = dir.join("s.col_img_rec");
        let rec = std::fs::read(&rec_p).unwrap();
        std::fs::write(&rec_p, &rec[..rec.len() / 2]).unwrap();
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        assert!(r.read_batch_images(0).is_ok(), "前缀记录不受尾部截短影响");
        let err = r.read_batch_images(2).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);

        // blob 文件截短 → 引用尾部的 batch 3 记录越界（写入端保证 Σlen == blob 长度；
        // batch 0 的记录仍在前缀内、不受影响——这本身也是边界正确性的一部分）
        let dir2 = tmp_dir("tamper2");
        write_dataset(&dir2);
        let blob_p = dir2.join("s.col_img_blob");
        let blob = std::fs::read(&blob_p).unwrap();
        std::fs::write(&blob_p, &blob[..blob.len() - 1]).unwrap();
        let mut r2 = MmSidecarReader::open_local(&dir2, "s").unwrap();
        assert!(r2.read_batch_images(0).is_ok(), "前缀记录不受尾部截短影响");
        let err2 = r2.read_batch_images(3).unwrap_err();
        assert_eq!(err2.kind(), std::io::ErrorKind::InvalidData);
        assert!(err2.to_string().contains("beyond"), "{err2}");
    }

    #[test]
    fn test_unknown_flags_fail_fast() {
        let dir = tmp_dir("flags");
        write_dataset(&dir);
        let rec_p = dir.join("s.col_img_rec");
        let mut rec = std::fs::read(&rec_p).unwrap();
        rec[16] = 9; // flags → 未知值
        std::fs::write(&rec_p, rec).unwrap();
        let mut r = MmSidecarReader::open_local(&dir, "s").unwrap();
        let err = r.read_batch_images(0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("flags 9"), "{err}");
    }
}

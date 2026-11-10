use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;
use flume::{Receiver, RecvTimeoutError, Sender, TryRecvError, bounded, unbounded};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::builder::metadata::{DatasetManifest, ShardMeta, StorageFormat};
use crate::builder::sidecar::ImageRecord;
use crate::cancel::CancellationToken;
use crate::io::CloudIoManager;
use crate::runtime::align::{AlignEntry, join_align_entries};
use crate::runtime::mm_sidecar::{
    BlobBytes, MmImageEntry, MmSidecarReader, MmVideoEntry, VideoSidecarReader,
};
use crate::runtime::sampler::{
    DistributedSampler, PartitionAlgo, ShardOrder, World, batch_permutation,
};

// =====================================================================
// 1. 跨线程通信协议定义 (代际栅栏 + 可中断)
// =====================================================================
pub enum BatchData {
    Arrow(RecordBatch),
    /// PT (Memmap/Zstd-MDS) 模式的 batch。
    /// - input_ids: 预打包的 token id（batch_size * max_seq_len 个 i32）
    /// - cu_seqlens: batch 内文档段边界（FlashAttention varlen 用）。
    ///   旧数据集无 .cu 文件时为空 Vec（标准 attention 语义，向后兼容）。
    /// - doc_ids / loss_weights: 文档级旁路列（列扩展）。`None` = 该数据集没有这一列
    ///   （构建期未开启，或旧数据集）——消费端按"dict 里没有这个 key"处理。
    /// - images: 多模态图像 sidecar（P1-T4b）。`None` = 数据集没有 mm sidecar
    ///   （旧数据集，schema 无 img_rec/img_blob/mm_cu 三 key）——同样按 key 缺失处理。
    /// - videos: 多模态视频 sidecar 记录（P3b）。`None` = 数据集无 video sidecar
    ///   （schema 无 vid_rec/vid_blob/mm_cu 三 key，或无 video 记录）；条目
    ///   **不含 blob 字节**（字节延迟：blob_ref 按需读，D8 不进预取窗口）。
    /// - video_pin: **P1-2（评审修复）**本 batch 引用的 video blob 句柄 pin
    ///   （RAII：随 batch drop 归还）。reader 切 shard/析构时注销只标记
    ///   retired，最后一个 pin 归还才真正关闭句柄——未消费 batch 的按需读
    ///   不会因代际切换拿到 NotFound。
    Memmap {
        input_ids: Vec<i32>,
        cu_seqlens: Vec<i32>,
        doc_ids: Option<Vec<i32>>,
        loss_weights: Option<Vec<u8>>,
        images: Option<Vec<MmImageEntry>>,
        videos: Option<Vec<MmVideoEntry>>,
        /// P3c：三模态对齐视图（请求 align 时；缺省不出现 = 旧 trainer 零影响）
        av_align: Option<Vec<AlignEntry>>,
        video_pin: Option<crate::runtime::video_blob::VideoBlobPin>,
    },
}

/// 文档级旁路列的文件后缀（列扩展）：`shard_00000.bin` → `.col_doc_ids` / `.col_loss_weights`
pub(crate) const COL_DOC_IDS_EXT: &str = "col_doc_ids";
pub(crate) const COL_LOSS_WEIGHTS_EXT: &str = "col_loss_weights";

/// 发送给主线程的数据包，盖上"生产时的代际戳"
pub struct PrefetchedBatch {
    pub data: BatchData,
    pub seek_gen: u64,
    /// **P1-T5（D8）**本 batch 的字节账单（四类计量，见
    /// `StreamReaderOptions::prefetch_bytes_limit`）。预算纪律：预取线程
    /// **send 成功后 +cost**；前台取走（含陈旧丢弃）**立即 -cost**——任何路径
    /// 上"账本与在途 batch"必须守恒，否则预算门会虚高卡死或虚低失效。
    pub byte_cost: u64,
}

pub enum ControlCmd {
    /// 包含目标 **(epoch, epoch 内位置)**，以及主线程分配的新代际戳。
    ///
    /// v3（R2/B1）：不再传"全局 batch 下标"——`epoch = idx // per_epoch` 的推导只在
    /// "所有 epoch 等长"时成立，而每个 epoch 的 shard 集合与尾块大小都在变（实测
    /// round_robin 下 5/8 个 worker 的轮长逐轮不同），除法会把 Python 侧的轮号
    /// 静默解到错误的 epoch。故直接传二元组，由 Rust 侧 `load_epoch_shards(epoch)`
    /// 定位。
    Seek {
        epoch: u64,
        in_epoch: u64,
        gen_id: u64,
    },
}

/// 预取线程的退出原因：`try_next` 返回 `Err` 时的载荷。
///
/// 修复"静默重试掩盖 IO 错误"：此前后台把 `None` 一律当 EOF，遇到 IO / 解压 /
/// 数据损坏只会 200ms 轮询后无限重试，消费端表现为"一直没数据"。现在三态分明：
///
/// - `Ok(Some)`：拿到 batch；
/// - `Ok(None)`：本次轮询超时（调用方重试）；
/// - `Err(ReaderExit::Eof)`：**本轮读完**（R1：预取线程自然 EOF 后保持存活并置
///   `at_eof`，`try_next` 据此**幂等**返回 Eof，直到下次 `seek_to_epoch_batch`
///   清除标志）；线程退出（取消 / 致命错误后的通道断开）同样落到这里；
/// - `Err(ReaderExit::Fatal(msg))`：**致命错误**，消费者必须上报（Python 侧抛 OSError），
///   绝不能再静默重试。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReaderExit {
    Eof,
    Fatal(String),
}

// 内部维护的纯物理状态（对主线程不可见）
//
// v3（R5/D1）：**没有** `num_epochs` 字段——epoch 归 Python 驱动（epoch-per-iter），
// Rust 侧"加载哪个 epoch 的 shards 读完即 EOF-park"，等 `seek_to_epoch_batch` 推进。
#[derive(Clone)]
struct InnerState {
    epoch: u64,
    shard_list_index: usize,
    batches_yielded_in_shard: u64,
    global_batch_idx: u64,
}

/// 单次 try_next 轮询的最长阻塞时间。保证主线程能周期性重获 GIL 以响应信号。
const POLL_TIMEOUT: Duration = Duration::from_millis(200);

/// PT 裸流一个 batch 的产出：(input_ids, cu_seqlens, doc_ids?, loss_weights?, images?, videos?, video_pin?, av_align?, byte_cost)。
///
/// 类型别名（原为四元组内联，`clippy::type_complexity` 长期报警）：`doc_ids` /
/// `loss_weights` 是列扩展旁路列，仅在构建期 `extra_columns=True` 时为 `Some`；
/// `images` 是多模态图像 sidecar（P1-T4b），仅当 manifest schema 三 key 齐全时为
/// `Some`；`byte_cost` 是 D8 预算账单（send 时入账、取走时归还）。
/// P3b：`videos` 为 `Some` 时必带 `video_pin`（P1-2：batch 存活期 pin 住
/// `col_vid_blob` 句柄，防切 shard 后未消费 batch 按需读失败）。
type PtBatchData = (
    Vec<i32>,
    Vec<i32>,
    Option<Vec<i32>>,
    Option<Vec<u8>>,
    Option<Vec<MmImageEntry>>,
    Option<(Vec<MmVideoEntry>, crate::runtime::video_blob::VideoBlobPin)>,
    Option<Vec<AlignEntry>>,
    u64,
);

/// P3c：从 manifest `align_fingerprint` 解析的对齐参数（读侧 join 的输入）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct AlignParams {
    /// 窗长（整数毫秒；终稿 DP1）
    pub w_ms: u32,
    /// ts token 区间基址（D6 alignment-marker 子区间）
    pub ts_base_id: i32,
    /// ts token 区间预留宽度
    pub ts_range_size: u32,
    /// ts token 桶宽（默认档 = W_ms；细粒度档独立）
    pub bucket_width_ms: u32,
}

/// P3c：align_fingerprint 的 modality id ranges 快照（非文本区间）。
///
/// P1-1（评审修复）：读侧无文本段边界，text token 区间末端旧实现取到行尾，
/// 把行内后续 Image/Video placeholder 纳入"文本区间"。现在按 manifest 的
/// modality id ranges（image/audio/video；**不含 timestamp**——ts token 本身
/// 就是区间边界）在首个非文本 token 处截断。
#[derive(Debug, Clone)]
pub(crate) struct VisualRanges {
    ranges: Vec<(i32, i32)>,
}

impl VisualRanges {
    pub fn as_slice(&self) -> &[(i32, i32)] {
        &self.ranges
    }
}

/// P3c §7.2：从 manifest 解析对齐参数（指纹门槛）。
///
/// - `Ok(None)` = 数据集未启用对齐（无 `align_fingerprint`）；
/// - `Ok(Some)` = 参数完备；
/// - `Err` = 指纹自相矛盾（缺 `W_ms` / ts_token 子键缺失或非法 / ts 区间与
///   文本 vocab 相交）——fail-fast，禁静默降级。
///
/// **P3-1（评审修复）：全部 checked conversion + 构造前校验**。旧实现
/// `u64→u32` / `i64→i32` 用 `as`（超范围静默回绕），且未校验
/// `W_ms>0` / `bucket_width_ms>0` / `range_size>0` / `base_id>=0` /
/// `base_id+range_size` 加法溢出。`W_ms=0` 时 text 的 `bucket_to_window` 与
/// A/V 的 `w_ms.max(1)` 防御语义不一致，会产生不同 window——零窗宽现在直接
/// 判非法。
pub(crate) fn parse_align_params(
    manifest: &DatasetManifest,
) -> std::io::Result<(Option<AlignParams>, Option<VisualRanges>)> {
    let Some(fp) = &manifest.align_fingerprint else {
        return Ok((None, None));
    };
    let get_u32 = |k: &str| -> Option<u32> {
        fp.get(k)
            .and_then(|v| v.as_u64())
            .and_then(|x| u32::try_from(x).ok())
    };
    let w_ms = get_u32("W_ms").ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "align_fingerprint is self-contradictory: missing W_ms (granularity=G2 requires it)",
        )
    })?;
    if w_ms == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "align_fingerprint.W_ms must be > 0 (zero window width would make text and \
             A/V window mapping diverge)",
        ));
    }
    let ts = fp.get("ts_token").ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "align_fingerprint is self-contradictory: missing ts_token (base_id/bucket_width_ms/range_size)",
        )
    })?;
    let ts_base_id = ts
        .get("base_id")
        .and_then(|v| v.as_i64())
        .and_then(|x| i32::try_from(x).ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "align_fingerprint.ts_token.base_id missing or not an integer in i32 range",
            )
        })?;
    if ts_base_id < 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "align_fingerprint.ts_token.base_id must be >= 0 (got {ts_base_id}): \
                 token ids are non-negative"
            ),
        ));
    }
    let ts_range_size = ts
        .get("range_size")
        .and_then(|v| v.as_u64())
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "align_fingerprint.ts_token.range_size missing or not an integer in u32 range",
            )
        })?;
    if ts_range_size == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "align_fingerprint.ts_token.range_size must be > 0 (zero-width ts token range \
             cannot encode any window bucket)",
        ));
    }
    let bucket_width_ms = ts
        .get("bucket_width_ms")
        .and_then(|v| v.as_u64())
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "align_fingerprint.ts_token.bucket_width_ms missing or not an integer in u32 range",
            )
        })?;
    if bucket_width_ms == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "align_fingerprint.ts_token.bucket_width_ms must be > 0 (zero bucket width would \
             collapse all buckets onto one window)",
        ));
    }
    // base_id + range_size 不得溢出 i32（ts token 区间 = [base_id, base_id+range_size)）
    let ts_end = ts_base_id
        .checked_add(ts_range_size as i32)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "align_fingerprint.ts_token base_id {ts_base_id} + range_size {ts_range_size} \
                 overflows i32 — manifest is corrupt"
                ),
            )
        })?;
    // ts 区间与文本 vocab 相交 → 构建期已挡（D6）；读取侧 vocab 已知时复核
    if let Some(vsz) = manifest.tokenizer.as_ref().and_then(|t| t.vocab_size)
        && (ts_base_id as u32) < vsz
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "align_fingerprint.ts_token.base_id {ts_base_id} overlaps text vocab [0,{vsz})                      — manifest self-contradictory"
            ),
        ));
    }
    // P1-1：非文本（image/audio/video）token id 区间——text 区间的截断依据。
    // 优先取指纹内的 modality_id_ranges 快照（含 ts 区间登记，须排除
    // "timestamp"  kind）；缺快照时回落 manifest 顶层 modality_id_ranges。
    // 两者皆缺而对齐启用 = manifest 自相矛盾（构建期恒写）→ fail-fast。
    let mut visual: Vec<(i32, i32)> = Vec::new();
    if let Some(snapshot) = fp.get("modality_id_ranges").and_then(|v| v.as_array()) {
        for r in snapshot {
            let kind = r.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            if kind == "timestamp" || kind == "text" {
                continue; // ts token 即区间边界，不计入"非文本"
            }
            let (Some(s), Some(e)) = (
                r.get("start")
                    .and_then(|v| v.as_i64())
                    .and_then(|x| i32::try_from(x).ok()),
                r.get("end")
                    .and_then(|v| v.as_i64())
                    .and_then(|x| i32::try_from(x).ok()),
            ) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "align_fingerprint.modality_id_ranges entry has non-i32 start/end: {r}"
                    ),
                ));
            };
            if s < e {
                visual.push((s, e));
            }
        }
    } else if let Some(ranges) = &manifest.modality_id_ranges {
        for r in ranges {
            if r.kind == "timestamp" || r.kind == "text" {
                continue;
            }
            if r.start < r.end {
                visual.push((r.start, r.end));
            }
        }
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "align_fingerprint present but no modality_id_ranges snapshot and no manifest \
             modality_id_ranges: text token ranges cannot be delimited from visual \
             placeholders (manifest self-contradictory)",
        ));
    }
    let params = AlignParams {
        w_ms,
        ts_base_id,
        ts_range_size,
        bucket_width_ms,
    };
    let _ = ts_end; // 已由 checked_add 校验（见上）
    Ok((Some(params), Some(VisualRanges { ranges: visual })))
}

/// 本地读失败的重试次数（**上限明确**：耗尽即按致命错误上报，绝不无限重试）。
///
/// 为什么需要：本地磁盘没有下载层的重试兜底，瞬时 EIO / NFS 抖动 / 半截缓存文件
/// 值得再试一次；为什么只限本地：云源在 `CloudIoManager` 内已有 3 次指数退避重试 +
/// 缓存校验兜底，再叠加会放大成 3×3 次（且掩盖真正的持续故障）。
const LOCAL_READ_RETRIES: usize = 2;
/// 本地读重试的退避间隔（毫秒，与 LOCAL_READ_RETRIES 一一对应）
const LOCAL_READ_RETRY_BACKOFF_MS: [u64; LOCAL_READ_RETRIES] = [50, 200];

/// predownload 滑窗节流的默认窗口（batch 数，借鉴 streaming 的 samples_ahead 节流）：
/// 预取"在途"batch 数达到窗口时，后台线程主动 sleep 让路，而不是打满硬缓冲。
/// 既削平 IO 突刺（避免尽量快地灌满 50 深的 channel），也限制预取内存上限。
/// ★ 评审 §3.8(d)：改为 `pub`，经 PyO3 导出给 Python（`megadata_rs.DEFAULT_PREDOWNLOAD`）。
///   `simulation.py` 此前独立硬编码 `16`，改这里不会同步仿真器 → 容量规划结论失真
///   （仿真报告说"16 够用"，真实引擎其实是另一个值）。现在单一事实源在 Rust。
pub const DEFAULT_PREDOWNLOAD: usize = 16;

/// `data_channel_capacity` 的下界：容量恒 >= 该值，用户窗口更大时随之扩容。
/// 同样导出给 Python 供仿真器/文档核对（见 §3.8(d)）。
pub const MIN_DATA_CHANNEL_CAPACITY: usize = 50;

/// predownload 窗口解析（纯函数，便于单测）：`0` → 默认窗口；过小（<4）保留
/// 原值但给出可操作建议（在途内存按实测 batch 字节量估算——旧实现硬编码
/// 6.5MB/batch，对 max_seq_len=8192 差 3 倍）。
fn resolve_predownload(raw: usize, batch_bytes: usize) -> usize {
    if raw == 0 {
        DEFAULT_PREDOWNLOAD
    } else if raw < 4 {
        let batch_mb = batch_bytes as f64 / (1024.0 * 1024.0);
        tracing::warn!(
            "predownload={raw} is too small: the prefetch window cannot hide IO latency, so training will likely starve the GPU.\
             Suggested value: {DEFAULT_PREDOWNLOAD} (about {:.1}MB of in-flight memory).",
            DEFAULT_PREDOWNLOAD as f64 * batch_mb
        );
        raw
    } else {
        raw
    }
}

/// data channel 容量与 predownload 窗口联动（纯函数，便于单测）：
/// 容量恒 >= 默认 50；用户窗口更大时随之扩容——否则 `predownload=100` 的
/// 节流判据永不成立，配置被静默截断（二轮 P2-3 回归锚点）。
fn data_channel_capacity(predownload: usize) -> usize {
    predownload.max(MIN_DATA_CHANNEL_CAPACITY)
}
/// 节流等待期间的轮询间隔（需要同时响应 Seek / 取消指令）
const THROTTLE_TICK: Duration = Duration::from_millis(10);

/// StreamReader 的扩展选项（Python 绑定以带默认值的参数透传）。
#[derive(Debug, Clone)]
pub struct StreamReaderOptions {
    /// 每 rank 的 DataLoader worker 数（worker 维度，避免多 worker 重复读同数据）
    pub num_workers: usize,
    /// 当前 worker 编号（0..num_workers）
    pub worker_id: usize,
    /// batch 粒度置换表 shuffle（Rust 侧纯函数，替代 Python buffer shuffle）
    pub shuffle: bool,
    /// shard 分区算法
    pub partition: PartitionAlgo,
    /// epoch 内 shard 访问顺序（F6；默认 Random = 历史行为）
    pub shard_order: ShardOrder,
    /// relaxed 分区的弹性锚点 num_canonical_nodes（0 = None = 用当前节点数）
    pub num_canonical_nodes: usize,
    /// 预取滑窗节流窗口（batch 数）。0 = 用默认值 DEFAULT_PREDOWNLOAD；
    /// 预取在途 batch 达到该窗口时后台线程主动让路，削平 IO 突刺。
    pub predownload: usize,
    /// **P1-T5（D8）预取字节预算**：预取在途的内存当量上限（字节）。0 = 不限
    /// （默认，行为与历史版本完全一致）。
    ///
    /// 计量四类字节（同 batch 合计入账，同一内存只计一次）：
    /// ① 主 token 压缩块（Zstd frame，裸流模式为 0）；② sidecar blob；
    /// ③ 解码中间态（zstd 解压缓冲 / 裸流直读的原始字节，零拷贝转交 i32 视图
    /// 后仍是同一块内存）；④ 组装后的 Python 对象（BatchData 里的 i32 向量 +
    /// images——blob 已在 ② 计过，不重复）。
    ///
    /// 超预算的 batch **延后预取**：预算门在"主列 + rec 已读、blob 未读"处
    /// 检查（blob 是大图场景的内存重头，超限时连 blob 都不读入）；同 batch
    /// 主列与 blob 亲和调度——作为整体一次读入、一次入窗，blob 不会单独排队。
    /// 单 batch 账单本身超预算时放行并 WARN 一次（保证活性，否则数据永远出不来）。
    ///
    /// PYTHON_GUIDE：建议值 ≈ 预取窗口深度 × 典型 batch 字节（主列
    /// `batch_size×max_seq_len×4` + sidecar blob 均值），训练侧一般配
    /// 数百 MiB；多模态大数据集务必显式配置，防止大图 blob 把窗口内存当量撑爆。
    pub prefetch_bytes_limit: u64,
    /// **P3b（D8）video 按需读字节上限**：collate 时点经 FFI 按需读
    /// `col_vid_blob` 的在途字节上限（video blob **不进预取窗口、不计
    /// prefetch_bytes_limit**——那是 D8 对视频 blob 的模态感知豁免）。
    /// 0 = 不限（默认）。单 clip 账单本身超限 → WARN 一次 + 放行（活性保证）。
    ///
    /// 进程级语义：上限登记在 `VideoBlobStore` 单例上（FFI 入口读它），
    /// 同进程多 reader 时后者覆盖前者。
    pub video_ondemand_bytes_limit: u64,
    /// **P3c**：请求三模态对齐视图（batch dict 透出 `av_align` 可选键）。
    /// false（默认）= 不执行 join、不透出该键（旧 trainer 零影响）。
    /// true 但 manifest 无 `align_fingerprint` → 构造期 fail-fast
    /// （缺指纹 = 数据集未启用对齐，禁静默降级）；指纹自相矛盾 → fail-fast。
    pub align: bool,
}

impl Default for StreamReaderOptions {
    fn default() -> Self {
        Self {
            num_workers: 0,
            worker_id: 0,
            shuffle: false,
            partition: PartitionAlgo::RoundRobin,
            shard_order: ShardOrder::Random,
            num_canonical_nodes: 0,
            predownload: 0,
            prefetch_bytes_limit: 0,
            video_ondemand_bytes_limit: 0,
            align: false,
        }
    }
}

// =====================================================================
// 2. 前台暴露层：StreamReader (代际栅栏 + 每实例取消)
// =====================================================================
pub struct StreamReader {
    /// 供 mixer 的 stream_format 与外部诊断读取（clippy 视角当前无人消费）
    #[allow(dead_code)]
    pub(crate) manifest: DatasetManifest,
    /// sampler 的克隆（纯函数持有者：分区 + 置换表的种子/拓扑）。前台需要它来回答
    /// "第 e 轮有多长"（R3 `epoch_total_batches`），不必惊动后台线程。
    sampler: DistributedSampler,
    my_shards_total_batches: u64,
    data_rx: Receiver<PrefetchedBatch>,
    cmd_tx: Sender<ControlCmd>,
    cancel_token: CancellationToken,
    seek_gen: u64,
    /// **R1（EOF 信号）**：预取线程读到本轮末尾时置 `true` 并 park（线程保持存活，
    /// 可被后续 Seek 唤醒）；`try_next` 见"通道空 + 标志置位"立即返回
    /// `Err(ReaderExit::Eof)`（幂等，直到下次 seek 清除）。
    ///
    /// 没有它，EOF 只能表现为"200ms 超时 → 绑定层 `continue` → 原地自旋"，
    /// 消费端既拿不到 StopIteration 也不报错（B2 实测：混采第 60 步卡死）。
    at_eof: Arc<AtomicBool>,
    /// 预取线程的致命错误（IO / 解压 / 数据损坏）。线程退出后由 `try_next` 读取，
    /// 保证错误"只上报一次"，而不是被当成 EOF 静默吞掉。
    fatal_error: Arc<Mutex<Option<String>>>,
    /// ★ N2 退出栅栏：预取线程体完整走完（含 panic 兜底写入）后才置位。
    /// 前台 `exit_reason` 在通道断开时若发现未置位，做一次有界自旋等待再读
    /// fatal 槽位——防止"槽位写入晚于 data_tx 断开"把 Fatal 降级成 Eof。
    worker_finished: Arc<AtomicBool>,
    /// **P1-T5（D8）**预取在途账本（与预取线程内的 SyncInnerReader 共享）：
    /// send +cost / 取走 -cost。前台侧只读（诊断与测试断言），写方为预取线程。
    in_flight_bytes: Arc<AtomicU64>,
}

impl StreamReader {
    /// 旧版兼容入口：单节点、单 worker、顺序读取、round-robin
    #[allow(dead_code)]
    pub fn new(
        index_path: &str,
        batch_size: usize,
        max_seq_len: usize,
        world_size: usize,
        rank: usize,
        global_seed: u64,
    ) -> std::io::Result<Self> {
        Self::with_options(
            index_path,
            batch_size,
            max_seq_len,
            world_size,
            rank,
            global_seed,
            StreamReaderOptions::default(),
        )
    }

    /// 全功能入口。
    ///
    /// 控制面校验（借鉴 streaming 的防御式配置）：
    /// - index.json 版本/结构（manifest.validate）
    /// - batch_size / max_seq_len 与构建期一致性
    /// - **shard 数 >= world_size**：shard 聚合后若 shard 数不足，高位 rank 会分到
    ///   0 个 shard（千卡场景 rank 空转），直接报错并给出可操作建议；
    ///   shard 数 < world_size×num_workers 时降级为 warning。
    pub fn with_options(
        index_path: &str,
        batch_size: usize,
        max_seq_len: usize,
        world_size: usize,
        rank: usize,
        global_seed: u64,
        options: StreamReaderOptions,
    ) -> std::io::Result<Self> {
        // ★ 单线程 runtime：所有用法都是"从同步上下文 block_on 一次 IO"（清单加载 / 分片
        //   下载），不需要多线程调度器。此前用 new_multi_thread() 会为**每个 reader** 起
        //   nCPU 个 worker 线程（实测 80 核机器：1 个 reader +81 线程、3 源 Mixer +324 线程），
        //   而它们几乎全程空闲（复核问题 4）。
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    std::io::Error::other(format!("Failed to build Tokio runtime: {e}"))
                })?,
        );
        let manifest = DatasetManifest::load_smart(&rt, index_path)?;

        // ★ 一致性校验：加载期 batch_size / max_seq_len 必须与构建期一致
        let build_bs = manifest.global_stats.batch_size;
        // ★ 评审 §3.6(b)：在**任何分配之前**校验 batch_size × max_seq_len 的乘积上限。
        //   二者都来自 Python 侧用户输入，且下游有多处按它开缓冲
        //   （`read_column_bytes` 的 `vec![0u8; ...]`、`resolve_predownload` 的
        //   batch_bytes 估算、`batch_bytes as f64` 的容量建议日志）。
        //   入口就报错，比在分配点 OOM 友好得多。
        const MAX_BATCH_ELEMS: usize = 1 << 28; // 2.68亿 token/batch ≈ 1GiB 的 i32流
        match batch_size.checked_mul(max_seq_len) {
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "batch_size ({batch_size}) × max_seq_len ({max_seq_len}) overflows usize"
                    ),
                ));
            }
            Some(elems) if elems > MAX_BATCH_ELEMS => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "batch_size ({batch_size}) × max_seq_len ({max_seq_len}) = {elems} elements exceeds the \
                         per-batch limit ({MAX_BATCH_ELEMS}). That is roughly {} GiB of i32 token ids per batch; \
                         please lower batch_size or max_seq_len.",
                        // ★ 评审 §3.4：elems 可达 usize::MAX（如 2^31 × 2^31），
                        //   `elems * 4` 会溢出（debug panic / release 静默回绕给出
                        //   错误的 GiB 数）——这个分支的全部意义就是给出友好错误，
                        //   自己却不能成为 panic 源。saturating_mul 后退化文案也正确。
                        elems.saturating_mul(4) / (1024 * 1024 * 1024)
                    ),
                ));
            }
            Some(_) => {}
        }
        if build_bs != 0 && build_bs != batch_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "batch_size mismatch: built with {}, loaded with {}. Both must be identical (data is pre-packed by batch_size at build time).",
                    build_bs, batch_size
                ),
            ));
        }
        if manifest.max_seq_len != max_seq_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "max_seq_len mismatch: built with {}, loaded with {}.",
                    manifest.max_seq_len, max_seq_len
                ),
            ));
        }

        // ★ shard 数下限校验（与 shard 聚合联动的硬约束，见设计文档 §2.1③）
        let mut world = World::from_sizes(world_size, 1, options.num_workers);
        if options.num_canonical_nodes > 0 {
            world.num_canonical_nodes = Some(options.num_canonical_nodes);
        }
        // ★ P2-4（评审修复）：拓扑校验**前移到 sampler 构造之前**。
        //   旧顺序先构造（`DistributedSampler::with_world` 内含 assert!）后做友好
        //   校验——`rank >= world_size` / `worker_id >= num_workers` 时用户拿到的是
        //   pyo3 PanicException 而非设计的 InvalidInput。现在先校验后构造，
        //   与 `epoch_total_batches` 纯函数入口同纪律。
        if rank >= world.world_size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "rank ({rank}) must be smaller than world_size ({}). Check the world_size / rank arguments (or the RANK environment variable).",
                    world.world_size()
                ),
            ));
        }
        if world.workers_per_rank > 0 && options.worker_id >= world.workers_per_rank {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "worker_id ({}) must be smaller than num_workers ({}). Check the DataLoader worker arguments.",
                    options.worker_id, world.workers_per_rank
                ),
            ));
        }
        let sampler = DistributedSampler::with_world(
            world.clone(),
            rank,
            options.worker_id,
            global_seed,
            options.partition,
        )
        .with_shard_order(options.shard_order);
        let total_shards = manifest.shards.len();
        if total_shards < world.world_size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shard count ({}) < world_size ({}): under round-robin/relaxed partitioning, high ranks would get 0 shards (training idles).\n\
                     Suggestion: rebuild the dataset with a smaller batches_per_shard (aggregation granularity must yield shard count >= world_size), or lower world_size.",
                    total_shards,
                    world.world_size()
                ),
            ));
        }
        if total_shards < world.total_workers() {
            tracing::warn!(
                "shard count ({}) < total worker count ({} = world_size {} × num_workers {}): some workers would have no data to read.",
                total_shards,
                world.total_workers(),
                world.world_size(),
                world.workers_per_rank
            );
        }

        let base_dir_or_url = if index_path.starts_with("s3://")
            || index_path.starts_with("gs://")
            || index_path.starts_with("http")
        {
            let last_slash = index_path.rfind('/').unwrap_or(0);
            index_path[..last_slash].to_string()
        } else {
            let path_obj = std::path::Path::new(index_path);
            path_obj
                .parent()
                .unwrap_or(std::path::Path::new(""))
                .to_string_lossy()
                .to_string()
        };
        // 本地源判定：只有本地读需要预取线程自己做有限次重试（见 LOCAL_READ_RETRIES）
        let is_local_source = !(base_dir_or_url.starts_with("s3://")
            || base_dir_or_url.starts_with("gs://")
            || base_dir_or_url.starts_with("http"));
        let max_read_retries = if is_local_source {
            LOCAL_READ_RETRIES
        } else {
            0
        };

        let initial_epoch = 0;
        let my_shards = sampler.get_my_shards(&manifest, initial_epoch);
        let my_shards_total_batches = my_shards.iter().map(|s| s.num_batches as u64).sum();

        let inner_state = InnerState {
            epoch: initial_epoch,
            shard_list_index: 0,
            batches_yielded_in_shard: 0,
            global_batch_idx: 0,
        };

        // predownload 窗口解析（0 = 默认；过小给出可操作建议）。
        // 放在 channel 创建之前：data channel 的容量必须与窗口联动——
        // 若容量固定 50 而用户配了 predownload=100，节流判据
        // `data_tx.len() >= predownload` 永不成立（channel 在 50 处已满），
        // 配置静默失效（二轮 P2-3）。
        let batch_bytes = batch_size.saturating_mul(max_seq_len).saturating_mul(4);
        let predownload = resolve_predownload(options.predownload, batch_bytes);
        // P3b：video 按需读上限登记进程级单例（FFI read_vid_blob 入口读它；
        // 进程级语义——同进程多 reader 后者覆盖前者）
        crate::runtime::video_blob::VideoBlobStore::global()
            .set_ondemand_limit(options.video_ondemand_bytes_limit);
        // P3c §7.2：对齐指纹门槛——请求 align 但数据集未启用（无指纹）→ 显式
        // 失败（缺指纹 = 未启用，禁静默降级）；指纹自相矛盾 → fail-fast。
        let (align_params, align_visual_ranges) = parse_align_params(&manifest)?;
        if options.align && align_params.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "align=True requested but the dataset has no align_fingerprint: the dataset \
                 was built without alignment (no audio/video modality ranges declared). \
                 Rebuild with audio/video id ranges to enable alignment.",
            ));
        }

        // data channel 容量与 predownload 联动：窗口大于默认容量 50 时扩容，
        // 否则用户的预取窗口配置会被 channel 容量悄悄截断。
        let (data_tx, data_rx) = bounded::<PrefetchedBatch>(data_channel_capacity(predownload));
        let (cmd_tx, cmd_rx) = unbounded::<ControlCmd>();
        let cancel_token = CancellationToken::new();
        // R1：EOF 信号（后台置位 / 前台 seek 清除 / try_next 读）
        let at_eof: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let at_eof_in_thread = at_eof.clone();

        let mut inner_reader = SyncInnerReader {
            manifest: manifest.clone(),
            sampler: sampler.clone(),
            state: inner_state,
            my_shards,
            base_dir_or_url,
            cloud_io: CloudIoManager::new(),
            rt,
            current_arrow_reader: None,
            batch_size,
            max_seq_len,
            current_cu_seqlens: Vec::new(),
            current_bin_offsets: None,
            current_bin_file: None,
            current_bin_len: None,
            current_doc_ids_file: None,
            current_loss_weights_file: None,
            current_mm_reader: None,
            current_vid_reader: None,
            current_vid_store_id: None,
            align_enabled: options.align,
            align_params,
            align_visual_ranges,
            current_row_sids_file: None,
            align_audio_cache: Vec::new(),
            align_audio_cache_key: None,
            align_audio_files: std::collections::HashMap::new(),
            // P3b：video 按需读上限登记进程级单例（FFI 入口读它）
            prefetch_bytes_limit: options.prefetch_bytes_limit,
            in_flight_bytes: Arc::new(AtomicU64::new(0)),
            budget_over_warned: AtomicBool::new(false),
            shuffled: options.shuffle,
            epoch_perm: Vec::new(),
            epoch_cursor: 0,
            open_shard: None,
            arrow_next_in_shard: None,
            arrow_contract_checked: false,
            seek_gen: 0,
            cancel_token: cancel_token.clone(),
        };
        if options.shuffle {
            inner_reader.epoch_perm =
                batch_permutation(inner_reader.epoch_total(), global_seed, initial_epoch);
        }
        // 预打开第一个物理位置（预取第一 shard，与旧行为一致）
        if let Some((shard_i, in_shard)) = inner_reader.position_target(0) {
            inner_reader.ensure_shard_open(shard_i, in_shard)?;
        }
        // 预取在途账本的前台句柄（D8 预算；写方在预取线程，前台只读诊断/断言）
        let in_flight_bytes = inner_reader.in_flight_bytes.clone();

        let bg_token = cancel_token.clone();
        // 致命错误槽位：预取线程写，前台 try_next 读（只上报一次）
        let fatal_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let fatal_in_thread = fatal_error.clone();
        // panic 捕获用的独立句柄：闭包把fatal_in_thread move 进去了，事后上报要用另一个。
        let fatal_for_panic = fatal_in_thread.clone();
        // ★ N2 退出栅栏：线程体完整走完（含 panic 兜底写入）才置位。
        //   `exit_reason` 在通道断开时若发现未置位，做一次有界等待再读槽位
        //   （详见 exit_reason 的注释）。
        let worker_finished: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let worker_finished_in_thread = worker_finished.clone();
        // ★ P0（20261007 评审 §3.1）：预取线程体必须包catch_unwind。
        //
        //   旧实现把整个 loop 直接放进 thread::spawn 且丢弃 JoinHandle，而
        //   `record_fatal` **只在显式的 `Err(e)` 分支被调用**。于是线程体内任何
        //   panic 都直接 unwind 出去 → fatal_error 保持 None → data_tx 被 drop →
        //   前台 exit_reason() 按 `ReaderExit::Eof` 处理 → 绑定层转
        //   ("exhausted", ...) → Python 侧视为"本轮正常读完"。后果是
        //   **训练数据被静默截断**：无异常、无 Python 侧告警、manifest 无痕迹，
        //   与 README "错误不静默" 的承诺直接矛盾。Mixer 场景下每个子流各有一个
        //   这样的线程，任一子流 panic 即截断整个混采数据集。
        //
        //   已确认的 panic 触发源（不是理论风险）：
        //   - io.rs `verify_local_file` 的 `local_verified.lock().unwrap()`：该
        //     CloudIoManager 随 SyncInnerReader 被 move 进本线程（stream.rs:339），
        //     Mutex 中毒后每次本地校验都 panic；
        //   - packing 热路径的 unwrap（greedy.rs / document_aware.rs）；
        //   - rayon 内部 panic、with_capacity 的 OOM。
        //
        //   panic 信息本身仍由默认 panic hook 打到 stderr（带 backtrace，若设了
        //   RUST_BACKTRACE=1）；这里只负责把它**上报到消费端**，避免降级成 EOF。
        let _worker = std::thread::spawn(move || {
            let prefetch_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // ★ N2 双保险之一：panic 守卫。必须放在 **catch_unwind 闭包体**的第一行，
                //   而不是外层 spawn 闭包体——unwind 时"体内局部量先于外层捕获量析构"，
                //   守卫 Drop 写 fatal 槽位严格早于 data_tx 断开（外层闭包持有 data_tx，
                //   循环被 panic 打 abort 后 catch_unwind 先返回、外层体才结束）。
                //   注：若把守卫放外层闭包第一行则**无效**——panic 被 catch 后外层体
                //   正常结束，std::thread::panicking() 已为 false，守卫不会触发
                //   （20261007 复核报告的建议位置即此，已按实测修正）。
                let _panic_guard = PanicFatalGuard {
                    slot: fatal_in_thread.clone(),
                };
                // 连续读失败计数（成功后清零）：本地源允许 LIMITED 次重试，云源不做叠加
                let mut read_failures: usize = 0;
                loop {
                    // 1. 优先处理控制指令（循环顶部同步状态）
                    while let Ok(cmd) = cmd_rx.try_recv() {
                        if let Err(e) = handle_control_cmd(&mut inner_reader, cmd) {
                            record_fatal(&fatal_in_thread, e);
                            return;
                        }
                    }

                    // 1.5 predownload 滑窗节流：在途 batch 达到窗口时主动让路。
                    //     等待期间持续响应 Seek/epoch 指令（seek 时前台会清空通道，
                    //     窗口自然回落）。借鉴 streaming 的 samples_ahead 节流。
                    while data_tx.len() >= predownload && !bg_token.is_cancelled() {
                        match cmd_rx.recv_timeout(THROTTLE_TICK) {
                            Ok(cmd) => {
                                if let Err(e) = handle_control_cmd(&mut inner_reader, cmd) {
                                    record_fatal(&fatal_in_thread, e);
                                    return;
                                }
                            }
                            Err(RecvTimeoutError::Timeout) => {}
                            Err(RecvTimeoutError::Disconnected) => return, // 前台已 drop reader
                        }
                    }
                    if bg_token.is_cancelled() {
                        return;
                    }

                    // 2. 读取下一批数据（三态：Some=数据 / Ok(None)=EOF / Err=致命）
                    let batch_res: std::io::Result<Option<(BatchData, u64)>> =
                        if inner_reader.manifest.format == StorageFormat::ArrowIpc {
                            // Arrow 分支不在 D8 预算计量范围内（PT/MDS 是多模态 sidecar
                            // 的主战场）：账单恒 0，预算门短路，行为与历史版本一致。
                            inner_reader
                                .next_arrow_batch()
                                .map(|o| o.map(|b| (BatchData::Arrow(b), 0)))
                        } else {
                            inner_reader.next_pt_batch().map(|o| {
                                o.map(
                                    |(
                                        ids,
                                        cu,
                                        doc_ids,
                                        loss_weights,
                                        images,
                                        videos,
                                        av_align,
                                        byte_cost,
                                    )| {
                                        // P1-2：videos 携带 (entries, pin)——pin 随 batch 存活，
                                        // drop 时归还 blob 句柄（切 shard 只标记 retired）。
                                        let (videos, video_pin) = match videos {
                                            Some((entries, pin)) => (Some(entries), Some(pin)),
                                            None => (None, None),
                                        };
                                        (
                                            BatchData::Memmap {
                                                input_ids: ids,
                                                cu_seqlens: cu,
                                                doc_ids,
                                                loss_weights,
                                                images,
                                                videos,
                                                av_align,
                                                video_pin,
                                            },
                                            byte_cost,
                                        )
                                    },
                                )
                            })
                        };

                    match batch_res {
                        Ok(Some((data, byte_cost))) => {
                            read_failures = 0;
                            // D8 账本：**send 之前**入账（Release）——程序序保证任何
                            // 前台的 -cost（Acquire）都发生在 +cost 之后，账本不会
                            // 下溢/虚高。send 失败（前台已 drop）则回滚。
                            if byte_cost > 0 {
                                inner_reader
                                    .in_flight_bytes
                                    .fetch_add(byte_cost, Ordering::Release);
                            }
                            let msg = PrefetchedBatch {
                                data,
                                seek_gen: inner_reader.seek_gen,
                                byte_cost,
                            };
                            if data_tx.send(msg).is_err() {
                                if byte_cost > 0 {
                                    inner_reader
                                        .in_flight_bytes
                                        .fetch_sub(byte_cost, Ordering::Release);
                                }
                                return;
                            }
                        }
                        Ok(None) => {
                            // EOF park 前把 read_failures 清零是有意设计：瞬时故障的
                            // 重试计数只针对"同一位置的连续失败"；EOF park 期间收到
                            // Seek 后，新位置的读失败与旧位置无关，从 0 重新计
                            // （重试本就是为瞬时故障设计的，跨位置不累积）。
                            read_failures = 0;
                            // ★ R1：本轮读完 → 置 EOF 标志并 park。
                            //   线程**保持存活**等新指令（Seek/续训回退），读完就退出会让
                            //   后续 seek 失去执行者；而置位让前台 `try_next` 能立刻把
                            //   "本轮结束"上报给消费端（而不是 200ms 超时 → 绑定自旋）。
                            at_eof_in_thread.store(true, Ordering::Release);
                            if bg_token.is_cancelled() {
                                return;
                            }
                            match cmd_rx.recv_timeout(POLL_TIMEOUT) {
                                Ok(cmd) => {
                                    if let Err(e) = handle_control_cmd(&mut inner_reader, cmd) {
                                        record_fatal(&fatal_in_thread, e);
                                        return;
                                    }
                                }
                                Err(RecvTimeoutError::Timeout) => {}
                                Err(RecvTimeoutError::Disconnected) => return,
                            }
                        }
                        Err(e) => {
                            // 取消（含 IO 中断）走正常退出
                            if bg_token.is_cancelled()
                                || e.kind() == std::io::ErrorKind::Interrupted
                            {
                                return;
                            }
                            read_failures += 1;
                            if read_failures <= max_read_retries {
                                // 有限次重试（仅本地源）：失败不会推进 cursor，
                                // 因此重试的就是同一位置。退避 sleep ≤200ms，指令会排队后立刻处理。
                                // 下标安全性：`LOCAL_READ_RETRY_BACKOFF_MS` 的长度与
                                // LOCAL_READ_RETRIES 一一对应，且本分支要求
                                // 1 <= read_failures <= max_read_retries <= LOCAL_READ_RETRIES
                                // （云源 max_read_retries=0 不会进入）。debug 构建兜底
                                // 暴露未来给云源加重试时引入的越界。
                                debug_assert!(
                                    read_failures >= 1
                                        && read_failures <= LOCAL_READ_RETRY_BACKOFF_MS.len(),
                                    "read_failures out of backoff table range"
                                );
                                debug_assert!(max_read_retries <= LOCAL_READ_RETRIES);
                                let delay = LOCAL_READ_RETRY_BACKOFF_MS[read_failures - 1];
                                tracing::warn!(
                                    "⚠️ Local read failed, retrying the same position in {delay}ms (retry {read_failures}/{max_read_retries}): {e}"
                                );
                                thread::sleep(Duration::from_millis(delay));
                                // 退避结束后、重试前刷新缓存的 .bin 长度：本地"半截文件被
                                // 外部续写修复"的自愈场景要求重试即刻看到新长度（放在 sleep
                                // 后会错过修复窗口；正常路径零成本，见方法注释）。
                                inner_reader.refresh_cached_bin_len_for_retry();
                                continue;
                            }
                            // 重试耗尽 → 致命上报（绝不无限重试）：
                            // drop data_tx → 前台 try_next 拿到 Disconnected + Fatal(msg)
                            record_fatal(&fatal_in_thread, e);
                            return;
                        }
                    }
                }
            }));
            if prefetch_outcome.is_err() {
                // 走到这里说明线程体内部 panic 了（显式 IO 错误路径不会到这里——它们走
                // record_fatal 后 return）。必须写 fatal 槽位：否则前台 exit_reason()
                // 会把它当成 Eof = "本轮正常读完"，造成训练数据静默截断。
                // （与 PanicFatalGuard 双写同一常量，幂等；守卫保证"提前写"，这里
                //  保证"一定写"，任一处被未来重构删掉另一处仍兜底。）
                record_fatal_msg(&fatal_for_panic, PANIC_FATAL_MSG);
            }
            // ★ N2 退出栅栏置位：必须是线程体最后一句——此时才早于本闭包捕获的
            //   data_tx 析构（即通道断开）。前台观察到 Disconnected 时，槽位写入
            //   与置位均已完成（Release/Acquire 配对），Fatal 不可能被降级成 Eof。
            worker_finished_in_thread.store(true, Ordering::Release);
        });

        Ok(Self {
            manifest,
            sampler,
            my_shards_total_batches,
            data_rx,
            cmd_tx,
            cancel_token,
            seek_gen: 0,
            at_eof,
            fatal_error,
            worker_finished,
            in_flight_bytes,
        })
    }

    /// 本 rank/worker 在 **epoch 0** 的 batch 数（构造期常量）。
    ///
    /// v3（R3/B1）：已降级为"epoch-0 名义值"——每个 epoch 的 shard 集合与尾块都在变，
    /// 真实轮长请用 `epoch_total_batches(epoch)`。
    pub fn total_batches_for_rank(&self) -> u64 {
        self.my_shards_total_batches
    }

    /// **R3**：第 `epoch` 轮的真实 batch 数（`len()` 的数据源）。
    ///
    /// 纯计算（分区 + 置换都是纯函数），不惊动后台线程、不做 IO。
    /// 走 `my_shard_batch_total` 轻量路径：不 clone ShardMeta（含 hashes
    /// HashMap）——本函数在每次 seek 的前台校验都会调用。
    pub fn epoch_total_batches(&self, epoch: u64) -> u64 {
        self.sampler.my_shard_batch_total(&self.manifest, epoch)
    }

    /// **R1**：本轮是否已读完（预取线程已 park 在 EOF）。
    ///
    /// 供 Mixer 判定子流耗尽（`is_exhausted(i)`）；幂等，直到下次 seek 清除。
    pub fn at_eof(&self) -> bool {
        self.at_eof.load(Ordering::Acquire)
    }

    /// 预取线程是否仍在运行。以前恒为 `true`（信息失真）；现在读完后线程虽保持存活
    /// 等 Seek（`Ok(None)`），但**取消 / 致命错误**后线程退出、通道断开 → `false`。
    pub fn is_running(&self) -> bool {
        !self.cancel_token.is_cancelled() && !self.data_rx.is_disconnected()
    }

    /// 是否已被请求取消。
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// **P1-T5（D8）**预取在途账本当前值（诊断/测试断言用；前台只读）。
    /// 预算开启时恒 ≤ `prefetch_bytes_limit`（单 batch 超预算放行除外）；
    /// 消费完/取消消费干通道后应归零（守恒）。
    #[allow(dead_code)] // 诊断/测试断言用
    pub fn in_flight_bytes(&self) -> u64 {
        self.in_flight_bytes.load(Ordering::Acquire)
    }

    /// 请求取消：后台 IO 尽快退出，前台 try_next 将返回 Err。
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    // ===============================================================
    // ★ 代际栅栏 Seek 协议（v3：按 (epoch, in_epoch) 定位）
    // ===============================================================
    /// **R2**：定位到 "第 `epoch` 轮的第 `in_epoch` 个 batch"。
    ///
    /// 顺序钉死（A5/C13）：**drain 通道 → 清除 `at_eof` → 发送 Seek**（与 seek_gen
    /// 自增同处）。清除必须由**前台**做：若交给后台线程处理 Seek 时再清，前台可能在
    /// 它被处理前读到陈旧 `true` → 伪 Eof。
    pub fn seek_to_epoch_batch(&mut self, epoch: u64, in_epoch: u64) -> std::io::Result<()> {
        // 前台先校验：越界就地报错，而不是让后台线程在 handle_control_cmd 里
        // record_fatal 后**退出**（那样整个 reader 会被一次错误 seek 打死）。
        let total = self.epoch_total_batches(epoch);
        if total > 0 && in_epoch >= total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "seek out of range: epoch {epoch} has only {total} batches, cannot seek to in_epoch={in_epoch}.\
                     The cursor must be the 'next position to produce' (normalize to (epoch+1, 0) when exhausted)."
                ),
            ));
        }
        self.seek_gen = self.seek_gen.wrapping_add(1);
        // drain 在途 batch：每个被丢弃的 batch 同步归还 D8 预算账本（与 try_next
        // 的取走归还同一纪律）。drain 后预取线程可能还在 send 旧 gen batch——
        // 它们入账后滞留通道，由后续 try_next 取走（陈旧丢弃）时归还，账本守恒。
        while let Ok(batch) = self.data_rx.try_recv() {
            if batch.byte_cost > 0 {
                self.in_flight_bytes
                    .fetch_sub(batch.byte_cost, Ordering::AcqRel);
            }
        }
        self.at_eof.store(false, Ordering::Release);
        let _ = self.cmd_tx.send(ControlCmd::Seek {
            epoch,
            in_epoch,
            gen_id: self.seek_gen,
        });
        Ok(())
    }

    /// 尝试获取下一个 batch，最多阻塞 POLL_TIMEOUT。
    ///
    /// **R1 的顺序**（关键）：
    /// ① 非阻塞 `try_recv` 先取在途数据（有数据按代际处理）；
    /// ② **通道空且 `at_eof`** → 立即 `Err(Eof)`（幂等，直到下次 seek 清除）；
    /// ③ 否则阻塞 `recv_timeout(POLL_TIMEOUT)`，超时后再查一次 `at_eof`。
    ///
    /// "Timeout ⇒ 通道空"成立，所以 ② / ③ 判据不会跳过在途的最后一批。
    /// - Ok(Some(data))：拿到有效 batch（代际匹配）
    /// - Ok(None)：超时或陈旧（代际不匹配），调用方应重试
    /// - Err(ReaderExit::Eof)：本轮读完（R1 可达）／线程退出（取消、drop）
    /// - Err(ReaderExit::Fatal(msg))：致命 IO／解压／数据错误，**必须上报**（不可当 EOF 吞掉）
    pub fn try_next(&self) -> Result<Option<BatchData>, ReaderExit> {
        // ① 在途数据（非阻塞）
        match self.data_rx.try_recv() {
            Ok(batch) => {
                // D8 账本归还：batch 已离开预取窗口（含陈旧丢弃），无论去向立即减账。
                if batch.byte_cost > 0 {
                    self.in_flight_bytes
                        .fetch_sub(batch.byte_cost, Ordering::AcqRel);
                }
                return Ok(if batch.seek_gen == self.seek_gen {
                    Some(batch.data)
                } else {
                    None // 陈旧 batch，丢弃并让调用方重试
                });
            }
            Err(TryRecvError::Disconnected) => return self.exit_reason(),
            Err(TryRecvError::Empty) => {}
        }
        // ② 通道空 + 本轮已读完
        if self.at_eof.load(Ordering::Acquire) {
            return Err(ReaderExit::Eof);
        }
        // ③ 阻塞等待
        match self.data_rx.recv_timeout(POLL_TIMEOUT) {
            Ok(batch) => {
                // D8 账本归还（同 ①）
                if batch.byte_cost > 0 {
                    self.in_flight_bytes
                        .fetch_sub(batch.byte_cost, Ordering::AcqRel);
                }
                Ok(if batch.seek_gen == self.seek_gen {
                    Some(batch.data)
                } else {
                    None
                })
            }
            Err(RecvTimeoutError::Timeout) => {
                if self.at_eof.load(Ordering::Acquire) {
                    Err(ReaderExit::Eof)
                } else {
                    Ok(None)
                }
            }
            Err(RecvTimeoutError::Disconnected) => self.exit_reason(),
        }
    }

    /// 通道断开（线程退出）后的退出原因：致命错误优先，否则按 Eof 处理。
    ///
    /// ★ N2（20261007 复核）有界等待：当前代码结构下"槽位写入 → 置位 → data_tx
    /// 断开"的程序序 + Release/Acquire 已保证置位必先于前台观察到 Disconnected，
    /// 等待是**纯冗余保险**；它防的是未来重构（把循环体抽成持有 data_tx 所有权
    /// 的独立函数、给 catch_unwind 闭包加 move 等）把槽位写入排到断开之后——
    /// 那种结构下等待把"必发的降级"缩为"最多 50ms 延迟 + 仍然上报 Fatal"。
    fn exit_reason(&self) -> Result<Option<BatchData>, ReaderExit> {
        if !self.worker_finished.load(Ordering::Acquire) {
            let deadline = std::time::Instant::now() + Duration::from_millis(50);
            while !self.worker_finished.load(Ordering::Acquire) {
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::yield_now();
            }
        }
        let fatal = self.fatal_error.lock().ok().and_then(|g| g.clone());
        match fatal {
            Some(msg) => Err(ReaderExit::Fatal(msg)),
            None => Err(ReaderExit::Eof),
        }
    }
}

/// 预取线程 panic 时上报给消费端的致命错误消息（与 catch_unwind 后的补写共用
/// 同一常量，保证守卫与兜底两条路径文案一致、回归测试可断言）。
const PANIC_FATAL_MSG: &str = "prefetch thread panicked unexpectedly (the reader loop was aborted by a panic; \
     see the panic message on stderr for details and backtrace)";

/// ★ N2 panic 守卫：构造于 catch_unwind 闭包体第一行，线程因 panic 而 unwind
/// 时在 Drop 中把 fatal 槽位写上（`std::thread::panicking()` 为 true 才写，
/// 正常退出路径零成本）。写入发生在**本闭包体内局部量析构阶段**，严格早于
/// 外层闭包持有的 data_tx 断开，从时序上根除"Fatal 被降级成 Eof"的窗口。
struct PanicFatalGuard {
    slot: Arc<Mutex<Option<String>>>,
}

impl Drop for PanicFatalGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            record_fatal_msg(&self.slot, PANIC_FATAL_MSG);
        }
    }
}

/// 记录致命错误：写共享槽位（前台 `try_next` → `Err(ReaderExit::Fatal(msg))`）。
/// 预取线程随后退出（drop `data_tx`），消费者立刻拿到错误而不是无限重试。
fn record_fatal(slot: &Arc<Mutex<Option<String>>>, e: std::io::Error) {
    record_fatal_msg(slot, &e.to_string());
}

/// 同上，但直接收字符串——供 `catch_unwind` 的 panic 分支使用（panic 不是
/// `io::Error`，且此时原始错误类型信息只存在于已打出的 panic message 里）。
fn record_fatal_msg(slot: &Arc<Mutex<Option<String>>>, msg: &str) {
    tracing::error!(
        "❌ Prefetch thread fatal error (stopping silent retries, reporting to consumer): {msg}"
    );
    //锁中毒不再 panic：本函数是 panic 兜底路径的最后一道防线，若它自己再 panic
    // 就彻底回到"静默 EOF"的老问题。毒化后 guard 不可用时退化为"写不进去"，
    // 至少不会二次 panic；此时前台退化为 Eof（唯一残留缺口，已在 §3.1 备注）。
    if let Ok(mut guard) = slot.lock() {
        *guard = Some(msg.to_string());
    }
}

/// 让"用户取消"以 Interrupted 在内部传播：预取线程见到 `ErrorKind::Interrupted`
/// 会走正常退出（`ReaderExit::Eof`），而不是当成致命错误上报。
fn cancel_err() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Interrupted, "Cancel")
}

// =====================================================================
// 3. 后台物理执行层：SyncInnerReader (物理状态全封闭 + 每实例取消)
// =====================================================================

/// 统一处理控制指令（后台循环的三处共用：循环顶部 / 节流等待 / EOF 等待）。
///
/// 返回 `Err` = 致命（目前仅 Seek 失败：定位 / 打开 shard 的 IO 错误），由调用方
/// 记录并退出线程——不再 `let _ =` 静默吞掉（Seek 失败若被忽略，后续会读到错位数据）。
fn handle_control_cmd(inner: &mut SyncInnerReader, cmd: ControlCmd) -> std::io::Result<()> {
    match cmd {
        ControlCmd::Seek {
            epoch,
            in_epoch,
            gen_id,
        } => {
            inner.seek_gen = gen_id;
            inner.seek_to_epoch_batch(epoch, in_epoch)
        }
    }
}

struct SyncInnerReader {
    manifest: DatasetManifest,
    sampler: DistributedSampler,
    state: InnerState,
    my_shards: Vec<ShardMeta>,
    base_dir_or_url: String,
    cloud_io: CloudIoManager,
    rt: Arc<tokio::runtime::Runtime>,
    current_arrow_reader: Option<FileReader<File>>,
    /// 已打开的 Arrow reader 将要吐出的 in-shard 位置（顺序读取快路径的判据）
    arrow_next_in_shard: Option<u64>,
    /// 当前 shard 的首个 batch 已通过 batch 级列（cu_seqlens/row_lens）契约校验
    arrow_contract_checked: bool,
    batch_size: usize,
    max_seq_len: usize,
    /// 当前 shard 的所有 batch 的 cu_seqlens（PT Memmap 模式）。
    /// open_current_shard 时从 .cu 文件加载；旧数据集无 .cu 文件时为空 Vec。
    current_cu_seqlens: Vec<Vec<i32>>,
    /// 当前 shard 的 .idx 偏移表（仅 Zstd-MDS 格式使用：压缩流字节偏移，n+1 条）
    current_bin_offsets: Option<Vec<u64>>,
    /// 当前 shard 主 `.bin` 的已缓存句柄（0.2 Tier-0 快赢项）。
    ///
    /// 背景：此前 `read_pt_batch_bytes` 每批都走 `open_cached`（open + fstat + seek + read
    /// 共 4 syscall），而旁路列（`.col_doc_ids` 等）反而有句柄缓存。本字段对齐旁路列
    /// 既有模式——在 `open_current_shard` 物化 `.bin` 后保存句柄，`read_pt_batch_bytes`
    /// 复用之（4→2 syscall/batch）。
    ///
    /// shard 切换时随 `open_current_shard` 重置为 None 再重开。持有打开的 fd 期间即使
    /// 缓存 LRU 把 `.bin` 删掉（unlink），inode 仍被 fd 保活、读路径不受影响（Unix
    /// open-fd 语义）——这也是 #38 mmap 的天然第一步（"保存 File"→"保存 Mmap"升级路径）。
    current_bin_file: Option<File>,
    /// 当前 shard 主 `.bin` 的文件长度（`current_bin_file` 打开时 stat 一次并缓存，
    /// 与之同步置位/重置）。裸流分支"长度是否装得下该 batch"的校验据此进行——若每批
    /// 再 `file.metadata()` 会白付一次 fstat，句柄缓存的收益从 4→2 syscall 缩水成 4→3。
    current_bin_len: Option<u64>,
    /// 文档级旁路列（列扩展）的已打开句柄；`None` = 数据集无该列（零开销、零行为变化）。
    /// 两个列文件都是**定宽**（每 batch `batch_size*max_seq_len` 个元素），因此不需要
    /// 额外的偏移表：直接按 `batch_idx * elem_size * capacity` 算术寻址。
    current_doc_ids_file: Option<File>,
    current_loss_weights_file: Option<File>,
    /// 多模态图像 sidecar 读取器（P1-T4b）。`None` = 数据集无 mm sidecar
    /// （旧数据集）——零 IO、零行为变化。打开判定见 `open_mm_sidecar`：
    /// schema 三 key 齐全才构造，部分 key = manifest 自相矛盾（fail-fast）。
    /// `MmSidecarReader` 自带前缀和游标（顺序迭代 O(1) 摊销），随 shard 切换重建。
    current_mm_reader: Option<MmSidecarReader>,
    /// P3b：video sidecar 读取器（`col_vid_rec` + 共享 `mm_cu`；**不持 blob 读
    /// 句柄**——字节经 `VideoBlobStore` 按需读，D8 不进预取窗口）。
    /// `None` = 数据集无 video sidecar（schema 无 vid 三 key）。
    current_vid_reader: Option<VideoSidecarReader>,
    /// P3b：当前 shard 的 `col_vid_blob` 在 `VideoBlobStore` 的句柄 id
    /// （代际切换时 unregister，防句柄泄漏）。
    current_vid_store_id: Option<u64>,
    /// P3c：对齐启用（请求 av_align；构造期经指纹门槛校验）
    align_enabled: bool,
    /// P3c：对齐参数（manifest 指纹解析；None = 数据集未启用对齐）
    align_params: Option<AlignParams>,
    /// P1-1：非文本（image/audio/video）token id 区间——text token 区间末端
    /// 在首个非文本 token 处截断的依据（指纹 modality_id_ranges 快照）。
    align_visual_ranges: Option<VisualRanges>,
    /// P3c：`col_row_sids` 列句柄（Vlm 流行 sid；schema key 判定存在性）
    current_row_sids_file: Option<File>,
    /// P3c：当前 Vlm shard 的音频范围 rec 缓存
    /// （`(全局 audio batch idx, batch 内 rec idx, rec)`；Vlm shard 切换重建）
    align_audio_cache: Vec<(u32, u32, crate::builder::sidecar::AudioRecord)>,
    /// 缓存对应的 Vlm shard idx（None = 未建）
    align_audio_cache_key: Option<usize>,
    /// P3c：audio shard 句柄缓存（`ashard_XXXXX` 的 col_aud_cu/rec；随 Vlm
    /// shard 切换整体重建，防句柄泄漏）
    align_audio_files: std::collections::HashMap<usize, AudioShardFiles>,
    /// **P1-T5（D8）**预取字节预算上限；0 = 不限（所有预算逻辑短路，零行为差异）。
    prefetch_bytes_limit: u64,
    /// 预取在途账本（与 `StreamReader` 共享）：send 成功 +byte_cost，前台取走
    /// （含陈旧丢弃、seek drain）立即 -byte_cost。预算门据此判定是否延后读 blob。
    in_flight_bytes: Arc<AtomicU64>,
    /// 单 batch 账单本身超预算的 WARN 只发一次（防刷屏）。
    budget_over_warned: AtomicBool,
    /// batch 粒度置换表 shuffle（P2：置换表替代 Python buffer）
    shuffled: bool,
    epoch_perm: Vec<u32>,
    /// 当前 epoch 内已消费的 batch 位置（置换模式下是"置换后的第几个"）
    epoch_cursor: u64,
    /// 当前已打开的 (epoch, shard_list_index)
    open_shard: Option<(u64, usize)>,
    seek_gen: u64,
    cancel_token: CancellationToken,
}

impl Drop for SyncInnerReader {
    /// **P2-3（评审修复）：reader 析构时注销最后持有的 video blob 句柄。**
    ///
    /// 旧实现只在"切 shard"时注销，reader 本身销毁（训练结束 / Mixer 子流
    /// 释放 / 构造失败回滚）时最后登记的 id 永久留在进程级注册表里（fd +
    /// HashMap 条目泄漏）。注销语义见 [`VideoBlobStore::unregister`]：仍被
    /// 未消费 batch pin 着的句柄会保活到最后一个 pin 归还——不破坏 P1-2 的
    /// "未消费 batch 可读"保证。
    fn drop(&mut self) {
        if let Some(id) = self.current_vid_store_id.take() {
            crate::runtime::video_blob::VideoBlobStore::global().unregister(id);
        }
    }
}

/// 测试专用“毒丸”标记：shard 文件名含此子串时，`next_pt_batch` 会在预取线程内
/// 主动 panic——用于 `prefetch_thread_panic_surfaces_fatal_never_eof` 行为级回归
/// （P0/N2：panic 必须上报 Fatal，不得降级成 Eof）。
/// `#[cfg(test)]` 保证该分支只进 lib 测试构建，发布/嵌入 Python 的构建不含。
#[cfg(test)]
const TEST_PANIC_PILL: &str = ".inject_panic.";

/// P3c：audio shard 的 sidecar 句柄（col_aud_cu 计数 + col_aud_rec 记录）。
struct AudioShardFiles {
    cu_file: File,
    cu_len: u64,
    rec_file: File,
}

impl SyncInnerReader {
    /// 当前**已加载** epoch 的 batch 数
    fn epoch_total(&self) -> u64 {
        self.my_shards.iter().map(|s| s.num_batches as u64).sum()
    }

    /// 指定 epoch 的 batch 数（不改内部状态；分区是纯函数，用于 seek 校验）。
    /// 同 `epoch_total_batches`：走零 clone 路径，避免 seek 高频路径上的分配。
    fn epoch_total_of(&self, epoch: u64) -> u64 {
        self.sampler.my_shard_batch_total(&self.manifest, epoch)
    }

    /// 加载指定 epoch 的 shard 列表（并重建置换表）
    fn load_epoch_shards(&mut self, epoch: u64) {
        self.my_shards = self.sampler.get_my_shards(&self.manifest, epoch);
        self.state.epoch = epoch;
        if self.shuffled {
            self.epoch_perm =
                batch_permutation(self.epoch_total(), self.sampler.global_seed, epoch);
        }
    }

    /// epoch 内位置 → (shard_list_index, in_shard)。顺序模式为恒等映射，
    /// 置换模式经 epoch_perm 查表。
    fn position_target(&self, in_epoch_pos: u64) -> Option<(usize, u64)> {
        let target = if self.shuffled {
            *self.epoch_perm.get(in_epoch_pos as usize)? as u64
        } else {
            in_epoch_pos
        };
        let mut rem = target;
        for (i, shard) in self.my_shards.iter().enumerate() {
            if rem < shard.num_batches as u64 {
                return Some((i, rem));
            }
            rem -= shard.num_batches as u64;
        }
        None
    }

    /// 确保物理读头位于 (当前 epoch, shard_i, in_shard)。
    /// - shard 未打开 → open_current_shard（含 .cu/.idx 加载、arrow skip）
    /// - shard 已打开但 Arrow reader 位置不符（置换跳读）→ 重开 + skip
    fn ensure_shard_open(&mut self, shard_i: usize, in_shard: u64) -> std::io::Result<()> {
        self.state.batches_yielded_in_shard = in_shard;
        let open_key = (self.state.epoch, shard_i);
        if self.open_shard != Some(open_key) {
            self.state.shard_list_index = shard_i;
            self.open_current_shard()?;
            self.open_shard = Some(open_key);
        } else if self.manifest.format == StorageFormat::ArrowIpc
            && self.arrow_next_in_shard != Some(in_shard)
        {
            // 置换跳读：不重开文件，按 footer block 索引直接定位（O(1)，支持前后跳）
            self.arrow_jump_to(in_shard)?;
        }
        Ok(())
    }

    /// 解析下一个 batch 的物理位置：置换表查表 + shard 定位。
    ///
    /// v3（D1/R5）：**不做 epoch 自动推进**——epoch 归 Python 驱动，本轮读完即
    /// `Ok(None)`（→ 预取线程置 `at_eof` 并 park），等 `seek_to_epoch_batch` 推进。
    ///
    /// - `Ok(Some((shard_i, in_shard)))`：有效位置；
    /// - `Ok(None)`：本轮读完 / 空 rank（Seek 后可继续，非错误）；
    /// - `Err(..)`：IO 错误 / 内部状态不一致 —— 上报，绝不再被当作 EOF 静默重试。
    fn resolve_next_position(&mut self) -> std::io::Result<Option<(usize, u64)>> {
        let total = self.epoch_total();
        if total == 0 {
            return Ok(None); // 空 rank（调用方已在构造期校验并告警）
        }
        if self.epoch_cursor >= total {
            return Ok(None); // 本轮读完（EOF，不是错误）
        }
        let Some((shard_i, in_shard)) = self.position_target(self.epoch_cursor) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "epoch_cursor {} out of range for this epoch's {} batches (internal state inconsistency)",
                    self.epoch_cursor, total
                ),
            ));
        };
        self.ensure_shard_open(shard_i, in_shard)?; // 取消 → Interrupted；其余为真 IO 错误
        Ok(Some((shard_i, in_shard)))
    }

    fn next_arrow_batch(&mut self) -> std::io::Result<Option<RecordBatch>> {
        if self.cancel_token.check().is_err() {
            return Err(cancel_err());
        }
        let Some((_shard_i, in_shard)) = self.resolve_next_position()? else {
            return Ok(None);
        };
        let Some(reader) = self.current_arrow_reader.as_mut() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Arrow reader not ready (shard open flow is abnormal)",
            ));
        };
        let batch = match reader.next() {
            Some(Ok(batch)) => batch,
            Some(Err(e)) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Arrow read failed (batch {in_shard}): {e}"),
                ));
            }
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "Arrow stream ended prematurely: batch {in_shard} does not exist (file truncated?)"
                    ),
                ));
            }
        };
        self.arrow_next_in_shard = Some(in_shard + 1);
        self.state.batches_yielded_in_shard = in_shard + 1;
        self.state.global_batch_idx += 1;
        self.epoch_cursor += 1;
        // 契约校验：每 shard 首个 batch 执行一次（校验逻辑与热路径无关）
        if !self.arrow_contract_checked {
            let shard = &self.my_shards[self.state.shard_list_index].filename;
            validate_arrow_batch_level_columns(&batch, self.batch_size, shard)?;
            self.arrow_contract_checked = true;
        }
        Ok(Some(batch))
    }

    fn next_pt_batch(&mut self) -> std::io::Result<Option<PtBatchData>> {
        // epoch 推进 / shard 切换全部由 resolve_next_position 处理，这里只需定位 → 读一个 batch
        if self.cancel_token.check().is_err() {
            return Err(cancel_err());
        }
        let Some((_shard_i, in_shard)) = self.resolve_next_position()? else {
            return Ok(None);
        };

        let shard_filename = self.my_shards[self.state.shard_list_index].filename.clone();
        // ★ 测试毒丸（P0/N2 行为级回归的注入点）：文件名带约定标记时在**预取线程内**
        //   制造一次真实 panic——unwind 路径、Drop 时序与生产代码完全同构，
        //   比任何 mock 都更接近真实竞态。仅 lib 测试构建编译。
        #[cfg(test)]
        if shard_filename.contains(TEST_PANIC_PILL) {
            panic!("[test-injected] prefetch thread panic (N2 regression)");
        }
        let (bytes, compressed_len) = self.read_pt_batch_bytes(&shard_filename, in_shard)?;
        let expected_len = self.batch_size * self.max_seq_len * 4;
        if bytes.len() != expected_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shard {shard_filename} batch {in_shard} decompressed/read byte count {} != expected {expected_len}, data corrupted",
                    bytes.len()
                ),
            ));
        }

        // Vec<u8> → Vec<i32>：满足对齐条件时零拷贝转交（生命周期由 DLPack capsule 接管），
        // 否则退化为一次拷贝 —— 旧实现无条件 transmute 是潜在 UB（复核问题 12）。
        let vec_i32 = bytes_into_i32_vec(bytes).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shard {shard_filename} batch {in_shard}: byte count is not a multiple of 4 (i32 width); \
                     refusing to silently drop the trailing bytes",
                ),
            )
        })?;
        // 取出当前 batch 的 cu_seqlens（若 .cu 文件存在且已加载）。
        // **P1-4（评审修复）**：`.cu` 存在时加载期已严格校验（记录数 ==
        // ShardMeta.num_batches、单条非空/首项 0/单调/不超 batch 容量）⇒ 此处
        // 缺失只可能是 manifest 与文件不一致，**不得**静默回退空 Vec（那会把
        // attention 语义从 varlen 悄悄改成标准 attention——跨文档注意力污染）。
        // 仅"旧格式明确无 .cu 文件"（加载结果为空 Vec）允许空边界兼容。
        let cu_seqlens = if self.current_cu_seqlens.is_empty() {
            Vec::new() // 旧数据集无 .cu 旁路：空 cu_seqlens = 标准 attention（向后兼容）
        } else {
            self.current_cu_seqlens
                .get(in_shard as usize)
                .cloned()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "shard {shard_filename} .cu covers {} batch(es) but batch {in_shard} \
                             was requested — manifest and .cu disagree (corrupt or truncated)",
                            self.current_cu_seqlens.len()
                        ),
                    )
                })?
        };
        // 文档级旁路列（列扩展）：定宽寻址，缺失时 None
        let doc_ids = self.read_doc_ids_column(&shard_filename, in_shard)?;
        let loss_weights = self.read_loss_weights_column(&shard_filename, in_shard)?;
        // 多模态图像 sidecar（P1-T4b）：无 key 数据集为 None（零行为差异）。
        // **P1-T5（D8）两段式读取 + 亲和调度**：先读 rec 元数据（不含 blob 字节），
        // 账单可精确计算后过预算门，通过才读 blob——超预算时 blob 字节根本不进
        // 内存（blob 是大图场景的内存重头）；主列与 blob 一次读齐、一次入窗，
        // 不会出现"blob 单独排队占窗"的失配。
        let mm_records = self.read_mm_records_meta(&shard_filename, in_shard)?;
        let blob_est = mm_records
            .as_ref()
            .map(|recs| recs.iter().map(|r| r.len as u64).sum::<u64>())
            .unwrap_or(0);
        // 账单（四类计量，同一内存只计一次）：
        // ① 主 token 压缩块（compressed_len，裸流恒 0）
        // ③ 解码中间态（expected_len：zstd 解压缓冲 / 裸流直读字节，零拷贝转交后仍是它）
        // ② sidecar blob（blob_est，读入后与 ④ 组装对象里的 images 同一块内存）
        // ④ 组装后的 Python 对象（BatchData：i32 向量与 ③ 同内存 + blob 已计）
        let byte_cost = compressed_len + expected_len as u64 + blob_est;
        self.await_prefetch_budget(byte_cost, &shard_filename, in_shard)?;
        let images = self.read_mm_blobs(mm_records, &shard_filename, in_shard)?;
        // P3b（D8）：video 记录元数据（**不含 blob 字节**——col_vid_blob 不进
        // 预取窗口、不计 byte_cost；字节由 collate 时点经 VideoBlobStore 按需读）。
        // P1-2：entries 与 blob 句柄 pin 成对产出（pin 随 batch 存活）。
        let videos = self.read_vid_records_meta(&shard_filename, in_shard)?;
        let video_pairs: Option<Vec<(u64, u32)>> = videos
            .as_ref()
            .map(|(entries, _)| entries.iter().map(|v| (v.sample_id, v.start_ms)).collect());
        // P3c：三模态对齐 join（请求 align 时；行 sids + 视频 rec + 音频范围 rec
        // 按 (sid, window_index) 归并）。不请求 → None（batch dict 无 av_align 键）。
        let av_align = if self.align_enabled {
            Some(self.build_av_align(
                &shard_filename,
                in_shard,
                &vec_i32,
                &cu_seqlens,
                &video_pairs,
            )?)
        } else {
            None
        };
        self.state.batches_yielded_in_shard = in_shard + 1;
        self.state.global_batch_idx += 1;
        self.epoch_cursor += 1;
        Ok(Some((
            vec_i32,
            cu_seqlens,
            doc_ids,
            loss_weights,
            images,
            videos,
            av_align,
            byte_cost,
        )))
    }

    /// P3c §6.2：构造当前 batch 的 `av_align` 条目（join 主函数的 pipeline 装配）。
    fn build_av_align(
        &mut self,
        shard_filename: &str,
        in_shard: u64,
        input_ids: &[i32],
        cu_seqlens: &[i32],
        videos: &Option<Vec<(u64, u32)>>,
    ) -> std::io::Result<Vec<AlignEntry>> {
        let params = self.align_params.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "align requested but align_fingerprint parsing produced no params                  (internal state inconsistency)",
            )
        })?;
        let row_sids = self.read_row_sids(shard_filename, in_shard)?;
        self.ensure_align_audio_cache(self.state.shard_list_index)?;
        // row_lens 由 cu_seqlens 逐差推导（每行有效 token 数）
        let row_lens: Vec<i32> = cu_seqlens.windows(2).map(|w| w[1] - w[0]).collect();
        let video_pairs: Vec<(u64, u32)> = videos.clone().unwrap_or_default();
        // 音频范围缓存是 **Vlm shard 级**（覆盖该 shard 全部样本）；join 到单个
        // batch 时必须按本批 row_sids 过滤——否则同一 shard 内后续 batch 会
        // 重复产出前面样本的音频条目（守恒破坏：rec 出现多次）。
        let audio_recs: Vec<(u32, u32, crate::builder::sidecar::AudioRecord)> = self
            .align_audio_cache
            .iter()
            .filter(|(_, _, rec)| row_sids.contains(&rec.sample_id))
            .cloned()
            .collect();
        // P1-1：非文本 id 区间（image/audio/video）——text 区间末端截断依据。
        // 缺区间 = manifest 自相矛盾（构造期 parse_align_params 已 fail-fast）。
        let visual = self.align_visual_ranges.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "align requested but modality id ranges were not parsed (internal state \
                 inconsistency: align_visual_ranges missing while align_params present)",
            )
        })?;
        Ok(join_align_entries(
            &row_sids,
            input_ids,
            &row_lens,
            self.max_seq_len,
            &video_pairs,
            &audio_recs,
            visual.as_slice(),
            params.ts_base_id,
            params.ts_range_size,
            params.bucket_width_ms,
            params.w_ms,
        ))
    }

    /// P3c：读当前 shard 第 `in_shard` 个 batch 的行 sids（`col_row_sids`，
    /// u64/行定宽；schema key `row_sids` 判定存在性）。align 请求但缺 key
    /// → fail-fast（manifest 自相矛盾：有指纹无列）。
    fn read_row_sids(&mut self, shard_filename: &str, in_shard: u64) -> std::io::Result<Vec<u64>> {
        if self.current_row_sids_file.is_none() {
            if !self.manifest.schema.contains_key("row_sids") {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "[{shard_filename}] align requested but the dataset has no col_row_sids \
                         column (schema key row_sids missing) — the manifest is self-contradictory \
                         (align_fingerprint present without the row-sid column)"
                    ),
                ));
            }
            let filename = sidecar_filename(shard_filename, "col_row_sids");
            let hash = self
                .my_shards
                .get(self.state.shard_list_index)
                .and_then(|sm| sm.hashes.get("sha256:col_row_sids"))
                .map(|s| s.as_str());
            let f = open_cached(
                &self.cloud_io,
                &self.rt,
                &self.base_dir_or_url,
                &filename,
                hash,
            )
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!("Failed to open col_row_sids file {filename} ({e})"),
                )
            })?;
            self.current_row_sids_file = Some(f);
        }
        let bs = self.batch_size as u64;
        let start = in_shard * bs * 8;
        let n = bs as usize;
        let mut buf = vec![0u8; n * 8];
        let f = self.current_row_sids_file.as_mut().expect("opened above");
        f.seek(std::io::SeekFrom::Start(start))?;
        f.read_exact(&mut buf)?;
        let (chunks, _) = buf.as_chunks::<8>();
        Ok(chunks.iter().map(|c| u64::from_le_bytes(*c)).collect())
    }

    /// P3c：确保当前 Vlm shard 的音频范围 rec 缓存已建（`align_aud_range` →
    /// 打开覆盖的 audio shard 句柄 → 读范围 rec）。Vlm shard 切换时由
    /// `open_current_shard` 清空重建。
    ///
    /// **P1-3（评审修复）：局部构建 → 成功后一次性提交**。旧实现"先清空缓存并
    /// 设置 cache key，再做可失败 I/O"：首次进入时若打开/读取 `col_aud_cu` /
    /// `col_aud_rec` 失败即返回错误，而 key 已设置——重试再次进入时因 key 相同
    /// 直接返回 `Ok(())`，**空缓存被当成有效缓存**，batch 正常产出但音频 rec
    /// 全缺（静默对齐数据丢失）。现在 key 只在全部校验/读取成功后设置，失败
    /// 不设 key，下次进入重新完整尝试；已打开的文件句柄保留复用（打开本身
    /// 成功、失败点在后续读取时无需重开）。
    fn ensure_align_audio_cache(&mut self, shard_list_index: usize) -> std::io::Result<()> {
        if self.align_audio_cache_key == Some(shard_list_index) {
            return Ok(());
        }
        let collected = self.build_align_audio_cache(shard_list_index)?;
        // 全部成功才提交（key + cache 原子切换；失败路径不碰既有状态）
        self.align_audio_cache = collected;
        self.align_audio_cache_key = Some(shard_list_index);
        // INV-2 软校验（读侧）：同 sid 内 start_ms 随 seg_order 单调性倒退 →
        // 限频 WARN（数据质量开关；对齐键由单条 rec 自足，不影响正确性）
        self.check_align_monotonic_soft();
        Ok(())
    }

    /// P1-3：`ensure_align_audio_cache` 的构建体（纯局部产出，不提前提交）。
    fn build_align_audio_cache(
        &mut self,
        shard_list_index: usize,
    ) -> std::io::Result<Vec<(u32, u32, crate::builder::sidecar::AudioRecord)>> {
        let Some(range) = self
            .my_shards
            .get(shard_list_index)
            .and_then(|sm| sm.align_aud_range)
        else {
            return Ok(Vec::new()); // 无 audio 流 / 该 shard 无对齐范围
        };
        // P3-2（评审修复）：声明了音频范围却无 audio 流 = manifest 自相矛盾。
        // 旧实现静默返回成功（音频静默缺失），与相邻分支的 fail-fast 不一致。
        if self.manifest.audio_shards.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shard {} declares align_aud_range [{},{}] but the manifest has no \
                     audio_shards — manifest is corrupt",
                    shard_list_index, range.first.0, range.last.0
                ),
            ));
        }
        // audio shard 元数据克隆（避免循环内持 self.manifest 借用与 &mut self 冲突）
        type ShardMetaTuple = (usize, u32, String, Option<String>, Option<String>);
        let shard_metas: Vec<ShardMetaTuple> = self
            .manifest
            .audio_shards
            .iter()
            .enumerate()
            .map(|(i, sm)| {
                (
                    i,
                    sm.num_batches as u32,
                    sm.filename.clone(),
                    sm.hashes.get("sha256:col_aud_cu").cloned(),
                    sm.hashes.get("sha256:col_aud_rec").cloned(),
                )
            })
            .collect();
        // 音频 batch 全局序号 = Σ_{j<shard} num_batches + batch。
        // P3-2：坐标越界**不钳位**（旧实现 `.min(nb-1)` 会把损坏坐标静默映射到
        // 最后一批，读到错误音频范围）；就地报错。
        let mut global_base = 0u32;
        let mut first_global: Option<u32> = None;
        let mut last_global: Option<u32> = None;
        for (s_idx, nb, _, _, _) in &shard_metas {
            let nb = *nb;
            if *s_idx == range.first.0 as usize {
                if range.first.1 >= nb {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "align_aud_range first batch {} is out of range for audio shard {} \
                             (has {} batches) — manifest is corrupt",
                            range.first.1, s_idx, nb
                        ),
                    ));
                }
                first_global = Some(global_base + range.first.1);
            }
            if *s_idx == range.last.0 as usize {
                if range.last.1 >= nb {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "align_aud_range last batch {} is out of range for audio shard {} \
                             (has {} batches) — manifest is corrupt",
                            range.last.1, s_idx, nb
                        ),
                    ));
                }
                last_global = Some(global_base + range.last.1);
            }
            global_base = global_base.saturating_add(nb);
        }
        let (Some(first_global), Some(last_global)) = (first_global, last_global) else {
            // P3c-3（评审）：范围越界（manifest 损坏）——违反 fail-fast 纪律
            // （格式冻结 §5、方案 A7）；返回 InvalidData 而非静默跳过。
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "align_aud_range [{},{}] refers to audio shard(s) beyond the \
                     manifest's audio_shards count — manifest is corrupt",
                    range.first.0, range.last.0
                ),
            ));
        };
        // P3-2：区间顺序校验（first > last = 空窗/反向区间，读侧不猜语义）
        if first_global > last_global {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "align_aud_range first (global batch {first_global}) is after last \
                     (global batch {last_global}) — manifest is corrupt"
                ),
            ));
        }
        // 逐 audio shard 读范围 rec（局部缓冲，避免与 self 的借用冲突）
        let mut collected: Vec<(u32, u32, crate::builder::sidecar::AudioRecord)> = Vec::new();
        let mut g_base = 0u32;
        for (s_idx, nb, filename, cu_hash, rec_hash) in &shard_metas {
            let nb = *nb;
            let (lo, hi) = (g_base, g_base + nb);
            g_base = hi;
            if hi <= first_global || lo > last_global {
                continue;
            }
            let b_lo = first_global.saturating_sub(lo);
            let b_hi = (last_global + 1).saturating_sub(lo).min(nb);
            if !self.align_audio_files.contains_key(s_idx) {
                let stem = filename.trim_end_matches(".aud_bin");
                let open = |ext: &str, hash: &Option<String>| -> std::io::Result<File> {
                    let fname = format!("{stem}.{ext}");
                    open_cached(
                        &self.cloud_io,
                        &self.rt,
                        &self.base_dir_or_url,
                        &fname,
                        hash.as_deref(),
                    )
                    .map_err(|e| {
                        std::io::Error::new(
                            e.kind(),
                            format!("Failed to open audio sidecar file {fname} ({e})"),
                        )
                    })
                };
                let cu_file = open("col_aud_cu", cu_hash)?;
                let rec_file = open("col_aud_rec", rec_hash)?;
                let cu_len = cu_file.metadata()?.len();
                self.align_audio_files.insert(
                    *s_idx,
                    AudioShardFiles {
                        cu_file,
                        cu_len,
                        rec_file,
                    },
                );
            }
            let files = self
                .align_audio_files
                .get_mut(s_idx)
                .expect("just inserted");
            // col_aud_cu 全量计数 + 前缀和
            let mut cu_buf = vec![0u8; files.cu_len as usize];
            files.cu_file.seek(std::io::SeekFrom::Start(0))?;
            files.cu_file.read_exact(&mut cu_buf)?;
            let (cu_chunks, _) = cu_buf.as_chunks::<16>();
            let counts: Vec<u32> = cu_chunks
                .iter()
                .map(|c| u32::from_le_bytes(c[4..8].try_into().unwrap()))
                .collect();
            let mut prefix = 0u32;
            for (b, &cnt) in counts.iter().enumerate() {
                let (s, e) = (prefix, prefix + cnt);
                prefix = e;
                if (b as u32) < b_lo || (b as u32) >= b_hi || cnt == 0 {
                    continue;
                }
                let start = s as u64 * 40;
                let mut rbuf = vec![0u8; cnt as usize * 40];
                files.rec_file.seek(std::io::SeekFrom::Start(start))?;
                files.rec_file.read_exact(&mut rbuf)?;
                let (rec_chunks, _) = rbuf.as_chunks::<40>();
                for (i, chunk) in rec_chunks.iter().enumerate() {
                    collected.push((
                        lo + b as u32,
                        i as u32,
                        crate::builder::sidecar::decode_aud_rec(chunk),
                    ));
                }
            }
        }
        // P1-3：只返回局部结果——提交（cache + key）由 ensure_align_audio_cache
        // 在全部成功后一次性完成。失败路径（上面任何 `?`）不碰既有状态，
        // 重试会重新完整尝试，绝不以"空缓存 + key 已设"冒充有效缓存。
        Ok(collected)
    }

    /// P3c §7（INV-2 软校验）：同 sid 内 start_ms 随 seg_order 倒退 → 限频 WARN。
    fn check_align_monotonic_soft(&self) {
        use std::collections::BTreeMap;
        let mut by_sid: BTreeMap<u64, Vec<(u32, u32)>> = BTreeMap::new();
        for (_, _, rec) in &self.align_audio_cache {
            by_sid
                .entry(rec.sample_id)
                .or_default()
                .push((rec.seg_order, rec.start_ms));
        }
        for (sid, mut segs) in by_sid {
            segs.sort_by_key(|(order, _)| *order);
            for w in segs.windows(2) {
                if w[1].1 < w[0].1 {
                    tracing::warn!(
                        "P3c INV-2 soft: audio start_ms regression sid={sid} seg_order {} → {} \
                         ({}ms → {}ms); alignment keys remain self-describing (WARN only)",
                        w[0].0,
                        w[1].0,
                        w[0].1,
                        w[1].1
                    );
                }
            }
        }
    }
    /// P3b：读当前 shard 第 `in_shard` 个 batch 的 video 记录元数据（40B 定宽
    /// 寻址 + 结构校验；**不含 blob 字节**——条目带 `blob_ref`，字节延迟到
    /// collate 时点按需读）。
    ///
    /// `None` = 数据集无 video sidecar（`open_vid_sidecar` 未构造 reader）；
    /// 有 reader 但该 batch 无 video 记录（纯图像/文本 batch）→ `Some((vec![], pin))`。
    ///
    /// **P1-2（评审修复）**：返回 `(entries, VideoBlobPin)`——产出 batch 时对
    /// 当前 `store_id` pin 一次，guard 随 batch 存活（channel / 前台陈旧丢弃 /
    /// Python GC 任何 drop 路径归还）。reader 切 shard 时 `unregister` 只标记
    /// retired，最后一个 pin 归还才真正关闭句柄——未消费 batch 的按需读不会
    /// 因代际切换拿到 `NotFound`（多 shard video 训练正常时序读取失败）。
    fn read_vid_records_meta(
        &mut self,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<Option<(Vec<MmVideoEntry>, crate::runtime::video_blob::VideoBlobPin)>>
    {
        let Some(reader) = self.current_vid_reader.as_mut() else {
            return Ok(None);
        };
        let store_id = self.current_vid_store_id.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "[{shard_filename}] video sidecar reader exists but its blob store id is \
                     missing (internal state inconsistency)"
                ),
            )
        })?;
        let recs = reader.read_batch_videos(in_shard).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("[{shard_filename}] video sidecar read failed for batch {in_shard}: {e}"),
            )
        })?;
        let entries = recs
            .into_iter()
            .map(|r| MmVideoEntry {
                sample_id: r.sample_id,
                seg_order: r.seg_order,
                n_frames: r.n_frames,
                width: r.width,
                height: r.height,
                flags: r.flags,
                fps_milli: r.fps_milli,
                start_ms: r.start_ms,
                blob_ref: (store_id, r.offset, r.len),
            })
            .collect();
        // pin 在 rec 读取成功后才取（失败不得留下悬空 pin）；条目已带 blob_ref，
        // 句柄必须保活到 batch 消费完毕。
        let pin = crate::runtime::video_blob::VideoBlobStore::global()
            .pin(store_id)
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "[{shard_filename}] failed to pin video blob store {store_id} for \
                         batch {in_shard}: {e} (the shard was retired mid-read; re-read the batch)"
                    ),
                )
            })?;
        Ok(Some((entries, pin)))
    }

    /// **P1-T5（D8）预算门**：在途账本 + 本 batch 账单超预算时延后（自旋等待
    /// 前台消费归还预算），直到 `in_flight + byte_cost <= limit`。
    ///
    /// - 0 预算（不限）直接放行——所有相关逻辑短路，行为与历史版本逐字节一致。
    /// - 等待期间响应取消（CancellationToken），tick 有界（50ms），不持锁。
    /// - **单 batch 账单本身 > 预算**：放行并 WARN 一次（活性保证——若死等，
    ///   `in_flight` 归零后条件仍不成立，数据永远出不来）。超配是用户配置问题，
    ///   窗口峰值退化为"单 batch 账单"，与不限预算的该 batch 相同。
    /// - 停等发生在这里（读 blob 之前）而非 loop 层：保证超限时 blob 不读入。
    ///   代价是等待期间不处理 Seek 指令（旧 gen batch 会被代际栅栏丢弃，
    ///   最长延迟 = 窗口回落时间，与既有节流门同量级）。
    fn await_prefetch_budget(
        &mut self,
        byte_cost: u64,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<()> {
        const BUDGET_WAIT_TICK: Duration = Duration::from_millis(50);
        if self.prefetch_bytes_limit == 0 {
            return Ok(());
        }
        if byte_cost > self.prefetch_bytes_limit {
            if !self.budget_over_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "⚠️ single batch byte cost {byte_cost} exceeds prefetch_bytes_limit {} \
                     (shard {shard_filename} batch {in_shard}); letting it through to keep liveness — \
                     consider raising prefetch_bytes_limit",
                    self.prefetch_bytes_limit
                );
            }
            return Ok(());
        }
        while self
            .in_flight_bytes
            .load(Ordering::Acquire)
            .saturating_add(byte_cost)
            > self.prefetch_bytes_limit
        {
            if self.cancel_token.check().is_err() {
                return Err(cancel_err());
            }
            thread::sleep(BUDGET_WAIT_TICK);
        }
        Ok(())
    }

    /// 读当前 shard 第 `in_shard` 个 batch 的图像记录**元数据**（P1-T4b/T5 两段式
    /// 第 1 段）：rec 定宽寻址 + 结构校验，**不含 blob 字节**——D8 预算门据此先算
    /// 出精确账单，超限时 blob 字节不进内存。
    ///
    /// `None` = 数据集无 mm sidecar（旧数据集，`open_mm_sidecar` 未构造 reader）。
    /// 底层错误已含 batch/sample_id 定位（`mm_sidecar.rs` fail-fast 语义），这里
    /// 统一补 shard 文件名与 batch 下标，保证消费端可定位。
    fn read_mm_records_meta(
        &mut self,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<Option<Vec<ImageRecord>>> {
        let Some(reader) = self.current_mm_reader.as_mut() else {
            return Ok(None);
        };
        reader.read_batch_images(in_shard).map(Some).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("[{shard_filename}] mm sidecar read failed for batch {in_shard}: {e}"),
            )
        })
    }

    /// 两段式第 2 段：按已校验的记录读 blob，组装透出条目（P1-T4b）。
    /// 仅在预算门放行后调用（亲和调度：与主列同批入窗）。
    ///
    /// **P1-T6（D8）共享缓冲**：本 batch 的图像字节按 span **一次连续读入**
    /// `SharedBlobBuffer`（磁盘 → 堆，全链路唯一一次数据拷贝），各条目是它的
    /// 零拷贝 view（`BlobBytes::Shared`，Arc 保活 = 驱逐保护；多 view 单次
    /// 释放）。占位记录（len=0）为空 view。无 VALID 记录（纯文本 batch）时
    /// 不分配缓冲、全部 `Owned` 空。
    fn read_mm_blobs(
        &mut self,
        records: Option<Vec<ImageRecord>>,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<Option<Vec<MmImageEntry>>> {
        use crate::runtime::ffi_dlpack::SharedBlobBuffer;
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        let Some(recs) = records else {
            return Ok(None);
        };
        let Some(reader) = self.current_mm_reader.as_mut() else {
            // records 非空 ⇒ reader 必在（两段间无 open_current_shard 插入点）
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "[{shard_filename}] mm sidecar reader lost between meta and blob reads \
                     for batch {in_shard} (internal state inconsistency)"
                ),
            ));
        };
        // 整 batch 一次读入（读失败 → batch 数据全弃、账本未入账，无泄漏）
        let shared = match MmSidecarReader::batch_blob_span(&recs) {
            None => None,
            Some((start, end)) => {
                let data = reader.read_blob_range(start, end - start).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!(
                            "[{shard_filename}] mm sidecar blob read failed for batch {in_shard} \
                             (range [{start}, {end})): {e}"
                        ),
                    )
                })?;
                Some((
                    start,
                    Arc::new(SharedBlobBuffer {
                        data,
                        release_count: Arc::new(AtomicUsize::new(0)),
                        copy_count: Arc::new(AtomicUsize::new(0)),
                    }),
                ))
            }
        };
        let mut out = Vec::with_capacity(recs.len());
        for rec in &recs {
            let bytes = match (&shared, rec.len > 0) {
                (Some((start, buf)), true) => {
                    debug_assert!(
                        rec.offset >= *start
                            && rec.offset.saturating_add(rec.len as u64)
                                <= start + buf.data.len() as u64,
                        "rec 区间必须落在 batch span 内（写入端 offset 连续不变量）"
                    );
                    BlobBytes::Shared(buf.clone(), rec.offset - *start, rec.len)
                }
                // 占位记录（len=0）或全占位 batch：空字节（语义与 read_blob 一致）
                _ => BlobBytes::Owned(Vec::new()),
            };
            out.push(MmImageEntry {
                bytes,
                sample_id: rec.sample_id,
                seg_order: rec.seg_order,
                grid_h: rec.grid_h,
                grid_w: rec.grid_w,
                flags: rec.flags,
            });
        }
        Ok(Some(out))
    }

    /// 读取 `doc_ids` 列（`[u8]` → `Vec<i32>`，复用 PT 主流的零拷贝/拷贝两态逻辑）。
    fn read_doc_ids_column(
        &mut self,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<Option<Vec<i32>>> {
        let capacity = self.batch_size * self.max_seq_len;
        match read_column_bytes(
            &mut self.current_doc_ids_file, capacity, COL_DOC_IDS_EXT, shard_filename, in_shard, 4,
        )? {
            Some(bytes) => Ok(Some(bytes_into_i32_vec(bytes).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "[{shard_filename}] doc_ids column of batch {in_shard} has a byte count that is not a \
                         multiple of 4 (i32 width); refusing to silently drop the trailing bytes",
                    ),
                )
            })?)),
            None => Ok(None),
        }
    }

    /// 读取 `loss_weights` 列（f16 LE 字节流，保持原样交给 DLPack，dtype 由绑定层声明）。
    fn read_loss_weights_column(
        &mut self,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<Option<Vec<u8>>> {
        let capacity = self.batch_size * self.max_seq_len;
        read_column_bytes(
            &mut self.current_loss_weights_file,
            capacity,
            COL_LOSS_WEIGHTS_EXT,
            shard_filename,
            in_shard,
            2,
        )
    }

    /// 读取一个 PT batch 的原始字节（自动适配裸流 / per-batch zstd frame 两种格式）。
    ///
    /// 返回 `(raw_bytes, compressed_len)`：`compressed_len` 是解压前压缩块的字节数
    /// （Zstd frame；裸流模式无压缩块恒 0）——P1-T5 预算账单据此把"压缩块 / 解码
    /// 中间态"两类字节分开计量（Zstd 模式两者短暂共存）。
    ///
    /// 所有失败都返回 `Err` 并带上定位信息（shard 名 + batch 下标）：这些是真实的
    /// IO / 数据损坏错误，必须上报给消费者，而不是返回 None 让上层当 EOF 无限重试。
    fn read_pt_batch_bytes(
        &mut self,
        shard_filename: &str,
        in_shard: u64,
    ) -> std::io::Result<(Vec<u8>, u64)> {
        // ★ 0.2 Tier-0 快赢项：复用 `open_current_shard` 缓存的 `.bin` 句柄，
        //   消每批 open+fstat（4→2 syscall/batch）。fd 持有期间缓存 LRU 删 .bin
        //   也不影响读（inode 保活）。
        let file = self.current_bin_file.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("shard {shard_filename} main .bin handle is not open (internal state inconsistency: open_current_shard not called?)"),
            )
        })?;
        // 给底层 IO 错误补定位信息：否则 read_exact 只报 "failed to fill whole buffer"，无法定位
        let locate = |e: std::io::Error| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "shard {shard_filename} batch {in_shard} read failed (file truncated / disk corrupted?): {e}"
                ),
            )
        };

        match self.manifest.format {
            StorageFormat::ZstdMds => {
                // per-batch zstd frame：.idx 为压缩流字节偏移（n+1 条）
                let offsets = self.current_bin_offsets.as_ref().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("shard {shard_filename} is missing its .idx offset table (required by Zstd-MDS), cannot locate batch {in_shard}"),
                    )
                })?;
                let i = in_shard as usize;
                let idx_err = |what: &str| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            ".idx offset table of shard {shard_filename} is {what} ({} entries in total), cannot locate batch {i}",
                            offsets.len()
                        ),
                    )
                };
                let start = *offsets.get(i).ok_or_else(|| idx_err("too short"))?;
                let end = *offsets.get(i + 1).ok_or_else(|| idx_err("too short"))?;
                if end <= start {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "shard {shard_filename} batch {i} has invalid .idx offsets (start={start}, end={end})"
                        ),
                    ));
                }
                file.seek(SeekFrom::Start(start)).map_err(locate)?;
                let mut compressed = vec![0u8; (end - start) as usize];
                file.read_exact(&mut compressed).map_err(locate)?;
                let compressed_len = compressed.len() as u64;
                zstd::stream::decode_all(&compressed[..])
                    .map(|raw| (raw, compressed_len))
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "shard {shard_filename} batch {i} zstd decompression failed: {e}"
                            ),
                        )
                    })
            }
            _ => {
                // 裸流：batch 定宽，算术寻址。文件长度在 open_current_shard 时已 stat
                // 并缓存（热路径零 fstat；每批仅 seek + read 共 2 次 syscall）。
                let bin_len = self.current_bin_len.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("shard {shard_filename} main .bin length is not cached (internal state inconsistency: open_current_shard not called?)"),
                    )
                })?;
                let offset_bytes = in_shard * (self.batch_size * self.max_seq_len) as u64 * 4;
                if offset_bytes >= bin_len {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "shard {shard_filename} length {bin_len} is insufficient to hold batch {in_shard} (needs offset {offset_bytes}); file truncated?",
                        ),
                    ));
                }
                file.seek(SeekFrom::Start(offset_bytes)).map_err(locate)?;
                let bytes_to_read = self.batch_size * self.max_seq_len * 4;
                let mut buffer = vec![0u8; bytes_to_read];
                file.read_exact(&mut buffer).map_err(locate)?;
                // 裸流无压缩块（第二返回值 = 解压前压缩块字节数，恒 0；
                // 读出的 buffer 本身就是"解码中间态"那一类）。
                Ok((buffer, 0))
            }
        }
    }

    /// 本地瞬时故障重试前刷新缓存的 `.bin` 长度（仅 error→retry 路径调用）。
    ///
    /// 长度在 `open_current_shard` 时 stat 一次缓存（热路径零 fstat），但本地
    /// "半截文件被外部续写修复"的自愈场景要求**重试能看到新长度**——否则长度校验
    /// 永远拿旧值，重试必然再次失败。远端/正常读取路径不会调用本方法（零成本）。
    ///
    /// 残余局限：若文件被**整体替换**（新 inode）而非原地续写，持有的 fd 仍指向旧
    /// inode；该场景由 shard 级重开兜底，单批重试不覆盖（与测试锁定语义对齐）。
    fn refresh_cached_bin_len_for_retry(&mut self) {
        if let Some(f) = self.current_bin_file.as_ref()
            && let Ok(m) = f.metadata()
        {
            self.current_bin_len = Some(m.len());
        }
    }

    fn open_current_shard(&mut self) -> std::io::Result<()> {
        self.current_bin_offsets = None;
        // 切换 shard 时关闭上一 shard 的主 .bin 与旁路列句柄
        self.current_bin_file = None;
        self.current_bin_len = None;
        self.current_doc_ids_file = None;
        self.current_loss_weights_file = None;
        self.current_mm_reader = None;
        // P3c：对齐缓存随 Vlm shard 切换重建（audio 句柄关闭，防泄漏）
        self.current_row_sids_file = None;
        self.align_audio_cache.clear();
        self.align_audio_cache_key = None;
        self.align_audio_files.clear();
        // P3b：video 句柄随 shard 切换注销。
        // **P1-2（评审修复）**：`unregister` 不再等于"立即关闭"——旧 shard 的
        // batch 可能仍在 channel / 已交付 Python 侧未消费，它们各持一个
        // `VideoBlobPin`，条目只标记 retired，最后一个 pin 归还才真正关闭
        // （详见 `VideoBlobStore::unregister`）。旧实现立即移除 ⇒ 未消费 batch
        // 的按需读拿到 NotFound（多 shard video 训练的正常时序下失败）。
        if let Some(id) = self.current_vid_store_id.take() {
            crate::runtime::video_blob::VideoBlobStore::global().unregister(id);
        }
        self.current_vid_reader = None;
        if self.my_shards.is_empty() {
            self.current_arrow_reader = None;
            self.current_cu_seqlens.clear();
            self.arrow_next_in_shard = None;
            self.arrow_contract_checked = false;
            return Ok(());
        }
        let shard = self.my_shards[self.state.shard_list_index].clone();
        if self.cancel_token.check().is_err() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Cancel",
            ));
        }
        let expected_sha256 = shard.hashes.get("sha256").map(|s| s.as_str());
        // 先物化 .bin（供下面 open_cached 直接命中本地路径；本地源等价于存在性检查）
        self.cloud_io.ensure_local_file_verified(
            &self.rt,
            &self.base_dir_or_url,
            &shard.filename,
            expected_sha256,
        )?;
        if self.cancel_token.check().is_err() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Cancel",
            ));
        }

        // PT 模式：加载 .cu 旁路文件（文档边界）。旧数据集无此文件则保持空。
        // 旁路文件也带 hash（构建期写入 manifest 的 sha256:cu / sha256:idx）：开启
        // MEGADATA_VALIDATE_CACHE=1 时同样走"校验 → 不符则重下"（复核问题 7）。
        if self.manifest.format != StorageFormat::ArrowIpc {
            // P1-4：文件存在即严格校验（记录数 == num_batches、单条结构合法），
            // 结构不符返回 InvalidData——静默截断会让后续 batch 文档边界消失、
            // attention 语义被悄悄改变。
            self.current_cu_seqlens = load_cu_seqlens_for_shard(
                &self.rt,
                &self.cloud_io,
                &self.base_dir_or_url,
                &shard.filename,
                shard.hashes.get("sha256:cu").map(|s| s.as_str()),
                shard.num_batches as u64,
                self.batch_size.saturating_mul(self.max_seq_len),
            )?;
            // Zstd-MDS：加载 .idx 压缩流字节偏移表
            if self.manifest.format == StorageFormat::ZstdMds {
                self.current_bin_offsets = load_bin_offsets_for_shard(
                    &self.rt,
                    &self.cloud_io,
                    &self.base_dir_or_url,
                    &shard.filename,
                    shard.hashes.get("sha256:idx").map(|s| s.as_str()),
                )?;
            }
        }

        // ★ 重新物化后打开：.cu/.idx 的下载会触发缓存限额驱逐，可能刚刚把本 shard 的
        //   .bin 删掉（缓存限额小于工作集时就是这么表现的）。直接 `File::open`
        //   只会抛 ENOENT，用户看到的是"数据读不出来"而不是"缓存太小"。
        let file = open_cached(
            &self.cloud_io,
            &self.rt,
            &self.base_dir_or_url,
            &shard.filename,
            expected_sha256,
        )
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open shard {} ({e})", shard.filename),
            )
        })?;
        if self.manifest.format == StorageFormat::ArrowIpc {
            let reader = FileReader::try_new(file, None).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to open Arrow file: {e}"),
                )
            })?;
            let expected = self.my_shards[self.state.shard_list_index].num_batches;
            if reader.num_batches() != expected {
                tracing::warn!(
                    "⚠️ Batch count mismatch for {}: manifest says {}, Arrow file says {} (trusting the file)",
                    shard.filename,
                    expected,
                    reader.num_batches()
                );
            }
            self.current_arrow_reader = Some(reader);
            self.arrow_next_in_shard = None;
            self.arrow_contract_checked = false;
            // 用 IPC footer 的 block 索引直接定位到目标 batch（O(1)），不再"从头跳过 N 个"
            self.arrow_jump_to(self.state.batches_yielded_in_shard)?;
        } else {
            self.current_arrow_reader = None;
            self.arrow_next_in_shard = None;
            self.arrow_contract_checked = false;
            // ★ 0.2 Tier-0 快赢项：缓存主 `.bin` 句柄（对齐旁路列既有模式）。
            //   此前 `file` 在 PT 分支被直接丢弃、`read_pt_batch_bytes` 每批再 open，
            //   现在保存句柄供 `read_pt_batch_bytes` 复用（open+fstat 每批各消一次）。
            //   fd 持有期间缓存 LRU 删 .bin 也不影响读（inode 保活）。
            //   顺手 stat 一次文件长度缓存：裸流分支的长度校验据此进行，热路径不再
            //   每批 fstat——最终每批仅 seek+read 共 2 次 syscall（兑现注释口径）。
            let bin_len = file.metadata().map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!("Failed to stat shard {} main .bin: {e}", shard.filename),
                )
            })?;
            self.current_bin_len = Some(bin_len.len());
            self.current_bin_file = Some(file);
            // 文档级旁路列（列扩展）：**按 manifest schema key 判定存在性**——旧数据集
            // 没有这些 key ⇒ 不打开、不报错（零兼容成本）。放在主文件重新物化**之后**：
            // 前面的 .cu/.idx 下载可能触发缓存驱逐，这里重开一次保证句柄有效。
            self.current_doc_ids_file = if self.manifest.schema.contains_key("doc_ids") {
                self.open_column_file(
                    &sidecar_filename(&shard.filename, COL_DOC_IDS_EXT),
                    shard.hashes.get("sha256:col_doc_ids").map(|s| s.as_str()),
                )?
            } else {
                None
            };
            self.current_loss_weights_file = if self.manifest.schema.contains_key("loss_weights") {
                self.open_column_file(
                    &sidecar_filename(&shard.filename, COL_LOSS_WEIGHTS_EXT),
                    shard
                        .hashes
                        .get("sha256:col_loss_weights")
                        .map(|s| s.as_str()),
                )?
            } else {
                None
            };
            // 多模态图像 sidecar（P1-T4b，格式冻结 §5）：schema 三 key 齐全才打开。
            // 同样放在主文件重新物化之后（前面的下载可能触发缓存驱逐）。
            self.current_mm_reader = self.open_mm_sidecar(&shard)?;
            // P3b：video sidecar（依赖 mm reader 的 mm_cu 共享句柄 ⇒ 在其后开）
            self.current_vid_reader = self.open_vid_sidecar(&shard)?;
        }
        Ok(())
    }

    /// 多模态图像 sidecar 的打开判定（P1-T4b，格式冻结 §5）：
    ///
    /// - schema **同时含** `img_rec` + `img_blob` + `mm_cu` 三 key ⇒ 经
    ///   `open_cached` 物化三文件（带 `sha256:col_img_blob` 等三键校验）并构造
    ///   `MmSidecarReader`；
    /// - 有部分 key ⇒ **manifest 自相矛盾**，显式报错（不静默跳过、不猜测）；
    /// - 三 key 全无 ⇒ 旧数据集，`Ok(None)`（零行为差异）。
    ///
    /// schema 声明了 key ⇒ 三文件**必须**存在且可打开（含云端物化），缺失/损坏
    /// fail-fast——与 `open_column_file` 的"无 hash 视为无列"兼容策略不同：
    /// 这里的存在性由 schema key 而非文件系统决定。
    fn open_mm_sidecar(&mut self, shard: &ShardMeta) -> std::io::Result<Option<MmSidecarReader>> {
        let has_rec = self.manifest.schema.contains_key("img_rec");
        let has_blob = self.manifest.schema.contains_key("img_blob");
        let has_cu = self.manifest.schema.contains_key("mm_cu");
        if !has_rec && !has_blob && !has_cu {
            return Ok(None);
        }
        if has_rec != has_blob || has_rec != has_cu {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shard {}: manifest schema has an inconsistent mm sidecar key set \
                     (img_rec={}, img_blob={}, mm_cu={}) — all three keys are required, \
                     refusing to guess",
                    shard.filename, has_rec, has_blob, has_cu
                ),
            ));
        }
        let open = |ext: &str, hash_key: &str| -> std::io::Result<File> {
            let filename = sidecar_filename(&shard.filename, ext);
            open_cached(
                &self.cloud_io,
                &self.rt,
                &self.base_dir_or_url,
                &filename,
                shard.hashes.get(hash_key).map(|s| s.as_str()),
            )
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to open mm sidecar file {filename} ({e}) — manifest declares \
                         the mm sidecar, so a missing/corrupt file is fatal",
                    ),
                )
            })
        };
        let reader = MmSidecarReader::from_files(
            open("col_img_blob", "sha256:col_img_blob")?,
            open("col_img_rec", "sha256:col_img_rec")?,
            open("mm_cu", "sha256:mm_cu")?,
        )
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open mm sidecar of shard {}: {e}", shard.filename),
            )
        })?;
        Ok(Some(reader))
    }

    /// P3b：打开 video sidecar（`col_vid_blob` + `col_vid_rec`；计数经共享
    /// `mm_cu`）。打开判定（格式冻结 §5 v1.4 + 评审 P3b-1 修法）：
    /// **按 per-shard hashes 判定**——shard 无 `sha256:col_vid_blob` 哈希即视为
    /// 该 shard 无 video（容忍缺文件，与 schema 全局 key 解耦）。这修复了混合
    /// video 数据集（部分 shard 无 video 样本）构建成功但读取必挂的问题。
    ///
    /// manifest schema 全局 key 仍保留（任一 shard 有 video 即有 vid_rec/vid_blob），
    /// 但 per-shard 缺哈希 = 该 shard 惰性跳过；schema 自相矛盾（有其一而缺其余）
    /// 仍 fail-fast。
    fn open_vid_sidecar(
        &mut self,
        shard: &ShardMeta,
    ) -> std::io::Result<Option<VideoSidecarReader>> {
        let has_rec = shard.hashes.contains_key("sha256:col_vid_rec");
        let has_blob = shard.hashes.contains_key("sha256:col_vid_blob");
        // per-shard 判定：该 shard 无 vid 哈希 = 无 video 记录（惰性创建的产物，
        // P3b §8.4）。与 schema 全局 key 解耦——混合 video 数据集不再读取必挂。
        if !has_rec && !has_blob {
            return Ok(None);
        }
        if !has_rec || !has_blob {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "shard {}: has vid hash for one sidecar file but not the other \
                     (vid_rec={}, vid_blob={}) — both or neither required",
                    shard.filename, has_rec, has_blob
                ),
            ));
        }
        let open =
            |ext: &str, hash_key: &str| -> std::io::Result<File> {
                let filename = sidecar_filename(&shard.filename, ext);
                open_cached(
                &self.cloud_io, &self.rt, &self.base_dir_or_url, &filename,
                shard.hashes.get(hash_key).map(|s| s.as_str()),
            )
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to open video sidecar file {filename} ({e}) — manifest declares \
                         the video sidecar, so a missing/corrupt file is fatal",
                    ),
                )
            })
            };
        // mm_cu 共享句柄（img reader 已开；try_clone 共享 fd，不重复打开）
        let mm_cu = self
            .current_mm_reader
            .as_ref()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "shard {}: video sidecar requires the mm_cu shared handle but the image \
                         sidecar reader is absent (internal state inconsistency)",
                        shard.filename
                    ),
                )
            })?
            .mm_cu_try_clone()?;
        let vid_blob = open("col_vid_blob", "sha256:col_vid_blob")?;
        let vid_blob_len = vid_blob.metadata()?.len();
        // blob 句柄交注册表（collate 按需读）；reader 只带长度
        let store_id =
            crate::runtime::video_blob::VideoBlobStore::global().register(vid_blob, vid_blob_len);
        // P2-3（评审修复）：注册后仍有可失败操作（col_vid_rec 打开 /
        // VideoSidecarReader::from_files）。旧实现构造失败时 `current_vid_store_id`
        // 尚未赋值，刚注册的 id 无法清理 → 注册表与 fd 持续增长（重试损坏
        // sidecar 时尤其明显）。用 RAII guard：构造失败自动注销，成功才移交。
        let mut store_guard =
            crate::runtime::video_blob::VideoBlobStore::global().register_guard(store_id);
        let reader = VideoSidecarReader::from_files(
            open("col_vid_rec", "sha256:col_vid_rec")?,
            mm_cu,
            vid_blob_len,
        )
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to open video sidecar of shard {}: {e}",
                    shard.filename
                ),
            )
        })?;
        store_guard.disarm();
        self.current_vid_store_id = Some(store_id);
        Ok(Some(reader))
    }

    /// 打开旁路列文件（含云端物化 + 缓存校验）。
    /// 文件不存在且 manifest 未记录其 hash ⇒ 视为"该数据集无此列"，返回 `Ok(None)`
    /// （与 `.cu` 的兼容策略一致），不打断读取。
    fn open_column_file(
        &self,
        filename: &str,
        expected_sha256: Option<&str>,
    ) -> std::io::Result<Option<File>> {
        match open_cached(
            &self.cloud_io,
            &self.rt,
            &self.base_dir_or_url,
            filename,
            expected_sha256,
        ) {
            Ok(f) => Ok(Some(f)),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound && expected_sha256.is_none() {
                    Ok(None)
                } else {
                    Err(std::io::Error::new(
                        e.kind(),
                        format!("Failed to open column file {filename} ({e})"),
                    ))
                }
            }
        }
    }

    /// 用 IPC footer 的 block 索引把当前 Arrow reader 直接定位到 shard 内第 `in_shard` 个 batch。
    ///
    /// `FileReader::set_index` 只记 block 下标，`next()` 时 seek 到该 block 的偏移并**只读
    /// 那一个 block** —— 随机访问是 O(1)。旧实现是"重开文件 + `next()` 跳过 in_shard 个
    /// batch"，在置换 shuffle（训练默认模式）下每个 batch 都要付出 O(shard 内 batch 数)
    /// 的解析代价（实测 k≈256 慢 2.1×、k=1009 慢 7.7×），还会吞掉跳过区间的解码错误。
    fn arrow_jump_to(&mut self, in_shard: u64) -> std::io::Result<()> {
        let Some(reader) = self.current_arrow_reader.as_mut() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Arrow reader not ready (shard open flow is abnormal)",
            ));
        };
        let total = reader.num_batches();
        let idx = in_shard as usize;
        if idx >= total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Arrow shard has only {total} batches, but asked to locate batch {idx} (file truncated / manifest inconsistent?)"
                ),
            ));
        }
        reader.set_index(idx).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Arrow random seek failed: {e}"),
            )
        })?;
        self.arrow_next_in_shard = Some(in_shard);
        Ok(())
    }

    /// **R2**：定位到 "第 `epoch` 轮的第 `in_epoch` 个 batch"。
    ///
    /// 与 v2 的 `seek_to_global_batch` 唯一区别：不再用 `idx // per_epoch` 反推 epoch
    /// （该推导要求所有 epoch 等长，见 B1 实测），而是直接接收二元组。
    ///
    /// 校验：`in_epoch >= epoch_total_of(epoch)` → 报错（游标一律是"下一产出位置"口径，
    /// 越界说明调用方算错了轮长）。
    pub fn seek_to_epoch_batch(&mut self, epoch: u64, in_epoch: u64) -> std::io::Result<()> {
        let total = self.epoch_total_of(epoch);
        if total == 0 {
            // 空 rank（本轮没有 shard）：加载该 epoch 的空 shard 集，读头立即 EOF
            self.load_epoch_shards(epoch);
            self.epoch_cursor = 0;
            self.state.global_batch_idx = 0;
            self.open_shard = None;
            return Ok(());
        }
        if in_epoch >= total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "seek out of range: epoch {epoch} has only {total} batches, cannot seek to in_epoch={in_epoch}.\
                     The cursor must be the 'next position to produce' (normalize to (epoch+1, 0) when exhausted)."
                ),
            ));
        }

        self.load_epoch_shards(epoch);
        self.epoch_cursor = in_epoch;
        self.state.global_batch_idx = in_epoch;
        self.open_shard = None;

        // 定位物理读头（下一次 next() 直接命中目标 batch）
        if let Some((shard_i, in_shard)) = self.position_target(in_epoch) {
            self.ensure_shard_open(shard_i, in_shard)?;
        }
        Ok(())
    }
}

/// Arrow batch 级列（cu_seqlens / row_lens）读取侧契约校验（每 shard 首个 batch 一次）。
///
/// 写侧约定（`writer.rs`）：cu_seqlens 与 row_lens 是 **batch 级列**，每个 RecordBatch
/// 只写第 0 行，读取端按第 0 行取。若文件被三方工具改写、或跨版本格式漂移导致
/// 约定被破坏，在此响亮报错——而不是静默读出错乱的 mask / 装箱切分。
fn validate_arrow_batch_level_columns(
    batch: &RecordBatch,
    batch_size: usize,
    shard: &str,
) -> std::io::Result<()> {
    use arrow::array::{Array, Int32Array, ListArray};

    let err = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    fn downcast_list<'a>(
        col: &'a Arc<dyn Array>,
        name: &str,
        shard: &str,
    ) -> std::io::Result<&'a ListArray> {
        col.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "[{shard}] {name} must be List<Int32>, got {}",
                    col.data_type()
                ),
            )
        })
    }
    let cu_list = downcast_list(
        batch
            .column_by_name("cu_seqlens")
            .ok_or_else(|| err(format!("[{shard}] Arrow shard missing cu_seqlens column")))?,
        "cu_seqlens",
        shard,
    )?;
    let row_list = downcast_list(
        batch
            .column_by_name("row_lens")
            .ok_or_else(|| err(format!("[{shard}] Arrow shard missing row_lens column")))?,
        "row_lens",
        shard,
    )?;
    // 写侧约定：一个 RecordBatch = 一个 packed batch（行数 == batch_size），
    // batch 级列（cu_seqlens / row_lens）**只填第 0 行**，其余行是空列表。
    if cu_list.len() != batch_size || row_list.len() != batch_size {
        return Err(err(format!(
            "[{shard}] batch-level columns must have batch_size ({batch_size}) rows \
             (write-side convention), got cu_seqlens={} row_lens={}",
            cu_list.len(),
            row_list.len()
        )));
    }
    if batch_size > 1 && (!cu_list.value(1).is_empty() || !row_list.value(1).is_empty()) {
        return Err(err(format!(
            "[{shard}] batch-level columns must be empty for rows > 0 (only row 0 is populated), \
             got row 1 cu_seqlens={} row_lens={} entries",
            cu_list.value(1).len(),
            row_list.value(1).len()
        )));
    }
    let int32_values = |list: &ListArray, name: &str| -> std::io::Result<Vec<i32>> {
        let inner = list.value(0);
        let arr = inner.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
            err(format!(
                "[{shard}] {name} values must be Int32, got {}",
                inner.data_type()
            ))
        })?;
        Ok((0..arr.len()).map(|i| arr.value(i)).collect())
    };
    let cu = int32_values(cu_list, "cu_seqlens")?;
    let row = int32_values(row_list, "row_lens")?;
    // cu_seqlens 契约：非空、首 0、非递减
    if cu.is_empty() || cu[0] != 0 {
        return Err(err(format!(
            "[{shard}] cu_seqlens must be non-empty and start at 0, got {cu:?}"
        )));
    }
    if cu.windows(2).any(|w| w[1] < w[0]) {
        return Err(err(format!(
            "[{shard}] cu_seqlens must be non-decreasing, got {cu:?}"
        )));
    }
    // row_lens 契约：每行有效长度，恒为 batch_size 长，总和 == cu_seqlens 末值
    if row.len() != batch_size {
        return Err(err(format!(
            "[{shard}] row_lens must always be batch_size ({batch_size}) long, got {}",
            row.len()
        )));
    }
    let row_sum: i32 = row.iter().sum();
    if row_sum != cu[cu.len() - 1] {
        return Err(err(format!(
            "[{shard}] sum(row_lens)={row_sum} must equal cu_seqlens last value ({})",
            cu[cu.len() - 1]
        )));
    }
    Ok(())
}

/// `Vec<u8>` → `Vec<i32>`（**长度必须是 4 的倍数**，否则返回 `None`）。
///
/// - 4 字节对齐且容量也是 4 的倍数：零拷贝转交（`mem::forget` 后新 Vec 接管同一块堆，
///   最终经 `vec_to_dlpack_capsule` 交给 DLPack deleter 释放）；
/// - 否则退化为一次拷贝：`Vec<u8>` 的分配对齐只有 1（`align_of::<i32>() == 4`），
///   把 `*mut u8` 当 `*mut i32` 用是 UB；容量不是 4 的倍数时 `from_raw_parts` 的 cap
///   还会小于真实分配，drop 时同样 UB（复核问题 12）。
///
/// ★ 评审 §3.6(b)：长度不变量原先只是 `debug_assert_eq!`，release 下失效时两个分支
///   **都会静默丢弃尾部 1-3 字节**（零拷贝分支 `len = bytes.len()/4` 截断；拷贝分支
///   `_rest` 直接不要了）——是数据损坏而非 UB，所以此前定级 Minor。但"数据被静默
///   丢 1-3 个 token"在训练侧表现为难以定位的诡异行为，因此现在改为**运行时校验 +
///   显式返回 None**，让调用方报错而不是继续。
///
/// 对齐/容量（真正与 UB 相关的部分）本来就是运行时检查，与 `bytemuck::try_cast_vec`
/// 同构，这部分保持不变。
fn bytes_into_i32_vec(bytes: Vec<u8>) -> Option<Vec<i32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    if (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<i32>())
        && bytes.capacity().is_multiple_of(4)
    {
        let mut bytes = bytes;
        let ptr = bytes.as_mut_ptr() as *mut i32;
        let len = bytes.len() / 4;
        let cap = bytes.capacity() / 4;
        std::mem::forget(bytes);
        // SAFETY: 对齐与容量都已校验；i32 与 u8 布局兼容（LE 平台字节序一致），
        // 长度已在上面做过运行时校验（4 的倍数）。
        Some(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    } else {
        tracing::debug!(
            "batch buffer does not satisfy i32 zero-copy conditions (ptr={:p} cap={}), falling back to one copy",
            bytes.as_ptr(),
            bytes.capacity()
        );
        let (chunks, _rest) = bytes.as_chunks::<4>();
        Some(chunks.iter().map(|c| i32::from_le_bytes(*c)).collect())
    }
}

/// "首次 NotFound → 重物化一次 → 再试"的通用骨架（open_cached / read_cached_bytes 共用）。
///
/// 背景：`MEGADATA_CACHE_LIMIT` 小于工作集时，后续下载触发的 LRU 驱逐会删掉
/// "已经交给调用方、但还没打开/读取"的缓存文件——直接 `File::open` 只会抛 ENOENT，
/// 用户看到的是"数据读不出来"而不是"缓存太小"。重取路径自身受 `just_added` 保护
/// （`enforce_cache_limit` 永不驱逐刚下载的文件），因此重试一次即可读通。
/// 第二次仍失败时补上文件名定位信息，不再裸抛。
fn with_cache_retry<T>(
    cloud_io: &CloudIoManager,
    rt: &tokio::runtime::Runtime,
    base_dir_or_url: &str,
    filename: &str,
    expected_sha256: Option<&str>,
    op: impl Fn(&str) -> std::io::Result<T>,
) -> std::io::Result<T> {
    let path =
        cloud_io.ensure_local_file_verified(rt, base_dir_or_url, filename, expected_sha256)?;
    match op(&path) {
        Ok(v) => Ok(v),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let path = cloud_io.ensure_local_file_verified(
                rt,
                base_dir_or_url,
                filename,
                expected_sha256,
            )?;
            op(&path).map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!("failed to open/read cached file {filename} even after re-materialization (cache dir over limit?): {e}"),
                )
            })
        }
        Err(e) => Err(e),
    }
}

/// 打开（可能是云端缓存下来的）文件；**缓存被驱逐时重新物化一次**再打开。
fn open_cached(
    cloud_io: &CloudIoManager,
    rt: &tokio::runtime::Runtime,
    base_dir_or_url: &str,
    filename: &str,
    expected_sha256: Option<&str>,
) -> std::io::Result<File> {
    with_cache_retry(
        cloud_io,
        rt,
        base_dir_or_url,
        filename,
        expected_sha256,
        |p| File::open(p),
    )
}

/// 同 [`open_cached`]，但一次性读全部字节（.cu / .idx 这类小旁路文件）。
fn read_cached_bytes(
    rt: &tokio::runtime::Runtime,
    cloud_io: &CloudIoManager,
    base_dir_or_url: &str,
    filename: &str,
    expected_sha256: Option<&str>,
) -> std::io::Result<Vec<u8>> {
    with_cache_retry(
        cloud_io,
        rt,
        base_dir_or_url,
        filename,
        expected_sha256,
        |p| std::fs::read(p),
    )
}

/// shard 文件名 → 旁路文件名（.bin → .cu / .idx）。
///
/// 用后缀替换（`Path::with_extension`）而非全局 `replace(".bin", ..)`：
/// 文件名中间含 ".bin" 子串时（如 `my.bin.data.bin`），replace 会把每一处
/// 都替换掉（`my.cu.data.cu`，错误），with_extension 只替换末尾扩展名。
/// 与构建期 MemmapWriter 的命名约定一致。
fn sidecar_filename(shard_filename: &str, ext: &str) -> String {
    std::path::Path::new(shard_filename)
        .with_extension(ext)
        .to_string_lossy()
        .to_string()
}

/// 定宽旁路列的通用读取：`batch_idx * elem_size * capacity`。
/// 句柄为 `None`（数据集无该列）时直接 `Ok(None)`，零 IO、零行为变化。
fn read_column_bytes(
    handle: &mut Option<File>,
    capacity: usize,
    ext: &str,
    shard_filename: &str,
    in_shard: u64,
    elem_size: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    let Some(file) = handle.as_mut() else {
        return Ok(None);
    };
    // ★ 评审 §3.6(b)：`saturating_mul` 饱和到 usize::MAX 后，`vec![0u8; len]`
    //   会直接 OOM/panic，而`batch_size` / `max_seq_len` / `in_shard` 都来自
    //   Python 侧用户输入或磁盘上的 index.json。这里改为**显式拒绝**超长读请求，
    //   把"分配时崩溃"换成"带上下文的可操作错误"。
    //   上限取 2GiB：单个 batch 的旁路列不可能合理地超过它（8192×65536×4B ≈ 2GiB
    //   已远超任何合理训练配置），而 Rust 侧 `vec!` 超过 isize::MAX 也会直接崩。
    const MAX_COLUMN_BYTES: usize = 2 * 1024 * 1024 * 1024;
    let offset = in_shard
        .checked_mul(capacity as u64)
        .and_then(|v| v.checked_mul(elem_size as u64))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "shard {shard_filename} .{ext} column: batch {in_shard} × {capacity} × {elem_size} \
                     overflows the addressable range (batch_size/max_seq_len too large?)"
                ),
            )
        })?;
    let len = capacity.checked_mul(elem_size).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "shard {shard_filename} .{ext} column: capacity {capacity} × elem_size {elem_size} \
                 overflows usize (batch_size/max_seq_len too large?)"
            ),
        )
    })?;
    if len > MAX_COLUMN_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "shard {shard_filename} .{ext} column: a single batch would need {len} bytes \
                 (> {MAX_COLUMN_BYTES} limit); refusing to allocate. Check batch_size × max_seq_len."
            ),
        ));
    }
    file.seek(SeekFrom::Start(offset)).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("shard {shard_filename} .{ext} column seek to batch {in_shard} failed: {e}"),
        )
    })?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!(
                "shard {shard_filename} .{ext} column batch {in_shard} read failed \
                 (file truncated / corrupted?): {e}"
            ),
        )
    })?;
    Ok(Some(buf))
}

/// 加载 shard 的 .cu 文件（cu_seqlens 旁路）。
///
/// 文件格式：`[u32 count LE][i32 * count LE]` 重复（每 batch 一条记录）。
/// - 旧数据集**没有** .cu 文件（manifest 也没记 hash）→ 返回空 Vec（标准 attention 语义，向后兼容）；
/// - manifest 记了 `sha256:cu` 却拿不到合格文件（缓存损坏且重下失败）→ 返回 `Err`：
///   此时静默退回空 cu_seqlens 会**悄悄改变 attention 语义**，必须让用户知道。
///
/// **P1-4（评审修复）：文件存在即严格校验，结构不符返回 `InvalidData`**。旧实现
/// 解析到剩余字节不足时只 `WARN + break` 返回已解析前缀——后续 batch 的文档
/// 边界消失，消费端可能退回标准 attention（跨文档注意力污染），而 WARN 不阻止
/// 语义变化，与"数据损坏应上报"契约不符。现在校验：
/// - 单条记录字节不足（截断）→ `InvalidData`（不返回前缀）；
/// - 解析出的记录数 != `expected_num_batches`（`ShardMeta.num_batches`）→ `InvalidData`；
/// - 单条 cu 非空、首项为 0、严格单调递增、末项 ≤ `batch_capacity`
///   （`batch_size × max_seq_len`）——任一不符 → `InvalidData`。
///   只有"旧格式明确无 .cu 文件"（NotFound 且无 hash）才允许空边界兼容。
fn load_cu_seqlens_for_shard(
    rt: &tokio::runtime::Runtime,
    cloud_io: &CloudIoManager,
    base_dir_or_url: &str,
    shard_filename: &str,
    expected_sha256: Option<&str>,
    expected_num_batches: u64,
    batch_capacity: usize,
) -> std::io::Result<Vec<Vec<i32>>> {
    // .bin → .cu（与构建期 MemmapWriter 的命名约定一致）
    let cu_filename = sidecar_filename(shard_filename, "cu");
    match cloud_io.ensure_local_file_verified(rt, base_dir_or_url, &cu_filename, expected_sha256) {
        Ok(_) => {}
        Err(_) if expected_sha256.is_none() => return Ok(Vec::new()), // 旧数据集无 .cu 文件
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{cu_filename} failed to fetch/verify: {e} (clear the cache and retry, or rebuild the dataset)"
                ),
            ));
        }
    }
    let bytes =
        match read_cached_bytes(rt, cloud_io, base_dir_or_url, &cu_filename, expected_sha256) {
            Ok(b) => b,
            // 文件不存在且 manifest 也没记 hash = 旧数据集没有 .cu 旁路（本地路径不会走下载分支）
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && expected_sha256.is_none() => {
                return Ok(Vec::new());
            }
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{cu_filename} failed to read: {e}"),
                ));
            }
        };

    let mut result = Vec::new();
    let mut offset = 0usize;
    while offset + 4 <= bytes.len() {
        let count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        // ★ 评审 §3.6(a)：`count * 4` 在 32 位平台上会溢出 usize（count 最大 2^32-1，
        //   ×4 超u32::MAX）。一旦溢出，下面 `offset + needed > bytes.len()` 的截断
        //   保护会变成"小值比较"而通过，随后 `Vec::with_capacity(count)` 按 40 亿
        //   分配 → OOM / DoS。`count` 来自用户提供的 .cu 文件内容，属**不可信输入**。
        //   现在先用 u64 做乘法再收窄，溢出即报错退出（不做截断猜测）。
        let needed = match (count as u64).checked_mul(4) {
            Some(n) if n <= usize::MAX as u64 => n as usize,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{cu_filename} is corrupted: record #{} declares {count} boundaries, which overflows the addressable range; refusing to allocate",
                        result.len()
                    ),
                ));
            }
        };
        if offset + needed > bytes.len() {
            // P1-4：截断 = 文件损坏，**不再** WARN+break 返回前缀（那会让后续 batch
            // 的文档边界消失、attention 语义被静默改变）。fail-fast。
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{cu_filename} is corrupted: parsed {} batch record(s), but the remaining {} \
                     bytes are insufficient for record #{} (declares {count} boundaries = \
                     {needed} bytes). Refusing to fall back to a partial prefix — that would \
                     silently drop document boundaries for later batches and change attention \
                     semantics (clear the cache and re-download, or rebuild the dataset).",
                    result.len(),
                    bytes.len() - offset,
                    result.len()
                ),
            ));
        }
        // 容量以实际剩余字节为上界：即便 count 合法，也不允许一次申请超过文件
        // 剩余量的内存（旧实现按声明值分配，损坏文件可借此放大内存占用）。
        let mut cu = Vec::with_capacity(count.min(bytes.len() - offset));
        for _ in 0..count {
            let val = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            cu.push(val);
            offset += 4;
        }
        // P1-4：单条 cu 结构校验——非空 / 首项 0 / 非递减 / 不超 batch 容量。
        // 任一不符 = 文件与构建期 writer 契约不符（损坏或格式演进），fail-fast。
        // 注意：**允许相等**——定宽 VLM 批的 padding 空行会产出重复边界
        // （如 `[0, 2, 2]`，batch_size=2 且仅 1 行有效），这是合法形态。
        let bad = |why: String| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{cu_filename} record #{} is malformed ({why}): {cu:?} — refusing to read \
                     with corrupt document boundaries (clear the cache and re-download, or \
                     rebuild the dataset)",
                    result.len()
                ),
            )
        };
        if cu.is_empty() {
            return Err(bad(
                "empty boundary list (every batch must have at least one document)".to_string(),
            ));
        }
        if cu[0] != 0 {
            return Err(bad("first boundary must be 0".to_string()));
        }
        if cu.windows(2).any(|w| w[1] < w[0]) {
            return Err(bad("boundaries must be non-decreasing".to_string()));
        }
        let last = *cu.last().expect("non-empty checked above");
        if last < 0 || last as usize > batch_capacity {
            return Err(bad(format!(
                "last boundary {last} outside [0, batch_capacity {batch_capacity}]"
            )));
        }
        result.push(cu);
    }
    // P1-4：记录数必须等于 manifest 的 num_batches（文件存在但缺记录 = 截断）
    if result.len() as u64 != expected_num_batches {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{cu_filename} contains {} batch record(s) but the manifest declares \
                 {expected_num_batches} — the file is truncated or the manifest is corrupt \
                 (clear the cache and re-download, or rebuild the dataset)",
                result.len()
            ),
        ));
    }
    Ok(result)
}

/// 加载 shard 的 .idx 文件（Zstd-MDS 格式：压缩流字节偏移，u64 LE，n+1 条）。
///
/// 与 .cu 同款策略：manifest 未记 hash 的旧数据集缺失 → `Ok(None)`（按裸流读）；
/// 记了 hash 却拿不到合格文件 → `Err`（否则会静默按错误格式读压缩流）。
fn load_bin_offsets_for_shard(
    rt: &tokio::runtime::Runtime,
    cloud_io: &CloudIoManager,
    base_dir_or_url: &str,
    shard_filename: &str,
    expected_sha256: Option<&str>,
) -> std::io::Result<Option<Vec<u64>>> {
    let idx_filename = sidecar_filename(shard_filename, "idx");
    if let Err(e) =
        cloud_io.ensure_local_file_verified(rt, base_dir_or_url, &idx_filename, expected_sha256)
    {
        if expected_sha256.is_none() {
            return Ok(None); // 旧数据集无 .idx 文件
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{idx_filename} failed to fetch/verify: {e} (clear the cache and retry, or rebuild the dataset)"
            ),
        ));
    }
    let bytes = match read_cached_bytes(
        rt,
        cloud_io,
        base_dir_or_url,
        &idx_filename,
        expected_sha256,
    ) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && expected_sha256.is_none() => {
            return Ok(None); // 旧数据集没有 .idx 旁路
        }
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{idx_filename} failed to read: {e}"),
            ));
        }
    };
    if bytes.len() % 8 != 0 {
        // 长度不对 = 文件损坏（曾经静默退化按裸流读 ⇒ 解压数据全错）；现在直接报错
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{idx_filename} length {} is not a multiple of 8; the offset table is corrupted (clear the cache and re-download, or rebuild the dataset)",
                bytes.len()
            ),
        ));
    }
    let (chunks, _rest) = bytes.as_chunks::<8>();
    Ok(Some(
        chunks.iter().map(|c| u64::from_le_bytes(*c)).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::metadata::{DatasetManifest, ShardMeta};

    /// 造一个"本地数据集"：index.json + 指定内容的 shard_00000.bin
    fn make_local_dataset(
        tag: &str,
        num_batches: usize,
        batch_size: usize,
        max_seq_len: usize,
        shard_bytes: usize,
    ) -> std::path::PathBuf {
        make_local_dataset_with_columns(
            tag,
            num_batches,
            batch_size,
            max_seq_len,
            shard_bytes,
            false,
        )
    }

    /// 同 [`make_local_dataset`]，可选地带上文档级旁路列（列扩展）：
    /// `.col_doc_ids` / `.col_loss_weights` 定宽文件 + manifest schema key。
    fn make_local_dataset_with_columns(
        tag: &str,
        num_batches: usize,
        batch_size: usize,
        max_seq_len: usize,
        shard_bytes: usize,
        extra_columns: bool,
    ) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("megadata_stream_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = DatasetManifest::new(StorageFormat::Memmap, max_seq_len);
        manifest.global_stats.batch_size = batch_size;
        manifest.shards.push(ShardMeta {
            shard_id: 0,
            filename: "shard_00000.bin".to_string(),
            num_batches,
            size_bytes: shard_bytes as u64,
            hashes: Default::default(),
            align_aud_range: None,
        });
        manifest
            .save_to_file(dir.join("index.json").to_str().unwrap())
            .unwrap();
        std::fs::write(dir.join("shard_00000.bin"), vec![0u8; shard_bytes]).unwrap();
        if extra_columns {
            manifest
                .schema
                .insert("doc_ids".to_string(), "list<int32>".to_string());
            manifest
                .schema
                .insert("loss_weights".to_string(), "list<f16>".to_string());
            manifest
                .save_to_file(dir.join("index.json").to_str().unwrap())
                .unwrap();
            let capacity = batch_size * max_seq_len;
            // doc_ids：每个 batch 的 token 全标该 batch 的下标（便于断言可区分）
            let mut doc_ids: Vec<i32> = Vec::new();
            // loss_weights：f16 1.0（0x3C00 LE）
            let mut weights: Vec<u8> = Vec::new();
            for b in 0..num_batches {
                doc_ids.extend(std::iter::repeat_n(b as i32, capacity));
                weights.extend(std::iter::repeat_n([0x00u8, 0x3Cu8], capacity).flatten());
            }
            std::fs::write(
                dir.join("shard_00000.col_doc_ids"),
                doc_ids
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
            std::fs::write(dir.join("shard_00000.col_loss_weights"), weights).unwrap();
        }
        dir
    }

    fn open_reader(dir: &std::path::Path, batch_size: usize, max_seq_len: usize) -> StreamReader {
        StreamReader::with_options(
            dir.join("index.json").to_str().unwrap(),
            batch_size,
            max_seq_len,
            1,
            0,
            42,
            StreamReaderOptions::default(),
        )
        .unwrap()
    }

    /// 同 [`make_local_dataset_with_columns`]，额外用 `MmSidecarWriter`（P1-T2）
    /// 写图像 sidecar 三文件 + manifest 插三 key（P1-T4b stream 接缝测试基建；
    /// Rust writer→reader roundtrip 由 `builder/sidecar` 与 `runtime/mm_sidecar`
    /// 单测覆盖，这里只测 stream 接缝）。
    ///
    /// sidecar 布局（`num_batches` 必须为 3）：
    /// - batch 0：1 张 VALID（10B，grid 2x2，sample_id=0，seg_order=0）
    /// - batch 1：纯文本（0 图）
    /// - batch 2：VALID（4B，grid 1x1，sample_id=3，seg_order=1）+ NO_IMAGE 占位（sample_id=4）
    ///
    /// 返回 `(dir, 各 VALID 图 bytes)`；manifest hashes 带 `finish()` 的真实 sha256。
    fn make_local_dataset_with_mm(
        tag: &str,
        num_batches: usize,
        batch_size: usize,
        max_seq_len: usize,
        shard_bytes: usize,
    ) -> (std::path::PathBuf, Vec<Vec<u8>>) {
        let imgs: Vec<Vec<u8>> = vec![vec![7u8; 10], vec![8u8; 4]];
        let per_batch = vec![
            vec![Some(imgs[0].clone())],
            vec![],
            vec![Some(imgs[1].clone()), None],
        ];
        let dir = make_mm_dataset_with_payloads(
            tag,
            num_batches,
            batch_size,
            max_seq_len,
            shard_bytes,
            &per_batch,
        );
        (dir, imgs)
    }

    /// 底层构造（T5 压力测试基建）：每 batch 的图像 payload 由调用方指定
    /// （`Some(bytes)` = VALID，`None` = NO_IMAGE 占位；空 Vec = 纯文本 batch），
    /// grid 统一 1x1、sample_id 递增、seg_order=0。hashes/schema 与
    /// [`make_local_dataset_with_mm`] 同款。
    fn make_mm_dataset_with_payloads(
        tag: &str,
        num_batches: usize,
        batch_size: usize,
        max_seq_len: usize,
        shard_bytes: usize,
        per_batch: &[Vec<Option<Vec<u8>>>],
    ) -> std::path::PathBuf {
        use crate::builder::sidecar::{ImageRecordInput, MmSidecarWriter, img_flags};
        assert_eq!(per_batch.len(), num_batches);
        let dir =
            std::env::temp_dir().join(format!("megadata_stream_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = DatasetManifest::new(StorageFormat::Memmap, max_seq_len);
        manifest.global_stats.batch_size = batch_size;
        manifest.shards.push(ShardMeta {
            shard_id: 0,
            filename: "shard_00000.bin".to_string(),
            num_batches,
            size_bytes: shard_bytes as u64,
            hashes: Default::default(),
            align_aud_range: None,
        });
        std::fs::write(dir.join("shard_00000.bin"), vec![0u8; shard_bytes]).unwrap();
        let mut w = MmSidecarWriter::create(&dir, "shard_00000").unwrap();
        let mut sample_id = 0u64;
        for imgs in per_batch {
            let inputs: Vec<ImageRecordInput> = imgs
                .iter()
                .map(|p| ImageRecordInput {
                    bytes: p.as_deref(),
                    grid_h: 1,
                    grid_w: 1,
                    flags: if p.is_some() {
                        img_flags::VALID
                    } else {
                        img_flags::NO_IMAGE
                    },
                    sample_id: {
                        sample_id += 1;
                        sample_id - 1
                    },
                    seg_order: 0,
                })
                .collect();
            w.append_mm_batch(&inputs, 0).unwrap();
        }
        let hashes = w.finish().unwrap();
        for (k, v) in hashes {
            manifest.shards[0].hashes.insert(k, v);
        }
        // schema 三 key（格式冻结 §5：对齐 doc_ids 无前缀先例）
        manifest
            .schema
            .insert("img_blob".to_string(), "bytes".to_string());
        manifest
            .schema
            .insert("img_rec".to_string(), "imgrec32".to_string());
        manifest
            .schema
            .insert("mm_cu".to_string(), "mmcu".to_string());
        manifest
            .save_to_file(dir.join("index.json").to_str().unwrap())
            .unwrap();
        dir
    }

    /// 从磁盘重载 mm 数据集的 manifest 并改写 schema key 集合后落回（负面测试用）。
    fn rewrite_mm_schema(dir: &std::path::Path, keep: &[&str]) {
        let mut manifest =
            DatasetManifest::load_from_file(dir.join("index.json").to_str().unwrap()).unwrap();
        for k in ["img_blob", "img_rec", "mm_cu"] {
            manifest.schema.remove(k);
        }
        for k in keep {
            manifest.schema.insert(k.to_string(), "x".to_string());
        }
        manifest
            .save_to_file(dir.join("index.json").to_str().unwrap())
            .unwrap();
    }

    /// 预期构造失败的打开：`StreamReader` 不实现 Debug，`unwrap_err()` 用不了，
    /// 手动 match 提取 io::Error（构造期预打开首个 shard 即触发负面路径）。
    fn open_reader_expect_err(
        dir: &std::path::Path,
        batch_size: usize,
        max_seq_len: usize,
    ) -> std::io::Error {
        match StreamReader::with_options(
            dir.join("index.json").to_str().unwrap(),
            batch_size,
            max_seq_len,
            1,
            0,
            42,
            StreamReaderOptions::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!(
                "expected with_options to fail (inconsistent manifest / missing sidecar file)"
            ),
        }
    }

    /// 带超时地取下一个"非超时"结果：None = 超时（数据一直不来）。
    fn next_within(
        reader: &StreamReader,
        timeout: Duration,
    ) -> Option<Result<Option<BatchData>, ReaderExit>> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match reader.try_next() {
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        return None;
                    }
                }
                other => return Some(other),
            }
        }
    }

    /// 毒丸数据集：shard 文件名带 [`TEST_PANIC_PILL`] 标记，读取时在预取线程内 panic。
    /// 字节内容是合法尺寸的（毒丸在读第一个 batch 前就炸，不依赖内容）。
    fn make_panic_pill_dataset(tag: &str) -> std::path::PathBuf {
        let (batch_size, max_seq_len, num_batches) = (2usize, 4usize, 2usize);
        let dir =
            std::env::temp_dir().join(format!("megadata_stream_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let filename = format!("shard_00000{TEST_PANIC_PILL}bin");
        let mut manifest = DatasetManifest::new(StorageFormat::Memmap, max_seq_len);
        manifest.global_stats.batch_size = batch_size;
        manifest.shards.push(ShardMeta {
            shard_id: 0,
            filename: filename.clone(),
            num_batches,
            size_bytes: (batch_size * max_seq_len * 4 * num_batches) as u64,
            hashes: Default::default(),
            align_aud_range: None,
        });
        manifest
            .save_to_file(dir.join("index.json").to_str().unwrap())
            .unwrap();
        std::fs::write(
            dir.join(&filename),
            vec![0u8; batch_size * max_seq_len * 4 * num_batches],
        )
        .unwrap();
        dir
    }

    /// 离开作用域时恢复此前的 panic hook：毒丸测试会触发 1000 次 panic，默认 hook
    /// 的输出会淹没并行运行的其它测试的失败信息（断言失败时同样需要可读输出）。
    /// （`std::panic::PanicHook` 是 std 私有别名，这里显式写出其真实类型。）
    struct HookRestore(Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static>);
    impl Drop for HookRestore {
        fn drop(&mut self) {
            std::panic::set_hook(std::mem::replace(&mut self.0, Box::new(|_| {})));
        }
    }

    /// ★ P0/N2 行为级回归（20261007 复核报告"立即"项 #3）：
    /// 预取线程 panic 必须让 `try_next()` 返回 `Err(ReaderExit::Fatal(..))`，
    /// **绝不出现** `Err(ReaderExit::Eof)`（那等于训练数据被静默截断）。
    ///
    /// 循环 1000 次防 flaky：竞态类缺陷（fatal 写入晚于 data_tx 断开）单次运行
    /// 可能碰巧不触发，只有大样本重复才能把概率压到可忽略。
    ///
    /// 每轮还断言 Fatal **可重复读取**（幂等）：消费端多拉一次不会突然变成 Eof。
    #[test]
    fn prefetch_thread_panic_surfaces_fatal_never_eof() {
        let _silence = {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            HookRestore(prev)
        };
        let dir = make_panic_pill_dataset("panic_pill");
        for round in 0..1000 {
            let reader = open_reader(&dir, 2, 4);
            match next_within(&reader, Duration::from_secs(5)) {
                Some(Err(ReaderExit::Fatal(msg))) => {
                    assert!(
                        msg.contains("panic"),
                        "round {round}: Fatal message should mention the panic: {msg}"
                    );
                }
                Some(Err(ReaderExit::Eof)) => {
                    panic!(
                        "round {round}: prefetch thread panic was downgraded to Eof (silent data truncation)"
                    );
                }
                Some(Ok(Some(_))) => panic!("round {round}: panic pill did not fire"),
                Some(Ok(None)) => panic!("round {round}: unexpected Ok(None) (timeout semantics)"),
                None => panic!("round {round}: no result within 5s (consumer would hang forever)"),
            }
            // Fatal 幂等：通道已断开，再次读取仍应是 Fatal（不会翻转为 Eof）
            match reader.try_next() {
                Err(ReaderExit::Fatal(msg2)) => {
                    assert!(
                        msg2.contains("panic"),
                        "round {round}: second read lost the panic message: {msg2}"
                    );
                }
                Err(ReaderExit::Eof) => {
                    panic!(
                        "round {round}: second try_next flipped to Eof (Fatal must be idempotent)"
                    );
                }
                Ok(_) => panic!("round {round}: second try_next unexpectedly returned data"),
            }
        }
    }

    /// 回归「静默重试掩盖 IO 错误」：截断的 shard 必须让 try_next 返回
    /// `Err(ReaderExit::Fatal(..))`，而不是永远 `Ok(None)` 无限重试
    /// （旧行为：消费端一直等不到数据、也不报错）。
    ///
    /// 同时锁定"重试有界"：本地源允许 `LOCAL_READ_RETRIES` 次重试，耗尽必须上报，
    /// 且总耗时应在退避量级（不是无限重试）。
    #[test]
    fn test_truncated_shard_surfaces_fatal_error_instead_of_silent_retry() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        // size_bytes=0 的 shard：模拟 bit rot / 传输截断
        let dir = make_local_dataset("fatal", 2, batch_size, max_seq_len, 0);
        let reader = open_reader(&dir, batch_size, max_seq_len);

        let started = std::time::Instant::now();
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Err(ReaderExit::Fatal(msg))) => {
                assert!(
                    msg.contains("shard_00000.bin"),
                    "error message should pinpoint the specific shard: {msg}"
                );
            }
            Some(Err(ReaderExit::Eof)) => panic!("IO error was treated as normal end (Eof)"),
            Some(Ok(Some(_))) => panic!("truncated shard should not produce a batch"),
            Some(Ok(None)) => panic!("should not return Ok(None) (semantics: timeout)"),
            None => panic!(
                "IO error was silently retried: no data and no error within 10s (consumer would hang forever)"
            ),
        }
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "error took {:?}, too long (unbounded retries suspected)",
            started.elapsed()
        );

        // 错误幂等上报：线程退出后重复调用仍是 Fatal，不会退化成 Eof
        assert!(
            matches!(reader.try_next(), Err(ReaderExit::Fatal(_))),
            "error should be re-readable"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 本地读有**有限次重试**：瞬时故障（半截文件 / 瞬时 EIO）应能自愈。
    ///
    /// 时序设计（确定性）：t≈0 首读失败（0 字节）→ t≈50ms 重试仍失败 →
    /// t≈120ms 测试修好文件 → t≈250ms 最后一次重试成功。
    /// 若重试次数为 0/1，此处必然拿到 Fatal（红）。
    #[test]
    fn test_local_read_retries_transient_failure_then_heals() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let batch_bytes = batch_size * max_seq_len * 4;
        let dir = make_local_dataset("heal", 2, batch_size, max_seq_len, 0); // 初始 0 字节
        let reader = open_reader(&dir, batch_size, max_seq_len);

        // 介于第 1 次（~50ms）与第 2 次（~250ms）重试之间修复文件
        std::thread::sleep(Duration::from_millis(120));
        std::fs::write(dir.join("shard_00000.bin"), vec![0x07u8; batch_bytes * 2]).unwrap();

        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap { input_ids, .. }))) => {
                assert_eq!(input_ids.len(), batch_size * max_seq_len);
                assert!(
                    input_ids.iter().all(|&v| v == 0x0707_0707),
                    "should read the repaired content (little-endian 0x07 fill)"
                );
            }
            _ => {
                panic!("transient failure should heal via bounded retries (retries not in effect)")
            }
        }
        // 重试后 reader 状态正常：第 2 个 batch 仍可读
        assert!(
            matches!(
                next_within(&reader, Duration::from_secs(10)),
                Some(Ok(Some(BatchData::Memmap { .. })))
            ),
            "should be able to read subsequent batches after self-healing"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **R1**：本轮读完后 `try_next` 立即**幂等**返回 `Err(Eof)`（而不是超时自旋），
    /// 且线程保持存活：Seek 回到本轮起点后仍能继续读（保证续训回退不被"读完就退出"破坏）。
    #[test]
    fn test_eof_signal_is_idempotent_and_seek_revives_reader() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let total_bytes = batch_size * max_seq_len * 4 * 2; // 2 个 batch
        let dir = make_local_dataset("eof", 2, batch_size, max_seq_len, total_bytes);
        let mut reader = open_reader(&dir, batch_size, max_seq_len);

        assert_eq!(
            reader.epoch_total_batches(0),
            2,
            "epoch 0 should have only 2 batches"
        );

        for i in 0..2 {
            match next_within(&reader, Duration::from_secs(10)) {
                Some(Ok(Some(BatchData::Memmap { .. }))) => {}
                Some(Err(ReaderExit::Fatal(msg))) => {
                    panic!("batch {i} should not raise a fatal error: {msg}")
                }
                Some(Err(ReaderExit::Eof)) => panic!("batch {i} should not EOF"),
                _ => panic!("batch {i} should be read normally"),
            }
        }

        // 读尽后：立即且幂等地拿到 Eof（B2 的挂死正是缺了这条）
        for _ in 0..3 {
            assert!(
                matches!(reader.try_next(), Err(ReaderExit::Eof)),
                "should return Err(Eof) immediately after the epoch is exhausted (idempotent), not spin on timeout"
            );
        }
        assert!(
            reader.at_eof(),
            "at_eof flag should be readable (for Mixer to detect sub-stream exhaustion)"
        );

        // Seek 回本轮起点 → 清除 EOF 标志 → 仍能继续读（线程必须保持存活）
        reader
            .seek_to_epoch_batch(0, 0)
            .expect("seek command should be sent successfully");
        assert!(
            !reader.at_eof(),
            "Seek must clear at_eof (otherwise pseudo Eof)"
        );
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap { .. }))) => {}
            Some(Err(ReaderExit::Eof)) => panic!(
                "after Seek, Eof not cleared or data not arrived yet (should not EOF immediately)"
            ),
            Some(Err(ReaderExit::Fatal(msg))) => {
                panic!("Seek should not raise a fatal error: {msg}")
            }
            _ => {
                panic!("after Seek back to start, reading should continue (thread must stay alive)")
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **R2**：`seek_to_epoch_batch` 的越界校验与"合法中位寻址后首个 try_next 不得返回 Eof"。
    #[test]
    fn test_seek_to_epoch_batch_bounds_and_mid_epoch_positioning() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let total_bytes = batch_size * max_seq_len * 4 * 4; // 4 个 batch
        let dir = make_local_dataset("seek", 4, batch_size, max_seq_len, total_bytes);
        let mut reader = open_reader(&dir, batch_size, max_seq_len);

        // 越界：in_epoch == epoch_total 即非法（游标是"下一产出位置"口径）
        assert!(
            reader.seek_to_epoch_batch(0, 4).is_err(),
            "in_epoch >= epoch_total(0)=4 should raise an error"
        );

        // 合法中位寻址：落在本轮第 2 个 batch，首个 try_next 必须给数据（不能给 Eof）
        reader
            .seek_to_epoch_batch(0, 2)
            .expect("mid-epoch seek should succeed");
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap { .. }))) => {}
            Some(Err(ReaderExit::Eof)) => {
                panic!("first try_next after mid-epoch seek must not return Eof (A5 acceptance)")
            }
            Some(Err(ReaderExit::Fatal(msg))) => {
                panic!("mid-epoch seek should not raise a fatal error: {msg}")
            }
            _ => panic!("first try_next after mid-epoch seek should return data"),
        }
        assert!(
            !matches!(reader.try_next(), Err(ReaderExit::Eof)),
            "only 1 batch remains after mid-epoch seek, should not EOF immediately"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// predownload 解析 + channel 容量联动（二轮 P2-3 回归锚点：
    /// `predownload > 50` 时配置不得被默认容量 50 静默截断）。
    #[test]
    fn predownload_resolution_and_channel_capacity_stay_linked() {
        assert_eq!(resolve_predownload(0, 2 * 4 * 4), DEFAULT_PREDOWNLOAD);
        // 过小值：保留原值（仅告警），不静默改写用户配置
        assert_eq!(resolve_predownload(3, 2 * 4 * 4), 3);
        assert_eq!(resolve_predownload(100, 2 * 4 * 4), 100);
        // 默认窗口 <= 默认容量：容量保持 50
        assert_eq!(data_channel_capacity(16), 50);
    }

    // ===============================================================
    // 评审 §3.1 / §3.6 的负向路径回归
    // ===============================================================

    /// ★ 评审 §3.1（P0）回归：`record_fatal_msg` 是 panic 兜底路径的**最后一道防线**，
    ///   它自己绝不能 panic。
    ///
    ///   失效链回顾：预取线程 panic → 未写 fatal 槽位 → `exit_reason()` 按 Eof 处理
    ///   → 绑定层转 `("exhausted", ...)` → **训练数据被静默截断**。
    ///   `io.rs::verify_local_file` 里那两处 `.lock().unwrap()` 正是触发源之一
    ///   （已改为毒化安全）。本测试锁定"槽位被毒化时 record_fatal_msg 仍不 panic"。
    #[test]
    fn record_fatal_msg_survives_poisoned_slot() {
        use std::sync::Arc;
        let slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // 正常路径：写入成功
        record_fatal_msg(&slot, "boom");
        assert_eq!(slot.lock().unwrap().as_deref(), Some("boom"));

        // 人为毒化：持锁线程 panic → std Mutex 进入 poisoned 状态
        let slot2: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let s2 = slot2.clone();
        let _ = std::thread::spawn(move || {
            let _g = s2.lock().unwrap();
            panic!("poison the mutex on purpose");
        })
        .join();
        assert!(
            slot2.lock().is_err(),
            "slot should be poisoned for this test to be meaningful"
        );

        // 关键断言：**不 panic**。能走到这里即通过。
        // 若 record_fatal_msg 内部用了 .lock().unwrap()，本测试会以 panic 失败——
        // 那正是评审 §3.1 要根除的"panic 套 panic"形态。
        record_fatal_msg(&slot2, "fatal while poisoned");
    }

    /// ★ 评审 §3.6(b) 回归：`bytes_into_i32_vec` 对非 4 倍数长度必须返回 `None`，
    ///   而不是静默丢弃尾部 1-3 字节（旧实现是 `debug_assert`，release 下静默截断）。
    #[test]
    fn bytes_into_i32_vec_rejects_non_multiple_of_four() {
        // 合法长度：正常转换
        let ok = bytes_into_i32_vec(vec![0u8; 8]).expect("8 bytes = 2 i32 should convert");
        assert_eq!(ok.len(), 2);

        // 非法长度：必须 None（不 panic、不静默截断）
        for bad_len in [1usize, 2, 3, 5, 7] {
            assert!(
                bytes_into_i32_vec(vec![0u8; bad_len]).is_none(),
                "length {bad_len} is not a multiple of 4 and must be rejected, not silently truncated"
            );
        }

        // 空 buffer 是合法的（0 % 4 == 0）
        assert!(bytes_into_i32_vec(Vec::new()).is_some());
    }

    /// ★ 评审 §3.6(b) 回归：`with_options` 入口的 `batch_size × max_seq_len`
    ///   上限校验——超大组合必须**在任何分配之前**就被拒绝。
    #[test]
    fn oversized_batch_shape_is_rejected_at_construction() {
        let dir = std::env::temp_dir().join(format!(
            "megadata_oversized_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (bs, msl) = (2usize, 4usize);
        let mut manifest = crate::builder::metadata::DatasetManifest::new(
            crate::builder::metadata::StorageFormat::Memmap,
            msl,
        );
        manifest.global_stats.batch_size = bs;
        manifest.shards.push(crate::builder::metadata::ShardMeta {
            shard_id: 0,
            filename: "shard_00000.bin".to_string(),
            num_batches: 1,
            size_bytes: (bs * msl * 4) as u64,
            hashes: Default::default(),
            align_aud_range: None,
        });
        manifest
            .save_to_file(dir.join("index.json").to_str().unwrap())
            .unwrap();
        // 正常形状的构造会预打开第一个 shard，文件必须真实存在
        std::fs::write(dir.join("shard_00000.bin"), vec![0u8; bs * msl * 4]).unwrap();
        let index = dir.join("index.json").to_str().unwrap().to_string();

        // 超大形状：1<<20 × 1<<20 = 2^40 远超上限，必须报错而不是尝试分配 4PiB
        let too_big = StreamReader::with_options(
            &index,
            1 << 20,
            1 << 20,
            1,
            0,
            42,
            StreamReaderOptions::default(),
        );
        assert!(
            too_big.is_err(),
            "oversized batch_size x max_seq_len must be rejected"
        );
        let msg = too_big.err().unwrap().to_string();
        assert!(
            msg.contains("exceeds the per-batch limit") || msg.contains("overflows"),
            "error must be actionable, got: {msg}"
        );

        // 正常形状仍然可以构造（确认上限没有误伤）
        let ok =
            StreamReader::with_options(&index, bs, msl, 1, 0, 42, StreamReaderOptions::default());
        assert!(
            ok.is_ok(),
            "normal shape must still construct: {:?}",
            ok.err()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 文档级旁路列（列扩展）读取：
    /// - manifest 有 schema key ⇒ 列文件存在时透出 `Some(...)`
    /// - 无 schema key（旧数据集）⇒ 透出 `None`，且**不产生任何错误/警告**
    #[test]
    fn pt_reader_exposes_doc_columns_and_degrades_without_them() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let capacity = batch_size * max_seq_len;
        let total_bytes = capacity * 4 * 4; // 4 个 batch

        // 1. 带列数据集
        let dir =
            make_local_dataset_with_columns("cols", 4, batch_size, max_seq_len, total_bytes, true);
        let reader = open_reader(&dir, batch_size, max_seq_len);
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap {
                doc_ids,
                loss_weights,
                ..
            }))) => {
                let doc_ids = doc_ids.expect("schema 声明了 doc_ids ⇒ 必须透出该列");
                assert_eq!(doc_ids.len(), capacity);
                assert!(
                    doc_ids.iter().all(|&d| d == 0),
                    "首个 batch 的 doc_ids 全为 0: {doc_ids:?}"
                );
                let weights = loss_weights.expect("schema 声明了 loss_weights ⇒ 必须透出该列");
                assert_eq!(weights.len(), capacity * 2);
                assert!(
                    weights.chunks(2).all(|w| w == [0x00, 0x3C]),
                    "权重应恒为 f16 1.0"
                );
            }
            _ => panic!("expected a Memmap batch with columns"),
        }
        std::fs::remove_dir_all(&dir).ok();

        // 2. 无列数据集（旧格式）：必须降级为 None，不能报错
        let dir2 = make_local_dataset("nocols", 4, batch_size, max_seq_len, total_bytes);
        let reader2 = open_reader(&dir2, batch_size, max_seq_len);
        match next_within(&reader2, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap {
                doc_ids,
                loss_weights,
                images,
                ..
            }))) => {
                assert!(doc_ids.is_none(), "无 schema key 的数据集不应透出 doc_ids");
                assert!(
                    loss_weights.is_none(),
                    "无 schema key 的数据集不应透出 loss_weights"
                );
                assert!(
                    images.is_none(),
                    "无 mm sidecar key 的数据集不应透出 images"
                );
            }
            _ => panic!("expected a Memmap batch"),
        }
        std::fs::remove_dir_all(&dir2).ok();
        // 窗口 > 默认容量：容量必须随之扩容（否则节流永不触发）
        assert_eq!(data_channel_capacity(100), 100);
        assert_eq!(data_channel_capacity(DEFAULT_PREDOWNLOAD), 50);
    }

    /// P1-T4b 主路径：schema 三 key 齐全 ⇒ 每个 batch 透出 `images`（含空 batch
    /// 与 NO_IMAGE 占位），记录数 / 元数据 / bytes 与写入一致。
    #[test]
    fn mm_sidecar_stream_exposes_images_per_batch() {
        use crate::builder::sidecar::img_flags;
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let capacity = batch_size * max_seq_len;
        let total_bytes = capacity * 4 * 3; // 3 个 batch
        let (dir, imgs) = make_local_dataset_with_mm("mm", 3, batch_size, max_seq_len, total_bytes);
        let reader = open_reader(&dir, batch_size, max_seq_len);

        // batch 0：1 张 VALID，bytes / 元数据与写入一致
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap { images, .. }))) => {
                let images = images.expect("schema 三 key 齐全 ⇒ 必须透出 images");
                assert_eq!(images.len(), 1);
                assert_eq!(images[0].bytes.as_slice(), imgs[0].as_slice());
                assert_eq!(images[0].sample_id, 0);
                assert_eq!(images[0].seg_order, 0);
                assert_eq!(images[0].grid_h, 1);
                assert_eq!(images[0].grid_w, 1);
                assert_eq!(images[0].flags, img_flags::VALID);
            }
            _ => panic!("expected a Memmap batch with images"),
        }
        // batch 1：纯文本 → 空 list（不是 None、不是错误）
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap { images, .. }))) => {
                let images = images.expect("纯文本 batch 也要透出 images 键");
                assert!(
                    images.is_empty(),
                    "纯文本 batch 的 images 应为空 list: {images:?}"
                );
            }
            _ => panic!("expected batch 1"),
        }
        // batch 2：VALID + NO_IMAGE 占位（bytes == b""；sample_id 按
        // make_mm_dataset_with_payloads 的连续分配 0/1/2，跳号透出由
        // runtime::mm_sidecar 的 roundtrip 单测覆盖）
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap { images, .. }))) => {
                let images = images.expect("batch 2 必须透出 images");
                assert_eq!(images.len(), 2);
                assert_eq!(images[0].bytes.as_slice(), imgs[1].as_slice());
                assert_eq!(images[0].sample_id, 1);
                assert_eq!(images[1].flags, img_flags::NO_IMAGE);
                assert!(images[1].bytes.is_empty(), "NO_IMAGE 占位 bytes 必须为空");
                assert_eq!(images[1].sample_id, 2);
            }
            _ => panic!("expected batch 2"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// P1-T4b 负面：schema 只有一部分 mm sidecar key ⇒ manifest 自相矛盾，
    /// fail-fast（错误信息含 "inconsistent"），不是静默跳过。
    /// 构造期预打开首个 shard 即触发（`with_options` 直接报错）。
    #[test]
    fn mm_sidecar_partial_schema_keys_fail_fast_inconsistent() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let capacity = batch_size * max_seq_len;
        let total_bytes = capacity * 4 * 3;

        // 组合 1：只有 img_rec（缺 img_blob / mm_cu）
        let (dir, _) =
            make_local_dataset_with_mm("mmpartial", 3, batch_size, max_seq_len, total_bytes);
        rewrite_mm_schema(&dir, &["img_rec"]);
        let err = open_reader_expect_err(&dir, batch_size, max_seq_len);
        assert!(err.to_string().contains("inconsistent"), "{err}");
        std::fs::remove_dir_all(&dir).ok();

        // 组合 2：img_rec + img_blob 有、mm_cu 缺
        let (dir2, _) =
            make_local_dataset_with_mm("mmpartial2", 3, batch_size, max_seq_len, total_bytes);
        rewrite_mm_schema(&dir2, &["img_rec", "img_blob"]);
        let err2 = open_reader_expect_err(&dir2, batch_size, max_seq_len);
        assert!(err2.to_string().contains("inconsistent"), "{err2}");
        std::fs::remove_dir_all(&dir2).ok();
    }

    /// P1-T4b 负面：manifest 声明了 sidecar（key + hash 齐全）但文件被删 ⇒
    /// fail-fast（不静默降级为无图数据集）。
    #[test]
    fn mm_sidecar_missing_file_fails_fast_not_silent() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let capacity = batch_size * max_seq_len;
        let total_bytes = capacity * 4 * 3;
        let (dir, _) =
            make_local_dataset_with_mm("mmmiss", 3, batch_size, max_seq_len, total_bytes);
        // 删 blob（manifest 仍有 key + sha256:col_img_blob）
        std::fs::remove_file(dir.join("shard_00000.col_img_blob")).unwrap();
        let err = open_reader_expect_err(&dir, batch_size, max_seq_len);
        assert!(
            err.to_string().contains("col_img_blob"),
            "错误信息必须能定位到缺失文件: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();

        // 同理：删 mm_cu
        let (dir2, _) =
            make_local_dataset_with_mm("mmmiss2", 3, batch_size, max_seq_len, total_bytes);
        std::fs::remove_file(dir2.join("shard_00000.mm_cu")).unwrap();
        let err2 = open_reader_expect_err(&dir2, batch_size, max_seq_len);
        assert!(err2.to_string().contains("mm_cu"), "{err2}");
        std::fs::remove_dir_all(&dir2).ok();
    }

    /// P1-T4b 负面：sidecar 声明与 shard batch 数不符（mm_cu 只覆盖部分 batch）
    /// ⇒ 读超出部分 fail-fast（mm_cu 条目不足 → UnexpectedEof），不静默给空图。
    #[test]
    fn mm_sidecar_mm_cu_shorter_than_batches_fails_fast() {
        use crate::builder::sidecar::img_flags;
        use crate::builder::sidecar::{ImageRecordInput, MmSidecarWriter};
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let capacity = batch_size * max_seq_len;
        let total_bytes = capacity * 4 * 3;
        let (dir, _) =
            make_local_dataset_with_mm("mmshort", 3, batch_size, max_seq_len, total_bytes);
        // 重写 mm_cu：只保留 1 条（batch 0）
        let mut w = MmSidecarWriter::create(&dir, "shard_00000").unwrap();
        w.append_mm_batch(
            &[ImageRecordInput {
                bytes: Some(&[1u8; 4]),
                grid_h: 1,
                grid_w: 1,
                flags: img_flags::VALID,
                sample_id: 0,
                seg_order: 0,
            }],
            0,
        )
        .unwrap();
        let _ = w.finish().unwrap();
        let reader = open_reader(&dir, batch_size, max_seq_len);
        // batch 0 正常
        assert!(matches!(
            next_within(&reader, Duration::from_secs(10)),
            Some(Ok(Some(BatchData::Memmap {
                images: Some(_),
                ..
            })))
        ));
        // batch 1：mm_cu 无此条目 → Fatal（含 batch 定位），不得静默当 EOF / 空图
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Err(ReaderExit::Fatal(msg))) => {
                assert!(msg.contains("batch 1"), "错误应含 batch 定位: {msg}");
            }
            _ => panic!(
                "batch 1 must fail fast with Fatal (mm_cu has no entry), not EOF/empty images"
            ),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // ==================== P1-T5（D8）预取字节预算 ====================

    /// 等待预取在途账本稳定（预算门阻塞 / EOF park 后不再增长）：
    /// 轮询 `probe`，连续 `stable_ms` 无变化或到达 `timeout` 即返回当前值。
    fn wait_ledger_stable<F: Fn() -> u64>(probe: F, timeout: Duration, stable_ms: u128) -> u64 {
        let deadline = std::time::Instant::now() + timeout;
        let mut last = probe();
        let mut since_change = std::time::Instant::now();
        while std::time::Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
            let cur = probe();
            if cur != last {
                last = cur;
                since_change = std::time::Instant::now();
            } else if since_change.elapsed().as_millis() >= stable_ms {
                return cur;
            }
        }
        last
    }

    /// D8 主验收（大图压力测试）：单 batch blob 2MB × 窗口深度 > 预算 ⇒
    /// 实测窗口峰值（在途账本）≤ 预算；无预算对照下同数据集能预取完全部
    /// batch（证明限制确实来自预算而非数据缺失）。消费完后账本归零（守恒）。
    #[test]
    fn prefetch_byte_budget_caps_window_peak() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let nbatches = 6usize;
        let main_bytes = batch_size * max_seq_len * 4; // 32B 主列
        let img_bytes = 1024 * 1024; // 1MB/图，2 图/batch
        let per_batch: Vec<Vec<Option<Vec<u8>>>> =
            vec![vec![Some(vec![9u8; img_bytes]); 2]; nbatches];
        let batch_cost = (img_bytes * 2 + main_bytes) as u64; // 裸流：compressed=0 + raw + blob
        let limit = batch_cost * 5 / 2; // 2.5 batch ≈ 5.25MB：窗口最多 2 batch 在途

        // 1. 预算开启：窗口峰值被预算封顶
        let dir = make_mm_dataset_with_payloads(
            "mmbudget",
            nbatches,
            batch_size,
            max_seq_len,
            main_bytes * nbatches,
            &per_batch,
        );
        let reader = StreamReader::with_options(
            dir.join("index.json").to_str().unwrap(),
            batch_size,
            max_seq_len,
            1,
            0,
            42,
            StreamReaderOptions {
                predownload: 8,
                prefetch_bytes_limit: limit,
                ..Default::default()
            },
        )
        .unwrap();
        let peak = wait_ledger_stable(|| reader.in_flight_bytes(), Duration::from_secs(5), 800);
        assert!(
            peak > 0 && peak >= batch_cost,
            "预算门不应阻断所有预取：账本 {peak} < 单 batch {batch_cost}"
        );
        assert!(
            peak <= limit,
            "窗口峰值 {peak} 必须被封在预算 {limit} 内（无预算时全窗口 ≈ {}）",
            batch_cost * nbatches as u64
        );
        // 消费回归：6 个 batch 全部可读、内容正确，读完后账本归零
        for b in 0..nbatches {
            match next_within(&reader, Duration::from_secs(10)) {
                Some(Ok(Some(BatchData::Memmap {
                    images: Some(imgs), ..
                }))) => {
                    assert_eq!(imgs.len(), 2);
                    assert_eq!(imgs[0].bytes.len(), img_bytes, "batch {b} blob 完整");
                    assert_eq!(imgs[1].bytes.len(), img_bytes, "batch {b} blob 完整");
                }
                _ => panic!("expected Memmap batch {b} with 2 images"),
            }
        }
        assert!(matches!(
            next_within(&reader, Duration::from_secs(10)),
            Some(Err(ReaderExit::Eof))
        ));
        assert_eq!(reader.in_flight_bytes(), 0, "消费完必须账本归零（守恒）");
        drop(reader);
        std::fs::remove_dir_all(&dir).ok();

        // 2. 无预算对照（limit=0 = 历史行为）：同数据集全部 6 batch 预取完
        let dir2 = make_mm_dataset_with_payloads(
            "mmbudget0",
            nbatches,
            batch_size,
            max_seq_len,
            main_bytes * nbatches,
            &per_batch,
        );
        let reader2 = StreamReader::with_options(
            dir2.join("index.json").to_str().unwrap(),
            batch_size,
            max_seq_len,
            1,
            0,
            42,
            StreamReaderOptions {
                predownload: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let peak2 = wait_ledger_stable(|| reader2.in_flight_bytes(), Duration::from_secs(5), 800);
        assert_eq!(
            peak2,
            batch_cost * nbatches as u64,
            "预算不限时全部 batch 应预取完（EOF park 前账本 = 全窗口）"
        );
        std::fs::remove_dir_all(&dir2).ok();
    }

    /// D8 取消路径：预算门阻塞中 / 通道有在途 batch 时取消，前台把通道内
    /// 剩余 batch 消干（try_next 内部逐个归还账本）后账本归零（无泄漏）。
    /// 既有 cancel 语义（prefetch 线程退出、try_next 返回 Eof）不回归。
    #[test]
    fn prefetch_byte_budget_cancel_returns_ledger_to_zero() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let nbatches = 4usize;
        let main_bytes = batch_size * max_seq_len * 4;
        let img_bytes = 256 * 1024; // 256KB/图 → batch_cost ≈ 512KB
        let per_batch: Vec<Vec<Option<Vec<u8>>>> =
            vec![vec![Some(vec![9u8; img_bytes]); 2]; nbatches];
        let batch_cost = (img_bytes * 2 + main_bytes) as u64;
        let limit = batch_cost + batch_cost / 2; // 1.5 batch：窗口只能放 1 个
        let dir = make_mm_dataset_with_payloads(
            "mmbudgetcancel",
            nbatches,
            batch_size,
            max_seq_len,
            main_bytes * nbatches,
            &per_batch,
        );
        let reader = StreamReader::with_options(
            dir.join("index.json").to_str().unwrap(),
            batch_size,
            max_seq_len,
            1,
            0,
            42,
            StreamReaderOptions {
                predownload: 8,
                prefetch_bytes_limit: limit,
                ..Default::default()
            },
        )
        .unwrap();
        // 预取线程在预算门处阻塞（第 2 个 batch 放不进来）
        let peak = wait_ledger_stable(|| reader.in_flight_bytes(), Duration::from_secs(5), 800);
        assert!(
            peak <= limit && peak > 0,
            "取消前账本应在预算内且非空: {peak}"
        );
        reader.cancel();
        // 前台消费干通道（每个取走的 batch 在 try_next 内归还），直至断开 → Eof
        while reader.try_next().is_ok() {}
        assert_eq!(
            reader.in_flight_bytes(),
            0,
            "取消 + 消费干通道后账本必须归零（无泄漏）"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// D8 预算不限（0，默认）= 历史行为：小数据集正常迭代，账本随消费归零。
    /// 全量回归（Rust + pytest）叠加保证零行为差异。
    #[test]
    fn prefetch_byte_limit_zero_keeps_ledger_conserved() {
        let (batch_size, max_seq_len) = (2usize, 4usize);
        let capacity = batch_size * max_seq_len;
        let total_bytes = capacity * 4 * 3;
        let (dir, imgs) =
            make_local_dataset_with_mm("mmzero", 3, batch_size, max_seq_len, total_bytes);
        let imgs0_snapshot = imgs[0].clone();
        let imgs1_snapshot = imgs[1].clone();
        let reader = open_reader(&dir, batch_size, max_seq_len); // 默认 options：limit=0
        // batch 0：imgs[0]；batch 1：纯文本（空 images）；batch 2：imgs[1] + 占位
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap {
                images: Some(imgs), ..
            }))) => {
                assert_eq!(imgs.len(), 1);
                assert_eq!(imgs[0].bytes.as_slice(), imgs0_snapshot.as_slice());
            }
            _ => panic!("expected Memmap batch 0"),
        }
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap {
                images: Some(imgs), ..
            }))) => {
                assert!(imgs.is_empty(), "batch 1 纯文本");
            }
            _ => panic!("expected Memmap batch 1"),
        }
        match next_within(&reader, Duration::from_secs(10)) {
            Some(Ok(Some(BatchData::Memmap {
                images: Some(imgs), ..
            }))) => {
                assert_eq!(imgs.len(), 2);
                assert_eq!(imgs[0].bytes.as_slice(), imgs1_snapshot.as_slice());
                assert!(imgs[1].bytes.is_empty());
            }
            _ => panic!("expected Memmap batch 2"),
        }
        assert!(matches!(
            next_within(&reader, Duration::from_secs(10)),
            Some(Err(ReaderExit::Eof))
        ));
        assert_eq!(
            reader.in_flight_bytes(),
            0,
            "limit=0 也必须守恒（取走归还）"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

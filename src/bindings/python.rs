use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use arrow::pyarrow::ToPyArrow;
use bbpe_rs::tokenizer::BBPETokenizer;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::builder::encoded_sample::IdRange;
use crate::builder::metadata::DatasetManifest;
use crate::builder::pipeline::{
    BuildProgress, PyBatchEncoder, PyModalityEncoder, VlmBuildConfig, run_offline_builder_pipeline,
};
use crate::builder::sample::{FailurePolicy, MmSampleSink, SampleIdAllocator, SegmentKind};
use crate::builder::source::{ChunkLimit, ReadSamplesOptions, SourceErrorCounts, read_samples};
use crate::cancel::CancellationToken;
use crate::packing::modality_aware::BudgetPolicy;
use crate::runtime::align::AlignEntry;
use crate::runtime::ffi_dlpack::{bytes_to_dlpack_capsule, vec_to_dlpack_capsule};
use crate::runtime::mixer::{MixerState, StreamMixer};
use crate::runtime::mm_sidecar::{MmImageEntry, MmVideoEntry};
use crate::runtime::sampler::{DistributedSampler, PartitionAlgo, ShardOrder, World};
use crate::runtime::stream::{BatchData, ReaderExit, StreamReader, StreamReaderOptions};
use crate::runtime::video_blob::VideoBlobPin;

/// P1-2：batch 持有的 video blob 句柄 pin（Python 对象生命周期承载）。
///
/// `BatchData.video_pin` 若只绑定到 Rust 局部变量，会随绑定函数返回立即 Drop——
/// 未消费 batch 的 blob 仍会被提前关闭（旧实现复现：多 shard 预取后
/// `read_video_blob` 拿到 NotFound）。本对象挂在 batch dict 的 `_video_pin`
/// 私有键上：dict 存活期 = pin 存活期（Python GC / 丢弃引用时归还）。
#[pyclass(name = "_VideoBlobPin", module = "megadata_rs")]
pub struct PyVideoBlobPin {
    _pin: VideoBlobPin,
}

/// 把 video blob pin 挂到 batch dict（`None` = 无 video sidecar，不设 key）。
fn set_video_pin(py: Python, dict: &Bound<'_, PyDict>, pin: Option<VideoBlobPin>) -> PyResult<()> {
    if let Some(pin) = pin {
        dict.set_item("_video_pin", Py::new(py, PyVideoBlobPin { _pin: pin })?)?;
    }
    Ok(())
}

/// 把文档级旁路列（列扩展）挂到 batch dict 上：`None` ⇒ 不设该 key（消费端按缺失处理）。
///
/// - `doc_ids`：i32（`DLDataType { code: 0, bits: 32 }`），形状 `[batch_size, max_seq_len]`
/// - `loss_weights`：f16（`DLDataType { code: 2, bits: 16 }`），形状同上，字节流原样透出
fn set_doc_columns(
    py: Python,
    dict: &Bound<'_, PyDict>,
    doc_ids: Option<Vec<i32>>,
    loss_weights: Option<Vec<u8>>,
    batch_size: usize,
    max_seq_len: usize,
) -> PyResult<()> {
    use crate::runtime::ffi_dlpack::DLDataType;

    if let Some(doc_ids) = doc_ids {
        let capsule = vec_to_dlpack_capsule(py, doc_ids, batch_size, max_seq_len)?;
        dict.set_item("doc_ids", capsule)?;
    }
    if let Some(weights) = loss_weights {
        let capsule = bytes_to_dlpack_capsule(
            py,
            weights,
            DLDataType {
                code: 2,
                bits: 16,
                lanes: 1,
            }, // 2 = Float
            vec![batch_size as i64, max_seq_len as i64],
            vec![max_seq_len as i64, 1],
        )?;
        dict.set_item("loss_weights", capsule)?;
    }
    Ok(())
}

/// 把多模态图像 sidecar（P1-T4b）挂到 batch dict 的 `"images"` 键：
/// `list[dict]`，每项 `{"bytes", "sample_id", "seg_order", "grid_h", "grid_w",
/// "flags"}`。占位记录（NO_IMAGE/EMPTY_IMAGE/ERRORED）的 `bytes` 为**零长
/// capsule**，消费端按 `flags` 判别（格式冻结 §3）。
///
/// **P1-T6（D8）**：`bytes` 为 DLPack capsule（u8 一维，torch 可
/// `from_dlpack`）——Owned 形态 move 转交、Shared 形态零拷贝 view（只增
/// Arc 引用计数：驱逐保护 + 多 view 单次释放），全链路无深拷贝。
fn set_mm_images(py: Python, dict: &Bound<'_, PyDict>, images: Vec<MmImageEntry>) -> PyResult<()> {
    use crate::runtime::ffi_dlpack::{
        DLDataType, bytes_to_dlpack_capsule, shared_bytes_view_to_dlpack_capsule,
    };
    use crate::runtime::mm_sidecar::BlobBytes;
    use pyo3::types::PyList;
    const U8_DTYPE: DLDataType = DLDataType {
        code: 1,
        bits: 8,
        lanes: 1,
    }; // 1=UInt
    let list = PyList::empty(py);
    for e in images {
        let d = PyDict::new(py);
        let capsule = match e.bytes {
            BlobBytes::Owned(v) => {
                let n = v.len() as i64;
                bytes_to_dlpack_capsule(py, v, U8_DTYPE, vec![n], vec![1])?
            }
            BlobBytes::Shared(buf, off, len) => {
                shared_bytes_view_to_dlpack_capsule(py, buf, off as usize, len as usize, U8_DTYPE)?
            }
        };
        d.set_item("bytes", capsule)?;
        d.set_item("sample_id", e.sample_id)?;
        d.set_item("seg_order", e.seg_order)?;
        d.set_item("grid_h", e.grid_h)?;
        d.set_item("grid_w", e.grid_w)?;
        d.set_item("flags", e.flags)?;
        list.append(d)?;
    }
    dict.set_item("images", list)?;
    Ok(())
}

/// P3b：video 记录 → batch dict `videos` 键（**字节延迟**——条目只带 rec 元数据
/// + `blob_ref = (store_id, offset, len)`，blob 字节由 collate 时点经
///   [`read_vid_blob`] FFI 按需读；D8 "video blob 不进预取窗口"的消费侧形态）。
///
/// `blob_ref` 是纯数据（跨 FFI 无所有权问题）；读取端无需跨 rec/跨 shard 二次
/// 查找（`offset/len` 自足定位）。
fn set_mm_videos(py: Python, dict: &Bound<'_, PyDict>, videos: Vec<MmVideoEntry>) -> PyResult<()> {
    use pyo3::types::PyList;
    let list = PyList::empty(py);
    for e in videos {
        let d = PyDict::new(py);
        d.set_item("sample_id", e.sample_id)?;
        d.set_item("seg_order", e.seg_order)?;
        d.set_item("n_frames", e.n_frames)?;
        d.set_item("width", e.width)?;
        d.set_item("height", e.height)?;
        d.set_item("flags", e.flags)?;
        d.set_item("fps_milli", e.fps_milli)?;
        d.set_item("start_ms", e.start_ms)?;
        let (store_id, offset, len) = e.blob_ref;
        d.set_item("blob_ref", (store_id, offset, len))?;
        list.append(d)?;
    }
    dict.set_item("videos", list)?;
    Ok(())
}

/// P3c：对齐视图 → batch dict `av_align` 键（每窗口一条；缺省不出现 =
/// 旧 trainer 零影响，DP6）。
///
/// 条目字段（终稿 §6.5 形态契约 + P1-1/P3-3 评审修订）：
/// `sample_id / window_index / video_rec_range（Vlm 流内 rec 下标列表）/
/// audio_rec_range（AudioRecRef 列表 (aud_batch, idx)）/ text_token_range
/// （Vlm 流内 token **区间列表**；空列表 = 无文本）`。
///
/// **契约变更（P1-1/P3-3）**：`text_token_range` 由"单个元组或 None"改为
/// **区间列表**——① 区间末端在首个 image/audio/video placeholder 处截断
/// （旧实现把视觉 token 当文本）；② text → 视觉段 → text 交错样本在同一窗内
/// 可有多段，全部保留（旧实现静默只保留第一段）。空列表保持"无文本"语义
/// （等价于旧的 None）。
fn set_av_align(py: Python, dict: &Bound<'_, PyDict>, entries: Vec<AlignEntry>) -> PyResult<()> {
    use pyo3::types::PyList;
    let list = PyList::empty(py);
    for e in entries {
        let d = PyDict::new(py);
        d.set_item("sample_id", e.sample_id)?;
        d.set_item("window_index", e.window_index)?;
        d.set_item("video_rec_range", e.video_rec_range)?;
        let audio: Vec<(u32, u32)> = e
            .audio_rec_range
            .iter()
            .map(|r| (r.aud_batch, r.idx))
            .collect();
        d.set_item("audio_rec_range", audio)?;
        d.set_item("text_token_range", e.text_token_range)?;
        list.append(d)?;
    }
    dict.set_item("av_align", list)?;
    Ok(())
}

// ==========================================
// 1. 离线构建器绑定
// ==========================================
// 说明：cancel_token / progress 用 Mutex 包成内部可变性，使 build_dataset 可以只取
// &self。这样 build_dataset 运行期间，其它线程（或 progress_cb 重入）调 cancel() /
// get_progress() 都是 &self 共享借用，PyO3 不再报 "Already mutably borrowed"。
/// bbpe 分词器的加载结果缓存（`probe_bbpe` / `probe_special_id` / `build_dataset`
/// 共用**一次**加载；失败原因也缓存，避免每次调用都重解一遍文件）。
struct BbpeCache {
    loaded: Option<Arc<BBPETokenizer>>,
    error: Option<String>,
}

#[pyclass(name = "DataBuilder", module = "megadata_rs")]
pub struct PyDataBuilder {
    tokenizer_path: String,
    max_seq_len: usize,
    cancel_token: Mutex<CancellationToken>,
    progress: Mutex<Option<Arc<BuildProgress>>>,
    bbpe: Mutex<Option<BbpeCache>>,
}

impl PyDataBuilder {
    /// 目录 → 目录下的 `tokenizer.json`（与 pipeline 内同样的解析规则）。
    fn tokenizer_file(&self) -> Option<String> {
        if self.tokenizer_path.is_empty() {
            return None;
        }
        Some(crate::builder::pipeline::resolve_tokenizer_file(
            &self.tokenizer_path,
        ))
    }

    /// 取（并缓存）bbpe 分词器。`tokenizer_path` 为空时返回 `Ok(None)`。
    fn load_bbpe(&self) -> Result<Option<Arc<BBPETokenizer>>, String> {
        let file = match self.tokenizer_file() {
            Some(f) => f,
            None => return Ok(None),
        };
        let mut guard = self.bbpe.lock().unwrap();
        if let Some(c) = &*guard {
            return match &c.error {
                Some(e) => Err(e.clone()),
                None => Ok(c.loaded.clone()),
            };
        }
        let cache = match BBPETokenizer::load(&file) {
            Ok(t) => BbpeCache {
                loaded: Some(Arc::new(t)),
                error: None,
            },
            Err(e) => BbpeCache {
                loaded: None,
                error: Some(format!("Failed to load tokenizer: {e}")),
            },
        };
        let out = match &cache.error {
            Some(e) => Err(e.clone()),
            None => Ok(cache.loaded.clone()),
        };
        *guard = Some(cache);
        out
    }
}

#[pymethods]
impl PyDataBuilder {
    #[new]
    pub fn new(tokenizer_path: String, max_seq_len: usize) -> Self {
        Self {
            tokenizer_path,
            max_seq_len,
            cancel_token: Mutex::new(CancellationToken::new()),
            progress: Mutex::new(None),
            bbpe: Mutex::new(None),
        }
    }

    /// 请求取消正在进行的构建。
    pub fn cancel(&self) {
        self.cancel_token.lock().unwrap().cancel();
    }

    /// 取回当前构建进度快照（dict）。
    pub fn get_progress(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        if let Some(p) = self.progress.lock().unwrap().as_ref() {
            d.set_item(
                "bytes_read",
                p.bytes_read.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "docs_read",
                p.docs_read.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "docs_tokenized",
                p.docs_tokenized.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "shards_written",
                p.shards_written.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "bytes_written",
                p.bytes_written.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            // F2：SFT 截断统计；F7：疑似 turn 标记注入计数
            d.set_item(
                "docs_truncated",
                p.docs_truncated.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "tokens_dropped_to_truncation",
                p.tokens_dropped_to_truncation
                    .load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "all_masked_docs",
                p.all_masked_docs.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "docs_flagged_injection",
                p.docs_flagged_injection
                    .load(std::sync::atomic::Ordering::Relaxed),
            )?;
            // F2-PT：被跨 batch 切分的文档数 / 这些文档的 token 总数
            // （PT 不截断，但长文档分段后 varlen 训练会丢跨 batch 上下文）
            d.set_item(
                "docs_split",
                p.docs_split.load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item(
                "tokens_in_split_docs",
                p.tokens_in_split_docs
                    .load(std::sync::atomic::Ordering::Relaxed),
            )?;
            d.set_item("elapsed_secs", p.start.elapsed().as_secs_f64())?;
        }
        Ok(d.into_any().unbind())
    }

    /// 判定 `tokenizer_path` 是否能被 bbpe_rs 加载（两级加载分派的第一级）。
    ///
    /// Python 侧据此决定走默认路径（Rust 直连）还是回退到第三方分词器。用**真实
    /// 的加载器**探测而非猜后缀：F4 已确认非 bbpe 文件会得到干净的失败信号
    /// （`InvalidModelFormat`），因此"探测成功"等价于"构建期加载也会成功"。
    pub fn probe_bbpe(&self, py: Python) -> PyResult<bool> {
        py.detach(|| Ok(self.load_bbpe().is_ok()))
    }

    /// 用 bbpe 分词器编码一段文本，返回其 id 序列（自动推导 eos / turn 标记 id 用）。
    ///
    /// 返回 `None` 表示"没有可用的 bbpe 分词器"（第三方分词器路径 / 文件不是 bbpe
    /// 模型）—— **注意与"编码结果不是单 token"区分开**：后者由调用方决定处置
    /// （turn 标记必须单 token → 报错；eos 只做分隔用 → 取首个并告警）。
    ///
    /// 有了它，bbpe 路径的 ID 自动推导不再依赖 `bbpe_tokenizer` 这个 Python 软依赖
    /// （方案 §3.5 的可选增强）：Rust 手上有真实分词器，推导必然与构建期一致。
    pub fn probe_special_ids(&self, py: Python, text: &str) -> PyResult<Option<Vec<i32>>> {
        py.detach(|| {
            // ★ 评审 §3.3：原本是 `self.load_bbpe().ok().flatten()`——把
            //   `load_bbpe`里**精心缓存的错误**（`Some(e) => Err(e.clone())`）整个
            //   丢掉，使路径写错 / InvalidModelFormat / OOM 全部退化成 `Ok(None)`，
            //   调用方无法区分"没有 bbpe 分词器"与"分词器加载失败"。
            //   现在保留错误语义：加载失败 → 抛；确实没有可用分词器 → None。
            let Some(tok) = self
                .load_bbpe()
                .map_err(pyo3::exceptions::PyIOError::new_err)?
            else {
                return Ok(None);
            };
            Ok(Some(tok.encode_with_types(text, &HashSet::new(), true).ids))
        })
    }

    /// `extra_columns`：是否落盘文档级旁路列 `doc_ids` / `loss_weights`
    /// （PT/Memmap 数据集；读取端按 manifest schema key 自动透出为 DLPack capsule）。
    /// 默认 false —— 零额外磁盘/内存开销；需要文档级 loss 加权或样本追溯时开启。
    #[pyo3(signature = (input_paths, output_dir, mode, template_str, batch_size, pad_id, eos_id, turn_start_id, turn_end_id, turn_start_str, turn_end_str, assistant_str, batches_per_shard=0, compression=None, progress_cb=None, batch_encoder=None, tokenizer_identity=None, tokenizer_vocab_size=None, extra_columns=false, modality_encoder=None, modality_id_ranges=None, max_visual_tokens_per_sample=None, image_label=-100, visual_budget_hard_fail=false, w_ms=2000, audio_window_token_rate=0, audio_max_seq_len=0, audio_batch_size=0, audio_batches_per_shard=0, audio_strict_monotonic=false, video_n_frames=16, video_resize=(336, 336), video_seed=0, video_fps_default=30.0, video_shard_max_bytes=536870912, field_map=None))]
    pub fn build_dataset(
        &self,
        py: Python,
        input_paths: Vec<String>,
        output_dir: String,
        mode: String,
        template_str: String,
        batch_size: usize,
        pad_id: i32,
        eos_id: i32,
        turn_start_id: i32,
        turn_end_id: i32,
        turn_start_str: String,
        turn_end_str: String,
        assistant_str: String,
        batches_per_shard: usize,
        compression: Option<String>,
        progress_cb: Option<Py<PyAny>>,
        batch_encoder: Option<Py<PyAny>>,
        tokenizer_identity: Option<String>,
        tokenizer_vocab_size: Option<u32>,
        extra_columns: bool,
        // ---- P2 (mm)：vlm 模式专用（pt/sft 不传 → None, None，行为零变更）----
        // 模态编码回调 `vlm_encode_callback(samples: list[dict]) -> list[dict]`
        modality_encoder: Option<Py<PyAny>>,
        // 各模态专属 token id 区间 `[(start, end, kind), ...]`（D6；vlm 必填）
        modality_id_ranges: Option<Vec<(i32, i32, String)>>,
        // D7 样本级总视觉预算；None ⇒ 取 max_seq_len（行宽即物理上限，不额外收紧）
        max_visual_tokens_per_sample: Option<u32>,
        // 图像段 label（T3，默认 -100）
        image_label: i32,
        // 视觉预算超限策略：false=TruncateAndCount（默认），true=HardFail
        visual_budget_hard_fail: bool,
        // ---- P3a (mm)：audio 流对齐/装箱配置（vlm 专用；pt/sft 不传即默认值）----
        // 对齐窗长 W_ms（整数毫秒；对齐终稿 DP1，默认 2000）
        w_ms: u32,
        // 音频窗内 token 率（tokens/s）；含 audio 段的构建必须显式声明（0 → fail-fast）
        audio_window_token_rate: u32,
        // audio 流 max_seq_len / batch_size / batches_per_shard（0 = 同 Vlm 流）
        audio_max_seq_len: usize,
        audio_batch_size: usize,
        audio_batches_per_shard: usize,
        // INV-2 软校验升级：start_ms 单调性倒退时 true=fail-fast（缺省 WARN）
        audio_strict_monotonic: bool,
        // ---- P3b (mm)：video 定帧采样确定性（入 align_fingerprint.video）----
        // 采样帧数 N（uniform 定帧；D9 定宽预算）
        video_n_frames: u16,
        // resize 目标分辨率 (w, h)（以指纹为准、rec 透传）
        video_resize: (u16, u16),
        // 采样器版本占位（uniform 策略不消费；seeded 随机采样预留）
        video_seed: u64,
        // fps 兜底值（MediaMeta.fps 缺失 → WARN + 此值 + 计数入 manifest）
        video_fps_default: f32,
        // Vlm 流 shard 字节均衡上限（0 = 不限；默认 512 MiB）
        video_shard_max_bytes: u64,
        // P4：源字段映射 dict（image_fields/text_fields/audio_fields/video_fields/
        // meta_fields/projection/timestamp_field）；None = 默认映射。仅 vlm 的
        // tar/parquet 源消费（JSONL 不读）。
        field_map: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let is_vlm = mode == "vlm";
        // P4-3（评审）：field_map 是 vlm 专属（tar/parquet 源映射）。其解析原本位于
        // 下方 `Some(cb)`（modality_encoder）分支内——若 pt/sft 传了 field_map 但未传
        // modality_encoder，会被静默忽略。此处在绑定层入口 loud-fail，与 builder.py
        // 的 vlm-only 校验对称（防直调 Rust 绑定者绕过 Python 层校验）。
        validate_field_map_scope(is_vlm, field_map.is_some())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        // 第三方分词器路径：把 Python callable 包成 PyBatchEncoder（模式烘焙进去）。
        // 有回调 ⇒ 不加载 bbpe 分词器，也不要求 tokenizer_path 指向真实文件。
        let encoder: Option<PyBatchEncoder> = batch_encoder.map(|cb| {
            PyBatchEncoder::new(
                cb,
                mode == "sft",
                tokenizer_identity.clone(),
                tokenizer_vocab_size,
            )
        });
        // P4：源字段映射（vlm 的 tar/parquet 源消费；pt/sft 不用默认值）
        let mut fm = crate::builder::source::FieldMap::default();
        // vlm 模式的编码工作全在 modality_encoder 回调（文本+视觉），不要求分词器；
        // 但回调本身缺失属于配置错误，binding 侧先显式报错（比 pipeline 晚到的
        // "missing tokenizer" 更准确）。
        if is_vlm && modality_encoder.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vlm mode requires modality_encoder: a Python callable implementing the \
                 samples -> encoded samples contract (see builder docs)",
            ));
        }
        if encoder.is_none() && modality_encoder.is_none() && self.tokenizer_path.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Missing tokenizer: provide tokenizer_path (bbpe model or HF tokenizer.json) \
                 or batch_encoder (third-party tokenizer callback)",
            ));
        }
        // bbpe 路径才加载分词器（复用缓存，构建期不再二次加载）。
        // ⚠️ 绝不能在 Python 路径也去 load：`tokenizer_path` 指向 HF tokenizer.json
        //    时它不是 bbpe 模型，会直接把"第三方分词器"这条路堵死。
        // vlm + modality_encoder：不加载 bbpe（golden e2e 可完全不依赖分词器）。
        let bbpe = match (&encoder, is_vlm) {
            (Some(_), _) => None,
            (None, true) => None,
            (None, false) => self
                .load_bbpe()
                .map_err(pyo3::exceptions::PyIOError::new_err)?,
        };

        // P2 (mm)：vlm 参数构造（spawn 前完成，配置错误提前抛给 Python）。
        // - vlm：D6 要求必须声明 id 区间（否则 token 校验无从谈起）；构造
        //   PyModalityEncoder（identity/vocab_size 供 manifest 指纹与 D6 文本 vocab 检查）
        //   + VlmBuildConfig。
        // - pt/sft 传了回调：照常构造并下传，由 pipeline 入口防护显式报错（防误用），
        //   binding 不静默吞掉；未传 → None, None 与之前完全一致。
        let (mm_encoder, vlm_cfg): (Option<PyModalityEncoder>, Option<VlmBuildConfig>) =
            match modality_encoder {
                None => (None, None),
                Some(cb) => {
                    let declared = if is_vlm {
                        Some(modality_id_ranges.ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(
                                "vlm mode requires modality_id_ranges: declare each modality's \
                             exclusive token id range as [(start, end, kind), ...] — without it \
                             the D6 token-range validation cannot be enforced",
                            )
                        })?)
                    } else {
                        modality_id_ranges
                    };
                    let id_ranges = declared
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(start, end, kind)| {
                            let kind = match kind.as_str() {
                                "text" => SegmentKind::Text,
                                "image" => SegmentKind::Image,
                                "audio" => SegmentKind::Audio,
                                "video" => SegmentKind::Video,
                                other => {
                                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                        "unknown segment kind {other:?} in modality_id_ranges \
                                 (expected text/image/audio/video)"
                                    )));
                                }
                            };
                            Ok(IdRange { start, end, kind })
                        })
                        .collect::<PyResult<Vec<IdRange>>>()?;
                    let me = PyModalityEncoder::new(cb, tokenizer_vocab_size, id_ranges);
                    // P4：field_map dict → FieldMap（缺失键保留默认；pt/sft 不用）
                    if let Some(d) = field_map {
                        fm = parse_field_map_dict(d)?;
                    }
                    let cfg = if is_vlm {
                        Some(VlmBuildConfig {
                            max_visual_tokens_per_sample: max_visual_tokens_per_sample
                                .unwrap_or(self.max_seq_len as u32),
                            image_label,
                            visual_budget_policy: if visual_budget_hard_fail {
                                BudgetPolicy::HardFail
                            } else {
                                BudgetPolicy::TruncateAndCount
                            },
                            // P3a：audio 流对齐/装箱配置（默认值 = 同 Vlm 流 / 不启用细项）
                            w_ms,
                            audio_window_token_rate,
                            audio_max_seq_len,
                            audio_batch_size,
                            audio_batches_per_shard,
                            audio_strict_monotonic,
                            // P3b：video 定帧采样确定性（入 align_fingerprint.video）
                            video_n_frames,
                            video_resize,
                            video_seed,
                            video_fps_default,
                            video_shard_max_bytes,
                        })
                    } else {
                        None
                    };
                    (Some(me), cfg)
                }
            };

        // 每次构建使用全新的令牌与进度（经内部可变性，无需 &mut self）
        let new_token = CancellationToken::new();
        let progress = BuildProgress::new();
        {
            let mut ct = self.cancel_token.lock().unwrap();
            *ct = new_token.clone();
            let mut pr = self.progress.lock().unwrap();
            *pr = Some(progress.clone());
        }

        let tokenizer_path = self.tokenizer_path.clone();
        let max_seq_len = self.max_seq_len;
        // cancel_token 用于本地轮询是否已取消；new_token 移入异步任务（两者同源，共享取消状态）
        let cancel_token = new_token.clone();
        // 读侧是唯一的 async 任务（tokio::fs 读 JSONL），写侧已是普通线程 → 单线程 runtime 够用
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to build Tokio runtime: {e}"))
            })?;

        let handle = rt.spawn(async move {
            run_offline_builder_pipeline(
                input_paths,
                output_dir,
                mode,
                &tokenizer_path,
                &template_str,
                max_seq_len,
                batch_size,
                pad_id,
                eos_id,
                turn_start_id,
                turn_end_id,
                turn_start_str,
                turn_end_str,
                assistant_str,
                batches_per_shard,
                compression,
                new_token,
                progress,
                encoder,
                bbpe,
                extra_columns,
                /*modality_encoder=*/ mm_encoder,
                /*vlm_config=*/ vlm_cfg,
                /*field_map=*/ fm,
            )
            .await
        });

        // ★ current_thread runtime 需要有人驱动调度器：用独立线程 block_on。
        //   不能只在主线程 `handle.is_finished()` 轮询——那样 spawn 出去的任务根本不会推进
        //   （主线程得以保持空闲来轮询进度 / 响应 Ctrl+C）。
        let worker = std::thread::spawn(move || rt.block_on(handle));

        let mut cancelled = false;
        loop {
            if worker.is_finished() {
                break;
            }

            // 轮询/回调出错：先请求取消并等构建线程收尾（避免留下后台写盘），再把错误抛出去
            let poll_result = (|| -> PyResult<()> {
                py.check_signals()?;
                if let Some(cb) = &progress_cb {
                    let dict = self.get_progress(py)?;
                    cb.call1(py, (dict,))?;
                }
                Ok(())
            })();
            if let Err(e) = poll_result {
                cancel_token.cancel();
                let _ = worker.join();
                return Err(e);
            }

            // 用本地克隆检查，避免每轮加锁（与 self.cancel_token 指向同一底层标志）
            if cancel_token.is_cancelled() {
                cancelled = true;
                break;
            }

            py.detach(|| {
                std::thread::sleep(std::time::Duration::from_millis(100));
            });
        }

        let join_result = worker
            .join()
            .map_err(|_| pyo3::exceptions::PyIOError::new_err("build worker thread panicked"))?
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("build task failed: {e}")))?;
        if cancelled || cancel_token.is_cancelled() {
            return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
                "Megadata build canceled by user",
            ));
        }
        join_result.map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(())
    }
}

// ==========================================
// 2. 单流读取器绑定
// ==========================================
#[pyclass(name = "RustStreamReader", module = "megadata_rs")]
pub struct PyRustStreamReader {
    inner: StreamReader,
    batch_size: usize,
    max_seq_len: usize,
}

#[pymethods]
impl PyRustStreamReader {
    #[new]
    #[pyo3(signature = (index_path, batch_size, max_seq_len, world_size, rank, seed,
                        num_workers=0, worker_id=0, shuffle=false,
                        partition="round_robin", shard_order="random",
                        num_canonical_nodes=0, predownload=0,
                        prefetch_bytes_limit=0, video_ondemand_bytes_limit=0, align=false))]
    pub fn new(
        py: Python,
        index_path: String,
        batch_size: usize,
        max_seq_len: usize,
        world_size: usize,
        rank: usize,
        seed: u64,
        num_workers: usize,
        worker_id: usize,
        shuffle: bool,
        partition: &str,
        shard_order: &str,
        num_canonical_nodes: usize,
        predownload: usize,
        prefetch_bytes_limit: u64,
        video_ondemand_bytes_limit: u64,
        align: bool,
    ) -> PyResult<Self> {
        let inner = py.detach(|| {
            let options = StreamReaderOptions {
                num_workers,
                worker_id,
                shuffle,
                partition: PartitionAlgo::parse(partition)?,
                shard_order: ShardOrder::parse(shard_order)?,
                num_canonical_nodes,
                predownload,
                // P1-T5（D8）：预取字节预算（0 = 不限 = 历史行为）；机制先在
                // Rust 侧落地，Python 训练侧按需透传。
                prefetch_bytes_limit,
                // P3b（D8）：video blob 按需读字节上限（0 = 不限）
                video_ondemand_bytes_limit,
                // P3c：三模态对齐视图请求
                align,
            };
            StreamReader::with_options(
                &index_path,
                batch_size,
                max_seq_len,
                world_size,
                rank,
                seed,
                options,
            )
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
        })?;
        Ok(Self {
            inner,
            batch_size,
            max_seq_len,
        })
    }

    /// 可中断的取 batch：周期性重获 GIL 以响应 Ctrl+C。
    ///
    /// 三态（R1 之后 `Eof` 才真正可达）：
    /// - `Ok(None)`：本轮读完（Rust `at_eof`）→ Python 侧 `StopIteration`；
    /// - `Err(OSError)`：致命 IO/解压/数据错误，必须上报；
    /// - `Ok(dict)`：拿到 batch。
    ///
    /// **C19-④**：拿到 `Eof` 后**再查一次 `is_cancelled()`**——否则 Ctrl+C 恰好落在
    /// EOF 置位之后会被静默转成 StopIteration（训练"正常结束"，正是 P6 禁止的静默）。
    pub fn next_batch_capsules<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        loop {
            py.check_signals()?;
            if self.inner.is_cancelled() {
                return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
                    "Megadata canceled by user",
                ));
            }
            let opt = py.detach(|| self.inner.try_next());
            match opt {
                Err(ReaderExit::Eof) => {
                    if self.inner.is_cancelled() {
                        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
                            "Megadata canceled by user",
                        ));
                    }
                    return Ok(None); // 本轮读完
                }
                Err(ReaderExit::Fatal(msg)) => {
                    return Err(pyo3::exceptions::PyIOError::new_err(format!(
                        "megadata stream fatal error (no more silent retries): {msg}"
                    )));
                }
                Ok(None) => continue, // 超时或陈旧，重试
                Ok(Some(data)) => {
                    let dict = PyDict::new(py);
                    match data {
                        BatchData::Arrow(arrow_batch) => {
                            dict.set_item("arrow_batch", arrow_batch.to_pyarrow(py)?)?;
                        }
                        BatchData::Memmap {
                            input_ids,
                            cu_seqlens,
                            doc_ids,
                            loss_weights,
                            images,
                            videos,
                            av_align,
                            video_pin,
                        } => {
                            let capsule = vec_to_dlpack_capsule(
                                py,
                                input_ids,
                                self.batch_size,
                                self.max_seq_len,
                            )?;
                            dict.set_item("input_ids", capsule)?;
                            if !cu_seqlens.is_empty() {
                                dict.set_item("cu_seqlens", cu_seqlens)?;
                            }
                            set_doc_columns(
                                py,
                                &dict,
                                doc_ids,
                                loss_weights,
                                self.batch_size,
                                self.max_seq_len,
                            )?;
                            if let Some(images) = images {
                                set_mm_images(py, &dict, images)?;
                            }
                            // P3b：video 记录（字节延迟；blob_ref 按需读）
                            if let Some(videos) = videos {
                                set_mm_videos(py, &dict, videos)?;
                            }
                            // P1-2：pin 随返回的 batch dict 存活（Python 侧 GC 时归还）
                            set_video_pin(py, &dict, video_pin)?;
                            // P3c：三模态对齐视图（请求 align 才透出）
                            if let Some(entries) = av_align {
                                set_av_align(py, &dict, entries)?;
                            }
                        }
                    }
                    return Ok(Some(dict));
                }
            }
        }
    }

    pub fn is_running(&self) -> bool {
        self.inner.is_running()
    }
    /// epoch-0 的名义 batch 数（v3 降级：真实轮长请用 `epoch_total_batches(epoch)`）
    pub fn total_batches_for_rank(&self) -> u64 {
        self.inner.total_batches_for_rank()
    }
    /// **R3**：第 `epoch` 轮的真实 batch 数（`len()` 的数据源）
    pub fn epoch_total_batches(&self, epoch: u64) -> u64 {
        self.inner.epoch_total_batches(epoch)
    }
    /// **R2**：定位到 "第 `epoch` 轮的第 `in_epoch` 个 batch"
    pub fn seek_to_epoch_batch(&mut self, epoch: u64, in_epoch: u64) -> PyResult<()> {
        self.inner
            .seek_to_epoch_batch(epoch, in_epoch)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
    /// **R1**：本轮是否已读完（供 Mixer 判定子流耗尽）
    pub fn at_eof(&self) -> bool {
        self.inner.at_eof()
    }
    /// 请求取消：正在阻塞的 next_batch_capsules 会尽快以 KeyboardInterrupt 返回。
    pub fn cancel(&self) {
        self.inner.cancel();
    }
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

// ==========================================
// 3. 多源混流器绑定
// ==========================================
#[pyclass(name = "RustStreamMixer", module = "megadata_rs")]
pub struct PyRustStreamMixer {
    inner: StreamMixer,
    batch_size: usize,
    max_seq_len: usize,
}

#[pymethods]
impl PyRustStreamMixer {
    #[new]
    #[pyo3(signature = (index_paths, weights, batch_size, max_seq_len, world_size, rank, seed,
                        num_workers=0, worker_id=0, shuffle=false,
                        partition="round_robin", shard_order="random",
                        num_canonical_nodes=0, predownload=0,
                        start_epoch=0))]
    pub fn new(
        py: Python,
        index_paths: Vec<String>,
        weights: Vec<f64>,
        batch_size: usize,
        max_seq_len: usize,
        world_size: usize,
        rank: usize,
        seed: u64,
        num_workers: usize,
        worker_id: usize,
        shuffle: bool,
        partition: &str,
        shard_order: &str,
        num_canonical_nodes: usize,
        predownload: usize,
        start_epoch: u64,
    ) -> PyResult<Self> {
        let inner = py.detach(|| {
            let mut streams = Vec::with_capacity(index_paths.len());
            for path in &index_paths {
                let options = StreamReaderOptions {
                    num_workers,
                    worker_id,
                    shuffle,
                    partition: PartitionAlgo::parse(partition)?,
                    shard_order: ShardOrder::parse(shard_order)?,
                    num_canonical_nodes,
                    predownload,
                    // 同单流：P1 暂不透出（0 = 不限）。
                    prefetch_bytes_limit: 0,
                    // P3b：video 按需读上限（mixer 路径同样登记全局单例）
                    video_ondemand_bytes_limit: 0,
                    // P3c：mixer 路径一期不请求对齐（缺省不出现 = 旧 trainer 零影响）
                    align: false,
                };
                let reader = StreamReader::with_options(
                    path,
                    batch_size,
                    max_seq_len,
                    world_size,
                    rank,
                    seed,
                    options,
                )
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                streams.push(reader);
            }
            // C25/H1：`start_epoch` 在 Rust 内部把全部子流 seek 到 (k, 0)
            let mixer = StreamMixer::new(streams, weights, seed, start_epoch)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok::<StreamMixer, PyErr>(mixer)
        })?;
        Ok(Self {
            inner,
            batch_size,
            max_seq_len,
        })
    }

    /// **R4③（C11/C18）取数协议 —— 三态**。
    ///
    /// 一次调用 = 一次 **in-flight attempt**：先 rng 抽取一次选定 `stream_idx`，随后
    /// 在该子流上循环取数——`Ok(None)`（在途超时）**重试同一 idx**、**不重新采样**
    /// （重新采样会让采样序列依赖 IO 时序：同 seed 不再同序列，C14 的 DDP 论证前提
    /// 随之塌掉，见 F2/C18）；只有"产出"或"该子流本轮已耗尽"才结束 attempt。
    ///
    /// 返回 `(status, stream_idx, batch|None)`：
    /// - `("data", i, dict)`：产出一个 batch（计数已推进）；
    /// - `("exhausted", i, None)`：子流 i **本轮读完**——是正常事件，**不等于 mixer
    ///   结束**（自然模式按轮边界处理，确定性模式 `wrap_stream(i)` 后继续）。
    pub fn next_batch_capsules<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(String, usize, Option<Bound<'py, PyDict>>)> {
        let stream_idx = self.inner.sample_stream_idx();
        loop {
            py.check_signals()?;
            if self.inner.is_cancelled() {
                return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
                    "Megadata canceled by user",
                ));
            }
            let opt = py.detach(|| self.inner.try_next_at(stream_idx));
            match opt {
                Err(ReaderExit::Eof) => {
                    if self.inner.is_cancelled() {
                        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
                            "Megadata canceled by user",
                        ));
                    }
                    return Ok(("exhausted".to_string(), stream_idx, None));
                }
                Err(ReaderExit::Fatal(msg)) => {
                    return Err(pyo3::exceptions::PyIOError::new_err(format!(
                        "megadata mixer sub-stream fatal error (no more silent retries): {msg}"
                    )));
                }
                Ok(None) => continue, // 在途超时 → 重试同一 idx
                Ok(Some(data)) => {
                    // 计数只在**成功产出**后推进（C3/B3）
                    self.inner.on_batch_produced(stream_idx);
                    let dict = PyDict::new(py);
                    match data {
                        BatchData::Arrow(arrow_batch) => {
                            dict.set_item("arrow_batch", arrow_batch.to_pyarrow(py)?)?;
                        }
                        BatchData::Memmap {
                            input_ids,
                            cu_seqlens,
                            doc_ids,
                            loss_weights,
                            images,
                            videos,
                            av_align,
                            video_pin,
                        } => {
                            let capsule = vec_to_dlpack_capsule(
                                py,
                                input_ids,
                                self.batch_size,
                                self.max_seq_len,
                            )?;
                            dict.set_item("input_ids", capsule)?;
                            if !cu_seqlens.is_empty() {
                                dict.set_item("cu_seqlens", cu_seqlens)?;
                            }
                            set_doc_columns(
                                py,
                                &dict,
                                doc_ids,
                                loss_weights,
                                self.batch_size,
                                self.max_seq_len,
                            )?;
                            if let Some(images) = images {
                                set_mm_images(py, &dict, images)?;
                            }
                            // P3b：video 记录（字节延迟；blob_ref 按需读）
                            if let Some(videos) = videos {
                                set_mm_videos(py, &dict, videos)?;
                            }
                            // P1-2：pin 随返回的 batch dict 存活（Python 侧 GC 时归还）
                            set_video_pin(py, &dict, video_pin)?;
                            // P3c：三模态对齐视图（请求 align 才透出）
                            if let Some(entries) = av_align {
                                set_av_align(py, &dict, entries)?;
                            }
                        }
                    }
                    return Ok(("data".to_string(), stream_idx, Some(dict)));
                }
            }
        }
    }

    pub fn update_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        self.inner
            .update_weights(&weights)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// 各子流"当前所在轮"的真实轮长（名义 `len()` 与跨 rank 校验用）
    pub fn substream_epoch_totals(&self) -> Vec<u64> {
        self.inner.substream_epoch_totals()
    }

    /// epoch-0 的名义总 batch 数（v3 降级：名义 len 请用 `substream_epoch_totals()`）
    pub fn total_batches_for_rank(&self) -> u64 {
        self.inner.total_batches_for_rank()
    }

    /// **R4①**：子流 i 本轮是否已读完（零消耗判定轮边界 / wrap 范围）
    pub fn is_exhausted(&self, i: usize) -> PyResult<bool> {
        if i >= self.inner.stream_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "sub-stream index {i} out of range ({} sub-streams in total)",
                self.inner.stream_count()
            )));
        }
        Ok(self.inner.is_exhausted(i))
    }

    /// **R4④**：把子流 i 推进到它自己的下一轮开头（确定性模式 wrap / 重入 wrap-all）
    ///
    /// ★ 评审 §3.2：补上与 [`is_exhausted`] 一致的前置下标校验。此前本方法直接把下标
    ///   透传给 Rust，而 Rust 侧只有 `debug_assert!`（release 下失效）→ 越界索引 panic
    ///   会以 pyo3 `PanicException` 越过 FFI 边界。这里提前拦成 `PyIndexError`，
    ///   给出与 `is_exhausted` 完全一致的可操作消息。
    pub fn wrap_stream(&mut self, i: usize) -> PyResult<()> {
        if i >= self.inner.stream_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "sub-stream index {i} out of range ({} sub-streams in total)",
                self.inner.stream_count()
            )));
        }
        self.inner
            .wrap_stream(i)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// 便捷封装：wrap 所有已耗尽的子流，返回被 wrap 的下标（自然模式重入用，B9'）
    pub fn wrap_all_exhausted(&mut self) -> PyResult<Vec<usize>> {
        let idxs = self.inner.exhausted_streams();
        for &i in &idxs {
            self.inner
                .wrap_stream(i)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        Ok(idxs)
    }

    /// 导出 O(1) 断点续训快照：`{steps_yielded, sub_stream_cursors, rng_word_pos}`。
    /// `sub_stream_cursors[i] = (epoch, in_epoch)`（"下一产出位置"口径；耗尽子流已
    /// 归一化为 `(e+1, 0)`）。几十字节即可精确恢复 RNG 采样状态。
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let s = self.inner.state_dict();
        let d = PyDict::new(py);
        d.set_item("steps_yielded", s.steps_yielded)?;
        d.set_item("sub_stream_cursors", s.sub_stream_cursors)?;
        d.set_item("rng_word_pos", s.rng_word_pos)?;
        Ok(d.into_any().unbind())
    }

    /// O(1) 恢复：state 为 state_dict() 的输出，weights 为快照时刻的采样权重。
    pub fn load_state_dict(
        &mut self,
        state: &Bound<'_, PyDict>,
        weights: Vec<f64>,
    ) -> PyResult<()> {
        let steps_yielded: u64 = state
            .get_item("steps_yielded")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("state missing steps_yielded"))?
            .extract()?;
        // 接受 [[e, p], ...]（JSON/Python 友好），内部转成 (u64, u64)
        let cursors_raw: Vec<[u64; 2]> = state.get_item("sub_stream_cursors")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                "state missing sub_stream_cursors (v3 schema: per-sub-stream [epoch, in_epoch])"))?
            .extract()?;
        let sub_stream_cursors: Vec<(u64, u64)> =
            cursors_raw.into_iter().map(|[e, p]| (e, p)).collect();
        let rng_word_pos: u128 = state
            .get_item("rng_word_pos")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("state missing rng_word_pos"))?
            .extract()?;
        let mixer_state = MixerState {
            steps_yielded,
            sub_stream_cursors,
            rng_word_pos,
        };
        self.inner
            .load_state(&mixer_state, &weights, None)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn cancel(&self) {
        self.inner.cancel();
    }
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

// ==========================================
// 4. 仿真器数据源：shard 读取序规划
// ==========================================
/// 产出每个全局 worker "每 step 应读的全局 shard id" 序列（仿真器用，零 IO）。
///
/// 与 `RustStreamReader` 的真实读取序一致（同一套分区 / 置换纯函数），
/// 供 `python/megadata/simulation.py` 在不跑训练的前提下做容量规划。
///
/// 返回 `out[worker][step] = 全局 shard 下标`；worker 扁平编号
/// `w = rank * num_workers + worker_id`。
#[pyfunction]
#[pyo3(signature = (index_path, world_size, num_workers=0, seed=42, epochs=1,
                    shuffle=true, partition="round_robin", shard_order="random",
                    num_canonical_nodes=0))]
fn plan_shard_sequence(
    py: Python,
    index_path: String,
    world_size: usize,
    num_workers: usize,
    seed: u64,
    epochs: u64,
    shuffle: bool,
    partition: &str,
    shard_order: &str,
    num_canonical_nodes: usize,
) -> PyResult<Vec<Vec<u32>>> {
    let algo = PartitionAlgo::parse(partition)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let order = ShardOrder::parse(shard_order)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    py.detach(move || {
        // 一次性纯计算任务：单线程 runtime 足够（避免为一次调用起 nCPU 个空闲线程）
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to build Tokio runtime: {e}"))
            })?;
        let manifest = DatasetManifest::load_smart(&rt, &index_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut world = World::from_sizes(world_size, 1, num_workers);
        if num_canonical_nodes > 0 {
            world.num_canonical_nodes = Some(num_canonical_nodes);
        }
        Ok(crate::runtime::sampler::plan_shard_sequence(
            &manifest, &world, algo, seed, epochs, shuffle, order,
        ))
    })
}

/// **纯函数**：第 `epoch` 轮本 (rank, worker) 将产出的 batch 数（`len()` 的数据源）。
///
/// A3 查询零副作用：不构造 reader、不启动预取线程、不做批量 IO——
/// 只读 manifest 元数据 + 跑一遍分区/置换纯函数（与 `plan_shard_sequence` 同源）。
/// 供 `BatchStream.__len__` / `StreamingDataset.__len__` 在没有（也不应有）
/// 活跃读头时回答问题。
#[pyfunction]
#[pyo3(signature = (index_path, epoch, world_size, rank, num_workers=0, worker_id=0,
                    seed=42, partition="round_robin", num_canonical_nodes=0))]
fn epoch_total_batches(
    py: Python,
    index_path: String,
    epoch: u64,
    world_size: usize,
    rank: usize,
    num_workers: usize,
    worker_id: usize,
    seed: u64,
    partition: &str,
    num_canonical_nodes: usize,
) -> PyResult<u64> {
    let algo = PartitionAlgo::parse(partition)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    py.detach(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to build Tokio runtime: {e}"))
            })?;
        let manifest = DatasetManifest::load_smart(&rt, &index_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut world = World::from_sizes(world_size, 1, num_workers);
        if num_canonical_nodes > 0 {
            world.num_canonical_nodes = Some(num_canonical_nodes);
        }
        let sampler = DistributedSampler::with_world(world, rank, worker_id, seed, algo);
        Ok(sampler
            .get_my_shards(&manifest, epoch)
            .iter()
            .map(|s| s.num_batches as u64)
            .sum())
    })
}

/// 读取数据集 manifest（`index.json`）并返回其 JSON 文本。
///
/// 支持本地路径 / `http(s)://` / `s3://`（对象存储）并带重试退避 —— 供 Python 侧
/// 解析 manifest 概要（如按 token 数配比 Mixer 权重）。旧实现是 Python `urllib`
/// 直连（无重试）且对 `s3://` 直接 `NotImplementedError`（复核问题 12）。
#[pyfunction]
fn read_manifest_json(py: Python<'_>, index_path: String) -> PyResult<String> {
    // ★ 必须释放 GIL：这是阻塞式网络/磁盘 IO。HTTP 服务端可能就是本进程的 Python 线程
    //   （测试里的 mock 云存储、用户自建数据服务），持 GIL 等响应会直接死锁
    //   （复核修复期间实测：整个测试套件卡死在这里）。
    py.detach(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to build Tokio runtime: {e}"))
            })?;
        crate::io::CloudIoManager::read_manifest_to_string(&rt, &index_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read manifest {index_path}: {e}"
            ))
        })
    })
}

// ==========================================
// 4b. 多模态 P0 读取诊断入口（仅测试/诊断用，非生产构建路径）
// ==========================================

/// 把 `"skip"` / `"fail"`（容忍 `skip_and_count` / `fail_fast` 等变体）解析为
/// `FailurePolicy`；未识别值回退到 `default`（保持宽容，便于测试构造）。
fn parse_failure_policy(s: &str, default: FailurePolicy) -> FailurePolicy {
    match s.to_ascii_lowercase().as_str() {
        "skip" | "skipandcount" | "skip_and_count" | "count" => FailurePolicy::SkipAndCount,
        "fail" | "failfast" | "fail_fast" => FailurePolicy::FailFast,
        _ => default,
    }
}

/// 从 Python dict 解析 `ReadSamplesOptions`（键全可选，缺省取方案默认值）。
/// P4：解析 field_map dict → FieldMap（缺失键保留默认）。
/// `parse_read_samples_options` 的嵌套键与 `build_dataset` 的直接 dict 共用。
/// 纯 Rust 侧的字段映射装配（**不触碰解释器**）。
///
/// P4-1（评审）：原先将 `parse_field_map_dict` 的语义全部写死在 pyo3 取参里，
/// 单测只能在 `feature = "pyo3-tests"` 下跑（默认 `cargo test` 不执行）。这里把
/// "键缺失 ⇒ 保留默认值" 的装配语义独立出来，默认 `cargo test` 即可回归。
fn build_field_map(
    image_fields: Option<Vec<String>>,
    text_fields: Option<Vec<String>>,
    audio_fields: Option<Vec<String>>,
    video_fields: Option<Vec<String>>,
    meta_fields: Option<Vec<String>>,
    projection: Option<Vec<String>>,
    timestamp_field: Option<String>,
) -> crate::builder::source::FieldMap {
    let mut out = crate::builder::source::FieldMap::default();
    if let Some(v) = image_fields {
        out.image_fields = v;
    }
    if let Some(v) = text_fields {
        out.text_fields = v;
    }
    if let Some(v) = audio_fields {
        out.audio_fields = v;
    }
    if let Some(v) = video_fields {
        out.video_fields = v;
    }
    if let Some(v) = meta_fields {
        out.meta_fields = v;
    }
    if projection.is_some() {
        out.projection = projection;
    }
    if timestamp_field.is_some() {
        out.timestamp_field = timestamp_field;
    }
    out
}

/// P4-3（评审）+ VR-4e（复审）：`field_map` 的 **vlm-only 作用域校验**（绑定层入口）。
///
/// 为何抽成纯 Rust 函数：`build_dataset` 是几十个参数的 pyfunction，无法在默认
/// `cargo test` 里直调；而 Python 层 `BuildConfig` 会先把这件事拦下，导致绑定层的
/// 对称校验长期无人锁。抽出来后可直接验证“直调 Rust 绑定也不静默忽略”这条
/// loud-fail 契约（默认测试套件即跑，不需 `--features pyo3-tests`/libpython）。
fn validate_field_map_scope(is_vlm: bool, field_map_is_some: bool) -> Result<(), String> {
    if !is_vlm && field_map_is_some {
        return Err(
            "field_map is a vlm-only argument (consumed by the tar/parquet source \
                    readers): pt/sft builds must not pass field_map (fail loudly rather \
                    than silently ignore)"
                .to_string(),
        );
    }
    Ok(())
}

fn parse_field_map_dict(fm: &Bound<'_, PyDict>) -> PyResult<crate::builder::source::FieldMap> {
    let get_list = |k: &str| -> PyResult<Option<Vec<String>>> {
        match fm.get_item(k)? {
            Some(v) if !v.is_none() => Ok(Some(v.extract::<Vec<String>>().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "field_map.{k} must be list[str]: {e}"
                ))
            })?)),
            _ => Ok(None),
        }
    };
    Ok(build_field_map(
        get_list("image_fields")?,
        get_list("text_fields")?,
        get_list("audio_fields")?,
        get_list("video_fields")?,
        get_list("meta_fields")?,
        get_list("projection")?,
        match fm.get_item("timestamp_field")? {
            Some(v) if !v.is_none() => Some(v.extract::<String>().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "field_map.timestamp_field must be str: {e}"
                ))
            })?),
            _ => None,
        },
    ))
}

fn parse_read_samples_options(opts: &Bound<'_, PyDict>) -> PyResult<ReadSamplesOptions> {
    let mut cfg = ReadSamplesOptions::default();
    // chunk 双约束（D5）
    let max_samples = opts
        .get_item("max_samples")?
        .map(|v| v.extract::<usize>())
        .transpose()?;
    let max_bytes = opts
        .get_item("max_bytes")?
        .map(|v| v.extract::<u64>())
        .transpose()?;
    if max_samples.is_some() || max_bytes.is_some() {
        cfg.chunk_limit = ChunkLimit::new(max_samples.unwrap_or(0), max_bytes.unwrap_or(0));
    }
    // 失败策略（按分类）
    if let Some(v) = opts
        .get_item("source_not_found")?
        .and_then(|v| v.extract::<String>().ok())
    {
        cfg.failure_policy.source_not_found =
            parse_failure_policy(&v, cfg.failure_policy.source_not_found);
    }
    if let Some(v) = opts
        .get_item("source_corrupt")?
        .and_then(|v| v.extract::<String>().ok())
    {
        cfg.failure_policy.source_corrupt =
            parse_failure_policy(&v, cfg.failure_policy.source_corrupt);
    }
    if let Some(v) = opts
        .get_item("decode_failed")?
        .and_then(|v| v.extract::<String>().ok())
    {
        cfg.failure_policy.decode_failed =
            parse_failure_policy(&v, cfg.failure_policy.decode_failed);
    }
    if let Some(v) = opts
        .get_item("encode_failed")?
        .and_then(|v| v.extract::<String>().ok())
    {
        cfg.failure_policy.encode_failed =
            parse_failure_policy(&v, cfg.failure_policy.encode_failed);
    }
    if let Some(b) = opts
        .get_item("download_url_media")?
        .map(|v| v.extract::<bool>())
        .transpose()?
    {
        cfg.download_url_media = b;
    }
    if let Some(n) = opts
        .get_item("download_attempts")?
        .map(|v| v.extract::<u32>())
        .transpose()?
    {
        cfg.download_attempts = n;
    }
    // P4：field_map（源字段 → 角色映射；tar/parquet reader 消费，JSONL 不读）
    if let Some(fm) = opts.get_item("field_map")? {
        let fm = fm.extract::<Bound<'_, PyDict>>().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("field_map must be a dict: {e}"))
        })?;
        cfg.field_map = parse_field_map_dict(&fm)?;
    }
    Ok(cfg)
}

/// 读取线程产出的聚合结果（error_counts / allocated 在线程内创建，结束时带回）。
struct ReadThreadOutcome {
    result: Result<(), String>,
    error_counts: SourceErrorCounts,
    allocated: u64,
}

/// 多模态 P0 读取诊断入口（P0-T6/T7）：驱动 `read_samples` 跑完输入文件，
/// 用 `MmSampleSink`（encoder 占位）跨 chunk 校验 sample_id 无漏无重，并回传
/// 跳过计数 / chunk 分布 / FailFast 中止信息。
///
/// **仅测试与诊断用**：生产 vlm 构建走 `build_dataset(mode="vlm")`（P2 packer
/// 落地后接通；P0 下因 0 shard 必然报错，无法产数据集）。本入口让 P0 集成测试
/// 能在不依赖 P2 packer 的前提下验证读取侧的全部行为契约。
///
/// 返回 dict：`{sample_ids, allocated, samples_seen, skipped_by_class,
/// fail_fast_triggered, chunk_count, chunk_sizes, chunk_byte_sizes, aborted,
/// abort_msg, id_check, bytes_read, docs_read}`。
#[pyfunction]
#[pyo3(signature = (input_paths, options=None))]
fn debug_read_samples(
    py: Python<'_>,
    input_paths: Vec<String>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // 先在 GIL 内解析 options → 纯 Rust 配置（detach 后不再触碰 Python 对象）
    let cfg = match options {
        Some(d) => parse_read_samples_options(d)?,
        None => ReadSamplesOptions::default(),
    };

    let summary = py
        .detach(move || -> Result<DebugSummary, String> {
            let (tx, rx) = flume::bounded::<Vec<crate::builder::sample::Sample>>(100);
            let progress = BuildProgress::new();
            let progress_for_read = progress.clone();

            // 读取线程：串行跑 input_paths，共享 id_alloc + error_counts
            let cfg_for_thread = cfg.clone();
            let paths_for_thread = input_paths.clone();
            let read_handle = std::thread::spawn(move || -> ReadThreadOutcome {
                let alloc = SampleIdAllocator::new();
                let mut error_counts = SourceErrorCounts::default();
                for path in &paths_for_thread {
                    if let Err(e) = read_samples(
                        path,
                        &cfg_for_thread,
                        &alloc,
                        &mut error_counts,
                        tx.clone(),
                        progress_for_read.clone(),
                    ) {
                        return ReadThreadOutcome {
                            result: Err(e.to_string()),
                            error_counts,
                            allocated: alloc.allocated(),
                        };
                    }
                }
                // tx 最后一个 clone 在闭包退出时 drop → rx 解除阻塞
                ReadThreadOutcome {
                    result: Ok(()),
                    error_counts,
                    allocated: alloc.allocated(),
                }
            });

            // 主线程：drain + MmSampleSink 校验 id 连续性
            let mut sink = MmSampleSink::new();
            let mut sample_ids: Vec<u64> = Vec::new();
            let mut chunk_sizes: Vec<usize> = Vec::new();
            let mut chunk_byte_sizes: Vec<u64> = Vec::new();
            let mut id_check = "ok".to_string();
            while let Ok(chunk) = rx.recv() {
                let sz = chunk.len();
                let mut bytes = 0u64;
                for s in &chunk {
                    sample_ids.push(s.sample_id.0);
                    bytes += s.payload_bytes();
                }
                if let Err(e) = sink.push_chunk(&chunk) {
                    id_check = format!("fail: {e}");
                    break;
                }
                chunk_sizes.push(sz);
                chunk_byte_sizes.push(bytes);
            }

            let thread_outcome = read_handle
                .join()
                .map_err(|_| "read thread panicked".to_string())?;
            let aborted = thread_outcome.result.is_err();
            let abort_msg = thread_outcome.result.err();

            Ok(DebugSummary {
                sample_ids,
                allocated: thread_outcome.allocated,
                samples_seen: sink.count(),
                skipped_by_class: thread_outcome.error_counts.skipped_by_class,
                fail_fast_triggered: thread_outcome.error_counts.fail_fast_triggered,
                chunk_count: chunk_sizes.len(),
                chunk_sizes,
                chunk_byte_sizes,
                aborted,
                abort_msg,
                id_check,
                bytes_read: progress
                    .bytes_read
                    .load(std::sync::atomic::Ordering::Relaxed),
                docs_read: progress
                    .docs_read
                    .load(std::sync::atomic::Ordering::Relaxed),
            })
        })
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    // GIL 已重新获取：构造返回 dict
    let d = PyDict::new(py);
    d.set_item("sample_ids", summary.sample_ids)?;
    d.set_item("allocated", summary.allocated)?;
    d.set_item("samples_seen", summary.samples_seen)?;
    let skip_map = PyDict::new(py);
    for (k, v) in &summary.skipped_by_class {
        skip_map.set_item(k, *v)?;
    }
    d.set_item("skipped_by_class", skip_map)?;
    d.set_item("fail_fast_triggered", summary.fail_fast_triggered)?;
    d.set_item("chunk_count", summary.chunk_count)?;
    d.set_item("chunk_sizes", summary.chunk_sizes)?;
    d.set_item("chunk_byte_sizes", summary.chunk_byte_sizes)?;
    d.set_item("aborted", summary.aborted)?;
    d.set_item("abort_msg", summary.abort_msg)?;
    d.set_item("id_check", summary.id_check)?;
    d.set_item("bytes_read", summary.bytes_read)?;
    d.set_item("docs_read", summary.docs_read)?;
    Ok(d.into_any().unbind())
}

/// P3b：video blob 按需读（FFI 入口；collate 时点调用）。
///
/// `store_id/offset/len` 来自 batch dict `videos[].blob_ref`。返回 **per-clip
/// Owned DLPack capsule**（u8 一维，torch 可 `from_dlpack`）——`Vec<u8>` 所有权
/// 经 managed tensor 转移（deleter = Rust drop），`copy_count = 0`（所有权
/// 转移非复制；P3b §7.3 per-clip Owned 形态）。
///
/// 账本与背压（§7.4）：在途字节经 `VideoBlobStore` 全局账本计量；超
/// `video_ondemand_bytes_limit` 时自旋等待（其它读交付后收敛），单 clip 本身
/// 超限 WARN 一次 + 放行（活性保证）。
///
/// 错误语义：store_id 未登记（shard 已切换/未注册）→ `FileNotFound` 风格
/// `PyIOError`；区间越界 → `InvalidData` 风格错误（blob/rec 不一致）。
#[pyfunction]
fn read_vid_blob(py: Python<'_>, store_id: u64, offset: u64, len: u32) -> PyResult<Py<PyAny>> {
    use crate::runtime::ffi_dlpack::{DLDataType, bytes_to_dlpack_capsule};
    const U8_DTYPE: DLDataType = DLDataType {
        code: 1,
        bits: 8,
        lanes: 1,
    }; // 1=UInt
    // P3b-2（评审）：read_tracked 内含背压自旋 + seek+read_exact 阻塞 IO，
    // pyo3 默认持 GIL 执行 → 违反"阻塞 IO 释放 GIL"纪律（handoff §3.6）。
    // 用 detach 包裹读段，返回后在 GIL 内做 capsule。
    let data = py.detach(move || {
        crate::runtime::video_blob::VideoBlobStore::global()
            .read_tracked(store_id, offset, len as u64)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    })?;
    bytes_to_dlpack_capsule(py, data, U8_DTYPE, vec![len as i64], vec![1])
}

/// `debug_read_samples` 的返回聚合（纯 Rust，跨 detach 边界传递）。
struct DebugSummary {
    sample_ids: Vec<u64>,
    allocated: u64,
    samples_seen: u64,
    skipped_by_class: std::collections::BTreeMap<String, u64>,
    fail_fast_triggered: u64,
    chunk_count: usize,
    chunk_sizes: Vec<usize>,
    chunk_byte_sizes: Vec<u64>,
    aborted: bool,
    abort_msg: Option<String>,
    id_check: String,
    bytes_read: u64,
    docs_read: u64,
}

// ==========================================
// 5. 全局注册
// ==========================================
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = tracing_subscriber::fmt().with_target(false).try_init();

    m.add_class::<PyDataBuilder>()?;
    m.add_class::<PyRustStreamReader>()?;
    m.add_class::<PyRustStreamMixer>()?;

    m.add_function(wrap_pyfunction!(plan_shard_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(epoch_total_batches, m)?)?;
    m.add_function(wrap_pyfunction!(read_manifest_json, m)?)?;
    m.add_function(wrap_pyfunction!(debug_read_samples, m)?)?;
    m.add_function(wrap_pyfunction!(read_vid_blob, m)?)?;

    // manifest 版本白名单（唯一事实源在 `builder::metadata`）：`megadata.validate`
    // 从这里读，避免 Python 侧手抄第二份、改 Rust 写入端后静默漂移。
    m.add(
        "SUPPORTED_MANIFEST_VERSIONS",
        crate::builder::metadata::SUPPORTED_MANIFEST_VERSIONS.to_vec(),
    )?;

    // ★ 评审 §3.8(d)：把 Rust 的预取窗口默认值**导出**给 Python，避免
    //   `python/megadata/simulation.py` 独立硬编码一份 `predownload=16`。
    //   原状下改Rust 的 DEFAULT_PREDOWNLOAD 不会同步仿真器，容量规划结论会失真
    //   （仿真说"16 够用"，真实引擎其实是另一个值）。现在单一事实源在 Rust，
    //   Python 侧在 import 时读取，两个模块不可能再漂移。
    m.add(
        "DEFAULT_PREDOWNLOAD",
        crate::runtime::stream::DEFAULT_PREDOWNLOAD,
    )?;
    // 同时导出数据通道容量下界，供仿真器/文档核对（`data_channel_capacity(0)`）。
    m.add(
        "MIN_DATA_CHANNEL_CAPACITY",
        crate::runtime::stream::MIN_DATA_CHANNEL_CAPACITY,
    )?;

    // 中断支持：v3（§6/D2）移除了 `request_cancel` / `is_cancel_requested` /
    // `raise_if_canceled` 的 Python 面（每实例 `cancel()` 已全覆盖）；
    // 2026-09-27 评审清理中 `reset_cancel` 与全局 CANCEL_FLAG 一并退役
    // （`uninstall_signal_handlers` 里的调用本就是 no-op）。

    Ok(())
}

// ==========================================
// 6. 单测（P4-1 评审补齐：field_map 装配）
// ==========================================
/// P4-1（评审）：`build_field_map` 的**纯 Rust 装配语义**测试——
/// 不用解释器、不依赖 `--features pyo3-tests`，默认 `cargo test` 即执行。
/// （取值/类型检查仍需解释器，见下方 pyo3-tests 模块。）
#[cfg(test)]
mod field_map_tests {
    use super::{build_field_map, validate_field_map_scope};
    use crate::builder::source::FieldMap;

    #[test]
    fn build_field_map_all_none_keeps_defaults() {
        let fm = build_field_map(None, None, None, None, None, None, None);
        let d = FieldMap::default();
        assert_eq!(fm.image_fields, d.image_fields);
        assert_eq!(fm.text_fields, d.text_fields);
        assert_eq!(fm.audio_fields, d.audio_fields);
        assert_eq!(fm.video_fields, d.video_fields);
        assert_eq!(fm.meta_fields, d.meta_fields);
        assert_eq!(fm.projection, None);
        assert_eq!(fm.timestamp_field, None);
        // P4-2：tar 扩展名形态与 parquet 列名形态共用单列表——两侧都必须在默认里
        for k in [
            "jpg",
            "png",
            "image",
            "image_bytes",
            "image_jpeg",
            "image_webp",
        ] {
            assert!(
                fm.image_fields.iter().any(|x| x == k),
                "missing image field {k}"
            );
        }
        for k in ["txt", "caption", "sentence"] {
            assert!(
                fm.text_fields.iter().any(|x| x == k),
                "missing text field {k}"
            );
        }
        for k in ["flac", "audio", "audio_bytes"] {
            assert!(
                fm.audio_fields.iter().any(|x| x == k),
                "missing audio field {k}"
            );
        }
        for k in ["mp4", "video", "video_bytes"] {
            assert!(
                fm.video_fields.iter().any(|x| x == k),
                "missing video field {k}"
            );
        }
        for k in ["json", "meta", "metadata"] {
            assert!(
                fm.meta_fields.iter().any(|x| x == k),
                "missing meta field {k}"
            );
        }
    }

    #[test]
    fn build_field_map_overrides_only_touched_keys() {
        let fm = build_field_map(
            Some(vec!["my_img".to_string()]),
            Some(vec!["my_text".to_string()]),
            Some(vec!["my_aud".to_string()]),
            Some(vec!["my_vid".to_string()]),
            Some(vec!["my_meta".to_string()]),
            Some(vec!["col_a".to_string()]),
            Some("ts".to_string()),
        );
        assert_eq!(fm.image_fields, vec!["my_img".to_string()]);
        assert_eq!(fm.text_fields, vec!["my_text".to_string()]);
        assert_eq!(fm.audio_fields, vec!["my_aud".to_string()]);
        assert_eq!(fm.video_fields, vec!["my_vid".to_string()]);
        assert_eq!(fm.meta_fields, vec!["my_meta".to_string()]);
        assert_eq!(fm.projection, Some(vec!["col_a".to_string()]));
        assert_eq!(fm.timestamp_field, Some("ts".to_string()));
    }

    /// 部分覆盖：只给 text_fields ⇒ 其余键保持默认（避免"给了就全清"的静默丢默认）。
    #[test]
    fn build_field_map_partial_override_keeps_other_defaults() {
        let fm = build_field_map(
            Some(vec!["only".to_string()]),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(fm.image_fields, vec!["only".to_string()]);
        assert_eq!(fm.text_fields, FieldMap::default().text_fields);
        assert_eq!(fm.timestamp_field, None);
        assert_eq!(fm.projection, None);
    }

    /// VR-4e（复审）：绑定层入口的 vlm-only loud-fail 直接锚点。
    ///
    /// 现有 Python 用例（`test_p4_field_map_rejected_for_pt_and_sft`）走的是
    /// `BuildConfig` 预校验，**先于**绑定层就抛了；这里直接锁住 Rust 侧对称校验，
    /// 防止后续重构把它扣掉变成“静默忽略”。
    #[test]
    fn validate_field_map_scope_rejects_pt_sft_but_allows_vlm() {
        let err = validate_field_map_scope(false, true).unwrap_err();
        assert!(err.contains("vlm-only"), "错误文案须点明 vlm-only：{err}");
        // 其余三种组合必须放行（防过度拦截）
        assert!(
            validate_field_map_scope(true, true).is_ok(),
            "vlm + field_map 应放行"
        );
        assert!(
            validate_field_map_scope(false, false).is_ok(),
            "pt/sft 不传应放行"
        );
        assert!(
            validate_field_map_scope(true, false).is_ok(),
            "vlm 不传应放行"
        );
    }
}

#[cfg(all(test, feature = "pyo3-tests"))]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    /// P4-1（评审）：`parse_field_map_dict` 默认值补齐 + 覆盖 + timestamp_field。
    #[test]
    fn parse_field_map_dict_defaults_and_overrides() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // 空 dict → 全默认值（含 parquet 列名，P4-2）
            let d = PyDict::new(py);
            let fm = parse_field_map_dict(&d).unwrap();
            assert!(fm.image_fields.contains(&"image_bytes".to_string()));
            assert!(fm.timestamp_field.is_none());

            // 覆盖 image_fields + timestamp_field
            let d = PyDict::new(py);
            d.set_item("image_fields", vec!["my_img".to_string()])
                .unwrap();
            d.set_item("timestamp_field", "ts").unwrap();
            let fm = parse_field_map_dict(&d).unwrap();
            assert_eq!(fm.image_fields, vec!["my_img".to_string()]);
            assert_eq!(fm.timestamp_field, Some("ts".to_string()));
            // 未覆盖的键保留默认
            assert!(fm.text_fields.contains(&"txt".to_string()));
        });
    }

    /// P4-1（评审）：类型错误 → PyTypeError。
    #[test]
    fn parse_field_map_dict_type_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("image_fields", 42i64).unwrap(); // 非列表
            assert!(parse_field_map_dict(&d).is_err());
        });
    }
}

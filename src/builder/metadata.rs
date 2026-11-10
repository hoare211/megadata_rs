use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

/// 支持的数据落盘格式
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StorageFormat {
    #[serde(rename = "Arrow-IPC")]
    ArrowIpc,
    #[serde(rename = "Memmap")]
    Memmap,
    /// PT 裸流 + per-batch zstd frame 压缩（借鉴 MDS 的「压缩与 encoding 正交」，
    /// 但压缩粒度是单个 batch 独立 frame，保留 O(1) 随机寻址）。
    #[serde(rename = "Zstd-MDS")]
    ZstdMds,
}

/// 记录单个 Shard 的微观元数据
///
/// `hashes`：物理文件的内容哈希（algo → hex），如 {"sha256": "..."}。
/// 构建期"写完回读"计算；读取端校验，失败删缓存重下（借鉴 streaming 的
/// validate_hash：解压/使用前校验）。
/// 旧版 index.json 只有 `checksum: "fast-hash"` 字段（假校验，无 hash 能力），
/// serde 会忽略未知字段、本字段缺省为空表 → 跳过校验，完全向后兼容。
///
/// 用 `BTreeMap` 而非 `HashMap`：序列化顺序稳定，保证**同输入两次构建产出的
/// index.json 逐字节一致**（HashMap 的迭代顺序随进程随机 → 构建不可复现）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMeta {
    pub shard_id: usize,
    pub filename: String,
    pub num_batches: usize,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hashes: BTreeMap<String, String>,
    /// P3c：对齐音频范围索引——覆盖本 Vlm shard 样本的 audio 流
    /// (shard, batch) **闭区间**（四元组坐标；reader 据此定位音频侧 rec，
    /// 按 key 过滤超集）。`None` = 无 audio 流 / 对齐未启用。
    ///
    /// **精确语义（VR-2 重构后）**：区间在 audio 流收尾后由 per-batch span
    /// （`AudioBatchSpan{min_sid,max_sid,ashard,batch}`，见 `pipeline.rs`）
    /// **构造性导出**——对本 shard 的样本 id 闭区间做 `partition_point` 二分，
    /// 得到与之重叠的 span 连续段，即“必需 audio rec 的精确最小超集”。
    /// 超集只出现在**边界样本**（Vlm 行跨 shard / audio rec 跨批 ⇒ 同一 rec
    /// 同时进入相邻 shard 的区间），属 join 语义所需，不是误差。
    ///
    /// 旧述“弱一致性（构建期游标快照，audio 流可能略超前/滞后）”已废弃——
    /// 那是对已删除的逐 shard 游标机制（`aud_cursor` + 水位屏障）的描述。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align_aud_range: Option<AlignAudRange>,
}

/// P3c：audio 流 (shard, batch) 闭区间（`align_aud_range` 的值类型）。
/// 坐标 = audio 流自身 shard 序列内的下标（与 Vlm 流 shard_idx 无关）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignAudRange {
    /// 起始（含）`(aud_shard, aud_batch_in_shard)`
    pub first: (u32, u32),
    /// 结束（含）`(aud_shard, aud_batch_in_shard)`
    pub last: (u32, u32),
}

/// 构建期使用的分词器指纹（第三方分词器接入方案 §6 / D7）。
///
/// **只写不读**：manifest 里记录"这份数据集是用哪个分词器构建的"，供事后追溯
/// （换分词器重建后 sha 变了能立刻发现）。读取端不接触分词器，因此**不做校验**——
/// 防呆靠构建工具与文档。
///
/// 与 v3 的 `manifest_fingerprint`（读取端数据集身份）是独立并存的两个字段。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenizerFingerprint {
    /// `"bbpe"`（Rust 直连默认路径）| `"python"`（第三方分词器回调路径）
    pub backend: String,
    /// bbpe：`sha256(文件)[:16] + "@v" + format_version`；
    /// python：adapter 报告的标识（`name_or_path` / 类名），无则 `"opaque"`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
    /// 词表大小；不可得时为 `None`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<u32>,
}

/// 记录全局统计信息的结构体
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GlobalStats {
    /// 真实有效 token 数（不含 padding；构建期累加 cu_seqlens 末值）。
    /// v2.0 起语义修复：旧版把 batch_size*max_seq_len 的 padding 也计入，
    /// 会系统性高估高 padding 率数据源在 MultiSourceMixLoader 温度权重中的占比。
    pub total_tokens: u64,
    pub total_batches: usize,
    pub total_shards: usize,
    /// 构建期使用的 batch_size（加载期校验一致性，防止维度错位）。
    /// 旧 index.json 无此字段时默认 0，表示跳过校验。
    #[serde(default)]
    pub batch_size: usize,
    /// F2 截断审计：SFT 构建期因超过 max_seq_len 被尾部截断的文档数
    /// （截断策略为 tail：超出部分直接丢弃）。
    /// 旧 index.json 无此字段时默认 0；为 0 时序列化整体省略
    /// （保持既有产物的 byte 级格式稳定，向后兼容）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub truncated_docs: u64,
    /// F2 截断审计：截断丢弃的 token 总数（语义同上）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub truncated_tokens: u64,
    /// PT 切分审计：因长度超过 `batch_size*max_seq_len` 而被**跨 batch 切分**的
    /// 文档数（PT 不截断，超长文档被连续铺到多个 batch，每段在 `cu_seqlens` 里
    /// 各自成为一个独立段）。
    ///
    /// 与 `truncated_docs` 的区别务必分清：SFT 是**丢弃**尾部（数据损失），
    /// PT 是**完整保留但分段**（数据不丢，但段间注意力被切断）。两者都非零都值得
    /// 警惕，但排查方向完全不同。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub split_docs: u64,
    /// PT 切分审计：被切分文档的原始 token 总数（含被分到后续 batch 的部分）。
    /// 配合 `max_seq_len` 可估算出这些文档平均被切成几段。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub split_tokens: u64,
    /// 多模态 P0-T5：读取期按错误分类跳过的样本数（键为
    /// `ErrorClass::as_str()` 的稳定字符串，如 `source_not_found` /
    /// `source_corrupt` / `decode_failed` / `encode_failed` / `id_conflict`）。
    ///
    /// **与 `truncated_docs` 的区别**：truncated 是 SFT 文本超长被**尾部丢弃**
    /// （token 级损失）；此处是样本级整条**跳过**（图像下载失败 / 解码失败等），
    /// 对应的 `sample_id` 出现空缺——加载端按 id 寻址时会看到 gap。
    ///
    /// `BTreeMap` 保证序列化顺序稳定 → 构建可复现；空表整体省略（旧产物
    /// byte 级兼容）。`sample_id` 与跳过位置的对应关系见 `read_samples`。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub skipped_samples_by_class: BTreeMap<String, u64>,
    /// 多模态 P0-T5：FailFast 策略触发次数（每次都中止构建，正常应为 0）。
    /// 非零意味着构建因 `encode_failed` 等 FailFast 分类被中止过——与
    /// `skipped_samples_by_class` 互补：后者记录 SkipAndCount 的"丢而继续"，
    /// 本字段记录"丢而中止"。为 0 时省略（旧格式兼容）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub fail_fast_triggered: u64,
    /// 多模态 P1-T3：模态拆分统计的口径版本（sidecar 格式冻结 §8）。
    /// `2` = 本版：`total_tokens == Σ 模态有效 token` 守恒成立。
    /// 缺省（旧 manifest）= `0`：读取语义上按 `total_text_tokens = total_tokens`、
    /// 其余模态 = 0 解释（见 [`Self::effective_modality_tokens`]），
    /// **不改写加载的 manifest**（读不改写纪律）。为 0 时省略（旧格式兼容）。
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub stats_version: u32,
    /// 文本有效 token（不含 padding；口径与 `total_tokens` 相同——D3 原文
    /// "total_tokens 含 padding"与代码事实不符，冻结时以代码为准修正，见格式冻结 §8）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_text_tokens: u64,
    /// visual placeholder 有效 token（P2 起由 vlm packer 累加；文本构建恒 0 省略）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_image_tokens: u64,
    /// P3a 音频（P3 前恒 0 省略；格式一次到位，见 D3）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_audio_tokens: u64,
    /// P3b 视频。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_video_tokens: u64,
    /// 图像记录数（VALID + EMPTY_IMAGE；不含 NO_IMAGE padding 位）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_images: u64,
    /// P3a 音频片段数。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_audio_clips: u64,
    /// P3b 视频片段数。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub total_video_clips: u64,
    /// P3a 音频总时长（秒；0.0 省略）。
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub total_audio_seconds: f64,
    /// P3b 视频总时长（秒；0.0 省略）。
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub total_video_seconds: f64,
    /// P3b §6.1：源缺 `MediaMeta.fps` 而采用构建默认 fps 的 video 段数
    /// （消费侧以此为信号核查源数据——默认 fps 写入的 rec 帧时间有偏差）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub video_fps_defaulted_clips: u64,
    /// P3a-2（评审）：空 payload / 零 token 的 audio 段数（SkipAndCount 语义，
    /// 不产 rec、不进流）。为 0 时省略（旧格式 byte 级兼容）。
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub audio_empty_segments: u64,
}

/// serde `skip_serializing_if` 助手：0 视为"未记录"（字段整体省略，旧格式兼容）。
fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// 同 [`is_zero_u64`]（u32 版）。
fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// 同 [`is_zero_u64`]（f64 版：0.0 省略，含 -0.0）。
fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

impl GlobalStats {
    /// 模态拆分的读取期解释（sidecar 格式冻结 §8）：
    /// - `stats_version >= 2`：直接用拆分字段；
    /// - 旧 manifest（字段缺失 = 0）：**明确解释**为 `text = total_tokens`、
    ///   其余模态 = 0，不静默错解（D3 / 评审 A 问题 6）。
    ///
    /// 返回 `(text, image, audio, video)` 有效 token 四元组。
    pub fn effective_modality_tokens(&self) -> (u64, u64, u64, u64) {
        if self.stats_version >= 2 {
            (
                self.total_text_tokens,
                self.total_image_tokens,
                self.total_audio_tokens,
                self.total_video_tokens,
            )
        } else {
            (self.total_tokens, 0, 0, 0)
        }
    }

    /// 守恒不变式（sidecar 格式冻结 §8 / §9.4，测试锚点）：
    /// `stats_version >= 2` 时必须 `total_tokens == Σ 模态有效 token`。
    /// 违反 = 统计路径丢数（packer/encoder 计数 bug），构建端硬失败，
    /// 不带病落盘 manifest。
    pub fn check_modality_conservation(&self) -> Result<(), String> {
        if self.stats_version < 2 {
            return Ok(()); // 旧口径：守恒由解释函数保证（text ≡ total）
        }
        let (t, i, a, v) = self.effective_modality_tokens();
        let sum = t.saturating_add(i).saturating_add(a).saturating_add(v);
        if sum == self.total_tokens {
            Ok(())
        } else {
            Err(format!(
                "modality stats conservation violated: total_tokens={total} != \
                 text({t}) + image({i}) + audio({a}) + video({v}) = {sum} \
                 (a counting bug in the packer/encoder statistics path dropped tokens)",
                total = self.total_tokens,
            ))
        }
    }
}

/// 大脑：数据集清单 (Manifest)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub version: String,
    pub format: StorageFormat,
    pub max_seq_len: usize,
    pub global_stats: GlobalStats,
    /// 列 schema（`BTreeMap` 保证序列化顺序稳定 → 构建可复现）。
    pub schema: BTreeMap<String, String>,
    pub shards: Vec<ShardMeta>,
    /// 构建期分词器指纹（§6）。`#[serde(default)]`：旧 index.json 无此字段时为
    /// `None`，读取端与校验逻辑均不受影响（向后兼容）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerFingerprint>,
    /// 多模态 P2-T4（D6）：各模态专属 token id 区间 `[start, end)`，per-modality。
    /// 构建期由 modality encoder 声明 → pipeline 校验与文本 vocab 不相交（冲突硬失败）
    /// → 透出 manifest。旧 manifest 无此字段 → `None`（纯文本数据集，读取端不校验）。
    /// `Vec` 而非 `BTreeMap`：区间按声明顺序透出（构建期已排序去重），可复现。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality_id_ranges: Option<Vec<ModalityIdRange>>,
    /// 多模态 P3a：audio 流独立 shard 序列（`ashard_` 文件族；与 Vlm 流 `shards`
    /// 平行的第二条 batch 序列，R1 路由下 audio 走 GreedyPacker 变长装箱）。
    /// 空 = 无 audio 流（纯文本/图像/视频构建）——旧 manifest 无此键零影响。
    /// `total_batches`/`total_shards` 保持 Vlm 流口径（audio 侧为推导量）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio_shards: Vec<ShardMeta>,
    /// 多模态 P3a（对齐终稿 DP5）：对齐指纹最小充分集（P3a 落最小集：
    /// `W_ms` / `granularity` / `audio{window_token_rate, codec}`；P3b 增 video 子键、
    /// P3c 增 ts_token/text_in_axis 与 modality_id_ranges 快照）。
    /// **不存在 = 对齐未启用**（旧数据集 / 未含 audio 的构建，读取端不 fail-fast）；
    /// 存在但参数自相矛盾（G2 却缺 W_ms 等）→ 读取端 fail-fast（P3c A7）。
    /// `BTreeMap` 保证序列化顺序稳定 → 构建可复现。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align_fingerprint: Option<BTreeMap<String, serde_json::Value>>,
    /// P4：本次构建用到的源格式集合（如 `["jsonl"]` / `["webdataset","parquet"]`；
    /// 去重排序可复现）。空 = 未记录（pt/sft 旧路径不落此字段）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_formats: Vec<String>,
    /// P4：FieldMap 映射摘要（角色 → 字段列表；`timestamp_field` 以单元素列表
    /// 呈现）——直接影响"哪些字节进哪个段"的构建期配置必须入 manifest
    /// （方案锚点 8：同分片 + 不同映射 = 不同数据集）。`None` = 未记录。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_field_map: Option<BTreeMap<String, Vec<String>>>,
}

/// manifest 中透出的模态 id 区间（D6，序列化形态；与 `encoded_sample::IdRange`
/// 同语义但 `kind` 用稳定字符串键，避免 `SegmentKind` 的 serde 演化耦合）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModalityIdRange {
    pub start: i32,
    pub end: i32,
    /// 模态名（`"image"` / `"audio"` / `"video"`），与 `SegmentKind` 一一对应。
    pub kind: String,
}

/// 当前支持读取的 index.json 版本。
/// - "1.0"：历史格式（1 shard = 1 batch，checksum 假值，无 hashes）
/// - "2.0"：本版本（shard 聚合 + 真 sha256 + 真实 total_tokens）
pub const SUPPORTED_MANIFEST_VERSIONS: [&str; 2] = ["1.0", "2.0"];

impl DatasetManifest {
    /// 构造一个新的空白清单
    pub fn new(format: StorageFormat, max_seq_len: usize) -> Self {
        let mut schema = BTreeMap::new();
        // 默认注入 SFT 必备的 Schema
        schema.insert("input_ids".to_string(), "int32".to_string());
        schema.insert("labels".to_string(), "int32".to_string());
        schema.insert("position_ids".to_string(), "int32".to_string());
        schema.insert("cu_seqlens".to_string(), "list<int32>".to_string());

        Self {
            version: "2.0".to_string(),
            format,
            max_seq_len,
            global_stats: GlobalStats::default(),
            schema,
            shards: Vec::new(),
            tokenizer: None,
            modality_id_ranges: None,
            audio_shards: Vec::new(),
            align_fingerprint: None,
            source_formats: Vec::new(),
            source_field_map: None,
        }
    }

    /// 挂上构建期分词器指纹（链式）。`None` 表示不记录（字段整体省略）。
    pub fn with_tokenizer(mut self, fp: Option<TokenizerFingerprint>) -> Self {
        self.tokenizer = fp;
        self
    }

    /// 在写盘流中动态追加一个 Shard 的记录
    pub fn append_shard(&mut self, meta: ShardMeta, tokens_in_shard: u64) {
        self.global_stats.total_batches += meta.num_batches;
        self.global_stats.total_tokens += tokens_in_shard;
        self.global_stats.total_shards += 1;
        self.shards.push(meta);
    }

    /// 结构校验：版本、format、shard 条目自洽性（借鉴 streaming 的
    /// 「报错 + 可操作建议」防御风格）。
    pub fn validate(&self) -> std::io::Result<()> {
        if !SUPPORTED_MANIFEST_VERSIONS.contains(&self.version.as_str()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unsupported index.json version: {} (supported: {:?}). Rebuild the dataset with the current megadata version.",
                    self.version, SUPPORTED_MANIFEST_VERSIONS
                ),
            ));
        }
        // P3a：两流均空才算空数据集（纯 audio 数据集只有 audio_shards，合法）
        if self.shards.is_empty() && self.audio_shards.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "index.json contains no shard records (neither Vlm/PT stream shards nor audio stream \
                 shards); the dataset may be an empty build or corrupted.",
            ));
        }
        for shard in &self.shards {
            if shard.filename.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "filename of shard {} is empty; the manifest is corrupted.",
                        shard.shard_id
                    ),
                ));
            }
            if shard.num_batches == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "num_batches of shard {} is 0; the manifest is corrupted.",
                        shard.shard_id
                    ),
                ));
            }
        }
        // 交叉校验统计量与 shard 列表（很便宜，但能兜住"manifest 被截断/手改"导致的
        // "看起来正常但少数据"：total_batches 是 LR scheduler 计算总步数的依据）。
        // 0 视为"未记录"（兼容旧 manifest），跳过校验。
        let sum_batches: usize = self.shards.iter().map(|s| s.num_batches).sum();
        if self.global_stats.total_batches != 0 && self.global_stats.total_batches != sum_batches {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "index.json is self-contradictory: global_stats.total_batches={} but the sum of shard num_batches={}.\
                     The dataset may be truncated or tampered with; please rebuild it.",
                    self.global_stats.total_batches, sum_batches
                ),
            ));
        }
        if self.global_stats.total_shards != 0
            && self.global_stats.total_shards != self.shards.len()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "index.json is self-contradictory: global_stats.total_shards={} but the shards entry count={}.\
                     The dataset may be truncated or tampered with; please rebuild it.",
                    self.global_stats.total_shards,
                    self.shards.len()
                ),
            ));
        }
        // P3a：audio 流独立 shard 序列同规格自校验（filename 非空 + num_batches>0；
        // total_batches/total_shards 是 Vlm 流口径，audio 侧不做 Σ 交叉校验）。
        for shard in &self.audio_shards {
            if shard.filename.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "filename of audio shard {} is empty; the manifest is corrupted.",
                        shard.shard_id
                    ),
                ));
            }
            if shard.num_batches == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "num_batches of audio shard {} is 0; the manifest is corrupted.",
                        shard.shard_id
                    ),
                ));
            }
        }
        // F2 截断审计：加载时顺带可见——让"数据损失"可审计，而不只是构建当时告警过
        if self.global_stats.truncated_docs > 0 {
            tracing::warn!(
                "This dataset was built with tail truncation: {} doc(s) exceeded max_seq_len \
                 ({} tokens dropped, see global_stats.truncated_docs / truncated_tokens in index.json).",
                self.global_stats.truncated_docs,
                self.global_stats.truncated_tokens
            );
        }
        // PT 切分审计：加载时顺带可见——长文档被分段会让 varlen 训练悄悄丢掉
        // 跨 batch 的上下文，这是"数据没丢但语义变了"的隐形损失，值得在加载期就说。
        if self.global_stats.split_docs > 0 {
            tracing::warn!(
                "This PT dataset contains {} document(s) longer than one batch ({} tokens total, \
                 see global_stats.split_docs / split_tokens in index.json). They were NOT dropped, but were \
                 split across batches: with FlashAttention varlen (cu_seqlens), each segment becomes an \
                 independent sequence starting at position 0, so cross-batch context inside one document is LOST. \
                 Consider raising max_seq_len so most documents fit in a single batch.",
                self.global_stats.split_docs,
                self.global_stats.split_tokens
            );
        }
        // 多模态 P0-T5：跳过样本审计——让"读取期整条样本被丢"在加载期可见。
        // 与 truncated（token 级）/ split（段级）并列的"样本级"损失维度：
        // 对应 sample_id 出现空缺，加载端按 id 寻址时需感知。禁止静默丢样本
        // 的兜底——若此处非零却没在构建期告警，说明计数链路漏报。
        let total_skipped: u64 = self.global_stats.skipped_samples_by_class.values().sum();
        if total_skipped > 0 || self.global_stats.fail_fast_triggered > 0 {
            tracing::warn!(
                "This multimodal dataset was built with {} skipped sample(s) across {} error class(es){}: \
                 {:?}. The corresponding sample_ids have gaps; samples were dropped during reading per the \
                 failure policy (see global_stats.skipped_samples_by_class in index.json).",
                total_skipped,
                self.global_stats.skipped_samples_by_class.len(),
                if self.global_stats.fail_fast_triggered > 0 {
                    format!(
                        ", fail_fast triggered {} time(s)",
                        self.global_stats.fail_fast_triggered
                    )
                } else {
                    String::new()
                },
                self.global_stats.skipped_samples_by_class,
            );
        }
        Ok(())
    }

    /// 落盘为 index.json（原子写：.tmp + rename，与云端缓存同一防半截文件语义）
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let tmp_path = std::path::Path::new(path).with_extension("json.tmp");
        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, self)?;
            writer.flush()?;
            // fsync 失败（磁盘写满/EDQUOTA）必须报错：index.json 代表"构建完成"的
            // 语义，静默吞掉落盘失败会在 rename 后留下一个崩溃后丢失/半截的
            // manifest——构建失败远好于静默产出不可靠数据集。
            writer.get_ref().sync_all().map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "failed to fsync index.json tmp file {}: {e} (disk full?); aborting to avoid publishing an unflushed manifest",
                        tmp_path.display()
                    ),
                )
            })?;
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// 从 index.json 加载（含版本/结构校验）。本地路径的公开读取入口
    #[allow(dead_code)]
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let manifest: Self = serde_json::from_reader(reader)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// 智能加载：支持本地或 S3/GCS 等云端路径（含版本/结构校验）
    pub fn load_smart(rt: &tokio::runtime::Runtime, path_or_url: &str) -> std::io::Result<Self> {
        let json_str = crate::io::CloudIoManager::read_manifest_to_string(rt, path_or_url)?;
        let manifest: Self = serde_json::from_str(&json_str)?;
        manifest.validate()?;
        Ok(manifest)
    }

    // =========================================================
    // 🌟 核心算法：分布式哈希路由 (Distributed Routing)
    // =========================================================
    /// 当 DataLoader 在多机多卡启动时调用此方法。
    /// 给定全局 rank 和 world_size，极其均匀且互不重叠地分配 Shard。
    ///
    /// 注意：仅保留 round-robin 作为旧版兼容入口；弹性扩缩容分区见
    /// runtime/sampler.rs 的 DistributedSampler（relaxed / canonical nodes）。
    #[allow(dead_code)] // 旧版 round-robin 公开 API，保持兼容
    pub fn get_shards_for_rank(&self, rank: usize, world_size: usize) -> Vec<ShardMeta> {
        self.shards
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % world_size == rank)
            .map(|(_, shard)| shard.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with(num_batches: &[usize]) -> DatasetManifest {
        let mut m = DatasetManifest::new(StorageFormat::Memmap, 128);
        for (i, &nb) in num_batches.iter().enumerate() {
            m.append_shard(
                ShardMeta {
                    shard_id: i,
                    filename: format!("shard_{i:05}.bin"),
                    num_batches: nb,
                    size_bytes: 1,
                    hashes: BTreeMap::new(),
                    align_aud_range: None,
                },
                nb as u64 * 128,
            );
        }
        m
    }

    /// validate 交叉校验统计量：能抓出"被截断/手改"的 manifest（看起来正常但少数据）。
    #[test]
    fn test_validate_cross_checks_global_stats() {
        let m = manifest_with(&[3, 4, 5]);
        assert_eq!(m.global_stats.total_batches, 12);
        assert_eq!(m.global_stats.total_shards, 3);
        assert!(m.validate().is_ok(), "self-consistent manifest should pass");

        let mut bad_batches = manifest_with(&[3, 4, 5]);
        bad_batches.global_stats.total_batches = 11; // 手改/截断
        assert!(
            bad_batches.validate().is_err(),
            "total_batches not matching the shard sum should raise an error"
        );

        let mut bad_shards = manifest_with(&[3, 4, 5]);
        bad_shards.global_stats.total_shards = 2; // 手改/截断
        assert!(
            bad_shards.validate().is_err(),
            "total_shards not matching the entry count should raise an error"
        );

        // 0 视为"未记录"（兼容旧 manifest / 手写 manifest）→ 跳过校验
        let mut legacy = manifest_with(&[3, 4, 5]);
        legacy.global_stats.total_batches = 0;
        legacy.global_stats.total_shards = 0;
        assert!(
            legacy.validate().is_ok(),
            "zero stats should be skipped (backward compatibility)"
        );
    }

    /// 向后兼容：旧 index.json 没有 `tokenizer` 字段时必须能反序列化（§6 / D7）。
    #[test]
    fn test_manifest_without_tokenizer_field_deserializes() {
        let legacy_json = r#"{
            "version": "2.0",
            "format": "Memmap",
            "max_seq_len": 128,
            "global_stats": {"total_tokens": 10, "total_batches": 1, "total_shards": 1, "batch_size": 1},
            "schema": {},
            "shards": [{"shard_id": 0, "filename": "shard_00000.bin", "num_batches": 1, "size_bytes": 4}]
        }"#;
        let m: DatasetManifest = serde_json::from_str(legacy_json)
            .expect("legacy manifest without a tokenizer field must deserialize");
        assert!(m.tokenizer.is_none());
        assert!(m.validate().is_ok());
    }

    /// 指纹写出：有值时写 `tokenizer` 块，为 None 时整体省略（保持旧格式）。
    #[test]
    fn test_tokenizer_fingerprint_roundtrip_and_omission() {
        let m = DatasetManifest::new(StorageFormat::Memmap, 128).with_tokenizer(Some(
            TokenizerFingerprint {
                backend: "python".to_string(),
                identity: Some("/path/to/tokenizer.json".to_string()),
                vocab_size: Some(151936),
            },
        ));
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"tokenizer\""));
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();
        let fp = back.tokenizer.expect("fingerprint should round-trip");
        assert_eq!(fp.backend, "python");
        assert_eq!(fp.vocab_size, Some(151936));

        let plain = DatasetManifest::new(StorageFormat::Memmap, 128);
        assert!(!serde_json::to_string(&plain).unwrap().contains("tokenizer"));
    }

    /// **构建可复现**：同输入的两次构建，index.json 必须逐字节一致。
    ///
    /// `HashMap` 的 `RandomState` 逐实例随机 ⇒ 序列化顺序不稳定，两次构建的
    /// index.json 会不同（不可复现、也没法用文件哈希做产物比对）。`schema` /
    /// `hashes` 因此用 `BTreeMap`。
    #[test]
    fn test_manifest_serialization_is_deterministic() {
        let a = DatasetManifest::new(StorageFormat::Memmap, 128);
        let b = DatasetManifest::new(StorageFormat::Memmap, 128);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "two manifests built with identical parameters must serialize identically (schema uses BTreeMap for stable ordering)"
        );
    }

    /// F2 回归：截断统计字段的序列化语义——为 0 时整体省略（旧格式 byte 级
    /// 兼容），> 0 时写入；旧 manifest 缺字段时反序列化为 0。
    #[test]
    fn test_truncation_stats_serialization_semantics() {
        // 未截断：字段整体省略（保持既有产物格式稳定）
        let clean = DatasetManifest::new(StorageFormat::ArrowIpc, 128);
        let json = serde_json::to_string(&clean).unwrap();
        assert!(
            !json.contains("truncated_docs"),
            "zero counts must be omitted"
        );
        assert!(
            !json.contains("truncated_tokens"),
            "zero counts must be omitted"
        );

        // 有截断：写入字段
        let mut truncated = DatasetManifest::new(StorageFormat::ArrowIpc, 128);
        truncated.global_stats.truncated_docs = 3;
        truncated.global_stats.truncated_tokens = 1234;
        let json = serde_json::to_string(&truncated).unwrap();
        assert!(json.contains("\"truncated_docs\":3"));
        assert!(json.contains("\"truncated_tokens\":1234"));
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.global_stats.truncated_docs, 3);
        assert_eq!(back.global_stats.truncated_tokens, 1234);

        // 旧 manifest（无该字段）→ 默认 0，向后兼容
        let legacy: DatasetManifest =
            serde_json::from_str(&serde_json::to_string(&clean).unwrap()).unwrap();
        assert_eq!(legacy.global_stats.truncated_docs, 0);
        assert_eq!(legacy.global_stats.truncated_tokens, 0);
    }

    /// ★ 评审 §3.5（F2-PT）回归：切分统计字段的序列化语义与 `truncated_*` 完全同构——
    ///   为 0 时省略（旧产物 byte 级稳定）、> 0 时写入、旧 manifest 缺字段时反序列化为 0。
    ///
    ///   两者**必须分开**建模：SFT 超长文档被**丢弃**（数据损失），
    ///   PT 超长文档被**分段**（数据不丢但 varlen 下丢跨 batch 上下文）。
    ///   若将来有人图省事把 split_* 并入 truncated_*，本测试的作用域就是提醒他：
    ///   字段语义不同，排查方向也不同。
    #[test]
    fn test_split_stats_serialization_semantics() {
        let clean = DatasetManifest::new(StorageFormat::Memmap, 128);
        let json = serde_json::to_string(&clean).unwrap();
        assert!(
            !json.contains("split_docs"),
            "zero split counts must be omitted"
        );
        assert!(
            !json.contains("split_tokens"),
            "zero split counts must be omitted"
        );

        let mut split = DatasetManifest::new(StorageFormat::Memmap, 128);
        split.global_stats.split_docs = 7;
        split.global_stats.split_tokens = 98765;
        let json = serde_json::to_string(&split).unwrap();
        assert!(json.contains("\"split_docs\":7"));
        assert!(json.contains("\"split_tokens\":98765"));
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.global_stats.split_docs, 7);
        assert_eq!(back.global_stats.split_tokens, 98765);

        // 旧 manifest（无该字段）→ 0，且**不与** truncated_* 混淆
        let legacy: DatasetManifest =
            serde_json::from_str(&serde_json::to_string(&clean).unwrap()).unwrap();
        assert_eq!(legacy.global_stats.split_docs, 0);
        assert_eq!(legacy.global_stats.split_tokens, 0);
        assert_eq!(legacy.global_stats.truncated_docs, 0);
        assert_eq!(legacy.global_stats.truncated_tokens, 0);
    }

    /// ★ 多模态 P0-T5 回归：跳过样本统计字段的序列化语义——
    ///   - `skipped_samples_by_class`：空表整体省略（旧产物 byte 级稳定）；
    ///     非空时按 ErrorClass 稳定键写入，`BTreeMap` 保证顺序可复现；
    ///   - `fail_fast_triggered`：0 省略、>0 写入；
    ///   - 旧 manifest 缺字段时反序列化为默认（空表 / 0），向后兼容。
    ///
    ///   与 truncated_*/split_* 同构但**语义不同**：truncated 是 token 级尾部丢弃
    ///   （SFT 文本超长），split 是段级切分（PT 长文档跨 batch），此处是样本级
    ///   整条跳过（多模态图像下载/解码失败），对应 sample_id 空缺。三者**不可合并**：
    ///   排查方向分别为"截断策略"、"max_seq_len"、"读取失败域"。
    #[test]
    fn test_skipped_stats_serialization_semantics() {
        // 无跳过：两个字段整体省略（保持既有产物格式稳定）
        let clean = DatasetManifest::new(StorageFormat::ArrowIpc, 128);
        let json = serde_json::to_string(&clean).unwrap();
        assert!(
            !json.contains("skipped_samples_by_class"),
            "empty map must be omitted for byte-level backward compat"
        );
        assert!(
            !json.contains("fail_fast_triggered"),
            "zero fail_fast must be omitted"
        );

        // SkipAndCount：有跳过分类 → 写入 map；fail_fast=0 仍省略
        let mut skipped = DatasetManifest::new(StorageFormat::ArrowIpc, 128);
        skipped
            .global_stats
            .skipped_samples_by_class
            .insert("source_not_found".to_string(), 2);
        skipped
            .global_stats
            .skipped_samples_by_class
            .insert("source_corrupt".to_string(), 1);
        let json = serde_json::to_string(&skipped).unwrap();
        assert!(json.contains("\"skipped_samples_by_class\""));
        // BTreeMap → 键有序，source_corrupt 在 source_not_found 之前（字典序）
        assert!(json.contains("\"source_corrupt\":1,\"source_not_found\":2"));
        assert!(
            !json.contains("fail_fast_triggered"),
            "zero fail_fast must still be omitted when skip map is non-empty"
        );
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.global_stats
                .skipped_samples_by_class
                .get("source_not_found"),
            Some(&2)
        );
        assert_eq!(
            back.global_stats
                .skipped_samples_by_class
                .get("source_corrupt"),
            Some(&1)
        );
        assert_eq!(back.global_stats.fail_fast_triggered, 0);

        // FailFast 触发：写 fail_fast_triggered；skip map 可同时为空（仅中止，无跳过）
        let mut ff = DatasetManifest::new(StorageFormat::ArrowIpc, 128);
        ff.global_stats.fail_fast_triggered = 1;
        let json = serde_json::to_string(&ff).unwrap();
        assert!(json.contains("\"fail_fast_triggered\":1"));
        assert!(
            !json.contains("skipped_samples_by_class"),
            "empty skip map must be omitted even when fail_fast is non-zero"
        );

        // 旧 manifest（无这些字段）→ 默认空表 / 0，向后兼容
        let legacy: DatasetManifest =
            serde_json::from_str(&serde_json::to_string(&clean).unwrap()).unwrap();
        assert!(legacy.global_stats.skipped_samples_by_class.is_empty());
        assert_eq!(legacy.global_stats.fail_fast_triggered, 0);
        // 且不与 truncated_/split_ 混淆
        assert_eq!(legacy.global_stats.truncated_docs, 0);
        assert_eq!(legacy.global_stats.split_docs, 0);

        // 构建可复现：同输入两次构建的序列化一致（BTreeMap 顺序稳定）
        let a = serde_json::to_string(&skipped).unwrap();
        let b = serde_json::to_string(&skipped).unwrap();
        assert_eq!(
            a, b,
            "skip map uses BTreeMap for deterministic serialization"
        );
    }

    /// ★ 多模态 P1-T3 回归：模态拆分统计的序列化与解释语义（sidecar 格式冻结 §8）——
    ///   - 零值字段整体省略（旧产物 byte 级稳定）：文本构建只多
    ///     `stats_version:2` + `total_text_tokens` 两个键；
    ///   - 守恒不变式 `total_tokens == Σ 模态`：stats_version>=2 时断言成立，
    ///     违反被 `check_modality_conservation` 捕获；
    ///   - 旧 manifest（无 stats_version）读取解释：text = total_tokens、
    ///     其余 0（`effective_modality_tokens`），**不静默错解**（D3）。
    #[test]
    fn test_modality_stats_serialization_semantics() {
        // 文本构建（pipeline 落盘路径）：stats_version=2 + text=total，模态字段省略
        let mut text = DatasetManifest::new(StorageFormat::Memmap, 128);
        text.global_stats.total_tokens = 1000;
        text.global_stats.stats_version = 2;
        text.global_stats.total_text_tokens = 1000;
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"stats_version\":2"));
        assert!(json.contains("\"total_text_tokens\":1000"));
        assert!(
            !json.contains("total_image_tokens"),
            "zero modal tokens must be omitted"
        );
        assert!(
            !json.contains("total_images"),
            "zero image count must be omitted"
        );
        assert!(
            !json.contains("total_audio_seconds"),
            "zero f64 seconds must be omitted"
        );
        // 守恒：text 构建 total == text + 0 + 0 + 0
        assert!(text.global_stats.check_modality_conservation().is_ok());

        // 多模态构建：守恒成立 / 破坏均被正确判定
        let mut mm = DatasetManifest::new(StorageFormat::Memmap, 128);
        mm.global_stats.total_tokens = 1000;
        mm.global_stats.stats_version = 2;
        mm.global_stats.total_text_tokens = 700;
        mm.global_stats.total_image_tokens = 300;
        mm.global_stats.total_images = 42;
        mm.global_stats.total_video_seconds = 0.0;
        assert!(mm.global_stats.check_modality_conservation().is_ok());
        let json = serde_json::to_string(&mm).unwrap();
        assert!(json.contains("\"total_images\":42"));
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.global_stats.total_image_tokens, 300);
        assert_eq!(back.global_stats.total_images, 42);

        mm.global_stats.total_image_tokens = 301; // 701+300 > 1000 破坏守恒
        let err = mm.global_stats.check_modality_conservation().unwrap_err();
        assert!(err.contains("conservation violated"), "err: {err}");
        mm.global_stats.total_image_tokens = 299; // 700+299 < 1000 亦破坏
        assert!(mm.global_stats.check_modality_conservation().is_err());

        // 旧 manifest（无 stats_version = 0）：解释为 text=total、其余 0；守恒不校验
        let mut legacy = DatasetManifest::new(StorageFormat::Memmap, 128);
        legacy.global_stats.total_tokens = 555; // stats_version 缺省 0
        assert_eq!(
            legacy.global_stats.effective_modality_tokens(),
            (555, 0, 0, 0)
        );
        assert!(legacy.global_stats.check_modality_conservation().is_ok());
        // 序列化不含新键（byte 级兼容：旧格式字段全部省略）
        let json = serde_json::to_string(&legacy).unwrap();
        assert!(!json.contains("stats_version"));
        assert!(!json.contains("total_text_tokens"));

        // stats_version>=2 的解释函数直读拆分字段（不再回退 text=total）
        let eff = mm.global_stats.effective_modality_tokens();
        assert_eq!(eff, (700, 299, 0, 0));
    }

    /// P3a：`audio_shards` / `align_fingerprint` 的省略（空/None 不出键）与
    /// roundtrip；旧 manifest（无两字段）反序列化零影响。
    #[test]
    fn test_audio_shards_and_align_fingerprint_omission_roundtrip() {
        let plain = DatasetManifest::new(StorageFormat::Memmap, 128);
        let json = serde_json::to_string(&plain).unwrap();
        assert!(
            !json.contains("audio_shards"),
            "empty audio_shards must be omitted"
        );
        assert!(
            !json.contains("align_fingerprint"),
            "None fingerprint must be omitted"
        );

        let mut m = DatasetManifest::new(StorageFormat::Memmap, 128);
        m.audio_shards.push(ShardMeta {
            shard_id: 0,
            filename: "ashard_00000.aud_bin".to_string(),
            num_batches: 3,
            size_bytes: 4096,
            hashes: BTreeMap::from([
                ("sha256:aud_bin".to_string(), "aa".to_string()),
                ("sha256:col_aud_blob".to_string(), "bb".to_string()),
            ]),
            align_aud_range: None,
        });
        let mut fp = BTreeMap::new();
        fp.insert("W_ms".to_string(), serde_json::json!(2000));
        fp.insert("granularity".to_string(), serde_json::json!("G2"));
        fp.insert(
            "audio".to_string(),
            serde_json::json!({"window_token_rate": 50, "codec": "flac"}),
        );
        m.align_fingerprint = Some(fp);
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("ashard_00000.aud_bin"));
        assert!(json.contains("\"W_ms\":2000"));
        assert!(json.contains("\"window_token_rate\":50"));
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.audio_shards.len(), 1);
        assert_eq!(back.audio_shards[0].num_batches, 3);
        assert_eq!(
            back.audio_shards[0].hashes.get("sha256:aud_bin").unwrap(),
            "aa"
        );
        assert_eq!(
            back.align_fingerprint
                .as_ref()
                .unwrap()
                .get("W_ms")
                .unwrap(),
            &serde_json::json!(2000)
        );
        assert!(back.validate().is_ok());

        // audio_shards 结构校验：空 filename / 0 batches → Err
        let mut bad = DatasetManifest::new(StorageFormat::Memmap, 128);
        bad.audio_shards.push(ShardMeta {
            shard_id: 0,
            filename: String::new(),
            num_batches: 1,
            size_bytes: 1,
            hashes: BTreeMap::new(),
            align_aud_range: None,
        });
        assert!(
            bad.validate().is_err(),
            "empty audio shard filename must fail"
        );
        let mut bad2 = DatasetManifest::new(StorageFormat::Memmap, 128);
        bad2.audio_shards.push(ShardMeta {
            shard_id: 0,
            filename: "ashard_00000.aud_bin".to_string(),
            num_batches: 0,
            size_bytes: 1,
            hashes: BTreeMap::new(),
            align_aud_range: None,
        });
        assert!(
            bad2.validate().is_err(),
            "zero audio shard num_batches must fail"
        );

        // 旧 manifest（无 audio_shards / align_fingerprint 键）反序列化兼容
        let legacy = r#"{
            "version": "2.0", "format": "Memmap", "max_seq_len": 128,
            "global_stats": {"total_tokens": 10, "total_batches": 1, "total_shards": 1, "batch_size": 1},
            "schema": {},
            "shards": [{"shard_id": 0, "filename": "shard_00000.bin", "num_batches": 1, "size_bytes": 4}]
        }"#;
        let m: DatasetManifest = serde_json::from_str(legacy).unwrap();
        assert!(m.audio_shards.is_empty());
        assert!(m.align_fingerprint.is_none());
        assert!(m.validate().is_ok());
    }
}

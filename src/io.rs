use fs2::FileExt;
use object_store::aws::{AmazonS3, AmazonS3Builder, AmazonS3ConfigKey};
use object_store::{ObjectStoreExt, path::Path as ObjectPath};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};
use url::Url;

/// 下载失败重试次数（首次 + 重试）
const DOWNLOAD_ATTEMPTS: u32 = 3;
/// 重试退避基期：第 n 次重试前 sleep `base * 2^n + U(0, base)`（指数退避 + 抖动，
/// 防止千卡同时重试打爆存储后端，借鉴 streaming 的 retry 语义）。
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(500);

/// P2-6（评审修复）：HTTP 连接超时。对端接受连接后不返回响应头/体时，旧
/// `reqwest::get` 无任何超时会无限等待（重试与 EOF/panic 上报均无法生效）。
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// P2-6：单次 body 读取超时（chunk 间停滞即失败；不限制持续但缓慢的传输总时长）。
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// P2-6：进程级 HTTP client（连接/读取超时显式配置）。
fn http_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .read_timeout(HTTP_READ_TIMEOUT)
            .build()
            .expect("failed to build reqwest HTTP client with timeouts")
    })
}

/// 根据 s3:// URL 构造一个已经注入好 endpoint/region/凭据 的 S3 客户端，
/// 避免走默认凭据链访问 EC2 IMDS (169.254.169.254)。
fn build_s3_store(parsed_url: &Url) -> std::io::Result<AmazonS3> {
    let bucket = parsed_url
        .host_str()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "S3 URL is missing a bucket",
            )
        })?
        .to_string();

    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket).with_region(
        std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string()),
    );

    // 自定义 endpoint (Moto / MinIO / LocalStack)
    if let Ok(endpoint) =
        std::env::var("AWS_ENDPOINT_URL").or_else(|_| std::env::var("AWS_ENDPOINT"))
        && !endpoint.is_empty()
    {
        builder = builder.with_endpoint(endpoint);
    }
    // 允许 HTTP（默认只允许 HTTPS）
    if let Ok(allow) = std::env::var("AWS_ALLOW_HTTP")
        && (allow == "true" || allow == "1")
    {
        builder = builder.with_allow_http(true);
    }
    // 路径式请求（兼容 Moto / MinIO）
    builder = builder.with_virtual_hosted_style_request(false);

    // 凭据：优先用环境变量；没有时跳过签名（Moto 默认放行匿名请求）
    let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok();
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
    match (access_key, secret_key) {
        (Some(ak), Some(sk)) if !ak.is_empty() && !sk.is_empty() => {
            builder = builder
                .with_config(AmazonS3ConfigKey::AccessKeyId, ak)
                .with_config(AmazonS3ConfigKey::SecretAccessKey, sk);
        }
        _ => {
            builder = builder.with_skip_signature(true);
        }
    }

    // 关闭 IMDS 元数据兜底，防止访问 169.254.169.254
    if let Ok(metadata_endpoint) = std::env::var("AWS_METADATA_ENDPOINT")
        && !metadata_endpoint.is_empty()
    {
        builder = builder.with_config(AmazonS3ConfigKey::MetadataEndpoint, metadata_endpoint);
    }

    builder
        .build()
        .map_err(|e| std::io::Error::other(e.to_string()))
}

/// 解析带可选单位后缀的容量字符串（如 "5368709120" / "10G" / "512MB" / "1T"）。
/// "0" 或解析失败返回 None（调用方决定默认值；0 语义为不限制）。
fn parse_size_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (digits, mult) = match s.chars().last()? {
        'K' | 'k' => (&s[..s.len() - 1], 1024u64),
        'M' | 'm' => (&s[..s.len() - 1], 1024 * 1024),
        'G' | 'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        'T' | 't' => (&s[..s.len() - 1], 1024u64 * 1024 * 1024 * 1024),
        'B' | 'b' => match s[..s.len() - 1].chars().last() {
            Some('K' | 'k') => (&s[..s.len() - 2], 1024),
            Some('M' | 'm') => (&s[..s.len() - 2], 1024 * 1024),
            Some('G' | 'g') => (&s[..s.len() - 2], 1024 * 1024 * 1024),
            Some('T' | 't') => (&s[..s.len() - 2], 1024u64 * 1024 * 1024 * 1024),
            _ => (s, 1),
        },
        _ => (s, 1),
    };
    let parsed = digits
        .trim()
        .parse::<u64>()
        .ok()
        .map(|v| v.saturating_mul(mult));
    if parsed.is_none() && !s.chars().all(|c| c.is_ascii_digit()) {
        // 拼错单位（如 "100X"）或非法输入时给出提示：静默 None 会让用户误以为配置已生效。
        // 当前仅用于 MEGADATA_CACHE_LIMIT 等环境变量解析，失败时调用方回退默认值。
        tracing::warn!(
            "invalid size string '{s}': expected digits with an optional unit suffix (K/M/G/T/B); ignoring and falling back to default"
        );
    }
    parsed
}

/// 流式计算文件的 sha256 hex（1MB 缓冲，供构建期写完回读与读取端校验共用）。
pub fn sha256_file_hex(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// 命中缓存时的 mtime 刷新间隔：LRU 粒度（1 分钟）足够，避免每个 batch 读都打一次 utimensat。
const CACHE_TOUCH_INTERVAL: Duration = Duration::from_secs(60);

/// F1：本地读完整性校验模式（环境变量 `MEGADATA_VERIFY_LOCAL`，默认关）。
///
/// 背景：manifest 记录的 sha256 此前只对云端下载生效——本地数据集读路径直接
/// return，位翻转/篡改后的文件照样读出（运行时抽查：743/743 batch 无任何报错）。
/// 本开关让本地读也能校验，默认仍关闭（避免双倍 IO，行为向后兼容）：
/// - `Off`：不设 / `0` / `false`（默认）
/// - `Full`：`1` / `true`，对带 sha256 记录的文件全量校验
/// - `Sample(p)`：`sample:P`（0≤P≤1），按文件**确定性**抽样校验比例 P
#[derive(Debug, Clone, Copy, PartialEq)]
enum LocalVerifyMode {
    Off,
    Full,
    Sample(f64),
}

impl LocalVerifyMode {
    fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false") {
            return Self::Off;
        }
        if s == "1" || s.eq_ignore_ascii_case("true") {
            return Self::Full;
        }
        if let Some(p_str) = s.strip_prefix("sample:") {
            return match p_str.trim().parse::<f64>() {
                Ok(p) if (0.0..=1.0).contains(&p) => Self::Sample(p),
                _ => {
                    tracing::warn!(
                        "invalid MEGADATA_VERIFY_LOCAL value {s:?}: expected `sample:P` with 0<=P<=1; disabling local verification"
                    );
                    Self::Off
                }
            };
        }
        tracing::warn!(
            "invalid MEGADATA_VERIFY_LOCAL value {s:?}: expected `1` or `sample:P`; disabling local verification"
        );
        Self::Off
    }
}

/// 命中缓存时"触摸"文件（把 mtime 推到 now），供 LRU 驱逐排序使用。
///
/// 为什么不用 atime：Linux 默认 relatime（甚至 noatime）挂载下反复读取**不会**刷新 atime，
/// "最久未访问"判据失真，可能把刚用过的 shard 当冷数据删掉（复核问题 6 实测确认）。
/// 失败时静默放弃（LRU 退化为按"下载时间"排序，不影响正确性）。
fn touch_cache_file(path: &Path) {
    let Ok(meta) = fs::metadata(path) else { return };
    let Ok(modified) = meta.modified() else {
        return;
    };
    if SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default()
        < CACHE_TOUCH_INTERVAL
    {
        return; // 刚触摸过，不必再写
    }
    if let Ok(f) = fs::OpenOptions::new().write(true).open(path) {
        let _ = f.set_times(fs::FileTimes::new().set_modified(SystemTime::now()));
    }
}

/// 智能路径解析与下载器
/// 屏蔽本地与云端的差异，按需将云端文件缓存到本地。
///
/// 生产化控制面（借鉴 streaming）：
/// - **原子写**：下载内容先写 `<target>.tmp.<pid>` 再 `rename`，进程被 kill /
///   磁盘写满都不会在最终路径留下半截文件；
/// - **校验**：manifest 带 sha256 时，下载后校验，不匹配删缓存重下；
///   `MEGADATA_VALIDATE_CACHE=1` 时缓存命中也校验（默认关，避免双倍 IO）；
/// - **重试**：指数退避 + 随机抖动，一次网络抖动不再直接失败；
/// - **单飞**：同节点多 rank 并发下载同一 shard 时，用 `<target>.lock` 文件锁去重，
///   只有一个进程真正下载，其余等锁后直接命中缓存；
/// - **有界缓存**：`MEGADATA_CACHE_LIMIT`（如 "512G"）超限后按最久未被触摸（mtime）
///   驱逐，`MEGADATA_CACHE_DIR` 可自定义缓存目录。
pub struct CloudIoManager {
    cache_dir: PathBuf,
    cache_limit_bytes: u64,
    validate_on_hit: bool,
    /// F1：本地读校验模式（`MEGADATA_VERIFY_LOCAL`，构造期解析一次）
    local_verify: LocalVerifyMode,
    /// F1：本地校验结果缓存 path → (size, mtime)：同一文件只 hash 一次，运行期零开销
    local_verified: Mutex<HashMap<PathBuf, (u64, SystemTime)>>,
}

impl CloudIoManager {
    pub fn new() -> Self {
        // 默认将缓存放在 /tmp/megadata_cache 下
        let cache_dir = std::env::var("MEGADATA_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("megadata_cache"));
        fs::create_dir_all(&cache_dir).unwrap_or_default();
        let cache_limit_bytes = std::env::var("MEGADATA_CACHE_LIMIT")
            .ok()
            .and_then(|s| parse_size_bytes(&s))
            .unwrap_or(256 * 1024 * 1024 * 1024); // 默认 256 GiB
        let validate_on_hit = std::env::var("MEGADATA_VALIDATE_CACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // ★ 评审 §3.4：`.tmp.<pid>` 孤儿清扫提前到构造期触发（进程内一次）。
        //   旧位置在 `enforce_cache_limit` 顶部，而后者**只在云端下载成功后**被
        //   调用——纯本地数据集 / 缓存全命中的工作负载永远走不到那里，孤儿文件
        //   （进程被 kill -9 / OOM killer 带走时留下）会无限期堆积。构造期目录
        //   必然已存在（上面 create_dir_all），且每个 reader 构造都会经过，
        //   两类工作负载就都能覆盖到。
        let mgr = Self {
            cache_dir,
            cache_limit_bytes,
            validate_on_hit,
            local_verify: LocalVerifyMode::parse(
                &std::env::var("MEGADATA_VERIFY_LOCAL").unwrap_or_default(),
            ),
            local_verified: Mutex::new(HashMap::new()),
        };
        mgr.sweep_orphan_tmp_files_once();
        mgr
    }

    /// F1：本地读的可选 sha256 校验（云端下载路径原本就有强制校验，此处只补
    /// 本地分支——本地数据集此前完全不校验，位翻转/篡改后照常读出）。
    ///
    /// - 默认不校验（`LocalVerifyMode::Off`），行为与旧版一致，不产生双倍 IO；
    /// - `MEGADATA_VERIFY_LOCAL=1` 全量校验、`=sample:P` 按文件确定性抽样
    ///   （同一路径每次决定一致，部分校验也可复现）；
    /// - 校验结果按 (size, mtime) 缓存：同一文件只 hash 一次，之后运行期零开销。
    ///   首次校验发生在首个 batch 读取时（命中 page cache，数百 MB/s 量级）；
    /// - mismatch 直接报错：本地文件没有"删缓存重下"的自愈手段，静默读坏数据
    ///   远比失败危险。
    fn verify_local_file(&self, path: &Path, expected_sha256: Option<&str>) -> std::io::Result<()> {
        let Some(expected) = expected_sha256 else {
            return Ok(());
        };
        let sampled = match self.local_verify {
            LocalVerifyMode::Off => return Ok(()),
            LocalVerifyMode::Full => true,
            LocalVerifyMode::Sample(p) => {
                // 按路径 hash 确定性抽样：p=1.0 全量、p=0.0 不校验，中间值可复现
                let mut h = Sha256::new();
                h.update(path.to_string_lossy().as_bytes());
                let x = u64::from_le_bytes(h.finalize()[..8].try_into().unwrap());
                (x as f64 / u64::MAX as f64) < p
            }
        };
        if !sampled {
            return Ok(());
        }

        let Ok(meta) = fs::metadata(path) else {
            // 文件不存在/元数据拿不到：留给真正的读路径报错（更贴近调用语义）
            return Ok(());
        };
        let stamp = (
            meta.len(),
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        );
        // 已校验过且文件未变（size+mtime 相同）。
        //
        // ★ P0（20261007 评审 §3.1）：这里原本是 `.lock().unwrap()`。本函数经
        //   CloudIoManager 被 SyncInnerReader 带着 move 进 stream.rs 的预取线程，
        //   Mutex 一旦中毒（任何持锁线程 panic），每次本地 shard 校验都会 panic
        //   → 落进"预取线程 panic 被降级成 EOF"的失效链 → 训练数据静默截断。
        //   现在改为毒化安全：毒化时降级为"缓存未命中"，最坏只是多算一次 sha256。
        let already_verified = self
            .local_verified
            .lock()
            .map(|cache| cache.get(path) == Some(&stamp))
            .unwrap_or(false);
        if already_verified {
            return Ok(());
        }
        match sha256_file_hex(path) {
            Ok(actual) if actual == expected => {
                if let Ok(mut cache) = self.local_verified.lock() {
                    cache.insert(path.to_path_buf(), stamp);
                }
                Ok(())
            }
            Ok(actual) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Checksum failure (local): {} (expected sha256={expected}, got {actual}); the file may be corrupted \
                     or tampered with. Re-copy/re-build the dataset, or unset MEGADATA_VERIFY_LOCAL to skip local verification.",
                    path.display()
                ),
            )),
            // ★ P1（20261007 评审 §3.3）：原本是 `Err(_) => Ok(())`，把"无法哈希"
            //   当成"校验通过"。这绕过了整套完整性保证，且触发面比原注释写的
            //   "权限等" 宽得多——EISDIR（路径其实是目录）、EIO（坏扇区）、
            //   EMFILE（fd 耗尽）、安全策略拒绝（SELinux/AppArmor）全被放行。
            //   更糟的是用户拿到的错误是"读不出数据"而不是"文件校验不了"，
            //   **排查方向被系统性带偏**。
            //   现在区分两种情况：NotFound 视为"文件确实不在"（与上面 metadata
            //   失败分支同款，交给读路径报错）；其余一律报错。
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(std::io::Error::new(
                e.kind(),
                format!(
                    "Local integrity check could not be performed: {} (expected sha256={expected}, but hashing the \
                     file failed: {e}). The file is present but unreadable, so its integrity is UNKNOWN — refusing \
                     to silently continue. Fix the underlying cause (permissions / disk error / fd exhaustion), \
                     or unset MEGADATA_VERIFY_LOCAL to skip local verification explicitly.",
                    path.display()
                ),
            )),
        }
    }

    /// 给定基础路径和相对文件名，返回可用于本地直接打开的绝对路径。
    ///
    /// 携带 manifest 中记录的 sha256 做完整性校验（`None` = 该文件未记 hash，如旧数据集
    /// 的 .cu/.idx 旁路）。云端路径：校验失败 → 删缓存重下 → 仍失败则**报错**（数据静默
    /// 损坏不再可能；复核清理时把原来的无校验入口 `ensure_local_file` 删掉了——它以默认
    /// 参数形式存在时容易让新调用点漏掉校验）。
    /// 本地路径：默认信任（零开销）；`MEGADATA_VERIFY_LOCAL` 开启后走
    /// [`Self::verify_local_file`]（F1，mismatch 直接报错）。
    pub fn ensure_local_file_verified(
        &self,
        rt: &tokio::runtime::Runtime,
        base_dir_or_url: &str,
        filename: &str,
        expected_sha256: Option<&str>,
    ) -> std::io::Result<String> {
        // 1. 本地路径处理
        if !base_dir_or_url.starts_with("s3://")
            && !base_dir_or_url.starts_with("gs://")
            && !base_dir_or_url.starts_with("http")
        {
            let full_path = Path::new(base_dir_or_url).join(filename);
            self.verify_local_file(&full_path, expected_sha256)?;
            return Ok(full_path.to_string_lossy().to_string());
        }

        // 1.5 缓存目录兜底：构造期 create_dir_all 之后，/tmp 可能被 tmpwatch /
        //     容器回收清掉。不重建的话下面 File::create(lock_path) 会以 ENOENT
        //     失败，用户看到的是"打不开 shard"而非"缓存目录消失"（长期训练
        //     数天~数周可能命中）。失败也继续走（真正的错误留给后续步骤报）。
        let _ = fs::create_dir_all(&self.cache_dir);

        // 2. 云端路径拼接
        let full_url_str = if base_dir_or_url.ends_with('/') {
            format!("{}{}", base_dir_or_url, filename)
        } else {
            format!("{}/{}", base_dir_or_url, filename)
        };

        let parsed_url = Url::parse(&full_url_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

        // 3. 构造缓存文件名
        //    前缀用 16 hex（64-bit）：8 hex 只有 32-bit 空间，跨数据集同名 shard
        //    （shard_00000.bin）存在碰撞风险（复核问题 8），碰撞会以 "Checksum failure" 形式暴露。
        let mut hasher = Sha256::new();
        hasher.update(full_url_str.as_bytes());
        let hash_hex = hex::encode(hasher.finalize());
        let cache_filename = format!("{}_{}", hash_hex.get(..16).unwrap_or(""), filename);
        let cache_path = self.cache_dir.join(&cache_filename);

        // 缓存命中：默认信任（原子写保证最终路径永远是完整文件）；
        // MEGADATA_VALIDATE_CACHE=1 时也校验，兜底防御磁盘 bit rot / 人为篡改。
        if cache_path.exists()
            && self.validate_on_hit
            && let Some(expected) = expected_sha256
        {
            match sha256_file_hex(&cache_path) {
                Ok(actual) if actual == expected => {
                    touch_cache_file(&cache_path);
                    return Ok(cache_path.to_string_lossy().to_string());
                }
                _ => {
                    tracing::warn!(
                        "⚠️ Cache file failed verification, deleting and re-downloading: {}",
                        cache_path.display()
                    );
                    let _ = fs::remove_file(&cache_path);
                }
            }
        }
        if cache_path.exists() {
            touch_cache_file(&cache_path);
            return Ok(cache_path.to_string_lossy().to_string());
        }

        tracing::info!("☁️ Downloading shard from cloud to cache: {}", full_url_str);

        // 4. 跨进程单飞：同节点多 rank 同时启动时，只有持锁进程下载，
        //    其余进程等锁（flock 随进程死亡自动释放，不会死锁）后直接命中缓存。
        let lock_path = self.cache_dir.join(format!("{cache_filename}.lock"));
        let lock_file = File::create(&lock_path)?;
        lock_file
            .lock_exclusive()
            .map_err(|e| std::io::Error::other(format!("Failed to acquire cache lock: {e}")))?;

        let result = (|| -> std::io::Result<String> {
            // 拿到锁后再查一次缓存：可能别的进程刚下载完
            if cache_path.exists() {
                touch_cache_file(&cache_path);
                return Ok(cache_path.to_string_lossy().to_string());
            }

            // 5. 下载 → 写临时文件 → 校验 → 原子 rename。失败时清理临时文件并退避重试。
            let tmp_path = self
                .cache_dir
                .join(format!("{cache_filename}.tmp.{}", std::process::id()));
            let mut last_err: Option<std::io::Error> = None;
            for attempt in 0..DOWNLOAD_ATTEMPTS {
                if attempt > 0 {
                    backoff_sleep(attempt - 1);
                }
                match self.download_to_atomic(
                    rt,
                    &parsed_url,
                    &full_url_str,
                    &tmp_path,
                    &cache_path,
                    expected_sha256,
                ) {
                    Ok(()) => return Ok(cache_path.to_string_lossy().to_string()),
                    Err(e) => {
                        let _ = fs::remove_file(&tmp_path);
                        tracing::warn!(
                            "Download failed (attempt {}/{}): {} — {}",
                            attempt + 1,
                            DOWNLOAD_ATTEMPTS,
                            full_url_str,
                            e
                        );
                        last_err = Some(e);
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| std::io::Error::other("download failed")))
        })();

        drop(lock_file); // 释放锁（Download 完成即释放，不串行化后续读）
        // ★ 评审 §3.7：成功路径上删除 `.lock` 文件本身。
        //   旧实现只 `drop(lock_file)` 释放 fd，文件留在磁盘上——唯一的删除路径是
        //   `enforce_cache_limit` 的 LRU 清理分支，而那只在缓存**超限**时才跑。
        //   于是长期使用的缓存目录会按 shard 数线性累积空 `.lock` 小文件。
        //   只在成功时删：失败时保留它（便于并发进程继续排队等待同一把锁，
        //   删掉会让其他进程另起一把锁、破坏单飞语义）。
        if result.is_ok() {
            let _ = fs::remove_file(&lock_path);
            self.enforce_cache_limit(&cache_path);
        }
        result
    }

    /// 清扫进程死亡后遗留的 `.tmp.<pid>` 孤儿文件（每个 manager 实例只跑一次）。
    ///
    /// **触发点**：`CloudIoManager::new()`（构造期，每个 reader 构造都会经过）。
    /// ★ 评审 §3.4：旧触发点在 `enforce_cache_limit` 顶部，而后者只在云端下载
    /// 成功后调用——纯本地数据集 / 缓存全命中的负载永远触发不到，孤儿文件无限堆积。
    ///
    /// ★ 评审 §3.7：`.tmp.<pid>` 原子写的中转文件在进程被 `kill -9` / OOM killer
    ///   带走时会成为孤儿（flock 会自动释放，但文件不会消失），长期堆积吃满缓存盘，
    ///   而且永远不会被 `enforce_cache_limit` 清理（它只认最终文件名，不认 `.tmp.*`）。
    ///
    /// 判据用**存活时长**而非"pid 是否存活"：pid 探活需要 `libc::kill(pid, 0)`
    /// （引入 unsafe + 平台差异），而本项目的下载有明确上界
    /// （`DOWNLOAD_ATTEMPTS=3`次 + 有限退避），因此**超过 1 小时未被触碰的
    /// `.tmp.*` 必然是孤儿**。这样完全依赖 std、无 unsafe、跨平台，且绝不会
    /// 误删并发 rank 的在途文件。
    ///
    /// 保守起见阈值给得很宽（1 小时）：误删在途临时文件的代价（一次下载失败重试）
    /// 大于多留一个孤儿文件一段时间的代价。
    fn sweep_orphan_tmp_files_once(&self) {
        static SWEPT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        if SWEPT.get().is_some() {
            return;
        }
        // 进程内首次触发时清扫；多 rank 同机各自进程都会各扫一次（幂等：删不到已删的）
        let _ = SWEPT.set(());
        const ORPHAN_TMP_AGE: std::time::Duration = std::time::Duration::from_secs(3600);
        let Ok(entries) = fs::read_dir(&self.cache_dir) else {
            return;
        };
        let now = std::time::SystemTime::now();
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // 形如 `shard_00000.bin.tmp.12345`：只认 `.tmp.` 后带**纯数字**的
            if !name.contains(".tmp.") {
                continue;
            }
            let Some((_, pid_str)) = name.rsplit_once(".tmp.") else {
                continue;
            };
            if pid_str.parse::<u32>().is_err() {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let Ok(modified) = meta.modified() else {
                continue;
            };
            let age = now.duration_since(modified).unwrap_or_default();
            if age < ORPHAN_TMP_AGE {
                continue; // 可能是在途文件，留着
            }
            if fs::remove_file(&path).is_ok() {
                tracing::warn!(
                    "🧹 Removed an orphaned cache temp file (untouched for {}s): {}",
                    age.as_secs(),
                    path.display()
                );
            }
        }
    }

    /// 下载 → 写 .tmp → （可选）sha256 校验 → 原子 rename 到最终路径。
    ///
    /// **流式写入**（不整文件驻留内存）：shard 可达 128MB，单飞锁下同一节点
    /// 仍有"预取线程 + Mixer 多子流"并发下载多个 shard，整文件缓冲会造成
    /// 内存峰值叠加。reqwest 用 `chunk()` 增量读；object_store 的 `GetResult`
    /// 实现 `tokio::io::AsyncRead`，用 1MB 缓冲分块读。sha256 在写盘后回读计算
    /// （`sha256_file_hex`），与本函数原有的"写后校验"顺序零冲突。
    fn download_to_atomic(
        &self,
        rt: &tokio::runtime::Runtime,
        parsed_url: &Url,
        full_url_str: &str,
        tmp_path: &Path,
        final_path: &Path,
        expected_sha256: Option<&str>,
    ) -> std::io::Result<()> {
        rt.block_on(async {
            // 先写临时文件（进程被 kill 只会留下 .tmp.<pid>，永不污染最终路径）
            let mut file = File::create(tmp_path)?;

            if parsed_url.scheme() == "http" || parsed_url.scheme() == "https" {
                // 普通 HTTP 协议，使用 reqwest 直接获取（chunk-by-chunk，无需 stream feature）
                // P2-6：显式 client（连接/单次读取超时），不再用无超时的 `reqwest::get`。
                let mut response = http_client().get(parsed_url.as_str()).send().await
                    .map_err(|e| std::io::Error::other(format!("HTTP download failed: {e}")))?;
                response.error_for_status_ref()
                    .map_err(|e| std::io::Error::other(format!("HTTP status error: {e}")))?;
                while let Some(chunk) = response.chunk().await
                    .map_err(|e| std::io::Error::other(format!("Failed to read response chunk: {e}")))?
                {
                    file.write_all(&chunk)?;
                }
            } else {
                // S3 / GCS 协议：自行构造 store，避免访问 EC2 IMDS
                let path_str = parsed_url.path().trim_start_matches('/');
                let response = if parsed_url.scheme() == "s3" {
                    let store = build_s3_store(parsed_url)?;
                    let location = ObjectPath::from(path_str);
                    store.get(&location).await
                        .map_err(|e| std::io::Error::other(format!("Object store fetch failed: {e}")))?
                } else {
                    // 其他对象存储（GCS / Azure）继续走 parse_url
                    let (store, location) = object_store::parse_url(parsed_url)
                        .map_err(|e| std::io::Error::other(format!("Failed to parse object store URL: {e}")))?;
                    store.get(&location).await
                        .map_err(|e| std::io::Error::other(format!("Object store fetch failed: {e}")))?
                };
                // GetResult::into_stream 分块搬运（1MB 级 chunk，避免整文件驻留内存）
                use futures_util::StreamExt as _;
                let mut stream = response.into_stream();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk
                        .map_err(|e| std::io::Error::other(format!("Failed to read object stream: {e}")))?;
                    file.write_all(&chunk)?;
                }
            }

            // 尽力落盘：磁盘写满/EDQUOTA 时不能完全静默——rename 后的文件可能
            // 未真正持久化，崩溃后丢失。此处保留错误日志（缓存文件完整性最终
            // 由 sha256 校验兜底），不因 fsync 失败放弃整个下载。
            if let Err(e) = file.sync_all() {
                tracing::error!(
                    "fsync failed on downloaded tmp file {}: {e} (file may not be durable after rename)",
                    tmp_path.display()
                );
            }

            // 校验（解压/使用前校验，借鉴 streaming 的 validate_hash 语义）
            if let Some(expected) = expected_sha256 {
                let actual = sha256_file_hex(tmp_path)?;
                if actual != expected {
                    let _ = fs::remove_file(tmp_path);
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Checksum failure: {full_url_str} (expected sha256={expected}, got {actual})"),
                    ));
                }
            }

            // 同目录原子 rename：要么完整出现，要么不存在
            fs::rename(tmp_path, final_path)?;
            Ok::<(), std::io::Error>(())
        })
    }

    /// 有界缓存：超过 cache_limit 后按最久未访问驱逐（LRU = least recently
    /// accessed，与 streaming 的 _evict_coldest_shard 语义一致）。
    ///
    /// **驱逐依据是 mtime 而非 atime**（复核问题 6）：命中缓存时 `touch_cache_file`
    /// 会刷新 mtime；而 Linux 普遍是 relatime/noatime 挂载，实测"反复读取并不会刷新
    /// atime"，用 atime 会导致 LRU 完全失效。
    ///
    /// 保护刚下载的文件与 .lock 文件。
    ///
    /// ★ 评审 §3.4：`.tmp.<pid>` 孤儿清扫**不在**本函数触发（旧实现在此调用）——
    ///   本函数只在云端下载成功后跑，纯本地 / 缓存全命中走不到；已前移到
    ///   `CloudIoManager::new()`（构造期一次，覆盖全部工作负载）。
    fn enforce_cache_limit(&self, just_added: &Path) {
        if self.cache_limit_bytes == 0 {
            return;
        }
        let Ok(entries) = fs::read_dir(&self.cache_dir) else {
            return;
        };
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
        let mut total: u64 = 0;
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_file() || p.extension().map(|e| e == "lock").unwrap_or(false) {
                continue;
            }
            // P2-7（评审修复）：在途下载临时文件（`<name>.tmp.<pid>`）不得计入
            // LRU 候选/驱逐——另一线程正在写它，删除会让随后 hash/rename 失败
            // 并触发重试风暴。旧实现只排除 `.lock` 与 just_added。
            if p.file_name()
                .map(|n| n.to_string_lossy().contains(".tmp."))
                .unwrap_or(false)
            {
                continue;
            }
            if p == just_added {
                continue; // 刚下载的文件最热，永不驱逐
            }
            if let Ok(meta) = entry.metadata() {
                // 排序依据用 mtime：命中缓存时会 touch_cache_file 刷新它。不能用 atime——
                // Linux 普遍是 relatime/noatime 挂载，实测"反复读取并不会刷新 atime"
                // （复核问题 6），据此驱逐可能把刚用过的 shard 当冷数据删掉。
                let stamp = meta
                    .modified()
                    .or_else(|_| meta.accessed())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                total += meta.len();
                files.push((p, meta.len(), stamp));
            }
        }
        if total <= self.cache_limit_bytes {
            return;
        }
        // 最冷（最久未被触摸）优先驱逐
        files.sort_by_key(|(_, _, t)| *t);
        for (path, size, _) in files {
            if total <= self.cache_limit_bytes {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                // 顺带清理同名 .lock
                if let Some(stem) = path.file_name() {
                    let _ = fs::remove_file(
                        self.cache_dir
                            .join(format!("{}.lock", stem.to_string_lossy())),
                    );
                }
                total = total.saturating_sub(size);
                tracing::debug!(
                    "🧹 Cache over limit, evicted: {} (-{} bytes)",
                    path.display(),
                    size
                );
            }
        }
    }

    /// 读取极小的元数据文件 (index.json)，带重试退避
    pub fn read_manifest_to_string(
        rt: &tokio::runtime::Runtime,
        path_or_url: &str,
    ) -> std::io::Result<String> {
        if !path_or_url.starts_with("s3://")
            && !path_or_url.starts_with("gs://")
            && !path_or_url.starts_with("http")
        {
            return std::fs::read_to_string(path_or_url);
        }

        let parsed_url = Url::parse(path_or_url)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

        let mut last_err: Option<std::io::Error> = None;
        for attempt in 0..DOWNLOAD_ATTEMPTS {
            if attempt > 0 {
                backoff_sleep(attempt - 1);
            }
            let result: std::io::Result<String> = rt.block_on(async {
                let bytes = if parsed_url.scheme() == "http" || parsed_url.scheme() == "https" {
                    // P2-6：显式 client（连接/单次读取超时）。
                    let response = http_client()
                        .get(parsed_url.as_str())
                        .send()
                        .await
                        .map_err(|e| std::io::Error::other(format!("HTTP download failed: {e}")))?;
                    response
                        .error_for_status()
                        .map_err(|e| std::io::Error::other(format!("HTTP status error: {e}")))?
                        .bytes()
                        .await
                        .map_err(|e| {
                            std::io::Error::other(format!("Failed to read response bytes: {e}"))
                        })?
                } else {
                    let path_str = parsed_url.path().trim_start_matches('/');
                    if parsed_url.scheme() == "s3" {
                        let store = build_s3_store(&parsed_url)?;
                        let location = ObjectPath::from(path_str);
                        let response = store.get(&location).await.map_err(|e| {
                            std::io::Error::other(format!("Object store fetch failed: {e}"))
                        })?;
                        response.bytes().await.map_err(|e| {
                            std::io::Error::other(format!("Failed to read response bytes: {e}"))
                        })?
                    } else {
                        let (store, location) =
                            object_store::parse_url(&parsed_url).map_err(|e| {
                                std::io::Error::other(format!(
                                    "Failed to parse object store URL: {e}"
                                ))
                            })?;
                        let response = store.get(&location).await.map_err(|e| {
                            std::io::Error::other(format!("Object store fetch failed: {e}"))
                        })?;
                        response.bytes().await.map_err(|e| {
                            std::io::Error::other(format!("Failed to read response bytes: {e}"))
                        })?
                    }
                };
                String::from_utf8(bytes.to_vec()).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid UTF-8 encoding: {e}"),
                    )
                })
            });
            match result {
                Ok(text) => return Ok(text),
                Err(e) => {
                    tracing::warn!(
                        "Failed to read manifest (attempt {}/{}): {} — {}",
                        attempt + 1,
                        DOWNLOAD_ATTEMPTS,
                        path_or_url,
                        e
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| std::io::Error::other("Failed to read manifest")))
    }
}

/// 指数退避 + 随机抖动的 sleep（第 n 次：base*2^n + U(0, base)）
///
/// `min(5)` 的依据：把指数钳在 5（500ms << 5 = 16s 上限）——既防止极端
/// retry_index 下 u64 位移溢出（overflow-checks 开启时 `<<` 会 panic），
/// 也避免退避增长到分钟级，配合同段抖动打散千卡集群的对齐重试。
fn backoff_sleep(retry_index: u32) {
    let base_ms = RETRY_BACKOFF_BASE.as_millis() as u64;
    let exp = base_ms << retry_index.min(5);
    let jitter = rand::rng().random_range(0..=base_ms.max(1));
    std::thread::sleep(Duration::from_millis(exp + jitter));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::SystemTime;

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size_bytes("512"), Some(512));
        assert_eq!(parse_size_bytes("10K"), Some(10 * 1024));
        assert_eq!(parse_size_bytes("512MB"), Some(512 * 1024 * 1024));
        assert_eq!(parse_size_bytes("2G"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_size_bytes("1T"), Some(1024u64 * 1024 * 1024 * 1024));
        assert_eq!(parse_size_bytes("0"), Some(0));
        assert_eq!(parse_size_bytes(""), None);
        assert_eq!(parse_size_bytes("abc"), None);
    }

    #[test]
    fn test_sha256_file_hex_roundtrip() {
        let dir = std::env::temp_dir().join(format!("megadata_test_sha_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("blob");
        fs::write(&p, b"hello megadata").unwrap();
        let h1 = sha256_file_hex(&p).unwrap();
        let h2 = sha256_file_hex(&p).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        // 已知向量
        assert_eq!(sha256_file_hex(&p).unwrap(), {
            use sha2::Digest;
            let mut h = Sha256::new();
            h.update(b"hello megadata");
            hex::encode(h.finalize())
        });
        fs::remove_dir_all(&dir).ok();
    }

    /// 极简 stub HTTP 服务器：任何 GET 都返回固定 body，并累计命中次数。
    fn spawn_stub_http(
        body: Vec<u8>,
        hits: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) -> String {
        use std::io::Write as _;
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut buf = [0u8; 8192];
                let _ = std::io::Read::read(&mut s, &mut buf); // 读掉请求行/头即可
                hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(&body);
                let _ = s.flush();
            }
        });
        format!("http://{addr}")
    }

    /// 测试用构造器：绕过环境变量，直接指定各字段（F1 的 local_verify 可显式注入）。
    fn make_test_manager(
        cache_dir: PathBuf,
        cache_limit_bytes: u64,
        validate_on_hit: bool,
        local_verify: LocalVerifyMode,
    ) -> CloudIoManager {
        CloudIoManager {
            cache_dir,
            cache_limit_bytes,
            validate_on_hit,
            local_verify,
            local_verified: Mutex::new(HashMap::new()),
        }
    }

    /// 回归：缓存命中的 sha256 校验失败 → 必须**真的重下**，且返回的路径必须指向存在的文件。
    ///
    /// 旧实现（已修）在删除损坏缓存后仍返回原路径，调用方 `File::open` 必然失败；
    /// 该错误当时被上层"None 当作 EOF → 200ms 轮询 → 重试"的循环掩盖（多付一次
    /// POLL_TIMEOUT，见 stream.rs 的 EOF 分支），所以端到端看起来"能用"。本测试直接
    /// 锁死本函数自身的契约，不再依赖上层重试。
    #[test]
    fn test_cache_hit_hash_mismatch_redownloads_and_returns_existing_path() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let body = b"megadata-shard-bytes".to_vec();
        let expected = {
            use sha2::Digest;
            let mut h = Sha256::new();
            h.update(&body);
            hex::encode(h.finalize())
        };
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_stub_http(body.clone(), hits.clone());

        let cache_dir =
            std::env::temp_dir().join(format!("megadata_test_cache_{}", std::process::id()));
        fs::create_dir_all(&cache_dir).unwrap();
        // 直接构造：不受环境变量影响（validate_on_hit 默认关，需要显式打开）
        let mgr = make_test_manager(cache_dir.clone(), 0, true, LocalVerifyMode::Off);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        // 1) 首次：下载 + 校验通过
        let p1 = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .unwrap();
        assert_eq!(
            fs::read(&p1).unwrap(),
            body,
            "first download should produce the correct content"
        );
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // 2) 篡改缓存（等长改写，模拟 bit rot / 人为损坏）
        fs::write(&p1, vec![b'x'; body.len()]).unwrap();
        assert_ne!(sha256_file_hex(Path::new(&p1)).unwrap(), expected);

        // 3) 再次获取：必须自愈重下；返回路径必须存在（旧实现返回已删除文件的路径）
        let p2 = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .unwrap();
        assert!(
            Path::new(&p2).exists(),
            "returned path must exist (the old implementation returned a path that had just been deleted)"
        );
        assert_eq!(
            fs::read(&p2).unwrap(),
            body,
            "corrupted cache should be repaired by re-downloading"
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "verification failure should trigger exactly one re-download"
        );

        // 4) 修复后再取：直接命中缓存，不再产生下载
        let p3 = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .unwrap();
        assert_eq!(p3, p2);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "no more downloads after repair"
        );

        fs::remove_dir_all(&cache_dir).ok();
    }

    /// 回归（二轮 P2-1 / 首轮并发§3）：缓存目录被外部删除（/tmp 被 tmpwatch 或
    /// 容器回收清掉）后，`ensure_local_file_verified` 必须自愈重建目录并成功下载，
    /// 而不是把 ENOENT 冒充成"打不开 shard"报给用户。
    #[test]
    fn test_cache_dir_deleted_externally_is_recreated() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let body = b"megadata-shard-bytes".to_vec();
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_stub_http(body.clone(), hits.clone());

        let cache_dir =
            std::env::temp_dir().join(format!("megadata_test_gone_dir_{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache_dir);
        // 只构造管理器、不提前建目录（模拟"构造期建过、运行中被外部删掉"的中间态）
        let mgr = make_test_manager(cache_dir.clone(), 0, false, LocalVerifyMode::Off);
        fs::create_dir_all(&cache_dir).unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        // 首次下载建立缓存内容
        let p1 = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", None)
            .unwrap();
        assert!(Path::new(&p1).exists());

        // ★ 模拟外部清理：把整个缓存目录（连同锁文件、shard）一起删掉
        fs::remove_dir_all(&cache_dir).unwrap();
        assert!(!cache_dir.exists());

        // 再取：必须自愈（重建目录 + 重新下载），而不是 ENOENT 报错
        let p2 = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", None)
            .unwrap();
        assert!(cache_dir.is_dir(), "cache dir must be recreated");
        assert!(
            Path::new(&p2).exists(),
            "second fetch must succeed after external deletion"
        );
        assert_eq!(fs::read(&p2).unwrap(), body);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "cache gone → exactly one re-download"
        );

        fs::remove_dir_all(&cache_dir).ok();
    }

    /// 回归（二轮 P1-3 / 首轮 P2-1）：大 shard（> 默认 8MB 写缓冲）流式下载
    /// 内容完整——`chunk()` 增量搬运不得丢尾字节或打乱顺序。
    /// （内存峰值改善本身由实现保证：整文件不再 `bytes().await` 驻留。）
    #[test]
    fn test_large_shard_downloads_streaming_and_intact() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // 12MB：超过 8MB 的 BufWriter 缓冲与常见 socket 缓冲，逼出多 chunk 搬运
        let big_len = 12 * 1024 * 1024;
        let body: Vec<u8> = (0..big_len).map(|i| (i % 251) as u8).collect();
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_stub_http(body.clone(), hits.clone());

        let cache_dir =
            std::env::temp_dir().join(format!("megadata_test_big_shard_{}", std::process::id()));
        let mgr = make_test_manager(cache_dir.clone(), 0, false, LocalVerifyMode::Off);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let p = mgr
            .ensure_local_file_verified(&rt, &base, "big.bin", None)
            .unwrap();
        let got = fs::read(&p).unwrap();
        assert_eq!(got.len(), big_len, "streaming download must not truncate");
        assert_eq!(got, body, "streaming download must preserve byte order");
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        fs::remove_dir_all(&cache_dir).ok();
    }

    /// 回归（复核问题 6）：命中缓存必须刷新 mtime，否则 LRU 依据（atime 在 relatime 下
    /// 不刷新）会失真，可能驱逐刚用过的 shard。
    #[test]
    fn test_cache_hit_touches_mtime_for_lru() {
        let dir = std::env::temp_dir().join(format!("megadata_touch_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("shard.bin");
        fs::write(&f, b"data").unwrap();

        // 把 mtime 设成 2 小时前
        let old = SystemTime::now() - Duration::from_secs(7200);
        fs::OpenOptions::new()
            .write(true)
            .open(&f)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(old))
            .unwrap();
        let before = fs::metadata(&f).unwrap().modified().unwrap();
        assert!(
            before < SystemTime::now() - Duration::from_secs(3600),
            "precondition: mtime should be old"
        );

        touch_cache_file(&f);
        let after = fs::metadata(&f).unwrap().modified().unwrap();
        assert!(
            after > before + Duration::from_secs(3300),
            "cache hit should refresh mtime (LRU basis); before={before:?} after={after:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// 回归（复核问题 6）：LRU 驱逐应淘汰"最久未被触摸"的那个，而不是最新写入的。
    #[test]
    fn test_cache_eviction_drops_least_recently_touched() {
        let dir = std::env::temp_dir().join(format!("megadata_evict_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mk = |name: &str, age_secs: u64| -> PathBuf {
            let p = dir.join(name);
            fs::write(&p, vec![0u8; 100]).unwrap();
            let t = SystemTime::now() - Duration::from_secs(age_secs);
            fs::OpenOptions::new()
                .write(true)
                .open(&p)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(t))
                .unwrap();
            p
        };
        let cold = mk("cold.bin", 10800); // 3 小时未触摸
        let warm = mk("warm.bin", 7200); // 2 小时未触摸
        let hot = mk("hot.bin", 0); // 刚写入

        // 参与统计的是 cold+warm=200B（just_added 受保护、不计入），上限 150B →
        // 必须驱逐 1 个：应选最久未被触摸的 cold.bin
        let mgr = make_test_manager(dir.clone(), 150, false, LocalVerifyMode::Off);
        mgr.enforce_cache_limit(&hot);

        assert!(
            !cold.exists(),
            "the least recently touched shard should be evicted"
        );
        assert!(warm.exists(), "a warmer shard should not be evicted");
        assert!(
            hot.exists(),
            "the just-written shard is protected by just_added and should not be evicted"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// P2-7 回归：在途 `.tmp.<pid>` 文件即使最旧也不得被 LRU 驱逐。
    #[test]
    fn test_cache_eviction_skips_inflight_tmp_files() {
        let dir = std::env::temp_dir().join(format!("megadata_evict_tmp_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mk = |name: &str, age_secs: u64| -> PathBuf {
            let p = dir.join(name);
            fs::write(&p, vec![0u8; 100]).unwrap();
            let t = SystemTime::now() - Duration::from_secs(age_secs);
            fs::OpenOptions::new()
                .write(true)
                .open(&p)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(t))
                .unwrap();
            p
        };
        // 在途 tmp 最旧（3h），但正在被另一线程写；可驱逐的只有 cold.bin（2h）
        let inflight = mk("shard.bin.tmp.12345", 10800);
        let cold = mk("cold.bin", 7200);
        let hot = mk("hot.bin", 0);

        // tmp 若被计入：total=300B > 150B ⇒ 先驱逐 inflight（最旧）→ 测试失败。
        // 正确行为：tmp 不计入（参与统计只有 cold=100B ≤ 150B，甚至无需驱逐）。
        let mgr = make_test_manager(dir.clone(), 150, false, LocalVerifyMode::Off);
        mgr.enforce_cache_limit(&hot);

        assert!(
            inflight.exists(),
            "in-flight .tmp.<pid> file must not be evicted"
        );
        assert!(
            cold.exists(),
            "with tmp excluded, cold.bin fits within the limit"
        );
        assert!(hot.exists(), "just-added file stays protected");
        let _ = fs::remove_dir_all(&dir);
    }

    /// F1 回归：本地路径 + 全量校验（`MEGADATA_VERIFY_LOCAL=1`）+ 篡改文件 →
    /// `ensure_local_file_verified` 报 `Checksum failure (local)`。
    /// 同款文件未被篡改时校验通过；默认（Off）下行为不变（不校验不报错）。
    #[test]
    fn test_local_verify_full_mode_detects_tampering() {
        let dir =
            std::env::temp_dir().join(format!("megadata_verify_local_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("shard.bin");
        let content = b"megadata-local-shard-bytes".to_vec();
        fs::write(&file, &content).unwrap();
        let expected = {
            let mut h = Sha256::new();
            h.update(&content);
            hex::encode(h.finalize())
        };
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        // 1) 内容正确 → 校验通过（Full 模式）
        let mgr = make_test_manager(dir.clone(), 0, false, LocalVerifyMode::Full);
        let p = mgr
            .ensure_local_file_verified(&rt, dir.to_str().unwrap(), "shard.bin", Some(&expected))
            .unwrap();
        assert_eq!(
            Path::new(&p),
            &file,
            "local mode must return the plain local path (not a cache path)"
        );

        // 2) 等长篡改（bit flip）→ mismatch 必须报错
        let mut tampered = content.clone();
        tampered[0] ^= 0x01;
        fs::write(&file, &tampered).unwrap();
        let err = mgr
            .ensure_local_file_verified(&rt, dir.to_str().unwrap(), "shard.bin", Some(&expected))
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("Checksum failure (local)"),
            "error must identify the local checksum failure: {err}"
        );

        // 3) 默认（Off）→ 行为不变：同一被篡改文件照样放行
        let mgr_off = make_test_manager(dir.clone(), 0, false, LocalVerifyMode::Off);
        mgr_off
            .ensure_local_file_verified(&rt, dir.to_str().unwrap(), "shard.bin", Some(&expected))
            .expect("with verification disabled the old (permissive) behavior must be preserved");

        // 4) 修复内容后 Full 模式恢复通过
        fs::write(&file, &content).unwrap();
        mgr.ensure_local_file_verified(&rt, dir.to_str().unwrap(), "shard.bin", Some(&expected))
            .unwrap();

        let _ = fs::remove_dir_all(&dir);
    }

    /// F1 回归：校验结果按 (size, mtime) 缓存——同一文件只 hash 一次；
    /// size+mtime 未变（同长度篡改 + 恢复原 mtime）时命中缓存放行（这是
    /// "运行期零开销"的既定取舍），size 或 mtime 变化后重新 hash 并报错。
    #[test]
    fn test_local_verify_result_cached_by_size_and_mtime() {
        let dir =
            std::env::temp_dir().join(format!("megadata_verify_cache_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("shard.bin");
        let content = b"megadata-local-cache-bytes".to_vec();
        fs::write(&file, &content).unwrap();
        let expected = {
            let mut h = Sha256::new();
            h.update(&content);
            hex::encode(h.finalize())
        };
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let mgr = make_test_manager(dir.clone(), 0, false, LocalVerifyMode::Full);
        let base = dir.to_str().unwrap().to_string();

        // 首次校验通过 → 结果进入缓存
        mgr.ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .unwrap();

        // 同长度篡改 + 恢复原 mtime → (size, mtime) 相同 → 缓存命中放行（不重 hash）
        let mtime = fs::metadata(&file).unwrap().modified().unwrap();
        let mut tampered = content.clone();
        tampered[5] ^= 0xFF;
        fs::write(&file, &tampered).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(mtime))
            .unwrap();
        mgr.ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .expect("unchanged (size, mtime) must hit the verification cache");

        // mtime 变化 → 缓存失效 → 重新 hash → 篡改被捕获
        let later = mtime + Duration::from_secs(3600);
        fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(later))
            .unwrap();
        let err = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .unwrap_err();
        assert!(
            err.to_string().contains("Checksum failure (local)"),
            "cache must invalidate on mtime change: {err}"
        );

        // 不同长度的篡改同理：size 变化即缓存失效
        let mut appended = tampered;
        appended.push(b'x');
        fs::write(&file, &appended).unwrap();
        let err = mgr
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&expected))
            .unwrap_err();
        assert!(err.to_string().contains("Checksum failure (local)"));

        let _ = fs::remove_dir_all(&dir);
    }

    /// F1 回归：抽样模式的边界与确定性——`Sample(1.0)` 等价全量、`Sample(0.0)`
    /// 等价关闭；同一文件路径的抽样决定是确定性的（可复现）。
    #[test]
    fn test_local_verify_sample_mode_bounds() {
        let dir =
            std::env::temp_dir().join(format!("megadata_verify_sample_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("shard.bin");
        fs::write(&file, b"megadata-sample-bytes").unwrap();
        // 故意给一个错误 hash：任何"被抽中"的文件都会报错
        let wrong_hash = "0".repeat(64);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let base = dir.to_str().unwrap().to_string();

        // Sample(1.0)：全量 → 必然校验 → 必然报错
        let mgr_all = make_test_manager(dir.clone(), 0, false, LocalVerifyMode::Sample(1.0));
        let err = mgr_all
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&wrong_hash))
            .unwrap_err();
        assert!(err.to_string().contains("Checksum failure (local)"));

        // Sample(0.0)：不抽样 → 放行（与 Off 一致）
        let mgr_none = make_test_manager(dir.clone(), 0, false, LocalVerifyMode::Sample(0.0));
        mgr_none
            .ensure_local_file_verified(&rt, &base, "shard.bin", Some(&wrong_hash))
            .unwrap();

        // 确定性：同一 manager、同一文件重复调用结论一致
        let mgr = make_test_manager(dir.clone(), 0, false, LocalVerifyMode::Sample(0.5));
        let first = mgr.ensure_local_file_verified(&rt, &base, "shard.bin", Some(&wrong_hash));
        for _ in 0..3 {
            assert_eq!(
                first.is_err(),
                mgr.ensure_local_file_verified(&rt, &base, "shard.bin", Some(&wrong_hash))
                    .is_err()
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// F1 回归：`MEGADATA_VERIFY_LOCAL` 值解析——`1`/`true` 全量、`sample:P`
    /// 抽样、`0`/空/未设关闭、非法值告警后关闭（不 fail-fast，避免环境变量
    /// 手误直接打挂训练）。
    #[test]
    fn test_local_verify_mode_parsing() {
        assert_eq!(LocalVerifyMode::parse(""), LocalVerifyMode::Off);
        assert_eq!(LocalVerifyMode::parse("0"), LocalVerifyMode::Off);
        assert_eq!(LocalVerifyMode::parse("false"), LocalVerifyMode::Off);
        assert_eq!(LocalVerifyMode::parse("1"), LocalVerifyMode::Full);
        assert_eq!(LocalVerifyMode::parse("TRUE"), LocalVerifyMode::Full);
        assert_eq!(
            LocalVerifyMode::parse(" sample:0.01 "),
            LocalVerifyMode::Sample(0.01)
        );
        assert_eq!(
            LocalVerifyMode::parse("sample:1.0"),
            LocalVerifyMode::Sample(1.0)
        );
        assert_eq!(
            LocalVerifyMode::parse("sample:0"),
            LocalVerifyMode::Sample(0.0)
        );
        assert_eq!(LocalVerifyMode::parse("sample:-0.1"), LocalVerifyMode::Off); // 越界 → 关闭
        assert_eq!(LocalVerifyMode::parse("sample:1.1"), LocalVerifyMode::Off);
        assert_eq!(LocalVerifyMode::parse("sample:abc"), LocalVerifyMode::Off);
        assert_eq!(LocalVerifyMode::parse("yes"), LocalVerifyMode::Off); // 未知值 → 关闭
    }
}

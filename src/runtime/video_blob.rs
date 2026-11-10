//! P3b：video blob 按需读（D8 "video blob 不进 batch 预取窗口" 的落地）。
//!
//! 架构（P3b 方案 §7.1–7.4）：
//! - **预取线程**只读 `col_vid_rec`（40B 定宽小）+ `mm_cu`，产出带 rec 元数据但
//!   **字节延迟**的条目：`blob_ref = (store_id, offset, len)`（纯数据，跨线程/
//!   跨 FFI 无所有权问题）；`col_vid_blob` **不读、不计预取预算**。
//! - **collate（Python 绑定层，`__getitem__`/batch 组装时点）**经 FFI 用
//!   `blob_ref` 按需读 blob 区间 → per-clip Owned 字节 → DLPack 透出
//!   （`copy_count = 0`：所有权转移非复制）。
//! - 句柄不能复用预取线程的（按需读发生在消费时点、不同线程）⇒ 本模块维护
//!   **进程级单例注册表** `VideoBlobStore`：每个 shard 的 `col_vid_blob`
//!   打开时登记（id 单调递增），Seek/新 epoch 代际切换时关闭并移除旧代句柄
//!   （与预取线程代际栅栏同模式，防句柄泄漏）。
//!
//! **P1-2（评审修复）：句柄生命周期 = 未消费 batch 引用**。注销不再是"切 shard
//! 立即移除"——预取线程跨 shard 时旧 shard 的 batch 可能仍在 channel/已交付
//! Python 侧未被消费，其 `blob_ref` 按需读会拿到 `NotFound`（多 shard video
//! 训练正常时序下读取失败）。现在每条产出 batch 持有 **pin**（RAII guard，
//! 随 batch drop 自动归还）；注销只把条目标记 retired，**pins 归零后才真正
//! 移除**——队列与已交付引用全部释放前句柄保活。
//!
//! 内存上界（§7.4）：`video_ondemand_bytes_limit`（`StreamReaderOptions`，
//! 默认 0 = 不限）；在途账本 `video_in_flight_bytes` 计量"正在读"的字节
//! （发起读 +、读完交付 −）。单 clip 账单本身超限 → WARN 一次 + 放行
//! （活性保证，同 prefetch 单 batch 超限先例）。
//! **P2-2（评审修复）**：上限检查与记账改为**原子预留**（`fetch_update` CAS
//! 循环）——旧"load 检查再 fetch_add"在多线程同时读各小于上限的 clip 时可
//! 同时观察到在途为 0，把在途量推过上限，配置的并发读内存上限不可靠。

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

// VR-1（复审）：按需读改用**定位读**（pread 语义），不读写共享文件偏移——
// `File::try_clone`（Unix `fcntl(F_DUPFD)` / Windows `DuplicateHandle`）新旧句柄
// **共享同一 open file description 的偏移**，锁外 `seek + read_exact` 在并发读
// 同一 `store_id` 时会交叉错读（T1 seek(A) → T2 seek(B) → T1 read ⇒ 拿到 B 处
// 字节，静默错数据）。`read_exact_at` / `seek_read` 直接把 offset 作为参数传给
// 内核（`pread(2)` / `ReadFile` + `OVERLAPPED`），不触碰共享偏移，天然并发安全。
#[cfg(unix)]
use std::os::unix::fs::FileExt as _PreadExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt as _PreadExt;

/// 定位读（pread 语义）：从 `offset` 处读足 `buf.len()` 字节，**不读写文件共享偏移**。
/// Unix 走 `read_exact_at`（`pread(2)`）；Windows 走 `seek_read`（`ReadFile` +
/// `OVERLAPPED`）。两者均天然并发安全（VR-1）。两平台对外语义统一为
/// **读满或报错**（Windows 侧手动补读，见下）。
#[cfg(unix)]
#[inline]
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    file.read_exact_at(buf, offset)
}

/// VR-4c（复审）：Unix `read_exact_at` 的契约是"读满或报错"，而 Windows
/// `seek_read` 返回**实际读到的字节数**，短于 `buf` 时同样返回 `Ok(n)`。
/// 本函数若直接 `FileExt::seek_read` 透传，会在罕见场景（句柄语义异常等）
/// 静默返回半截数据、且被上层当作完整 blob 交付。此处循环补读，使
/// 两平台契约一致（短读=0 即 `UnexpectedEof`，同 `read_exact` 语义）。
#[cfg(windows)]
#[inline]
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.seek_read(&mut buf[filled..], offset + filled as u64) {
            // 0 字节 = 提前 EOF（文件被截断 / rec 与 blob 不一致）→ 不能当成功。
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "failed to fill whole buffer: read {filled} of {} bytes at offset {offset}",
                        buf.len()
                    ),
                ));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// 进程级 video blob 句柄注册表（单例）。
pub struct VideoBlobStore {
    files: Mutex<HashMap<u64, StoreEntry>>,
    next_id: AtomicU64,
    /// **P3b §7.4 在途账本**：collate 按需读"正在读"的字节（发起读 +、读完
    /// 交付 −）。消费侧持有的字节不計（属消费侧内存）。
    in_flight_bytes: AtomicU64,
    /// 按需读字节上限（0 = 不限；进程级——reader 构造时经
    /// `StreamReaderOptions.video_ondemand_bytes_limit` 设置，多 reader 进程
    /// 后者覆盖前者）。
    ondemand_limit: AtomicU64,
    /// 单 clip 账单本身超限的 WARN 只发一次（防刷屏；活性保证同 prefetch 先例）。
    over_warned: AtomicBool,
}

/// 注册表条目（P1-2：pin 计数 + retired 标记）。
struct StoreEntry {
    file: VideoBlobFile,
    /// 未消费 batch 持有的 pin 数（[`VideoBlobPin`] RAII guard 归还）。
    pins: u32,
    /// 已被注销（shard 切换 / reader 析构 / 构造失败回滚）——`pins` 归零后移除。
    retired: bool,
}

/// **P1-2：batch 持有的 blob 句柄 pin（RAII）**。
///
/// 预取线程产出带 `blob_ref` 的 batch 时对当前 `store_id` pin 一次，guard 随
/// batch 走（channel、前台陈旧丢弃、Python 侧 GC——任何 drop 路径都会归还）；
/// reader 切 shard / 析构时 `unregister` 只标记 retired，**最后一个 pin 归还时
/// 才真正关闭句柄**。这保证"旧 shard 的 batch 尚未消费完"期间其 video blob
/// 按需读不失败（评审复现：多 shard 预取后读第一个 batch 的 blob → NotFound）。
#[derive(Debug)]
pub struct VideoBlobPin {
    id: u64,
}

impl VideoBlobPin {
    /// 句柄 id（诊断用）。
    #[allow(dead_code)]
    pub fn store_id(&self) -> u64 {
        self.id
    }
}

impl Drop for VideoBlobPin {
    fn drop(&mut self) {
        VideoBlobStore::global().unpin(self.id);
    }
}

/// **P2-3（评审修复）：注册 RAII guard**。
///
/// `open_vid_sidecar` 注册 blob 句柄后仍有可失败操作（`col_vid_rec` 打开 /
/// `VideoSidecarReader::from_files`）；旧实现失败时 `current_vid_store_id`
/// 尚未赋值，刚注册的 id 无法清理——注册表与 fd 持续增长（长期反复创建/
/// 销毁 reader、重试损坏 sidecar 时泄漏可见）。guard 在失败路径（`?` 提前
/// 返回 / panic unwind）自动注销；构造成功后调用 [`Self::disarm`] 移交
/// 所有权给 reader。
#[derive(Debug)]
pub struct RegisterGuard {
    id: u64,
    armed: bool,
}

impl RegisterGuard {
    /// 移交所有权（构造成功；后续注销由 reader 生命周期负责）。
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RegisterGuard {
    fn drop(&mut self) {
        if self.armed {
            VideoBlobStore::global().unregister(self.id);
        }
    }
}

struct VideoBlobFile {
    file: File,
    blob_len: u64,
}

impl VideoBlobStore {
    fn new_empty() -> Self {
        VideoBlobStore {
            files: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            in_flight_bytes: AtomicU64::new(0),
            ondemand_limit: AtomicU64::new(0),
            over_warned: AtomicBool::new(false),
        }
    }

    /// 进程级单例（懒初始化；跨 reader/epoch 共享，代际由 register/unregister 管理）。
    pub fn global() -> &'static Self {
        static STORE: OnceLock<VideoBlobStore> = OnceLock::new();
        STORE.get_or_init(VideoBlobStore::new_empty)
    }

    /// 登记一个已打开的 `col_vid_blob` 句柄，返回 `store_id`（注册表持有句柄）。
    pub fn register(&self, file: File, blob_len: u64) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut files) = self.files.lock() {
            files.insert(
                id,
                StoreEntry {
                    file: VideoBlobFile { file, blob_len },
                    pins: 0,
                    retired: false,
                },
            );
        }
        id
    }

    /// 注销句柄（Seek/新 epoch/reader 析构/构造失败回滚）。
    ///
    /// **P1-2**：不再立即移除——仍有未消费 batch pin 着时条目标记 `retired`
    /// （可读、不可再新 pin），最后一个 pin 归还（[`VideoBlobPin::drop`] →
    /// [`Self::unpin`]）时才关闭文件。无 pin 时立即移除（旧行为）。
    pub fn unregister(&self, id: u64) {
        if let Ok(mut files) = self.files.lock() {
            let remove_now = match files.get_mut(&id) {
                Some(e) => {
                    e.retired = true;
                    e.pins == 0
                }
                None => false, // 未知 id 静默忽略（幂等）
            };
            if remove_now {
                files.remove(&id);
            }
        }
    }

    /// **P1-2**：pin 一个已登记句柄，返回随 batch 存活的 RAII guard。
    ///
    /// 条目不存在（已注销且无 pin 引用、句柄已关闭）→ `NotFound`——调用方不得
    /// 产出携带死 `blob_ref` 的 batch。
    pub fn pin(&self, id: u64) -> std::io::Result<VideoBlobPin> {
        let mut files = self
            .files
            .lock()
            .map_err(|_| std::io::Error::other("video blob store lock poisoned"))?;
        let entry = files.get_mut(&id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "video blob store id {id} not found — the shard was closed \
                     (seek/new epoch) or never registered; re-read the batch before \
                     reading its video blobs"
                ),
            )
        })?;
        if entry.retired {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "video blob store id {id} was retired (shard switched); its blob handle \
                     is only kept alive for batches still pinning it"
                ),
            ));
        }
        entry.pins = entry.pins.saturating_add(1);
        Ok(VideoBlobPin { id })
    }

    /// **P1-2**：归还一个 pin；归零且已 retired ⇒ 真正关闭句柄。
    fn unpin(&self, id: u64) {
        if let Ok(mut files) = self.files.lock() {
            let remove_now = match files.get_mut(&id) {
                Some(e) => {
                    e.pins = e.pins.saturating_sub(1);
                    e.retired && e.pins == 0
                }
                None => false,
            };
            if remove_now {
                files.remove(&id);
            }
        }
    }

    /// 当前 pin 着该 id 的未消费 batch 数（诊断/测试断言用）。
    #[allow(dead_code)]
    pub fn pin_count(&self, id: u64) -> u32 {
        self.files
            .lock()
            .map(|f| f.get(&id).map(|e| e.pins).unwrap_or(0))
            .unwrap_or(0)
    }

    /// **P2-3**：为一个**已注册**的 id 创建 RAII 回滚 guard（构造失败自动注销）。
    pub fn register_guard(&self, id: u64) -> RegisterGuard {
        RegisterGuard { id, armed: true }
    }

    /// 设置按需读字节上限（0 = 不限；进程级，reader 构造时调用）。
    ///
    /// **P3b-4（评审）**：注册表是进程级单例，多 reader/多数据集并存时后设置者
    /// 覆盖先前值（语义不变，仍以最后一次为准）。为避免"静默覆盖"排障困难，
    /// 此处做冲突校验：当已存在非零上限、且新值与之不同时发 WARN（单 reader
    /// 或首次设置不触发；后者覆盖前者的语义保留）。
    pub fn set_ondemand_limit(&self, limit: u64) {
        let prev = self.ondemand_limit.swap(limit, Ordering::Relaxed);
        if prev != 0 && prev != limit {
            tracing::warn!(
                "VideoBlobStore 为进程级单例：video 按需读字节上限被覆盖（{prev} → {limit}），\
                 以最后一次设置为准。多数据集需各自独立上限时，请分进程运行。"
            );
        }
    }

    /// 当前在途字节（诊断/测试断言用）。
    #[allow(dead_code)]
    pub fn in_flight_bytes(&self) -> u64 {
        self.in_flight_bytes.load(Ordering::Acquire)
    }

    /// 按需读 `[offset, offset+len)`（**定位读**／pread 语义；越界 fail-fast，同
    /// `MmSidecarReader::read_blob_range` 语义）。
    ///
    /// P3b-3（评审）+ VR-1（复审）：锁内只做 `try_clone`（共享 fd）+ 越界校验，
    /// **锁外走 `read_exact_at` / `seek_read`**（内核 `pread(2)` / `ReadFile` +
    /// `OVERLAPPED`）——不读写共享文件偏移，多 worker DataLoader 下并发按需读
    /// 同一 `store_id` 也不会交叉错读（旧"锁外 seek+read"修法在共享偏移的
    /// clone 上有竞态，已改）。
    ///
    /// VR-4d（复审）：**先校验再分配**——越界/未知 id 必须在 `vec![0; len]` 之前
    /// 判掉。旧顺序下损坏 rec 携带超大 `len` 会先触发巨型分配（abort）而非干净的
    /// `InvalidData`，既掩盖了真实错误、也可能直接把进程打死。
    pub fn read(&self, store_id: u64, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
        // 零长区间合法（占位记录），且不触碰注册表（旧行为保留：未知 id + 零长
        // 也返回空，不报 NotFound）。
        if len == 0 {
            return Ok(Vec::new());
        }
        let file = {
            let files = self
                .files
                .lock()
                .map_err(|_| std::io::Error::other("video blob store lock poisoned"))?;
            let entry = files.get(&store_id).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "video blob store id {store_id} not found — the shard was closed \
                         (seek/new epoch) or never registered; re-read the batch before \
                         reading its video blobs"
                    ),
                )
            })?;
            if offset.saturating_add(len) > entry.file.blob_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "video blob range [{offset}, {}) beyond col_vid_blob length {} — \
                         blob/rec inconsistent",
                        offset + len,
                        entry.file.blob_len
                    ),
                ));
            }
            entry.file.file.try_clone()?
        };
        // 校验通过后 len ≤ blob_len，分配规模受文件实际长度约束（VR-4d）。
        let mut buf = vec![0u8; len as usize];
        // 阻塞 IO 在锁外执行（多并发按需读不再串行化）；定位读不读写共享偏移，
        // 并发读同一 store_id 也安全（VR-1）。
        pread_exact(&file, &mut buf, offset)?;
        Ok(buf)
    }

    /// **带账本 + 背压的按需读**（FFI 入口，P3b §7.4）：
    /// - 账本：发起读 `+len`、读完交付 `-len`（在途 = 并发读压力）；
    /// - 背压：上限 > 0 且在途超限 → 50ms tick 自旋等待（其它读交付后自然收敛）；
    /// - **单 clip 账单本身超限** → WARN 一次 + 放行（活性保证——若死等，
    ///   in_flight 归零后条件仍不成立，该 clip 永远读不出）。
    ///
    /// **P2-2（评审修复）：原子预留**。旧实现"`load` 检查在途 → `fetch_add`"两步
    /// 之间无同步，多线程同时读各小于上限的 clip 时可同时观察到在途为 0，随后
    /// 把在途量推过上限——配置的并发读内存上限不可靠。现在用
    /// `fetch_update`（CAS 循环）**连同检查带占用一步完成**：任一线程观察到
    /// "cur + len > limit" 即重试，成功即视为已预留该额度。
    pub fn read_tracked(&self, store_id: u64, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
        const ONDEMAND_WAIT_TICK: std::time::Duration = std::time::Duration::from_millis(50);
        let limit = self.ondemand_limit.load(Ordering::Relaxed);
        if limit > 0 {
            if len > limit {
                if !self.over_warned.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "⚠️ single video clip on-demand read {len} bytes exceeds \
                         video_ondemand_bytes_limit {limit}; letting it through to keep \
                         liveness — consider raising video_ondemand_bytes_limit"
                    );
                }
                // 单 clip 超限仍记账（活性放行，不参与等待判断）
                self.in_flight_bytes.fetch_add(len, Ordering::Acquire);
            } else {
                // 原子预留：CAS 成功 = 额度已占用（检查与占用之间无窗口）
                while self
                    .in_flight_bytes
                    .fetch_update(Ordering::Acquire, Ordering::Acquire, |cur| {
                        if cur.saturating_add(len) > limit {
                            None // 超限：不占用，重试等待
                        } else {
                            Some(cur + len)
                        }
                    })
                    .is_err()
                {
                    std::thread::sleep(ONDEMAND_WAIT_TICK);
                }
            }
        } else {
            // 不限：仅记账（诊断口径不变）
            self.in_flight_bytes.fetch_add(len, Ordering::Acquire);
        }
        let result = self.read(store_id, offset, len);
        // 交付：任何路径（含读失败）都归还额度，账本守恒
        self.in_flight_bytes.fetch_sub(len, Ordering::Release);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// 测试专用文件命名计数器：`tmp_file` 必须为每次调用生成**唯一路径**——
    /// 旧实现用 `next_id.load()`（不自增），并发测试会路径碰撞互相覆盖文件
    /// （`register_read_unregister` 读到并发用例的字节）。用独立原子计数器隔离。
    static TMP_FILE_SEQ: AtomicU64 = AtomicU64::new(0);

    fn tmp_file(bytes: &[u8]) -> File {
        let seq = TMP_FILE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "megadata_vidblob_{}_{}_{seq}",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("t")
                .replace("::", "_"),
        ));
        std::fs::write(&path, bytes).unwrap();
        File::options().read(true).open(&path).unwrap()
    }

    #[test]
    fn register_read_unregister() {
        let store = VideoBlobStore::global();
        let data = vec![7u8; 100];
        let id = store.register(tmp_file(&data), data.len() as u64);
        let got = store.read(id, 10, 20).unwrap();
        assert_eq!(got, vec![7u8; 20]);
        // 零长区间合法（占位记录）
        assert!(store.read(id, 100, 0).unwrap().is_empty());
        // 越界 fail-fast
        assert!(store.read(id, 90, 20).is_err());
        // 未知 id fail-fast（句柄已随代际移除）
        store.unregister(id);
        assert!(store.read(id, 0, 1).is_err());
    }

    /// **VR-4d 回归锁**：越界校验必须先于分配。
    ///
    /// 损坏 rec 携带超大 `len`（这里是 `u64::MAX`）时，旧顺序会先做
    /// `vec![0u8; len as usize]` → 巨型分配 abort（进程直接死，而非返回错误）。
    /// 新顺序下应拿到干净的 `InvalidData`，且 `saturating_add` 不会溢出。
    #[test]
    fn oversized_len_fails_fast_before_allocation() {
        let store = VideoBlobStore::global();
        let data = vec![3u8; 64];
        let id = store.register(tmp_file(&data), data.len() as u64);
        let err = store.read(id, 0, u64::MAX).unwrap_err();
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "越界 len 必须是 InvalidData（而非分配 abort/其它错误）：{err}"
        );
        // 零长区间仍合法（本轮顺序调整不影响占位记录语义）
        assert!(store.read(id, 0, 0).unwrap().is_empty());
        store.unregister(id);
    }

    /// **VR-1 回归锁**：并发读同一 `store_id` 不得交叉错读。
    ///
    /// 背景：旧"锁外 seek+read"修法在 `try_clone` 共享偏移的句柄上会竞态——
    /// T1 `seek(A)` → T2 `seek(B)` → T1 `read` ⇒ T1 拿到 B 处字节（静默错数据）。
    /// 本用例用**内容互异**的字节模式 + N 线程 × M 次随机 `[offset,len)` 并发读，
    /// 逐一断言字节正确：seek+read 形态下负载足够即稳定复现错读；pread 形态恒绿。
    #[test]
    fn concurrent_reads_same_store_id_do_not_cross_contaminate() {
        // 内容互异：每个 64B 块的首字节 = 块序号（错读则首字节不匹配）。
        const BLOCK: usize = 64;
        const N_BLOCKS: usize = 64; // 4 KiB
        let mut data = vec![0u8; BLOCK * N_BLOCKS];
        for b in 0..N_BLOCKS {
            for i in 0..BLOCK {
                data[b * BLOCK + i] = (b as u8) ^ (i as u8);
            }
        }
        let store = VideoBlobStore::global();
        let id = store.register(tmp_file(&data), data.len() as u64);

        // N 线程 × M 次随机读；负载足够大以在旧实现下稳定复现竞态。
        const N_THREADS: usize = 8;
        const M_READS: usize = 200;
        let data_arc = Arc::new(data);
        let mut handles = Vec::with_capacity(N_THREADS);
        for t in 0..N_THREADS {
            let data = data_arc.clone();
            handles.push(std::thread::spawn(move || {
                // 确定性伪随机（不依赖 rand crate；线性同余）。
                let mut seed = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF;
                let mut rng = move || {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (seed >> 33) as usize
                };
                for _ in 0..M_READS {
                    let block = rng() % N_BLOCKS;
                    // 读长度跨块边界时也能验证（offset+len 不超总长即可）。
                    let len = 1 + (rng() % (BLOCK * 2));
                    let offset = block * BLOCK;
                    if offset + len > data.len() {
                        continue;
                    }
                    let got = store.read(id, offset as u64, len as u64)
                        .unwrap_or_else(|e| panic!("read({offset},{len}) failed: {e}"));
                    assert_eq!(
                        got,
                        data[offset..offset + len].to_vec(),
                        "VR-1 并发错读：thread={t} offset={offset} len={len}——若为 seek+read 共享偏移竞态，\n\
                         此处会拿到其它线程 seek 后的字节（首字节应为 {}，实际为 {}）",
                        data[offset],
                        got[0],
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("worker panicked");
        }
        store.unregister(id);
    }

    /// **VR-1 语义锁**：定位读不读写共享偏移（`pread` 语义的直接验证）。
    ///
    /// 对同一文件的两个 clone 句柄：一个 `seek(4)` 后另一个 `pread_exact(…, 0)` 读出
    /// 的仍是偏移 0 起的数据；且 `pread` 后原句柄的偏移不变（仍为 4）——与旧
    /// "锁外 seek+read"修法形成对照（后者会使两 clone 互相干扰偏移）。
    #[test]
    fn pread_does_not_disturb_shared_offset() {
        use std::io::{Read, Seek, SeekFrom};
        let data = b"AAAABBBBCCCC";
        let mut f1 = tmp_file(data);
        let mut f2 = f1.try_clone().unwrap();
        // f1 seek 到 4（"BBBB" 起点）；f2 偏移随 f1（共享 open file description）。
        f1.seek(SeekFrom::Start(4)).unwrap();
        assert_eq!(f2.stream_position().unwrap(), 4, "try_clone 共享偏移");
        // 对 f2 走 pread_exact（offset=0）：应读出 "AAAA"，且不改变共享偏移。
        let mut buf = [0u8; 4];
        pread_exact(&f2, &mut buf, 0).unwrap();
        assert_eq!(&buf, b"AAAA", "pread 应从 offset=0 读，不受共享偏移影响");
        assert_eq!(f2.stream_position().unwrap(), 4, "pread 不读写共享偏移");
        // 对照：f1 从当前（共享）偏移 read_exact → "BBBB"（证明偏移未被 pread 扰乱）。
        let mut buf2 = [0u8; 4];
        f1.read_exact(&mut buf2).unwrap();
        assert_eq!(&buf2, b"BBBB");
    }

    /// **P1-2 回归锁**：注销后未消费 batch 的 pin 必须保活句柄。
    ///
    /// 复现评审场景：多 shard 预取——预取线程切到新 shard 立即 `unregister`
    /// 旧 `store_id`，但旧 batch 可能尚未被消费。旧实现下 `read` 拿到
    /// `NotFound`（"store id N not found"）；新实现下 pin 未归零前句柄可读，
    /// 最后一个 pin drop 后才真正移除。
    #[test]
    fn pin_keeps_blob_readable_after_unregister_until_last_pin_drops() {
        let store = VideoBlobStore::global();
        let data = vec![0x5Au8; 128];
        let id = store.register(tmp_file(&data), data.len() as u64);

        // 两个未消费 batch 各持一个 pin（模拟 channel 里 + 已交付 Python 侧）
        let pin_a = store.pin(id).expect("registered entry must be pinnable");
        let pin_b = store.pin(id).expect("registered entry must be pinnable");
        assert_eq!(store.pin_count(id), 2);

        // shard 切换：注销旧代（旧实现在此移除句柄）
        store.unregister(id);
        assert_eq!(store.pin_count(id), 2, "pin 未归零 ⇒ 条目不移除");
        // 未消费 batch 的按需读必须仍然成功（P1-2 核心断言）
        assert_eq!(store.read(id, 0, 32).unwrap(), vec![0x5Au8; 32]);

        // retired 条目不可再新 pin（新 batch 必须来自新 shard 的新 store_id）
        assert!(store.pin(id).is_err(), "retired 条目不得再 pin");

        drop(pin_a);
        assert_eq!(store.pin_count(id), 1, "还有一个未消费 batch");
        assert_eq!(store.read(id, 96, 32).unwrap(), vec![0x5Au8; 32]);

        drop(pin_b);
        assert_eq!(store.pin_count(id), 0, "全部归还");
        // 最后一个 pin 归还 ⇒ 句柄真正移除
        assert!(store.read(id, 0, 1).is_err(), "pins 归零后句柄必须已移除");
    }

    /// **P1-2 边界**：无 pin 时注销立即移除（旧行为保留——不泄漏句柄）。
    #[test]
    fn unregister_without_pin_removes_immediately() {
        let store = VideoBlobStore::global();
        let data = vec![1u8; 16];
        let id = store.register(tmp_file(&data), data.len() as u64);
        store.unregister(id);
        assert!(store.read(id, 0, 1).is_err(), "无 pin ⇒ 立即移除");
    }

    /// **P2-2 回归锁**：并发按需读的在途字节峰值不得超过
    /// `video_ondemand_bytes_limit`。
    ///
    /// 旧实现"load 检查 → fetch_add"两步之间无同步：N 个线程同时读各小于上限的
    /// clip 时可同时观察到在途为 0，把在途量推过上限（这里 4×2MiB > 4MiB）。
    /// 新实现 `fetch_update` 原子预留，峰值恒 ≤ 上限。watchdog 线程持续采样，
    /// 任一时刻超限即失败（对新实现恒绿；对旧实现在负载足够时稳定复现）。
    #[test]
    fn concurrent_read_tracked_never_exceeds_ondemand_limit() {
        const CLIP: u64 = 2 * 1024 * 1024; // 2 MiB/clip
        const LIMIT: u64 = 4 * 1024 * 1024; // 4 MiB 上限（并发 2 路即满）
        const THREADS: usize = 4;
        const ROUNDS: usize = 3;
        // 隔离实例：进程级单例的 ondemand_limit 会被并发用例（reader 构造）
        // 覆盖为 0，导致本用例在并行测试下假失败；独立 store 不受干扰。
        let store = Arc::new(VideoBlobStore::new_empty());
        store.set_ondemand_limit(LIMIT);
        let data = vec![0x11u8; CLIP as usize];
        let id = store.register(tmp_file(&data), CLIP);
        let stop = Arc::new(AtomicBool::new(false));
        let violation = Arc::new(AtomicU64::new(0));

        // watchdog：持续采样在途字节，超限即记录（新实现下永不触发）
        let watchdog = {
            let store = store.clone();
            let stop = stop.clone();
            let violation = violation.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let cur = store.in_flight_bytes();
                    if cur > LIMIT {
                        violation.fetch_max(cur, Ordering::Relaxed);
                    }
                }
            })
        };

        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..ROUNDS {
                    let got = store
                        .read_tracked(id, 0, CLIP)
                        .unwrap_or_else(|e| panic!("read_tracked failed: {e}"));
                    assert_eq!(got.len(), CLIP as usize);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker panicked");
        }
        stop.store(true, Ordering::Relaxed);
        watchdog.join().expect("watchdog panicked");

        let peak = violation.load(Ordering::Relaxed);
        assert_eq!(
            peak, 0,
            "在途字节峰值 {peak} 超过上限 {LIMIT}（P2-2 原子预留失效）"
        );
        assert_eq!(store.in_flight_bytes(), 0, "账本守恒：全部交付后归零");
        store.unregister(id);
    }
}

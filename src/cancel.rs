// src/cancel.rs
// 中断信号：让 Rust 同步函数在 IO 边界快速响应取消请求。
//
// 设计：
// 1. **每实例 CancellationToken**：每个 StreamReader / StreamMixer / DataBuilder 持有
//    独立的令牌，支持 scope 化取消（取消一个加载器不影响其它）。
// 2. **零依赖**：仅 std atomic。
// 3. 旧版的全局 CANCEL_FLAG（request_cancel / reset_cancel 等）已于 2026-09-27
//    评审清理中整体退役——它已无任何消费者（v3 全部走每实例令牌，
//    `uninstall_signal_handlers` 里的 `reset_cancel()` 亦为 no-op）。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// 错误类型。Python 端会捕获并当作中断处理。
#[derive(Debug)]
pub struct Canceled;

impl std::fmt::Display for Canceled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Megadata operation canceled")
    }
}
impl std::error::Error for Canceled {}

// =========================================================
// 每实例取消令牌
// =========================================================

/// 每实例取消令牌。Clone 共享同一底层 AtomicBool。
#[derive(Clone)]
pub struct CancellationToken {
    inner: Arc<AtomicBool>,
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 请求取消。
    pub fn cancel(&self) {
        self.inner.store(true, Ordering::Release);
    }

    /// 是否已被请求取消。
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::Acquire)
    }

    /// 重置为未取消（一般不需要：新实例自带全新令牌）。
    pub fn reset(&self) {
        self.inner.store(false, Ordering::Release);
    }

    /// 在 IO 边界检查；如已取消返回 Err(Canceled)。
    #[inline]
    pub fn check(&self) -> Result<(), Canceled> {
        if self.is_cancelled() {
            return Err(Canceled);
        }
        Ok(())
    }

    /// 阻塞等待取消或超时。用于让后台线程在 EOF 时仍能响应取消。
    pub fn wait_cancelled_or(&self, dur: Duration) -> bool {
        // 简单轮询：分段睡眠以便及时响应取消
        let step = Duration::from_millis(50);
        let mut elapsed = Duration::ZERO;
        while elapsed < dur {
            if self.is_cancelled() {
                return true;
            }
            std::thread::sleep(step.min(dur - elapsed));
            elapsed += step;
        }
        self.is_cancelled()
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

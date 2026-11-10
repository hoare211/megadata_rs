use pyo3::ffi::{PyCapsule_GetPointer, PyCapsule_New, PyCapsule_SetName, PyErr_Clear};
use pyo3::prelude::*;
use std::os::raw::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// ==========================================
// 1. 定义 DLPack 标准的 C 结构体 (ABI)
// ==========================================
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32, // 1 代表 CPU，2 代表 CUDA
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,   // 0=Int, 1=UInt, 2=Float, 3=Bfloat
    pub bits: u8,   // 32 位 (i32)
    pub lanes: u16, // 1
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void, // 借给 C 的上下文（装载我们的 Rust Box 指针）
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>, // 析构回调
}

struct TensorPayload {
    /// 字节流（dtype 信息在 `dtype` 字段描述）。
    /// 泛型化基础层：`Vec<i32>` 硬编码会让
    /// doc_ids / loss_weights 等 i8/f16 列无法走零拷贝通道；字节流 + dtype 描述
    /// 是 DLPack 标准姿势，下游列扩展（落地层）只需新增 dtype/shape 组合。
    data: Vec<u8>,
    /// **P1-T6**共享缓冲 view 形态：非 `None` 时 `data` 为空占位，DLTensor.data
    /// 指向共享缓冲的 `[view_offset, view_offset+data_len)` 区间，payload 持有
    /// Arc 强引用保活（D8"驱逐保护"堆形态：tensor 存活期 = 引用计数 > 0 ⇒
    /// buffer 不回收不复用）。deleter drop payload ⇒ Arc 减一，最后一个引用
    /// 释放时触发 `SharedBlobBuffer::drop`（release_count +1，多 view 单次释放）。
    shared: Option<(Arc<SharedBlobBuffer>, usize)>,
    dtype: DLDataType,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

/// **P1-T6（D8）**batch 级共享 blob 缓冲：整 batch 的图像字节一次读入
/// （磁盘 → 堆，全链路**唯一一次**数据拷贝），各图像条目是它的定宽 view。
///
/// - **驱逐保护**（堆形态语义）：capsule/_tensor 持有 `Arc` 强引用 ⇒ 引用计数
///   大于 0 期间底层内存不被回收、不被复用；#38 真 mmap 落地后升级为
///   "cache 持有映射页时 LRU 禁止驱逐"，接口形状不变。
/// - **多 view 单次释放**：N 个 view 各持一个 Arc 克隆，deleter 各调一次
///   （N 次），底层 buffer 只在最后一个引用消失时 Drop 一次（release_count
///   断言 == 1，测试锚点）。
/// - **copy_count**：观测性计数器——当前实现没有任何深拷贝路径（view 只有
///   `as_slice` 只读借用，Arc::clone 只增引用计数），恒 0；未来若引入
///   "共享缓冲 → 独立副本"的降级路径，必须显式 +1，测试据此断言零额外拷贝。
pub struct SharedBlobBuffer {
    pub data: Vec<u8>,
    /// Drop 次数（最后一个 Arc 引用消失时 +1；多 view 断言 == 1）
    pub release_count: Arc<AtomicUsize>,
    /// 深拷贝次数（观测；当前无路径会 +1，见类型注释；断言点在 pyo3-tests）
    #[allow(dead_code)]
    pub copy_count: Arc<AtomicUsize>,
}

impl Drop for SharedBlobBuffer {
    fn drop(&mut self) {
        self.release_count.fetch_add(1, Ordering::Release);
    }
}

unsafe extern "C" fn dlpack_deleter(managed_tensor: *mut DLManagedTensor) {
    if managed_tensor.is_null() {
        return;
    }
    // ★ 评审 §3.4(a)：整个函数体包 catch_unwind。
    //   这是交给 C / CPython GC 的析构回调，`extern "C"` fn 内 panic 在 Rust 语义上
    //   是**未定义行为**（实际会 abort，或在 CPython 里表现为 "Exception ignored in: "
    //   + 潜在状态损坏）。当前 drop 链（Vec<u8>/Vec<i64>）不会 panic，所以还不是活
    //   bug；但这是整个代码库最脆弱的边界——给 TensorPayload 加一个带 Drop 且可能
    //   panic 的字段（如 Arc<Mutex<..>> 的锁中毒），就变成**进程级崩溃**。
    //   修复成本极低（catch_unwind），故在此永久兜底。
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { reclaim_managed_tensor(managed_tensor) };
    }));
    if outcome.is_err() {
        // 释放路径 panic 意味着可能泄漏，但**绝不能**把 panic 泄进 CPython：
        // 那会 abort 整个训练进程（批量作业直接全崩），比泄漏严重得多。
        tracing::error!(
            "dlpack_deleter panicked while reclaiming the tensor (possible leak, process kept alive)"
        );
    }
}

/// 真正的释放逻辑（从 `extern "C"` 边界剥离，便于 catch_unwind 包裹）。
///
/// # Safety
/// `managed_tensor` 必须非空，且来自 [`bytes_to_dlpack_capsule`] 的 `Box::into_raw`。
unsafe fn reclaim_managed_tensor(managed_tensor: *mut DLManagedTensor) {
    // SAFETY: 见调用方约定。
    let ctx = unsafe { (*managed_tensor).manager_ctx as *mut TensorPayload };
    // ★ 评审 §3.4(d)（重入防御）：**先**把 manager_ctx 置空，再释放 payload。
    //   这样即使本函数在释放过程中被重入（消费端 bug 或异常路径二次析构），
    //   第二次进来会看到 null ctx 而直接返回，不会二次 free。
    //   注意：真正的"调用两次 deleter"（消费端在第一次之后又调一次，且此时 shell
    //   已被 free）是 DLPack 协议违约，无法在不泄漏一个标志位分配的前提下检测——
    //   任何幂等标志都必须比 payload 活得久，而那意味着刻意的堆泄漏。
    //   这里能挡住的正是现实中的重入路径（含本析构器自身的重入）。
    unsafe { (*managed_tensor).manager_ctx = std::ptr::null_mut() };

    if !ctx.is_null() {
        // 夺回所有权，让 Rust 安全 Drop 掉这块内存（Vec 的 dealloc）
        let _payload = unsafe { Box::from_raw(ctx) };
    }
    // shell 一律回收（ctx 为空的畸形输入也回收，否则该路径泄漏 DLManagedTensor）
    let _managed = unsafe { Box::from_raw(managed_tensor) };
}

/// PyCapsule 析构器（#22a）：capsule 被 GC 时触发，兜底回收未消费的 DLManagedTensor。
///
/// DLPack 协议约定消费端正常消费后**自行调用 `DLManagedTensor.deleter` 并把 capsule
/// 名字改掉**（不再是 "dltensor"，见 PyTorch 的 `torch.utils.dlpack` 实现）。因此本
/// 析构器只在 capsule 仍名为 "dltensor"（即消费端未取走资源）时兜底释放。
///
/// #38 真 mmap 落地后泄漏形态从堆内存升级为**映射页**（每次一个 shard 大小的虚拟映射，
/// 不会 SIGBUS 但累积致命）—— 本析构器是其硬前置；在此先以堆形态实现，语义不变。
///
/// ★ N1（20261007 复核）："释放前把 capsule 标记为已消费"只能靠**改名**，
///   不能靠 `PyCapsule_SetPointer(NULL)`——ctypes 实测（Python 3.12.9）：
///   - `PyCapsule_SetPointer(cap, NULL)` 返回 **-1** 并设置
///     `ValueError: PyCapsule_SetPointer called with null pointer`（CPython 显式
///     拒绝 null pointer），**指针根本不会被清空**——原实现既没起到标记作用，
///     又往解释器里留了一个永不清除的异常；
///   - `PyCapsule_SetName(cap, "used_dltensor")` 返回 0 无异常，是 PyTorch
///     同款惯例；改名后 `PyCapsule_GetPointer(cap, "dltensor")` 因名字不匹配
///     返回 NULL——重入（或消费端已接管后的 GC）天然 no-op。
///
///   另：`PyCapsule_GetPointer` 名字不匹配时**也会**设置 ValueError
///   （`PyCapsule_GetPointer called with incorrect name`，复核实测新发现），
///   因此提前返回路径必须先 `PyErr_Clear()`，否则异常会泄进解释器、污染后续
///   任意一次 Python API 调用（表现为远处莫名的 ValueError）。
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() {
        return;
    }
    // ★ 评审 §3.4(b)：判定"是否被消费"直接用 `PyCapsule_GetPointer(capsule, "dltensor")`
    //   —— 它本身就做"名字精确匹配 + 取指针"两件事。
    //   旧实现是"PyCapsule_GetName + CStr::to_str().is_ok_and(...)"两步，其中：
    //     1) `GetName` 返回 NULL（错误态）时 unconsumed=false → 直接 return → **泄漏**；
    //     2) `to_str()` 对非 UTF-8 名字返回 Err → 同样走 return；
    //     3) 两处都依赖"名字是 UTF-8"这个不必要的假设。
    //   现在单步完成，且顺序颠倒为"先改名、后释放"：
    //   万一析构器被重入，第二次看到名字不匹配直接 no-op，不会二次释放。
    let managed =
        unsafe { PyCapsule_GetPointer(capsule, c"dltensor".as_ptr()) } as *mut DLManagedTensor;
    if managed.is_null() {
        // 名字不是 "dltensor"（消费端已按协议改名并接管所有权，或本析构器重入时
        // 自己已改名）或 capsule 畸形 → 不介入。
        // ★ N1：GetPointer 名字不匹配时返回 NULL **且设置 ValueError**——
        //   不清除就会把异常留给解释器里下一个无辜的 API 调用方。
        unsafe { PyErr_Clear() };
        return;
    }
    // 先改名再释放：把"重入"从 double-free 降级为安全 no-op。
    // 旧实现在释放**之后**才清空、且用的 SetPointer(NULL) 根本不生效、并忽略返回值。
    let renamed = unsafe { PyCapsule_SetName(capsule, c"used_dltensor".as_ptr()) };
    if renamed != 0 {
        // 改名失败意味着 capsule 状态异常；资源指针已经摘走，不释放就是泄漏，
        // 只能继续释放并记日志。残留异常一并清除（此时 raise 出去只会打断
        // 无关调用方——GC 析构路径没有合理的异常上报对象）。
        tracing::error!(
            "capsule_destructor: PyCapsule_SetName(\"used_dltensor\") failed (returned {renamed}); \
             proceeding with reclaim"
        );
        unsafe { PyErr_Clear() };
    }
    unsafe { dlpack_deleter(managed) };
}

/// 将 Rust 的 `Vec<u8>` 字节流按给定 dtype/shape 零拷贝转换为 DLPack 胶囊。
///
/// 泛型化基础层：调用方自带 dtype 与 shape/strides 描述，`data` 以字节流形式
/// 交给 DLPack（零拷贝所有权转移，deleter 负责回收）。i32 便利封装见
/// [`vec_to_dlpack_capsule`]；新增 i8/f16 等列 dtype 时只需提供对应
/// `DLDataType`，无需再动 FFI 层。
pub fn bytes_to_dlpack_capsule(
    py: Python,
    data: Vec<u8>,
    dtype: DLDataType,
    shape: Vec<i64>,
    strides: Vec<i64>,
) -> PyResult<Py<PyAny>> {
    payload_to_capsule(
        py,
        TensorPayload {
            data,
            shared: None,
            dtype,
            shape,
            strides,
        },
    )
}

/// **P1-T6（D8）**共享缓冲的零拷贝 view capsule：`buffer` 的
/// `[offset, offset+len)` 区间以 u8 一维张量透出（图像压缩字节流）。
///
/// - **零拷贝**：DLTensor.data 直接指向共享缓冲内部，payload 只携带
///   Arc 引用 + 元数据，`copy_count` 不增；
/// - **驱逐保护**：payload 持有 Arc 强引用，tensor 存活期 = buffer 存活期；
/// - **单次释放**：多 view 各自 drop（deleter 各一次），底层 buffer 仅在
///   最后一个引用消失时 Drop 一次（`release_count` 断言锚点）。
///
/// dtype/shape 由调用方声明（图像压缩字节 = u8 一维；未来扩展列自带描述）。
pub fn shared_bytes_view_to_dlpack_capsule(
    py: Python,
    buffer: Arc<SharedBlobBuffer>,
    offset: usize,
    len: usize,
    dtype: DLDataType,
) -> PyResult<Py<PyAny>> {
    if offset.saturating_add(len) > buffer.data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shared blob view [{offset}, {}) out of bounds (buffer len {})",
            offset + len,
            buffer.data.len()
        )));
    }
    payload_to_capsule(
        py,
        TensorPayload {
            data: Vec::new(), // 空占位：真实数据在 shared（DLTensor.data 指向 view 起点）
            shared: Some((buffer, offset)),
            dtype,
            shape: vec![len as i64],
            strides: vec![1],
        },
    )
}

/// payload → DLManagedTensor → capsule 的公共装配路径（评审 §3.4(c) 的
/// "先 into_raw 再取指针"不变量在这里集中维护；Owned / Shared 两种形态
/// 只在 `DLTensor.data` 指针来源上分叉）。
fn payload_to_capsule(py: Python, payload: TensorPayload) -> PyResult<Py<PyAny>> {
    // SAFETY: payload 先 into_raw（堆上稳定不动，结构上不可能 realloc），再通过
    //   裸指针读字段填 DLTensor；读完立刻把所有权打包进 DLManagedTensor 交给 capsule。
    let payload_ptr = Box::into_raw(Box::new(payload));
    let dl_tensor = unsafe {
        let p = &mut *payload_ptr;
        let data_ptr: *mut c_void = match &p.shared {
            Some((buf, off)) => buf.data[*off..].as_ptr() as *mut c_void,
            None => p.data.as_mut_ptr() as *mut c_void,
        };
        DLTensor {
            data: data_ptr,
            device: DLDevice {
                device_type: 1,
                device_id: 0,
            }, // 1 = CPU
            ndim: p.shape.len() as i32,
            dtype: p.dtype,
            shape: p.shape.as_mut_ptr(),
            strides: p.strides.as_mut_ptr(),
            byte_offset: 0,
        }
    };

    let payload_ptr = payload_ptr as *mut c_void;

    let managed_tensor = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: payload_ptr,
        deleter: Some(dlpack_deleter),
    });

    let managed_ptr = Box::into_raw(managed_tensor);
    // 必须使用 'static 生命周期的 C 字符串，因为 PyCapsule_New 仅借用指针，name 在函数返回后会被释放。
    let name = c"dltensor";

    // ★ #22a：补 capsule 析构器（第三参 None → Some(capsule_destructor)）。
    //   此前 capsule 无析构器，"创建后未被消费"路径会泄漏 DLManagedTensor + payload；
    //   #38 mmap 后泄漏形态升级为映射页，本析构器是其硬前置。
    let capsule = unsafe {
        PyCapsule_New(
            managed_ptr as *mut c_void,
            name.as_ptr() as *const _,
            Some(capsule_destructor),
        )
    };
    if capsule.is_null() {
        // SAFETY: capsule 创建失败意味着引用链断裂（DLPack 协议下 capsule 是唯一
        // 所有权入口），两个 Box 已 into_raw 且无人再持有——必须手动回收，
        // 否则 DLManagedTensor 与 payload（含 Vec<i32> 数据）双重泄漏。
        // Shared 形态的 Arc 引用随 payload Drop 归还（不泄漏共享缓冲）。
        unsafe {
            drop(Box::from_raw(managed_ptr));
            drop(Box::from_raw(payload_ptr as *mut TensorPayload));
        }
        return Err(pyo3::exceptions::PyValueError::new_err(
            "PyCapsule_New returned NULL",
        ));
    }

    // DLPack capsule 的零拷贝生命周期绑定到自定义 deleter，暂不迁移 pyo3 Bound API
    #[allow(deprecated)]
    let capsule_obj: Py<PyAny> = unsafe { Py::from_owned_ptr(py, capsule) };
    Ok(capsule_obj)
}

/// 便利封装：`Vec<i32>`（batch_size × seq_len 的 input_ids）零拷贝转 DLPack 胶囊。
///
/// i32 → 字节流零拷贝（`bytemuck::try_cast_vec`，降宽转换不搬数据、只调整容量记账）；
/// i8/f16 等新列 dtype 请直接调 [`bytes_to_dlpack_capsule`]。
pub fn vec_to_dlpack_capsule(
    py: Python,
    data: Vec<i32>,
    batch_size: usize,
    seq_len: usize,
) -> PyResult<Py<PyAny>> {
    let bytes = match bytemuck::try_cast_vec(data) {
        Ok(bytes) => bytes,
        // 失败只可能是对齐/大小不变量被破坏（i32→u8 理论上恒可降宽），退化为拷贝兜底
        Err((_, data)) => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    bytes_to_dlpack_capsule(
        py,
        bytes,
        DLDataType {
            code: 0,
            bits: 32,
            lanes: 1,
        }, // 0=Int, 32bit
        vec![batch_size as i64, seq_len as i64],
        vec![seq_len as i64, 1],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// deleter 回收路径：构造真实布局的 payload + DLManagedTensor，调用 deleter，
    /// 验证"夺回所有权 → Drop"链路不 panic（泄漏由 ASAN/miri 构建兜底检测）。
    #[test]
    fn deleter_reclaims_payload_and_tensor() {
        let mut payload = Box::new(TensorPayload {
            data: vec![1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
            shared: None,
            dtype: DLDataType {
                code: 0,
                bits: 32,
                lanes: 1,
            },
            shape: vec![2, 2],
            strides: vec![2, 1],
        });
        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: payload.data.as_mut_ptr() as *mut c_void,
                device: DLDevice {
                    device_type: 1,
                    device_id: 0,
                },
                ndim: 2,
                dtype: payload.dtype,
                shape: payload.shape.as_mut_ptr(),
                strides: payload.strides.as_mut_ptr(),
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(payload) as *mut c_void,
            deleter: Some(dlpack_deleter),
        });
        let managed_ptr = Box::into_raw(managed);
        unsafe { dlpack_deleter(managed_ptr) };
        // 回收后同量级分配应正常可用（冒烟：堆未损坏）
        let _probe: Vec<u8> = vec![0u8; 4096];
    }

    /// 畸形输入不 panic：null managed_tensor / null manager_ctx（纵深防御路径）。
    #[test]
    fn deleter_null_paths_are_noop() {
        unsafe {
            dlpack_deleter(std::ptr::null_mut());
            let managed = Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: DLDevice {
                        device_type: 1,
                        device_id: 0,
                    },
                    ndim: 0,
                    dtype: DLDataType {
                        code: 0,
                        bits: 32,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(dlpack_deleter),
            });
            dlpack_deleter(Box::into_raw(managed));
        }
    }

    /// ★ 评审 §3.4(d) 的单次调用回归：deleter 走"先置空 manager_ctx、再释放
    ///   payload 与 shell"的路径，第一次调用即干净回收（不 panic、不损坏堆）。
    ///
    /// **命名/注释诚实化**（短期项 #5）：原测试名 `deleter_is_reentrancy_safe` 与
    /// 注释宣称"第二次调用不会二次 free"，但代码里**并没有第二次调用**——字面的
    /// 二次 deleter 调用是对已 free 的 shell 解引用（UAF），无法在单测里安全执行，
    /// 那个不变量只能由 capsule 级测试覆盖（见下面
    /// `capsule_destructor_noops_after_consumer_rename`：改名后析构器 no-op，
    /// 全程不触碰已释放指针）。本测试只保留它真实断言的部分。
    #[test]
    fn deleter_single_call_reclaims_cleanly() {
        let mut payload = Box::new(TensorPayload {
            data: vec![7u8; 32],
            shared: None,
            dtype: DLDataType {
                code: 0,
                bits: 32,
                lanes: 1,
            },
            shape: vec![2, 4],
            strides: vec![4, 1],
        });
        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: payload.data.as_mut_ptr() as *mut c_void,
                device: DLDevice {
                    device_type: 1,
                    device_id: 0,
                },
                ndim: 2,
                dtype: payload.dtype,
                shape: payload.shape.as_mut_ptr(),
                strides: payload.strides.as_mut_ptr(),
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(payload) as *mut c_void,
            deleter: Some(dlpack_deleter),
        });
        // 第一次（也是唯一一次安全可执行的）调用：ctx 非空 → 回收 payload + shell。
        // 若"先置空 ctx 再释放"的顺序被破坏，本测试仍可能通过（单次调用无观测点），
        // 该不变量的真正回归在 capsule 级重入测试（pyo3-tests 特性下）。
        let ptr = Box::into_raw(managed);
        unsafe { dlpack_deleter(ptr) };
        // 同量级分配应正常可用（冒烟：堆未损坏）
        let _probe: Vec<u8> = vec![0u8; 4096];
    }

    /// capsule 创建 → drop 触发 deleter 全链路（需要 Python 解释器）。
    /// pyo3 `extension-module` 特性下测试二进制不链接 libpython，
    /// 因此仅在 `pyo3-tests` 特性下编译运行：
    /// `cargo test --features pyo3-tests ffi`
    #[cfg(feature = "pyo3-tests")]
    #[test]
    fn capsule_create_and_drop_roundtrip() {
        use pyo3::types::PyCapsule;
        // pyo3 0.28：`with_gil` → `Python::attach`；`auto-initialize` 特性下
        // 首次 attach 自动初始化解释器，无需 prepare_freethreaded_python
        Python::attach(|py| {
            let capsule = vec_to_dlpack_capsule(py, vec![1, 2, 3, 4, 5, 6], 2, 3)
                .expect("capsule creation failed");
            // DLPack 协议要求 capsule 名为 "dltensor"（PyTorch 按名校验）
            let obj = capsule.bind(py).as_ptr();
            let name = unsafe { pyo3::ffi::PyCapsule_GetName(obj) };
            let name = unsafe { std::ffi::CStr::from_ptr(name) };
            assert_eq!(name.to_str().unwrap(), "dltensor");
            assert!(capsule.bind(py).is_instance_of::<PyCapsule>());
            drop(capsule); // 触发 deleter：回收 DLManagedTensor + payload
            let capsule2 = vec_to_dlpack_capsule(py, vec![0; 8], 2, 4)
                .expect("second capsule creation failed");
            drop(capsule2);
        })
    }

    /// ★ N1 capsule 级重入回归（模拟 PyTorch 的完整消费路径）：
    /// 消费端按 DLPack 协议"取指针 → 改名 → 自行调 deleter"接管所有权后，
    /// capsule 被 GC 时析构器必须 **no-op**（不二次释放）**且不留 Python 异常**。
    ///
    /// 这是 `deleter_single_call_reclaims_cleanly` 覆盖不到的真正重入路径：
    /// 这里 capsule 仍持有（已悬空的）指针，但名字已不是 "dltensor" →
    /// `PyCapsule_GetPointer` 返回 NULL + 设置 ValueError → 析构器清除异常后
    /// 直接返回，全程不触碰已释放的 shell。N1 修复前，残留的 ValueError 会泄进
    /// 解释器，污染后续任意一次 Python API 调用。
    #[cfg(feature = "pyo3-tests")]
    #[test]
    fn capsule_destructor_noops_after_consumer_rename() {
        Python::attach(|py| {
            let capsule =
                vec_to_dlpack_capsule(py, vec![9i32, 8, 7], 3, 1).expect("capsule creation failed");
            let obj = capsule.bind(py).as_ptr();
            // 模拟 PyTorch 消费端：按名取指针 → 改名 → 自行调 deleter（接管所有权）
            let managed = unsafe { PyCapsule_GetPointer(obj, c"dltensor".as_ptr()) };
            assert!(!managed.is_null(), "消费端应能按 \"dltensor\" 名字取到指针");
            let renamed = unsafe { PyCapsule_SetName(obj, c"used_dltensor".as_ptr()) };
            assert_eq!(renamed, 0, "PyTorch 同款改名应成功");
            unsafe { dlpack_deleter(managed as *mut DLManagedTensor) };
            // capsule drop → 析构器：名字不匹配 → no-op。
            // N1 关键断言：不得留下 Python 异常（旧实现会漏一个 ValueError）。
            drop(capsule);
            assert!(
                unsafe { pyo3::ffi::PyErr_Occurred() }.is_null(),
                "capsule_destructor 提前返回路径必须清除 GetPointer 设置的 ValueError"
            );
        })
    }

    // ==================== P1-T6（D8）共享 blob 生命周期 ====================

    #[cfg(feature = "pyo3-tests")]
    const U8_DTYPE: DLDataType = DLDataType {
        code: 1,
        bits: 8,
        lanes: 1,
    }; // 1=UInt

    #[cfg(feature = "pyo3-tests")]
    fn make_shared_blob(
        data: Vec<u8>,
    ) -> (Arc<SharedBlobBuffer>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let release_count = Arc::new(AtomicUsize::new(0));
        let copy_count = Arc::new(AtomicUsize::new(0));
        let buf = Arc::new(SharedBlobBuffer {
            data,
            release_count: release_count.clone(),
            copy_count: copy_count.clone(),
        });
        (buf, release_count, copy_count)
    }

    /// 从 capsule 取回 DLManagedTensor 的 data 指针并读前 n 字节（模拟消费端
    /// tensor 读取——不依赖 torch，直接按 DLPack ABI 验证数据可读且正确）。
    #[cfg(feature = "pyo3-tests")]
    unsafe fn tensor_bytes(capsule: &Py<PyAny>, py: Python, n: usize) -> Vec<u8> {
        let obj = capsule.bind(py).as_ptr();
        let managed =
            unsafe { PyCapsule_GetPointer(obj, c"dltensor".as_ptr()) } as *mut DLManagedTensor;
        assert!(!managed.is_null(), "capsule 应仍持未消费的 dltensor");
        unsafe { PyErr_Clear() };
        let t = unsafe { &(*managed).dl_tensor };
        let base = t.data as *const u8;
        unsafe { std::slice::from_raw_parts(base, n).to_vec() }
    }

    /// D8 负面测试 1+3：多 view 共享 blob——每个 view 一个 capsule（deleter
    /// 各调一次），底层 buffer **只释放一次**（release_count == 1）；
    /// 全链路 `copy_count` 恒 0（零额外拷贝）。
    #[cfg(feature = "pyo3-tests")]
    #[test]
    fn shared_blob_multiple_views_single_release_zero_copies() {
        Python::attach(|py| {
            let (buf, release_count, copy_count) =
                make_shared_blob(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            // 3 个 view（模拟一 batch 多图共享同一 blob 缓冲）
            let v0 = shared_bytes_view_to_dlpack_capsule(py, buf.clone(), 0, 4, U8_DTYPE).unwrap();
            let v1 = shared_bytes_view_to_dlpack_capsule(py, buf.clone(), 4, 4, U8_DTYPE).unwrap();
            let v2 = shared_bytes_view_to_dlpack_capsule(py, buf.clone(), 8, 2, U8_DTYPE).unwrap();
            assert_eq!(buf.data.len(), 10);
            assert_eq!(
                copy_count.load(Ordering::Acquire),
                0,
                "capsule 链路不得深拷贝"
            );

            // 各 view 数据正确（区间切分无误）
            assert_eq!(unsafe { tensor_bytes(&v0, py, 4) }, vec![1, 2, 3, 4]);
            assert_eq!(unsafe { tensor_bytes(&v1, py, 4) }, vec![5, 6, 7, 8]);
            assert_eq!(unsafe { tensor_bytes(&v2, py, 2) }, vec![9, 10]);

            // 数据侧引用全部消失（eviction 语义）：只要 tensor 还活着 buffer 不释放
            drop(buf);
            assert_eq!(release_count.load(Ordering::Acquire), 0);

            // 逐个 drop（模拟 tensor 各自生命周期结束）：deleter 各触发一次，
            // 但 buffer 在最后一个引用消失前不得释放。
            drop(v0);
            assert_eq!(
                release_count.load(Ordering::Acquire),
                0,
                "仍有 view 持有 buffer"
            );
            drop(v1);
            assert_eq!(release_count.load(Ordering::Acquire), 0);
            drop(v2);
            assert_eq!(
                release_count.load(Ordering::Acquire),
                1,
                "多 view 共享 blob 必须只释放一次（deleter 计数 == 1 的 buffer 级等价）"
            );
            assert_eq!(copy_count.load(Ordering::Acquire), 0);
        })
    }

    /// D8 负面测试 1：eviction 后 tensor 仍安全——原始所有者（BatchData 侧的
    /// Arc 强引用）全部 drop（模拟缓存驱逐/缓冲归还），只要 tensor（capsule
    /// payload 的 Arc）还活着，读旧 tensor 必须不 crash、不读到复用后的脏数据。
    #[cfg(feature = "pyo3-tests")]
    #[test]
    fn tensor_survives_owner_drop_and_buffer_reuse() {
        Python::attach(|py| {
            let (buf, release_count, _copy) = make_shared_blob(vec![42u8; 16]);
            let cap = shared_bytes_view_to_dlpack_capsule(py, buf, 0, 16, U8_DTYPE).unwrap();
            // 模拟驱逐：原始引用全部消失（Arc 只剩 capsule payload 一个持有者）
            // —— reader 侧继续迭代/重写同位置内存也无法触达这块 buffer。
            let sentinel: Vec<u8> = vec![7u8; 16]; // 同量级分配（冒烟：旧内存若被复用会变 7）
            drop(sentinel);
            assert_eq!(
                release_count.load(Ordering::Acquire),
                0,
                "tensor 存活期 buffer 不得释放"
            );
            assert_eq!(
                unsafe { tensor_bytes(&cap, py, 16) },
                vec![42u8; 16],
                "eviction 后 tensor 数据必须原样"
            );
            drop(cap);
            assert_eq!(release_count.load(Ordering::Acquire), 1);
        })
    }

    /// D8 负面测试 4（失败/部分失败路径）：view 创建中途越界报错时，已创建的
    /// view 正常持有 buffer（Arc 计数正确），错误不留悬挂——调用方 drop 已有
    /// capsule 后 buffer 精确释放一次；越界请求本身零分配、零泄漏。
    #[cfg(feature = "pyo3-tests")]
    #[test]
    fn failed_view_creation_leaves_no_leak_and_prior_views_intact() {
        Python::attach(|py| {
            let (buf, release_count, _copy) = make_shared_blob(vec![1u8; 8]);
            let ok = shared_bytes_view_to_dlpack_capsule(py, buf.clone(), 0, 8, U8_DTYPE).unwrap();
            // 失败路径：越界 view 创建被拒（Err），不影响已创建的 view
            let err = shared_bytes_view_to_dlpack_capsule(py, buf.clone(), 4, 8, U8_DTYPE);
            assert!(err.is_err(), "越界 view 必须报错而非静默截断");
            assert_eq!(
                release_count.load(Ordering::Acquire),
                0,
                "成功 view 仍持有 buffer"
            );
            assert_eq!(
                Arc::strong_count(&buf),
                2, // 调用方 + 成功 view 的 payload
                "失败请求不得改变引用计数"
            );
            drop(ok);
            drop(buf);
            assert_eq!(
                release_count.load(Ordering::Acquire),
                1,
                "全路径结束后 buffer 恰好释放一次"
            );
        })
    }
}

pub(crate) mod align;
pub(crate) mod ffi_dlpack;
pub(crate) mod mixer;
pub(crate) mod mm_sidecar;
pub(crate) mod sampler;
pub(crate) mod stream;
pub(crate) mod video_blob; // P3b：video blob 按需读（进程级句柄注册表 + FFI 读原语） // P3c：三模态对齐 join（ts token 解码 + (sid,window) 归并）

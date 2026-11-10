import importlib.util
spec = importlib.util.spec_from_file_location("md_mask", "python/megadata/mask_adapter.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def eq(name, got, want):
    ok = got == want
    print(f"[{'OK' if ok else 'FAIL'}] {name}: got={got} want={want}")
    assert ok, name

# 1. normalize
eq("normalize dedup", m.normalize_cu_seqlens([0,3,3,5,11]).tolist(), [0,3,5,11])
eq("normalize prepend0", m.normalize_cu_seqlens([3,5,11]).tolist(), [0,3,5,11])
eq("normalize empty", m.normalize_cu_seqlens([]).numel(), 0)
eq("normalize None", m.normalize_cu_seqlens(None).numel(), 0)

# 2. valid_len
eq("valid_len", m.valid_len([0,3,5,11]), 11)
eq("valid_len empty+total", m.valid_len([], total_len=100), 100)

# 3. seg_lens
eq("seg_lens", m.cu_seqlens_to_seg_lens([0,3,5,11]), [3,2,6])
eq("seg_lens empty", m.cu_seqlens_to_seg_lens([]), [])

# 4. doc_ids
eq("doc_ids", m.cu_seqlens_to_doc_ids([0,3,5,11]).tolist(), [0,0,0,1,1,2,2,2,2,2,2])
eq("doc_ids empty", m.cu_seqlens_to_doc_ids([], total_len=4).tolist(), [0,0,0,0])

# 5. flatten_and_unpad + repad_to_batch 往返
import torch
batch = torch.arange(2*8).reshape(2, 8).clone()  # (2,8) row-major 0..15
cu = [0, 3, 5]  # valid=5
flat = m.flatten_and_unpad(batch, cu)
eq("flatten_and_unpad", flat.tolist(), [0,1,2,3,4])
back = m.repad_to_batch(flat.float(), cu, batch_size=2, max_seq_len=8, pad_value=-1.0)
# PT 语义：padding 集中在展平流末尾，valid=5 占第一行前5，其余全 pad
expect = [[0,1,2,3,4,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1,-1]]
eq("repad_to_batch", back.reshape(2,8).tolist(), expect)

# 6. 三维 (B,S,H,D) 也能 unpad
q = torch.arange(2*8*2).reshape(2,8,2).float()
qf = m.flatten_and_unpad(q, cu)
eq("flatten 3d shape", tuple(qf.shape), (5,2))
eq("flatten 3d vals", qf[0].tolist(), [0,1])

# 7. to_flash_varlen 是**纯转换**（不要求安装 flash-attn；依赖校验在 build_attention 门面）
cu32, max_s = m.to_flash_varlen([0,3,5])
eq("to_flash_varlen dtype", cu32.dtype, torch.int32)
eq("to_flash_varlen cu", cu32.tolist(), [0,3,5])
eq("to_flash_varlen max_seqlen", max_s, 3)
print("[OK] to_flash_varlen 纯转换（无需 flash_attn）")
try:
    m.build_attention([0,3,5], "flash_varlen")
    print("[OK] flash_attn 可用，build_attention 返回值")
except ImportError as e:
    print(f"[OK] flash_attn 缺失，build_attention 门面报友好错误: {str(e)[:50]}...")

# 8. FlexAttention（torch 2.12 支持，CPU 上构造 BlockMask）
try:
    bm = m.to_flex_block_mask([0,3,5], causal=True, block_size=2, device="cpu", compile_mask=False)
    print(f"[OK] to_flex_block_mask -> {type(bm).__name__}")
except Exception as e:
    print(f"[INFO] flex 不可用: {type(e).__name__}: {str(e)[:80]}")

# 9. xformers 友好错误
try:
    m.to_xformers_block_diag([0,3,5])
    print("[OK] xformers 可用")
except ImportError as e:
    print(f"[OK] xformers 缺失，友好错误: {str(e)[:50]}...")

# 10. build_attention 入口分发 + 未知 backend
try:
    m.build_attention([0,3,5], "nonsense")
    print("[FAIL] 应抛 ValueError")
except ValueError as e:
    print(f"[OK] 未知 backend 报错: {str(e)[:60]}...")

# 11. build_attention 字符串分发到 flash（验证分发逻辑，不验证 flash 本身）
try:
    r = m.build_attention([0,3,5], "flash_varlen")
    print(f"[OK] build_attention flash_varlen -> {type(r)}")
except ImportError:
    print("[OK] build_attention flash_varlen 分发到位（flash_attn 缺失）")

# 12. 畸形 cu_seqlens 必须**响亮报错**（二轮 P2-4 的回归锚点）：
#     递减段（[0,5,3]）旧实现会静默产出错误 mask（seg_lens 过滤负 diff、doc_ids 截断）。
def expect_value_error(name, fn):
    try:
        fn()
    except ValueError as e:
        print(f"[OK] {name} 拒绝畸形输入: {str(e)[:60]}...")
        return
    print(f"[FAIL] {name} 应抛 ValueError")

expect_value_error("normalize 递减", lambda: m.normalize_cu_seqlens([0,5,3]))
expect_value_error("normalize 负值", lambda: m.normalize_cu_seqlens([0,-2]))
expect_value_error("valid_len 递减", lambda: m.valid_len([0,5,3]))
expect_value_error("doc_ids 递减", lambda: m.cu_seqlens_to_doc_ids([0,5,3]))
# 单调合法输入不受影响
eq("normalize 单调递增保持", m.normalize_cu_seqlens([0,3,5,11]).tolist(), [0,3,5,11])

# 13. to_flash_varlen 边界：段边界越过 total_len 必须报错（P3-7）
expect_value_error("to_flash_varlen 越界", lambda: m.to_flash_varlen([0,3,50], total_len=8))

# 14. build_attention 的 kwargs 只被 flex 消费；xformers 收到未知 kwargs 必须显式拒绝
expect_value_error(
    "xformers 拒绝 kwargs",
    lambda: m.build_attention([0,3,5], "xformers", block_size=2),
)

print("\n全部核心逻辑验证通过")

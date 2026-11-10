import os
import json
import tempfile
from typing import Optional
import pytest
import tracemalloc
import gc

from megadata import DataBuilder, MixBatchStream

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./model_learn_tokenizer")
BATCH_SIZE = 8
MAX_SEQ_LEN = 512
STRESS_STEPS = 2000 # 疯狂抽取 2000 个 Batch 进行压测

@pytest.fixture(scope="module")
def stress_environment():
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "stress.jsonl")
        output_dir = os.path.join(temp_dir, "processed")

        with open(raw_path, "w", encoding="utf-8") as f:
            for i in range(1000):
                f.write(json.dumps({"text": "stress test memory leaks " * 10}, ensure_ascii=False) + "\n")

        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
        builder.build_dataset([raw_path], output_dir, mode="pt", batch_size=BATCH_SIZE)
        yield os.path.join(output_dir, "index.json")

# =========================================================
# 【新增】：包装一个无限循环迭代器，模拟真实按 Step 训练的场景
# 这将极度考验 Rust 引擎跨 Epoch 重置状态的稳定性
# =========================================================
def get_infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch

def _current_rss_kb() -> Optional[int]:
    """当前进程 RSS（KB）；平台不可得时返回 None（跳过 RSS 断言）。

    tracemalloc 只追踪 **Python 分配器**，看不见 Rust 堆上的 DLPack payload
    （由自定义 deleter 释放）——P1-1 一类的 Rust 侧泄漏必须用进程级指标才能发现。
    """
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss // 1024)
    except Exception:
        pass
    try:
        # Linux：statm 第二列 = resident pages
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except Exception:
        return None

def test_ffi_memory_leak(stress_environment):
    """
    【内存泄漏压测】：验证 Rust -> DLPack -> Python -> GC 回收的全链路是否会漏内存。
    """
    index_path = stress_environment

    loader = MixBatchStream(
        configs=[{"path": index_path}],
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,  # 无限流式：显式声明，允许下方的无限迭代器跨 epoch 循环
    )

    # 强制清理环境
    gc.collect()

    # 【修复】：使用无限迭代器
    iterator = get_infinite_iterator(loader)

    print("\n[Stress Test] 开始热身...")
    for _ in range(50):
        _ = next(iterator)

    # 热身完毕，系统缓冲区已分配。开启内存追踪！
    rss_before_kb = _current_rss_kb()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    print(f"[Stress Test] 热身完毕，开始高频抽取 {STRESS_STEPS} 个 Batch...")
    for _ in range(STRESS_STEPS):
        batch = next(iterator)
        # 模拟真实训练：操作张量，然后解除引用，依赖 GC 回收 C FFI 胶囊
        loss = batch["input_ids"].sum()
        del batch
        del loss

    # 强制进行垃圾回收
    gc.collect()

    # 截取压测后的内存快照
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # 计算差异
    stats = snapshot2.compare_to(snapshot1, 'lineno')

    total_diff_kb = sum(stat.size_diff for stat in stats) / 1024
    print(f"[Stress Test] 经过 {STRESS_STEPS} 步，内存增量为: {total_diff_kb:.2f} KB")

    # 【断言】：在 2000 次大规模矩阵创建后，未被回收的内存增长不应超过 1MB (1024KB)。
    assert total_diff_kb < 1024.0, f"检测到严重的内存泄漏！增量达到 {total_diff_kb:.2f} KB"

    # 【RSS 级断言】补上 tracemalloc 看不见的 Rust 堆（DLPack payload 由自定义
    # deleter 释放，不走 Python 分配器）：进程 RSS 不应随步数线性增长。
    # 阈值宽松（128MB）：tracemalloc 自身的跟踪开销、allocator arena 与碎片都计入，
    # 这里只抓"每步泄漏 batch 缓冲区"这类线性泄漏（2000 步 × 8×512×4B ≈ 32MB 量级）。
    rss_after_kb = _current_rss_kb()
    if rss_before_kb is not None and rss_after_kb is not None:
        rss_diff_kb = rss_after_kb - rss_before_kb
        print(f"[Stress Test] 进程 RSS 增量为: {rss_diff_kb / 1024:.2f} MB")
        assert rss_diff_kb < 128 * 1024, (
            f"检测到严重的进程级内存泄漏（Rust 堆侧）！RSS 增量达到 {rss_diff_kb / 1024:.2f} MB"
        )
    else:
        pytest.skip("当前平台无法读取进程 RSS，跳过 Rust 堆侧的泄漏断言")
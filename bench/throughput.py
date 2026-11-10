"""
megadata_rs 吞吐量基准测试

为 README 中"零拷贝极限吞吐"的宣称提供可复现的公开数字。

用法:
    # 单流基准
    python bench/throughput.py --index data/pt_processed/index.json --steps 1000

    # 多源混采基准
    python bench/throughput.py --index data/pt/index.json data/sft/index.json --steps 1000

    # 自定义 batch / seq_len
    python bench/throughput.py --index data/pt/index.json --batch-size 32 --max-seq-len 4096 --steps 2000

指标:
    - batches/sec: 每秒产出 batch 数
    - tokens/sec:  每秒产出 token 数（batch_size * max_seq_len * batches/sec）
    - p50/p99 latency: 单次 next_batch 延迟（毫秒）
"""
import argparse
import time

from megadata import MultiSourceMixLoader


def bench(index_paths, steps, batch_size, max_seq_len, num_epochs):
    configs = [{"path": p} for p in index_paths]
    loader = MultiSourceMixLoader(
        datasets_config=configs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_epochs=num_epochs or 1,
    )

    latencies = []
    total_tokens = 0

    # 预热：让后台预取线程填满 pipeline、OS page cache 就位
    warmup_steps = min(50, steps)
    print(f"[warmup] 预热 {warmup_steps} 步...")
    it = iter(loader)
    for _ in range(warmup_steps):
        try:
            next(it)
        except StopIteration:
            break

    print(f"[bench] 开始测量 {steps} 步...")
    it = iter(loader)
    measured = 0
    start = time.perf_counter()
    for i in range(steps):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            print(f"[bench] 在第 {i} 步提前耗尽，实际测量 {i} 步")
            break
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
        total_tokens += batch["input_ids"].numel()
        measured += 1
    end = time.perf_counter()

    elapsed = end - start
    batches_per_sec = measured / elapsed if elapsed > 0 else 0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    print("\n========== megadata_rs 吞吐量基准 ==========")
    print(f"配置: batch_size={batch_size}, max_seq_len={max_seq_len}, streams={len(index_paths)}")
    print(f"测量步数: {measured}")
    print(f"总耗时: {elapsed:.3f} s")
    print(f"吞吐: {batches_per_sec:.2f} batches/sec")
    print(f"吞吐: {tokens_per_sec:,.0f} tokens/sec ({tokens_per_sec / 1e6:.2f}M tokens/sec)")
    if latencies:
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p99 = lat_sorted[min(int(len(lat_sorted) * 0.99), len(lat_sorted) - 1)]
        print(f"延迟: p50={p50:.2f}ms, p99={p99:.2f}ms, max={max(latencies):.2f}ms")
    print("============================================")


def main():
    parser = argparse.ArgumentParser(description="megadata_rs 吞吐量基准")
    parser.add_argument("--index", nargs="+", required=True, help="一个或多个 index.json 路径")
    parser.add_argument("--steps", type=int, default=1000, help="测量步数 (默认 1000)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--num-epochs", type=int, default=0, help="0=无限流(测到 steps 为止)")
    args = parser.parse_args()

    bench(
        index_paths=args.index,
        steps=args.steps,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()

"""仿真器测试：合成 plan 的纯函数语义 + 真实 index.json 的 plan 一致性 + CLI 冒烟。

仿真模型假设见 ``megadata.simulation`` 模块 docstring。
"""
import json
import os
import tempfile

import pytest

from megadata import DataBuilder
from megadata.simulation import (
    SimulationConfig,
    format_report_markdown,
    format_sweep_markdown,
    load_shard_sizes,
    main,
    plan_from_rust,
    simulate,
    simulate_plan,
    sweep_cache,
)

TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH",
    os.environ.get("BBPE_TOKENIZER_DIR", "./tokenizer"),
)
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)
BATCH_SIZE = 4
MAX_SEQ_LEN = 64
TOTAL_DOCS = 400
BATCHES_PER_SHARD = 4


def _cfg(**kw) -> SimulationConfig:
    base = dict(
        index_path="",
        world_size=1,
        num_nodes=1,
        num_workers=0,
        seed=42,
        epochs=1,
        shuffle="permutation",
        partition="round_robin",
        num_canonical_nodes=None,
        bandwidth_mbps=1000.0,
        cache_limit_bytes=0,
        compute_ms_per_batch=250.0,
        predownload=16,
        max_steps=0,
    )
    base.update(kw)
    return SimulationConfig(**base)


# ============================================================
# 一、纯函数仿真核心（合成 plan，毫秒级）
# ============================================================
def test_zero_cache_every_first_touch_misses():
    """零缓存（限 1 字节）：每个 shard 都只命中一次 → 全 miss，下载量 = Σ size。"""
    sizes = [10_000_000] * 8
    plan = [list(range(8))]  # 单 worker，8 个 shard 各用一次
    cfg = _cfg(bandwidth_mbps=10_000, compute_ms_per_batch=100, predownload=0, cache_limit_bytes=1)
    r = simulate_plan(cfg, plan, sizes)
    assert r.steps == 8
    assert r.cache_hits == 0
    assert r.cache_misses == 8
    assert r.hit_rate == 0.0
    assert r.total_download_bytes == sum(sizes)
    assert r.peak_cache_bytes <= max(sizes)  # 缓存装不下第二个 shard


def test_huge_cache_hit_rate_equals_one_minus_first_touch():
    """不限缓存、无预取：命中率 == 1 - 每 epoch 首次触达数 / 总步数。"""
    sizes = [1_000_000] * 4
    plan = [[0, 1, 2, 3] * 2]  # 8 步 / 4 个 shard，每 shard 第二次使用必命中
    cfg = _cfg(bandwidth_mbps=100_000, compute_ms_per_batch=100,
               cache_limit_bytes=0, predownload=0)
    r = simulate_plan(cfg, plan, sizes)
    assert r.steps == 8
    assert r.cache_misses == 4
    assert r.cache_hits == 4
    assert r.hit_rate == pytest.approx(0.5)
    assert r.hit_rate == pytest.approx(1 - 4 / 8)
    # 本模型下 miss 必然伴随等待（见模块 docstring）
    assert r.starved_steps == r.cache_misses
    assert r.avg_gpu_util < 1.0


def test_prefetch_converts_first_touch_into_hit():
    """同样的不限缓存，开启足够大的预取窗口后，首次触达也能在消费前就绪。"""
    sizes = [1_000_000] * 4
    plan = [[0, 1, 2, 3] * 2]
    cfg = _cfg(bandwidth_mbps=100_000, compute_ms_per_batch=100,
               cache_limit_bytes=0, predownload=16)
    r = simulate_plan(cfg, plan, sizes)
    assert r.cache_misses == 1, "只有第 0 步来不及（窗口从当前步开始预取）"
    assert r.hit_rate == pytest.approx(7 / 8)


def test_low_bandwidth_causes_starvation():
    """低带宽：每 shard 下载 10s，训练被显著饿住。"""
    sizes = [10_000_000] * 6
    plan = [[0, 1, 2, 3, 4, 5, 0, 1]]  # 6 个首次触达 + 2 个复用
    cfg = _cfg(bandwidth_mbps=1.0, compute_ms_per_batch=250, cache_limit_bytes=0, predownload=4)
    r = simulate_plan(cfg, plan, sizes)
    assert r.starve_ratio == pytest.approx(6 / 8), f"低带宽应显著饥饿，实际 {r.starve_ratio}"
    # wall ≈ 6 次下载 × 10s + 尾部 compute，远大于 busy（8 × 0.25s）
    assert r.wall_time_s > 60.0
    assert r.avg_gpu_util < 0.1


def test_predownload_reduces_starvation():
    """同样的带宽下，更大的预取窗口应带来更多命中（提前入队 > 临时抱佛脚）。"""
    sizes = [1_000_000] * 16
    plan = [list(range(16))]
    cfg_small = _cfg(bandwidth_mbps=20.0, compute_ms_per_batch=250, predownload=1, cache_limit_bytes=0)
    cfg_large = _cfg(bandwidth_mbps=20.0, compute_ms_per_batch=250, predownload=16, cache_limit_bytes=0)
    r_small = simulate_plan(cfg_small, plan, sizes)
    r_large = simulate_plan(cfg_large, plan, sizes)
    assert r_large.hit_rate > r_small.hit_rate
    assert r_large.starve_ratio < r_small.starve_ratio


def test_deterministic_across_runs():
    """同配置两次运行报告完全一致（多 worker / 多节点 / 有限缓存）。"""
    sizes = [3_000_000] * 32
    plan = [
        [(w * 7 + i * 3) % 32 for i in range(40)] for w in range(8)
    ]
    cfg = _cfg(world_size=8, num_nodes=2, num_workers=1, bandwidth_mbps=200.0,
               cache_limit_bytes=9_000_000, predownload=8, compute_ms_per_batch=180)
    r1 = simulate_plan(cfg, plan, sizes)
    r2 = simulate_plan(cfg, plan, sizes)
    assert r1 == r2
    assert r1.steps == 8 * 40


def test_max_steps_truncates():
    sizes = [1_000_000] * 4
    plan = [[0, 1, 2, 3]]
    r = simulate_plan(_cfg(max_steps=2, cache_limit_bytes=0), plan, sizes)
    assert r.steps == 2


def test_empty_plan():
    r = simulate_plan(_cfg(), [], [])
    assert r.steps == 0
    assert r.wall_time_s == 0.0


# ============================================================
# 二、真实 manifest：Rust plan 的一致性与守恒
# ============================================================
@pytest.fixture(scope="module")
def mini_dataset():
    """构建一个小型 PT 数据集（聚合 shard），供 plan / CLI 测试使用。"""
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ds")
        os.makedirs(out, exist_ok=True)
        raw = os.path.join(tmp, "raw.jsonl")
        with open(raw, "w", encoding="utf-8") as f:
            for i in range(TOTAL_DOCS):
                f.write(json.dumps({"text": f"doc {i} the quick brown fox jumps over the lazy dog {i * 7}"}) + "\n")
        DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN).build_dataset(
            [raw], out, mode="pt", batch_size=BATCH_SIZE, eos_id=2,
            batches_per_shard=BATCHES_PER_SHARD, compression="none",
        )
        yield os.path.join(out, "index.json")


def _manifest_stats(index_path):
    with open(index_path) as f:
        m = json.load(f)
    total_batches = sum(int(s["num_batches"]) for s in m["shards"])
    return len(m["shards"]), total_batches


@pytest.mark.parametrize("algo", ["round_robin", "relaxed"])
def test_plan_disjoint_and_cover(mini_dataset, algo):
    """shuffle=none：各 worker 的 shard 集合互斥且并集为全集（epoch 0）。

    注：只测 epoch 0——plan 没有起始 epoch 参数，而各 epoch 的 shard 分配本身
    （每 epoch 重新打散、per-worker 分片数可变）由 Rust 侧单测覆盖。
    """
    index_path = mini_dataset
    num_shards, _total_batches = _manifest_stats(index_path)
    world_size, num_workers = 4, 2
    cfg = _cfg(index_path=index_path, world_size=world_size, num_workers=num_workers,
               shuffle="none", epochs=1, partition=algo)
    plan = plan_from_rust(cfg)
    assert len(plan) == world_size * num_workers

    sets = [set(seq) for seq in plan]
    union = set().union(*sets)
    assert union == set(range(num_shards)), f"{algo}: 并集应为全集"
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"{algo}: worker {i}/{j} shard 集合重叠"


@pytest.mark.parametrize("algo", ["round_robin", "relaxed"])
@pytest.mark.parametrize("shuffle", ["none", "permutation"])
def test_plan_step_conservation(mini_dataset, algo, shuffle):
    """步数守恒：Σ worker 步数 == total_batches × epochs；且引用合法。"""
    index_path = mini_dataset
    num_shards, total_batches = _manifest_stats(index_path)
    cfg = _cfg(index_path=index_path, world_size=4, num_workers=2, shuffle=shuffle,
               epochs=2, partition=algo)
    plan = plan_from_rust(cfg)
    steps = sum(len(seq) for seq in plan)
    assert steps == total_batches * 2, f"{algo}/{shuffle}: 步数应守恒"
    for seq in plan:
        assert all(0 <= s < num_shards for s in seq)

    # permutation 与非 shuffle 的步数分布应一致（只是顺序不同）
    cfg_none = _cfg(index_path=index_path, world_size=4, num_workers=2, shuffle="none",
                    epochs=2, partition=algo)
    plan_none = plan_from_rust(cfg_none)
    assert sorted(len(s) for s in plan) == sorted(len(s) for s in plan_none)


def test_plan_permutation_is_stepwise_bijection(mini_dataset):
    """置换模式：每个 worker 每 epoch 内步数不变，且 shard 多重集与顺序模式一致。"""
    index_path = mini_dataset
    common = dict(index_path=index_path, world_size=2, num_workers=1, epochs=1, partition="round_robin")
    plan_seq = plan_from_rust(_cfg(shuffle="none", **common))
    plan_perm = plan_from_rust(_cfg(shuffle="permutation", **common))
    for a, b in zip(plan_seq, plan_perm):
        assert len(a) == len(b)
        assert sorted(a) == sorted(b), "置换只改顺序，不改 shard 多重集"


def test_simulate_and_sweep_on_real_manifest(mini_dataset):
    """端到端：真实 index.json → simulate / sweep_cache（不读 shard 数据）。"""
    index_path = mini_dataset
    sizes = load_shard_sizes(index_path)
    assert sizes and all(s > 0 for s in sizes)

    cfg = _cfg(index_path=index_path, world_size=4, num_workers=1, epochs=1,
               bandwidth_mbps=500.0, compute_ms_per_batch=250)
    r = simulate(cfg)
    assert r.steps > 0
    assert r.cache_hits + r.cache_misses == r.steps
    assert r.total_download_bytes > 0
    assert r.avg_gpu_util <= 1.0, "多 worker 并行时利用率按 worker 数摊平，不应超过 1"

    results = sweep_cache(cfg)
    assert [label for label, _ in results] == ["2×shard", "4×shard", "8×shard", "16×shard", "32×shard", "∞"]
    hit_rates = [r.hit_rate for _label, r in results]
    assert hit_rates[-1] == max(hit_rates), "不限缓存应取得最高命中率"
    # 表格可渲染
    assert "hit_rate" in format_sweep_markdown(cfg, results)
    assert "wall_time" in format_report_markdown(results[-1][1])


# ============================================================
# 三、CLI 冒烟
# ============================================================
def test_cli_single_run(mini_dataset, capsys):
    rc = main([mini_dataset, "--world-size", "4", "--num-workers", "1",
               "--bandwidth-mbps", "500", "--cache-limit-gb", "0.01"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hit_rate" in out and "starve_ratio" in out


def test_cli_sweep(mini_dataset, capsys):
    rc = main([mini_dataset, "--world-size", "4", "--sweep-cache"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cache_limit sweep" in out
    assert "∞" in out


def test_cli_missing_index(capsys):
    rc = main(["/nonexistent/index.json"])
    assert rc == 2

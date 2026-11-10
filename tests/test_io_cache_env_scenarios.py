"""IO / 缓存 / 环境变量 / 信号处理 全场景测试。

覆盖真实部署里会遇到的：本地数据被破坏、HTTP 远端数据集、缓存命中与修复、
缓存目录/限额、环境变量探测、Ctrl+C / SIGTERM 的信号链。
"""

from __future__ import annotations

import contextlib
import http.server
import os
import shutil
import signal
import socket
import threading
import urllib.parse
from collections import Counter

import pytest

from _scenario_kit import (
    BATCH_SIZE,
    MAX_SEQ_LEN,
    batch_key,
    build_dataset,
    load_index,
    pt_records,
    read_batches,
    sft_records,
)


# ==============================================================================
# 夹具：可计数的 HTTP 服务器（模拟对象存储 / CDN）
# ==============================================================================
@contextlib.contextmanager
def serve_dir(directory: str):
    """把目录作为 HTTP 根对外提供，并统计每个路径的 GET 次数。"""
    counts: Counter = Counter()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, *args):        # 静音
            pass

        def do_GET(self):
            counts[urllib.parse.urlparse(self.path).path] += 1
            super().do_GET()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", counts
    finally:
        srv.shutdown()
        srv.server_close()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def local_pt(tmp_path_factory):
    d = tmp_path_factory.mktemp("io_local")
    return build_dataset(str(d / "pt"), pt_records(200), "pt", batches_per_shard=4,
                         compression="zstd")


@pytest.fixture(scope="module")
def local_sft(tmp_path_factory):
    d = tmp_path_factory.mktemp("io_sft")
    return build_dataset(str(d / "sft"), sft_records(120), "sft", batches_per_shard=4)


@pytest.fixture
def served_pt(local_pt, tmp_path, monkeypatch):
    """把 PT 数据集复制到临时目录并通过 HTTP 提供；缓存目录也在临时目录里。"""
    src = os.path.dirname(local_pt)
    root = tmp_path / "served"
    shutil.copytree(src, root)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEGADATA_CACHE_DIR", str(cache))
    monkeypatch.delenv("MEGADATA_VALIDATE_CACHE", raising=False)
    monkeypatch.delenv("MEGADATA_CACHE_LIMIT", raising=False)
    with serve_dir(str(root)) as (base, counts):
        yield base, counts, root, cache


# ==============================================================================
# 1. 本地数据被破坏（bit rot / 人为改动）
# ==============================================================================
def test_local_shard_bitrot_is_detected(local_pt, tmp_path):
    """本地 shard 被改写（非截断）→ 读取必须报错，且定位到具体 shard。"""
    root = tmp_path / "ds"
    shutil.copytree(os.path.dirname(local_pt), root)
    bins = sorted(f for f in os.listdir(root) if f.endswith(".bin"))
    assert bins
    victim = root / bins[0]
    data = bytearray(victim.read_bytes())
    for i in range(len(data) // 2, min(len(data) // 2 + 64, len(data))):
        data[i] ^= 0xFF                    # 翻转中段字节（zstd frame 结构被破坏）
    victim.write_bytes(bytes(data))

    from megadata import BatchStream

    loader = BatchStream(index_json_path=str(root / "index.json"), batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
    with pytest.raises(OSError) as exc:
        for i, _ in enumerate(loader):
            if i > 100_000:
                pytest.fail("读了 10 万个 batch 仍无错误：IO 错误被静默吞掉（回归）")
    assert bins[0] in str(exc.value), f"错误信息应定位到损坏的 shard：{exc.value}"


def test_local_shard_truncation_is_detected(local_pt, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(os.path.dirname(local_pt), root)
    bins = sorted(f for f in os.listdir(root) if f.endswith(".bin"))
    (root / bins[0]).write_bytes(b"")

    from megadata import BatchStream

    loader = BatchStream(index_json_path=str(root / "index.json"), batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
    with pytest.raises(OSError):
        for i, _ in enumerate(loader):
            if i > 100_000:
                pytest.fail("截断的 shard 未报错")


def test_missing_shard_file_is_detected(local_pt, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(os.path.dirname(local_pt), root)
    bins = sorted(f for f in os.listdir(root) if f.endswith(".bin"))
    os.remove(root / bins[0])

    from megadata import BatchStream

    loader = BatchStream(index_json_path=str(root / "index.json"), batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
    with pytest.raises(OSError):
        for i, _ in enumerate(loader):
            if i > 100_000:
                pytest.fail("缺失的 shard 未报错")


# ==============================================================================
# 2. HTTP 远端数据集：端到端 + 缓存复用
# ==============================================================================
def test_http_dataset_end_to_end(served_pt, local_pt):
    """HTTP 数据集读出的内容必须与本地读出的完全一致。"""
    from megadata import BatchStream

    base, counts, _root, _cache = served_pt
    remote = [batch_key(b) for b in BatchStream(
        index_json_path=f"{base}/index.json", batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)]
    local = [batch_key(b) for b in read_batches(local_pt)]
    assert remote == local, "HTTP 读取结果与本地不一致"
    assert any(p.endswith(".bin") for p in counts), "应发生过 shard 下载"
    assert counts[f"/{os.path.basename(local_pt)}"] >= 1, "应拉取过 index.json"


def test_http_cache_is_reused(served_pt):
    """第二次读同一数据集：shard 不应再次下载（缓存命中）。"""
    from megadata import BatchStream

    base, counts, _root, _cache = served_pt

    def read_once():
        return [batch_key(b) for b in BatchStream(
            index_json_path=f"{base}/index.json", batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)]

    first = read_once()
    shard_gets = {p: c for p, c in counts.items() if p.endswith(".bin")}
    assert shard_gets and sum(shard_gets.values()) >= 1

    before = dict(counts)
    second = read_once()
    assert second == first
    # 按**增量**比较（不能按 key 是否存在比较，否则断言恒真）
    extra = sum(c - before.get(p, 0) for p, c in counts.items()
                if p.endswith(".bin") and c > before.get(p, 0))
    assert extra == 0, f"缓存未命中，重复下载了 {extra} 次 shard：{counts}"


def test_cache_dir_is_honored(served_pt):
    from megadata import BatchStream

    base, _counts, _root, cache = served_pt
    list(BatchStream(index_json_path=f"{base}/index.json", batch_size=BATCH_SIZE,
                      max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    cached = list(cache.iterdir())
    assert cached, f"MEGADATA_CACHE_DIR 未生效（{cache} 为空）"
    assert any(p.name.endswith(".bin") for p in cached), "缓存里应有 shard"


def test_cache_limit_small_still_correct(served_pt, monkeypatch):
    """把缓存限额设到极小：允许反复重下，但**数据必须仍然正确**。"""
    from megadata import BatchStream

    base, counts, _root, _cache = served_pt
    monkeypatch.setenv("MEGADATA_CACHE_LIMIT", "1")
    keys = [batch_key(b) for b in BatchStream(
        index_json_path=f"{base}/index.json", batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)]
    assert keys, "极小缓存下仍应读得到数据"


def test_validate_cache_repairs_tampered_cache(served_pt, monkeypatch):
    """缓存被篡改 + MEGADATA_VALIDATE_CACHE=1 → 校验失败后重下，数据仍正确。"""
    from megadata import BatchStream

    base, counts, _root, cache = served_pt

    def read_once():
        return [batch_key(b) for b in BatchStream(
            index_json_path=f"{base}/index.json", batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)]

    ref = read_once()
    cached_bins = sorted(p for p in cache.iterdir() if p.name.endswith(".bin"))
    assert cached_bins, "应先有缓存"
    # 篡改第一个缓存文件
    victim = cached_bins[0]
    data = bytearray(victim.read_bytes())
    data[0] ^= 0xFF
    victim.write_bytes(bytes(data))

    monkeypatch.setenv("MEGADATA_VALIDATE_CACHE", "1")
    before = dict(counts)
    again = read_once()
    assert again == ref, "缓存损坏后读出的数据不对（未重下修复）"
    redownloaded = sum(c - before.get(p, 0) for p, c in counts.items()
                       if p.endswith(".bin") and c > before.get(p, 0))
    assert redownloaded >= 1, f"损坏的缓存未被重新下载：{counts}"


def test_http_missing_shard_is_reported(local_pt, tmp_path, monkeypatch):
    """远端缺 shard → 必须报错（不能静默空转）。"""
    from megadata import BatchStream

    root = tmp_path / "served"
    shutil.copytree(os.path.dirname(local_pt), root)
    bins = sorted(f for f in os.listdir(root) if f.endswith(".bin"))
    os.remove(root / bins[0])
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEGADATA_CACHE_DIR", str(cache))

    with serve_dir(str(root)) as (base, _counts):
        loader = BatchStream(index_json_path=f"{base}/index.json", batch_size=BATCH_SIZE,
                              max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
        with pytest.raises(OSError):
            for i, _ in enumerate(loader):
                if i > 100_000:
                    pytest.fail("远端 404 未报错")


def test_read_manifest_over_http(served_pt, local_pt):
    from megadata import read_manifest

    base, _counts, _root, _cache = served_pt
    remote = read_manifest(f"{base}/index.json")
    assert remote == load_index(local_pt)


def test_read_manifest_missing_path(tmp_path):
    from megadata import read_manifest

    with pytest.raises(Exception):
        read_manifest(str(tmp_path / "nope.json"))


# ==============================================================================
# 3. 环境变量探测
# ==============================================================================
def test_world_size_rank_from_env(local_pt, monkeypatch):
    """未显式传 world_size/rank 时从环境变量探测（DDP 启动脚本的常规做法）。"""
    from megadata import BatchStream

    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "1")
    loader = BatchStream(index_json_path=local_pt, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN)
    assert loader._world_size == 3 and loader._rank == 1
    assert len(loader) > 0, "rank=1 应有数据（不是空转 rank）"


def test_explicit_world_size_overrides_env(local_pt, monkeypatch):
    from megadata import BatchStream

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")
    loader = BatchStream(index_json_path=local_pt, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    assert loader._world_size == 1 and loader._rank == 0


def test_disable_signal_handler_env(local_pt, monkeypatch):
    """MEGADATA_DISABLE_SIGNAL_HANDLER=1：不接管信号（让框架托管）。"""
    import megadata._signals as sig          # F-36：信号状态/函数已抽到 _signals
    import megadata.dataloader as dl         # noqa: F401
    # dataloader 的 re-export 必须指向同一函数（向后兼容）
    assert dl.install_signal_handlers is sig.install_signal_handlers

    monkeypatch.setenv("MEGADATA_DISABLE_SIGNAL_HANDLER", "1")
    monkeypatch.setattr(sig, "_SIGNAL_INSTALLED", False)
    monkeypatch.setattr(sig, "_ORIGINAL_HANDLERS", {})
    before = signal.getsignal(signal.SIGINT)
    sig.install_signal_handlers()
    assert signal.getsignal(signal.SIGINT) is before, "禁用开关未生效：仍然接管了 SIGINT"
    assert sig._SIGNAL_INSTALLED is False


# ==============================================================================
# 4. 信号处理：幂等 + 链式 + 取消全部活跃实例
# ==============================================================================
def test_install_uninstall_signal_handlers_idempotent(local_pt):
    import megadata.dataloader as dl

    dl.uninstall_signal_handlers()
    orig = signal.getsignal(signal.SIGINT)

    dl.install_signal_handlers()
    once = signal.getsignal(signal.SIGINT)
    dl.install_signal_handlers()          # 幂等：不叠加
    assert signal.getsignal(signal.SIGINT) is once
    assert once is not orig

    dl.uninstall_signal_handlers()
    assert signal.getsignal(signal.SIGINT) is orig, "卸载后应恢复原始 handler"


def test_signal_handler_chains_previous(local_pt):
    """链式：接管前已有的 handler 仍会被调用（框架自身信号处理不受影响）。"""
    import megadata.dataloader as dl

    called = []
    dl.uninstall_signal_handlers()
    orig = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, lambda s, f: called.append(s))
        dl.install_signal_handlers()
        os.kill(os.getpid(), signal.SIGINT)
        assert called == [signal.SIGINT], "前一个 handler 未被链接调用"
    finally:
        dl.uninstall_signal_handlers()
        signal.signal(signal.SIGINT, orig)


def test_signal_cancels_all_active_instances(local_pt):
    """收到信号时，所有活跃加载器都应被取消（而不是只取消一个）。"""
    import megadata.dataloader as dl
    from megadata import MegadataInterrupted, BatchStream

    dl.uninstall_signal_handlers()
    origin = signal.getsignal(signal.SIGINT)
    try:
        loaders = [BatchStream(index_json_path=local_pt, batch_size=BATCH_SIZE,
                                max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
                   for _ in range(2)]
        iters = [iter(x) for x in loaders]
        for it in iters:
            next(it)
        dl.install_signal_handlers()
        os.kill(os.getpid(), signal.SIGINT)
        # 前一个 handler 是 Python 默认的（抛 KeyboardInterrupt）——这里只关心取消已扩散
        for it in iters:
            with pytest.raises((MegadataInterrupted, KeyboardInterrupt)):
                next(it)
    except KeyboardInterrupt:
        pass
    finally:
        dl.uninstall_signal_handlers()
        signal.signal(signal.SIGINT, origin)


def test_sigterm_terminates_process(local_pt):
    """SIGTERM 不能被静默吞掉（k8s / timeout 依赖它能杀掉进程）。"""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import os, signal, time, sys
        sys.path.insert(0, {os.path.dirname(os.path.abspath("tests"))!r})
        from megadata import BatchStream
        loader = BatchStream(index_json_path={local_pt!r}, batch_size={BATCH_SIZE},
                              max_seq_len={MAX_SEQ_LEN}, world_size=1, rank=0)
        print("READY", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        print("SHOULD_NOT_REACH", flush=True)
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=60, cwd=os.getcwd())
    assert "READY" in proc.stdout
    assert "SHOULD_NOT_REACH" not in proc.stdout, "SIGTERM 被吞掉，进程存活（回归）"
    assert proc.returncode != 0, f"进程应以非 0 退出，实际 {proc.returncode}"


# ==============================================================================
# 5. 缓存/IO 的边角
# ==============================================================================
def test_index_json_is_valid_and_self_consistent(local_pt, local_sft):
    """两种格式的 manifest 都满足文档承诺的结构契约。"""
    for idx in (local_pt, local_sft):
        m = load_index(idx)
        assert m["version"] in ("1.0", "2.0")
        assert m["format"] in ("Memmap", "Zstd-MDS", "Arrow-IPC")
        assert m["global_stats"]["total_batches"] == sum(s["num_batches"] for s in m["shards"])
        assert m["global_stats"]["total_shards"] == len(m["shards"])
        assert m["global_stats"]["total_tokens"] > 0
        assert m["global_stats"]["batch_size"] == BATCH_SIZE
        for s in m["shards"]:
            assert s["hashes"].get("sha256"), "每个 shard 都应记录 sha256"
            assert s["size_bytes"] > 0


def test_pt_sidecar_hash_recorded(local_pt):
    """PT 数据集的 .cu / .idx 旁路文件也进 manifest 的 hash 表（可校验）。"""
    m = load_index(local_pt)
    for s in m["shards"]:
        assert "sha256:cu" in s["hashes"], "缺 .cu 的 hash（损坏只能被静默忽略）"
        assert "sha256:idx" in s["hashes"], "缺 .idx 的 hash"


def test_sft_dataset_has_no_sidecars(local_sft):
    m = load_index(local_sft)
    assert m["format"] == "Arrow-IPC"
    for s in m["shards"]:
        assert not any(k.startswith("sha256:") for k in s["hashes"]), \
            "SFT (Arrow IPC) 不应有旁路文件 hash"


def test_cloud_scheme_dispatch_error_message(local_pt):
    """不支持的云协议 / 不存在的 bucket：错误信息要可读（不能是裸 panic）。"""
    from megadata import BatchStream

    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path="s3://definitely-not-a-real-bucket-xyz/index.json",
                     batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    msg = str(exc.value)
    assert "definitely-not-a-real-bucket-xyz" in msg or "失败" in msg, \
        f"云路径失败信息不可操作：{msg}"

import os
import json
import tempfile
import threading
import http.server
import pytest
import time
import shutil

from megadata import DataBuilder, MixBatchStream

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./model_learn_tokenizer")
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
TOTAL_DOCS = 100

class QuietSimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """一个安静的 HTTP 处理器，不在控制台打印无用的请求日志"""
    def log_message(self, format, *args):
        pass

@pytest.fixture(scope="module")
def cloud_mock_environment():
    """
    环境夹具：
    1. 生成假数据并构建为一个本地的预训练(PT)数据集。
    2. 在后台启动一个 HTTP Server，把这个本地数据集目录挂载为 Web 服务。
    3. 返回挂载的 URL，模拟真实的云端对象存储环境。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "raw_data.jsonl")
        output_dir = os.path.join(temp_dir, "processed_pt")

        # 1. 生成并构建假数据
        with open(raw_path, "w", encoding="utf-8") as f:
            for i in range(TOTAL_DOCS):
                f.write(json.dumps({"text": f"This is cloud document {i}"}) + "\n")

        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
        builder.build_dataset([raw_path], output_dir, mode="pt", batch_size=BATCH_SIZE)

        # 2. 启动后台 HTTP 服务器，根目录指向 output_dir
        old_cwd = os.getcwd()
        os.chdir(output_dir) # 切换工作目录让 server 挂载
        handler = QuietSimpleHTTPRequestHandler

        # 寻找一个空闲端口
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]

        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # 等待服务器启动
        time.sleep(1)

        # 返回模拟的“云端 URL” (相当于 s3://my-bucket/index.json)
        cloud_url = f"http://127.0.0.1:{port}/index.json"
        print(f"\n[Mock Cloud] 已在后台启动虚拟云存储: {cloud_url}")

        yield cloud_url

        # 清理
        os.chdir(old_cwd)  # 恢复工作目录，避免污染同进程内其他测试的相对路径
        httpd.shutdown()
        httpd.server_close()


def test_cloud_streaming_download(cloud_mock_environment):
    """
    【云原生网络验证】：验证引擎能否通过 URL 远程拉取 index.json，
    并自动在后台将 shard 下载到 /tmp/megadata_cache 缓存中。
    """
    cloud_url = cloud_mock_environment

    # 强制清理本机的缓存目录，保证测试的绝对严谨性
    cache_dir = os.path.join(tempfile.gettempdir(), "megadata_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print(f"\n[Cloud Test] 尝试从云端加载数据: {cloud_url}")

    # 1. 实例化 Loader，传入的不再是本地路径，而是 HTTP URL！
    loader = MixBatchStream(
        configs=[{"path": cloud_url}],
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

    batch_count = 0
    # 2. 迭代读取，引擎应该会触发底层的 Tokio 网络下载
    for batch in loader:
        assert "input_ids" in batch
        assert batch["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        batch_count += 1

    print(f"[Cloud Test] 成功从网络流式加载了 {batch_count} 个 Batch！")

    # 3. 最关键的物理验证：检查本地缓存目录是否有文件落地！
    assert os.path.exists(cache_dir), "缓存目录未被创建，说明没有走 CloudIoManager 逻辑！"
    cached_files = os.listdir(cache_dir)
    assert len(cached_files) > 0, "缓存目录为空，说明分片没有被成功下载！"

    print(f"[Cloud Test] 验证通过！云端文件已成功缓存至本地: {cached_files}")
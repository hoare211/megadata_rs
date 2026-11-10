import os
import json
import tempfile
import shutil
import pytest

# 可选测试依赖（20261031 评审工程项）：未安装时整文件 skip，而不是让全量收集中断。
pytest.importorskip("boto3", reason="test_s3_io requires boto3 (install dev extras)")
pytest.importorskip("moto", reason="test_s3_io requires moto (install dev extras)")

import boto3
from moto.server import ThreadedMotoServer

from megadata import DataBuilder, MixBatchStream

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./model_learn_tokenizer")
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
TOTAL_DOCS = 100

@pytest.fixture(scope="module")
def s3_mock_environment():
    """
    环境夹具：
    1. 启动本地 Moto S3 模拟服务器。
    2. 生成假数据，使用 DataBuilder 本地构建。
    3. 将构建好的 .bin 和 index.json 使用 boto3 上传至模拟 S3 桶。
    4. 返回 s3:// 协议的虚拟路径。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")

    # 1. 启动后端的 S3 模拟服务器
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    port = server._server.server_address[1]
    endpoint_url = f"http://127.0.0.1:{port}"

    # 2. 设置环境变量：这一步极其关键！
    os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # 给 boto3 用的

    os.environ["AWS_ENDPOINT"] = endpoint_url       # 给 Rust object_store 用的
    os.environ["AWS_ALLOW_HTTP"] = "true"  # 允许 S3 走 HTTP 而非 HTTPS

    # 3. 使用 boto3 连接并创建一个 Bucket
    s3_client = boto3.client("s3", endpoint_url=endpoint_url, region_name="us-east-1")
    bucket_name = "test-llm-bucket"
    s3_client.create_bucket(Bucket=bucket_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "raw_data.jsonl")
        output_dir = os.path.join(temp_dir, "processed_pt")

        # 4. 生成并构建假数据到本地临时目录
        with open(raw_path, "w", encoding="utf-8") as f:
            for i in range(TOTAL_DOCS):
                f.write(json.dumps({"text": f"This is an S3 cloud document {i}"}) + "\n")

        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
        builder.build_dataset([raw_path], output_dir, mode="pt", batch_size=BATCH_SIZE)

        # 5. 将本地生成的数据文件，全部推送到模拟的 S3 桶中
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            s3_key = f"datasets/my_pt_data/{filename}"
            s3_client.upload_file(file_path, bucket_name, s3_key)

        s3_url = f"s3://{bucket_name}/datasets/my_pt_data/index.json"
        print(f"\n[Mock S3] 数据已全部上传至模拟云存储: {s3_url}")

        yield s3_url

    # 清理关闭模拟服务器
    server.stop()


def test_s3_streaming_download(s3_mock_environment):
    """
    【云原生 S3 验证】：测试底层 Rust 引擎能否通过对象存储协议拉取分片。
    """
    s3_url = s3_mock_environment

    # 清理本地缓存，确实验证走的是下载
    cache_dir = os.path.join(tempfile.gettempdir(), "megadata_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print(f"\n[S3 Test] 尝试从 S3 云端加载数据: {s3_url}")

    # 显式传入 weight=1.0，绕过 Python 端目前需要本地解析 index.json 算总量的逻辑
    # 直接让 Rust 接管 S3 路径！
    loader = MixBatchStream(
        configs=[{"path": s3_url, "weight": 1.0}],
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

    batch_count = 0
    # 开始迭代！此时底层的 Tokio 会利用 object_store 疯狂去模拟 S3 服务器里拖拽 .bin 分片！
    for batch in loader:
        assert "input_ids" in batch
        assert batch["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        batch_count += 1

    print(f"[S3 Test] 成功从 S3 网络流式加载了 {batch_count} 个 Batch！")

    # 物理验证
    assert os.path.exists(cache_dir), "缓存目录未被创建，说明没有走 CloudIoManager 逻辑！"
    cached_files = os.listdir(cache_dir)
    assert len(cached_files) > 0, "缓存目录为空，分片下载失败！"

    print(f"[S3 Test] 验证通过！S3 文件已成功缓存至本地: {cached_files}")
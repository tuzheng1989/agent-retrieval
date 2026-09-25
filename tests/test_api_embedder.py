"""ApiEmbedder 与工厂的契约测试（自 evochat test_embedders.py 迁移）。

models.yaml 加载与 flag 门控是宿主 wiring 的事（本包只收显式 EmbeddingConfig），
那些用例留在宿主。网络调用打桩、依赖缺失用 sys.modules 注入模拟，不依赖真实端点。
"""

import sys

import pytest
import requests

from agent_retrieval.core.ports import QueryEmbeddingError
from agent_retrieval.embedders.api import ApiEmbedder
from agent_retrieval.embedders.config import EmbeddingConfig
from agent_retrieval.embedders.factory import build_embedder, vector_available


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_embedding_config_identity_is_stable_triple():
    """identity 三元组进入快照 id 与 meta：model_version 缺省回落 model_name。"""
    assert EmbeddingConfig(kind="api", provider="p", model_name="m").identity() == ("p", "m", "m")
    assert EmbeddingConfig(kind="api", provider="p", model_name="m", model_version="v1").identity() == ("p", "m", "v1")


def test_api_embedder_posts_openai_compatible_batch_and_normalizes(monkeypatch):
    """ApiEmbedder 走 OpenAI 兼容 /embeddings：Bearer 凭据、批量分块、L2 归一化。"""
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        inputs = json["input"]
        # 故意返回未归一化向量：|[3,4]| = 5，归一化后应为 (0.6, 0.8)。
        return _FakeResponse({"data": [
            {"index": i, "embedding": [3.0 * (i + 1), 4.0 * (i + 1)]}
            for i in range(len(inputs))
        ]})

    monkeypatch.setattr("requests.post", fake_post)
    embedder = ApiEmbedder(
        base_url="https://emb.example/v1", api_key="secret", model_name="emb-x",
        batch_size=2, timeout_seconds=7,
    )

    vectors = embedder.embed_corpus(["a", "b", "c"])

    assert captured["url"] == "https://emb.example/v1/embeddings"
    assert captured["headers"] == {"Authorization": "Bearer secret"}  # noqa: S105 — 桩断言
    assert captured["timeout"] == 7
    # 分块：batch_size=2、语料 3 条 → 两批请求（2 + 1）。
    assert len(captured["payload"]["input"]) == 1
    assert captured["payload"]["model"] == "emb-x"
    assert [tuple(round(v, 6) for v in vector) for vector in vectors] == [
        (0.6, 0.8), (0.6, 0.8), (0.6, 0.8),
    ]
    # 单条查询与语料同管线：同文本逐位一致（端口契约）。
    assert embedder.embed_query("a") == vectors[0]


def test_api_embedder_error_is_wrapped_as_query_embedding_error(monkeypatch):
    """端点异常包装成检索面降级异常：调用方按类型退回 BM25，不裸穿 requests 异常。"""

    def boom(*args, **kwargs):
        raise requests.ConnectionError("断网")

    monkeypatch.setattr("requests.post", boom)
    embedder = ApiEmbedder(base_url="https://emb.example/v1", api_key="k", model_name="m")

    with pytest.raises(QueryEmbeddingError, match="断网"):
        embedder.embed_query("hello")


def test_api_embedder_wrapped_errors_pass_through_untouched(monkeypatch):
    """维度校验抛出的 QueryEmbeddingError 不得被二次包装（降级语义类型面稳定）。"""
    payload = {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}

    def fake_post(*args, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr("requests.post", fake_post)
    embedder = ApiEmbedder(
        base_url="https://emb.example/v1", api_key="k", model_name="m", dimensions=8,
    )

    with pytest.raises(QueryEmbeddingError, match="维度不符"):
        embedder.embed_query("hello")


def test_api_embedder_missing_dependency_reports_extras(monkeypatch):
    """requests 缺失（纯 core 安装）：干净 RuntimeError 指明 extras，不裸穿 ImportError。"""
    monkeypatch.setitem(sys.modules, "requests", None)

    embedder = ApiEmbedder(base_url="https://emb.example/v1", api_key="k", model_name="m")

    with pytest.raises(RuntimeError, match=r"agent-retrieval\[api\]"):
        embedder.embed_query("hello")


def test_factory_builds_api_embedder_from_explicit_config():
    """显式传参契约：工厂只认 EmbeddingConfig，api 形态产出 ApiEmbedder、诊断报 api。"""
    config = EmbeddingConfig(
        kind="api", provider="p", model_name="m",
        base_url="https://emb.example/v1", api_key="k",
    )

    assert isinstance(build_embedder(config), ApiEmbedder)
    assert vector_available(config) == "api"


def test_factory_reports_none_when_api_credentials_missing():
    """无 key ≠ 故障：凭据缺失时工厂返回 None、诊断报 none（未配置形态不告警）。"""
    config = EmbeddingConfig(kind="api", provider="p", model_name="m")

    assert build_embedder(config) is None
    assert vector_available(config) == "none"


def test_factory_reports_none_for_unknown_kind():
    config = EmbeddingConfig(kind="carrier-pigeon", provider="p", model_name="m")

    assert build_embedder(config) is None
    assert vector_available(config) == "none"


def test_local_embedder_raises_clean_error_without_optional_dependencies(monkeypatch):
    """缺 onnxruntime/transformers 时 LocalEmbedder 干净报错（RuntimeError 带装法指引）。"""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    from agent_retrieval.embedders.local import LocalEmbedder

    with pytest.raises(RuntimeError, match=r"agent-retrieval\[local\]"):
        LocalEmbedder(model_dir="models/gte")


def test_factory_reports_none_when_local_kind_dependencies_missing(monkeypatch):
    """kind: local 且依赖缺失：工厂吞掉干净报错返回 None、诊断报 none（不炸启动）。"""
    config = EmbeddingConfig(kind="local", provider="p", model_name="m", model_dir="models/gte")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert build_embedder(config) is None
    assert vector_available(config) == "none"


def test_vector_available_reports_local_when_kind_local_and_dependencies_importable(monkeypatch):
    """kind: local 且依赖可导入：诊断报 local（离线形态的一等公民，非降级）。"""
    import types

    config = EmbeddingConfig(kind="local", provider="p", model_name="m", model_dir="models/gte")
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应构造会话"))
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert vector_available(config) == "local"


def test_vector_available_reports_none_without_config():
    assert vector_available(None) == "none"

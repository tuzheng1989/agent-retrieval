"""OpenAI 兼容 ``/embeddings`` 端点的嵌入器（生产 API 路，extras: ``api``）。"""
from __future__ import annotations

import math
from collections.abc import Sequence

from agent_retrieval.core.ports import QueryEmbeddingError


def _normalize(vector: Sequence[float], dimensions: int) -> tuple[float, ...]:
    values = [float(item) for item in vector]
    if dimensions and len(values) != dimensions:
        raise QueryEmbeddingError(f"embedding 维度不符: 期望 {dimensions}，返回 {len(values)}")
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


class ApiEmbedder:
    """OpenAI 兼容 ``/embeddings`` 端点的嵌入器（``requests`` 为可选依赖）。

    向量 L2 归一化：cosine 消费方按单位向量点积实现（``agent_retrieval.core.ports``
    契约），部分端点返回未归一化向量，不归一会按模长静默缩放分数、系统性扭曲排序。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        batch_size: int = 16,
        timeout_seconds: float = 20.0,
        dimensions: int = 0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._batch_size = batch_size
        self._timeout = timeout_seconds
        self._dimensions = dimensions

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start:start + self._batch_size])
            vectors.extend(self._request(batch))
        return vectors

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._request([text])[0]

    def _request(self, batch: list[str]) -> list[tuple[float, ...]]:
        # requests 是 api extra 的依赖：下沉到调用点惰性导入，让零依赖安装（纯 core
        # 用法）在 import 期不受影响，缺依赖时给出指明 extras 的报错。
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "ApiEmbedder 缺依赖 requests；先安装 agent-retrieval[api]",
            ) from exc
        try:
            response = requests.post(  # noqa: S113 — 超时显式传入
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model_name, "input": batch},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
        except QueryEmbeddingError:
            raise
        except Exception as exc:
            raise QueryEmbeddingError(f"embedding 端点调用失败: {exc}") from exc
        ordered = sorted(data, key=lambda item: int(item.get("index") or 0))
        if len(ordered) != len(batch):
            raise QueryEmbeddingError(
                f"embedding 端点返回条数不符: 请求 {len(batch)} 条，返回 {len(ordered)} 条",
            )
        return [_normalize(item["embedding"], self._dimensions) for item in ordered]

"""嵌入器工厂与部署形态三态判定。

查询/发布两侧共用的装配入口。「向量路总开关」是宿主概念（宿主自己的配置体系决定
本次部署是否启用向量路）：开关关闭时宿主不应构造本工厂，而不是传一个 flag 进来。
"""
from __future__ import annotations

import logging
from typing import Literal

from agent_retrieval.core.ports import Embedder
from agent_retrieval.embedders.api import ApiEmbedder
from agent_retrieval.embedders.config import EmbeddingConfig
from agent_retrieval.embedders.local import LocalEmbedder

log = logging.getLogger("agent_retrieval.embeddings")


def build_embedder(config: EmbeddingConfig) -> Embedder | None:
    """按配置构造嵌入器；``None`` 表示「本部署此刻没有可用的向量路」的正常形态。

    ``None`` 不是异常（local 依赖未装 / 本地模型目录缺失），调用方按 BM25 现状继续。
    api 形态缺 base_url/api_key 同样返回 ``None``（未配置 ≠ 故障）。
    """
    if config.kind == "api":
        if not (config.base_url and config.api_key):
            return None
        return ApiEmbedder(
            base_url=config.base_url,
            api_key=config.api_key,
            model_name=config.model_name,
            batch_size=config.batch_size,
            timeout_seconds=config.timeout_seconds,
            dimensions=config.dimensions,
        )
    if config.kind == "local":
        try:
            return LocalEmbedder(model_dir=config.model_dir)
        except (RuntimeError, ImportError) as exc:
            log.warning("[embeddings] 本地嵌入器不可用，向量路按缺席处理: %s", exc)
            return None
    log.warning("[embeddings] 不支持的嵌入形态: %s", config.kind)
    return None


def vector_available(config: EmbeddingConfig | None) -> Literal["api", "local", "none"]:
    """部署形态三态判定（启动诊断与观测共用）。

    ``"none"`` 覆盖三种正常形态：未配置（``None``）、api 缺凭据、local 依赖未装——
    无 key/无依赖 ≠ 故障，不告警。
    """
    if config is None:
        return "none"
    if config.kind == "api":
        return "api" if (config.base_url and config.api_key) else "none"
    if config.kind == "local":
        try:
            import onnxruntime  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return "none"
        return "local"
    return "none"

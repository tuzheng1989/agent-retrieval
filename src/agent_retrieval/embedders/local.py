"""本地 ONNX 嵌入器（信创/离线路；extras: ``local``，缺依赖时干净报错）。

依赖（onnxruntime/transformers/tokenizers/numpy）装在 ``local`` extra——Windows
ARM64 无 onnxruntime wheel，进主依赖会直接炸开发机安装。
"""
from __future__ import annotations

from collections.abc import Sequence

from agent_retrieval.core.ports import QueryEmbeddingError


class LocalEmbedder:
    """ONNX Runtime + transformers tokenizer 的本地嵌入器（mean-pooling + L2 归一化）。"""

    def __init__(self, *, model_dir: str, max_length: int = 512) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocalEmbedder 缺本地嵌入依赖（onnxruntime/transformers/tokenizers/numpy）；"
                "先安装 agent-retrieval[local]",
            ) from exc
        if not model_dir:
            raise RuntimeError("LocalEmbedder 需要 model_dir 指向 ONNX 模型目录")
        self._np = np
        self._session = ort.InferenceSession(model_dir, providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._max_length = max_length

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        try:
            encoded = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="np",
            )
            input_names = {item.name for item in self._session.get_inputs()}
            last_hidden = self._session.run(
                None, {k: v for k, v in encoded.items() if k in input_names},
            )[0]
        except Exception as exc:
            raise QueryEmbeddingError(f"本地嵌入推理失败: {exc}") from exc
        mask = encoded["attention_mask"][..., None]
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clip(min=1e-9)
        emb = summed / counts
        emb = emb / (self._np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return [tuple(row) for row in emb]

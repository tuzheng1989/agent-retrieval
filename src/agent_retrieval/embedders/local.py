"""本地 ONNX 嵌入器（信创/离线路；extras: ``local``，缺依赖时干净报错）。

依赖（onnxruntime/transformers/tokenizers/numpy）装在 ``local`` extra——Windows
ARM64 无 onnxruntime wheel，进主依赖会直接炸开发机安装。

pooling（B6）：``mean``（attention_mask 加权平均，历史行为）/ ``cls``（每行首个
非 pad 位置——左填充时 ``[:, 0]`` 取到的是 pad，必须按 mask 找索引）/ ``last_token``
（每行最后非 pad 位置——右填充时 ``[:, -1]`` 同理）。
"""
from __future__ import annotations

from collections.abc import Sequence

from agent_retrieval.core.ports import QueryEmbeddingError
from agent_retrieval.embedders.config import POOLINGS


class LocalEmbedder:
    """ONNX Runtime + transformers tokenizer 的本地嵌入器（pooling 可选 + L2 归一化）。"""

    def __init__(
        self,
        *,
        model_dir: str,
        max_length: int = 512,
        pooling: str = "mean",
        query_prefix: str = "",
        corpus_prefix: str = "",
    ) -> None:
        if pooling not in POOLINGS:
            raise ValueError(
                f"pooling 必须是 {'/'.join(POOLINGS)} 之一，收到: {pooling!r}",
            )
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
        self._pooling = pooling
        #: 两路前缀（config 级能力）：查询/语料各自进 tokenizer 前拼接，空前缀是恒等。
        self._query_prefix = query_prefix
        self._corpus_prefix = corpus_prefix

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._embed([self._corpus_prefix + text for text in texts])

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed([self._query_prefix + text])[0]

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
        mask = encoded["attention_mask"]
        if self._pooling == "mean":
            weight = mask[..., None]
            summed = (last_hidden * weight).sum(1)
            counts = weight.sum(1).clip(min=1e-9)
            emb = summed / counts
        elif self._pooling == "cls":
            # 每行首个非 pad 索引：mask 0/1 下 argmax 即首个 1 的位置（左填充安全）。
            first_non_pad = mask.argmax(axis=1)
            emb = last_hidden[self._np.arange(last_hidden.shape[0]), first_non_pad]
        else:  # last_token
            # 每行最后非 pad 索引：反转后 argmax 即从尾部数首个 1（右填充安全）。
            last_non_pad = last_hidden.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
            emb = last_hidden[self._np.arange(last_hidden.shape[0]), last_non_pad]
        emb = emb / (self._np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return [tuple(row) for row in emb]

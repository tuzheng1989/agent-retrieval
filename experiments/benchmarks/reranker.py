"""Cohere-style ``/rerank`` API client (experiments-side host code).

Format is the de-facto standard shared by SiliconFlow, Jina and Cohere:
``POST {base}/rerank`` with ``{model, query, documents, top_n}``, answering
``{results: [{index, relevance_score}, ...]}``. ``results`` may be sparse or
unordered, so it is re-aligned to the input document order by ``index``.

Errors raise :class:`RerankError`; the benchmark runner preflights with one
probe so a misconfigured endpoint fails fast instead of silently leaving the
fusion order untouched.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

#: Re-rank inputs are budgeted per PAIR (query + document): the gateway rejects
#: on query+document length (code 1214) with per-backend limits that vary, and a
#: cross-encoder consumes only a leading window of each text anyway. A 2700-char
#: pair budget (~700 tokens) covers >95% of ToolRet's tool documents verbatim.
RERANK_PAIR_BUDGET_CHARS = 2700
#: A document is never truncated below this, however long the query is — a
#: degenerate 20-char document would make reranking pointless.
RERANK_DOC_FLOOR_CHARS = 200


class RerankError(RuntimeError):
    """Rerank 端点调用失败——基准 runner 按类型捕获并在预检时快速失败。"""


def parse_rerank_response(payload: dict, expected: int) -> list[float]:
    """Align a rerank response to input order.

    Entries the endpoint omitted (some honor a smaller ``top_n``) score
    ``-inf`` — they sort to the tail of the re-ranked window, which is the
    honest reading of "the reranker declined to score this". Out-of-range
    indices are structural errors and raise.
    """
    scores = [float("-inf")] * expected
    for item in payload.get("results", []):
        index = int(item["index"])
        if not 0 <= index < expected:
            raise RerankError(f"rerank 返回越界 index: {index} (documents={expected})")
        scores[index] = float(item.get("relevance_score") or item.get("score") or 0.0)
    return scores


class ApiReranker:
    """Rerank 端点客户端：查询 + 候选文档 → 与输入对齐的相关性分数列表。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout_seconds

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        try:
            import requests  # noqa: PLC0415 — 可选实验依赖，与 ApiEmbedder 同模式
        except ImportError as exc:
            raise RerankError("ApiReranker 缺依赖 requests；先 pip install requests") from exc
        # 网关会把瞬时限流/上游故障也报成 4xx（实测同一请求先 400 后成功），
        # 因此除明确的鉴权失败外都做指数退避重试；限流窗口是 10s 级，退避要够长。
        last_error: Exception | None = None
        for attempt, pause in enumerate((0, 2.0, 5.0, 10.0)):
            if pause:
                time.sleep(pause)
            try:
                response = requests.post(
                    f"{self._base_url}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model_name,
                        "query": query,
                        "documents": list(documents),
                        "top_n": len(documents),
                    },
                    timeout=self._timeout,
                )
                if response.status_code in (401, 403):
                    raise RerankError(
                        f"rerank 鉴权失败 (HTTP {response.status_code})，不重试: {response.text[:120]}",
                    )
                if response.status_code >= 400:
                    # 保留响应体：4xx 的具体拒绝原因（模型名/参数/token 超限）全在里面。
                    raise RerankError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                    )
                return parse_rerank_response(response.json(), len(documents))
            except RerankError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
        raise RerankError(f"rerank 端点调用失败（已重试 3 次）: {last_error}")

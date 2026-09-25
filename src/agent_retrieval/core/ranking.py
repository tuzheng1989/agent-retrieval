"""候选检索面：BM25 + 可选向量路双路融合 → 全量排序注解清单。

角色是「检索当参谋，调用方做决定」：本面提供视野与排序，业务裁决（准入、绑定、
截断）全部留在调用方。

三条硬边界：

1. **语料由调用方投影**（``corpus_text``）：库对「什么算声明面」零建模。惯例是检索面
   与判定面同语料（如资源的 id/name/description 等身份字段拼接）；正文是否进语料是
   调用方的权限决策——索引语料一旦含正文，正文里的噪声词就获得了抬分能力；
2. **候选集由调用方先过滤后融合**：资格审查不合格者根本不进检索，检索永远不能把
   不合格者排进来。本函数不重复准入，也不截断；
3. **全量召回、不在此截断**：k 只截输出，截断发生在查询面（展示层）——点名不受
   截断限制，截断只影响「看见」。

向量路经 :class:`Embedder` 端口注入；缺省不构造，融合退化为 BM25 单路——RRF 形态
保持不变（单路 fused = 1/(K+rank)）。嵌入失败按 :class:`QueryEmbeddingError` 捕获，
同样退化为 BM25 单路，不外抛。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from agent_retrieval.core.bm25 import tokenize
from agent_retrieval.core.fusion import BM25Hit, VectorHit, fuse
from agent_retrieval.core.ports import Embedder, QueryEmbeddingError
from agent_retrieval.core.resource_index import ResourceBM25Index, substantive_term

T = TypeVar("T")


@dataclass(frozen=True)
class CandidateHit(Generic[T]):
    """排序注解后的单个候选，供调用方消费与展示。

    ``matched_terms`` 是查询实词与该候选**语料**实词的交集（字典序）：检索理由的
    证据必须落在调用方投影的语料上，与 ``ResourceBM25Index`` 的判定面口径一致。
    其余注解字段从融合结果透传（缺席路为 ``None``，区别于 0.0 真值）。
    """

    item: T
    fused_score: float
    matched_terms: tuple[str, ...]
    exact: bool = False
    bm25_score: float | None = None
    bm25_relevance: float | None = None
    cosine: float | None = None


def _cosine(query_vector: tuple[float, ...], corpus_vector: tuple[float, ...]) -> float:
    # 端口契约：嵌入向量按惯例归一化，点积即 cosine（见 agent_retrieval.core.ports.MockEmbedder）。
    return sum(a * b for a, b in zip(query_vector, corpus_vector))


def rank_candidates(
    candidates: Iterable[T],
    query: str,
    *,
    corpus_text: Callable[[T], str],
    item_id: Callable[[T], str],
    item_name: Callable[[T], str] | None = None,
    vector: Embedder | None = None,
) -> list[CandidateHit[T]]:
    """对候选池做双路融合排序，返回全量降序注解清单。

    ``query`` 为空串时 BM25 查询无实词、全部零分，调用方保证非空（或自行处理空结果）。
    ``item_name`` 参与 exact 全等判定（查询串与 id 或 name 归一化后全等即置顶）；缺省
    只按 ``item_id`` 判定。

    判定证据口径说明：本函数不做「是否真的相关」的判定（无阈值闸），融合分只决定展示
    顺序与理由注解——需要判定闸的调用方拿 ``bm25_relevance`` 配 ``clearly_related``
    自行裁决。
    """
    pool = list(candidates)
    if not pool:
        return []

    def _identity(item: T) -> tuple[str, str]:
        return (item_id(item), item_name(item) if item_name is not None else "")

    # 检索面与判定面同语料（调用方投影）：没有「正文召回」的隐式建模。
    index = ResourceBM25Index(
        [(item, corpus_text(item)) for item in pool],
        declared=corpus_text,
    )
    bm25_hits = index.search(query, limit=len(pool), identity=_identity)
    bm25_ranked = [
        BM25Hit(
            item_id=item_id(hit.item),
            score=hit.score,
            relevance=hit.relevance,
            overlap=hit.overlap,
            declared_overlap=hit.declared_overlap,
            exact=hit.exact,
        )
        for hit in bm25_hits
    ]

    vector_ranked: list[VectorHit] = []
    if vector is not None:
        try:
            corpus_vectors = vector.embed_corpus([corpus_text(item) for item in pool])
            query_vector = vector.embed_query(query)
        except QueryEmbeddingError:
            # 嵌入失败：融合退化为 BM25 单路（RRF 形态不变），与未注入向量路连续。
            # Run 级冻结与观测由调用方的缓存包装层（agent_retrieval.run）记账。
            vector_ranked = []
        else:
            # cosine ≤ 0 的条目照常传入：fuse 按「零与负分无证据」契约自行滤除、不占排名位。
            vector_ranked = sorted(
                (
                    VectorHit(item_id=item_id(item), cosine=_cosine(query_vector, corpus))
                    for item, corpus in zip(pool, corpus_vectors)
                ),
                key=lambda hit: (-hit.cosine, hit.item_id),
            )

    fused = fuse(bm25_ranked, vector_ranked)
    query_terms = {term for term in tokenize(query) if substantive_term(term)}
    corpus_tokens = {
        item_id(item): {
            term for term in tokenize(corpus_text(item)) if substantive_term(term)
        }
        for item in pool
    }
    by_id = {item_id(item): item for item in pool}
    return [
        CandidateHit(
            item=by_id[hit.item_id],
            fused_score=hit.fused_score,
            matched_terms=tuple(sorted(query_terms & corpus_tokens[hit.item_id])),
            exact=hit.exact,
            bm25_score=hit.bm25_score,
            bm25_relevance=hit.bm25_relevance,
            cosine=hit.cosine,
        )
        for hit in fused
    ]

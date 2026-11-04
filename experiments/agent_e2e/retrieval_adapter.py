"""e2e 检索适配器（E2）：资格过滤 → :func:`rank_candidates` 全量排序 → 前 k 展示窗口。

A0 固定形态（方案 E-D3）：先资格过滤（allowed_ids ∩ kind 匹配）再检索——资格审查
不合格者根本不进候选（ranking.py「候选集由调用方先过滤后融合」纪律）；本适配器
不截断排序，k 只截 ``shown``（展示层），全量排序证据留在 ``evidence`` 供宿主记录。

检索语料 = 声明面身份字段 ``(id, name, description)`` 的 :class:`CorpusProjection`
声明式投影（E-D4；catalog-v1 无 tags 字段，技能正文不进一阶段检索文本——契约第一节）。
不注入向量路：冒烟固定 BM25 单路。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_retrieval import CorpusProjection, rank_candidates

from experiments.agent_e2e.loading import Catalog, ResourceSpec


@dataclass(frozen=True)
class SearchResult:
    """一次 search 的双面产出。

    ``shown`` 是 Agent 可见的前 k 个 :class:`ResourceSpec`；``evidence`` 是宿主留存的
    全量排序证据（未截断）：``{"query", "kind", "ranking": [{id, fused_score,
    matched_terms, exact}, ...]}``——契约第一节「宿主记录未截断的排序证据」的落地。
    """

    shown: list[ResourceSpec]
    evidence: dict


class RetrievalAdapter:
    """单 episode 检索适配器：持 catalog，无 episode 内可变状态（可跨 episode 共享）。"""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._corpus_text = CorpusProjection(("id", "name", "description")).to_callable()

    def search(
        self,
        query: str,
        allowed_ids: frozenset[str],
        kind: str | None = None,
        k: int = 5,
    ) -> SearchResult:
        """资格过滤（allowed_ids ∩ kind）后全量排序；前 k 进 shown，全量进 evidence。"""
        eligible = [
            resource
            for resource in self.catalog.resources
            if resource.id in allowed_ids and (kind is None or resource.kind == kind)
        ]
        hits = rank_candidates(
            eligible,
            query,
            corpus_text=self._corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
        )
        ranking = [
            {
                "id": hit.item.id,
                "fused_score": hit.fused_score,
                "matched_terms": list(hit.matched_terms),
                "exact": hit.exact,
            }
            for hit in hits
        ]
        return SearchResult(
            shown=[hit.item for hit in hits[:k]],
            evidence={"query": query, "kind": kind, "ranking": ranking},
        )

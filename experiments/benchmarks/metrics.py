"""Retrieval quality metrics over full ranked candidate lists.

Both ToolRet and SkillRet ship binary relevance labels only (``relevance == 1``),
so NDCG degenerates to the binary form. Metrics consume the FULL ranking produced
by ``rank_candidates`` — the kernel returns full recall and truncation at k happens
here at the presentation layer (kernel behavioral contract #2).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], golds: Iterable[str], k: int) -> float:
    """Fraction of gold items present in the top k."""
    gold_set = set(golds)
    if not gold_set:
        return 0.0
        # 判定空 gold 不合法由加载层保证；这里防御性返回零，让聚合不崩。
    found = gold_set.intersection(ranked_ids[:k])
    return len(found) / len(gold_set)


def completeness_at_k(ranked_ids: Sequence[str], golds: Iterable[str], k: int) -> float:
    """1.0 when every gold item is within the top k (SkillRet's Completeness@k)."""
    gold_set = set(golds)
    if not gold_set:
        return 0.0
    return 1.0 if gold_set.issubset(ranked_ids[:k]) else 0.0


def mrr(ranked_ids: Sequence[str], golds: Iterable[str]) -> float:
    """Reciprocal rank of the first gold hit; 0.0 when none appears."""
    gold_set = set(golds)
    for position, item in enumerate(ranked_ids, start=1):
        if item in gold_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], golds: Iterable[str], k: int) -> float:
    """Binary-gain NDCG@k over the full ranking truncated at k."""
    gold_set = set(golds)
    if not gold_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(ranked_ids[:k])
        if item in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg


def score_case(
    ranked_ids: Sequence[str],
    golds: Iterable[str],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """All per-case metrics in one dict, keyed ``metric@k`` (``mrr`` is k-free)."""
    scores: dict[str, float] = {"mrr": mrr(ranked_ids, golds)}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(ranked_ids, golds, k)
        scores[f"completeness@{k}"] = completeness_at_k(ranked_ids, golds, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, golds, k)
    return scores


def aggregate(
    rows: Sequence[dict[str, float]],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Mean of every per-case metric across cases, rounded to 4 decimals."""
    if not rows:
        return {name: 0.0 for name in _metric_names(k_values)}
    means = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in _metric_names(k_values)
    }
    return {name: round(value, 4) for name, value in means.items()}


def _metric_names(k_values: Sequence[int]) -> list[str]:
    names = ["mrr"]
    for k in k_values:
        names.extend((f"recall@{k}", f"completeness@{k}", f"ndcg@{k}"))
    return names

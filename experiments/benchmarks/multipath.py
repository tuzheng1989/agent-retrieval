"""Field-level multi-path retrieval with caller-side RRF merge (experiments side).

The kernel fuses exactly two paths per call (BM25 + vector) over one projected
corpus. Splitting the declared face into per-field paths is a caller decision:
each field gets its own ``BM25Index`` so a noisy field cannot dilute another's
evidence — SkillRet's own paper reports field-separate beats concatenated hybrid
retrieval. The N-way merge reuses the kernel's ``RRF_K`` so caller-side fusion
stays numerically consistent with kernel fusion.

Determinism: per-field rankings sort by ``(-score, id)`` and the merge sorts by
``(-score, id)`` — same inputs are bit-identical, matching the kernel contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_retrieval import BM25Index, Embedder
from agent_retrieval.core.fusion import RRF_K
from agent_retrieval.core.resource_index import substantive_term


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Merge N rankings with reciprocal-rank fusion; ranks start at 1.

    Scores accumulate across paths; ties break by lexicographic id, so the
    merged order is deterministic even when two paths fully agree.
    """
    totals: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking):
            totals[item_id] = totals.get(item_id, 0.0) + 1.0 / (k + position + 1)
    return [item_id for item_id, _ in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))]


def vector_ranking(
    resources: Sequence,
    corpus_text: Callable,
    query: str,
    embedder: Embedder,
) -> list[str]:
    """Cosine ranking over the projected corpus; cosine <= 0 carries no evidence
    and takes no rank slot (same contract as the kernel's vector path)."""
    vectors = embedder.embed_corpus([corpus_text(resource) for resource in resources])
    query_vector = embedder.embed_query(query)
    scored = []
    for resource, vector in zip(resources, vectors):
        cosine = sum(a * b for a, b in zip(query_vector, vector))
        if cosine > 0:
            scored.append((resource.id, cosine))
    return [item_id for item_id, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))]


class FieldRetriever:
    """Prebuilt per-field BM25 indexes over a fixed pool, RRF-merged on query.

    Indexes are built once (pool is fixed for a whole benchmark run) and each
    query only scores — a 44k pool costs ~10 s per field to build, ~1 s to score.
    """

    def __init__(self, resources: Sequence, paths: Sequence[tuple[str, Callable]]) -> None:
        self._items = tuple(resources)
        self._fields: list[tuple[str, BM25Index]] = [
            (label, BM25Index([(resource, projection(resource)) for resource in resources]))
            for label, projection in paths
        ]

    def bm25_ranking(self, query: str) -> list[str]:
        """RRF merge of the per-field BM25 rankings (score > 0 only)."""
        rankings = []
        for _, index in self._fields:
            scored = index.score(query, term_filter=substantive_term)
            rankings.append([
                match.item.id
                for match in sorted(
                    (m for m in scored if m.score > 0), key=lambda m: (-m.score, m.item.id),
                )
            ])
        return rrf_merge(rankings)

    def rank(self, query: str, embedder: Embedder | None = None,
             corpus_text: Callable | None = None) -> list[str]:
        """Fields-BM25 ranking, optionally RRF-merged with a vector path."""
        rankings = [self.bm25_ranking(query)]
        if embedder is not None:
            if corpus_text is None:
                raise ValueError("corpus_text is required when a vector path is attached")
            rankings.append(vector_ranking(self._items, corpus_text, query, embedder))
        return rrf_merge(rankings)

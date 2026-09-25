"""并行双路（BM25 + 向量）RRF 融合的确定性纯函数内核。"""
from __future__ import annotations

from dataclasses import dataclass

# RRF 的 K 是内核常数，循 DEFAULT_BM25_K1/B 先例不进 configs：它是排序形态的一部分
# 而非运维参数，改它等于换公式，必须走规格与评测闸，而不是改配置。
RRF_K = 60


@dataclass(frozen=True)
class BM25Hit:
    """BM25 路排名中的单个候选，判定面信号全量携带（fusion-ranking §3.1/§3.4）。

    ``exact`` 是全等命中标记；BM25 路对 exact 候选的保留（``score > 0`` 或
    ``exact`` 入排名）是调用方契约——见 :func:`fuse`。
    """

    item_id: str
    score: float
    relevance: float
    overlap: int
    declared_overlap: int
    exact: bool


@dataclass(frozen=True)
class VectorHit:
    """向量路排名中的单个候选，只携带余弦分。"""

    item_id: str
    cosine: float


@dataclass(frozen=True)
class FusedHit:
    """融合后的单个候选（fusion-ranking §3.4）。

    ``kind`` 是不透明占位字符串：kind 过滤发生在候选集构造（先过滤后融合），内核
    不建模 kind 语义，恒为 ``""``——调用方投影时用 ``dataclasses.replace(hit,
    kind=...)`` 填充并映射 RegistryKind。
    """

    item_id: str
    fused_score: float
    exact: bool
    bm25_score: float | None
    bm25_relevance: float | None
    overlap: int | None
    declared_overlap: int | None
    cosine: float | None
    in_bm25: bool
    in_vector: bool
    kind: str = ""


def fuse(
    bm25_ranked: list[BM25Hit],
    vector_ranked: list[VectorHit],
) -> list[FusedHit]:
    """对两路全量排名做 RRF 融合，返回排好序的 ``FusedHit`` 列表。

    接口契约（调用方必读）：

    - **入参是两路全量排名**：路内排名由入参顺序携带（rank = 位置 + 1），内核不重排
      路内序、不截断（fusion-ranking §4：k 只截断输出，是查询面的事）；同一路内
      ``item_id`` 必须唯一，重复行为未定义；
    - **exact 置顶是调用方契约**：``exact`` 只能由 BM25 路携带，内核不生成、不补录。
      排序键 ``(not exact, -fused, id)`` 只依赖候选自身的 exact 标记，但前提是调用方
      把 exact 候选放进至少一路入参（BM25 适配器按 ``score > 0`` 过滤时必须保留
      ``exact=True`` 的元素）——「任一路都不含」的 exact 候选根本不进入内核，置顶会
      静默失效且内核不可检测（它无法知道入参之外还有什么）。输出候选集恒等于两路
      入参的并集（见下），不存在「入参之外被置顶」的候选；
    - **cosine ≤ 0 不入向量排名**（fusion-ranking §3.1：零与负分无证据）：不计向量路
      RRF 贡献、``in_vector=False``；只出现在向量路且 cosine ≤ 0 的候选不属于任何
      一路，不出现在输出中；
    - **缺席路字段为 None**：``None`` = 该路未提供此候选（证据缺席）；非 None（含
      0.0 与负值）= 该路明确给出的值。0.0 在 BM25 语义里是合法真值（exact 命中即
      score=0），用 0.0 表达缺席会混淆两种状态，故选 None；
    - **确定性**（fusion-ranking §6）：同一候选的贡献按固定顺序累加（先 BM25 后
      向量），排序键含 id tie-break——同输入两次调用输出逐位一致。

    RRF：``fused(d) = Σ 1/(K + rank_p(d))``，K = :data:`RRF_K`（见其注释）。
    """
    vector_ranks: dict[str, int] = {}
    vector_cosines: dict[str, float] = {}
    vector_position = 0
    for hit in vector_ranked:
        vector_cosines[hit.item_id] = hit.cosine
        if hit.cosine > 0:
            # 负分/零分条目不占排名位：调用方已过滤时二者等价，防御路径下重编号才
            # 符合「cosine > 0 入排名」的语义——被滤者根本不在排名里。
            vector_position += 1
            vector_ranks[hit.item_id] = vector_position
    bm25_ranks = {hit.item_id: rank for rank, hit in enumerate(bm25_ranked, start=1)}
    bm25_by_id = {hit.item_id: hit for hit in bm25_ranked}

    candidates = list(bm25_ranks)
    candidates.extend(item_id for item_id in vector_ranks if item_id not in bm25_ranks)

    hits: list[FusedHit] = []
    for item_id in candidates:
        # 先 BM25 后向量固定顺序累加：浮点加法不可交换，顺序固定才有逐位可复现。
        bm25_part = 1 / (RRF_K + bm25_ranks[item_id]) if item_id in bm25_ranks else 0.0
        vector_part = 1 / (RRF_K + vector_ranks[item_id]) if item_id in vector_ranks else 0.0
        bm25_hit = bm25_by_id[item_id] if item_id in bm25_by_id else None
        hits.append(
            FusedHit(
                item_id=item_id,
                fused_score=bm25_part + vector_part,
                exact=bm25_hit.exact if bm25_hit else False,
                bm25_score=bm25_hit.score if bm25_hit else None,
                bm25_relevance=bm25_hit.relevance if bm25_hit else None,
                overlap=bm25_hit.overlap if bm25_hit else None,
                declared_overlap=bm25_hit.declared_overlap if bm25_hit else None,
                cosine=vector_cosines.get(item_id),
                in_bm25=item_id in bm25_ranks,
                in_vector=item_id in vector_ranks,
            )
        )
    hits.sort(key=lambda hit: (not hit.exact, -hit.fused_score, hit.item_id))
    return hits

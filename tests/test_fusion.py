"""检索内核融合层：RRF 纯函数与存储/嵌入端口的回归测试。"""

import pytest

from agent_retrieval.core.fusion import BM25Hit, VectorHit, fuse
from agent_retrieval.core.ports import Embedder, InMemoryVectorStore, MockEmbedder, VectorStore


def test_dual_path_rank_one_tops_with_hand_computed_rrf_score():
    """验收 1：X 双路 rank1，fused == 2/61，置顶；Y（BM25 rank2、向量 rank3）次之。

    向量排名 Y(3) 隐含一个只出现在向量路的 rank2 候选 Z——单路候选照常参与融合，
    fused = 1/62，落在 Y 之后。
    """
    hits = fuse(
        [
            BM25Hit(
                item_id="X", score=3.0, relevance=0.9, overlap=2, declared_overlap=1, exact=False
            ),
            BM25Hit(
                item_id="Y", score=1.0, relevance=0.3, overlap=1, declared_overlap=0, exact=False
            ),
        ],
        [
            VectorHit(item_id="X", cosine=0.9),
            VectorHit(item_id="Z", cosine=0.6),
            VectorHit(item_id="Y", cosine=0.3),
        ],
    )

    assert [hit.item_id for hit in hits] == ["X", "Y", "Z"]
    # 手算：X 双路 rank1 → 1/61 + 1/61 = 2/61；期望值是算术字面量，不是实现同式重算。
    assert hits[0].fused_score == 2 / 61
    assert hits[1].fused_score == 1 / 62 + 1 / 63
    assert hits[2].fused_score == 1 / 62


def test_exact_candidate_tops_without_relying_on_rank_or_vector_path():
    """验收 2：exact 候选 E 在 BM25 路排名末位、向量路不含 E，仍输出首名。

    置顶只依赖候选自身的 exact 标记（fusion-ranking §3.3 排序键第一层），
    与排名位置和向量路共识无关；向量路缺席字段为 None（证据缺席，区别于零分）。
    """
    hits = fuse(
        [
            BM25Hit(
                item_id="A", score=3.0, relevance=0.9, overlap=2, declared_overlap=1, exact=False
            ),
            BM25Hit(
                item_id="B", score=1.0, relevance=0.3, overlap=1, declared_overlap=0, exact=False
            ),
            BM25Hit(
                item_id="E", score=0.0, relevance=0.0, overlap=0, declared_overlap=0, exact=True
            ),
        ],
        [VectorHit(item_id="A", cosine=0.8)],
    )

    assert [hit.item_id for hit in hits] == ["E", "A", "B"]
    assert hits[0].exact is True
    assert hits[0].in_bm25 is True
    assert hits[0].in_vector is False
    assert hits[0].cosine is None
    assert hits[0].bm25_score == 0.0


def test_tied_fused_scores_break_by_lexicographic_item_id():
    """验收 3：两候选同 fused 分（各单路 rank1 → 1/61），输出按 id 字典序。

    tie-break 不得依赖入参顺序或路的先后——A 在向量路、B 在 BM25 路，
    字典序仍要求 A 先于 B（fusion-ranking §6：无并列输出）。
    """
    hits = fuse(
        [
            BM25Hit(
                item_id="B", score=2.0, relevance=0.5, overlap=1, declared_overlap=1, exact=False
            )
        ],
        [VectorHit(item_id="A", cosine=0.7)],
    )

    assert [hit.item_id for hit in hits] == ["A", "B"]
    assert hits[0].fused_score == hits[1].fused_score == 1 / 61
    assert hits[0].in_bm25 is False and hits[0].in_vector is True
    assert hits[1].in_bm25 is True and hits[1].in_vector is False


def test_non_positive_cosine_never_enters_the_vector_ranking():
    """cosine ≤ 0 无证据，不入向量排名：不计 RRF 贡献、in_vector=False（fusion-ranking §3.1）。

    - NEGV 只在向量路且 cosine=-0.1：不属于任何一路排名，不得凭空出现在输出里；
    - Z 在 BM25 路 rank1、向量路给出 cosine=-0.5：单路照常参与，fused 只含 BM25
      贡献 1/61（不是 1/61 + 1/62），负分原样保留；
    - W 在 BM25 路 rank2、向量路给出 cosine=0.0：fused = 1/62，零分原样保留
      （0.0 是"明确给出的零分"，区别于缺席的 None）。
    """
    hits = fuse(
        [
            BM25Hit(
                item_id="Z", score=2.0, relevance=0.5, overlap=1, declared_overlap=0, exact=False
            ),
            BM25Hit(
                item_id="W", score=1.0, relevance=0.25, overlap=1, declared_overlap=1, exact=False
            ),
        ],
        [
            VectorHit(item_id="Z", cosine=-0.5),
            VectorHit(item_id="W", cosine=0.0),
            VectorHit(item_id="NEGV", cosine=-0.1),
        ],
    )

    assert [hit.item_id for hit in hits] == ["Z", "W"]
    assert hits[0].in_vector is False and hits[0].cosine == -0.5
    assert hits[0].fused_score == 1 / 61
    assert hits[1].in_vector is False and hits[1].cosine == 0.0
    assert hits[1].fused_score == 1 / 62


def test_verdict_signals_pass_through_from_the_bm25_path():
    """判定面信号（relative relevance / overlap / declared_overlap）原样透传不换算。

    A 层 clearly_related 判据读这些字段（fusion-ranking §8），融合层只搬运不加工。
    """
    hits = fuse(
        [
            BM25Hit(
                item_id="R", score=1.7, relevance=0.85, overlap=3, declared_overlap=2, exact=False
            )
        ],
        [],
    )

    hit = hits[0]
    assert (hit.bm25_score, hit.bm25_relevance) == (1.7, 0.85)
    assert (hit.overlap, hit.declared_overlap) == (3, 2)


def test_same_input_produces_bit_identical_output_across_calls():
    """确定性：同输入两次调用逐位一致（评审要点）——无随机、无时间因子、求和顺序固定。"""
    bm25_ranked = [
        BM25Hit(
            item_id=f"cand-{i}",
            score=10.0 - i,
            relevance=0.5,
            overlap=1,
            declared_overlap=0,
            exact=False,
        )
        for i in range(20)
    ]
    vector_ranked = [
        VectorHit(item_id=f"cand-{i}", cosine=0.9 - i * 0.01) for i in range(20)
    ]

    first = fuse(bm25_ranked, vector_ranked)
    second = fuse(bm25_ranked, vector_ranked)

    assert first == second


def test_in_memory_vector_store_roundtrip_and_write_once():
    """快照写入后不可变：读出的是拷贝，同一 snapshot_id 二次 publish 拒绝。

    write-once 是 B7 发布管线「发布前 read_meta 预检，id 一致零重嵌」的前提——
    若 store 允许覆盖，预检与写入之间的竞态会静默换掉快照，破坏快照确定语义。
    """
    store: VectorStore = InMemoryVectorStore()
    assert store.load("missing") is None
    assert store.read_meta("missing") is None

    store.publish("snap-1", {"doc-a": [0.5, 0.25]}, {"model": "mock", "corpus_hash": "abc"})
    assert store.read_meta("snap-1") == {"model": "mock", "corpus_hash": "abc"}
    assert store.load("snap-1") == {"doc-a": (0.5, 0.25)}

    # 读出的是拷贝：改返回值不得渗入存储（快照写入后不可变）。
    loaded = store.load("snap-1")
    assert loaded is not None
    loaded["doc-a"] = (9.9,)
    assert store.load("snap-1") == {"doc-a": (0.5, 0.25)}
    meta = store.read_meta("snap-1")
    assert meta is not None
    meta["model"] = "tampered"
    assert store.read_meta("snap-1") == {"model": "mock", "corpus_hash": "abc"}

    with pytest.raises(ValueError):
        store.publish("snap-1", {"doc-a": [1.0]}, {"model": "other"})


def test_mock_embedder_is_deterministic_and_normalizes():
    """同文本两次嵌入逐位一致；embed_corpus 与 embed_query 逐元素一致。

    归一化成单位向量：真实嵌入器（B7 ApiEmbedder/LocalEmbedder）产出单位向量，
    Mock 与之同语义，融合臂的 cosine 在 Mock 与生产嵌入器之间行为一致。
    """
    embedder: Embedder = MockEmbedder(dimensions=8)

    first = embedder.embed_query("推演剧本编排")
    second = embedder.embed_query("推演剧本编排")
    assert first == second
    assert len(first) == 8
    assert sum(value * value for value in first) == pytest.approx(1.0)

    corpus = embedder.embed_corpus(["推演剧本编排", "render sales as chart"])
    assert corpus == [
        embedder.embed_query("推演剧本编排"),
        embedder.embed_query("render sales as chart"),
    ]
    assert corpus[0] != corpus[1]

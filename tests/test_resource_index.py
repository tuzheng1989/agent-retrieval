"""资源 BM25 适配器（检索面/判定面分离）的回归测试。

自 evochat test_bm25_core.py 适配器用例迁移；语料以裸字符串承载，
不引入任何宿主资源模型——适配器本就只认 ``(item, text)`` 对。
"""

from agent_retrieval.core.resource_index import ResourceBM25Index, clearly_related


def test_resource_exact_identity_remains_a_policy_not_core_behavior():
    index = ResourceBM25Index([("resource", "completely unrelated body")])

    hits = index.search(
        "resource-id",
        identity=lambda _item: ("resource-id", "Resource Name"),
    )

    assert len(hits) == 1
    assert hits[0].exact is True
    assert hits[0].score == 0
    assert hits[0].overlap == 0


def test_single_cjk_function_words_never_establish_resource_relevance():
    """单字中文虚词是索引召回手段，不是相关性证据。

    tokenize 保留一元组是为了让"图"这类单字查询在关键词检索里召回得到；代价是长自然
    语言查询里"的/于/天/关"必然与任何中文资源重叠。资源发现侧必须把它们从查询中剔除，
    否则"写一首关于秋天的诗"会稳定命中毫不相干的资源。
    """
    documents = [("scenario", "推演剧本编排\n覆盖关键节点于每天的推演流程")]
    index = ResourceBM25Index(documents, declared=lambda _item: "scenario\n推演剧本编排")

    noise = index.search("写一首关于秋天的诗", identity=lambda item: (item, item))
    assert [hit for hit in noise if clearly_related(hit, minimum_relevance=0.15)] == []

    real = index.search("帮我做推演剧本", identity=lambda item: (item, item))
    assert clearly_related(real[0], minimum_relevance=0.15)


def test_body_only_evidence_is_recall_not_proof_of_relevance():
    """正文 n-gram 可以召回，但判定必须落在资源声明面上。

    正文切分会产出"一张""的机"这类跨词边界噪声词，它们照样能凑够分数和实词重叠数，
    但不可能出现在资源的 id/name/description/tags/use_when 里。
    """
    documents = [("flow", "决策推演\n正文提到过一张明细表以及的机制说明")]
    index = ResourceBM25Index(documents, declared=lambda _item: "flow\n决策推演")

    hits = index.search("帮我订一张明天去上海的机票", identity=lambda item: (item, item))
    assert hits and hits[0].score > 1.0 and hits[0].overlap >= 2
    assert hits[0].declared_overlap == 0
    assert not clearly_related(hits[0], minimum_relevance=0.15)


def _padded_index(extra: int):
    """同一个目标资源，外加 ``extra`` 条互不相干的填充资源。"""
    documents = [("scenario", "推演剧本编排\n覆盖关键节点于每天的推演流程")]
    documents += [
        (f"pad-{i}", f"填充资源{i}\n与推演无关的会计报销流程说明{i}") for i in range(extra)
    ]
    return ResourceBM25Index(documents, declared=lambda item: (
        "scenario\n推演剧本编排" if item == "scenario" else f"{item}\n填充资源"
    ))


def test_relevance_survives_corpus_growth_while_raw_score_does_not():
    """判定必须与语料规模无关——这是 ``clearly_related`` 改用相对分的全部理由。

    BM25 的 idf 是 ``log(1+(N-df+0.5)/(df+0.5))``，随语料变大整体抬升：同一份查询与
    同一个文档，原始分会翻好几倍。于是任何**绝对**分数线都只在标定时那个规模上成立，
    资源库一长大就悄悄失效——判定越来越松、不报错、只是慢慢变差。相对分
    ``score/ideal_score`` 的分子分母同步缩放，比值稳定。
    """
    raw, rel = {}, {}
    for extra in (0, 10, 50):
        hit = _padded_index(extra).search("帮我做推演剧本", identity=lambda i: (i, i))[0]
        assert hit.item == "scenario"
        raw[extra], rel[extra] = hit.score, hit.relevance

    # 原始分随语料显著抬升——正是绝对阈值失效的机制。
    assert raw[50] > raw[0] * 5
    # 相对分几乎不动。不断言"完全相等"：tf 与文档长度效应仍在，做不到严格不变，
    # 要的是"阈值不会因为语料变大而被跨过"。
    assert abs(rel[50] - rel[0]) / rel[0] < 0.25
    assert abs(rel[50] - rel[10]) / rel[10] < 0.05
    assert all(clearly_related(_padded_index(n).search(
        "帮我做推演剧本", identity=lambda i: (i, i))[0], minimum_relevance=0.15)
        for n in (0, 10, 50))


def test_noise_stays_rejected_as_the_corpus_grows():
    """反向同理：噪声查询在语料变大后也不能因为分数被抬高而混进来。"""
    for extra in (0, 10, 50):
        hits = _padded_index(extra).search("写一首关于秋天的诗", identity=lambda i: (i, i))
        assert [h for h in hits if clearly_related(h, minimum_relevance=0.15)] == []


def test_generic_items_work_without_any_resource_model():
    """泛型 T 用最小 dataclass 承载：适配器对宿主资源模型零依赖（包化守卫）。"""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Stub:
        key: str
        label: str

    index = ResourceBM25Index(
        [(Stub("a", "推演剧本编排"), "推演剧本编排"), (Stub("b", "邮件收发"), "邮件收发")],
        declared=lambda stub: stub.label,
    )

    hits = index.search("推演剧本编排", identity=lambda stub: (stub.key, stub.label))

    assert [hit.item.key for hit in hits] == ["a"]

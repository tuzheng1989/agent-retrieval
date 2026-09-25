"""BM25 数学内核的回归测试（自 evochat test_bm25_core.py 内核用例迁移）。"""

import pytest

from agent_retrieval.core.bm25 import BM25Index, content_length, tokenize


def test_shared_core_scores_multilingual_documents_and_reports_overlap():
    index = BM25Index([
        ("chart", "把销售数据画成柱状图 render sales data as a chart"),
        ("mail", "发送邮件并查询收件箱"),
    ])

    scores = {match.item: match for match in index.score("销售数据 chart")}

    assert scores["chart"].score > scores["mail"].score
    assert scores["chart"].overlap >= 2
    assert content_length("销售 data") == 3
    assert "销售" in tokenize("销售数据")


def test_relevance_ignores_query_term_repetition():
    """相对分只反映文档覆盖度，不得随**查询自身**的用词重复度抬升。

    ``_score`` 迭代未去重的查询 token（同一个词出现 q 次就累加 q 份贡献），所以
    ``ideal_score`` 也必须按重数计价。分母若去重，比值会线性膨胀、越过 1，最吃亏的是
    "查询=资源全文"那个调用点：重复度天生最高，啰嗦的文档会系统性压过简洁的。
    """
    index = BM25Index([("a", "alpha beta gamma delta"), ("b", "zeta eta theta")])

    relevances = [
        index.score(query)[0].score / index.ideal_score(query)
        for query in ("alpha beta", "alpha alpha beta", "alpha alpha alpha beta")
    ]

    assert relevances[0] <= 1.0
    assert relevances[1] == pytest.approx(relevances[0])
    assert relevances[2] == pytest.approx(relevances[0])


def test_shared_core_keeps_empty_corpus_and_query_deterministic():
    assert BM25Index([]).score("chart") == ()
    scores = BM25Index([("chart", "chart")]).score("")
    assert [(match.item, match.score, match.overlap) for match in scores] == [
        ("chart", 0.0, 0),
    ]

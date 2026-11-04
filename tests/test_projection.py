"""CorpusProjection 声明式投影的契约测试（SDK P1 / C1）。

测试面：构造即校验（空 fields、含空串抛 ValueError）、frozen 不可变、
``to_callable()`` 投影行为（空 part 过滤 + separator 拼接，与
``experiments/golden/gate.py`` 的 ``_plain_text`` 过滤语义逐位对齐）、
未知字段抛 AttributeError、fields 可 JSON 序列化、与 ``rank_candidates``
内核的组合（投影函数与手写函数产出逐位一致的 hits）。
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, dataclass

import pytest

from agent_retrieval import CorpusProjection, rank_candidates


@dataclass(frozen=True)
class Resource:
    """宿主资源最小形状（照 experiments/jev_retrieval.py 的 Resource，id/name/
    description/tags/use_when 五字段）——字段语义由宿主建模，内核零建模。"""

    id: str
    name: str
    description: str
    tags: str = ""
    use_when: str = ""


def _plain_text(resource: Resource) -> str:
    """独立真值：手写复现 gate._plain_text 的四字段投影（空 part 过滤 + 单空格 join）。"""
    return " ".join(
        part for part in (resource.id, resource.name, resource.description, resource.tags) if part
    )


class TestConstructionValidation:
    def test_empty_fields_raises(self):
        with pytest.raises(ValueError, match="fields"):
            CorpusProjection(())

    def test_empty_string_field_raises(self):
        with pytest.raises(ValueError, match="fields"):
            CorpusProjection(("name", ""))

    def test_frozen(self):
        projection = CorpusProjection(("id", "name"))
        with pytest.raises(FrozenInstanceError):
            projection.fields = ("other",)  # type: ignore[misc]


class TestProjectionBehavior:
    _RESOURCES = [
        Resource(id="r1", name="Joke API", description="serves jokes", tags="fun", use_when="想笑"),
        Resource(id="r2", name="Chart Tool", description="draws charts", tags=""),
        Resource(id="r3", name="Weather", description="", tags="weather"),
        # dataclass 不做类型检查：None 取值链照空串语义丢弃（_plain_text 的 if part 同口径）。
        Resource(id="r4", name="Emailer", description=None, tags=""),  # type: ignore[arg-type]
        Resource(id="r5", name="", description="", tags=""),
    ]

    def test_four_field_projection_matches_plain_text_truth_bitwise(self):
        """四字段投影与手写 _plain_text（独立真值，非同义反复）对同一资源逐位一致。"""
        project = CorpusProjection(("id", "name", "description", "tags")).to_callable()
        for resource in self._RESOURCES:
            assert project(resource) == _plain_text(resource), resource.id

    def test_three_field_combination_constructs(self):
        """("name","description","tags") 三字段组合可构造（skillret 字段集同构）。"""
        projection = CorpusProjection(("name", "description", "tags"))
        project = projection.to_callable()
        assert project(Resource(id="r", name="N", description="D", tags="T")) == "N D T"

    def test_empty_description_produces_no_double_space(self):
        project = CorpusProjection(("id", "name", "description", "tags")).to_callable()
        corpus = project(Resource(id="r3", name="Weather", description="", tags="weather"))
        assert "  " not in corpus
        assert corpus == "r3 Weather weather"

    def test_all_empty_fields_return_empty_string(self):
        project = CorpusProjection(("id", "name", "description", "tags")).to_callable()
        assert project(Resource(id="", name="", description="", tags="")) == ""

    def test_none_value_part_is_dropped(self):
        project = CorpusProjection(("id", "name", "description", "tags")).to_callable()
        corpus = project(Resource(id="r4", name="Emailer", description=None, tags=""))  # type: ignore[arg-type]
        assert corpus == "r4 Emailer"

    def test_unknown_field_raises_attribute_error(self):
        """字段缺失不进构造校验（零建模纪律）：getattr 在调用时天然抛。"""
        project = CorpusProjection(("id", "nonexistent")).to_callable()
        with pytest.raises(AttributeError):
            project(Resource(id="r1", name="N", description=""))

    def test_non_default_separator_takes_effect(self):
        project = CorpusProjection(("id", "name"), separator="|").to_callable()
        assert project(Resource(id="r1", name="Joke API", description="")) == "r1|Joke API"

    def test_fields_are_json_serializable(self):
        """D5 证据：fields 天然可 JSON，宿主自行 json.dumps(list(...))，包内无序列化 API。"""
        projection = CorpusProjection(("id", "name", "description", "tags"))
        assert json.loads(json.dumps(list(projection.fields))) == [
            "id", "name", "description", "tags",
        ]


def _hit_row(hit):
    """hits 注解字段全量展平：item 序列与全部注解逐位比较。"""
    return (
        hit.item.id, hit.fused_score, hit.matched_terms, hit.exact,
        hit.bm25_score, hit.bm25_relevance, hit.cosine,
    )


class TestRankCandidatesCombination:
    """接缝③：CorpusProjection.to_callable() 经 rank_candidates(corpus_text=...) 与内核组合。"""

    _RESOURCES = [
        Resource(id="chart-tool", name="Chart Tool", description="图表统计报表柱状图绘图"),
        Resource(id="jokes", name="Joke API", description="serves jokes about charts", tags="fun"),
        Resource(id="weather", name="Weather", description="", tags="weather forecast"),
    ]
    _QUERY = "图表统计"

    def test_hits_match_handwritten_projection_bitwise(self):
        """投影函数与手写函数各跑一遍 rank_candidates：hits 的 item 序列与注解字段逐位一致。"""
        projection_hits = rank_candidates(
            self._RESOURCES, self._QUERY,
            corpus_text=CorpusProjection(("id", "name", "description", "tags")).to_callable(),
            item_id=lambda r: r.id, item_name=lambda r: r.name,
        )
        handwritten_hits = rank_candidates(
            self._RESOURCES, self._QUERY,
            corpus_text=_plain_text,
            item_id=lambda r: r.id, item_name=lambda r: r.name,
        )

        assert [_hit_row(hit) for hit in projection_hits] == [
            _hit_row(hit) for hit in handwritten_hits
        ]
        # 语料确实参与排序：强命中者置顶（零分候选不进 hits 是 resource_index.py:89
        # 的既有契约——rank_candidates 经 search 继承）。
        assert projection_hits[0].item.id == "chart-tool"
        assert any(hit.matched_terms for hit in projection_hits)

"""Unit tests for doc2query generation parsing and corpus projection."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.dataset import Resource
from experiments.benchmarks.docgen import (
    docgen_corpus_text,
    load_docgen,
    parse_generated_queries,
)


def test_parse_generated_queries_strips_numbering_and_dedupes():
    text = (
        "1. find a joke by its ID\n"
        "2. look up a specific joke\n"
        "3. FIND A JOKE BY ITS ID\n"   # 大小写去重后与第 1 条重复
        "\n"
        "4. get joke number 123\n"
        "   5. get joke number 123\n"  # 重复
    )
    assert parse_generated_queries(text, 5) == [
        "find a joke by its ID",
        "look up a specific joke",
        "get joke number 123",
    ]


def test_parse_generated_queries_caps_at_k():
    text = "a\nb\nc\nd\ne\nf\ng"
    assert parse_generated_queries(text, 3) == ["a", "b", "c"]


def test_docgen_corpus_text_appends_expansion_and_falls_back():
    base = lambda r: f"{r.name} {r.description}"  # noqa: E731 — 测试内一行投影
    expansions = {"r1": ["how do I query data", "lookup records"]}
    r1, r2 = Resource("r1", "Geo", "population data"), Resource("r2", "Doc", "documents")
    corpus = docgen_corpus_text(base, expansions)
    assert corpus(r1) == "Geo population data how do I query data lookup records"
    assert corpus(r2) == base(r2)  # 无扩展的资源回退原投影


def test_load_docgen_reads_jsonl(tmp_path: Path):
    path = tmp_path / "docgen-skillret.jsonl"
    path.write_text(
        json.dumps({"resource_id": "a", "queries": ["q1"], "model": "m"}) + "\n"
        + "\n"
        + json.dumps({"resource_id": "b", "queries": ["q2", "q3"], "model": "m"}) + "\n",
        encoding="utf-8",
    )
    assert load_docgen(tmp_path, "skillret") == {"a": ["q1"], "b": ["q2", "q3"]}
    assert load_docgen(tmp_path, "toolret") == {}  # 缺文件 → 空（不阻塞评测）

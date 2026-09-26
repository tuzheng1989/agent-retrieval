"""SkillRet adapter: pinned test split projected onto the benchmark shape.

Dataset: https://huggingface.co/datasets/ThakiCloud/SKILLRET (Apache-2.0).
Protocol (official): retrieve skills for natural-language queries from the
evaluation skill pool; binary labels; NDCG@k / Recall@k / Completeness@k.

The Hub head is mutable, so downloads are pinned to revision ``a050ad2`` (the
revision independent replications pin — see the dataset card). Bare ``requests``
only; no ``datasets`` dependency.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource

#: Dataset revision pinned for reproducibility (mutable-head caveat in the card).
SKILLRET_REVISION = "a050ad2"
_BASE = f"https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{SKILLRET_REVISION}"
_SKILLS_FILE = "data/skills/test.jsonl"
_QUERIES_FILE = "data/queries/test.jsonl"
_QRELS_FILE = "data/qrels/test.jsonl"


def download(data_dir: Path) -> None:
    """Fetch the three test-split files once; existing files are kept as-is."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for remote in (_SKILLS_FILE, _QUERIES_FILE, _QRELS_FILE):
        target = data_dir / remote.replace("/", "_")
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"[skillret] downloading {remote} (revision {SKILLRET_REVISION}) ...", file=sys.stderr)
        with requests.get(f"{_BASE}/{remote}", stream=True, timeout=300) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)


def load(data_dir: Path) -> BenchmarkDataset:
    """Load the evaluation pool and labeled queries from downloaded files."""
    skills_path = data_dir / _SKILLS_FILE.replace("/", "_")
    queries_path = data_dir / _QUERIES_FILE.replace("/", "_")
    qrels_path = data_dir / _QRELS_FILE.replace("/", "_")
    missing = [path.name for path in (skills_path, queries_path, qrels_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SkillRet files missing, run download() first: {missing}")

    resources: list[Resource] = []
    pool_ids: set[str] = set()
    with skills_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            resource = skill_resource(row)
            resources.append(resource)
            pool_ids.add(resource.id)

    golds_by_query: dict[str, set[str]] = defaultdict(set)
    with qrels_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("relevance") or 0) > 0 and row["skill_id"] in pool_ids:
                golds_by_query[row["query_id"]].add(row["skill_id"])

    cases: list[Case] = []
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            case = skill_case(row, golds_by_query)
            if case is not None:
                cases.append(case)
    return BenchmarkDataset(
        name="skillret",
        revision=SKILLRET_REVISION,
        corpus_text=skill_corpus_text,
        resources=resources,
        cases=cases,
    )


def skill_resource(row: dict) -> Resource:
    """Project one ``skills.jsonl`` row; SKILL.md body stays out of the face."""
    tags = " ".join(
        str(row.get(key) or "")
        for key in ("namespace", "major", "sub", "primary_action", "primary_object", "domain")
    )
    return Resource(
        id=str(row["id"]),
        name=str(row.get("name") or ""),
        description=str(row.get("description") or ""),
        tags=tags,
    )


def skill_case(row: dict, golds_by_query: dict[str, set[str]]) -> Case | None:
    """Project one query row; cases whose golds fell outside the pool are dropped."""
    golds = frozenset(golds_by_query.get(row["id"], set()))
    if not golds:
        return None
    return Case(id=str(row["id"]), query=str(row["query"]), golds=golds)


def skill_corpus_text(resource: Resource) -> str:
    """Declared face: identity fields + taxonomy tags. The Markdown body is a
    permission decision and never enters the lexical corpus (kernel discipline #1)."""
    return " ".join((resource.name, resource.description, resource.tags))

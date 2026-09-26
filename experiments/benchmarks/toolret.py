"""ToolRet adapter: parquet shards projected onto the benchmark shape.

Datasets: https://huggingface.co/datasets/mangopy/ToolRet-Queries and
https://huggingface.co/datasets/mangopy/ToolRet-Tools (35 source tasks over a
heterogeneous ~43k-tool corpus; code / customized / web categories).
Protocol (official): retrieve from the FULL corpus per query; binary labels from
the per-query ``labels`` JSON string; corpus text is the ``documentation`` field
verbatim.

Parquet shards are read through ``duckdb`` — an optional benchmark dependency with
Windows-ARM64 wheels (``pyarrow`` has none), imported lazily with an install hint,
mirroring how ``agent_retrieval[api]`` treats ``requests``. Bare ``requests`` for
downloads; no ``datasets`` dependency.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource

#: All 35 source tasks in the official config listing (stable alphabetical order).
TASKS: tuple[str, ...] = (
    "apibank", "apigen", "appbench", "autotools-food", "autotools-music",
    "autotools-weather", "craft-math-algebra", "craft-tabmwp", "craft-vqa",
    "gorilla-huggingface", "gorilla-pytorch", "gorilla-tensor", "gpt4tools", "gta",
    "metatool", "mnms", "restgpt-spotify", "restgpt-tmdb", "reversechain",
    "rotbench", "t-eval-dialog", "t-eval-step", "taskbench-daily",
    "taskbench-huggingface", "taskbench-multimedia", "tool-be-honest", "toolace",
    "toolalpaca", "toolbench-sam", "toolbench", "toolemu", "tooleyes", "toollens",
    "ultratool",
)
CATEGORIES: tuple[str, ...] = ("code", "customized", "web")
_QUERIES_URL = (
    "https://huggingface.co/datasets/mangopy/ToolRet-Queries/resolve/main"
    "/{task}/queries-00000-of-00001.parquet"
)
_TOOLS_URL = (
    "https://huggingface.co/datasets/mangopy/ToolRet-Tools/resolve/main"
    "/{category}/tools-00000-of-00001.parquet"
)


def download(data_dir: Path) -> None:
    """Fetch all parquet shards once; existing files are kept as-is."""
    data_dir.mkdir(parents=True, exist_ok=True)
    targets = [(_QUERIES_URL.format(task=task), f"queries_{task}.parquet") for task in TASKS]
    targets += [(_TOOLS_URL.format(category=cat), f"tools_{cat}.parquet") for cat in CATEGORIES]
    for url, name in targets:
        target = data_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"[toolret] downloading {name} ...", file=sys.stderr)
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)


def load(data_dir: Path, limit: int | None = None) -> BenchmarkDataset:
    """Load the full tool corpus and (optionally round-robin sampled) queries.

    ``limit`` samples cases round-robin across the 35 tasks — a plain prefix
    would concentrate on whichever tasks sort first and skew the mix.
    """
    rows_by_task = _read_parquet_rows(data_dir, TASKS, "queries_{name}.parquet")
    picked = round_robin(rows_by_task, limit)
    del rows_by_task

    tool_rows = _read_parquet_rows(data_dir, CATEGORIES, "tools_{name}.parquet")
    resources = [
        tool_resource(row)
        for category in CATEGORIES
        for row in tool_rows[category]
    ]
    pool_ids = {resource.id for resource in resources}

    cases: list[Case] = []
    for row in picked:
        case = tool_case(row, pool_ids)
        if case is not None:
            cases.append(case)
    return BenchmarkDataset(
        name="toolret",
        revision="parquet-main",
        corpus_text=tool_corpus_text,
        resources=resources,
        cases=cases,
    )


def round_robin(rows_by_task: dict[str, list[dict]], limit: int | None) -> list[dict]:
    """Interleave task rows so a ``limit`` keeps the source-task mix balanced."""
    queues = [list(rows) for rows in rows_by_task.values()]
    if limit is None:
        return [row for queue in queues for row in queue]
    picked: list[dict] = []
    while len(picked) < limit:
        progressed = False
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return picked


def tool_resource(row: dict) -> Resource:
    """Project one tool row: documentation verbatim, name extracted for the
    exact-match contract (queries rarely equal ids, so this stays honest)."""
    documentation = str(row.get("documentation") or "")
    return Resource(
        id=str(row["id"]),
        name=_tool_name(documentation),
        description=documentation,
    )


def tool_case(row: dict, pool_ids: set[str]) -> Case | None:
    """Project one query row; ``labels`` is a JSON string needing a second parse."""
    labels = json.loads(row.get("labels") or "[]")
    golds = frozenset(
        str(label["id"]) for label in labels
        if int(label.get("relevance") or 0) > 0 and str(label["id"]) in pool_ids
    )
    if not golds:
        return None
    return Case(
        id=str(row["id"]),
        query=str(row.get("query") or ""),
        golds=golds,
        instruction=str(row.get("instruction") or ""),
    )


def tool_corpus_text(resource: Resource) -> str:
    """Official protocol: the ``documentation`` field is the corpus text verbatim."""
    return resource.description


def _tool_name(documentation: str) -> str:
    try:
        return str(json.loads(documentation).get("name") or "")
    except json.JSONDecodeError:
        return ""


def _read_parquet_rows(
    data_dir: Path,
    names: tuple[str, ...],
    pattern: str,
) -> dict[str, list[dict]]:
    """Read parquet shards by name pattern (``{name}`` placeholder), one list per shard."""
    try:
        import duckdb  # noqa: PLC0415 — optional benchmark dependency
    except ImportError as exc:
        raise RuntimeError(
            "ToolRet parquet shards need duckdb; install it with: pip install duckdb"
        ) from exc
    grouped: dict[str, list[dict]] = defaultdict(list)
    for name in names:
        path = data_dir / pattern.format(name=name)
        cursor = duckdb.connect()
        rows = cursor.execute(
            "SELECT * FROM read_parquet(?)", [str(path)],
        ).fetchall()
        columns = [column[0] for column in cursor.description]
        cursor.close()
        grouped[name] = [dict(zip(columns, row)) for row in rows]
    return grouped

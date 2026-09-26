"""Run ToolRet / SkillRet retrieval benchmarks against the agent_retrieval kernel.

Caller-side experiment: nothing in ``agent_retrieval`` changes. Two arms per run:

  bm25    — ``rank_candidates`` without a vector path (kernel default)
  fusion  — BM25 + vector RRF fusion through an OpenAI-compatible embedder

Embedder configuration comes from environment variables (the package never reads
the environment — host discipline; this script IS the host). They may live in a
``.env`` file at the repository root (gitignored; parsed via python-dotenv when
installed) or be exported in the shell — existing environment variables WIN over
``.env``, so one-off overrides stay possible:

  AGENT_RETRIEVAL_EMBEDDING_API_KEY      required for the fusion arm
  AGENT_RETRIEVAL_EMBEDDING_BASE_URL     default https://open.bigmodel.cn/api/paas
  AGENT_RETRIEVAL_EMBEDDING_MODEL        default embedding-3
  AGENT_RETRIEVAL_EMBEDDING_DIMENSIONS   default 1024

Without a key the fusion arm is skipped and the report says so explicitly — the
BM25 arm still runs, because both benchmarks evaluate against the full candidate
pool and the baseline arm needs no network at all. A configured fusion arm is
preflighted with one probe embedding: an unreachable endpoint or rejected key
fails the run instead of silently reporting BM25 numbers under the fusion label.

Protocol notes: both benchmarks use binary relevance (``relevance == 1`` only),
so the reported NDCG is the binary-gain form. Rankings are the kernel's FULL
output; truncation at k happens in the metrics layer. ``--limit`` samples cases
(prefix for SkillRet's file order, round-robin across ToolRet's 35 source tasks)
and never touches the candidate pool — recall against a shrunken pool would
flatter every arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

from agent_retrieval import (
    Embedder,
    EmbeddingConfig,
    QueryEmbeddingError,
    build_embedder,
    rank_candidates,
    vector_available,
)

from experiments.benchmarks import skillret, toolret
from experiments.benchmarks.dataset import BenchmarkDataset, Case
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.metrics import aggregate, score_case

_DEFAULT_DATA_DIR = Path(__file__).with_name("data")
_DEFAULT_CACHE_PATH = Path(__file__).with_name(".cache").joinpath("vectors.sqlite3")
#: ``experiments/benchmarks/run_benchmark.py`` → repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file() -> None:
    """Load ``<repo root>/.env`` without overriding real environment variables.

    python-dotenv is an optional experiment dependency: absent it degrades to
    shell-exported variables only, with a hint instead of a crash.
    """
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 — optional experiment dependency
    except ImportError:
        print(
            f"[env] {env_path.name} found but python-dotenv is not installed — "
            "install it (pip install python-dotenv) or export the variables directly.",
            file=sys.stderr,
        )
        return
    load_dotenv(dotenv_path=env_path, override=False)


@dataclass(frozen=True)
class ArmResult:
    """Aggregate outcome of one arm: metric means plus run shape."""

    metrics: dict[str, float]
    cases: int
    elapsed_seconds: float


def load_benchmark(name: str, data_dir: Path, limit: int | None) -> BenchmarkDataset:
    """Download (once) and load the named benchmark; sampling applies to cases only."""
    if name == "skillret":
        skillret.download(data_dir)
        dataset = skillret.load(data_dir)
    elif name == "toolret":
        toolret.download(data_dir)
        dataset = toolret.load(data_dir, limit=limit)
    else:
        raise ValueError(f"unknown benchmark: {name}")
    if name == "skillret" and limit is not None:
        dataset = BenchmarkDataset(
            name=dataset.name,
            revision=dataset.revision,
            corpus_text=dataset.corpus_text,
            resources=dataset.resources,
            cases=dataset.cases[:limit],
        )
    return dataset


def embedder_from_env() -> EmbeddingConfig:
    """Host-side config assembly; empty api_key is the documented no-vector form."""
    raw_base = os.environ.get(
        "AGENT_RETRIEVAL_EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas"
    )
    return EmbeddingConfig(
        kind="api",
        provider="zhipu",
        model_name=os.environ.get("AGENT_RETRIEVAL_EMBEDDING_MODEL", "embedding-3"),
        # ApiEmbedder 拼接 "{base}/embeddings"：宽容处理填了完整端点的常见写法。
        base_url=raw_base.removesuffix("/embeddings"),
        api_key=os.environ.get("AGENT_RETRIEVAL_EMBEDDING_API_KEY", ""),
        # 0 = 信任端点（ApiEmbedder 不向端点传 dimensions，此值仅本地校验；
        # 预检后按实际维度收紧，见 main）。
        dimensions=int(os.environ.get("AGENT_RETRIEVAL_EMBEDDING_DIMENSIONS", "0")),
        # RTT 主导的语料嵌入：大 batch 摊薄每条延迟（两个基准的文档都短，无超限风险）。
        batch_size=64,
    )


def case_query_text(case: Case, query_mode: str) -> str:
    """Query text fed to the kernel, per the official ToolRet ``is_inst`` protocol:
    the task-aware instruction is prepended to the raw user query."""
    if query_mode == "instruction":
        if not case.instruction:
            raise ValueError(f"case {case.id} has no instruction for --query-mode instruction")
        return f"Instruct: {case.instruction}\nQuery: {case.query}"
    return case.query


def run_arm(
    dataset: BenchmarkDataset,
    k_values: tuple[int, ...],
    *,
    embedder: Embedder | None = None,
    label: str,
    query_mode: str = "query",
) -> ArmResult:
    """Rank every case once and aggregate per-case metrics; full-recall rankings."""
    rows: list[dict[str, float]] = []
    started = perf_counter()
    for index, case in enumerate(dataset.cases):
        hits = rank_candidates(
            dataset.resources,
            case_query_text(case, query_mode),
            corpus_text=dataset.corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
            vector=embedder,
        )
        ranked = [hit.item.id for hit in hits]
        rows.append(score_case(ranked, case.golds, k_values))
        if (index + 1) % 100 == 0:
            print(f"[{label}] {index + 1}/{len(dataset.cases)} cases ranked", file=sys.stderr)
    return ArmResult(
        metrics=aggregate(rows, k_values),
        cases=len(rows),
        elapsed_seconds=round(perf_counter() - started, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bench", choices=("skillret", "toolret"), required=True)
    parser.add_argument("--arm", choices=("bm25", "fusion", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap evaluated cases (sampled, pool untouched); default all")
    parser.add_argument("--k", type=int, nargs="+", default=(5, 10),
                        help="k values for recall/completeness/ndcg (default: 5 10)")
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query",
                        help="instruction mode prepends the task-aware instruction to each "
                             "query (ToolRet official main protocol, is_inst=True)")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
                        help="dataset download directory (gitignored)")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the persistent vector cache (fusion arm re-embeds every call)")
    parser.add_argument("--output", type=Path, help="write the full JSON report here")
    args = parser.parse_args()
    k_values = tuple(sorted({k for k in args.k if k > 0})) or (5, 10)
    if args.query_mode == "instruction" and args.bench != "toolret":
        print(f"Benchmark error: {args.bench} ships no instructions — "
              "--query-mode instruction is ToolRet-only", file=sys.stderr)
        return 2
    load_env_file()

    try:
        dataset = load_benchmark(args.bench, args.data_dir, args.limit)
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2
    print(
        f"[bench] {dataset.name} (revision {dataset.revision}): "
        f"{len(dataset.resources)} resources, {len(dataset.cases)} cases",
        file=sys.stderr,
    )

    arms: dict[str, ArmResult] = {}
    fusion_config: EmbeddingConfig | None = None
    cached: CachedEmbedder | None = None
    if args.arm in ("fusion", "both"):
        config = embedder_from_env()
        if vector_available(config) != "api":
            print(
                "[fusion] AGENT_RETRIEVAL_EMBEDDING_API_KEY not set — fusion arm skipped, "
                "BM25 arm still runs. See --help for the environment variables.",
                file=sys.stderr,
            )
            if args.arm == "fusion":
                return 2
        else:
            fusion_config = config
            embedder = build_embedder(config)
            assert embedder is not None  # vector_available said "api"
            # 实验侧 fail fast：端点不可达/密钥被拒时立刻退出，而不是让内核的
            # 降级语义把 fusion 臂静默折算成一份 BM25 数字。预检向量同时用于
            # 收紧维度校验（端点自报 → 实际值），缓存身份随之对齐真实维度。
            try:
                probe = embedder.embed_query("benchmark preflight")
            except QueryEmbeddingError as exc:
                print(f"[fusion] preflight failed — fix the embedder config: {exc}", file=sys.stderr)
                return 2
            if config.dimensions != len(probe):
                print(
                    f"[fusion] endpoint returned {len(probe)}-dim vectors — "
                    "dimensions pinned accordingly",
                    file=sys.stderr,
                )
                config = replace(config, dimensions=len(probe))
            if args.no_cache:
                arms["fusion"] = run_arm(dataset, k_values, embedder=embedder, label="fusion",
                                         query_mode=args.query_mode)
            else:
                cached = CachedEmbedder(
                    embedder,
                    db_path=_DEFAULT_CACHE_PATH,
                    identity=f"{config.provider}:{config.model_name}:{config.dimensions}",
                )
                arms["fusion"] = run_arm(dataset, k_values, embedder=cached, label="fusion",
                                         query_mode=args.query_mode)
    if args.arm in ("bm25", "both"):
        arms["bm25"] = run_arm(dataset, k_values, embedder=None, label="bm25",
                               query_mode=args.query_mode)

    result: dict[str, object] = {
        "dataset": {
            "name": dataset.name,
            "revision": dataset.revision,
            "resources": len(dataset.resources),
            "cases": len(dataset.cases),
        },
        "configuration": {
            "k_values": list(k_values),
            "query_mode": args.query_mode,
            "embedder": fusion_config.identity() if fusion_config else None,
            "vector_cache": bool(cached),
            "vectors_embedded_this_run": cached.embedded_texts if cached else None,
        },
        "arms": {name: asdict(arm) for name, arm in arms.items()},
    }
    if cached is not None:
        cached.close()

    _print_summary(
        dataset_name=dataset.name,
        revision=dataset.revision,
        resources=len(dataset.resources),
        cases=len(dataset.cases),
        embedder_identity=fusion_config.identity() if fusion_config else None,
        arms=arms,
    )
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"[bench] full report written to {args.output}", file=sys.stderr)
    return 0


def _print_summary(
    *,
    dataset_name: str,
    revision: str,
    resources: int,
    cases: int,
    embedder_identity: str | None,
    arms: dict[str, ArmResult],
) -> None:
    print(f"\n=== {dataset_name} @ {revision}: {resources} resources × {cases} cases ===")
    print(f"embedder: {embedder_identity or 'none (BM25-only run)'}")
    for arm_name, arm in arms.items():
        cells = "  ".join(f"{name}={value:.4f}" for name, value in arm.metrics.items())
        print(f"[{arm_name}] ({arm.cases} cases, {arm.elapsed_seconds}s)  {cells}")


if __name__ == "__main__":
    raise SystemExit(main())

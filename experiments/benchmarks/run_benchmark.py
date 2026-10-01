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
  AGENT_RETRIEVAL_RERANK_API_KEY         required for the rerank arm
  AGENT_RETRIEVAL_RERANK_BASE_URL        default https://api.siliconflow.cn/v1
  AGENT_RETRIEVAL_RERANK_MODEL           default BAAI/bge-reranker-v2-m3

Without a key the fusion/rerank arm is skipped and the report says so explicitly — the
BM25 arm still runs, because both benchmarks evaluate against the full candidate
pool and the baseline arm needs no network at all. A configured fusion arm is
preflighted with one probe embedding: an unreachable endpoint or rejected key
fails the run instead of silently reporting BM25 numbers under the fusion label.
The rerank arm (``--arm rerank``) re-ranks the fusion top-N
(``--rerank-candidates``, default 100 — the official ToolRet second stage) through
a Cohere-style ``/rerank`` endpoint (SiliconFlow / Jina / Cohere compatible).

Protocol notes: both benchmarks use binary relevance (``relevance == 1`` only),
so the reported NDCG is the binary-gain form. Rankings are the kernel's FULL
output; truncation at k happens in the metrics layer. ``--limit`` samples cases
(prefix for SkillRet's file order, round-robin across ToolRet's 35 source tasks)
and never touches the candidate pool — recall against a shrunken pool would
flatter every arm.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections.abc import Callable
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
from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource
from experiments.benchmarks.docgen import docgen_corpus_text, load_docgen
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.metrics import aggregate, score_case
from experiments.benchmarks.multipath import FieldRetriever
from experiments.benchmarks.reranker import (
    RERANK_DOC_FLOOR_CHARS,
    RERANK_PAIR_BUDGET_CHARS,
    ApiReranker,
    RerankError,
)

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
    #: rerank 臂：重试后仍失败、退化为 fusion 原序的 case 数（0 = 全程成功）。
    degraded_cases: int = 0


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


def reranker_from_env() -> ApiReranker:
    """Host-side rerank config assembly; empty api_key reads as no-rerank form."""
    raw_base = os.environ.get("AGENT_RETRIEVAL_RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
    return ApiReranker(
        # 客户端拼接 "{base}/rerank"：宽容处理填了完整端点的常见写法。
        base_url=raw_base.removesuffix("/rerank"),
        api_key=os.environ.get("AGENT_RETRIEVAL_RERANK_API_KEY", ""),
        model_name=os.environ.get("AGENT_RETRIEVAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
    )


def run_rerank_arm(
    dataset: BenchmarkDataset,
    k_values: tuple[int, ...],
    *,
    embedder: Embedder | None,
    reranker: ApiReranker,
    candidates: int,
    label: str,
    query_mode: str = "query",
) -> ArmResult:
    """Two-stage arm: fusion retrieval → cross-encoder re-rank of the top-N.

    Only the fusion top-N window can be re-ordered; candidates beyond it keep
    their fusion order — a reranker cannot rescue what retrieval never surfaced,
    and that boundary is exactly what this arm is designed to measure.
    """
    rows: list[dict[str, float]] = []
    started = perf_counter()
    degraded = consecutive_failures = 0
    for index, case in enumerate(dataset.cases):
        query = case_query_text(case, query_mode)
        hits = rank_candidates(
            dataset.resources,
            query,
            corpus_text=dataset.corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
            vector=embedder,
        )
        window = hits[:candidates]
        # 精排输入按 query+document 总长预算截断（网关 code 1214 拒收超长对，
        # 且多后端限制不一）；精排本就不消费全文。
        doc_cap = max(RERANK_DOC_FLOOR_CHARS, RERANK_PAIR_BUDGET_CHARS - len(query))
        try:
            scores = reranker.rerank(
                query,
                [dataset.corpus_text(hit.item)[:doc_cap] for hit in window],
            )
        except RerankError as exc:
            # 重试后仍失败：该 case 退化为 fusion 原序（rerank 是增强不是依赖），
            # 计入 degraded；连续失败到阈值则中止——那是端点彻底不可用，不是抖动。
            consecutive_failures += 1
            degraded += 1
            print(f"[{label}] case {case.id} rerank failed, keeping fusion order: {exc}",
                  file=sys.stderr)
            if consecutive_failures >= 5:
                print(f"[{label}] 5 consecutive failures — rerank endpoint is down, aborting",
                      file=sys.stderr)
                raise
            # 连续失败说明大概率在限流窗口内：冷却后再进下一个 case，
            # 否则下个 case 的重试会立刻撞回同一面墙。
            time.sleep(min(10.0 * consecutive_failures, 30.0))
            scores = None
        else:
            consecutive_failures = 0
            if scores and max(scores) - min(scores) < 1e-9:
                # 常量分 = reranker 没有产生任何区分度（实测 paratera 的
                # GLM-Rerank 恒返 1.0）。按重排会塌缩成 id 字典序、把有效
                # 排序打乱——保留 fusion 序并如实计入退化。
                scores = None
                degraded += 1
                print(f"[{label}] case {case.id}: rerank scores are constant — "
                      "endpoint gave no discrimination, keeping fusion order", file=sys.stderr)
        if scores is None:
            ranked = [hit.item.id for hit in hits]
        else:
            reranked = sorted(
                zip(window, scores), key=lambda pair: (-pair[1], pair[0].item.id),
            )
            ranked = [hit.item.id for hit, _ in reranked]
            ranked += [hit.item.id for hit in hits[candidates:]]
        rows.append(score_case(ranked, case.golds, k_values))
        if (index + 1) % 50 == 0:
            print(f"[{label}] {index + 1}/{len(dataset.cases)} cases ranked", file=sys.stderr)
    return ArmResult(
        metrics=aggregate(rows, k_values),
        cases=len(rows),
        elapsed_seconds=round(perf_counter() - started, 1),
        degraded_cases=degraded,
    )


def field_paths(dataset: BenchmarkDataset) -> list[tuple[str, Callable[[Resource], str]]]:
    """Per-field projections of the declared face, one BM25 path each.

    SkillRet's paper reports field-separate beats concatenated hybrid retrieval;
    ToolRet's documentation is heterogeneous JSON, so only the parsed tool name
    gets its own path and everything else stays on the verbatim path.
    """
    if dataset.name == "skillret":
        return [
            ("name", lambda resource: resource.name),
            ("description", lambda resource: resource.description),
            ("tags", lambda resource: resource.tags),
        ]
    return [
        ("name", lambda resource: resource.name),
        ("documentation", lambda resource: resource.description),
    ]


def run_arm(
    dataset: BenchmarkDataset,
    k_values: tuple[int, ...],
    *,
    embedder: Embedder | None = None,
    label: str,
    query_mode: str = "query",
    retriever: str = "single",
    field_retriever: FieldRetriever | None = None,
) -> ArmResult:
    """Rank every case once and aggregate per-case metrics; full-recall rankings."""
    rows: list[dict[str, float]] = []
    started = perf_counter()
    for index, case in enumerate(dataset.cases):
        query = case_query_text(case, query_mode)
        if retriever == "fields":
            if field_retriever is None:
                raise ValueError("fields retriever needs a prebuilt FieldRetriever")
            ranked = field_retriever.rank(query, embedder=embedder,
                                          corpus_text=dataset.corpus_text)
        else:
            hits = rank_candidates(
                dataset.resources,
                query,
                corpus_text=dataset.corpus_text,
                item_id=lambda resource: resource.id,
                item_name=lambda resource: resource.name,
                vector=embedder,
            )
            ranked = [hit.item.id for hit in hits]
        rows.append(score_case(ranked, case.golds, k_values))
        if (index + 1) % 25 == 0:
            # 全池无状态检索每次调用重建 44k 文档索引，临时对象量大且分代
            # GC 回收不及时——周期性强制回收压住内存峰值（OOM 杀任务的教训）。
            gc.collect()
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
    parser.add_argument("--arm", choices=("bm25", "fusion", "rerank", "both"), default="both")
    parser.add_argument("--rerank-candidates", type=int, default=100,
                        help="rerank arm: fusion top-N window re-ranked by the cross-encoder "
                             "(default 100, matching the official ToolRet second stage)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap evaluated cases (sampled, pool untouched); default all")
    parser.add_argument("--k", type=int, nargs="+", default=(5, 10),
                        help="k values for recall/completeness/ndcg (default: 5 10)")
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query",
                        help="instruction mode prepends the task-aware instruction to each "
                             "query (ToolRet official main protocol, is_inst=True)")
    parser.add_argument("--retriever", choices=("single", "fields"), default="single",
                        help="fields mode splits the declared face into per-field BM25 "
                             "paths merged by RRF (vector path unchanged)")
    parser.add_argument("--corpus", choices=("plain", "docgen"), default="plain",
                        help="docgen mode appends doc2query generated queries to each "
                             "resource's retrieval corpus (needs experiments/benchmarks/"
                             "data/docgen-{bench}.jsonl from the docgen CLI)")
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
    reranker: ApiReranker | None = None
    reranker_model: str | None = None
    field_retriever = (
        FieldRetriever(dataset.resources, field_paths(dataset))
        if args.retriever == "fields" else None
    )
    if args.corpus == "docgen":
        expansions = load_docgen(_DEFAULT_DATA_DIR, args.bench)
        dataset = BenchmarkDataset(
            name=dataset.name,
            revision=f"{dataset.revision}+docgen",
            corpus_text=docgen_corpus_text(dataset.corpus_text, expansions),
            resources=dataset.resources,
            cases=dataset.cases,
        )
        covered = sum(1 for resource in dataset.resources
                      if resource.id in expansions)
        print(f"[corpus] docgen: {covered}/{len(dataset.resources)} resources expanded",
              file=sys.stderr)
    if args.arm == "rerank":
        config = embedder_from_env()
        if vector_available(config) != "api":
            print("[rerank] the rerank arm retrieves via fusion first — set "
                  "AGENT_RETRIEVAL_EMBEDDING_API_KEY too.", file=sys.stderr)
            return 2
        fusion_config = config
        embedder = build_embedder(config)
        assert embedder is not None  # vector_available said "api"
        cached = CachedEmbedder(
            embedder,
            db_path=_DEFAULT_CACHE_PATH,
            identity=f"{config.provider}:{config.model_name}:{config.dimensions}",
        )
        reranker = reranker_from_env()
        reranker_model = os.environ.get("AGENT_RETRIEVAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        try:
            reranker.rerank("benchmark preflight", ["alpha document", "beta document"])
        except RerankError as exc:
            print(f"[rerank] preflight failed — fix the rerank config: {exc}", file=sys.stderr)
            return 2
        arms["rerank"] = run_rerank_arm(
            dataset, k_values, embedder=cached, reranker=reranker,
            candidates=args.rerank_candidates, label="rerank", query_mode=args.query_mode,
        )
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
                                         query_mode=args.query_mode, retriever=args.retriever,
                                         field_retriever=field_retriever)
            else:
                cached = CachedEmbedder(
                    embedder,
                    db_path=_DEFAULT_CACHE_PATH,
                    identity=f"{config.provider}:{config.model_name}:{config.dimensions}",
                )
                arms["fusion"] = run_arm(dataset, k_values, embedder=cached, label="fusion",
                                         query_mode=args.query_mode, retriever=args.retriever,
                                         field_retriever=field_retriever)
    if args.arm in ("bm25", "both"):
        arms["bm25"] = run_arm(dataset, k_values, embedder=None, label="bm25",
                               query_mode=args.query_mode, retriever=args.retriever,
                               field_retriever=field_retriever)

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
            "retriever": args.retriever,
            "corpus": args.corpus,
            "embedder": fusion_config.identity() if fusion_config else None,
            "reranker": reranker_model,
            "rerank_candidates": args.rerank_candidates if args.arm == "rerank" else None,
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
    if arms and "rerank" in arms:
        print(f"reranker: {os.environ.get('AGENT_RETRIEVAL_RERANK_MODEL', 'BAAI/bge-reranker-v2-m3')} "
              f"(top-{os.environ.get('AGENT_RETRIEVAL_RERANK_CANDIDATES', '100')} window)")
    for arm_name, arm in arms.items():
        cells = "  ".join(f"{name}={value:.4f}" for name, value in arm.metrics.items())
        print(f"[{arm_name}] ({arm.cases} cases, {arm.elapsed_seconds}s)  {cells}")


if __name__ == "__main__":
    raise SystemExit(main())

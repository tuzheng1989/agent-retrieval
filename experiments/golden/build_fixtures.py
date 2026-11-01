"""金版本 fixture 导出器：采样池 + cases + docgen 扩展 + 预计算向量 → gzip JSON。

在有 ``data/`` 与向量缓存（``experiments/benchmarks/.cache/vectors.sqlite3``）的机器上
运行一次的再生成工具（CI 不跑它）::

    python -m experiments.golden.build_fixtures --bench skillret
    python -m experiments.golden.build_fixtures --bench toolret

产物 ``tests/fixtures/golden/{bench}.json.gz``：池 = cases 的全部 gold + 固定种子随机
干扰补足（500 资源）；向量含 plain / docgen 两套语料向量与查询向量（ToolRet 按官方
instruction 形态），全部取自既有缓存（会话零嵌入 = 全命中，说明实验跑过、缓存覆盖
完整）；对抗负例是三条 trivia 查询——知识问答既非 SkillRet 技能域也非 ToolRet 工具
域，"不绑定"语义可辩护（jev_cases.json 的负例针对 14 资源合成池，移植到真实池会被
池内合理资源误杀）。gate.py 消费 fixture 离线跑四臂（bm25/fusion × plain/docgen）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from dataclasses import replace
from pathlib import Path

from experiments.benchmarks.dataset import BenchmarkDataset, Resource
from experiments.benchmarks.docgen import docgen_corpus_text, load_docgen
from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.jev_rerank import load_sampled_benchmark
from experiments.benchmarks.run_benchmark import (
    _DEFAULT_CACHE_PATH,
    _DEFAULT_DATA_DIR,
    case_query_text,
    load_env_file,
)

#: 池规模与采样种子：种子固定 → 同一 data/ 修订下 fixture 逐位可复现（内核确定性）。
#: gold 截留：SkillRet 单 case 的 gold 集可达百余条，全量入池会让 fixture 膨胀到
#: 数百 MB；每 case 确定性截留前 ``GOLDS_PER_CASE`` 条，case.golds 同步截断——指标
#: 语义是「截断 gold 集上的质量」，作为同一 fixture 上的相对红线自洽。
POOL_SIZE = 500
CASE_COUNT = 50
GOLDS_PER_CASE = 4
SAMPLE_SEED = 20261101

#: 对抗负例：**无任务语义且无声明面高频词**的文本。trivia 不行（"Fibonacci" 对数学 API
#: 是真相关，relevance 0.27–0.52）；元对话也不行（"name" 是工具名/参数名高频词，实测
#: relevance 0.53–0.77 过线误绑）。问候 / 无义串 / 诗意句是实测稳的负例形态。
ADVERSARIAL_QUERIES = (
    "Hello, good morning",
    "purple elephant dancing tango",
    "The quiet river flows under the old stone bridge at dawn",
)
ADVERSARIAL_RELEVANCE_CEILING = 0.2

#: ToolRet 按官方主协议（is_inst=True）喂 instruction 形态查询。
QUERY_MODE = {"toolret": "instruction", "skillret": "query"}

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "golden"


def kept_golds(golds: frozenset[str]) -> list[str]:
    """确定性截留前 ``GOLDS_PER_CASE`` 条 gold（排序去随机性）。"""
    return sorted(golds)[:GOLDS_PER_CASE]


def sample_pool(dataset: BenchmarkDataset, cases) -> list[Resource]:
    """采样 cases 截留后的 gold ∪ 固定种子随机干扰，补足到 ``POOL_SIZE``。"""
    gold_ids: set[str] = set()
    for case in cases:
        gold_ids.update(kept_golds(case.golds))
    rng = random.Random(SAMPLE_SEED)
    others = [resource.id for resource in dataset.resources if resource.id not in gold_ids]
    rng.shuffle(others)
    pool_ids = sorted(gold_ids.union(others[: max(POOL_SIZE - len(gold_ids), 0)]))
    by_id = {resource.id: resource for resource in dataset.resources}
    return [by_id[resource_id] for resource_id in pool_ids]


def build_fixture(bench: str, *, cache_path: Path) -> dict:
    dataset, cases = load_sampled_benchmark(bench, _DEFAULT_DATA_DIR, CASE_COUNT, corpus="plain")
    resources = sample_pool(dataset, cases)
    expansions = load_docgen(_DEFAULT_DATA_DIR, bench)

    def docgen_text(resource: Resource) -> str:
        return docgen_corpus_text(dataset.corpus_text, expansions)(resource)

    # 向量查缓存：历史 fusion/docgen 实验已嵌全池，此处全命中（新嵌即 fixture 不可离线复现）。
    from agent_retrieval import build_embedder, vector_available

    from experiments.benchmarks.run_benchmark import embedder_from_env

    config = embedder_from_env()
    if vector_available(config) != "api":
        raise RuntimeError("AGENT_RETRIEVAL_EMBEDDING_API_KEY required to pin the embedder identity")
    live = build_embedder(config)
    assert live is not None
    probe = live.embed_query("benchmark preflight")
    config = replace(config, dimensions=len(probe))
    cache = CachedEmbedder(
        live, db_path=cache_path,
        identity="|".join(config.identity()),  # R5-P1：身份串含 pooling/前缀 flags
    )
    try:
        vectors: dict[str, list[float]] = {}
        for resource in resources:
            # 6 位小数：归一化向量的截断对排序影响 ~1e-6，远小于门控语义，JSON 体积减半。
            vectors[f"corpus_plain:{resource.id}"] = [round(v, 6) for v in cache.embed_query(dataset.corpus_text(resource))]
            vectors[f"corpus_docgen:{resource.id}"] = [round(v, 6) for v in cache.embed_query(docgen_text(resource))]
        for case in cases:
            query = case_query_text(case, QUERY_MODE[bench])
            vectors[f"query:{case.id}"] = [round(v, 6) for v in cache.embed_query(query)]
    finally:
        cache.close()
    if cache.embedded_texts:
        raise RuntimeError(
            f"{cache.embedded_texts} texts needed live embedding — the vector cache does not "
            "cover this fixture; run the fusion and docgen arms once before exporting",
        )

    return {
        "name": f"{bench}-golden-v1",
        "revision": dataset.revision,
        "embedder_identity": config.identity(),
        "query_mode": QUERY_MODE[bench],
        "resources": [
            {"id": r.id, "name": r.name, "description": r.description, "tags": r.tags}
            for r in resources
        ],
        "cases": [
            {"id": c.id, "query": c.query, "instruction": c.instruction,
             "golds": kept_golds(c.golds)}
            for c in cases
        ],
        "docgen": {resource_id: queries for resource_id, queries in expansions.items()
                   if resource_id in {r.id for r in resources}},
        "vectors": vectors,
        "adversarial": {
            "queries": list(ADVERSARIAL_QUERIES),
            "relevance_ceiling": ADVERSARIAL_RELEVANCE_CEILING,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=("skillret", "toolret"), required=True)
    parser.add_argument("--cache-path", type=Path, default=_DEFAULT_CACHE_PATH)
    args = parser.parse_args()
    load_env_file()
    fixture = build_fixture(args.bench, cache_path=args.cache_path)
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    output = _FIXTURE_DIR / f"{args.bench}.json.gz"
    payload = json.dumps(fixture, ensure_ascii=False).encode("utf-8")
    with gzip.open(output, "wb", compresslevel=9) as handle:
        handle.write(payload)
    print(
        f"[golden] {fixture['name']}: {len(fixture['resources'])} resources, "
        f"{len(fixture['cases'])} cases, {len(fixture['vectors'])} vectors -> "
        f"{output} ({output.stat().st_size / 1e6:.1f} MB gzip)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

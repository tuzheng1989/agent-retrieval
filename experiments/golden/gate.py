"""金版本零退化门控：fixture 四臂逐指标 >= golden.json（CI 常驻的检索质量红线）。

四臂 = 检索路径 {bm25, fusion} × 语料 {plain, docgen}——覆盖每基准当前最优臂与关键
基线臂（SkillRet 最优 docgen+BM25、ToolRet 最优 docgen+fusion，其余两臂为基线锚）。
fusion 臂的向量来自 fixture 预计算值（FixtureEmbedder 替身），全程零外部服务、零
下载、零模型调用。对抗段：trivia 查询对发现面（``ResourceBM25Index`` +
``clearly_related``）不得误绑。

红线语义是 **零退化**（``current >= golden - ε``，改进放行）；改进要沉淀为新金版本，
显式重跑 ``python -m experiments.golden.gate --update`` 并随改动一起提交——金版本升级
因此是显式行为，不是静默漂移。中文格（zh-plot-v1）在自建评测集构建立项后接线，
golden.json 中占位标注。
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from agent_retrieval import (
    CandidateHit,
    Embedder,
    ResourceBM25Index,
    clearly_related,
    rank_candidates,
)

from experiments.benchmarks.dataset import Case, Resource
from experiments.benchmarks.docgen import docgen_corpus_text
from experiments.benchmarks.metrics import aggregate, score_case

K_VALUES = (5, 10)
#: 逐指标零退化比对的浮点容差：kernel 确定性下同机逐位一致，跨平台 last-bit 漂移护住。
EPSILON = 1e-9
_BENCHES = ("skillret", "toolret")
#: 中文格占位：zh-plot-v1（自建评测集方案已冻结、构建未立项）接线前的显式标注。
_PENDING = {"zh_plot": "待 zh-plot-v1 构建立项后接线（03 票冻结方案）"}

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "golden"
_GOLDEN_PATH = _FIXTURE_DIR / "golden.json"


class FixtureEmbedder(Embedder):
    """fixture 预计算向量的替身：文本查表，缺失即 fail-fast（fixture 不完整是导出器 bug）。"""

    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self._mapping = mapping

    def embed_corpus(self, texts) -> list[tuple[float, ...]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._mapping[text]


def load_fixture(bench: str) -> dict:
    path = _FIXTURE_DIR / f"{bench}.json.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"golden fixture missing: {path} — run experiments.golden.build_fixtures --bench {bench}",
        )
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def load_golden() -> dict:
    if not _GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"golden baseline missing: {_GOLDEN_PATH} — run experiments.golden.gate --update",
        )
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


def _plain_text(resource: Resource) -> str:
    return " ".join(
        part for part in (resource.id, resource.name, resource.description, resource.tags) if part
    )


def _query_text(fixture: dict, case: Case) -> str:
    if fixture.get("query_mode") == "instruction" and case.instruction:
        return f"Instruct: {case.instruction}\nQuery: {case.query}"
    return case.query


def _fixture_dataset(fixture: dict, corpus: str):
    """fixture → (resources, cases, corpus_text 投影, Embedder 用的 文本→向量 映射)。"""
    resources = [Resource(**row) for row in fixture["resources"]]
    cases = [
        Case(id=row["id"], query=row["query"], instruction=row.get("instruction", ""),
             golds=frozenset(row["golds"]))
        for row in fixture["cases"]
    ]
    expansions = {rid: list(qs) for rid, qs in fixture.get("docgen", {}).items()}
    corpus_text = _plain_text if corpus == "plain" else docgen_corpus_text(_plain_text, expansions)
    vectors = fixture["vectors"]
    prefix = "corpus_plain" if corpus == "plain" else "corpus_docgen"
    # Embedder 按文本查表：向量以「投影后语料文本 / 查询文本」为键展开。
    mapping = {
        corpus_text(resource): tuple(vectors[f"{prefix}:{resource.id}"])
        for resource in resources
    }
    mapping.update({_query_text(fixture, case): tuple(vectors[f"query:{case.id}"])
                    for case in cases})
    return resources, cases, corpus_text, mapping


def run_arm(fixture: dict, corpus: str, vector: bool) -> dict[str, float]:
    resources, cases, corpus_text, mapping = _fixture_dataset(fixture, corpus)
    embedder = FixtureEmbedder(mapping) if vector else None
    rows = []
    for case in cases:
        hits: list[CandidateHit[Resource]] = rank_candidates(
            resources,
            case.query if fixture.get("query_mode") != "instruction"
            else f"Instruct: {case.instruction}\nQuery: {case.query}",
            corpus_text=corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
            vector=embedder,
        )
        rows.append(score_case([hit.item.id for hit in hits], case.golds, K_VALUES))
    return aggregate(rows, K_VALUES)


def run_adversarial(fixture: dict) -> list[dict]:
    """trivia 查询对发现面不得误绑：top-1 不得 exact、不得过 clearly_related 线。"""
    resources, _cases, corpus_text, _mapping = _fixture_dataset(fixture, "plain")
    index = ResourceBM25Index(
        [(resource, corpus_text(resource)) for resource in resources],
        declared=lambda resource: f"{resource.name} {resource.description}",
    )
    ceiling = fixture["adversarial"]["relevance_ceiling"]
    results = []
    for query in fixture["adversarial"]["queries"]:
        hits = index.search(query, limit=1)
        bound = bool(hits) and (
            hits[0].exact
            # 内核签名按 ResourceHit[object] 声明，泛型不变性对具体资源类型报误——运行时按结构匹配。
            or clearly_related(hits[0], minimum_relevance=ceiling)  # type: ignore[arg-type]
        )
        results.append({
            "query": query,
            "bound": bound,
            "top_relevance": round(hits[0].relevance, 4) if hits else None,
        })
    return results


def run_gate() -> dict:
    """跑两基准四臂 + 对抗段，返回与 golden.json 同构的当前读数。"""
    report: dict = {"pending": _PENDING}
    for bench in _BENCHES:
        fixture = load_fixture(bench)
        report[bench] = {
            "fixture": fixture["name"],
            "revision": fixture["revision"],
            "embedder_identity": fixture["embedder_identity"],
            "arms": {
                f"{retriever}+{corpus}": run_arm(fixture, corpus, retriever == "fusion")
                for retriever in ("bm25", "fusion")
                for corpus in ("plain", "docgen")
            },
            "adversarial": run_adversarial(fixture),
        }
    return report


def violations(current: dict, golden: dict) -> list[str]:
    """逐基准逐臂逐指标零退化违规 + 对抗误绑 + fixture 修订漂移。"""
    problems: list[str] = []
    for bench in _BENCHES:
        if bench not in golden:
            problems.append(f"{bench}: golden.json 无该基准条目")
            continue
        if (current[bench]["fixture"], current[bench]["revision"]) != (
            golden[bench]["fixture"], golden[bench]["revision"],
        ):
            problems.append(
                f"{bench}: fixture 修订漂移 {golden[bench]['revision']} → "
                f"{current[bench]['revision']} — 重跑 --update 固化新金版本",
            )
        for arm, metrics in current[bench]["arms"].items():
            baseline = golden[bench]["arms"].get(arm)
            if baseline is None:
                problems.append(f"{bench}/{arm}: golden.json 无该臂")
                continue
            for metric, value in metrics.items():
                floor = baseline[metric] - EPSILON
                if value < floor:
                    problems.append(
                        f"{bench}/{arm}/{metric}: {value:.4f} < golden {baseline[metric]:.4f}",
                    )
        for row in current[bench]["adversarial"]:
            if row["bound"]:
                problems.append(f"{bench}/adversarial: 误绑 {row['query']!r} "
                                f"(relevance={row['top_relevance']})")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="以当前读数重写 golden.json（改进沉淀为新金版本的显式动作）")
    args = parser.parse_args()
    current = run_gate()
    if args.update:
        payload = {"pending": _PENDING, **{bench: current[bench] for bench in _BENCHES}}
        _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        _GOLDEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        print(f"[golden] golden.json rewritten ({_GOLDEN_PATH})")
        return 0
    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    problems = violations(current, golden)
    for bench in _BENCHES:
        for arm, metrics in current[bench]["arms"].items():
            cells = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[{bench}/{arm}]  {cells}")
    if problems:
        print("\n".join(f"[golden] VIOLATION {p}" for p in problems), file=sys.stderr)
        return 1
    print("[golden] all arms at or above the golden baseline; no adversarial misbind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""领域微调嵌入配对实验（08 票）：三模型 vs GLM-Embedding-3 基线，同语料同采样。

两种运行形态：

``--embed-only``（**用户在 PowerShell 手动跑**——载 sentence-transformers 模型的
CPU 密集长任务，Claude 后台跑会被内存收割）：逐模型加载 → 标定 100 条墙钟 →
嵌入全部语料与查询入 ``CachedEmbedder``（sqlite，identity 按模型区分）→ 释放 →
下一模型。中断可续（已嵌文本跳过）。

默认模式（低内存，向量全走缓存）：跑 fusion 臂逐 case 指标（jev 报告同构，
``ablation_stats pair`` 直接可配），外加票面核心读数 gold-in-window@{5,20,50}。
基线 = GLM-Embedding-3 fusion（既有 jev 报告 baseline 臂；w/o instruction 档用
``--model glm`` 补跑）。

模型契约（01 票 findings §3）：SKILLRET-0.6B / harrier 查询侧带 Instruct 前缀
（harrier 用 ST 自带 default prompt，SKILLRET 用模型卡任务句），last-token
pooling；ToolRet-trained-bge-large 查询侧 bge 惯例前缀，CLS——pooling 全部由
sentence-transformers 的模型自带配置处理，本脚本零 pooling 代码（SDK 侧 ONNX
pooling 协议按 ADR-0001 P0 另行落地，与本实验路径解耦）。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from agent_retrieval import (
    build_embedder,
    rank_candidates,
    vector_available,
)

from experiments.benchmarks.embedder_cache import CachedEmbedder
from experiments.benchmarks.jev_rerank import load_sampled_benchmark
from experiments.benchmarks.metrics import aggregate, score_case
from experiments.benchmarks.run_benchmark import (
    _DEFAULT_CACHE_PATH,
    _DEFAULT_DATA_DIR,
    case_query_text,
    embedder_from_env,
    load_env_file,
)

K_VALUES = (5, 10)
#: 票面核心读数：一阶段 gold 覆盖（any-gold 命中 top-k 的 case 比例）。
WINDOW_KS = (5, 20, 50)

#: 模型注册表：repo、查询侧前缀（文档侧一律裸文本）、ST 默认 prompt 优先。
#: 模型注册表：repo、查询侧前缀（文档侧一律裸文本）、ST 默认 prompt 优先。
#: ``faces`` = 允许嵌入/评测的基准面（01 票 findings：Edge 域外崩塌，禁入 ToolRet 面）。
#: ``feasible=False`` = 2026-11-01 CPU 标定 S1 判死（SKILLRET-0.6B 实测 79,702 ms/doc，
#: 44k 外推 ≈ 40 天；harrier 同族同量级 Qwen3-0.6B decoder，家族推断连带判死），
#: spec 保留供 GPU 环境复用。probe_size：标定探针条数（判死模型只留 10 条备查）。
MODELS: dict[str, dict] = {
    "skillret-edge": {
        # S1 替代臂（01 票 §4 备选）：教师平位（79.21 NDCG@10）、CPU 原生定位。
        # 卡面契约 CLS pooling / max 256 / 裸查询；repo 内 fp16 safetensors 可被
        # transformers 直接加载，ST 侧手工装配 CLS 管线。
        "repo": "ThakiCloud/SKILLRET-Edge-109M-int4",
        "prefix": "",
        "use_default_prompt": False,
        "pipeline": "cls",
        "max_seq_length": 256,
        "faces": ("skillret",),
        "probe_size": 100,
    },
    "toolret-bge": {
        "repo": "mangopy/ToolRet-trained-bge-large-en-v1.5",
        "prefix": "Represent this sentence for searching relevant passages: ",
        "use_default_prompt": False,
        "faces": ("toolret",),
        "probe_size": 100,
    },
    "skillret": {
        "repo": "ThakiCloud/SKILLRET-Embedding-0.6B",
        "prefix": "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ",
        "use_default_prompt": False,
        "faces": ("toolret", "skillret"),
        "probe_size": 10,
        "feasible": False,
    },
    "harrier": {
        # ST 自带 default prompt（web_search_query 任务句）优先；此值仅作无 prompt 时回退。
        "repo": "microsoft/harrier-oss-v1-0.6b",
        "prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "use_default_prompt": True,
        "faces": ("toolret", "skillret"),
        "probe_size": 10,
        "feasible": False,
    },
}

#: 预注册 S1 熔断（issue #9）按其原文语义实现：「外推 44k > 6h 判不可行」。
#: 限面模型按其 faces 外推（Edge 6k 面：2026-11-01 标定 733 ms/doc × 6,006 ≈ 73 min
#: < 6h，通过）；标定的绝对 ms/doc 照常打印留档。0.6B 判死记录不变。
S1_MAX_FACE_MINUTES = 360


class STEmbedder:
    """sentence-transformers 适配 Embedder 端口：pooling 归模型配置，前缀归本类。"""

    def __init__(self, spec: dict, *, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer, models

        if spec.get("pipeline") == "cls":
            # 卡面只给 transformers 权重、无 ST 配置的模型（Edge）：按卡面契约手工
            # 装配 CLS 管线；encode 侧照常 normalize，max_seq_length 按评测口径。
            word = models.Transformer(spec["repo"], max_seq_length=spec["max_seq_length"])
            pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
            self._model = SentenceTransformer(modules=[word, pooling])
        else:
            # CPU 无原生 fp16/bf16 逐算模拟（卡面警告实测显著拖慢）：强制 fp32
            # 加载，语料级嵌入的诚实形态；identity 不含 dtype——缓存清理代替
            # （换 dtype 时清该 identity 的旧行）。
            self._model = SentenceTransformer(
                spec["repo"], model_kwargs={"dtype": "float32"},
            )
        self._batch = batch_size
        self._prefix = spec["prefix"]
        if spec.get("use_default_prompt") and getattr(self._model, "default_prompt_name", None):
            # ST 自带 prompt 按完整 {prompt} 占位模板使用（卡面任务句的最保真形态）。
            self._prefix = self._model.prompts[self._model.default_prompt_name]
            print(f"[st] using model default prompt: {self._prefix!r}", file=sys.stderr)
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed_corpus(self, texts) -> list[tuple[float, ...]]:
        vectors = self._model.encode(
            list(texts), batch_size=self._batch, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [tuple(float(v) for v in vector) for vector in vectors]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed_corpus([self._prefix + text])[0]


def st_identity(spec: dict, dimensions: int) -> str:
    # R5-P1：身份串与 API 路径统一 ``|`` 拼接形态（信息不变：local-st + repo + dims）；
    # 旧 ``:`` 拼接行随键形状变更一并失效（embed-only 重跑即补齐）。
    return "|".join(("local-st", spec["repo"], str(dimensions)))


class CachedLookupEmbedder(CachedEmbedder):
    """检索模式的只读缓存面：miss 即抛错——嵌入必须在 ``--embed-only`` 会话完成。"""

    def __init__(self, *, db_path: Path, identity: str) -> None:
        super().__init__(inner=None, db_path=db_path, identity=identity)  # type: ignore[arg-type]

    def embed_corpus(self, texts) -> list[tuple[float, ...]]:
        from experiments.benchmarks.embedder_cache import _cache_key

        # embed-only 会话以语料入口（entry="c"）写入全部向量，lookup 侧按同键读取。
        keys = [_cache_key("c", text) for text in texts]
        cached = self._load_cached(keys)
        missing = [key for key in keys if key not in cached]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(keys)} vectors missing from cache — "
                "run --embed-only first (retrieval mode never loads a model)",
            )
        return [cached[key] for key in keys]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed_corpus([text])[0]


def build_lookup_embedder(model_name: str) -> tuple[CachedLookupEmbedder, str]:
    """identity 从缓存 sqlite 的 dim 列直查——检索会话零模型加载、零内存峰值。"""
    import sqlite3

    spec = MODELS[model_name]
    with sqlite3.connect(_DEFAULT_CACHE_PATH) as connection:
        row = connection.execute(
            "SELECT identity, dim FROM vectors WHERE identity LIKE ? LIMIT 1",
            (f"local-st|{spec['repo']}|%",),  # R5-P1：匹配 | 拼接的新 identity_str 形态
        ).fetchone()
    if row is None:
        raise RuntimeError(
            f"no cached vectors for {model_name} — run --embed-only first",
        )
    identity, _dim = row
    return CachedLookupEmbedder(db_path=_DEFAULT_CACHE_PATH, identity=identity), identity


def _bench_configs(faces: tuple[str, ...]):
    """(bench, query_mode, limit)：faces 内的档位（同面双 query 档共享语料缓存）。"""
    for bench, query_mode, limit in (
        ("toolret", "instruction", 100),
        ("toolret", "query", 100),
        ("skillret", "query", 300),
    ):
        if bench in faces:
            yield bench, query_mode, limit


def run_embed_only(model_names: list[str]) -> int:
    """逐模型：标定探针 → S1 熔断检查 → 语料与查询入缓存 → 释放。中断可续。"""
    for name in model_names:
        spec = MODELS[name]
        if not spec.get("feasible", True):
            print(f"[embed-only] {name}: S1 判死（见 MODELS 注册表注释），跳过", flush=True)
            continue
        print(f"\n[embed-only] === {name} ({spec['repo']}) ===", flush=True)
        embedder = STEmbedder(spec)
        identity = st_identity(spec, embedder.dimensions)
        cache = CachedEmbedder(embedder, db_path=_DEFAULT_CACHE_PATH, identity=identity)
        try:
            # 标定：首个 faces 的 docgen 语料探针（该模型实际要嵌的最重文本形态）。
            first_face = spec["faces"][0]
            dataset, _cases = load_sampled_benchmark(
                first_face, _DEFAULT_DATA_DIR, 10, corpus="docgen",
            )
            probe_size = spec.get("probe_size", 100)
            probe_texts = [dataset.corpus_text(r) for r in dataset.resources[:probe_size]]
            started = perf_counter()
            embedder.embed_corpus(probe_texts)
            per_doc = (perf_counter() - started) / len(probe_texts)
            pool_docs = len(dataset.resources)
            print(
                f"[calibrate] {name}: {per_doc * 1000:.0f} ms/doc (probe {probe_size}) "
                f"-> pool {pool_docs} ≈ {per_doc * pool_docs / 60:.0f} min",
                flush=True,
            )
            face_minutes = per_doc * pool_docs / 60
            if face_minutes > S1_MAX_FACE_MINUTES:
                print(
                    f"[embed-only] {name}: S1 ABORT — 单面外推 {face_minutes:.0f} min "
                    f"超过 {S1_MAX_FACE_MINUTES} min（标定 {per_doc * 1000:.0f} ms/doc），"
                    "语料级嵌入不可行",
                    flush=True,
                )
                continue
            for bench, query_mode, limit in _bench_configs(spec["faces"]):
                dataset, cases = load_sampled_benchmark(
                    bench, _DEFAULT_DATA_DIR, limit, corpus="docgen",
                )
                texts = [dataset.corpus_text(r) for r in dataset.resources]
                texts += [case_query_text(case, query_mode) for case in cases]
                print(f"[embed] {bench}/{query_mode}: {len(texts)} texts", flush=True)
                cache.embed_corpus(texts)
        finally:
            cache.close()
            del embedder, cache
            gc.collect()
        print(f"[embed-only] {name} done (identity {identity})", flush=True)
    return 0


def gold_in_window(ranked: list[str], golds: frozenset[str], k: int) -> float:
    """any-gold 覆盖：top-k 含任一 gold 记 1（逐 case 0/1，聚合即比例）。"""
    return 1.0 if golds.intersection(ranked[:k]) else 0.0


def build_glme_embedder() -> tuple[CachedEmbedder, str]:
    config = embedder_from_env()
    if vector_available(config) != "api":
        raise RuntimeError("AGENT_RETRIEVAL_EMBEDDING_API_KEY required for --model glm")
    live = build_embedder(config)
    assert live is not None
    probe = live.embed_query("benchmark preflight")
    config = replace(config, dimensions=len(probe))
    cache = CachedEmbedder(
        live, db_path=_DEFAULT_CACHE_PATH,
        identity="|".join(config.identity()),  # R5-P1：身份串含 pooling/前缀 flags
    )
    return cache, config.identity()


def run_paired(args: argparse.Namespace) -> dict:
    dataset, cases = load_sampled_benchmark(
        args.bench, args.data_dir, args.limit, corpus=args.corpus,
    )
    if args.model == "glm":
        embedder, identity = build_glme_embedder()
    else:
        embedder, identity = build_lookup_embedder(args.model)
    rows: list[dict] = []
    started = perf_counter()
    try:
        for index, case in enumerate(cases, start=1):
            query = case_query_text(case, args.query_mode)
            hits = rank_candidates(
                dataset.resources,
                query,
                corpus_text=dataset.corpus_text,
                item_id=lambda resource: resource.id,
                item_name=lambda resource: resource.name,
                vector=embedder,
            )
            ranked = [hit.item.id for hit in hits]
            # 逐 case 指标嵌套进臂名键（jev/rerank_paired 报告约定，ablation_stats
            # pair 的 load_arm_cases 按 row[arm] 取数）。
            row = {"case_id": case.id, "fusion": score_case(ranked, case.golds, K_VALUES)}
            row["gold_in_window"] = {
                f"@{k}": gold_in_window(ranked, case.golds, k) for k in WINDOW_KS
            }
            rows.append(row)
            if index % 10 == 0:
                print(f"[{args.bench}/{args.query_mode}/{args.model}] {index}/{len(cases)}",
                      file=sys.stderr)
    finally:
        embedder.close()

    def window_means() -> dict[str, float]:
        return {
            key: round(sum(row["gold_in_window"][key] for row in rows) / len(rows), 4)
            for key in rows[0]["gold_in_window"]
        }

    return {
        "dataset": {
            "name": dataset.name, "revision": dataset.revision,
            "resources": len(dataset.resources), "cases": len(cases),
            "case_ids": [case.id for case in cases],
        },
        "configuration": {
            "model": args.model, "embedder_identity": identity,
            "query_mode": args.query_mode, "corpus": args.corpus,
            "k_values": list(K_VALUES), "window_ks": list(WINDOW_KS),
        },
        "metrics": {"fusion": aggregate([row["fusion"] for row in rows], K_VALUES)},
        "gold_in_window": window_means(),
        "elapsed_seconds": round(perf_counter() - started, 1),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help=f"{'/'.join([*MODELS, 'glm'])}；--embed-only 可传 all")
    parser.add_argument("--embed-only", action="store_true",
                        help="只嵌入入缓存（用户手动跑的长任务），不跑检索")
    parser.add_argument("--bench", choices=("toolret", "skillret"))
    parser.add_argument("--query-mode", choices=("query", "instruction"), default="query")
    parser.add_argument("--corpus", choices=("plain", "docgen"), default="docgen")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_env_file()

    if args.embed_only:
        if args.model in (None, "all"):
            # all = 仅可行模型（判死的两个已被 S1 记录，见 MODELS 注释）。
            names = [key for key, value in MODELS.items() if value.get("feasible", True)]
        else:
            names = [args.model]
        return run_embed_only(names)
    if not args.model or not args.bench or not args.output:
        parser.error("default mode needs --model, --bench and --output")
    if args.query_mode == "instruction" and args.bench != "toolret":
        print("Benchmark error: instruction mode is ToolRet-only", file=sys.stderr)
        return 2
    if args.model != "glm" and args.bench not in MODELS[args.model].get("faces", ()):
        print(
            f"Benchmark error: {args.model} is restricted to faces "
            f"{MODELS[args.model]['faces']}（01 findings：域外无证据面禁跑）",
            file=sys.stderr,
        )
        return 2
    if args.limit is None:
        args.limit = 300 if args.bench == "skillret" else 100
    try:
        report = run_paired(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Embedding paired error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"dataset": report["dataset"], "configuration": report["configuration"],
                      "metrics": report["metrics"], "gold_in_window": report["gold_in_window"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

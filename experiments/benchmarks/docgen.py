"""doc2query corpus expansion: offline LLM generation of realistic user queries.

For every resource in a benchmark pool, a chat model writes k short questions a
user might naturally ask that the resource could answer. The generated queries
are appended to the resource's retrieval corpus, so query-side vocabulary gains
a pre-computed bridge into the index — the doc2query-- pattern: expansion text
feeds the *retrieval* face only.

Production note for hosts: the kernel passes ``declared=corpus_text`` (same
function) into ``ResourceBM25Index``, so expansion text lands on the judgment
face too. Evaluation here only consumes ranking order, but a real host should
pass ``declared=<original projection>`` to keep generated text out of the
adjudication evidence.

实现已收编为一等组件 ``agent_retrieval.expansion``（extras: ``[docgen]``）——
HTTP 纪律、提示词模板与解析都住在包里；本文件只留环境组装（``resolve_docgen_config``）、
jsonl IO（``load_docgen``）、CLI（``main()``）与两个兼容入口。Config comes from
``.env`` / environment (host discipline; ``load_env_file`` in the runners loads it):

  AGENT_RETRIEVAL_DOCGEN_MODEL        required (e.g. glm-5.3-flash)
  AGENT_RETRIEVAL_DOCGEN_BASE_URL     falls back to EMBEDDING_BASE_URL (same gateway)
  AGENT_RETRIEVAL_DOCGEN_API_KEY      falls back to EMBEDDING_API_KEY

Results append to ``data/docgen-{bench}.jsonl`` (one resource per line) and
reruns skip resource ids already present — gateway interruptions and memory
reaps cost nothing (both have happened in this project).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from agent_retrieval.expansion import (
    DocGenClient,
    DocGenConfig,
    DocGenError,
    build_docgen,
    docgen_corpus_text,
    parse_generated_queries,
)

_DATA_DIR = Path(__file__).with_name("data")

__all__ = [
    "DocGenConfig",
    "DocGenError",
    "build_docgen",
    "chat_text",
    "docgen_corpus_text",
    "generate_queries",
    "load_docgen",
    "parse_generated_queries",
    "resolve_docgen_config",
]


def resolve_docgen_config() -> tuple[str, str, str]:
    """(base_url, api_key, model)；未配模型视为未启用，返回空 model 让调用方裁决。

    fallback 链（生成模型与嵌入模型常不在同一网关）：
    base_url:  DOCGEN_BASE_URL → EMBEDDING_BASE_URL（再剥 /chat/completions 与
               /embeddings 尾巴——完整端点写法是常见配置形态）
    api_key:   DOCGEN_API_KEY → 系统环境变量 ZHIPUAI_API_KEY → EMBEDDING_API_KEY
    """
    model = os.environ.get("AGENT_RETRIEVAL_DOCGEN_MODEL", "")
    base_url = os.environ.get(
        "AGENT_RETRIEVAL_DOCGEN_BASE_URL",
        os.environ.get("AGENT_RETRIEVAL_EMBEDDING_BASE_URL", ""),
    )
    base_url = base_url.removesuffix("/chat/completions").removesuffix("/embeddings")
    api_key = (
        os.environ.get("AGENT_RETRIEVAL_DOCGEN_API_KEY")
        or os.environ.get("ZHIPUAI_API_KEY")
        or os.environ.get("AGENT_RETRIEVAL_EMBEDDING_API_KEY")
        or ""
    )
    return base_url, api_key, model


def chat_text(config: DocGenConfig, prompt: str) -> str:
    """自由文本生成的兼容入口：单轮传输基元，返回模型原文（groupdesc/summarize 走此）。"""
    return DocGenClient(config).complete(prompt)


def generate_queries(
    resource_name: str,
    resource_description: str,
    resource_tags: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    k: int = 5,
) -> list[str]:
    """One chat call → k distinct user queries for the resource（原签名兼容入口）。"""
    config = DocGenConfig(base_url=base_url, api_key=api_key, model_name=model)
    return DocGenClient(config).generate(resource_name, resource_description, resource_tags, k=k)


def load_docgen(data_dir: Path, bench: str) -> dict[str, list[str]]:
    """Load generated queries as ``{resource_id: queries}``; absent file → empty."""
    path = data_dir / f"docgen-{bench}.jsonl"
    if not path.exists():
        return {}
    loaded: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            loaded[row["resource_id"]] = list(row["queries"])
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bench", choices=("toolret", "skillret"), required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="generate for the first N not-yet-generated resources only")
    parser.add_argument("--workers", type=int, default=2,
                        help="generation threads; the gateway rate-limits "
                             "GLM-4.5-Flash somewhere below 4-thread pace (429s)")
    args = parser.parse_args()

    from experiments.benchmarks.run_benchmark import load_benchmark, load_env_file

    load_env_file()
    base_url, api_key, model = resolve_docgen_config()
    if not model:
        print("AGENT_RETRIEVAL_DOCGEN_MODEL is not set — nothing to run", file=sys.stderr)
        return 2
    if not (base_url and api_key):
        print("docgen needs a base_url and api_key (DOCGEN_* or EMBEDDING_* fallbacks)",
              file=sys.stderr)
        return 2

    dataset = load_benchmark(args.bench, _DATA_DIR, None)
    out_path = _DATA_DIR / f"docgen-{args.bench}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            done = {json.loads(line)["resource_id"] for line in handle if line.strip()}
    pending = [r for r in dataset.resources if r.id not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[docgen] {args.bench}: {len(done)} done, {len(pending)} pending, "
          f"model={model}, k={args.k}", file=sys.stderr)

    started = time.perf_counter()
    generated = 0
    failed = 0
    write_lock = Lock()
    with out_path.open("a", encoding="utf-8") as out:
        def work(resource):
            try:
                result = resource, generate_queries(
                    resource.name, resource.description, resource.tags,
                    base_url=base_url, api_key=api_key, model=model, k=args.k,
                ), None
            except DocGenError as exc:
                return resource, [], exc
            # 账户级 RPM 限流实测：4 线程即触发 429（GLM-4.5-Flash）。
            # 每线程每请求后停顿，把总速率压到 ~30 RPM 的限流线下。
            time.sleep(1.0)
            return result

        # 2 线程 + 每请求 1s 停顿 ≈ 30 RPM；失败的不落盘，重跑自动补齐。
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, resource) for resource in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                resource, queries, error = future.result()
                if error is not None:
                    failed += 1
                    print(f"[docgen] {resource.id} failed: {error}", file=sys.stderr)
                    continue
                if not queries:
                    # 空结果不落盘：资源保持"未生成"，下次重跑自动重试
                    #（典型成因是上游截断，重试往往能出结果）。
                    print(f"[docgen] {resource.id} produced no queries — left for retry",
                          file=sys.stderr)
                    continue
                with write_lock:
                    out.write(json.dumps(
                        {"resource_id": resource.id, "queries": queries, "model": model},
                        ensure_ascii=False,
                    ) + "\n")
                    out.flush()
                generated += 1
                if index % 100 == 0 or index == len(pending):
                    elapsed = time.perf_counter() - started
                    print(f"[docgen] {index}/{len(pending)} generated "
                          f"({generated} ok, {failed} failed, {elapsed:.0f}s)", file=sys.stderr)
    print(f"[docgen] done: {generated} newly generated, {failed} failed, output {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

Config comes from ``.env`` / environment (host discipline; ``load_env_file``
in the runners loads it):

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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

_DATA_DIR = Path(__file__).with_name("data")


class DocGenError(RuntimeError):
    """doc2query 生成调用失败——批处理按类型捕获并落盘已成功部分。"""


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


def parse_generated_queries(text: str, k: int) -> list[str]:
    """Split model output into k distinct queries: one per line, numbering stripped.

    Lines that collapse to duplicates or empties are dropped; fewer than k
    distinct lines is accepted as-is (the model's honest yield beats padding).
    """
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. )\t")
        if not cleaned:
            continue
        if cleaned.lower() not in {q.lower() for q in queries}:
            queries.append(cleaned)
    return queries[:k]


PROMPT_TEMPLATE = """You generate realistic user queries for a retrieval index.

Resource:
Name: {name}
Description: {description}
Tags: {tags}

Write {k} distinct questions a user might naturally ask that this resource could
answer. Vary wording, intent and vocabulary — synonyms, colloquial phrasings,
different entry angles; do not reuse the resource's own words verbatim in every
query. English only. One query per line, no numbering, no explanations."""


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
    """One chat call → k distinct user queries for the resource."""
    try:
        import requests  # noqa: PLC0415 — 可选实验依赖，与 ApiEmbedder 同模式
    except ImportError as exc:
        raise DocGenError("docgen 缺依赖 requests；先 pip install requests") from exc
    prompt = PROMPT_TEMPLATE.format(
        name=resource_name, description=resource_description, tags=resource_tags, k=k,
    )
    # 网关把瞬时限流/上游故障也报成 4xx（rerank 实测教训），除鉴权失败外都退避重试。
    last_error: Exception | None = None
    for attempt, pause in enumerate((0, 2.0, 5.0, 10.0)):
        if pause:
            time.sleep(pause)
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    # GLM-4.5 系是推理模型：thinking token 计入 max_tokens，会
                    # 把 content 挤空（finish_reason=length）——显式关闭后稳定。
                    "thinking": {"type": "disabled"},
                    "max_tokens": 2000,
                    "stream": False,
                },
                timeout=60,
            )
            if response.status_code in (401, 403):
                raise DocGenError(
                    f"docgen 鉴权失败 (HTTP {response.status_code})，不重试: {response.text[:120]}",
                )
            if response.status_code >= 400:
                raise DocGenError(f"HTTP {response.status_code}: {response.text[:200]}")
            content = response.json()["choices"][0]["message"]["content"]
            return parse_generated_queries(content, k)
        except DocGenError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
    raise DocGenError(f"docgen 调用失败（已重试 3 次）: {last_error}")


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


def docgen_corpus_text(base_corpus_text: Callable[[object], str],
                       expansions: dict[str, list[str]]) -> Callable[[object], str]:
    """Wrap a corpus projection so expansion queries ride on the retrieval face.

    Resources without generated queries fall back to the base projection —
    partial generation never blocks evaluation.
    """
    def corpus_text(resource: object) -> str:
        extra = expansions.get(resource.id, [])
        return " ".join((base_corpus_text(resource), *extra)) if extra else base_corpus_text(resource)
    return corpus_text


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

"""Documentation summaries for the Jev state (truncate-vs-summary ablation).

The Jev runner clips every candidate description at 2,000 chars because
System One rejects a full 20-window of ToolRet's raw documentation JSON (long
tail to ~14k chars). Clipping mid-JSON discards endpoints and parameter names
— exactly the declared-capability evidence Jev's question asks about. This
tool measures how often that bite actually happens (``--measure-only``,
pre-registered stop gate: <20% truncated → premise fails) and generates
faithful structured summaries for the window's unique resources only —
~1.6k calls, not the 44k pool.

Anti-hallucination matters here more than usual: Jev judges "declared
capability", so a summary that invents a capability would *manufacture*
relevance. The prompt hard-constrains the model to the documentation's own
content, and ``--audit N`` prints side-by-side pairs for manual tracing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.benchmarks.docgen import (
    DocGenConfig,
    DocGenError,
    chat_text,
    resolve_docgen_config,
)
from experiments.benchmarks.jev_rerank import JEV_DESCRIPTION_MAX_CHARS
from experiments.benchmarks.run_benchmark import load_benchmark, load_env_file

_DATA_DIR = Path(__file__).with_name("data")

PROMPT_TEMPLATE = """You compress one tool's documentation for a retrieval judge.

Name: {name}
Documentation (verbatim):
{documentation}

Write a structured summary of AT MOST {target_chars} characters that keeps:
what the tool does (capability statement), its endpoints/operations, and the
parameter names with their types. Use the documentation's own wording for
capabilities and parameters. Include NOTHING that is not stated in the
documentation above — no inferred abilities, no generic filler. Plain text."""


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile; single-value windows must not crash the gate."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def measure_truncation(report_path: Path, dataset) -> dict:
    """Length distribution and clip statistics over one report's windowed pool.

    Reads ``cases[].baseline_top`` (the Jev window ids) and measures the raw
    descriptions those windows actually carried — the pre-registered premise
    gate is the fraction exceeding the 2,000-char clip.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    window_ids: set[str] = set()
    for row in report["cases"]:
        window_ids.update(row["baseline_top"])
    by_id = {resource.id: resource for resource in dataset.resources}
    lengths = [
        len(by_id[item_id].description)
        for item_id in sorted(window_ids)
        if item_id in by_id
    ]
    if not lengths:
        raise ValueError("no window ids found in the report")
    clipped = [length for length in lengths if length > JEV_DESCRIPTION_MAX_CHARS]
    split_mid_json = 0
    for item_id in sorted(window_ids):
        resource = by_id.get(item_id)
        if resource is None or len(resource.description) <= JEV_DESCRIPTION_MAX_CHARS:
            continue
        try:
            json.loads(resource.description[:JEV_DESCRIPTION_MAX_CHARS])
        except json.JSONDecodeError:
            split_mid_json += 1
    return {
        "window_resources": len(lengths),
        "chars_p50": _percentile(lengths, 0.50),
        "chars_p90": _percentile(lengths, 0.90),
        "chars_p95": _percentile(lengths, 0.95),
        "chars_max": max(lengths),
        "clipped_count": len(clipped),
        "clipped_fraction": round(len(clipped) / len(lengths), 4),
        "clip_splits_mid_json": split_mid_json,
        "clip_chars": JEV_DESCRIPTION_MAX_CHARS,
    }


def load_summaries(path: Path) -> dict[str, str]:
    """``{resource_id: summary}``; absent file → empty (generation fills it)."""
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            loaded[row["resource_id"]] = row["summary"]
    return loaded


def summarize_resource(
    name: str,
    documentation: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    target_chars: int = 1200,
) -> str:
    """One chat call → structured summary bounded near the target length."""
    prompt = PROMPT_TEMPLATE.format(
        name=name, documentation=documentation, target_chars=target_chars,
    )
    config = DocGenConfig(base_url=base_url, api_key=api_key, model_name=model)
    return chat_text(config, prompt).strip()


def window_unique_ids(report_path: Path) -> list[str]:
    """Sorted unique resource ids across one report's Jev windows."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for row in report["cases"]:
        ids.update(row["baseline_top"])
    return sorted(ids)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--measure-only", action="store_true",
                        help="print window truncation statistics and exit (premise gate)")
    parser.add_argument("--window-source", type=Path, required=True,
                        help="a jev_rerank report whose baseline_top windows define the pool")
    parser.add_argument("--output", type=Path,
                        default=_DATA_DIR / "summaries-toolret.jsonl")
    parser.add_argument("--audit", type=int, default=0,
                        help="print N summary-vs-source pairs for manual tracing")
    args = parser.parse_args()

    load_env_file()
    # 全池一次加载（窗口 id 散布全库）；duckdb 可选依赖在 loader 内提示。
    dataset = load_benchmark("toolret", _DATA_DIR, None)
    by_id = {resource.id: resource for resource in dataset.resources}

    if args.measure_only:
        print(json.dumps(measure_truncation(args.window_source, dataset),
                         ensure_ascii=False, indent=2))
        return 0

    stats = measure_truncation(args.window_source, dataset)
    print(f"[summarize] premise: {stats['clipped_fraction']:.1%} of window resources "
          f"exceed the {JEV_DESCRIPTION_MAX_CHARS}-char clip "
          f"(p50={stats['chars_p50']}, p95={stats['chars_p95']}, max={stats['chars_max']})",
          file=sys.stderr)
    if stats["clipped_fraction"] < 0.20:
        print("[summarize] premise FAILS (<20% clipped) — the ablation has nothing "
              "to bite on; report 'premise not met' instead of generating summaries",
              file=sys.stderr)
        return 2

    base_url, api_key, model = resolve_docgen_config()
    if not model:
        print("AGENT_RETRIEVAL_DOCGEN_MODEL is not set — nothing to run", file=sys.stderr)
        return 2
    if not (base_url and api_key):
        print("summarize needs a base_url and api_key (DOCGEN_* or EMBEDDING_* fallbacks)",
              file=sys.stderr)
        return 2

    done = set(load_summaries(args.output))
    pending = [item_id for item_id in window_unique_ids(args.window_source)
               if item_id not in done and item_id in by_id]
    print(f"[summarize] {len(done)} done, {len(pending)} pending, model={model}",
          file=sys.stderr)

    generated = failed = 0
    with args.output.open("a", encoding="utf-8") as out:
        for index, item_id in enumerate(pending, start=1):
            resource = by_id[item_id]
            try:
                summary = summarize_resource(
                    resource.name, resource.description,
                    base_url=base_url, api_key=api_key, model=model,
                )
            except DocGenError as exc:
                failed += 1
                print(f"[summarize] {item_id} failed: {exc}", file=sys.stderr)
                continue
            if not summary:
                print(f"[summarize] {item_id} produced no text — left for retry",
                      file=sys.stderr)
                continue
            out.write(json.dumps(
                {
                    "resource_id": item_id,
                    "summary": summary,
                    "model": model,
                    "source_chars": len(resource.description),
                },
                ensure_ascii=False,
            ) + "\n")
            out.flush()
            generated += 1
            if index % 25 == 0:
                print(f"[summarize] {index}/{len(pending)} ({generated} ok, {failed} failed)",
                      file=sys.stderr)
            # 与 docgen 相同的 ~30 RPM 限流线下纪律。
            from time import sleep
            sleep(1.0)
    print(f"[summarize] done: {generated} generated, {failed} failed, output {args.output}",
          file=sys.stderr)

    if args.audit:
        summaries = load_summaries(args.output)
        pairs = list(summaries.items())[:args.audit]
        for item_id, summary in pairs:
            source = by_id[item_id].description
            print(f"\n=== {item_id} (source {len(source)} chars) ===")
            print(f"[summary] {summary[:600]}")
            print(f"[source head] {source[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Two-tier retrieval ablation on SkillRet: groups as first-class candidates.

Borrowed from OpenViking's directory-aware retrieval (scope-before-rank): the
first ``rank_candidates`` call ranks *groups* (each with a text surface), the
top-G groups' members are pooled, and a second call ranks resources inside that
pooled subset. The kernel is untouched — ``rank_candidates`` builds a fresh
BM25 index over whatever pool it receives, so tier two is literally "pass the
subset and call again".

Ablation logic (pre-registered):

* M1 — semantic two-tier (``sub``, G=3, names surface, plain corpus) vs the
  single-tier arm in the same process: the *total* effect, which conflates the
  structural prior with the pool-shrinkage side effect (idf/avgdl change with
  the pool).
* M2 — semantic two-tier vs ``random`` two-tier (same group-size distribution,
  seeded shuffle): the random arm absorbs the shrinkage side effect, so
  semantic − random is the *net* structural-prior effect. Both are reported.

Diagnostics per case: chosen groups, per-gold group ranks, second-pool size,
forced exact injections, and the group hit rates alongside their hypergeometric
chance baselines — 47% of SkillRet cases have gold in 2-3 groups, so a single
pool-wide hit rate would read structural loss as surface weakness.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from agent_retrieval import rank_candidates

from experiments.benchmarks.ablation_stats import chance_baseline
from experiments.benchmarks.docgen import docgen_corpus_text, load_docgen
from experiments.benchmarks.metrics import aggregate, score_case
from experiments.benchmarks.run_benchmark import _DEFAULT_DATA_DIR, load_benchmark, load_env_file
from experiments.benchmarks.skillret import SKILLS_FILE

#: Cap on the concatenated member-name surface. BM25's b=0.75 length
#: normalization punishes long documents, and group size spans 21x here —
#: without a budget the first tier systematically favors small groups while
#: gold concentrates in large ones. The random-group arm carries the same
#: budget so the shrinkage+length side effects cancel in the M2 difference.
SURFACE_BUDGET_CHARS = 6000
#: Diagnostic window of ids kept per case (metrics still consume full rankings).
DIAGNOSTIC_TOP = 50


@dataclass(frozen=True)
class GroupCandidate:
    """First-tier candidate: one group and its retrieval surface."""

    key: str
    surface: str


def skill_group_keys(data_dir: Path, key_field: str) -> dict[str, str]:
    """``{resource_id: grouping key}`` re-read from the raw skills jsonl.

    ``Resource.tags`` joins six fields with spaces and the values themselves
    contain spaces, so the key is not recoverable from the projection — the
    raw row is the only honest source. The coverage assertion catches revision
    drift silently producing empty groups.
    """
    path = data_dir / SKILLS_FILE.replace("/", "_")
    keys: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys[str(row["id"])] = str(row.get(key_field) or "")
    return keys


def build_groups(
    resources: list,
    keys: dict[str, str],
) -> dict[str, list]:
    """Group the pool by key, preserving pool order inside each group."""
    groups: dict[str, list] = {}
    for resource in resources:
        groups.setdefault(keys[resource.id], []).append(resource)
    return groups


def random_groups(
    resources: list,
    sizes: list[int],
    seed: int,
) -> dict[str, list]:
    """Seeded shuffle carved into pseudo-groups with the semantic size profile.

    Same group-size distribution as the semantic arm → same second-pool sizes,
    same surface lengths, same shrinkage side effects; only the *semantics* of
    the grouping differ. Two calls with one seed give identical groups.
    """
    shuffled = list(resources)
    random.Random(seed).shuffle(shuffled)
    groups: dict[str, list] = {}
    cursor = 0
    for index, size in enumerate(sizes):
        groups[f"random-{index:02d}"] = shuffled[cursor:cursor + size]
        cursor += size
    remainder = shuffled[cursor:]
    if remainder:
        groups[f"random-{len(sizes):02d}"] = remainder
    return groups


def build_surface(
    key: str,
    members: list,
    variant: str,
    descriptions: dict[str, str] | None = None,
) -> str:
    """First-tier retrieval text for one group.

    ``names`` — group key plus member names in id order, capped at the budget.
    ``llm``   — group key plus a generated use-when sentence (naturally short,
    so the length-normalization bias does not apply).
    """
    parts = [key]
    if variant == "llm":
        if not descriptions or key not in descriptions:
            raise ValueError(f"no llm description for group {key!r} — run groupdesc first")
        parts.append(descriptions[key])
        return " ".join(parts)
    total = len(key)
    for member in sorted(members, key=lambda item: item.id):
        piece = member.name
        if total + 1 + len(piece) > SURFACE_BUDGET_CHARS:
            break
        parts.append(piece)
        total += 1 + len(piece)
    return " ".join(parts)


def surface_coverage(members: list, surface: str) -> float:
    """Fraction of member names that fit into the surface (budget diagnostics)."""
    if not members:
        return 1.0
    included = sum(1 for member in members if member.name and member.name in surface)
    return round(included / len(members), 4)


def exact_forced_groups(
    resources: list,
    keys: dict[str, str],
    query: str,
) -> set[str]:
    """Groups of resources whose id/name exactly equals the normalized query.

    Mirrors the kernel's exact semantics (NFKC + strip + casefold on the query,
    casefold on ids/names — resource_index.py). The kernel contract #4 wants an
    exact hit fed to at least one path; group ranking can miss the group, so
    its groups are force-injected *without* consuming the G quota (expected to
    stay ~0 on natural-language queries — counted, not assumed).
    """
    exact_query = unicodedata.normalize("NFKC", query).strip().casefold()
    if not exact_query:
        return set()
    forced: set[str] = set()
    for resource in resources:
        if exact_query in (resource.id.casefold(), resource.name.casefold()):
            forced.add(keys[resource.id])
    return forced


def two_tier_rank(
    query: str,
    groups: dict[str, list],
    surfaces: dict[str, str],
    g: int,
    corpus_text: Callable,
    golds: frozenset[str],
    keys: dict[str, str],
    forced: set[str],
) -> dict:
    """One case through both tiers, with mechanism diagnostics attached."""
    candidates = [GroupCandidate(key, surfaces[key]) for key in groups]
    group_hits = rank_candidates(
        candidates,
        query,
        corpus_text=lambda group: group.surface,
        item_id=lambda group: group.key,
        item_name=lambda group: group.key,
    )
    ranked_groups = [hit.item.key for hit in group_hits]
    chosen = ranked_groups[:g] + sorted(forced - set(ranked_groups[:g]))

    members: list = []
    seen: set[str] = set()
    for key in chosen:
        for resource in groups[key]:
            if resource.id not in seen:
                seen.add(resource.id)
                members.append(resource)
    hits = rank_candidates(
        members,
        query,
        corpus_text=corpus_text,
        item_id=lambda resource: resource.id,
        item_name=lambda resource: resource.name,
    )
    ranked = [hit.item.id for hit in hits]

    group_rank_of = {key: position for position, key in enumerate(ranked_groups, start=1)}
    gold_groups = {keys[gold] for gold in golds if gold in keys}
    span = len(gold_groups)
    chance = chance_baseline(len(groups), g, span)
    return {
        "ranked": ranked,
        "chosen_groups": chosen,
        "gold_group_ranks": sorted(group_rank_of[key] for key in gold_groups if key in group_rank_of),
        "second_pool_size": len(members),
        "forced_injections": sorted(forced),
        "gold_group_span": span,
        "group_hit_any": 1.0 if gold_groups.intersection(chosen) else 0.0,
        "group_hit_all": 1.0 if gold_groups.issubset(chosen) else 0.0,
        "group_chance_any": chance,
    }


def _single_rows(dataset, cases, k_values, label: str) -> list[dict]:
    """Single-tier baseline rows, identical kernel parameters to tier two."""
    rows: list[dict] = []
    started = perf_counter()
    for index, case in enumerate(cases, start=1):
        hits = rank_candidates(
            dataset.resources,
            case.query,
            corpus_text=dataset.corpus_text,
            item_id=lambda resource: resource.id,
            item_name=lambda resource: resource.name,
        )
        ranked = [hit.item.id for hit in hits]
        rows.append({
            "case_id": case.id,
            "gold_count": len(case.golds),
            "single": score_case(ranked, case.golds, k_values),
            "single_top": ranked[:DIAGNOSTIC_TOP],
        })
        if index % 25 == 0:
            print(f"[{label}] {index}/{len(cases)} cases", file=sys.stderr)
    print(f"[{label}] single tier done in {perf_counter() - started:.0f}s", file=sys.stderr)
    return rows


def _two_tier_rows(dataset, cases, groups, surfaces, g, keys, k_values, label: str) -> list[dict]:
    rows: list[dict] = []
    started = perf_counter()
    for index, case in enumerate(cases, start=1):
        forced = exact_forced_groups(dataset.resources, keys, case.query)
        outcome = two_tier_rank(
            case.query, groups, surfaces, g, dataset.corpus_text, case.golds, keys, forced,
        )
        rows.append({
            "case_id": case.id,
            "gold_count": len(case.golds),
            "two_tier": score_case(outcome["ranked"], case.golds, k_values),
            "two_tier_top": outcome["ranked"][:DIAGNOSTIC_TOP],
            "chosen_groups": outcome["chosen_groups"],
            "gold_group_ranks": outcome["gold_group_ranks"],
            "second_pool_size": outcome["second_pool_size"],
            "forced_injections": outcome["forced_injections"],
            "gold_group_span": outcome["gold_group_span"],
            "group_hit_any": outcome["group_hit_any"],
            "group_hit_all": outcome["group_hit_all"],
            "group_chance_any": round(outcome["group_chance_any"], 4),
        })
        if index % 25 == 0:
            print(f"[{label}] {index}/{len(cases)} cases", file=sys.stderr)
    print(f"[{label}] two-tier done in {perf_counter() - started:.0f}s", file=sys.stderr)
    return rows


def _load_single_from(path: Path) -> list[dict]:
    """Reuse a previous run's single-tier rows (same sampler → same cases)."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in report["cases"] if "single" in row]


def run(args: argparse.Namespace) -> dict:
    if args.g < 1:
        raise ValueError("--g must be positive")
    if args.surface == "llm" and not args.llm_desc:
        raise ValueError("--surface llm needs --llm-desc (groupdesc CLI output)")

    dataset = load_benchmark("skillret", args.data_dir, None)
    if args.corpus == "docgen":
        expansions = load_docgen(args.data_dir, "skillret")
        if not expansions:
            raise ValueError("no docgen expansions found — run the docgen CLI first")
        dataset = type(dataset)(
            name=dataset.name,
            revision=f"{dataset.revision}+docgen",
            corpus_text=docgen_corpus_text(dataset.corpus_text, expansions),
            resources=dataset.resources,
            cases=dataset.cases,
        )
    # Spread sampling identical to jev_rerank._sample: even coverage of the
    # pinned split; a prefix would concentrate on whichever authors sort first.
    total = len(dataset.cases)
    cases = (
        dataset.cases
        if args.limit >= total
        else [dataset.cases[(index * total) // args.limit] for index in range(args.limit)]
    )
    case_ids = {case.id for case in cases}

    keys = skill_group_keys(args.data_dir, args.groupby)
    pool_ids = {resource.id for resource in dataset.resources}
    if set(keys) != pool_ids:
        raise ValueError(
            f"group-key map covers {len(keys)} ids but the pool has {len(pool_ids)} "
            "— dataset revision drift, refusing to group on a partial map",
        )

    if args.groupby == "random":
        semantic = build_groups(dataset.resources, skill_group_keys(args.data_dir, "sub"))
        sizes = [len(members) for members in semantic.values()]
        groups = random_groups(dataset.resources, sizes, args.seed)
    else:
        groups = build_groups(dataset.resources, keys)

    descriptions = None
    if args.surface == "llm":
        descriptions = {
            json.loads(line)["group"]: json.loads(line)["description"]
            for line in args.llm_desc.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    surfaces = {
        key: build_surface(key, members, args.surface, descriptions)
        for key, members in groups.items()
    }

    k_values = tuple(sorted(set(args.k)))
    single_rows = (
        _load_single_from(args.single_from)
        if args.skip_single
        else _single_rows(dataset, cases, k_values, args.label)
    )
    if args.skip_single:
        loaded_ids = {row["case_id"] for row in single_rows}
        if loaded_ids != case_ids:
            raise ValueError("--single-from cases differ from this run's sampling")
    two_tier = _two_tier_rows(
        dataset, cases, groups, surfaces, args.g, keys, k_values, args.label,
    )
    by_id = {row["case_id"]: row for row in single_rows}

    return {
        "dataset": {
            "name": dataset.name,
            "revision": dataset.revision,
            "resources": len(dataset.resources),
            "cases": len(cases),
            "case_ids": sorted(case_ids),
        },
        "configuration": {
            "groupby": args.groupby,
            "g": args.g,
            "surface": args.surface,
            "corpus": args.corpus,
            "seed": args.seed,
            "sampling": "spread across pinned test split",
            "surface_budget_chars": SURFACE_BUDGET_CHARS,
            "group_sizes": {key: len(members) for key, members in sorted(groups.items())},
            "surface_lengths": {key: len(text) for key, text in sorted(surfaces.items())},
            "surface_coverage": {
                key: surface_coverage(members, surfaces[key])
                for key, members in sorted(groups.items())
            },
        },
        "metrics": {
            "single": aggregate([row["single"] for row in single_rows], k_values),
            "two_tier": aggregate([row["two_tier"] for row in two_tier], k_values),
        },
        "group_diagnostics": {
            "hit_any": round(sum(row["group_hit_any"] for row in two_tier) / len(two_tier), 4),
            "hit_all": round(sum(row["group_hit_all"] for row in two_tier) / len(two_tier), 4),
            "chance_any": round(sum(row["group_chance_any"] for row in two_tier) / len(two_tier), 4),
            "mean_second_pool_size": round(
                sum(row["second_pool_size"] for row in two_tier) / len(two_tier), 1,
            ),
            "forced_injection_cases": sum(1 for row in two_tier if row["forced_injections"]),
        },
        "cases": [
            {**by_id[row["case_id"]], **row}
            for row in two_tier
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--groupby", choices=("sub", "domain", "random"), default="sub")
    parser.add_argument("--g", type=int, default=3)
    parser.add_argument("--surface", choices=("names", "llm"), default="names")
    parser.add_argument("--llm-desc", type=Path, default=None,
                        help="groupdesc CLI output jsonl (required with --surface llm)")
    parser.add_argument("--corpus", choices=("plain", "docgen"), default="plain")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10, 20, 50))
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--skip-single", action="store_true",
                        help="reuse the single-tier rows from --single-from instead of rerunning")
    parser.add_argument("--single-from", type=Path, default=None,
                        help="a two-tier report whose single-tier rows this run reuses")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.label = f"two-tier/{args.groupby}/G{args.g}/{args.surface}"
    load_env_file()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"two-tier error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("dataset", "configuration", "metrics", "group_diagnostics")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate one use-when description per SkillRet group (two-tier llm surface).

The ``names`` surface concatenates member names under a character budget; a
generated sentence describes *when to visit the group* instead — OpenViking's
directory abstract idea, and naturally length-balanced across groups, which
removes the size/length collinearity that the budget only bounds.

One chat call per group (~18 requests for ``sub``): group key, 30 representative
member names, 3 representative descriptions → a single sentence, ≤600 chars.
Same batch discipline as docgen (shared ``DocGenClient.complete``): backoff retries,
thinking disabled, failures not persisted — rerun fills the gaps.
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
from experiments.benchmarks.run_benchmark import load_benchmark, load_env_file
from experiments.benchmarks.two_tier import build_groups, skill_group_keys

_DATA_DIR = Path(__file__).with_name("data")

#: Members shown to the model per group; a slice, not a summary — names are the
#: vocabulary the first tier will match against.
REPRESENTATIVE_NAMES = 30
REPRESENTATIVE_DESCRIPTIONS = 3

PROMPT_TEMPLATE = """You describe a category of skills for a retrieval index.

Category: {group}
Example skills in it: {names}
Example skill descriptions: {descriptions}

Write ONE sentence describing when a user's request should be routed to this
category — the kind of task it covers. Reuse the concrete vocabulary of the
example skills. English only, at most 60 words, no explanations, no quotes."""


def generate_description(
    group: str,
    member_names: list[str],
    member_descriptions: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    prompt = PROMPT_TEMPLATE.format(
        group=group,
        names=", ".join(member_names[:REPRESENTATIVE_NAMES]),
        descriptions=" | ".join(
            description[:200] for description in member_descriptions[:REPRESENTATIVE_DESCRIPTIONS]
        ),
    )
    config = DocGenConfig(base_url=base_url, api_key=api_key, model_name=model)
    return chat_text(config, prompt).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--groupby", choices=("sub", "domain"), default="sub")
    parser.add_argument("--output", type=Path,
                        default=_DATA_DIR / "groupdesc-skillret-sub.jsonl")
    args = parser.parse_args()

    load_env_file()
    base_url, api_key, model = resolve_docgen_config()
    if not model:
        print("AGENT_RETRIEVAL_DOCGEN_MODEL is not set — nothing to run", file=sys.stderr)
        return 2
    if not (base_url and api_key):
        print("groupdesc needs a base_url and api_key (DOCGEN_* or EMBEDDING_* fallbacks)",
              file=sys.stderr)
        return 2

    dataset = load_benchmark("skillret", _DATA_DIR, None)
    keys = skill_group_keys(_DATA_DIR, args.groupby)
    groups = build_groups(dataset.resources, keys)

    done: set[str] = set()
    if args.output.exists():
        done = {
            json.loads(line)["group"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [key for key in sorted(groups) if key not in done]
    print(f"[groupdesc] {args.groupby}: {len(done)} done, {len(pending)} pending, "
          f"model={model}", file=sys.stderr)

    generated = failed = 0
    with args.output.open("a", encoding="utf-8") as out:
        for index, key in enumerate(pending, start=1):
            members = groups[key]
            try:
                description = generate_description(
                    key,
                    [member.name for member in members],
                    [member.description for member in members],
                    base_url=base_url, api_key=api_key, model=model,
                )
            except DocGenError as exc:
                failed += 1
                print(f"[groupdesc] {key} failed: {exc}", file=sys.stderr)
                continue
            if not description:
                print(f"[groupdesc] {key} produced no text — left for retry", file=sys.stderr)
                continue
            out.write(json.dumps(
                {"group": key, "description": description, "model": model},
                ensure_ascii=False,
            ) + "\n")
            out.flush()
            generated += 1
            # 账户级限流线下压速率，与 docgen 相同的 ~30 RPM 纪律。
            from time import sleep
            sleep(1.0)
            print(f"[groupdesc] {index}/{len(pending)}: {key}", file=sys.stderr)
    print(f"[groupdesc] done: {generated} generated, {failed} failed, output {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

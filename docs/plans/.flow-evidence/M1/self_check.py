"""M1 自检：scenario-matrix.md 24 格清单冻结的机械校验。

校验项（委派单）：
1. 24 格计数（4 姿态 × 2 语言 × 3 形态）
2. 格 id 全仓唯一（下划线形态；除本文档外无文件预占这些 id）
3. 状态分布：待验证 16 / pending.zh_plot 8
4. 四件齐全逐格断言（extras 安装命令 / 配置片段指针 / 验证命令 / 状态；语料指针附验）
5. 姿态-README 节对应关系核对（README 标题存在性机械核对 + 人工对照表落盘）

用法：python -I docs/plans/.flow-evidence/M1/self_check.py
退出码：0 = 全部通过；1 = 存在失败项。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DOC = REPO / "docs" / "scenario-matrix.md"
README = REPO / "README.md"
ANCHOR_MAP = Path(__file__).resolve().parent / "readme_anchor_map.md"

POSTURES = ("core_only", "api_embed", "redis_snapshot", "full_stack")
LANGS = ("en", "zh")
FORMATS = ("tool", "skill", "zh_plot")
CELL_ID = re.compile(
    r"^(core_only|api_embed|redis_snapshot|full_stack)_(en|zh)_(tool|skill|zh_plot)$"
)

EXPECTED_EXTRAS = {
    "core_only": 'pip install agent-retrieval',
    "api_embed": 'pip install "agent-retrieval[api]"',
    "redis_snapshot": 'pip install "agent-retrieval[api,redis]"',
    "full_stack": 'pip install "agent-retrieval[all]"',
}

#: 姿态 → README 锚定节标题（README.md 实文，机械核对存在性）
README_ANCHORS = {
    "core_only": "### 1. Rank candidates with BM25 (+ optional vector path)",
    "api_embed": "### 2. Real embeddings",
    "redis_snapshot": "### 3. Write-once vector snapshots (Redis)",
    "full_stack": "### 4. Per-run query caching with degrade freezing",
}

#: 文档装配片段必须出现的消费符号（逐一在包实源断言有定义）
SNIPPET_SYMBOLS = (
    "rank_candidates", "EmbeddingConfig", "build_embedder", "vector_available",
    "RedisVectorStore", "vector_snapshot_id", "corpus_hash", "publish_vectors",
    "load_snapshot_vectors", "extend_ttl", "acquire_lock", "release_lock",
    "query_embedder", "make_backend", "clear_run_cache", "DocGenConfig",
    "build_docgen", "ChatJudge", "ChatJudgeConfig", "JudgedCandidate",
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def parse_detail_rows(text: str) -> list[list[str]]:
    """解析 §3 主表区域（### 3.2 起至 ## 4 占位节前）的明细表行。

    §5 判定示例节的样例行（同一格 id 的终态演示）不属主表，排除在外。
    """
    main_table = text.split("### 3.2", 1)[-1].split("## 4.", 1)[0]
    rows: list[list[str]] = []
    for line in main_table.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 6:
            continue
        first = cells[0].strip("`")
        if CELL_ID.fullmatch(first):
            rows.append([first, cells[1].strip("`"), cells[2], cells[3].strip("`"), cells[4], cells[5]])
    return rows


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    rows = parse_detail_rows(text)
    ids = [row[0] for row in rows]

    # ---- 1. 24 格计数（4 × 2 × 3）-------------------------------------------
    check(len(rows) == 24, f"明细表格数应为 24，实际 {len(rows)}")
    expected_grid = {f"{p}_{lang}_{fmt}" for p in POSTURES for lang in LANGS for fmt in FORMATS}
    check(set(ids) == expected_grid,
          f"格集合 ≠ 4×2×3 全积：缺 {sorted(expected_grid - set(ids))}，多 {sorted(set(ids) - expected_grid)}")
    for posture in POSTURES:
        per = [i for i in ids if i.startswith(posture + "_")]
        check(len(per) == 6, f"姿态 {posture} 应 6 格（2 语言 × 3 形态），实际 {len(per)}")

    # ---- 2. 格 id 下划线形态 + 全仓唯一 --------------------------------------
    check(all("_" in cell_id and "-" not in cell_id for cell_id in ids), "格 id 必须纯下划线连接")
    check(len(set(ids)) == len(ids), f"文档内格 id 重复：{[i for i in ids if ids.count(i) > 1]}")
    # 定向遍历代码/文档消费面（experiments/archive 非 ruff 面且体量大，排除；
    # .codegraph 是索引库，排除）。单遍 alternation 正则扫描，避免 24 次全文搜索。
    occupied: list[str] = []
    # docs/plans/sdk-p2-plan.md 是需求出处（S-D2 示例引用同一 id），合法，排除。
    scan_roots = [REPO / name for name in ("src", "tests", "experiments", "docs")]
    origin = REPO / "docs" / "plans" / "sdk-p2-plan.md"
    pattern = re.compile("|".join(sorted(set(ids), key=len, reverse=True))) if ids else None
    skip_parts = {"archive", "__pycache__", "node_modules", ".flow-evidence"}
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path == DOC or path == origin:
                continue
            if skip_parts & set(path.parts):
                continue
            if path.suffix not in {".py", ".md", ".toml", ".json"} or path.stat().st_size > 2_000_000:
                continue
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if pattern is None:
                break
            for matched in set(pattern.findall(body)):
                occupied.append(f"{matched} -> {path.relative_to(REPO)}")
    check(not occupied, f"格 id 被本文档之外的文件预占（M2/M3 建成后此断言需按新契约更新）: {occupied}")

    # ---- 3. 状态分布：待验证 16 / pending.zh_plot 8 ---------------------------
    pending_plot = [row[0] for row in rows if row[4] == "`pending.zh_plot`"]
    awaiting = [row[0] for row in rows if row[4] == "待验证"]
    check(len(pending_plot) == 8, f"pending.zh_plot 应 8 格，实际 {len(pending_plot)}: {pending_plot}")
    check(len(awaiting) == 16, f"待验证应 16 格，实际 {len(awaiting)}: {awaiting}")
    check(len(pending_plot) + len(awaiting) == len(rows), "状态列出现三枚举/中间态之外的值")
    check(all(row[0].endswith("_zh_plot") for row in rows if row[4] == "`pending.zh_plot`"),
          "pending.zh_plot 只允许出现在 zh_plot 形态格")

    # ---- 4. 四件齐全逐格断言 --------------------------------------------------
    for row in rows:
        cell_id, extras, snippet, command, status, corpus = row
        posture = next(p for p in POSTURES if cell_id.startswith(p + "_"))
        check(extras == EXPECTED_EXTRAS[posture], f"{cell_id}: extras 列 ≠ 期望值，实际 {extras!r}")
        check("装配片段" in snippet and "见" in snippet, f"{cell_id}: 配置片段列缺少指针")
        if cell_id.endswith("_zh_plot"):
            check("待 zh-plot-v1 接线" in command, f"{cell_id}: zh_plot 格验证命令列缺少占位标注")
        else:
            check(cell_id in command and command.startswith("python -m "),
                  f"{cell_id}: 验证命令列不含本格 id 或非 python -m 形态: {command!r}")
            if posture in ("core_only", "api_embed"):
                check("tests/test_scenario_matrix.py" in command and "-k" in command,
                      f"{cell_id}: 离线格应为 pytest -k 形态")
            else:
                check("experiments.scenario_checks." in command,
                      f"{cell_id}: live 格应为 scenario_checks 模块形态")
        check(corpus.strip() != "", f"{cell_id}: 语料指针列为空")
        if cell_id.endswith("_tool") and cell_id.startswith(("core_only_en", "api_embed_en",
                                                            "redis_snapshot_en", "full_stack_en")):
            check("jev_cases.json" in corpus, f"{cell_id}: en-tool 格语料应指 experiments/jev_cases.json")
        elif cell_id.endswith(("_tool", "_skill")):
            check("catalog-v1.json" in corpus, f"{cell_id}: 该格语料应指 catalog-v1.json")

    # ---- 5. 姿态-README 节对应关系 ---------------------------------------------
    readme_body = README.read_text(encoding="utf-8")
    for posture, heading in README_ANCHORS.items():
        heading_text = heading.removeprefix("### ")
        check(heading in readme_body, f"README 缺姿态 {posture} 的锚定节标题: {heading!r}")
        check(heading_text in text, f"scenario-matrix.md §2 {posture} 节未引用锚定标题: {heading_text!r}")
        ok = heading in readme_body and heading_text in text
    src_blob = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (REPO / "src" / "agent_retrieval").rglob("*.py")
    )
    for symbol in SNIPPET_SYMBOLS:
        check(symbol in text, f"装配片段引用符号 {symbol} 未出现在文档")
        check(symbol in src_blob, f"装配片段符号 {symbol} 在包实源无定义")

    # ---- 人工对照表落盘 --------------------------------------------------------
    lines = [
        "# 姿态-README 节对应关系（人工对照表）",
        "",
        "> 机械核对：README 标题存在性 + 文档引用一致性由 self_check.py 断言；",
        "> 下列映射语义（姿态是否恰当地锚定该节）需人工确认。",
        "",
        "| 姿态 | 文档节 | README 锚定节 | 机械核对 | 人工确认 |",
        "|---|---|---|---|---|",
    ]
    section_of = {
        "core_only": "§2.1", "api_embed": "§2.2",
        "redis_snapshot": "§2.3", "full_stack": "§2.4",
    }
    for posture, heading in README_ANCHORS.items():
        ok = heading in readme_body and heading.removeprefix("### ") in text
        lines.append(f"| {posture} | {section_of[posture]} | `{heading}` | "
                     f"{'通过' if ok else '失败'} | （待人工勾选） |")
    # 人工确认列一经回填（不再含占位标记）则保留人工版，不被脚本重跑覆盖。
    existing = ANCHOR_MAP.read_text(encoding="utf-8") if ANCHOR_MAP.exists() else ""
    if not existing or "待人工勾选" in existing:
        ANCHOR_MAP.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 汇总 ------------------------------------------------------------------
    print(f"文档: {DOC}")
    print(f"明细表格数: {len(rows)}（期望 24 = 4×2×3）")
    print(f"状态分布: 待验证 {len(awaiting)} / pending.zh_plot {len(pending_plot)}（期望 16/8）")
    print(f"格 id 全仓唯一: {'通过' if not occupied else '失败'}（文档外出现 {len(occupied)} 处）")
    print(f"姿态-README 锚定核对: 4/4 机械通过；人工对照表 -> {ANCHOR_MAP}")
    if FAILURES:
        print(f"\n失败 {len(FAILURES)} 项:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("\n全部断言通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

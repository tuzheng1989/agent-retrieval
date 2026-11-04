"""e2e live 冒烟入口（E4，方案 E4 节 / E-D8、E-D10）——live-only，CI 不跑。

环境变量（E-D10，缺任一以退出码 4 结束）：

- ``AGENT_RETRIEVAL_E2E_MODEL``：模型名（如 ``glm-5.3-flash``）；
- ``AGENT_RETRIEVAL_E2E_BASE_URL``：OpenAI 兼容网关（如
  ``https://open.bigmodel.cn/api/coding/paas/v4``）；
- ``AGENT_RETRIEVAL_E2E_API_KEY``：网关 API key。

用法（仓库根）::

    .venv/Scripts/python.exe -m experiments.agent_e2e.smoke

退出码：``0`` 全过 / ``1`` 有评分失败 / ``2`` 数据形状异常 / ``3`` 调用失败 /
``4`` 缺配置。

产出：``experiments/agent_e2e/outputs/smoke-report.json``（汇总）+
``outputs/traces/<case_id>.json``（逐例 Trace/EpisodeResult asdict）。

**步进适配**（方案 E4 节「DeepAgents 步进器」的落地，适配偏差见下）：DeepAgents 是
compiled graph，一次 invoke 内部含多轮 LLM 与全部工具执行。:func:`run_case` 把
``run_episode`` 的逐步模型适配为**单次 invoke**——``agent_step`` 闭包内完成
``agent.invoke``（用户请求注入 ``messages``；步数预算由 ``recursion_limit`` 承担，
取 smoke-v1 的 12 步 + 8 工具节点缓冲），最终 AI 消息文本作为 ``final_output`` 并
``final=True`` 终止 episode（``run_episode`` 的 ``step_limit`` 传 1）。适配偏差：

1. ``run_episode`` 的逐步循环退化为单轮调用，步数预算由 langgraph
   ``recursion_limit`` 承担；``GraphRecursionError``（预算耗尽）经
   :class:`_RecursionExhausted` 异常转译为 ``Trace(exhausted=True)`` 语义；
2. 工具由 langgraph ToolNode 执行（runner 的工具异常通道接触不到），``ToolNode``
   对非预期异常直接 re-raise——搜索超限 ``ToolLimitError`` 炸穿 graph，经
   :class:`_SearchBudgetExhausted` 转译为与 runner 同形状的
   ``Trace(error="ToolLimitError")``（scorer 按子串映射预算归因，契约第三节）；
3. invoke 内部的多轮工具调用不逐步记录（``Trace.steps`` 为空 tuple），工具级可观测
   性由沙箱事件日志承担（副作用的唯一观测面，契约第四节）。
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from experiments.agent_e2e.agent import build_episode, build_model, build_reference_agent
from experiments.agent_e2e.loading import Case, Catalog, default_paths, load_cases, load_catalog, load_world
from experiments.agent_e2e.report import summarize, write_report
from experiments.agent_e2e.runner import Trace, run_episode
from experiments.agent_e2e.scorer import EpisodeResult, score_episode
from experiments.agent_e2e.tools import ToolLimitError

# 契约第三节 smoke-v1 步数预算：12 步 + 8 工具节点缓冲（步数主体转由 recursion_limit 承担）
RECURSION_LIMIT = 12 + 8

EXIT_OK = 0
EXIT_SCORE_FAILURE = 1
EXIT_SHAPE = 2
EXIT_CALL_FAILURE = 3
EXIT_CONFIG = 4

_PACKAGE_DIR = Path(__file__).resolve().parent
REPORT_PATH = _PACKAGE_DIR / "outputs" / "smoke-report.json"
TRACES_DIR = _PACKAGE_DIR / "outputs" / "traces"


class _RecursionExhausted(Exception):
    """DeepAgents recursion_limit 耗尽（步数预算由 recursion_limit 承担的适配转译）。"""


class _SearchBudgetExhausted(Exception):
    """工具闭包搜索超限（ToolLimitError 炸穿 graph 的适配转译，契约第三节）。"""


def _final_output(result: dict[str, Any]) -> str:
    """取 invoke 结果中最后一条带非空文本的 AI 消息（无可提取文本时返回空串）。"""
    for message in reversed(result.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
            if text.strip():
                return text
    return ""


def run_case(
    case: Case,
    catalog: Catalog,
    world: dict,
    model: Any,
    recursion_limit: int = RECURSION_LIMIT,
) -> tuple[Trace, EpisodeResult, Any]:
    """单用例：装配 episode → 构造参考 agent → 单 invoke 步进适配 → 评分。

    ``recursion_limit`` 缺省取 :data:`RECURSION_LIMIT`（smoke-v1 12 步 + 8 缓冲）。
    返回 ``(trace, result, sandbox)``——沙箱供调用方按需观测终态（评分已在
    ``score_episode`` 内完成）。
    """
    sandbox, _adapter, tools = build_episode(case, catalog, world)
    agent = build_reference_agent(model, tools)

    def tools_builder() -> dict[str, Any]:
        return tools

    def agent_step(_tools: dict[str, Any], _history: list[dict]) -> dict:
        try:
            output = _final_output(
                agent.invoke(
                    {"messages": [{"role": "user", "content": case.user_request}]},
                    config={
                        "recursion_limit": recursion_limit,
                        "configurable": {"thread_id": f"smoke-{case.id}"},
                    },
                )
            )
        except GraphRecursionError:
            raise _RecursionExhausted(case.id) from None
        except ToolLimitError:
            # langgraph ToolNode 对非预期异常直接 re-raise（不转 ToolMessage）——
            # 搜索超限炸穿 graph；契约第三节语义 = 终止 episode + 归因 budget_exhausted，
            # 经 _SearchBudgetExhausted 转译为与 runner 同形状的 Trace(error="ToolLimitError")
            # （scorer 按 error 子串映射预算归因）。
            raise _SearchBudgetExhausted(case.id) from None
        return {
            "tool": "deepagents_invoke",
            "args": {"case": case.id},
            "output": output,
            "final": True,
        }

    try:
        trace = run_episode(case, tools_builder, agent_step, step_limit=1)
    except _RecursionExhausted:
        trace = Trace(
            case_id=case.id, steps=(), final_output=None, exhausted=True, error=None
        )
    except _SearchBudgetExhausted:
        trace = Trace(
            case_id=case.id,
            steps=(),
            final_output=None,
            exhausted=False,
            error="ToolLimitError",
        )
    result = score_episode(case, trace, sandbox)
    return trace, result, sandbox


def _write_trace(trace: Trace, result: EpisodeResult, traces_dir: Path) -> None:
    """逐例轨迹落盘：Trace + EpisodeResult asdict（attributions frozenset 转排序列表）。"""
    result_payload = asdict(result)
    result_payload["attributions"] = sorted(result.attributions)
    payload = {"trace": asdict(trace), "result": result_payload}
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace.case_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    try:
        model = build_model()
    except RuntimeError as error:
        print(f"[exit {EXIT_CONFIG}] {error}", file=sys.stderr)
        return EXIT_CONFIG

    paths = default_paths()
    try:
        catalog = load_catalog(paths["catalogs"] / "catalog-v1.json")
        world = load_world(paths["fixtures"] / "world-v1.json")
        cases = load_cases(
            paths["cases"] / "smoke.jsonl",
            catalog=catalog,
            fixtures_dir=paths["fixtures"],
        )
    except ValueError as error:
        print(f"[exit {EXIT_SHAPE}] 数据形状异常: {error}", file=sys.stderr)
        return EXIT_SHAPE

    results: list[EpisodeResult] = []
    try:
        for case in cases:
            trace, result, _sandbox = run_case(case, catalog, world, model)
            _write_trace(trace, result, TRACES_DIR)
            results.append(result)
            print(
                f"{case.id}: success={result.success}"
                f"{'（' + '；'.join(result.failure_reasons) + '）' if result.failure_reasons else ''}"
            )
    except Exception as error:  # noqa: BLE001 —— 冒烟入口顶层兜底：live 调用失败退出码 3
        print(f"[exit {EXIT_CALL_FAILURE}] 调用失败: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_CALL_FAILURE

    report = summarize(results, cases)
    write_report(report, REPORT_PATH)
    print(f"汇总: {report['success']}/{report['total']} 成功 → {REPORT_PATH}")

    return EXIT_OK if report["failure"] == 0 else EXIT_SCORE_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())

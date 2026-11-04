"""e2e 运行器（E3）：LLM 无关的通用执行循环（方案 E3 节 + 契约第三节预算口径）。

- ``tools_builder()`` 每 episode 调一次（重放 = 同输入再调一次；episode 间零共享由
  builder 内部保证——如 E1 ``Sandbox.from_world`` 深拷贝）；
- ``agent_step(tools, history)`` 为步进回调，返回 ``{"tool": 名, "args": {...},
  "output": 文本, "final": bool(可选)}``；``final=True`` 时 ``output`` 即最终回答，
  循环终止（不执行工具）；LLM 冒烟用 DeepAgents 步进器（E4），离线测试用脚本化步进器；
- 每步记录 ``{"tool", "args", "output"}``（异常步为 ``{"tool", "args", "error"}``）
  进 ``history``（append-only，供下一步决策）与 ``Trace.steps``；
- **终止路径恰三条**：``final=True``；步数耗尽（``exhausted=True``）；
  ``ToolLimitError``——按**异常类名**识别（不导入 E2 的 tools 模块：E3 与 E2 文件集
  互斥、W3 并行），捕获后终止 episode，``error`` 记 ``"ToolLimitError"``、
  ``exhausted=False``，预算归因由 scorer 经 ``trace.error`` 字符串映射（契约第三节：
  搜索超限执行主体是工具闭包）；
- 其余工具异常捕获记入该步（``error`` 键）后 episode 继续，不中断。
"""

from collections.abc import Callable
from dataclasses import dataclass

from experiments.agent_e2e.loading import Case


@dataclass(frozen=True)
class Trace:
    """单 episode 执行轨迹（方案 E3 节冻结字段；``steps`` 元素形状见模块 docstring）。"""

    case_id: str
    steps: tuple[dict, ...]
    final_output: str | None
    exhausted: bool
    error: str | None


def run_episode(
    case: Case,
    tools_builder: Callable[[], dict[str, Callable]],
    agent_step: Callable[[dict, list[dict]], dict],
    step_limit: int = 12,
) -> Trace:
    """执行单 episode：步进回调决策 → 工具执行 → 记录，直至 final / 耗尽 / ToolLimitError。

    ``step_limit`` 缺省 12 = 契约第三节 ``smoke-v1`` 步数上限（runner 为执行主体）。
    """
    tools = tools_builder()
    history: list[dict] = []
    steps: list[dict] = []
    final_output: str | None = None
    exhausted = False
    error: str | None = None

    for _ in range(step_limit):
        decision = agent_step(tools, history)
        if decision.get("final"):
            final_output = decision.get("output")
            break
        entry: dict = {"tool": decision["tool"], "args": dict(decision.get("args", {}))}
        try:
            entry["output"] = tools[entry["tool"]](**entry["args"])
        except Exception as exc:
            if type(exc).__name__ == "ToolLimitError":
                entry["error"] = "ToolLimitError"
                history.append(entry)
                steps.append(entry)
                error = "ToolLimitError"
                break
            entry["error"] = f"{type(exc).__name__}: {exc}"
        history.append(entry)
        steps.append(entry)
    else:
        exhausted = True

    return Trace(
        case_id=case.id,
        steps=tuple(steps),
        final_output=final_output,
        exhausted=exhausted,
        error=error,
    )

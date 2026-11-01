"""Jev 判定的裸 HTTP 参考实现：System One ``/v1/systemone`` 端点（extras: ``judge``）。

实现前调研（typesafe-sdk 源码核实，SDK P0 / B2；B3 对拍双臂共用 ``judge_state``）：

- **Noul 问题 wire 形状**（``typesafe_sdk/_core/question_types.py`` + ``_schemas/models.py``
  的 ``NoulQuestion``）：pydantic 模型序列化后为
  ``{"type": "noul", "instructions": <str>, "criteria": {"true": <str>, "false": <str>}}``；
  ``_Question._omit_none`` 序列化器把值为 None 的字段从 wire JSON 省略——本实现三键
  全非 None，恰为上形。``normalize_questions``（``_core/questions.py``）对 Noul 对象
  直接放行，仅要求至少一个问题。
- **请求体顶层键**（``_core/endpoints.py`` 的 ``prepare_system_one``）：恰三键
  ``state`` / ``model`` / ``questions``；POST 到 ``_core/constants.py`` 的
  ``SYSTEM_ONE_PATH = "/v1/systemone"``。本实现的 ``judge_state`` 产出不含 model 的
  ``{"state": ..., "questions": ...}`` 纯 JSON dict（B3 两臂唯一构造点），``JevJudge``
  加 ``model`` 键包装后发送——SDK 臂与裸 HTTP 臂共用同一 dict。
- **鉴权头**（``_core/transport.py`` 的 ``prepare``）：``Authorization: Bearer <api_key>``，
  ``Content-Type: application/json``（requests ``json=`` 自动带）。SDK 专属自我标识头
  （User-Agent: typesafe-sdk/<version>、X-TypeSafe-SDK、X-TypeSafe-Runtime、
  X-TypeSafe-Retry-Count）是 SDK 侧的，裸 HTTP 参考实现不复刻。
- **响应体**（``_core/response_types.py`` 的 ``SystemOneResponse`` + wire ``NoulAnswer``/
  ``Usage``）：``{"answers": {"c0": {"type": "noul", "noul": <float 0-1>}}, "model":
  <str>, "usage": {"input_tokens": <int|None>, "output_tokens": <int|None>}}``。

判定语义继承 ``experiments/benchmarks/jev_rerank.py:71-95`` 版（Law 1 实验与全部
findings 数字的来源）：task / retrieval_instruction / 逐候选「具体步骤」判定。候选
投影按 R3 定案只含 ``{name, description}`` 两键（tags 不进端口；use_when/avoid_when
旧版本就未进 state）——对拍差异恰为 tags 键删除一项；描述截断 2000 字符随投影继承
（jev_rerank.py:55/:67-68 的常量与切片语义）。

HTTP 纪律以 ``experiments/benchmarks/docgen.py`` 的 ``_chat_completion`` 包内实现
（``agent_retrieval/expansion/api.py``）为语义基准：退避 (0, 2.0, 5.0, 10.0)；
401/403 鉴权失败抛 :class:`JudgeResponseError` 不重试；其余 4xx 记入 ``last_error``
退避重试 3 次；网络/解析异常同循环；超时显式传参。
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from agent_retrieval.core.ports import JudgedCandidate, Judgment

#: 每候选描述截断上限（继承 jev_rerank.py:55——ToolRet 描述长尾到 ~14k 字符，
#: 20 候选窗口不截断会顶爆 System One 输入预算 HTTP 400 max_tokens_exceeded）。
JEV_DESCRIPTION_MAX_CHARS = 2000

_INSTRUCTIONS_TEMPLATE = (
    "Considering `task`, `retrieval_instruction`, and `candidates.c{index}`, "
    "is this resource suitable for at least one concrete step needed to "
    "fulfill the task? Judge its declared capability, not shared keywords."
)
_CRITERIA_TRUE = "The resource can perform a specific operation needed for the task."
_CRITERIA_FALSE = "Only a related topic, a wrong operation, or insufficient stated capability."


class JudgeResponseError(RuntimeError):
    """Jev 判定调用失败——编排面按类型捕获处理，不裸穿 requests 异常。"""


@dataclass(frozen=True)
class JevJudgeConfig:
    """单个判定模型与 System One 端点的寻址与凭据（宿主配置体系的解析结果）。"""

    base_url: str
    api_key: str
    model_name: str
    timeout_seconds: float = 60.0


def judge_state(task: str, retrieval_instruction: str, candidates: Sequence[JudgedCandidate]) -> dict:
    """构造 System One 判定请求体的纯函数（B3 对拍两臂的唯一构造点）。

    返回 ``{"state": ..., "questions": ...}`` 纯 JSON dict（不含 model 键——``model``
    由各传输臂按自己的配置包装）：SDK 臂拆键传 ``TypeSafeClient.system_one``，裸 HTTP
    臂（:class:`JevJudge`）加 ``model`` 键后 POST。候选 id 重复抛 ``ValueError``（与
    B1 ``select_with_judge`` 语义一致——重复时 probabilities 键集无法与候选一一对应）。
    """
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("judge 候选 id 必须唯一：重复 id 无法与概率键一一对应")
    state = {
        "task": task,
        "retrieval_instruction": retrieval_instruction,
        "candidates": {
            f"c{index}": {
                # R3 投影定案：只投影 {name, description}——描述按 jev_rerank 旧版
                # 切片语义截断 2000 字符；tags/use_when/avoid_when 不进 state。
                "name": candidate.name,
                "description": candidate.description[:JEV_DESCRIPTION_MAX_CHARS],
            }
            for index, candidate in enumerate(candidates)
        },
    }
    questions = {
        f"c{index}": {
            "type": "noul",
            "instructions": _INSTRUCTIONS_TEMPLATE.format(index=index),
            "criteria": {"true": _CRITERIA_TRUE, "false": _CRITERIA_FALSE},
        }
        for index in range(len(candidates))
    }
    return {"state": state, "questions": questions}


class JevJudge:
    """System One 端点的裸 HTTP 判定器，实现 B1 的 :class:`Judge` 端口。

    ``judge_state`` 构造 + model 包装 + HTTP 传输；响应 ``answers`` 按 ``c{index}``
    → 候选 id 重对齐成 ``Judgment.probabilities``。
    """

    def __init__(self, config: JevJudgeConfig) -> None:
        self._config = config

    def judge(
        self, query: str, candidates: Sequence[JudgedCandidate], *, instruction: str = ""
    ) -> Judgment:
        payload = {**judge_state(query, instruction, candidates), "model": self._config.model_name}
        started = perf_counter()
        probabilities, model, usage = self._post(payload, candidates)
        return Judgment(
            probabilities=probabilities,
            model=model,
            latency_seconds=perf_counter() - started,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def _post(
        self, payload: dict, candidates: Sequence[JudgedCandidate]
    ) -> tuple[dict[str, float], str, dict]:
        """发送请求并解析响应（解析在重试循环内——解析异常与其他瞬态故障同退避循环）。"""
        config = self._config
        # requests 是 judge extra 的依赖：下沉到调用点惰性导入，让零依赖安装（纯
        # core 用法）在 import 期不受影响，缺依赖时给出指明 extras 的报错。
        try:
            import requests  # noqa: PLC0415 — 与 ApiEmbedder/DocGenClient 同模式
        except ImportError as exc:
            raise JudgeResponseError(
                "JevJudge 缺依赖 requests；先安装 agent-retrieval[judge]",
            ) from exc
        # 网关把瞬时限流/上游故障也报成 4xx（docgen 实测教训），除鉴权失败外都退避重试。
        last_error: Exception | None = None
        for attempt, pause in enumerate((0, 2.0, 5.0, 10.0)):
            if pause:
                time.sleep(pause)
            try:
                response = requests.post(
                    f"{config.base_url.rstrip('/')}/v1/systemone",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json=payload,
                    timeout=config.timeout_seconds,
                )
                if response.status_code in (401, 403):
                    raise JudgeResponseError(
                        f"judge 鉴权失败 (HTTP {response.status_code})，"
                        f"不重试: {response.text[:120]}",
                    )
                if response.status_code >= 400:
                    # 非鉴权 4xx 按瞬时故障处理：记入 last_error 进入退避重试。
                    last_error = JudgeResponseError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                    )
                    continue
                body = response.json()
                probabilities = self._parse_answers(body, candidates)
                usage = body.get("usage") or {}
                return probabilities, body["model"], usage
            except JudgeResponseError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
        raise JudgeResponseError(f"judge 调用失败（已重试 3 次）: {last_error}")

    def _parse_answers(self, body: dict, candidates: Sequence[JudgedCandidate]) -> dict[str, float]:
        """响应 ``c{index}`` 键 → 候选 id 重对齐；缺键视为实现错误（不重试）。"""
        answers = body["answers"]
        probabilities: dict[str, float] = {}
        for index, candidate in enumerate(candidates):
            key = f"c{index}"
            if key not in answers:
                raise JudgeResponseError(
                    f"judge 响应缺候选键 {key}（实现错误，不重试）: {str(body)[:200]}",
                )
            probabilities[candidate.id] = float(answers[key]["noul"])
        return probabilities

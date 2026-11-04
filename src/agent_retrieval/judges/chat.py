"""ChatJudge：chat completions 端点的裸 HTTP 判定实现（extras: ``judge``）。

与 :class:`agent_retrieval.judges.api.JevJudge` 是并列实现（SDK P1 / D1 裁决）：
System One wire 形状（Noul 问题）与 chat 自由文本是两种线协议，不互相包装——
``Judge`` 端口（B1）已抽象掉差异。一次调用判全部候选（D2：单 prompt 输出 JSON
概率映射，与 JevJudge 调用形状对齐）。

判定语义继承 ``judge_state`` 的判定语义（task / retrieval_instruction / 逐候选
「具体步骤」判定，判声明能力不判共享关键词）；候选投影共用
:func:`agent_retrieval.judges.api.projected_candidates`（D4：投影口径单一出处，
漂移即两类 judge 语义分叉）。与 JevJudge 不做数值对拍（D3：判定载体不同概率
不可比，live 冒烟只验管线与输出形状）。

HTTP 纪律照 ``JevJudge._post`` 平移：退避 (0, 2.0, 5.0, 10.0)；401/403 鉴权失败
抛 :class:`JudgeResponseError` 不重试；其余 4xx 记入 ``last_error`` 退避重试 3 次；
网络/解析异常同循环；超时显式传参。requests 惰性导入缺失报
「先安装 agent-retrieval[judge]」（extras 复用 ``[judge]``）。
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from agent_retrieval.core.ports import JudgedCandidate, Judgment
from agent_retrieval.judges.api import JudgeResponseError, projected_candidates

#: 判定 prompt 的语义版本（进 identity）：prompt 措辞变更即升版本。
CHAT_JUDGE_PROMPT_VERSION = "1"


@dataclass(frozen=True)
class ChatJudgeConfig:
    """单个判定模型与 chat completions 端点的寻址与凭据（宿主配置体系的解析结果）。

    ``temperature`` 是判定质量的组成部分（D10：判定是分类任务，低温默认 0.0），
    进 :meth:`identity`——采样参数影响输出语义，身份串不含它会掩盖复现差异。
    """

    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.0
    timeout_seconds: float = 60.0

    def identity(self) -> tuple[str, str, str]:
        """生成判定组件身份三元组（model、prompt 版本、temperature），供宿主记录与展示。"""
        return (self.model_name, CHAT_JUDGE_PROMPT_VERSION, str(self.temperature))


#: chat 判定 prompt 模板（措辞冻结于 ``CHAT_JUDGE_PROMPT_VERSION = "1"``）：判定
#: 语义继承 ``judge_state`` 的 instructions 模板与 true/false criteria（task /
#: retrieval_instruction / 逐候选「具体步骤」判定，判声明能力不判共享关键词），
#: criteria 以分数语义翻译进正文。
_PROMPT_TEMPLATE = """You judge retrieval candidates for an agent task.

Task:
{task}

{instruction_section}For each candidate below, decide: is this resource suitable for at least one concrete step needed to fulfill the task? Judge its declared capability, not shared keywords.

Scoring criteria: a candidate scoring near 1.0 can perform a specific operation needed for the task; a candidate scoring near 0.0 is only a related topic, a wrong operation, or has insufficient stated capability.

Candidates:
{candidate_lines}

Respond with a single JSON object and no other text: keys are exactly "c0" through "c{last_index}" (one key per candidate, in order), values are floats in the [0, 1] range estimating the probability that the candidate is suitable for at least one concrete step.
"""


def _chat_judge_prompt(task: str, instruction: str, candidates: Sequence[JudgedCandidate]) -> str:
    """构造 chat 判定 prompt 的纯函数（判定 prompt 唯一构造点）。

    候选清单用 :func:`agent_retrieval.judges.api.projected_candidates` 输出逐行
    渲染 ``c{i}: {name} — {description}``（D4 投影口径单一出处：use_when/
    avoid_when 不进判定输入）。``instruction`` 非空时作 retrieval_instruction 节
    并列（与 ``Judge`` 端口 instruction 逐次调用参数语义一致），默认空串时无该节。
    """
    instruction_section = f"Retrieval instruction:\n{instruction}\n\n" if instruction else ""
    candidate_lines = "\n".join(
        f"{key}: {entry['name']} — {entry['description']}"
        for key, entry in projected_candidates(candidates).items()
    )
    return _PROMPT_TEMPLATE.format(
        task=task,
        instruction_section=instruction_section,
        candidate_lines=candidate_lines,
        last_index=len(candidates) - 1,
    )


#: markdown 围栏形态：`` ```json\n{...}\n``` ``（首行语言标记可有可无）。
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*)\n?```$", re.DOTALL)


def _extract_json(content: str) -> dict:
    """content 的 JSON 提取三态：直接 ``json.loads`` → 剥 markdown 围栏再 loads。

    解析失败抛 ``ValueError`` / ``JSONDecodeError``（**不抛**
    :class:`JudgeResponseError`）：语义是「可重试的解析异常」，由 ``_post`` 的
    退避循环捕获——键集/概率校验失败才是实现错误（不重试）。
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = _FENCE_RE.match(content.strip())
    if match is None:
        raise ValueError(f"content 不是 JSON 也不含 ```json 围栏: {content[:120]}")
    return json.loads(match.group(1))


class ChatJudge:
    """chat completions 端点的裸 HTTP 判定器，实现 B1 的 :class:`Judge` 端口。

    ``_chat_judge_prompt`` 构造 → POST ``{base_url}/chat/completions`` → 解析
    ``choices[0].message.content`` 为 JSON 概率映射 → ``c{index}`` 键对齐候选 id
    成 ``Judgment.probabilities``。HTTP 纪律照 ``JevJudge._post`` 平移（退避
    (0, 2.0, 5.0, 10.0)；401/403 单发即抛；其余 4xx 记 ``last_error`` 重试 3 次；
    网络/解析异常同循环）。
    """

    def __init__(self, config: ChatJudgeConfig) -> None:
        self._config = config

    def judge(
        self, query: str, candidates: Sequence[JudgedCandidate], *, instruction: str = ""
    ) -> Judgment:
        prompt = _chat_judge_prompt(query, instruction, candidates)
        started = perf_counter()
        probabilities, model, usage = self._post(prompt, candidates)
        return Judgment(
            probabilities=probabilities,
            model=model,
            latency_seconds=perf_counter() - started,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def _post(self, prompt: str, candidates: Sequence[JudgedCandidate]) -> tuple[dict[str, float], str, dict]:
        """发送请求并解析响应（解析在重试循环内——解析异常与其他瞬态故障同退避循环）。"""
        config = self._config
        # requests 是 judge extra 的依赖：下沉到调用点惰性导入，缺依赖时给出指明
        # extras 的报错（extras 复用 [judge]，与 JevJudge 同面）。
        try:
            import requests  # noqa: PLC0415 — 与 ApiEmbedder/DocGenClient/JevJudge 同模式
        except ImportError as exc:
            raise JudgeResponseError(
                "ChatJudge 缺依赖 requests；先安装 agent-retrieval[judge]",
            ) from exc
        # 网关把瞬时限流/上游故障也报成 4xx（docgen 实测教训），除鉴权失败外都退避重试。
        last_error: Exception | None = None
        for attempt, pause in enumerate((0, 2.0, 5.0, 10.0)):
            if pause:
                time.sleep(pause)
            try:
                response = requests.post(
                    f"{config.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json={
                        "model": config.model_name,
                        "temperature": config.temperature,
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
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
                probabilities = self._parse_probabilities(body, candidates)
                usage = body.get("usage") or {}
                return probabilities, body["model"], usage
            except JudgeResponseError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
        raise JudgeResponseError(f"judge 调用失败（已重试 3 次）: {last_error}")

    def _parse_probabilities(
        self, body: dict, candidates: Sequence[JudgedCandidate]
    ) -> dict[str, float]:
        """content 的 JSON 提取 → ``c{index}`` 键集校验 → 候选 id 重对齐。

        键集不一致（缺键/多键）与概率越界抛 :class:`JudgeResponseError` 不重试
        （模型结构性输出错误，温度 0.0 下重试无意义；越界不 clamp——失控信号）。
        """
        content = body["choices"][0]["message"]["content"]
        data = _extract_json(content)
        expected = {f"c{index}" for index in range(len(candidates))}
        if not isinstance(data, dict) or set(data) != expected:
            details = sorted(set(data) ^ expected) if isinstance(data, dict) else type(data).__name__
            raise JudgeResponseError(
                f"judge 响应键集与候选不一一对应（实现错误，不重试；差异: {details}）"
                f": {str(data)[:200]}",
            )
        probabilities: dict[str, float] = {}
        for index, candidate in enumerate(candidates):
            key = f"c{index}"
            probability = float(data[key])
            if not 0 <= probability <= 1:
                raise JudgeResponseError(
                    f"judge 概率越界（不 clamp）: {key}={probability}",
                )
            probabilities[candidate.id] = probability
        return probabilities

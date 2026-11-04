"""ChatJudge chat 端点判定参考实现的契约测试（SDK P1 / C2，替身 HTTP）。

测试面：``ChatJudgeConfig``（frozen、缺省、identity 三元组分化）、
``_chat_judge_prompt`` 纯函数（判定语义、投影口径、instruction 两态）、
``ChatJudge.judge`` 的请求体（endpoint/Bearer/payload）、解析三态（纯 JSON /
markdown 围栏 / 不可解析进重试）、失败语义（401 单发、键集缺失不重试、概率越
界）、usage 透传、依赖缺失指路 extras、``Judge`` 协议满足、L2 判停编排。网络
调用打桩、依赖缺失用 sys.modules 注入模拟，不依赖真实端点（live 冒烟属
``experiments/benchmarks/judge_chat_smoke.py``）。
"""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, replace

import pytest

import agent_retrieval
from agent_retrieval import (
    CandidateHit,
    Judge,
    JudgedCandidate,
    Judgment,
    JudgeResponseError,
    select_with_judge,
)
from agent_retrieval.judges.api import JEV_DESCRIPTION_MAX_CHARS
from agent_retrieval.judges.chat import (
    CHAT_JUDGE_PROMPT_VERSION,
    ChatJudge,
    ChatJudgeConfig,
    _chat_judge_prompt,
)

_CANDIDATES = [
    JudgedCandidate(id="jokes", name="Joke API", description="serves jokes"),
    JudgedCandidate(
        id="charts", name="Chart Tool", description="draws charts",
        use_when="需要图表时", avoid_when="纯文本任务",
    ),
]


def _config() -> ChatJudgeConfig:
    # base_url 是网关前缀（项目内 docgen/embedders 同约定）；端点路径由实现拼接。
    return ChatJudgeConfig(
        base_url="https://judge.example", api_key="secret", model_name="glm-judge",
    )


class _FakeResponse:
    """requests.post 替身的最小响应面（judge 只消费 json/status_code/text）。"""

    def __init__(self, status_code: int = 200, body: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self._body = body if body is not None else {}

    def json(self) -> dict:
        return self._body


def _chat_body(content: str, *, model: str = "glm-judge", usage: dict | None = None) -> dict:
    """chat/completions 响应替身的最小 body：choices[0].message.content + model。"""
    body: dict = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "model": model,
    }
    if usage is not None:
        body["usage"] = usage
    return body


@pytest.fixture
def chat_wire(monkeypatch):
    """把 requests.post 换成序列响应记录器；install(*responses) 按调用次序回放。"""
    store: dict = {"calls": []}

    def install(*responses: _FakeResponse) -> None:
        def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001 — 替身接口形状
            store["url"] = url
            store["headers"] = headers
            store["payload"] = json
            store["timeout"] = timeout
            store["calls"].append(json)
            index = min(len(store["calls"]) - 1, len(responses) - 1)
            return responses[index]

        import requests

        monkeypatch.setattr(requests, "post", fake_post)
        # 退避序列 (0, 2.0, 5.0, 10.0) 不打真睡眠：重试用例只断言请求计数与结果。
        monkeypatch.setattr("time.sleep", lambda seconds: None)

    return store, install


class TestChatJudgeConfig:
    def test_frozen(self):
        config = _config()
        with pytest.raises(FrozenInstanceError):
            config.model_name = "other"

    def test_defaults(self):
        config = _config()
        assert config.temperature == 0.0
        assert config.timeout_seconds == 60.0

    def test_identity_triple(self):
        """identity = (model_name, prompt 版本, str(temperature)) 三元组。"""
        config = replace(_config(), temperature=0.2)
        assert config.identity() == ("glm-judge", CHAT_JUDGE_PROMPT_VERSION, "0.2")

    def test_identity_changes_when_any_component_changes(self):
        """任一元变（model / temperature）→ identity 变（prompt 版本由三元组携带）。"""
        base = _config()
        assert replace(base, model_name="other").identity() != base.identity()
        assert replace(base, temperature=0.5).identity() != base.identity()


class TestChatJudgePrompt:
    def test_contains_task_candidate_names_and_truncated_description(self):
        """task、候选 name 与截断后 description（投影口径）出现在 prompt 正文。"""
        long_candidate = JudgedCandidate(id="big", name="Big", description="x" * 3000)
        prompt = _chat_judge_prompt("找笑话工具", "", [_CANDIDATES[0], long_candidate])

        assert "找笑话工具" in prompt
        assert "c0: Joke API" in prompt
        assert "serves jokes" in prompt
        # 截断口径与 projected_candidates 一致：3000 字符描述按 2000 切片进 prompt。
        assert "x" * JEV_DESCRIPTION_MAX_CHARS in prompt
        assert "x" * (JEV_DESCRIPTION_MAX_CHARS + 1) not in prompt

    def test_excludes_use_when_and_avoid_when(self):
        """投影口径（D4）：use_when/avoid_when 不进 prompt。"""
        prompt = _chat_judge_prompt("任务", "", _CANDIDATES)
        assert "需要图表时" not in prompt
        assert "纯文本任务" not in prompt

    def test_instruction_included_when_non_empty(self):
        prompt = _chat_judge_prompt("任务", "仅限本地文件操作", _CANDIDATES)
        assert "仅限本地文件操作" in prompt
        assert "Retrieval instruction" in prompt

    def test_no_instruction_section_when_default_empty(self):
        """默认空串 → prompt 无 retrieval_instruction 节（逐次调用参数两态）。"""
        prompt = _chat_judge_prompt("任务", "", _CANDIDATES)
        assert "Retrieval instruction" not in prompt

    def test_concrete_step_criteria_in_prompt_body(self):
        """judge_state 判定语义翻译进正文：判具体步骤所需能力，不判共享关键词。"""
        prompt = _chat_judge_prompt("任务", "", _CANDIDATES)
        assert "concrete step" in prompt
        assert "declared capability, not shared keywords" in prompt


class TestChatJudgeHttp:
    def test_posts_chat_completions_with_bearer_and_payload(self, chat_wire):
        """请求面：endpoint {base_url}/chat/completions、Bearer 头、payload 含
        model/temperature/max_tokens/messages（单 user 消息）/stream=False、timeout 透传。"""
        store, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.1}')))
        judge = ChatJudge(_config())

        judgment = judge.judge("找笑话工具", _CANDIDATES)

        assert store["url"] == "https://judge.example/chat/completions"
        assert store["headers"] == {"Authorization": "Bearer secret"}  # noqa: S105 — 桩断言
        assert store["payload"]["model"] == "glm-judge"
        assert store["payload"]["temperature"] == 0.0
        assert store["payload"]["max_tokens"] == 4096
        assert store["payload"]["stream"] is False
        assert store["payload"]["messages"] == [
            {"role": "user", "content": _chat_judge_prompt("找笑话工具", "", _CANDIDATES)},
        ]
        assert store["timeout"] == 60.0
        # 纯 JSON 200 → probabilities 键与候选 id 一一对应、值逐位一致。
        assert judgment.probabilities == {"jokes": 0.9, "charts": 0.1}
        assert judgment.model == "glm-judge"
        assert judgment.latency_seconds >= 0.0

    def test_instruction_argument_flows_into_prompt(self, chat_wire):
        """instruction 是逐次调用参数：经 _chat_judge_prompt 进入 user 消息正文。"""
        store, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.1}')))
        judge = ChatJudge(_config())

        judge.judge("任务", _CANDIDATES, instruction="仅限本地文件操作")

        content = store["payload"]["messages"][0]["content"]
        assert "仅限本地文件操作" in content

    def test_lazy_export_from_top_level(self):
        """顶层 _LAZY_EXPORTS 接线：agent_retrieval.ChatJudge / ChatJudgeConfig 可达。"""
        assert isinstance(agent_retrieval.ChatJudge(_config()), Judge)
        config = agent_retrieval.ChatJudgeConfig(
            base_url="https://judge.example", api_key="secret", model_name="glm-judge",
        )
        assert config.identity() == ("glm-judge", CHAT_JUDGE_PROMPT_VERSION, "0.0")

    def test_fenced_json_content_parses(self, chat_wire):
        """```json 围栏包裹的 content → 剥围栏后解析成功，概率逐位一致。"""
        _, install = chat_wire
        install(_FakeResponse(body=_chat_body('```json\n{"c0": 0.9, "c1": 0.1}\n```')))
        judge = ChatJudge(_config())

        judgment = judge.judge("任务", _CANDIDATES)

        assert judgment.probabilities == {"jokes": 0.9, "charts": 0.1}

    def test_unparseable_status_then_success_retries(self, chat_wire):
        """400→200：非鉴权 4xx 记 last_error 退避重试，第二次成功（请求计数 2）。"""
        store, install = chat_wire
        install(
            _FakeResponse(status_code=400, text="bad request"),
            _FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.1}')),
        )
        judge = ChatJudge(_config())

        judgment = judge.judge("任务", _CANDIDATES)

        assert judgment.probabilities == {"jokes": 0.9, "charts": 0.1}
        assert len(store["calls"]) == 2

    def test_unparseable_content_200_retries_in_same_loop(self, chat_wire):
        """HTTP 200 但 content 不可解析 → 按解析异常进同一退避循环重试。"""
        store, install = chat_wire
        install(
            _FakeResponse(body=_chat_body("I cannot answer that in JSON.")),
            _FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.1}')),
        )
        judge = ChatJudge(_config())

        judgment = judge.judge("任务", _CANDIDATES)

        assert judgment.probabilities == {"jokes": 0.9, "charts": 0.1}
        assert len(store["calls"]) == 2

    def test_auth_failure_raises_without_retry(self, chat_wire):
        """401 单次请求即抛 JudgeResponseError（请求计数 1，不退避重试）。"""
        store, install = chat_wire
        install(_FakeResponse(status_code=401, text="unauthorized"))
        judge = ChatJudge(_config())

        with pytest.raises(JudgeResponseError, match="鉴权失败"):
            judge.judge("任务", _CANDIDATES)

        assert len(store["calls"]) == 1

    def test_forbidden_raises_without_retry(self, chat_wire):
        _, install = chat_wire
        install(_FakeResponse(status_code=403, text="forbidden"))
        judge = ChatJudge(_config())

        with pytest.raises(JudgeResponseError, match="403"):
            judge.judge("任务", _CANDIDATES)

    def test_missing_candidate_key_raises_without_retry(self, chat_wire):
        """响应缺候选键（c1）视为实现错误抛 JudgeResponseError，单发不重试。"""
        store, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 0.9}')))
        judge = ChatJudge(_config())

        with pytest.raises(JudgeResponseError, match="c1"):
            judge.judge("任务", _CANDIDATES)

        assert len(store["calls"]) == 1

    def test_extra_candidate_key_raises_without_retry(self, chat_wire):
        """键集校验对称覆盖多键方向：多出 c2 → JudgeResponseError 不重试。"""
        store, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.1, "c2": 0.5}')))
        judge = ChatJudge(_config())

        with pytest.raises(JudgeResponseError, match="c2"):
            judge.judge("任务", _CANDIDATES)

        assert len(store["calls"]) == 1

    def test_probability_out_of_range_raises(self, chat_wire):
        """概率 1.2 越界 → JudgeResponseError（不 clamp——越界是模型失控信号）。"""
        _, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 1.2, "c1": 0.1}')))
        judge = ChatJudge(_config())

        with pytest.raises(JudgeResponseError):
            judge.judge("任务", _CANDIDATES)

    def test_usage_present_passes_through(self, chat_wire):
        """usage 键存在 → input/output_tokens 逐项透传。"""
        _, install = chat_wire
        install(_FakeResponse(body=_chat_body(
            '{"c0": 0.9, "c1": 0.1}', usage={"input_tokens": 120, "output_tokens": 12},
        )))

        judgment = ChatJudge(_config()).judge("任务", _CANDIDATES)

        assert judgment.input_tokens == 120
        assert judgment.output_tokens == 12

    def test_usage_absent_passes_none_through(self, chat_wire):
        """端点不计量 token：usage 缺席 → None 透传（不伪 0 值）。"""
        _, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.1}')))

        judgment = ChatJudge(_config()).judge("任务", _CANDIDATES)

        assert judgment.input_tokens is None
        assert judgment.output_tokens is None


class TestDependencyAndProtocol:
    def test_missing_dependency_reports_extras(self, monkeypatch):
        """requests 缺失：干净报错指明 agent-retrieval[judge]，不裸穿 ImportError。"""
        monkeypatch.setitem(sys.modules, "requests", None)

        with pytest.raises(JudgeResponseError, match=r"agent-retrieval\[judge\]"):
            ChatJudge(_config()).judge("任务", _CANDIDATES)

    def test_satisfies_judge_protocol(self):
        """@runtime_checkable：ChatJudge 实现 judge 方法即满足 B1 的 Judge 端口。"""
        assert isinstance(ChatJudge(_config()), Judge)


class TestL2Orchestration:
    def test_select_with_judge_consumes_chat_judgment(self, chat_wire):
        """L2 编排：ChatJudge（替身 200）→ Judgment → select_with_judge 判停消费。

        probabilities 经 minimum_probability 门槛取 id；门槛抬高到最佳概率之上
        → abstain（None 是正常返回值）。
        """
        _, install = chat_wire
        install(_FakeResponse(body=_chat_body('{"c0": 0.9, "c1": 0.4}')))
        hits = [
            CandidateHit(item="jokes", fused_score=1.0, matched_terms=(), exact=False),
            CandidateHit(item="charts", fused_score=0.8, matched_terms=(), exact=False),
        ]
        judgment: Judgment = ChatJudge(_config()).judge("任务", _CANDIDATES)

        selected = select_with_judge(
            hits, judgment, minimum_probability=0.7, item_id=lambda item: item,
        )
        abstained = select_with_judge(
            hits, judgment, minimum_probability=0.95, item_id=lambda item: item,
        )

        assert selected == "jokes"
        assert abstained is None

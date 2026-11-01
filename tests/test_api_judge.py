"""JevJudge 裸 HTTP 参考实现的契约测试（SDK P0 / B2，替身 HTTP）。

测试面：``judge_state`` 纯函数的投影与问题形状、``JevJudge.judge`` 的请求体
（= ``judge_state`` 输出 + model 包装）、重试纪律（401/403 单发即抛、其余 4xx
退避重试）、usage 透传、``Judge`` 协议满足、候选 id 唯一性。网络调用打桩、
依赖缺失用 sys.modules 注入模拟，不依赖真实端点（live 对拍属 B3）。
"""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError

import pytest

from agent_retrieval import Judge, JevJudge, JevJudgeConfig, JudgedCandidate, JudgeResponseError
from agent_retrieval.judges.api import JEV_DESCRIPTION_MAX_CHARS, judge_state

_CANDIDATES = [
    JudgedCandidate(id="jokes", name="Joke API", description="serves jokes"),
    JudgedCandidate(
        id="charts", name="Chart Tool", description="draws charts",
        use_when="需要图表时", avoid_when="纯文本任务",
    ),
]

_RESPONSE_BODY = {
    "answers": {
        "c0": {"type": "noul", "noul": 0.98},
        "c1": {"type": "noul", "noul": 0.02},
    },
    "model": "glm-judge",
    "usage": {"input_tokens": 120, "output_tokens": 12},
}


def _config() -> JevJudgeConfig:
    # base_url 是网关前缀（项目内 docgen/embedders 同约定）；端点路径含 /v1。
    return JevJudgeConfig(
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


@pytest.fixture
def judge_wire(monkeypatch):
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


class TestJudgeState:
    def test_projects_name_and_description_only(self):
        """R3 投影定案：每候选恰 {name, description} 两键——tags/use_when/avoid_when 不进 state。"""
        payload = judge_state("找笑话工具", "仅当能直接完成任务", _CANDIDATES)
        assert payload["state"]["task"] == "找笑话工具"
        assert payload["state"]["retrieval_instruction"] == "仅当能直接完成任务"
        assert payload["state"]["candidates"]["c0"] == {
            "name": "Joke API", "description": "serves jokes",
        }
        assert payload["state"]["candidates"]["c1"] == {
            "name": "Chart Tool", "description": "draws charts",
        }

    def test_empty_instruction_defaults_to_empty_string(self):
        payload = judge_state("找笑话工具", "", _CANDIDATES)
        assert payload["state"]["retrieval_instruction"] == ""

    def test_truncates_description_at_2000_chars(self):
        long_candidate = JudgedCandidate(id="big", name="Big", description="x" * 3000)
        payload = judge_state("任务", "", [long_candidate])
        projected = payload["state"]["candidates"]["c0"]["description"]
        assert len(projected) == JEV_DESCRIPTION_MAX_CHARS == 2000
        assert projected == "x" * 2000

    def test_question_shape_is_wire_form_noul(self):
        """问题对象为核实后的 wire 形状：type 标记 + instructions + true/false criteria。"""
        payload = judge_state("任务", "", _CANDIDATES)
        assert set(payload["questions"]) == {"c0", "c1"}
        question = payload["questions"]["c0"]
        assert question["type"] == "noul"
        assert "candidates.c0" in question["instructions"]
        assert "concrete step" in question["instructions"]
        assert question["criteria"] == {
            "true": "The resource can perform a specific operation needed for the task.",
            "false": "Only a related topic, a wrong operation, or insufficient stated capability.",
        }

    def test_duplicate_candidate_ids_raise(self):
        """与 B1 语义一致：重复候选 id 抛 ValueError（probabilities 键集无法一一对应）。"""
        duplicated = [JudgedCandidate(id="same", name="A", description="d")] * 2
        with pytest.raises(ValueError, match="唯一"):
            judge_state("任务", "", duplicated)


class TestJevJudgeConfig:
    def test_frozen(self):
        config = _config()
        with pytest.raises(FrozenInstanceError):
            config.model_name = "other"


class TestJevJudgeHttp:
    def test_posts_state_plus_model_with_bearer_and_timeout(self, judge_wire):
        """请求体 = judge_state 输出 + model 包装（精确 dict 断言）。"""
        store, install = judge_wire
        install(_FakeResponse(body=_RESPONSE_BODY))
        judge = JevJudge(_config())

        judge.judge("找笑话工具", _CANDIDATES, instruction="仅当能直接完成任务")

        expected = judge_state("找笑话工具", "仅当能直接完成任务", _CANDIDATES)
        assert store["payload"] == {**expected, "model": "glm-judge"}
        assert store["url"] == "https://judge.example/v1/systemone"
        assert store["headers"] == {"Authorization": "Bearer secret"}  # noqa: S105 — 桩断言
        assert store["timeout"] == 60.0

    def test_instruction_argument_flows_into_state(self, judge_wire):
        """instruction 是逐次调用参数：经 judge_state 进入请求体的 retrieval_instruction。"""
        store, install = judge_wire
        install(_FakeResponse(body=_RESPONSE_BODY))
        judge = JevJudge(_config())

        judge.judge("任务", _CANDIDATES, instruction="ToolRet is_inst 指令")

        assert store["payload"]["state"]["retrieval_instruction"] == "ToolRet is_inst 指令"

    def test_judgment_aligns_probabilities_to_candidate_ids(self, judge_wire):
        """200 → probabilities 键与候选 id 一一对应（响应 c{index} → id 重对齐）。"""
        store, install = judge_wire
        install(_FakeResponse(body=_RESPONSE_BODY))
        judge = JevJudge(_config())

        judgment = judge.judge("任务", _CANDIDATES)

        assert judgment.probabilities == {"jokes": 0.98, "charts": 0.02}
        assert judgment.model == "glm-judge"
        assert judgment.input_tokens == 120
        assert judgment.output_tokens == 12
        assert judgment.latency_seconds >= 0.0

    def test_usage_absent_passes_none_through(self, judge_wire):
        """端点不计量 token：usage 缺席 → input/output_tokens 为 None（不伪 0）。"""
        _, install = judge_wire
        body = {"answers": {"c0": {"type": "noul", "noul": 1.0}}, "model": "m"}
        install(_FakeResponse(body=body))
        judge = JevJudge(_config())

        judgment = judge.judge("任务", [_CANDIDATES[0]])

        assert judgment.input_tokens is None
        assert judgment.output_tokens is None

    def test_auth_failure_raises_without_retry(self, judge_wire):
        """401 单次请求即抛 JudgeResponseError（请求计数 1，不退避重试）。"""
        store, install = judge_wire
        install(_FakeResponse(status_code=401, text="unauthorized"))
        judge = JevJudge(_config())

        with pytest.raises(JudgeResponseError, match="鉴权失败"):
            judge.judge("任务", _CANDIDATES)

        assert len(store["calls"]) == 1

    def test_forbidden_raises_without_retry(self, judge_wire):
        _, install = judge_wire
        install(_FakeResponse(status_code=403, text="forbidden"))
        judge = JevJudge(_config())

        with pytest.raises(JudgeResponseError, match="403"):
            judge.judge("任务", _CANDIDATES)

    def test_retryable_status_then_success(self, judge_wire):
        """400→200：非鉴权 4xx 记 last_error 退避重试，第二次成功（请求计数 2）。"""
        store, install = judge_wire
        install(
            _FakeResponse(status_code=400, text="bad request"),
            _FakeResponse(body=_RESPONSE_BODY),
        )
        judge = JevJudge(_config())

        judgment = judge.judge("任务", _CANDIDATES)

        assert judgment.probabilities == {"jokes": 0.98, "charts": 0.02}
        assert len(store["calls"]) == 2

    def test_missing_answer_key_raises_implementation_error(self, judge_wire):
        """响应缺某候选键（c1）视为实现错误抛 JudgeResponseError，不重试。"""
        store, install = judge_wire
        body = {"answers": {"c0": {"type": "noul", "noul": 0.9}}, "model": "m"}
        install(_FakeResponse(body=body))
        judge = JevJudge(_config())

        with pytest.raises(JudgeResponseError, match="c1"):
            judge.judge("任务", _CANDIDATES)

        assert len(store["calls"]) == 1


class TestDependencyAndProtocol:
    def test_missing_dependency_reports_extras(self, monkeypatch):
        """requests 缺失：干净报错指明 agent-retrieval[judge]，不裸穿 ImportError。"""
        monkeypatch.setitem(sys.modules, "requests", None)
        judge = JevJudge(_config())

        with pytest.raises(JudgeResponseError, match=r"agent-retrieval\[judge\]"):
            judge.judge("任务", _CANDIDATES)

    def test_satisfies_judge_protocol(self):
        """@runtime_checkable：JevJudge 实现 judge 方法即满足 B1 的 Judge 端口。"""
        assert isinstance(JevJudge(_config()), Judge)

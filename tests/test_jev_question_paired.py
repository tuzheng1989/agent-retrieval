"""jev_questions 与配对 runner 的离线单测（零 API、零数据目录）。

四组断言：noul 构造与 judge_parity 冻结存档逐字节一致（机器证明的冻结复现）、
score 量规形状与共享措辞、折叠数值与畸形分布拒绝、报告契约（离线跑通 runner
→ stats 消费 → refold 重构自校验）。
"""

import argparse
import json

import pytest

import experiments.benchmarks.jev_question_paired as paired_module
from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource
from experiments.benchmarks.jev_questions import (
    SCORE_INSTRUCTIONS_TEMPLATE,
    SCORE_LEVELS,
    SCORE_RUBRIC,
    build_judge_state,
    build_noul_questions,
    build_score_questions,
    distribution_problems,
    fold_at_least,
    fold_expected,
    fold_top,
    refold_report,
)
from experiments.benchmarks.judge_parity import legacy_judge_payload
from agent_retrieval.stats import load_arm_cases, paired_report


def _resources(count: int = 3) -> list[Resource]:
    return [Resource(f"r{index}", f"Res {index}", f"description {index}", "tags")
            for index in range(count)]


class TestNoulFrozenParity:
    """noul 构造 = judge_parity 冻结存档（措辞冻结的机器证明）。"""

    def test_questions_match_legacy_payload_byte_for_byte(self):
        resources = _resources(3)
        legacy = legacy_judge_payload("task text", "", resources)
        mine = {
            key: dict(question.model_dump())
            for key, question in build_noul_questions(3).items()
        }
        assert mine == legacy["questions"]

    def test_state_matches_legacy_payload_field_for_field(self):
        resources = _resources(2)
        legacy = legacy_judge_payload("task text", "", resources)
        assert build_judge_state("task text", "", resources) == legacy["state"]

    def test_clip_matches_legacy_description_cap(self):
        long_resource = Resource("long", "Long", "x" * 3000, "tags")
        legacy = legacy_judge_payload("t", "", [long_resource])
        assert build_judge_state("t", "", [long_resource]) == legacy["state"]

    def test_rejects_zero_candidates(self):
        with pytest.raises(ValueError):
            build_noul_questions(0)
        with pytest.raises(ValueError):
            build_score_questions(0)


class TestScoreQuestions:
    """score 量规形状：4 档 verbatim 文案 + 与 noul 共享的措辞框架。"""

    def test_wire_shape_and_verbatim_rubric(self):
        questions = build_score_questions(2)
        assert set(questions) == {"c0", "c1"}
        dump = dict(questions["c0"].model_dump())
        assert dump["type"] == "score"
        assert tuple(dump["criteria"]) == SCORE_RUBRIC
        assert len(SCORE_RUBRIC) == SCORE_LEVELS == 4

    def test_shares_noul_wording_frame(self):
        noul_dump = dict(build_noul_questions(1)["c0"].model_dump())
        score_dump = dict(build_score_questions(1)["c0"].model_dump())
        prefix = "Considering `task`, `retrieval_instruction`, and `candidates.c0`, "
        guard = "Judge its declared capability, not shared keywords."
        assert noul_dump["instructions"].startswith(prefix)
        assert SCORE_INSTRUCTIONS_TEMPLATE.format(index=0).startswith(prefix)
        assert noul_dump["instructions"].endswith(guard)
        assert score_dump["instructions"].endswith(guard)

    def test_partial_level_absorbs_noul_true_criterion(self):
        # 语义对齐锚点：第 2 档与 noul criteria.true 共享逐字判定短语。
        noul_dump = dict(build_noul_questions(1)["c0"].model_dump())
        anchor = "perform a specific operation needed for the task"
        assert anchor in SCORE_RUBRIC[2]
        assert anchor in noul_dump["criteria"]["true"]


class TestFolds:
    """折叠数值与畸形分布拒绝。"""

    DIST = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

    def test_expected_normalizes_expectation(self):
        assert fold_expected(self.DIST) == pytest.approx((0.2 + 0.6 + 1.2) / 3)

    def test_threshold_folds(self):
        assert fold_at_least(self.DIST, 2) == pytest.approx(0.7)
        assert fold_top(self.DIST) == pytest.approx(0.4)
        assert fold_at_least(self.DIST, 1) == pytest.approx(0.9)

    def test_subset_keys_fold_as_zero(self):
        assert fold_expected({2: 0.6, 3: 0.4}) == pytest.approx(0.8)
        assert fold_top({3: 1.0}) == 1.0

    @pytest.mark.parametrize("dist", [
        {},                                  # 空
        {4: 1.0},                            # 键越界
        {0: 0.5},                            # 和值出界
        {0: 1.2, 1: -0.2},                   # 负概率
        {0: 0.96},                           # 和 0.96 出 [0.97, 1.03] 之外
    ])
    def test_malformed_distributions_raise(self, dist):
        assert distribution_problems(dist)
        with pytest.raises(ValueError):
            fold_expected(dist)

    def test_quantization_drift_within_tolerance(self):
        # 端点概率 0.01 量化：和 1.00±量化漂移应判良构。
        assert not distribution_problems({0: 0.34, 1: 0.33, 2: 0.33, 3: 0.0})


def _canned_judgment(*, malformed: bool = False):
    """构造罐头 PairJudgment：noul 升序（逆转基线序），折叠序与端点期望分一致。"""
    from types import SimpleNamespace

    from experiments.benchmarks.jev_question_paired import PairJudgment

    count = 6
    noul_scores = {f"c{index}": 0.4 + 0.1 * index for index in range(count)}
    distributions = {}
    endpoint = {}
    for index in range(count):
        if malformed and index == 0:
            distributions[f"c{index}"] = {0: 0.5}  # 和 0.5 —— malformed
            endpoint[f"c{index}"] = 0.5
            continue
        top = 3 - index % 4
        p_top = round(top / 3, 4)
        dist = {0: round(1.0 - top / 3, 4), 3: p_top}
        distributions[f"c{index}"] = dist
        endpoint[f"c{index}"] = top
    usage = {"model": "jev-test", "latency_seconds": 0.1,
             "input_tokens": 100, "output_tokens": 10}
    return PairJudgment(
        noul_scores=noul_scores,
        score_distributions=distributions,
        endpoint_scores=endpoint,
        legend_echo_fraction=0.0 if malformed else 1.0,
        model="jev-test",
        noul_usage=usage,
        score_usage=usage,
    )


@pytest.fixture()
def offline_report(tmp_path, monkeypatch):
    """bm25 臂 + 罐头 judge_pair 的完整离线 runner 报告（第 2 case malformed）。"""
    resources = [
        Resource(f"r{index}", f"name {index}", "shared", "tags") for index in range(1, 7)
    ]
    corpus_text = lambda resource: "shared"  # noqa: E731 — 全池同文档 → 基线序 = id 序
    cases = [
        Case("q-one", "shared alpha", frozenset({"r2"})),
        Case("q-two", "shared beta", frozenset({"r5"})),
        Case("q-three", "shared gamma", frozenset({"r1"})),
    ]
    dataset = BenchmarkDataset("skillret", "test", corpus_text, resources, cases)
    monkeypatch.setattr(paired_module, "load_sampled_benchmark",
                        lambda *a, **k: (dataset, cases))

    def fake_judge_pair(state, count):
        return _canned_judgment(malformed=state["task"] == "shared beta")

    monkeypatch.setattr(paired_module, "judge_pair", fake_judge_pair)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    args = argparse.Namespace(
        bench="skillret", arm="bm25", corpus="plain", query_mode="query",
        limit=3, window=20, k=(5, 10), data_dir=tmp_path, cache_path=tmp_path / "c.db",
    )
    report = paired_module.run(args)
    output = tmp_path / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report, output


class TestReportContract:
    """报告形状契约：四臂行 → stats 消费 → 良构记账 → refold 自校验。"""

    def test_four_arms_and_degradation_accounting(self, offline_report):
        report, _output = offline_report
        assert set(report["metrics"]) == {"baseline", "jev_noul", "jev_score", "oracle"}
        degradation = report["degradation"]
        assert degradation["malformed_score_cases"] == 1
        assert degradation["judge_error_cases"] == 0
        assert degradation["judged_cases"] == 3
        assert degradation["malformed_fraction"] == pytest.approx(1 / 3)
        assert report["configuration"]["model"] == ["jev-test"]

    def test_malformed_case_keeps_fusion_order_for_score_arm(self, offline_report):
        report, _output = offline_report
        row = next(row for row in report["cases"] if row["case_id"] == "q-two")
        assert row["jev_score"] == row["baseline"]
        assert row["noul_scores"]  # noul 臂不受 malformed 影响
        well = next(row for row in report["cases"] if row["case_id"] == "q-one")
        assert well["jev_score"] != well["baseline"]

    def test_usage_ledger_per_question_type(self, offline_report):
        report, _output = offline_report
        assert report["usage"]["noul"]["requests"] == 3
        assert report["usage"]["score"]["input_tokens"] == 300
        assert report["usage"]["noul"]["input_tokens"] == 300

    def test_stats_pipeline_consumes_report(self, offline_report):
        _report, output = offline_report
        assert load_arm_cases(output, "jev_score")
        result = paired_report(output, "jev_score", output, "jev_noul",
                               ["mrr", "ndcg@10", "recall@5"])
        assert result["pairing"]["shared_cases"] == 3
        assert "ndcg@10" in result["metrics"]

    def test_malformed_gate_trips(self, offline_report):
        report, _output = offline_report
        assert paired_module.malformed_gate(report)

    def test_refold_expected_reproduces_primary_arm(self, offline_report):
        _report, output = offline_report
        refolded = refold_report(output, "expected")  # 内部自校验：不一致即抛
        assert {row["case_id"] for row in refolded["cases"]} == {"q-one", "q-two", "q-three"}

    def test_refold_sensitivity_arm_pairable(self, offline_report, tmp_path):
        _report, output = offline_report
        refolded = refold_report(output, "top",
                                 output_path=tmp_path / "refold-top.json")
        arm = refolded["configuration"]["arm"]
        assert arm == "jev_score_top"
        result = paired_report(tmp_path / "refold-top.json", arm, output, "jev_noul",
                               ["ndcg@10"])
        assert result["pairing"]["shared_cases"] == 3


class TestJudgePairTransport:
    """judge_pair 的 model 一致性断言（假客户端，不构造 SDK 响应模型）。"""

    @staticmethod
    def _fake_client(models_by_type: dict):
        from types import SimpleNamespace

        usage = SimpleNamespace(input_tokens=1, output_tokens=1)

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def system_one(self, *, state, questions):
                is_score = any(getattr(question, "type", None) == "score"
                               for question in questions.values())
                model = models_by_type["score" if is_score else "noul"]
                if is_score:
                    answers = {
                        key: SimpleNamespace(score=1.0, legend={}, probabilities={0: 1.0})
                        for key in questions
                    }
                else:
                    answers = {key: SimpleNamespace(noul=0.5) for key in questions}
                return SimpleNamespace(model=model, answers=answers, usage=usage)

        return FakeClient

    def test_model_mismatch_raises(self, monkeypatch):
        import typesafe_sdk

        monkeypatch.setattr(typesafe_sdk, "TypeSafeClient",
                            self._fake_client({"noul": "model-a", "score": "model-b"}))
        with pytest.raises(RuntimeError, match="different models"):
            paired_module.judge_pair({"state": 1}, 2)

    def test_matching_models_pass(self, monkeypatch):
        import typesafe_sdk

        monkeypatch.setattr(typesafe_sdk, "TypeSafeClient",
                            self._fake_client({"noul": "model-a", "score": "model-a"}))
        judgment = paired_module.judge_pair({"state": 1}, 2)
        assert judgment.model == "model-a"
        assert judgment.noul_scores == {"c0": 0.5, "c1": 0.5}
        assert judgment.legend_echo_fraction == 0.0  # 假 legend 不回显量规


class TestSdkAnswerShape:
    """SDK 公共层字段名锁定的快检（int 键强转 + 期望分字段在位）。"""

    def test_score_answer_fields(self):
        from typesafe_sdk._core.response_types import ScoreAnswer

        answer = ScoreAnswer(score=1.5, confidence=0.8,
                             legend={0: "a", 1: "b", 2: "c", 3: "d"},
                             probabilities={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
        assert answer.probabilities[2] == 0.3
        assert answer.score == 1.5

"""Exercise the caller-side experiment without claiming simulated Jev accuracy."""

from experiments.jev_retrieval import (
    Case,
    Judgment,
    Resource,
    run_experiment,
    select_with_jev,
    shortlist,
)


class ScriptedJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, query: str, candidates: list[Resource]) -> Judgment:
        self.calls.append(query)
        winner = "geo" if query == "people in California" else None
        return Judgment(
            probabilities={resource.id: 0.9 if resource.id == winner else 0.1
                           for resource in candidates},
            model="scripted-test-only",
            latency_seconds=0.01,
            input_tokens=10,
            output_tokens=2,
        )


def test_experiment_reorders_abstains_and_preserves_exact_match():
    resources = [
        Resource("docs", "Documents", "Search files mentioning California people"),
        Resource("geo", "Geo Data", "Retrieve California population by place and year"),
    ]
    cases = [
        Case("rerank", "people in California", "geo"),
        Case("none", "Translate California documents", None),
        Case("exact", "geo", "geo"),
    ]
    judge = ScriptedJudge()

    report = run_experiment(resources, cases, k=2, threshold=0.7, judge=judge)

    assert report["candidate_recall_at_k"] == 1.0
    assert report["cases"][0]["baseline"] == "docs"
    assert report["cases"][0]["jev"] == "geo"
    assert report["jev"]["top1_accuracy"] == 1.0
    assert report["jev"]["negative_false_bind_rate"] == 0.0
    assert report["jev_usage"] == {
        "requests": 2,
        "latency_seconds": 0.02,
        "input_tokens": 20,
        "output_tokens": 4,
    }
    assert judge.calls == ["people in California", "Translate California documents"]
    assert report["cases"][2]["jev_skipped"] == "exact_match"


def test_missing_or_invalid_jev_probability_fails_closed():
    hits = shortlist([Resource("geo", "Geo Data", "Population lookup")], "population", 1)
    for probabilities in ({}, {"geo": 1.1}, {"geo": float("nan")}):
        try:
            select_with_jev(hits, Judgment(probabilities, "test", 0), 0.7)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid Jev judgment was accepted")

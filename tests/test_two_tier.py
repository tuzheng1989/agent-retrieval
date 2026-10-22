"""Unit tests for the two-tier runner's grouping and ranking mechanics."""

from __future__ import annotations

import json

import pytest

from experiments.benchmarks.dataset import Resource
from experiments.benchmarks.two_tier import (
    SURFACE_BUDGET_CHARS,
    build_groups,
    build_surface,
    exact_forced_groups,
    random_groups,
    skill_group_keys,
    surface_coverage,
    two_tier_rank,
)


def _resource(suffix: str, name: str, group: str) -> Resource:
    return Resource(id=f"id-{suffix}", name=name, description=f"desc {suffix}", tags=group)


class TestSkillGroupKeys:
    def test_reads_keys_and_asserts_coverage(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # loader flattens "data/skills/test.jsonl" → "data_skills_test.jsonl"
        rows = [
            {"id": "a", "sub": "Development"},
            {"id": "b", "sub": "Technical Search"},
            {"id": "c", "sub": ""},
        ]
        (data_dir / "data_skills_test.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8",
        )
        keys = skill_group_keys(data_dir, "sub")
        assert keys == {"a": "Development", "b": "Technical Search", "c": ""}


class TestRandomGroups:
    def test_same_seed_same_groups(self):
        resources = [_resource(str(index), f"name{index}", "x") for index in range(20)]
        first = random_groups(resources, [10, 6, 4], seed=7)
        second = random_groups(resources, [10, 6, 4], seed=7)
        assert {key: [item.id for item in members] for key, members in first.items()} == {
            key: [item.id for item in members] for key, members in second.items()
        }

    def test_preserves_size_profile_and_membership(self):
        resources = [_resource(str(index), f"name{index}", "x") for index in range(20)]
        groups = random_groups(resources, [10, 6, 4], seed=7)
        assert sorted(len(members) for members in groups.values()) == [4, 6, 10]
        pooled = [item.id for members in groups.values() for item in members]
        assert sorted(pooled) == sorted(item.id for item in resources)


class TestBuildSurface:
    def test_names_respects_budget(self):
        members = [_resource(str(index), f"very-long-name-{index:04d}", "g") for index in range(500)]
        surface = build_surface("g", members, "names")
        assert len(surface) <= SURFACE_BUDGET_CHARS

    def test_llm_requires_description(self):
        members = [_resource("1", "n", "g")]
        with pytest.raises(ValueError):
            build_surface("g", members, "llm", descriptions=None)
        surface = build_surface("g", members, "llm", descriptions={"g": "when to use"})
        assert surface == "g when to use"

    def test_coverage_counts_included_names(self):
        members = [_resource(str(index), f"n{index}", "g") for index in range(10)]
        surface = "g " + " ".join(member.name for member in members[:4])
        assert surface_coverage(members, surface) == 0.4


class TestExactForcedGroups:
    def test_name_exact_match_forces_group(self):
        resources = [_resource("1", "Deploy Skill", "ops"), _resource("2", "other", "web")]
        keys = {"id-1": "ops", "id-2": "web"}
        assert exact_forced_groups(resources, keys, "deploy skill") == {"ops"}

    def test_natural_language_matches_nothing(self):
        resources = [_resource("1", "Deploy Skill", "ops")]
        keys = {"id-1": "ops"}
        assert exact_forced_groups(resources, keys, "how do I deploy services?") == set()


class TestTwoTierRank:
    @pytest.fixture
    def small_setup(self):
        groups = {
            "alpha": [_resource("a1", "alpha one", "alpha"), _resource("a2", "shared", "alpha")],
            "beta": [_resource("b1", "beta one", "beta")],
            "gamma": [_resource("g1", "gamma one", "gamma")],
        }
        keys = {"id-a1": "alpha", "id-a2": "alpha", "id-b1": "beta", "id-g1": "gamma"}
        corpus_text = lambda resource: " ".join((resource.id, resource.name, resource.description))  # noqa: E731
        return groups, keys, corpus_text

    def test_ranking_deterministic_and_complete(self, small_setup):
        groups, keys, corpus_text = small_setup
        surfaces = {key: build_surface(key, members, "names") for key, members in groups.items()}
        first = two_tier_rank("beta one", groups, surfaces, 1, corpus_text,
                              frozenset({"id-b1"}), keys, set())
        second = two_tier_rank("beta one", groups, surfaces, 1, corpus_text,
                               frozenset({"id-b1"}), keys, set())
        assert first["ranked"] == second["ranked"]
        # G=1 keeps only the winning group's members in the second pool.
        assert first["second_pool_size"] == len(groups[first["chosen_groups"][0]])
        assert set(first["ranked"]) <= {item.id for members in groups.values() for item in members}

    def test_hit_flags_against_chosen_groups(self, small_setup):
        groups, keys, corpus_text = small_setup
        surfaces = {key: build_surface(key, members, "names") for key, members in groups.items()}
        outcome = two_tier_rank("beta one", groups, surfaces, 3, corpus_text,
                                frozenset({"id-b1", "id-g1"}), keys, set())
        assert outcome["gold_group_span"] == 2
        assert outcome["group_hit_any"] == 1.0
        assert outcome["group_hit_all"] == 1.0
        assert outcome["group_chance_any"] == pytest.approx(1.0)  # G=3 of 3 groups

    def test_forced_group_injected_without_consuming_quota(self, small_setup):
        groups, keys, corpus_text = small_setup
        surfaces = {key: build_surface(key, members, "names") for key, members in groups.items()}
        # exact hit in gamma, but the first tier is told nothing about it (surfaces
        # for gamma stripped) and G=1 goes to whichever group wins without it.
        stripped = dict(surfaces)
        stripped["gamma"] = "gamma"
        outcome = two_tier_rank("g1", groups, stripped, 1, corpus_text,
                                frozenset({"id-g1"}), keys, forced={"gamma"})
        assert "gamma" in outcome["chosen_groups"]
        assert outcome["forced_injections"] == ["gamma"]
        assert "id-g1" in outcome["ranked"]

    def test_empty_golds_get_zero_baseline(self, small_setup):
        groups, keys, corpus_text = small_setup
        surfaces = {key: build_surface(key, members, "names") for key, members in groups.items()}
        outcome = two_tier_rank("anything", groups, surfaces, 1, corpus_text,
                                frozenset(), keys, set())
        assert outcome["gold_group_span"] == 0
        assert outcome["group_chance_any"] == 0.0
        assert outcome["group_hit_any"] == 0.0


class TestBuildGroups:
    def test_key_map_drives_partition(self):
        resources = [_resource("1", "n1", "x"), _resource("2", "n2", "y"),
                     _resource("3", "n3", "x")]
        keys = {"id-1": "x", "id-2": "y", "id-3": "x"}
        groups = build_groups(resources, keys)
        assert [item.id for item in groups["x"]] == ["id-1", "id-3"]
        assert [item.id for item in groups["y"]] == ["id-2"]

"""Benchmark adapters (ToolRet, SkillRet) for the agent_retrieval kernel.

Caller-side experiments only: the package core is untouched. Entry points:

- :func:`run_benchmark.main` — CLI (``python -m experiments.benchmarks.run_benchmark``)
- :mod:`metrics` — recall@k / completeness@k / NDCG@k / MRR over full rankings
- :mod:`skillret`, :mod:`toolret` — dataset download + projection
- :mod:`embedder_cache` — persistent vector cache for benchmark arms
"""

from experiments.benchmarks.dataset import BenchmarkDataset, Case, Resource
from experiments.benchmarks.metrics import aggregate, score_case

__all__ = [
    "BenchmarkDataset",
    "Case",
    "Resource",
    "aggregate",
    "score_case",
]

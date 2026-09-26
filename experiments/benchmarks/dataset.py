"""Shared benchmark shape: candidate resources, labeled cases, corpus projection.

A benchmark is loaded into a :class:`BenchmarkDataset` so the runner stays generic
across ToolRet and SkillRet. The corpus projection (the "declared face" the kernel
indexes) is part of the dataset — each benchmark decides which fields are retrieval
evidence, mirroring the kernel discipline that the library itself models nothing
about what a declaration is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resource:
    """One candidate resource in a benchmark pool."""

    id: str
    name: str
    description: str
    #: Extra declared fields (taxonomy tags) folded into the lexical corpus.
    tags: str = ""


@dataclass(frozen=True)
class Case:
    """One benchmark query with its gold resource ids (binary relevance)."""

    id: str
    query: str
    golds: frozenset[str] = field(default_factory=frozenset)
    #: Optional task-aware retrieval instruction (ToolRet ships one per query;
    #: official main protocol feeds it alongside the query — ``is_inst=True``).
    instruction: str = ""


@dataclass(frozen=True)
class BenchmarkDataset:
    """A loaded benchmark: candidate pool plus labeled cases and its corpus face."""

    name: str
    revision: str
    corpus_text: Callable[[Resource], str]
    resources: list[Resource]
    cases: list[Case]

"""声明式语料投影：字段选择（``fields``）+ ``to_callable()`` 产出投影函数。

双形态条款（ADR-0001）：

- **声明式为默认**：``fields`` 是 ``tuple[str, ...]``，天然可 JSON——宿主自行
  ``json.dumps(list(projection.fields))`` 即可把它写进场景矩阵配置文档，包内不设
  序列化 API；
- **callable 逃生口保留**：:func:`agent_retrieval.rank_candidates` 的 ``corpus_text``
  参数签名不变，宿主仍可直接传函数；本组件只是声明式形态的载体，不改内核契约。

宿主资源有哪些字段内核不建模（零建模纪律）：构造时不做字段白名单校验，字段缺失由
``getattr`` 在调用时天然抛 ``AttributeError``。取值链为 ``None`` 或空串的 part 丢弃
（字段缺席不产生双空格，过滤语义照 ``experiments/golden/gate.py`` 的 ``_plain_text``）；
全字段皆空返回空串，不抛。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CorpusProjection:
    """声明式语料投影：逐字段 ``getattr`` 取值、过滤空 part、``separator`` 拼接。"""

    fields: tuple[str, ...]
    separator: str = " "

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("CorpusProjection.fields 不能为空")
        for field in self.fields:
            if not field:
                raise ValueError("CorpusProjection.fields 不能含空串字段")

    def to_callable(self) -> Callable[[T], str]:
        """产出投影函数：``rank_candidates(corpus_text=...)`` 直接可消费的形态。"""
        fields = self.fields
        separator = self.separator

        def project(item: T) -> str:
            return separator.join(
                part for part in (getattr(item, field) for field in fields) if part
            )

        return project

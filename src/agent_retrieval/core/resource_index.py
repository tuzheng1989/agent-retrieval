"""注册资源 BM25 适配器；评分内核与 Tool Search 共用，领域阈值保持独立。"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from agent_retrieval.core.bm25 import BM25Index, DEFAULT_BM25_B, DEFAULT_BM25_K1, tokenize

T = TypeVar("T")

_SINGLE_CJK_RE = re.compile(r"^[㐀-䶿一-鿿]$")


def substantive_term(term: str) -> bool:
    """判断一个查询 token 是否算证据。

    资源发现的查询是整段自然语言目标，不是 Tool Search 那种可能只有一两个字的关键词。
    此时单字 CJK token 几乎全是虚词（的、于、天、关……）：它们在任何资源正文里都必然出现，
    既抬分又凑重叠数，会让"写一首关于秋天的诗"稳定命中推演类 Flow。长查询侧直接把它们从
    **查询**中剔除；索引仍保留一元组，Tool Search 的单字查询召回不受影响。
    """
    return not _SINGLE_CJK_RE.match(term)


@dataclass(frozen=True)
class ResourceHit(Generic[T]):
    item: T
    score: float
    overlap: int
    #: 命中词中落在资源**声明面**（id/name/description/tags/use_when）上的个数。
    declared_overlap: int
    exact: bool
    #: 与语料规模无关的相对分：``score / ideal_score``，即"覆盖了查询证据的多大比例"。
    #: 判定一律用它，不用 ``score``——原始分随注册资源变多而整体抬升（idf 随 N 增长），
    #: 任何写死的分数线都只在标定时那个规模上成立。见 ``search.bm25.ideal_score``。
    relevance: float = 0.0


class ResourceBM25Index(Generic[T]):
    """为 Flow、Capability Skill、Agent 和 DAG Node 提供领域搜索结果。

    检索面和**判定面**分开：``documents`` 含资源正文，负责召回——正文里的领域词是长目标
    唯一够用的信号，真实注册表的 ``use_when``/``avoid_when`` 往往是空的，只靠元数据召回会
    塌。``declared`` 只含资源自己声明的身份字段，负责判定"是否真的相关"：正文长且杂，
    n-gram 会切出 ``一张``、``的机`` 这种跨词边界的噪声词，它们照样能凑够分数和重叠数
    （"帮我订一张明天去上海的机票"因此稳定命中推演 Flow），但**不可能**出现在资源的名称或
    描述里。因此 ``clearly_related`` 要求证据落在声明面上。
    """

    def __init__(
        self,
        documents: list[tuple[T, str]],
        *,
        declared: Callable[[T], str] | None = None,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        self._index = BM25Index(documents, k1=k1, b=b)
        # 取值函数而不是平行列表：判定面必须与检索面严格同序，让调用方自己维护两个等长
        # 列表迟早会错位。省略 declared 时判定面退化为检索面（DAG 节点没有"正文"之分）。
        self._declared_tokens = tuple(
            {
                term
                for term in tokenize(declared(item) if declared is not None else text)
                if substantive_term(term)
            }
            for item, text in documents
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        identity=lambda item: (str(item), ""),
    ) -> list[ResourceHit[T]]:
        exact_query = unicodedata.normalize("NFKC", query).strip().casefold()
        query_terms = {term for term in tokenize(query) if substantive_term(term)}
        ideal = self._index.ideal_score(query, term_filter=substantive_term)
        hits: list[ResourceHit[T]] = []
        for index, match in enumerate(
            self._index.score(query, term_filter=substantive_term)
        ):
            item_id, item_name = identity(match.item)
            exact = exact_query in {item_id.casefold(), item_name.casefold()}
            if match.score > 0 or exact:
                hits.append(
                    ResourceHit(
                        item=match.item,
                        score=match.score,
                        overlap=match.overlap,
                        declared_overlap=len(query_terms & self._declared_tokens[index]),
                        exact=exact,
                        relevance=(match.score / ideal) if ideal > 0 else 0.0,
                    )
                )
        hits.sort(
            key=lambda hit: (
                -(hit.score + (1000.0 if hit.exact else 0.0)),
                identity(hit.item)[0],
            )
        )
        return hits[:limit]


def clearly_related(hit: ResourceHit[object], *, minimum_relevance: float) -> bool:
    """要求精确身份命中，或足够强且**落在资源声明面上**的词汇证据。

    三个条件缺一不可：相对分够高（弱相关排除）、实词重叠够多（单点巧合排除）、且至少有一个
    命中词出现在资源自己声明的身份字段里（正文噪声 n-gram 排除）。

    **判据用相对分而不是原始 BM25 分**：原始分随注册资源变多整体抬升（idf ∝ log N，单个
    命中词从 N=1 的 0.29 涨到 N=50 的 3.5），任何写死的分数线都只在标定时那个规模上成立。
    实测四条 Flow 的语料上，绝对线 ``1.0`` 已经拦不住噪声——一次"帮我订机票"对推演 Flow
    的巧合命中就能拿到 2.29 分，真正在拦的只剩 ``declared_overlap``。资源库一长大，
    自动发现会稳定绑满上限、相关性持续下降，而且**不报错、只是慢慢变差**。

    相对分 ``score / ideal_score`` 的分子分母同随 idf 缩放，比值稳定。同一份语料上实测：
    真命中 0.32–0.72，噪声命中 0.045–0.054，差一个数量级。

    ⚠ 阈值的**绝对大小与查询长度有关**（``ideal_score`` 对查询里每个实词求和），所以各调用
    点按自己的查询形态各自标定，不要跨调用点搬数值。``minimum_relevance`` 因此**没有默认值**：
    一个能被"忘了传"的默认，等于给跨调用点搬数值开了一个不需要理由的口子。
    """
    return hit.exact or (
        hit.relevance >= minimum_relevance
        and hit.overlap >= 2
        and hit.declared_overlap >= 1
    )

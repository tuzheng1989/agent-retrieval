"""Tool、Flow、Skill 与 Agent 共用的确定性 BM25 数学内核。"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

TOKENIZER_VERSION = "bm25-natural-v3"
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_IDENTIFIER_SEPARATORS_RE = re.compile(r"[_\-.:/\\]+")
_CAMEL_ACRONYM_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


def _normalize(text: str) -> str:
    """执行 NFKC、驼峰拆分、标识符分隔符归一和小写化。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = _CAMEL_ACRONYM_RE.sub(r"\1 \2", normalized)
    normalized = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", normalized)
    return _IDENTIFIER_SEPARATORS_RE.sub(" ", normalized).lower()


def tokenize(text: str) -> tuple[str, ...]:
    """把中英文自然语言转成稳定 token。

    英文按单词切分；中文对每个连续片段同时产**一元、二元、三元**组，四字片段另补整串。
    该实现不依赖分词模型，离线可用且跨进程结果一致。

    一元组不能省：文档侧只产 n≥2 的 token 时，"图"这类单字查询在任何文档里都找不到对应
    token，BM25 必然零分，而子串兜底只看 id/name——单字中文查询于是**永远**召回不到东西。
    整串只在 4 字时单独补：2/3 字片段的整串已经由二/三元组产出，再 append 一次等于把同一
    个词计两遍，凭空让它的 TF 翻倍。

    代价是**单字中文 token 同时也是虚词**（的、于、天、关……）。短查询里它们是唯一召回手段，
    长自然语言查询里它们纯是噪声：任意两个虚词就能凑出"重叠"。因此长查询侧的适配器应当用
    ``score(..., term_filter=...)`` 把单字 CJK 词从**查询**中剔除，而不是从索引里删——
    删索引会把上面那条单字查询的召回一起废掉。见 ``agent_retrieval.core.resource_index.substantive_term``。
    """
    normalized = _normalize(text)
    tokens: list[str] = _LATIN_RE.findall(normalized)
    for run in _CJK_RE.findall(normalized):
        tokens.extend(run)
        if len(run) >= 2:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
        if len(run) == 4:
            tokens.append(run)
    return tuple(tokens)


def content_length(text: str) -> int:
    """BM25 长度归一化用的**语言中立**文档长度：拉丁词计 1，中文按字计 1。

    不能直接用 ``len(tokenize(text))``：中文的 n-gram 展开会把同等信息量的文档撑成三倍
    长度，而 avgdl 是全语料共用的，于是 ``b`` 归一化会系统性压制中文描述的资源——同一个
    Catalog 里混着中文和英文声明时，英文条目凭"文档短"白拿一截分。
    """
    normalized = _normalize(text)
    latin = len(_LATIN_RE.findall(normalized))
    cjk = sum(len(run) for run in _CJK_RE.findall(normalized))
    return latin + cjk


@dataclass(frozen=True)
class BM25Score(Generic[T]):
    """通用内核对单个文档给出的分数和唯一查询词重叠数。"""

    item: T
    score: float
    overlap: int


class BM25Index(Generic[T]):
    """只负责 token、TF/DF、长度归一化和 BM25 评分，不承载领域筛选策略。"""

    def __init__(
        self,
        documents: list[tuple[T, str]] | tuple[tuple[T, str], ...],
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        self.items = tuple(item for item, _ in documents)
        texts = tuple(text for _, text in documents)
        self.k1 = k1
        self.b = b
        tokens = tuple(tokenize(text) for text in texts)
        self._term_frequencies = tuple(Counter(document_tokens) for document_tokens in tokens)
        self._document_lengths = tuple(content_length(text) for text in texts)
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for document_tokens in tokens:
            document_frequency.update(set(document_tokens))
        self._document_frequency = document_frequency

    def score(
        self,
        query: str,
        *,
        term_filter: Callable[[str], bool] | None = None,
    ) -> tuple[BM25Score[T], ...]:
        """按原始文档顺序返回全部分数，由领域适配器决定过滤、兜底与排序。

        ``term_filter`` 只裁剪**查询**词，索引不受影响：领域适配器据此决定哪些 token 算
        证据（例如长目标匹配剔除单字 CJK 虚词），内核本身不持有任何词表或语言策略。
        """
        query_tokens = tokenize(query)
        if term_filter is not None:
            query_tokens = tuple(term for term in query_tokens if term_filter(term))
        query_set = set(query_tokens)
        return tuple(
            BM25Score(
                item=item,
                score=self._score(query_tokens, index),
                overlap=len(query_set.intersection(self._term_frequencies[index])),
            )
            for index, item in enumerate(self.items)
        )

    def _idf_for(self, document_frequency: int) -> float:
        return math.log(
            1 + (len(self.items) - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def _idf(self, term: str) -> float:
        return self._idf_for(self._document_frequency.get(term, 0))

    def ideal_score(
        self,
        query: str,
        *,
        term_filter: Callable[[str], bool] | None = None,
    ) -> float:
        """一篇"理想文档"能拿到的分：每个查询词各命中一次，且长度恰为平均长度。

        用途是把绝对分数换算成**与语料规模无关**的相对分。BM25 的 idf 是
        ``log(1+(N-df+0.5)/(df+0.5))``，随语料规模剧烈变化：单个命中词在 N=1 时约 0.29、
        N=50 时约 3.5（12 倍）。于是任何写死的分数线都只在标定时的语料规模上成立——
        资源库一长大，阈值就悄悄失效，判定越来越松，且**不报错、只是慢慢变差**。

        用 ``score / ideal_score`` 做判据则分子分母同随 idf 缩放，比值稳定，含义也直白：
        "这篇文档覆盖了查询证据的多大比例"。

        代入 ``tf=1``、``dl=avgdl`` 时长度归一化项恰为 ``1+k1``，与 ``(k1+1)`` 约掉，
        故每个词的理想贡献就是它的 idf。

        查询词按**出现次数**计价，不去重：``_score`` 迭代的是未去重的查询 token，同一个词
        在查询里出现 q 次就累加 q 份贡献。分母若按唯一词求和，比值就随查询自身的用词重复度
        线性抬升（``alpha beta`` 0.94 → ``alpha alpha alpha beta`` 1.88），既超过 1、也不再是
        "覆盖比例"。受害最重的是"查询=Skill 全文"那个调用点（见 ``dispatch.resolver._assign``）：
        重复度天生最高，于是啰嗦的 SKILL.md 会系统性压过简洁的——和规模漂移一样，不报错，
        只是慢慢变差。分子分母同按重数计价后，比值重新有界且只反映文档的覆盖程度。

        语料里一次都没出现的词（``df=0``）**照常计入**——"查询大半内容本语料里没有"正是
        判定不相关的关键证据，把它们剔掉会让小语料下的相对分统统塌到 1.0 附近、噪声与真
        命中挤在一起（实测四条 Flow 上噪声 0.949 vs 真命中 1.070，区分度尽失）。

        但它们按 ``df=1`` 计价，不按真实的 ``df=0``：两者的增长速率不同（``df=0`` 是
        ``log(2N+2)``、``df=1`` 是 ``log(N/1.5)``），前者在 N 小时高出数倍、N 大时趋同，
        于是比值会随语料变大而系统性抬升——实测 N=1→51 时相对分从 0.16 漂到 0.34，
        规模无关性正是毁在这里。统一按 ``df=1`` 计价后，分母各项与分子同速缩放，
        稀释作用保留、漂移消除。
        """
        terms = tokenize(query)
        if term_filter is not None:
            terms = tuple(term for term in terms if term_filter(term))
        return sum(
            self._idf_for(max(self._document_frequency.get(term, 0), 1)) * count
            for term, count in Counter(terms).items()
        )

    def _score(self, query_tokens: tuple[str, ...], document_index: int) -> float:
        frequencies = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            # idf 走 ``_idf``，不在这里再写一遍公式：``ideal_score`` 要用同一份定义，
            # 两处各抄一份则任何调参都得改两处，漏一处就让相对分失去可比性。
            inverse_document_frequency = self._idf(term)
            length_normalization = frequency + self.k1 * (
                1 - self.b
                + self.b * document_length / max(self._average_document_length, 1.0)
            )
            score += inverse_document_frequency * frequency * (self.k1 + 1) / length_normalization
        return score

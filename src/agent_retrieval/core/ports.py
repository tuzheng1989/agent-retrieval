"""检索内核的存储与嵌入端口：Protocol + 确定性内存/替身实现。

端口只定义检索内核需要的最小面，不 import 平台资源模型——实现方（如 B7 的
Redis 快照与生产嵌入器）在端口外自行装配。
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    # 运行期不导入：core.ranking 反向依赖本模块（Embedder/QueryEmbeddingError），
    # 顶层导入会成环——注解经 ``from __future__ import annotations`` 惰性求值。
    from agent_retrieval.core.ranking import CandidateHit

T = TypeVar("T")


@runtime_checkable
class VectorStore(Protocol):
    """向量快照的 write-once 存储。

    ``publish`` 同一 ``snapshot_id`` 只允许一次：写入后快照不可变，重复发布抛
    ``ValueError``。读取面返回拷贝，调用方改返回值不影响存储内容。
    """

    def publish(
        self,
        snapshot_id: str,
        vectors: Mapping[str, Sequence[float]],
        meta: Mapping[str, str],
    ) -> None: ...

    def load(self, snapshot_id: str) -> dict[str, tuple[float, ...]] | None: ...

    def read_meta(self, snapshot_id: str) -> dict[str, str] | None: ...


@dataclass
class InMemoryVectorStore:
    """dict 存储的 :class:`VectorStore`，进程内生命周期。

    发布即转存不可变形态（向量 tuple、meta 浅拷贝 dict），此后拒绝对同一
    snapshot_id 的再次发布——write-once 语义是快照确定语义（R3）的存储侧前提。
    """

    _snapshots: dict[str, dict[str, tuple[float, ...]]]
    _meta: dict[str, dict[str, str]]

    def __init__(self) -> None:
        self._snapshots = {}
        self._meta = {}

    def publish(
        self,
        snapshot_id: str,
        vectors: Mapping[str, Sequence[float]],
        meta: Mapping[str, str],
    ) -> None:
        if snapshot_id in self._snapshots:
            raise ValueError(f"vector snapshot already published: {snapshot_id}")
        self._snapshots[snapshot_id] = {
            item_id: tuple(vector) for item_id, vector in vectors.items()
        }
        self._meta[snapshot_id] = dict(meta)

    def load(self, snapshot_id: str) -> dict[str, tuple[float, ...]] | None:
        """返回快照向量的拷贝；快照不存在返回 ``None``（缺失是正常路径，非异常）。"""
        snapshot = self._snapshots.get(snapshot_id)
        return dict(snapshot) if snapshot is not None else None

    def read_meta(self, snapshot_id: str) -> dict[str, str] | None:
        snapshot_meta = self._meta.get(snapshot_id)
        return dict(snapshot_meta) if snapshot_meta is not None else None


class QueryEmbeddingError(RuntimeError):
    """查询/语料嵌入失败——检索面按类型捕获并退回 BM25，不裸穿 requests 异常。"""


@runtime_checkable
class Embedder(Protocol):
    """文本嵌入端口：语料批量与查询单条两个入口。

    实现方在嵌入失败时应抛 :class:`QueryEmbeddingError`（而不是裸穿第三方异常）：
    融合检索面（``core.ranking``）按该类型捕获并自动退化为 BM25 单路。
    """

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


# ---------------------------------------------------------------------------
# Judge 端口与判停辅助（SDK P0 / B1）：判定面端口 + shortlist 判停消费。
# 窗口与判停门槛（τ）是宿主编排配置，不进端口（ADR-0001：端口只收数据不收调参）。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgedCandidate:
    """送入判定面的单个候选声明面。

    与宿主资源池（如 ``experiments/jev_cases.json`` 的资源条目）同构：id/name/
    description 是身份字段，use_when/avoid_when 是适用边界说明。五字段是候选的
    完整声明面——各 judge 实现把哪些字段投影进自己的判定 state 是实现自己的事，
    端口不做约束。
    """

    id: str
    name: str
    description: str
    use_when: str = ""
    avoid_when: str = ""


@dataclass(frozen=True)
class Judgment:
    """一次判定调用的结果：候选 id → 概率映射 + 调用元数据。

    ``probabilities`` 的键集必须与判定请求的候选 id 集一一对应（消费侧
    :func:`select_with_judge` 按此校验）。``input_tokens``/``output_tokens`` 可空——
    部分端点不计量 token 用量，缺席用 ``None`` 表达而非伪 0 值。
    """

    probabilities: dict[str, float]
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class Judge(Protocol):
    """判定端口：给定查询与候选声明面，返回逐候选概率判定。

    ``instruction`` 是**逐次调用参数**（ToolRet is_inst 协议把它与 query 并列传入
    判定 state），默认空串 = 无附加指令场景。判定窗口与判停门槛（τ）是宿主编排
    配置，不进端口（ADR-0001）。
    """

    def judge(
        self, query: str, candidates: Sequence[JudgedCandidate], *, instruction: str = ""
    ) -> Judgment: ...


def select_with_judge(
    hits: Sequence[CandidateHit[T]],
    judgment: Judgment,
    *,
    minimum_probability: float,
    item_id: Callable[[T], str],
) -> str | None:
    """消费已有 :class:`Judgment`，从 shortlist 选出最佳候选 id；门槛不足则弃权。

    契约前提是 ``judgment`` 已存在：exact 命中短路发生在调用方编排层（在调 judge
    之前先检查 ``hit.exact``，见 ``experiments/jev_retrieval.py`` 的编排模式）——本
    函数只消费判定结果，不触发判定。

    行为契约：

    - ``hits`` 为空返回 ``None``（空 shortlist 是正常路径，不抛）；
    - 候选 id 重复（如 ``[a, a]``）抛 ``ValueError``——重复检测在键集比较之前：纯
      集合比较对 ``[a, a]`` 与 ``[a]`` 无法区分；
    - ``judgment.probabilities`` 键集与候选 id 集不一致（缺键/多键）抛 ``ValueError``；
    - 任一概率不在 ``[0, 1]`` 抛 ``ValueError``（``not 0 <= p <= 1`` 对 NaN 恒真，
      NaN 天然按越界拦截）；
    - ``max()`` 并列时保留 ``hits`` 检索序（Python max 返回首个最大者）；
    - 最佳候选概率 ``< minimum_probability`` 返回 ``None``（abstain 弃权）。

    ``minimum_probability`` 无默认值（照 :func:`core.resource_index.clearly_related`
    的必填关键字风格）：判停门槛各调用点按查询形态各自标定，无默认等于不给跨调用点
    搬数值开口子。
    """
    if not hits:
        return None
    ids = [item_id(hit.item) for hit in hits]
    if len(ids) != len(set(ids)):
        raise ValueError("shortlist candidate ids must be unique")
    if set(judgment.probabilities) != set(ids):
        raise ValueError("judgment must return exactly one probability per candidate id")
    if any(not 0 <= probability <= 1 for probability in judgment.probabilities.values()):
        raise ValueError("judgment probabilities must be within [0, 1]")
    # Python max 并列保留检索序；门槛不足即弃权（abstain 是正常返回值，不是异常）。
    best = max(hits, key=lambda hit: judgment.probabilities[item_id(hit.item)])
    best_probability = judgment.probabilities[item_id(best.item)]
    if best_probability < minimum_probability:
        return None
    return item_id(best.item)


class MockEmbedder:
    """文本 hash → 确定性单位向量的真实替身（可运行实现，非打桩）。

    确定性来源是 ``hashlib.shake_256``：同文本任意两次调用逐位一致、跨进程一致，
    评测与测试因此可以离线复现（fusion-ranking §6）。归一化成单位向量是为了与
    真实嵌入器同语义——真实嵌入向量按惯例归一化，cosine 消费方在 Mock 与生产
    嵌入器之间行为一致。嵌入质量为零：只证明管线，不证明检索质量。
    """

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    def embed_corpus(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        digest = hashlib.shake_256((text or "").encode("utf-8")).digest(self._dimensions)
        vector = tuple(byte / 255.0 for byte in digest)
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)

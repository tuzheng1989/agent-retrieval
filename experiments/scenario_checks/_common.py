"""scenario_checks 共享装配面（SDK P2 / M3）——全部 8 格的装配唯一出处。

格脚本（``experiments/scenario_checks/<格 id>.py``）只做「格 id → 装配 → 执行 → 断言
→ exit」；语料加载、环境变量解析、快照发布/回读、Run 缓存检索、判定/docgen 参与全部
收在本模块（方案 M3：「共享装配工具函数放 _common.py，单一出处」）。

环境变量（包不读环境变量——解析发生在本宿主装配层，构造参数注入包）：

- ``REDIS_URL``（8 格必需）：Redis 连接串，如 ``redis://localhost:6379/0``；
- ``AGENT_RETRIEVAL_EMBEDDING_{BASE_URL,API_KEY,MODEL}``（8 格必需）：OpenAI 兼容
  嵌入网关，变量口径同 ``experiments/benchmarks/run_benchmark.py`` 的
  ``embedder_from_env``；
- ``AGENT_RETRIEVAL_E2E_{BASE_URL,API_KEY,MODEL}``（仅 full_stack 的判定/docgen 面，
  可选）：照 e2e 冒烟（``experiments/agent_e2e/agent.py``）同构变量名；缺任一 →
  判定面/docgen 走「未配置降级」，脚本输出明示，不影响 exit 0。

缺任一必需变量：打印缺失项 + 指路 + ``sys.exit(4)``（冒烟同构退出码语义 0/4）。

live 纪律：脚本不读 ``.env``（run_benchmark 的 ``load_env_file`` 会破坏 ``env -u``
清变量后缺配置路径的确定性）；M4 逐格真跑由主控以进程环境变量注入。运行期故障
（Redis 不可达、HTTP 失败、断言失败）以非零退出码显式失败，不假装通过
（「Redis 不可达的显式失败优于跳过」，方案 M3 评审要点）。

判定标准（S-D4，matrix.md §1.2）：组件存在性断言，非阈值——消费面非空、top-1 存在、
融合/快照/Run 缓存组件参与；top-1 是否命中 gold 只打印为观察项，不设质量断言
（质量红线归金版本 gate）。

用法（仓库根，逐格）::

    .venv/Scripts/python.exe -m experiments.scenario_checks.redis_snapshot_en_tool
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agent_retrieval import (
    CandidateHit,
    ChatJudge,
    ChatJudgeConfig,
    CorpusProjection,
    EmbeddingConfig,
    JudgedCandidate,
    RedisVectorStore,
    build_embedder,
    clear_run_cache,
    corpus_hash,
    make_backend,
    query_embedder,
    rank_candidates,
    select_with_judge,
    vector_snapshot_id,
)
from agent_retrieval.expansion import DocGenConfig, build_docgen
from agent_retrieval.run.cache import embedder_identity_digest
from experiments.agent_e2e.loading import load_catalog

EXIT_OK = 0
EXIT_CONFIG = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
_JEV_CASES_PATH = REPO_ROOT / "experiments" / "jev_cases.json"
_CATALOG_PATH = REPO_ROOT / "experiments" / "agent_e2e" / "catalogs" / "catalog-v1.json"

ENV_REDIS_URL = "REDIS_URL"
ENV_EMBED_BASE_URL = "AGENT_RETRIEVAL_EMBEDDING_BASE_URL"
ENV_EMBED_API_KEY = "AGENT_RETRIEVAL_EMBEDDING_API_KEY"
ENV_EMBED_MODEL = "AGENT_RETRIEVAL_EMBEDDING_MODEL"
REDIS_ENV_VARS = (ENV_REDIS_URL,)
EMBEDDING_ENV_VARS = (ENV_EMBED_BASE_URL, ENV_EMBED_API_KEY, ENV_EMBED_MODEL)

ENV_GATEWAY_BASE_URL = "AGENT_RETRIEVAL_E2E_BASE_URL"
ENV_GATEWAY_API_KEY = "AGENT_RETRIEVAL_E2E_API_KEY"
ENV_GATEWAY_MODEL = "AGENT_RETRIEVAL_E2E_MODEL"
GATEWAY_ENV_VARS = (ENV_GATEWAY_BASE_URL, ENV_GATEWAY_API_KEY, ENV_GATEWAY_MODEL)

#: 快照与 Run 缓存共用键前缀：scenario_checks 专属键域，不与应用数据混域。
KEY_PREFIX = "scenario_checks"
#: 快照与 Run 缓存的保留期（宿主侧常量，照 README §3 的 7 天口径）。
RETENTION_SECONDS = 7 * 24 * 3600
#: select_with_judge 的弃权门槛示例值：本脚本不判质量，仅为判定消费面走通取值。
MINIMUM_PROBABILITY = 0.5

#: 语料声明面投影：照 matrix.md §3.6 declared 口径（id+name+description；
#: tags/use_when/avoid_when 不进检索语料与判定输入）。
_DECLARED_PROJECTION = CorpusProjection(("id", "name", "description"))


@dataclass(frozen=True)
class GridItem:
    """格候选的统一形状：两语料源（jev_cases / catalog-v1）归一后的声明面条目。"""

    id: str
    name: str
    description: str
    kind: str


@dataclass(frozen=True)
class GridSpec:
    """格装配参数（matrix.md §3 主表逐格：语料指针 + 查询/gold 观察项）。"""

    #: 格 id（权威载体 matrix.md §3 主表；文档/脚本/模块名三方共用同一串）。
    grid_id: str
    #: 语料指针："jev"（experiments/jev_cases.json）| "catalog"（catalog-v1.json），
    #: 与 matrix.md §3.4/§3.5 主表的语料指针列逐格一致。
    corpus: str
    #: 宿主资源池版本标签（进快照 id 与 owner 指针）；jev_cases.json 无版本字段，
    #: 取脚本侧常量；catalog 格**硬编码 "catalog-v1"**（M3 审计 sug-1 契约注记：
    #: catalog 升版时须同步更新 SPECS 各行字面量，不会自动跟随文件实值）。
    owner_version: str
    #: 该格检索查询（宿主侧脚本常量；top-1 命中 gold 是观察项非断言）。
    query: str
    #: gold 资源 id（观察项：打印 top-1 对照，不设质量断言）。
    gold_id: str


SPECS: dict[str, GridSpec] = {
    spec.grid_id: spec
    for spec in (
        GridSpec("redis_snapshot_en_tool", "jev", "jev-cases-v1",
                 "Find population data for California in 2024", "geo-query"),
        GridSpec("redis_snapshot_en_skill", "catalog", "catalog-v1",
                 "Schedule meetings and notify attendees via calendar and email",
                 "meeting-notes-skill"),
        GridSpec("redis_snapshot_zh_tool", "catalog", "catalog-v1",
                 "按地理区域与年份查询人口统计数据", "geo-query"),
        GridSpec("redis_snapshot_zh_skill", "catalog", "catalog-v1",
                 "安排会议并向参会人发送通知", "meeting-notes-skill"),
        GridSpec("full_stack_en_tool", "jev", "jev-cases-v1",
                 "Find population data for California in 2024", "geo-query"),
        GridSpec("full_stack_en_skill", "catalog", "catalog-v1",
                 "Schedule meetings and notify attendees via calendar and email",
                 "meeting-notes-skill"),
        GridSpec("full_stack_zh_tool", "catalog", "catalog-v1",
                 "按地理区域与年份查询人口统计数据", "geo-query"),
        GridSpec("full_stack_zh_skill", "catalog", "catalog-v1",
                 "安排会议并向参会人发送通知", "meeting-notes-skill"),
    )
}


def require_env(*groups: tuple[str, ...]) -> None:
    """必需环境变量预检：缺任一打印缺失项 + 指路 + ``sys.exit(4)``（冒烟同构）。"""
    missing = [name for group in groups for name in group if not os.environ.get(name)]
    if not missing:
        return
    print(f"[exit {EXIT_CONFIG}] 缺少必需环境变量: {missing}", file=sys.stderr)
    print(
        f"指路: {ENV_REDIS_URL}=redis://localhost:6379/0（本机 Redis，如 "
        f"docker run -p 6379:6379 redis:7）；{ENV_EMBED_BASE_URL} / {ENV_EMBED_API_KEY}"
        f" / {ENV_EMBED_MODEL}=OpenAI 兼容嵌入网关（变量口径同 "
        f"experiments/benchmarks/run_benchmark.py embedder_from_env）。"
        f"格变量清单见 docs/scenario-matrix.md §3.4/§3.5。",
        file=sys.stderr,
    )
    sys.exit(EXIT_CONFIG)


def embedding_config_from_env() -> EmbeddingConfig:
    """嵌入配置（宿主解析环境变量后构造传参——包不读环境变量）。"""
    return EmbeddingConfig(
        kind="api",
        provider="zhipu",
        model_name=os.environ[ENV_EMBED_MODEL],
        # ApiEmbedder 拼 "{base}/embeddings"：宽容处理填了完整端点的常见写法（同
        # run_benchmark.embedder_from_env 的口径）。
        base_url=os.environ[ENV_EMBED_BASE_URL].removesuffix("/embeddings"),
        api_key=os.environ[ENV_EMBED_API_KEY],
    )


def _gateway_from_env() -> dict[str, str] | None:
    """full_stack 判定/docgen 网关（可选）：三变量齐 → 配置；否则 None（未配置降级）。"""
    values = {name: os.environ.get(name, "") for name in GATEWAY_ENV_VARS}
    if not all(values.values()):
        return None
    return {
        # ChatJudge/DocGenClient 自行拼接端点后缀：宽容处理填了完整端点的常见写法。
        "base_url": values[ENV_GATEWAY_BASE_URL].removesuffix("/chat/completions"),
        "api_key": values[ENV_GATEWAY_API_KEY],
        "model": values[ENV_GATEWAY_MODEL],
    }


def load_grid_items(spec: GridSpec) -> list[GridItem]:
    """按格语料指针加载候选池（matrix.md §3.6：en-tool=jev_cases；其余 7 格=catalog-v1）。"""
    if spec.corpus == "jev":
        rows = json.loads(_JEV_CASES_PATH.read_text(encoding="utf-8"))["resources"]
        return [GridItem(row["id"], row["name"], row["description"], "tool") for row in rows]
    catalog = load_catalog(_CATALOG_PATH)
    return [
        GridItem(resource.id, resource.name, resource.description, resource.kind)
        for resource in catalog.resources
    ]


def _corpus_texts(items: list[GridItem]) -> dict[str, str]:
    """快照语料映射 ``{kind}:{id}`` → 声明面文本（matrix.md §2.3 形态；声明面同语料）。"""
    text = _DECLARED_PROJECTION.to_callable()
    return {f"{item.kind}:{item.id}": text(item) for item in items}


async def _publish_and_readback(
    spec: GridSpec, config: EmbeddingConfig, items: list[GridItem],
) -> dict:
    """write-once 快照最小验证（matrix.md §2.3 装配片段）：预检/锁 → 发布或续期 → 回读。"""
    embedder = build_embedder(config)
    texts = _corpus_texts(items)
    item_keys = list(texts)
    snapshot_id = vector_snapshot_id(
        spec.owner_version, config.identity(), corpus_hash(texts),
    )
    store = RedisVectorStore(redis_url=os.environ[ENV_REDIS_URL], key_prefix=KEY_PREFIX)
    published: dict[str, tuple[float, ...]] | None = None
    try:
        if await store.read_meta(snapshot_id) is None and await store.acquire_lock(snapshot_id):
            # 锁内发布：全量嵌入一次 + write-once 写入（向量 + meta + owner 指针）。
            published = dict(zip(item_keys, embedder.embed_corpus(list(texts.values()))))
            await store.publish_vectors(
                snapshot_id=snapshot_id,
                owner_version=spec.owner_version,
                vectors=published,
                meta={
                    "provider": config.provider,
                    "dim": str(len(next(iter(published.values())))),
                },
                retention_seconds=RETENTION_SECONDS,
            )
            await store.release_lock(snapshot_id)
            branch = "publish"
        else:
            # 同 id 重发布 = write-once 预检命中：只续 TTL，零重嵌（matrix.md §2.3）。
            await store.extend_ttl(snapshot_id, spec.owner_version, RETENTION_SECONDS)
            branch = "extend_ttl"
        meta = await store.read_meta(snapshot_id)
        loaded = await store.load_snapshot_vectors(spec.owner_version, item_keys)
    finally:
        await store.aclose()
    return {
        "snapshot_id": snapshot_id,
        "branch": branch,
        "meta": meta,
        "loaded": loaded,
        "published": published,
        "item_keys": item_keys,
    }


def _snapshot_participation(snapshot: dict) -> dict[str, bool]:
    """快照面的 S-D4 组件存在性断言字段：meta 回读、键集一致、维度一致、发布回读一致。"""
    loaded: dict = snapshot["loaded"]
    dims = {len(vector) for vector in loaded.values()}
    published = snapshot["published"]
    return {
        "snapshot_meta_present": snapshot["meta"] is not None,
        "snapshot_readback_keys": set(loaded) == set(snapshot["item_keys"]),
        "snapshot_readback_dims": len(dims) == 1 and max(dims, default=0) > 0,
        # publish 分支验「发布 → 读回」逐向量一致（JSON float 往返精确）；
        # extend_ttl 分支（同语料同身份 → 同 snapshot_id）无需本地对照，键集/维度已验。
        "snapshot_publish_roundtrip": published is None or loaded == published,
    }


def run_redis_grid(grid_id: str) -> dict:
    """redis_snapshot 格装配执行（matrix.md §2.3）：嵌入 → 快照发布/续期 → 回读一致。"""
    spec = SPECS[grid_id]
    config = embedding_config_from_env()
    items = load_grid_items(spec)
    snapshot = asyncio.run(_publish_and_readback(spec, config, items))
    loaded = snapshot["loaded"]
    print(
        f"[{grid_id}] 快照 {snapshot['snapshot_id'][:12]}…（{snapshot['branch']}）"
        f" owner={spec.owner_version} 语料 {len(items)} 条"
        f" 回读 {len(loaded)}/{len(snapshot['item_keys'])} 键",
    )
    return {"grid_id": grid_id, "participation": _snapshot_participation(snapshot)}


def run_full_stack_grid(grid_id: str) -> dict:
    """full_stack 格装配执行（matrix.md §2.4）：§2.3 快照基础上 + Run 缓存检索
    + 判定面（ChatJudge 判检索 top-3 → select_with_judge 消费）+ docgen 返 None 语义。"""
    spec = SPECS[grid_id]
    config = embedding_config_from_env()
    items = load_grid_items(spec)
    snapshot = asyncio.run(_publish_and_readback(spec, config, items))

    # ① Run 级查询缓存（README §4）：查询向量每 Run 只嵌一次；首败冻结整 Run 向量路。
    backend = make_backend(redis_url=os.environ[ENV_REDIS_URL], key_prefix=KEY_PREFIX)
    run_id = f"scenario-{grid_id}"
    cached_query_embedder = query_embedder(
        embedder=build_embedder(config),
        run_id=run_id,
        backend=backend,
        embedder_identity=config.identity(),
    )
    hits = rank_candidates(
        items,
        spec.query,
        corpus_text=_DECLARED_PROJECTION.to_callable(),
        item_id=lambda item: item.id,
        item_name=lambda item: item.name,
        vector=cached_query_embedder,
    )
    run_cache_participated = cached_query_embedder is not None and backend.get_vector(
        run_id, embedder_identity_digest(config.identity()), "q", spec.query,
    ) is not None
    print(
        f"[{grid_id}] Run 缓存: run_id={run_id}"
        f" 查询向量{'已入' if run_cache_participated else '未入'}后端缓存（README §4 两级缓存）",
    )

    # ② 判定面 + ③ docgen（可选网关：缺配置明示降级，不影响 exit 0）。
    gateway = _gateway_from_env()
    _judge_face(grid_id, spec, hits, gateway)
    _docgen_face(grid_id, gateway)
    clear_run_cache()
    print(f"[{grid_id}] Run 收尾: clear_run_cache() 完成")
    return {
        "grid_id": grid_id,
        "gold_id": spec.gold_id,
        "hits": hits,
        "participation": {
            **_snapshot_participation(snapshot),
            "bm25_participated": any(hit.matched_terms for hit in hits),
            "vector_participated": any(hit.cosine is not None for hit in hits),
            "run_cache_participated": run_cache_participated,
        },
    }


def _judge_face(
    grid_id: str, spec: GridSpec, hits: list[CandidateHit[GridItem]],
    gateway: dict[str, str] | None,
) -> None:
    """判定面：网关配置 → ChatJudge 判检索 top-3，select_with_judge 消费；
    未配置 → 「未配置降级」明示输出（S-D4 口径：Judgment 产出或明示降级）。"""
    if gateway is None:
        missing = [name for name in GATEWAY_ENV_VARS if not os.environ.get(name)]
        print(f"[{grid_id}] 判定面: 未配置降级（缺 {missing}）→ 明示不判定")
        return
    judge = ChatJudge(ChatJudgeConfig(
        base_url=gateway["base_url"], api_key=gateway["api_key"], model_name=gateway["model"],
    ))
    shortlist = hits[:3]
    # 判定输入按 declared 口径只投影 name+description（matrix.md §3.6；use_when/
    # avoid_when 不进判定输入——JudgedCandidate 后两字段留缺省空串）。
    judgment = judge.judge(spec.query, [
        JudgedCandidate(id=item.id, name=item.name, description=item.description)
        for item in (hit.item for hit in shortlist)
    ])
    selected = select_with_judge(
        shortlist, judgment, minimum_probability=MINIMUM_PROBABILITY,
        item_id=lambda item: item.id,
    )
    probabilities = {key: round(value, 3) for key, value in judgment.probabilities.items()}
    print(
        f"[{grid_id}] 判定面: ChatJudge 判 top-{len(shortlist)}"
        f" → probabilities={probabilities} select_with_judge(selected={selected}，"
        f"门槛 {MINIMUM_PROBABILITY}，None=弃权)",
    )


def _docgen_face(grid_id: str, gateway: dict[str, str] | None) -> None:
    """docgen 面：build_docgen 返 None 语义验证——未配置 ≠ 故障，不强制真生成。"""
    config = DocGenConfig(
        base_url=gateway["base_url"] if gateway else "",
        api_key=gateway["api_key"] if gateway else "",
        model_name=gateway["model"] if gateway else "",
    )
    if build_docgen(config) is None:
        print(f"[{grid_id}] docgen: build_docgen → None（未配置 = 本部署未启用 docgen 的正常形态）")
    else:
        print(
            f"[{grid_id}] docgen: build_docgen → DocGenClient"
            f"（identity={config.identity()}，已配置，不强制真生成）",
        )


def check_grid(result: dict) -> None:
    """S-D4 断言面（matrix.md §1.2 组件存在性口径，非阈值）：消费面非空 + top-1 存在
    + 组件参与字段全真；gold 对照只打印为观察项（质量红线归金版本 gate）。"""
    grid_id = result["grid_id"]
    failed = [name for name, ok in result["participation"].items() if not ok]
    if failed:
        raise AssertionError(f"[{grid_id}] S-D4 组件参与断言失败: {failed}")
    hits = result.get("hits")
    if hits is not None:
        if not hits:
            raise AssertionError(f"[{grid_id}] S-D4 断言失败: 检索候选为空")
        top1 = getattr(hits[0].item, "id", "")
        if not top1:
            raise AssertionError(f"[{grid_id}] S-D4 断言失败: top-1 无 id")
        print(
            f"[{grid_id}] 检索: 候选 {len(hits)} 个，top-1 = {top1}"
            f"（gold = {result['gold_id']}，观察项不断言）",
        )
    print(f"[{grid_id}] 组件参与: {result['participation']}")
    print(f"[{grid_id}] PASS（S-D4 组件存在性断言全过）")

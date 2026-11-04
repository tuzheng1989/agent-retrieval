"""ChatJudge live 冒烟（SDK P1 / C2）：chat completions 端点判定管线 sanity。

live-only，CI 不跑。只验管线、输出形状与最弱语义合理性（D3：ChatJudge 判定
质量归 P2 scenario-matrix，本脚本的正例/负例断言是冒烟 sanity 而非质量结论）。

固定样本：任务 ``"Research the population of California in 2024, then email a
summary to Mei"``，候选取 ``experiments/jev_cases.json`` 资源池 4 项——正例
``geo-query``（查人口——步骤 1）与 ``email-send``（发摘要——步骤 2）；负例
``chart-render``（话题相关但任务无图表步骤）；干扰项 ``weather-forecast``（无
任务语义）。断言：``max(p(geo-query), p(email-send)) > p(chart-render)``、全部
概率 ∈ [0,1]、键集恰为 4 候选 id。

配置从环境变量读（experiments 是宿主，读环境变量是宿主权利；chat completions
网关回落链照 ``docgen.resolve_docgen_config`` 同构——judge 模型与生成/嵌入模型
常不在同一网关，judge 专属变量优先）：

- model:    ``AGENT_RETRIEVAL_CHAT_JUDGE_MODEL`` → ``AGENT_RETRIEVAL_DOCGEN_MODEL``
- base_url: ``AGENT_RETRIEVAL_CHAT_JUDGE_BASE_URL`` → ``AGENT_RETRIEVAL_DOCGEN_BASE_URL``
            → ``AGENT_RETRIEVAL_EMBEDDING_BASE_URL``（剥 /chat/completions 与
            /embeddings 尾巴——完整端点写法是常见配置形态）
- api_key:  ``AGENT_RETRIEVAL_CHAT_JUDGE_API_KEY`` → ``AGENT_RETRIEVAL_DOCGEN_API_KEY``
            → ``ZHIPUAI_API_KEY`` → ``AGENT_RETRIEVAL_EMBEDDING_API_KEY``

退出码：0 冒烟全过；1 语义 sanity 断言失败；2 输出形状断言失败（键集/越界）；
3 判定调用失败；4 缺配置。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agent_retrieval import ChatJudge, ChatJudgeConfig, JudgedCandidate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _REPO_ROOT / "experiments" / "jev_cases.json"

#: 冒烟任务文本（含两个明确步骤：查人口 → 发邮件摘要）。
_TASK = "Research the population of California in 2024, then email a summary to Mei"
#: 候选固定组合：正例 geo-query + email-send，负例 chart-render，干扰 weather-forecast。
_CANDIDATE_IDS = ("geo-query", "email-send", "chart-render", "weather-forecast")
_POSITIVE_IDS = ("geo-query", "email-send")
_NEGATIVE_ID = "chart-render"


def resolve_chat_judge_config() -> ChatJudgeConfig:
    """宿主配置解析：judge 专属变量优先，回落链照 docgen 同构（见模块 docstring）。"""
    model = (
        os.environ.get("AGENT_RETRIEVAL_CHAT_JUDGE_MODEL")
        or os.environ.get("AGENT_RETRIEVAL_DOCGEN_MODEL")
        or ""
    )
    base_url = (
        os.environ.get("AGENT_RETRIEVAL_CHAT_JUDGE_BASE_URL")
        or os.environ.get("AGENT_RETRIEVAL_DOCGEN_BASE_URL")
        or os.environ.get("AGENT_RETRIEVAL_EMBEDDING_BASE_URL")
        or ""
    ).removesuffix("/chat/completions").removesuffix("/embeddings")
    api_key = (
        os.environ.get("AGENT_RETRIEVAL_CHAT_JUDGE_API_KEY")
        or os.environ.get("AGENT_RETRIEVAL_DOCGEN_API_KEY")
        or os.environ.get("ZHIPUAI_API_KEY")
        or os.environ.get("AGENT_RETRIEVAL_EMBEDDING_API_KEY")
        or ""
    )
    if not (model and base_url and api_key):
        missing = [
            label
            for label, value in (("model", model), ("base_url", base_url), ("api_key", api_key))
            if not value
        ]
        raise RuntimeError(
            f"live 冒烟缺配置（{', '.join(missing)}）——按模块 docstring 的变量名设置"
            "（AGENT_RETRIEVAL_CHAT_JUDGE_* 或回落 DOCGEN_*/ZHIPUAI_API_KEY/*）",
        )
    return ChatJudgeConfig(base_url=base_url, api_key=api_key, model_name=model)


def load_smoke_candidates() -> list[JudgedCandidate]:
    """从 jev_cases.json 资源池取固定 4 项，转 JudgedCandidate 五字段。"""
    raw = json.loads(_DEFAULT_CASES.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in raw["resources"]}
    return [
        JudgedCandidate(
            id=entry["id"], name=entry["name"], description=entry["description"],
            use_when=entry.get("use_when", ""), avoid_when=entry.get("avoid_when", ""),
        )
        for entry in (by_id[candidate_id] for candidate_id in _CANDIDATE_IDS)
    ]


def main() -> int:
    try:
        config = resolve_chat_judge_config()
    except RuntimeError as exc:
        print(f"chat-judge smoke error: {exc}", file=sys.stderr)
        return 4

    candidates = load_smoke_candidates()
    print(f"[smoke] model={config.model_name} candidates={[c.id for c in candidates]}",
          file=sys.stderr)
    try:
        judgment = ChatJudge(config).judge(_TASK, candidates)
    except Exception as exc:  # noqa: BLE001 — live 失败如实上报，不为凑绿吞掉
        print(
            f"chat-judge smoke error: 判定调用失败 {type(exc).__name__}: {exc}"
            f"（先核对网关可达与 key/model 配置）",
            file=sys.stderr,
        )
        return 3

    # 输出形状断言：键集恰为 4 候选 id、全部概率 ∈ [0,1]。
    if set(judgment.probabilities) != set(_CANDIDATE_IDS):
        print(
            f"chat-judge smoke FAIL: 键集 {sorted(judgment.probabilities)} "
            f"!= 候选 id {sorted(_CANDIDATE_IDS)}",
            file=sys.stderr,
        )
        return 2
    out_of_range = {k: p for k, p in judgment.probabilities.items() if not 0 <= p <= 1}
    if out_of_range:
        print(f"chat-judge smoke FAIL: 概率越界 {out_of_range}", file=sys.stderr)
        return 2

    # 逐候选概率表到 stdout。
    print(f"[smoke] task: {_TASK}")
    print(f"[smoke] model: {judgment.model}  latency: {judgment.latency_seconds:.2f}s")
    for candidate in candidates:
        print(f"[smoke]   {candidate.id:<18} p={judgment.probabilities[candidate.id]:.3f}")

    # 最弱语义合理性：正例（任务步骤 1/2）至少一项高于负例（话题相关但无图表步骤）。
    positive_max = max(judgment.probabilities[candidate_id] for candidate_id in _POSITIVE_IDS)
    negative_p = judgment.probabilities[_NEGATIVE_ID]
    if not positive_max > negative_p:
        print(
            f"chat-judge smoke FAIL: max(p({_POSITIVE_IDS[0]}), p({_POSITIVE_IDS[1]}))="
            f"{positive_max:.3f} <= p({_NEGATIVE_ID})={negative_p:.3f}",
            file=sys.stderr,
        )
        return 1
    print("[smoke] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""B3 判定对拍：``judge_state``（SDK P0 端口构造）× jev_rerank 旧内联构造。

方案 B3 的双判定设计：

- **判定一（构造层语义保留，离线可复跑）**：同 case + 同候选下，``judge_state``
  输出与 ``jev_rerank.jev_scores`` 的旧内联构造（:71-95）逐字段对照。完整请求
  （state 全字段 + 每问题 instructions/criteria + 候选顺序）与响应键→id 映射分开
  记录；差异必须恰为已登记项——候选投影 ``tags`` 键删除一项（R3 投影定案；
  use_when/avoid_when 两侧都不在 state，无差异）。旧构造的完整输出冻结存
  ``judge-parity-legacy.json``（payload 与响应映射两节）。
- **判定二（传输层等价，live）**：SDK 臂（typesafe-sdk ``TypeSafeClient``）与
  JevJudge 臂（裸 HTTP）对同一 payload（``judge_state`` 生成，两臂唯一构造点）
  各判一遍。请求体对照取「捕获后 JSON 解析、结构逐字段一致」（SDK 序列化器不受
  我方控制，不做字节对照）——**硬门不变**；概率门 ``|Δp| ≤ 0.05``（主控裁决修订：
  System One 端点概率按 0.01 量化、采样跨档抖动，2026-11-01 实测同臂自比
  max|Δp| = 0.02、跨臂 max = 0.03——原 ``1e-9`` 判据在该非确定端点上结构性不可达
  （同臂自比即超差），0.05 = 量化步长 × 5 并覆盖实测噪声界）；超差打印两侧原始
  响应片段，并提示跑 ``--jitter-baseline`` 取与两臂实现无关的同臂噪声基线。

配置从 ``TYPESAFE_API_KEY`` / ``TYPESAFE_BASE_URL`` / ``TYPESAFE_DEFAULT_MODEL``
读（宿主纪律）；base_url/model 未设时回落 typesafe-sdk 内置默认（两臂同源）。
live-only，CI 不跑。

退出码：0 全过；2 判定一出现未登记差异；3 判定二超差或双臂调用失败；4 缺配置。

样本口径（R3-P1 / R4-P2）：资源池窗口须含非空 use_when/avoid_when 的候选（jev_cases
池八资源全带）；jev_cases 描述均为短句，超 2000 字符样本由 :func:`long_description`
在脚本内构造——同一文本同时适配旧 ``benchmarks.Resource`` 与新 ``JudgedCandidate``
两型（单一原始条目分叉两型），判定一断言两侧截断后的 description 逐位一致。
jev_cases 无 instruction 字段，另加一例 ``instruction="示例检索指令"`` 的合成 case
验证 instruction 通道。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_retrieval import JevJudge, JevJudgeConfig, JudgedCandidate
from agent_retrieval.judges import judge_state

from experiments.benchmarks.dataset import Resource
from experiments.benchmarks.jev_rerank import JEV_DESCRIPTION_MAX_CHARS
from experiments.benchmarks.run_benchmark import load_env_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = Path(__file__).resolve().parents[1] / "jev_cases.json"
_DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "judge-parity-report.json"
_DEFAULT_LEGACY = Path(__file__).resolve().parents[1] / "judge-parity-legacy.json"

#: 判定二概率对照容差（主控裁决修订，原方案 B3 的 1e-9 予以移除）：System One
#: 端点概率按 0.01 量化、采样跨档抖动——2026-11-01 实测同臂自比 max|Δp| = 0.02、
#: 跨臂 max = 0.03，1e-9 在该非确定端点上结构性不可达（同臂自比即超差）。
#: 0.05 = 量化步长 × 5，覆盖实测噪声界。
_PARITY_TOLERANCE = 0.05
#: live 判定的真实 case 数（另加一例合成 instruction case；合计 ≥ 3）。
_LIVE_CASES = 3
#: 已登记差异的路径形态：候选投影 tags 键删除（legacy 有、new 无）。
_REGISTERED_REMOVAL = re.compile(r"^state\.candidates\.c\d+\.tags$")

#: ToolRet 文档长尾的合成等价物：>2000 字符，供两侧截断语义对拍（R4-P2）。
_LONG_DESCRIPTION = "ToolRet documentation tail paragraph. " * 80


def long_description() -> str:
    """构造 >2000 字符的候选 description（同一文本适配两型的对拍探针）。"""
    return _LONG_DESCRIPTION


@dataclass(frozen=True)
class ParityCase:
    """一个判定对拍样本：task + instruction + 候选窗口（单一原始条目分叉两型）。"""

    case_id: str
    task: str
    instruction: str
    resources: list[Resource]
    candidates: list[JudgedCandidate]


def build_window(entries: list[dict]) -> tuple[list[Resource], list[JudgedCandidate]]:
    """资源池窗口：同一原始条目（jev_cases 形态）分叉旧 ``Resource`` 与新
    ``JudgedCandidate`` 两型——两型适配的单一事实来源，杜绝两侧候选漂移。"""
    resources = [
        Resource(id=entry["id"], name=entry["name"], description=entry["description"],
                 tags=entry.get("tags", ""))
        for entry in entries
    ]
    candidates = [
        JudgedCandidate(id=entry["id"], name=entry["name"], description=entry["description"],
                        use_when=entry.get("use_when", ""), avoid_when=entry.get("avoid_when", ""))
        for entry in entries
    ]
    return resources, candidates


def build_parity_cases(raw_cases: list[dict], resources: list[dict], window: int) -> list[ParityCase]:
    """组装对拍样本：真实 case × 窗口候选，另加一例非空 instruction 合成 case。"""
    # R4-P2：jev_cases 描述均为短句——追加一条构造的超长 description 候选（同文本
    # 两型适配），断言两侧 2000 字符截断逐位一致。
    probe = {
        "id": "long-doc", "name": "Long Doc", "description": long_description(), "tags": "docs",
    }
    entries = [*resources[: min(window, len(resources))], probe]
    window_resources, window_candidates = build_window(entries)
    cases = [
        ParityCase(case_id=entry["id"], task=entry["query"], instruction="",
                   resources=window_resources, candidates=window_candidates)
        for entry in raw_cases[:_LIVE_CASES]
    ]
    # jev_cases 无 instruction 字段：合成一例非空指令验证 retrieval_instruction 通道。
    cases.append(ParityCase(
        case_id="instruction-probe", task=raw_cases[0]["query"], instruction="示例检索指令",
        resources=window_resources, candidates=window_candidates,
    ))
    return cases


def legacy_judge_payload(task: str, retrieval_instruction: str, resources: list[Resource]) -> dict:
    """复刻 ``jev_rerank.jev_scores`` :71-96 的旧内联构造（不含 HTTP 调用）。

    冻结语义：字面量照抄旧实现（模板未在旧文件提常量），本函数是旧语义的存档——
    对拍通过后 jev_rerank 切换新端口，此处不再跟随变化。questions 的 ``Noul``
    对象照旧构造后经 ``model_dump()`` 转 wire dict（SDK 序列化的等价形态，B2 已
    核实），使两侧行同一 dict 基准对照、legacy.json 可直接落盘。
    """
    from typesafe_sdk import Noul

    state = {
        "task": task,
        "retrieval_instruction": retrieval_instruction,
        "candidates": {
            f"c{index}": {
                "name": resource.name,
                "description": resource.description[:JEV_DESCRIPTION_MAX_CHARS],
                "tags": resource.tags,
            }
            for index, resource in enumerate(resources)
        },
    }
    questions = {
        f"c{index}": dict(Noul(
            instructions=(
                f"Considering `task`, `retrieval_instruction`, and `candidates.c{index}`, "
                "is this resource suitable for at least one concrete step needed to "
                "fulfill the task? Judge its declared capability, not shared keywords."
            ),
            criteria={
                "true": "The resource can perform a specific operation needed for the task.",
                "false": "Only a related topic, a wrong operation, or insufficient stated capability.",
            },
        ).model_dump())
        for index in range(len(resources))
    }
    return {"state": state, "questions": questions}


def new_judge_payload(task: str, retrieval_instruction: str, candidates: list[JudgedCandidate]) -> dict:
    """新端口构造：``judge_state`` 是两臂唯一构造点（B2 收口事实）。"""
    return judge_state(task, retrieval_instruction, candidates)


def diff_payloads(legacy: dict, new: dict) -> list[dict]:
    """完整请求的字段级 diff：``{path, legacy, new}`` 清单（dict 键序无关）。

    对照面：state 全字段、每候选投影键集（含键的有无）、每问题 instructions 与
    criteria、候选顺序（由 c{index} 键位对位核验）。差异值以 ``repr`` 记录，长
    文本差异不会撑爆报告。
    """
    diffs: list[dict] = []

    def walk(path: str, legacy_value: Any, new_value: Any) -> None:
        if isinstance(legacy_value, dict) and isinstance(new_value, dict):
            for key in sorted(set(legacy_value) | set(new_value)):
                child = f"{path}.{key}" if path else key
                if key not in legacy_value:
                    diffs.append({"path": child, "legacy": None, "new": repr(new_value[key])})
                elif key not in new_value:
                    diffs.append({"path": child, "legacy": repr(legacy_value[key]), "new": None})
                else:
                    walk(child, legacy_value[key], new_value[key])
        elif legacy_value != new_value:
            diffs.append({"path": path, "legacy": repr(legacy_value), "new": repr(new_value)})

    walk("", legacy, new)
    return diffs


def registered_tag_removals(diffs: list[dict]) -> list[dict]:
    """已登记差异子集：候选投影 tags 键删除（legacy→new 方向的单键移除）。"""
    return [
        diff for diff in diffs
        if _REGISTERED_REMOVAL.match(diff["path"]) and diff["new"] is None
    ]


def unregistered_diffs(diffs: list[dict]) -> list[dict]:
    """未登记差异子集——判定一的 FAIL 清单（应恒为空）。"""
    registered_paths = {diff["path"] for diff in registered_tag_removals(diffs)}
    return [diff for diff in diffs if diff["path"] not in registered_paths]


def response_id_mapping(ids: list[str]) -> dict[str, str]:
    """响应 ``c{index}`` 键 → 候选 id 映射（候选顺序的显式记录，与 payload 分开存）。"""
    return {f"c{index}": item_id for index, item_id in enumerate(ids)}


def _run_judgment_one(cases: list[ParityCase]) -> tuple[bool, list[dict], list[dict]]:
    """逐 case 对照新旧构造；返回 (全过, 逐 case 摘要, 失败汇总)。"""
    case_reports: list[dict] = []
    failures: list[dict] = []
    for parity_case in cases:
        legacy = legacy_judge_payload(parity_case.task, parity_case.instruction, parity_case.resources)
        new = new_judge_payload(parity_case.task, parity_case.instruction, parity_case.candidates)
        diffs = diff_payloads(legacy, new)
        unregistered = unregistered_diffs(diffs)
        legacy_mapping = response_id_mapping([resource.id for resource in parity_case.resources])
        new_mapping = response_id_mapping([candidate.id for candidate in parity_case.candidates])
        mapping_equal = legacy_mapping == new_mapping
        # R4-P2 断言：超长 description 两侧截断逐位一致（diff 面之外显式记录）。
        long_index = len(parity_case.resources) - 1
        truncation_identical = (
            legacy["state"]["candidates"][f"c{long_index}"]["description"]
            == new["state"]["candidates"][f"c{long_index}"]["description"]
            == long_description()[:JEV_DESCRIPTION_MAX_CHARS]
        )
        case_ok = not unregistered and mapping_equal and truncation_identical
        case_reports.append({
            "case_id": parity_case.case_id,
            "n_candidates": len(parity_case.resources),
            "registered_diffs": registered_tag_removals(diffs),
            "unregistered_diffs": unregistered,
            "response_mapping_equal": mapping_equal,
            "long_description_truncation_identical": truncation_identical,
            "pass": case_ok,
        })
        if not case_ok:
            failures.append({
                "case_id": parity_case.case_id,
                "unregistered_diffs": unregistered,
                "response_mapping_equal": mapping_equal,
                "long_description_truncation_identical": truncation_identical,
            })
    return not failures, case_reports, failures


def _judge_config() -> JevJudgeConfig:
    """宿主配置解析：env 优先，回落 typesafe-sdk 内置默认（两臂同源端点与模型）。"""
    from typesafe_sdk.constants import DEFAULT_BASE_URL, DEFAULT_MODEL

    api_key = os.environ.get("TYPESAFE_API_KEY") or ""
    if not api_key:
        raise RuntimeError(
            "live 对拍缺环境变量 TYPESAFE_API_KEY——先 export TYPESAFE_API_KEY，"
            "或加 --skip-live 只跑离线判定一",
        )
    return JevJudgeConfig(
        base_url=os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL,
        api_key=api_key,
        model_name=os.environ.get("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
    )


def _run_sdk_arm(payload: dict, model_name: str) -> tuple[dict, dict]:
    """SDK 臂：显式传 model（不继承 client 默认——B2 审计提示），捕获实际请求体。

    经自建 ``httpx2.Client`` 实例传入 ``TypeSafeClient``，包装其 ``request`` 记录
    发送的 ``content``（bytes）后照常转发——SDK 序列化产物原样留证。
    """
    import httpx2
    from typesafe_sdk import TypeSafeClient

    captured: dict = {}
    http_client = httpx2.Client()
    original_request = http_client.request

    def capturing_request(method, url, **kwargs):
        captured["body"] = json.loads(kwargs["content"].decode("utf-8"))
        return original_request(method, url, **kwargs)

    http_client.request = capturing_request
    with TypeSafeClient(http_client=http_client) as client:
        response = client.system_one(
            state=payload["state"], questions=payload["questions"], model=model_name,
        )
    return captured["body"], {
        "answers": {key: float(answer.noul) for key, answer in response.answers.items()},
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


def _run_jev_arm(parity_case: ParityCase, config: JevJudgeConfig) -> tuple[dict, dict]:
    """JevJudge 臂（裸 HTTP）：包装 ``requests.post`` 捕获实际请求体后照常转发。"""
    import requests

    captured: dict = {}
    original_post = requests.post

    def capturing_post(url, headers=None, json=None, timeout=None):  # noqa: A002 — requests.post 形参名
        captured["body"] = json
        return original_post(url, headers=headers, json=json, timeout=timeout)

    requests.post = capturing_post
    try:
        judgment = JevJudge(config).judge(
            parity_case.task, parity_case.candidates, instruction=parity_case.instruction,
        )
    finally:
        requests.post = original_post
    return captured["body"], {
        "probabilities": dict(judgment.probabilities),
        "model": judgment.model,
        "usage": {"input_tokens": judgment.input_tokens, "output_tokens": judgment.output_tokens},
    }


def _run_judgment_two(cases: list[ParityCase], config: JevJudgeConfig) -> tuple[bool, list[dict]]:
    """逐 case 双臂对拍：请求体结构逐字段一致 + 概率 |Δp| ≤ 1e-9。"""
    case_reports: list[dict] = []
    all_ok = True
    for parity_case in cases:
        payload = new_judge_payload(parity_case.task, parity_case.instruction, parity_case.candidates)
        sdk_body, sdk_result = _run_sdk_arm(payload, config.model_name)
        jev_body, jev_result = _run_jev_arm(parity_case, config)
        # 请求体对照：两臂捕获体解析后逐字段相等（含 model 键）。
        bodies_equal = sdk_body == jev_body
        # 概率对照：SDK 臂 c{index} 键经响应映射对位候选 id，与 JevJudge 臂逐项求差。
        deltas = {
            candidate.id: abs(
                sdk_result["answers"][f"c{index}"] - jev_result["probabilities"][candidate.id],
            )
            for index, candidate in enumerate(parity_case.candidates)
        }
        max_delta = max(deltas.values(), default=0.0)
        within = max_delta <= _PARITY_TOLERANCE
        case_ok = bodies_equal and within
        all_ok = all_ok and case_ok
        report: dict = {
            "case_id": parity_case.case_id,
            "request_bodies_equal": bodies_equal,
            "max_abs_delta": max_delta,
            "deltas": deltas,
            "sdk_model": sdk_result["model"],
            "pass": case_ok,
        }
        if not case_ok:
            # 超差/结构不一致时才附两侧原始响应片段（验收标准：仅超差时打印）。
            report["sdk_raw"] = str(sdk_result)[:400]
            report["jev_raw"] = str(jev_result)[:400]
            report["sdk_request_body"] = str(sdk_body)[:400]
        case_reports.append(report)
        print(
            f"[parity] {parity_case.case_id}: bodies={'=' if bodies_equal else 'MISMATCH'} "
            f"max|Δp|={max_delta:.3e}",
            file=sys.stderr,
        )
    if not all_ok:
        print(
            "[parity] 概率超差可能源于端点采样非确定性（概率按 0.01 量化、跨档抖动）——"
            "跑 --jitter-baseline 可得与两臂实现无关的同臂噪声基线",
            file=sys.stderr,
        )
    return all_ok, case_reports


def _run_jitter_baseline(cases: list[ParityCase], config: JevJudgeConfig) -> list[dict]:
    """诊断模式：同一 payload 用 SDK 臂连跑两次，输出逐 case 概率自差表。

    端点采样非确定性会让同一请求两次返回不同概率——本表给出「与两臂实现无关」
    的噪声基线，供判定二超差归因。诊断不设门：只打印数字与对照容差的超差标记，
    退出码恒 0（噪声界修订是主控裁决事项，脚本不自行裁决）。
    """
    reports: list[dict] = []
    for parity_case in cases:
        payload = new_judge_payload(parity_case.task, parity_case.instruction, parity_case.candidates)
        _body_one, first = _run_sdk_arm(payload, config.model_name)
        _body_two, second = _run_sdk_arm(payload, config.model_name)
        self_deltas = {
            f"c{index}": abs(first["answers"][f"c{index}"] - second["answers"][f"c{index}"])
            for index in range(len(parity_case.candidates))
        }
        max_delta = max(self_deltas.values(), default=0.0)
        over = {key: value for key, value in self_deltas.items() if value > _PARITY_TOLERANCE}
        reports.append({
            "case_id": parity_case.case_id,
            "self_deltas": self_deltas,
            "max_self_delta": max_delta,
            "over_tolerance": over,
        })
        print(
            f"[jitter] {parity_case.case_id}: max self|Δp|={max_delta:.3g} "
            f"over({len(over)}/{len(self_deltas)})={over or '{}'}",
            file=sys.stderr,
        )
    return reports


def run(args: argparse.Namespace) -> int:
    raw = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = build_parity_cases(raw["cases"], raw["resources"], args.window)
    print(f"[parity] {len(cases)} cases × {len(cases[0].resources)} candidates", file=sys.stderr)

    # ---- 判定一（离线，每次运行都跑） ----------------------------------------
    one_ok, case_reports, failures = _run_judgment_one(cases)
    # 冻结参照：旧构造的完整输出存两节（payload + 响应映射），供审计与离线复跑比对。
    legacy_snapshot = {
        "payload": [
            {
                "case_id": parity_case.case_id,
                "request": legacy_judge_payload(
                    parity_case.task, parity_case.instruction, parity_case.resources,
                ),
                "response_mapping": response_id_mapping(
                    [resource.id for resource in parity_case.resources],
                ),
            }
            for parity_case in cases
        ],
    }
    args.legacy.write_text(
        json.dumps(legacy_snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"[parity] legacy snapshot -> {args.legacy}", file=sys.stderr)

    report: dict = {
        "verdict_one": {"pass": one_ok, "cases": case_reports, "failures": failures},
    }

    exit_code = 0
    if not one_ok:
        print("[parity] 判定一 FAIL：出现未登记差异", file=sys.stderr)
        for failure in failures:
            print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        exit_code = 2

    # ---- 判定二（live） ------------------------------------------------------
    if args.skip_live:
        report["verdict_two"] = {"pass": None, "note": "skipped (--skip-live)"}
    elif args.jitter_baseline:
        # 缺 key 同完整 live：上抛 main → 退出码 4（诊断也需要真实端点）。
        config = _judge_config()
        try:
            jitter_reports = _run_jitter_baseline(cases, config)
        except Exception as exc:  # noqa: BLE001 — 诊断失败如实上报
            report["verdict_two"] = {
                "pass": None, "jitter_baseline": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[parity] jitter 基线失败 {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 3
        else:
            report["verdict_two"] = {
                "pass": None, "note": "jitter-baseline diagnostic (no gate)",
                "tolerance": _PARITY_TOLERANCE, "cases": jitter_reports,
            }
    else:
        # 缺 key 的 RuntimeError 不进下方 try：让它上抛 main → 退出码 4（指路语义，
        # 与双臂调用失败的 3 区分）。
        config = _judge_config()
        try:
            two_ok, two_reports = _run_judgment_two(cases, config)
        except Exception as exc:  # noqa: BLE001 — live 失败如实上报，不为凑绿吞掉
            report["verdict_two"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[parity] 判定二 FAIL：双臂调用失败 {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 3
        else:
            report["verdict_two"] = {
                "pass": two_ok, "tolerance": _PARITY_TOLERANCE, "cases": two_reports,
            }
            if not two_ok:
                print("[parity] 判定二 FAIL：请求体不一致或概率超差", file=sys.stderr)
                exit_code = 3

    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"[parity] report -> {args.report}", file=sys.stderr)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="判定二口径（主控裁决修订）：请求体捕获后 JSON 解析逐字段一致（硬门）"
               "+ |Δp| ≤ 0.05——端点概率按 0.01 量化、采样跨档抖动，实测同臂自比 "
               "≤0.02 / 跨臂 ≤0.03，0.05 = 量化步长 5 倍并覆盖实测噪声界。",
    )
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES,
                        help="对拍样本源（默认 experiments/jev_cases.json）")
    parser.add_argument("--window", type=int, default=10,
                        help="资源池窗口大小（默认 10；池小于窗口时取全池）")
    live_group = parser.add_mutually_exclusive_group()
    live_group.add_argument("--skip-live", action="store_true",
                            help="只跑判定一（离线构造层对照），不触发双臂 live 调用")
    live_group.add_argument("--jitter-baseline", action="store_true",
                            help="诊断模式：同一 payload 用 SDK 臂连跑两次，输出逐 case "
                                 "概率自差表（与两臂实现无关的端点噪声基线；不设门，exit 0）")
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--legacy", type=Path, default=_DEFAULT_LEGACY)
    args = parser.parse_args()
    load_env_file()
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"judge-parity error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

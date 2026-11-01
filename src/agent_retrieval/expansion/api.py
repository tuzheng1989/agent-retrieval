"""doc2query 语料扩展客户端：OpenAI 兼容端点的单轮文本生成（extras: ``docgen``）。

HTTP 纪律（批量离线生成共用，平移自 experiments 批生成器）：temperature 0.2；
思考显式关闭（GLM-4.5/5.3 系的 reasoning token 计入 max_tokens，会把正文挤空）；
2,000 token 预算；除鉴权失败（401/403）外一切 HTTP 错误退避重试——网关把瞬时限流
也报成普通 4xx，长批处理依赖重试兜底。

与向量快照 ``corpus_hash`` 的连接口径：``redis_store`` 只哈希传入的语料文本——
**生成后的语料文本实际变化时才产生新哈希**（配置变化不一定改变生成输出；
``identity()`` 本身未接入该哈希），嵌入层按内容寻址自然重嵌。
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

_ResourceT = TypeVar("_ResourceT")


class DocGenError(RuntimeError):
    """doc2query 生成调用失败——批处理按类型捕获并落盘已成功部分。"""


@dataclass(frozen=True)
class DocGenConfig:
    """单个生成模型的寻址与凭据（宿主配置体系的解析结果）。

    ``identity()`` 是生成组件身份三元组，供宿主记录与展示；注意它未接入语料
    ``corpus_hash``——哈希只看语料文本内容，见模块 docstring 的连接口径。
    """

    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    #: 端点风格：``chat``（``/chat/completions``）或 ``responses``（``/responses``）。
    api_style: Literal["chat", "responses"] = "chat"
    #: ``generate`` 缺省每资源生成的问题数。
    questions_per_resource: int = 5
    timeout_seconds: float = 60.0

    def identity(self) -> tuple[str, str, int]:
        """生成组件身份三元组（model、api 风格、每资源问题数）。"""
        return (self.model_name, self.api_style, self.questions_per_resource)


PROMPT_TEMPLATE = """You generate realistic user queries for a retrieval index.

Resource:
Name: {name}
Description: {description}
Tags: {tags}

Write {k} distinct questions a user might naturally ask that this resource could
answer. Vary wording, intent and vocabulary — synonyms, colloquial phrasings,
different entry angles; do not reuse the resource's own words verbatim in every
query. English only. One query per line, no numbering, no explanations."""


def _chat_content(body: dict) -> str:
    """chat 响应取正文。"""
    return body["choices"][0]["message"]["content"]


def _responses_content(body: dict) -> str:
    """responses 响应取正文：遍历 output[].content[].text 拼接。"""
    texts = [
        block["text"]
        for item in body.get("output", [])
        for block in item.get("content", [])
        if "text" in block
    ]
    return "\n".join(texts)


class DocGenClient:
    """OpenAI 兼容端点的单轮文本生成客户端。

    ``complete`` 是通用单轮传输基元（自由文本生成，提示词由调用方组装——
    分组描述、文档摘要等分析工具走它）；``generate`` 是 doc2query 专用封装
    （模板 + 基元 + 解析），叠在基元之上。
    """

    def __init__(self, config: DocGenConfig) -> None:
        self._config = config

    def complete(self, prompt: str) -> str:
        """单轮补全：发送 prompt，返回模型文本（不解析、不裁剪）。

        退避序列 (0, 2.0, 5.0, 10.0)；401/403 鉴权失败不重试，其余 4xx 按瞬时
        故障记入 last_error 重试。重试耗尽抛 :class:`DocGenError`。
        """
        config = self._config
        # requests 是 docgen extra 的依赖：下沉到调用点惰性导入，让零依赖安装
        # （纯 core 用法）在 import 期不受影响，缺依赖时给出指明 extras 的报错。
        try:
            import requests  # noqa: PLC0415 — 与 ApiEmbedder 同模式
        except ImportError as exc:
            raise DocGenError(
                "DocGenClient 缺依赖 requests；先安装 agent-retrieval[docgen]",
            ) from exc
        if config.api_style == "responses":
            url, payload, extract = self._responses_call(prompt)
        else:
            url, payload, extract = self._chat_call(prompt)
        # 网关把瞬时限流/上游故障也报成 4xx（rerank 实测教训），除鉴权失败外都退避重试。
        last_error: Exception | None = None
        for attempt, pause in enumerate((0, 2.0, 5.0, 10.0)):
            if pause:
                time.sleep(pause)
            try:
                response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json=payload,
                    timeout=config.timeout_seconds,
                )
                if response.status_code in (401, 403):
                    raise DocGenError(
                        f"docgen 鉴权失败 (HTTP {response.status_code})，"
                        f"不重试: {response.text[:120]}",
                    )
                if response.status_code >= 400:
                    # 非鉴权 4xx 按瞬时故障处理：记入 last_error 进入退避重试。
                    last_error = DocGenError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                    )
                    continue
                return extract(response.json())
            except DocGenError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
        raise DocGenError(f"docgen 调用失败（已重试 3 次）: {last_error}")

    def generate(
        self,
        resource_name: str,
        resource_description: str,
        resource_tags: str,
        k: int | None = None,
    ) -> list[str]:
        """doc2query：一个资源 → 去重的用户查询列表（不足 k 条按模型产出接受）。"""
        count = self._config.questions_per_resource if k is None else k
        prompt = PROMPT_TEMPLATE.format(
            name=resource_name, description=resource_description, tags=resource_tags, k=count,
        )
        return parse_generated_queries(self.complete(prompt), count)

    def _chat_call(self, prompt: str) -> tuple[str, dict, Callable[[dict], str]]:
        """chat 分支：``/chat/completions`` 端点，thinking 显式关闭。"""
        payload = {
            "model": self._config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            # GLM-4.5 系是推理模型：thinking token 计入 max_tokens，会
            # 把 content 挤空（finish_reason=length）——显式关闭后稳定。
            "thinking": {"type": "disabled"},
            "max_tokens": 2000,
            "stream": False,
        }
        return f"{self._config.base_url}/chat/completions", payload, _chat_content

    def _responses_call(self, prompt: str) -> tuple[str, dict, Callable[[dict], str]]:
        """responses 分支：``/responses`` 端点，reasoning minimal 为对应的关思考参数。"""
        payload = {
            "model": self._config.model_name,
            "input": prompt,
            "temperature": 0.2,
            "reasoning": {"effort": "minimal"},
            "max_output_tokens": 2000,
            "stream": False,
        }
        return f"{self._config.base_url}/responses", payload, _responses_content


def build_docgen(config: DocGenConfig) -> DocGenClient | None:
    """按配置构造生成客户端；``None`` 表示「本部署未启用 docgen」的正常形态。

    缺 base_url 或 api_key 返 ``None``（未配置 ≠ 故障），语义对齐
    ``embedders.factory.build_embedder``。
    """
    if not (config.base_url and config.api_key):
        return None
    return DocGenClient(config)


def parse_generated_queries(text: str, k: int) -> list[str]:
    """Split model output into k distinct queries: one per line, numbering stripped.

    Lines that collapse to duplicates or empties are dropped; fewer than k
    distinct lines is accepted as-is (the model's honest yield beats padding).
    """
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. )\t")
        if not cleaned:
            continue
        if cleaned.lower() not in {q.lower() for q in queries}:
            queries.append(cleaned)
    return queries[:k]


def docgen_corpus_text(base_corpus_text: Callable[[_ResourceT], str],
                       expansions: dict[str, list[str]]) -> Callable[[_ResourceT], str]:
    """Wrap a corpus projection so expansion queries ride on the retrieval face.

    Resources without generated queries fall back to the base projection —
    partial generation never blocks evaluation.
    """
    def corpus_text(resource: object) -> str:
        extra = expansions.get(resource.id, [])
        return " ".join((base_corpus_text(resource), *extra)) if extra else base_corpus_text(resource)
    return corpus_text

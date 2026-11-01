"""agent_retrieval.expansion 包面单元测试（替身 HTTP；B4 docgen 收编）。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent_retrieval.expansion import (
    DocGenClient,
    DocGenConfig,
    DocGenError,
    build_docgen,
    docgen_corpus_text,
    parse_generated_queries,
)


class _FakeResponse:
    """requests.post 替身的最小响应面（批生成纪律下只消费 json/status_code/text）。"""

    def __init__(self, body: dict) -> None:
        self._body = body
        self.status_code = 200
        self.text = ""

    def json(self) -> dict:
        return self._body


@pytest.fixture
def captured(monkeypatch):
    """把 requests.post 换成记录器；install(body) 后每次调用记录请求并回放 body。"""
    store: dict = {}

    def install(body: dict) -> None:
        def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ARG001 — 替身接口形状
            store["url"] = url
            store["headers"] = headers
            store["payload"] = json
            store["timeout"] = timeout
            return _FakeResponse(body)

        import requests

        monkeypatch.setattr(requests, "post", fake_post)

    return store, install


class TestDocGenConfig:
    def test_identity_triple_and_changes(self):
        config = DocGenConfig(base_url="u", api_key="k", model_name="m")
        assert config.identity() == ("m", "chat", 5)
        assert DocGenConfig(
            base_url="u", api_key="k", model_name="m", api_style="responses",
        ).identity() == ("m", "responses", 5)
        assert DocGenConfig(
            base_url="u", api_key="k", model_name="m", questions_per_resource=7,
        ).identity() == ("m", "chat", 7)
        assert DocGenConfig(base_url="u", api_key="k", model_name="other").identity() != (
            "m",
            "chat",
            5,
        )

    def test_frozen(self):
        config = DocGenConfig(base_url="u", api_key="k", model_name="m")
        with pytest.raises(FrozenInstanceError):
            config.model_name = "other"


class TestBuildDocgen:
    def test_missing_credentials_return_none(self):
        assert build_docgen(DocGenConfig(base_url="", api_key="k", model_name="m")) is None
        assert build_docgen(DocGenConfig(base_url="u", api_key="", model_name="m")) is None

    def test_full_credentials_return_client(self):
        client = build_docgen(DocGenConfig(base_url="u", api_key="k", model_name="m"))
        assert isinstance(client, DocGenClient)


class TestChatBranch:
    def test_complete_payload_and_free_text(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": "自由文本第一行\n第二行"}}]})
        client = DocGenClient(
            DocGenConfig(base_url="http://gw", api_key="secret", model_name="glm-x"),
        )
        assert client.complete("任意提示词") == "自由文本第一行\n第二行"
        assert store["url"] == "http://gw/chat/completions"
        assert store["headers"] == {"Authorization": "Bearer secret"}
        assert store["payload"]["model"] == "glm-x"
        assert store["payload"]["messages"] == [{"role": "user", "content": "任意提示词"}]
        assert store["payload"]["thinking"] == {"type": "disabled"}
        assert store["payload"]["max_tokens"] == 2000
        assert store["payload"]["temperature"] == 0.2
        assert store["timeout"] == 60.0


class TestResponsesBranch:
    def test_payload_and_output_concat(self, captured):
        store, install = captured
        install({"output": [
            {"content": [{"type": "output_text", "text": "第一段\n续行"}]},
            {"content": [{"type": "output_text", "text": "第二段"}]},
        ]})
        client = DocGenClient(DocGenConfig(
            base_url="http://gw", api_key="secret", model_name="glm-x",
            api_style="responses",
        ))
        assert client.complete("任意提示词") == "第一段\n续行\n第二段"
        assert store["url"] == "http://gw/responses"
        assert store["payload"]["model"] == "glm-x"
        assert store["payload"]["input"] == "任意提示词"
        assert store["payload"]["reasoning"] == {"effort": "minimal"}
        assert store["payload"]["max_output_tokens"] == 2000
        assert store["payload"]["temperature"] == 0.2
        # chat 专属键不得漏进 responses 分支。
        assert "thinking" not in store["payload"]
        assert "messages" not in store["payload"]
        assert "max_tokens" not in store["payload"]


class TestGenerate:
    def test_parses_dedupes_caps_and_prompt_carries_resource(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": (
            "1. find a joke by its ID\n"
            "2. look up a specific joke\n"
            "3. FIND A JOKE BY ITS ID\n"
            "\n"
            "4. get joke number 123\n"
            "   5. get joke number 123\n"
            "6. one more beyond the cap\n"
        )}}]})
        client = DocGenClient(DocGenConfig(base_url="u", api_key="k", model_name="m"))
        queries = client.generate("Joke API", "serves jokes", "humor", k=3)
        assert queries == [
            "find a joke by its ID",
            "look up a specific joke",
            "get joke number 123",
        ]
        prompt = store["payload"]["messages"][0]["content"]
        assert "Name: Joke API" in prompt
        assert "Description: serves jokes" in prompt
        assert "Tags: humor" in prompt
        assert "Write 3 distinct questions" in prompt

    def test_k_defaults_to_config(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": "q1\nq2"}}]})
        client = DocGenClient(DocGenConfig(
            base_url="u", api_key="k", model_name="m", questions_per_resource=2,
        ))
        assert client.generate("n", "d", "t") == ["q1", "q2"]
        assert "Write 2 distinct questions" in store["payload"]["messages"][0]["content"]


class TestPureFunctions:
    def test_parse_strips_numbering_and_dedupes(self):
        text = (
            "1. find a joke by its ID\n"
            "2. look up a specific joke\n"
            "3. FIND A JOKE BY ITS ID\n"   # 大小写去重后与第 1 条重复
            "\n"
            "4. get joke number 123\n"
            "   5. get joke number 123\n"  # 重复
        )
        assert parse_generated_queries(text, 5) == [
            "find a joke by its ID",
            "look up a specific joke",
            "get joke number 123",
        ]

    def test_corpus_text_appends_expansion_and_falls_back(self):
        base = lambda r: f"{r.name} {r.description}"  # noqa: E731 — 测试内一行投影
        expansions = {"r1": ["how do I query data", "lookup records"]}
        r1 = SimpleNamespace(id="r1", name="Geo", description="population data")
        r2 = SimpleNamespace(id="r2", name="Doc", description="documents")
        corpus = docgen_corpus_text(base, expansions)
        assert corpus(r1) == "Geo population data how do I query data lookup records"
        assert corpus(r2) == base(r2)  # 无扩展的资源回退原投影


class TestShellCompat:
    """薄壳兼容入口：原签名含原参数名的关键字调用（main() 零改动的前提）。"""

    def test_generate_queries_keyword_call(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": (
            "1. how do I query data\n2. lookup records\n2. lookup records\n"
        )}}]})
        from experiments.benchmarks.docgen import generate_queries

        queries = generate_queries(
            "Geo Data", "population statistics", "census",
            base_url="http://gw", api_key="secret", model="glm-x", k=2,
        )
        assert queries == ["how do I query data", "lookup records"]
        assert store["url"] == "http://gw/chat/completions"
        assert store["payload"]["model"] == "glm-x"

    def test_chat_text_roundtrip(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": "one sentence."}}]})
        from experiments.benchmarks.docgen import DocGenConfig as ShellDocGenConfig
        from experiments.benchmarks.docgen import chat_text

        config = ShellDocGenConfig(
            base_url="http://gw", api_key="secret", model_name="glm-x",
        )
        assert chat_text(config, "describe the category") == "one sentence."
        assert store["payload"]["messages"][0]["content"] == "describe the category"

    def test_shell_reexports(self):
        # 既有调用方与 tests/test_docgen.py 依赖的薄壳转发名。
        from experiments.benchmarks.docgen import (  # noqa: F401 — 导入面本身即断言
            DocGenError,
            build_docgen,
            docgen_corpus_text,
            parse_generated_queries,
        )

    def test_shell_error_is_package_error(self):
        from experiments.benchmarks.docgen import DocGenError as ShellDocGenError

        assert ShellDocGenError is DocGenError


class TestConsumerGenerators:
    """groupdesc/summarize 生成函数替身直调（方案 B4 验收）：提示词出自自身文件
    且逐位一致、返回自由文本。"""

    def test_groupdesc_generate_description(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": "Route here for joke tasks."}}]})
        from experiments.benchmarks.groupdesc import PROMPT_TEMPLATE, generate_description

        description = generate_description(
            "jokes", ["Joke API", "Humor DB"], ["serves jokes", "stores humor"],
            base_url="http://gw", api_key="secret", model="glm-x",
        )
        assert description == "Route here for joke tasks."
        expected = PROMPT_TEMPLATE.format(
            group="jokes",
            names="Joke API, Humor DB",
            descriptions="serves jokes | stores humor",
        )
        assert store["payload"]["messages"][0]["content"] == expected

    def test_summarize_summarize_resource(self, captured):
        store, install = captured
        install({"choices": [{"message": {"content": "The tool serves jokes."}}]})
        from experiments.benchmarks.summarize import PROMPT_TEMPLATE, summarize_resource

        summary = summarize_resource(
            "Joke API", '{"endpoints": ["/joke"]}',
            base_url="http://gw", api_key="secret", model="glm-x",
        )
        assert summary == "The tool serves jokes."
        expected = PROMPT_TEMPLATE.format(
            name="Joke API", documentation='{"endpoints": ["/joke"]}', target_chars=1200,
        )
        assert store["payload"]["messages"][0]["content"] == expected

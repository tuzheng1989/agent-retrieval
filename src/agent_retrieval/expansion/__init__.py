"""doc2query 语料扩展组件：LLM 生成用户查询附加进检索语料（extras: ``docgen``）。"""

from agent_retrieval.expansion.api import (
    DocGenClient,
    DocGenConfig,
    DocGenError,
    build_docgen,
    docgen_corpus_text,
    parse_generated_queries,
)

__all__ = [
    "DocGenClient",
    "DocGenConfig",
    "DocGenError",
    "build_docgen",
    "docgen_corpus_text",
    "parse_generated_queries",
]

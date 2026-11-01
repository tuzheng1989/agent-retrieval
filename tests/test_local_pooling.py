"""LocalEmbedder 三 pooling 数学与双前缀的契约测试（替身 session，不加载真模型）。

- pooling 数学：同一 last_hidden + attention_mask，mean/cls/last_token 输出确定、
  互异、L2 归一化；
- padding 方向陷阱（B6 评审）：pad 在前与 pad 在后两种 mask，cls 与 last_token
  都必须取到同一内容 token——左填充时 ``[:, 0]`` 取到 pad、右填充时 ``[:, -1]``
  取到 pad， naive 索引在对应方向必错；
- 前缀拼接：embed_query 带 query_prefix、embed_corpus 带 corpus_prefix（替身
  tokenizer 断言收到的文本）；
- 非法 pooling 构造即 ValueError；默认（mean + 空前缀）输出与改动前逐位一致
  （期望字面量 = 改动前探针现值）。

onnxruntime/transformers 经 sys.modules 注入替身（tests/test_api_embedder.py 同款
模式），numpy 用真实库——pooling 数学本身是 numpy 代码，替身只会掩盖真问题。
"""

import sys
import types

import numpy as np
import pytest

from agent_retrieval.embedders.local import LocalEmbedder

# 内容 token 向量：三个互异单位向量 + 一个 pad 槽位向量（pooling 数学必须跳过它）。
_C0 = [1.0, 0.0, 0.0]
_C1 = [0.0, 1.0, 0.0]
_C2 = [0.0, 0.0, 1.0]
_PAD = [9.0, 9.0, 9.0]


class _FakeTokenizer:
    """替身 tokenizer：按预设 mask/ids 编码，记录收到的文本供前缀断言。"""

    def __init__(self, attention_mask, input_ids=None):
        self._attention_mask = np.asarray(attention_mask, dtype=np.int64)
        self._input_ids = np.asarray(
            input_ids if input_ids is not None
            else np.ones_like(self._attention_mask),
            dtype=np.int64,
        )
        self.received_texts: list[str] = []

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="np"):
        self.received_texts = list(texts)
        # 真实 tokenizer 按当前批动态 padding：预设数据按本批行数裁剪（单条查询取首行）。
        batch = len(texts)
        return {
            "input_ids": self._input_ids[:batch],
            "attention_mask": self._attention_mask[:batch],
        }


class _FakeSession:
    """替身 ONNX session：恒返预设 last_hidden，按本批行数裁剪（batch 与输入一致）。"""

    def __init__(self, last_hidden):
        self._last_hidden = np.asarray(last_hidden, dtype=np.float64)

    def get_inputs(self):
        return [
            types.SimpleNamespace(name="input_ids"),
            types.SimpleNamespace(name="attention_mask"),
        ]

    def run(self, output_names, feed):
        batch = feed["attention_mask"].shape[0]
        return [self._last_hidden[:batch]]


def _install_fake_stack(monkeypatch, *, attention_mask, last_hidden):
    """注入假 onnxruntime/transformers 模块；返回替身 tokenizer（供前缀断言）。"""
    tokenizer = _FakeTokenizer(attention_mask)
    session = _FakeSession(last_hidden)
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: session
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    return tokenizer


def _make_embedder(monkeypatch, *, attention_mask, last_hidden, pooling="mean",
                   query_prefix="", corpus_prefix=""):
    """注入假依赖后构造 LocalEmbedder（_embed 全链路可测，零真模型）。"""
    tokenizer = _install_fake_stack(
        monkeypatch, attention_mask=attention_mask, last_hidden=last_hidden,
    )
    embedder = LocalEmbedder(
        model_dir="fake", pooling=pooling,
        query_prefix=query_prefix, corpus_prefix=corpus_prefix,
    )
    return embedder, tokenizer


def _right_padding_hidden():
    """右 padding 批数据：行 1 内容 3 token（pad 在后）、行 2 内容 4 token 中前 3 个有效。

    mask ``[[1,1,1,0],[1,1,1,1]]``，hidden 按内容摆放，pad 槽位放 _PAD（取到即炸断言）。
    """
    return np.array([
        [_C0, _C1, _C2, _PAD],
        [_C0, _C1, _C2, _C1],
    ], dtype=np.float64)


def test_pooling_variants_are_deterministic_distinct_and_l2_normalized(monkeypatch):
    """同一 last_hidden + mask：三 pooling 输出确定、两两互异、每行 L2 归一化。"""
    outputs = {}
    for pooling in ("mean", "cls", "last_token"):
        embedder, _tokenizer = _make_embedder(
            monkeypatch, attention_mask=[[1, 1, 1, 0], [1, 1, 1, 1]],
            last_hidden=_right_padding_hidden(), pooling=pooling,
        )
        vectors = embedder.embed_corpus(["甲", "乙丙"])
        assert vectors == embedder.embed_corpus(["甲", "乙丙"])  # 确定
        outputs[pooling] = vectors

    assert outputs["mean"] != outputs["cls"]
    assert outputs["cls"] != outputs["last_token"]
    assert outputs["mean"] != outputs["last_token"]
    for vectors in outputs.values():
        for row in vectors:
            assert pytest.approx(np.linalg.norm(row), abs=1e-9) == 1.0


def test_cls_and_last_token_immune_to_padding_direction(monkeypatch):
    """pad 在前与 pad 在后：cls 取到内容首 token、last_token 取到内容尾 token（同批混向）。

    行 1 右 padding（mask [1,1,1,0]，hidden [c0,c1,c2,pad]）与行 2 左 padding
    （mask [0,1,1,1]，hidden [pad,c0,c1,c2]）同一批：逐行独立按 mask 取索引，
    两行取到的都是 c0 / c2——naive ``[:, 0]``/``[:, -1]`` 在对应方向取到 pad 向量。
    """
    hidden_mixed = np.array([
        [_C0, _C1, _C2, _PAD],   # 右 padding：pad 在后
        [_PAD, _C0, _C1, _C2],   # 左 padding：pad 在前
    ], dtype=np.float64)
    masks = [[1, 1, 1, 0], [0, 1, 1, 1]]

    cls_embedder, _t = _make_embedder(
        monkeypatch, attention_mask=masks, last_hidden=hidden_mixed, pooling="cls",
    )
    last_embedder, _t2 = _make_embedder(
        monkeypatch, attention_mask=masks, last_hidden=hidden_mixed, pooling="last_token",
    )

    assert list(cls_embedder.embed_corpus(["甲", "乙"])) == [
        pytest.approx(tuple(_C0)), pytest.approx(tuple(_C0)),
    ]
    assert list(last_embedder.embed_corpus(["甲", "乙"])) == [
        pytest.approx(tuple(_C2)), pytest.approx(tuple(_C2)),
    ]


def test_mean_pooling_matches_mask_weighted_average_regardless_of_padding_side(monkeypatch):
    """mean 对 pad 方向不敏感（attention_mask 加权）：左/右 padding 同内容同输出。

    两行同内容向量、mask 方向相反：输出都是 (c0+c1+c2) 的归一化均值——pad 槽位的
    _PAD 大向量若泄漏进均值，数值会显著偏离。
    """
    hidden = np.array([
        [_C0, _C1, _C2, _PAD],   # 右 padding
        [_PAD, _C0, _C1, _C2],   # 左 padding
    ], dtype=np.float64)
    expected_first = np.array(_C0) + np.array(_C1) + np.array(_C2)
    expected_first = (expected_first / np.linalg.norm(expected_first)).tolist()

    embedder, _t = _make_embedder(
        monkeypatch, attention_mask=[[1, 1, 1, 0], [0, 1, 1, 1]], last_hidden=hidden,
    )
    rows = embedder.embed_corpus(["甲", "乙"])
    assert rows[0] == pytest.approx(tuple(expected_first))
    assert rows[1] == pytest.approx(tuple(expected_first))


def test_prefixes_are_prepended_per_entry(monkeypatch):
    """两路前缀（config 级能力）：embed_query 拼查询前缀、embed_corpus 拼语料前缀。"""
    hidden = np.array([[_C0, _C1, _C2, _PAD], [_C1, _C2, _C0, _PAD]], dtype=np.float64)
    embedder, tokenizer = _make_embedder(
        monkeypatch, attention_mask=[[1, 1, 1, 0], [1, 1, 1, 0]], last_hidden=hidden,
        query_prefix="查询: ", corpus_prefix="文档: ",
    )

    embedder.embed_query("推演剧本")
    embedder.embed_corpus(["a", "b"])

    assert tokenizer.received_texts == ["文档: a", "文档: b"]


def test_unknown_pooling_rejected_at_construction():
    """非法 pooling 构造即 ValueError（参数边界校验先于依赖导入，替身都不需要）。"""
    with pytest.raises(ValueError, match="mean"):
        LocalEmbedder(model_dir="fake", pooling="bogus")


def test_factory_passes_pooling_and_prefixes_to_local_embedder(monkeypatch):
    """工厂 local 分支透传 pooling 与双前缀：config 级能力经装配后生效。"""
    from agent_retrieval.embedders.config import EmbeddingConfig
    from agent_retrieval.embedders.factory import build_embedder

    tokenizer = _install_fake_stack(
        monkeypatch,
        attention_mask=[[1, 1, 1, 0], [1, 1, 1, 0]],
        last_hidden=np.array([[_C0, _C1, _C2, _PAD], [_C0, _C1, _C2, _PAD]], dtype=np.float64),
    )
    config = EmbeddingConfig(
        kind="local", provider="p", model_name="m", model_dir="fake",
        pooling="cls", query_prefix="Q: ", corpus_prefix="D: ",
    )

    embedder = build_embedder(config)
    assert isinstance(embedder, LocalEmbedder)

    vectors = embedder.embed_corpus(["y1", "y2"])
    assert tokenizer.received_texts == ["D: y1", "D: y2"]  # 语料前缀已透传
    assert list(vectors[0]) == [pytest.approx(v) for v in _C0]  # cls pooling 已透传（首非 pad）
    embedder.embed_query("x")
    assert tokenizer.received_texts == ["Q: x"]  # 查询前缀已透传


def test_default_mean_empty_prefix_output_matches_pre_change_values(monkeypatch):
    """默认回归（mean + 空前缀）：输出与改动前逐位一致。

    期望字面量 = B6 改动前探针现值（固定 last_hidden/mask 下改前实现的输出，
    float64 确定性数学，逐位比较）。
    """
    last_hidden = np.array([
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [9.0, 9.0, 9.0, 9.0]],
        [[1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0], [2.1, 2.2, 2.3, 2.4], [9.0, 9.0, 9.0, 9.0]],
    ], dtype=np.float64)
    embedder, _t = _make_embedder(
        monkeypatch, attention_mask=[[1, 1, 0, 0], [1, 1, 1, 0]], last_hidden=last_hidden,
    )

    assert embedder.embed_query("甲") == (
        0.3234983196099664, 0.4313310928132885, 0.5391638660166106, 0.6469966392199329,
    )
    assert embedder.embed_corpus(["甲", "乙丙"]) == [
        (0.3234983196099664, 0.4313310928132885, 0.5391638660166106, 0.6469966392199329),
        (0.4586227087021482, 0.4856005150963923, 0.5125783214906362, 0.5395561278848803),
    ]

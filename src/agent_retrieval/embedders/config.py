"""嵌入器配置：显式传参的不可变值对象。

库纪律：包不读环境变量、不读配置文件——宿主从自己的配置体系（yaml/env/密钥管理）
解析出凭据后构造本对象传入。``identity()`` 四元组进入向量快照 id 与快照元数据，
是「固定嵌入模型与版本」承诺的落点：换模型/版本/推理语义（pooling/前缀）即换快照。

兼容性（B6）：公开返回从三元变四元属 0.x 阶段可见变化。三字段全默认时 ``flags``
为空串且前三元与三元时代逐位一致——默认配置的快照 id 不变（既有快照零重嵌）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

#: pooling 合法值域：mean（掩码加权平均）/ cls（首非 pad）/ last_token（尾非 pad）。
POOLINGS = ("mean", "cls", "last_token")


@dataclass(frozen=True)
class EmbeddingConfig:
    """单个嵌入模型的寻址与凭据（宿主配置体系的解析结果）。"""

    #: 形态：``api``（OpenAI 兼容端点）或 ``local``（本地 ONNX 模型目录）
    kind: str
    provider: str
    model_name: str
    base_url: str = ""
    api_key: str = ""
    #: 嵌入模型版本承诺的元数据落点；缺省即 model_name。
    model_version: str = ""
    #: 0 = 端点自报维度（api）；local 由模型决定。
    dimensions: int = 0
    batch_size: int = 16
    timeout_seconds: float = 20.0
    #: local 形态的 ONNX 模型目录。
    model_dir: str = ""
    #: 池化方式（local 消费）：``mean``/``cls``/``last_token``；api 形态不消费、
    #: 仅随 identity 进快照语义。
    pooling: str = "mean"
    #: 查询路前缀（embed_query 请求前拼接）；空 = 不拼。
    query_prefix: str = ""
    #: 语料路前缀（embed_corpus 请求前拼接）；空 = 不拼。
    corpus_prefix: str = ""

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(
                f"pooling 必须是 {'/'.join(POOLINGS)} 之一，收到: {self.pooling!r}",
            )

    def identity(self) -> tuple[str, str, str, str]:
        """进入快照 id 与快照 meta 的嵌入器身份四元组。

        第四元 ``flags``：pooling/双前缀推理语义的紧凑指纹——三字段全默认为 ``""``；
        任一非默认则取规范 JSON 序列化（JSON 数组结构无双解，分隔符拼接可碰撞）。
        """
        flags = ""
        if (self.pooling, self.query_prefix, self.corpus_prefix) != ("mean", "", ""):
            flags = json.dumps(
                [self.pooling, self.query_prefix, self.corpus_prefix],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return (self.provider, self.model_name, self.model_version or self.model_name, flags)

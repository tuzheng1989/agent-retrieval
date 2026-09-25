"""嵌入器配置：显式传参的不可变值对象。

库纪律：包不读环境变量、不读配置文件——宿主从自己的配置体系（yaml/env/密钥管理）
解析出凭据后构造本对象传入。``identity()`` 三元组进入向量快照 id 与快照元数据，
是「固定嵌入模型与版本」承诺的落点：换模型/版本即换快照。
"""
from __future__ import annotations

from dataclasses import dataclass


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

    def identity(self) -> tuple[str, str, str]:
        """进入快照 id 与快照 meta 的嵌入器身份三元组。"""
        return (self.provider, self.model_name, self.model_version or self.model_name)

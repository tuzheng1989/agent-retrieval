"""格 `full_stack_zh_skill`（full_stack × zh × skill）live 机制检查（SDK P2 / M3）。

姿态装配：matrix.md §2.4（§2.3 快照基础上 + Run 缓存检索 + 判定面 + docgen）。
语料指针：``experiments/agent_e2e/catalogs/catalog-v1.json`` 全 10 资源（matrix.md §3.5
本格行；zh 格消费双语声明面，中文查询，S-D2 载体裁决）。
环境变量：``REDIS_URL`` + ``AGENT_RETRIEVAL_EMBEDDING_{BASE_URL,API_KEY,MODEL}``
（必需，缺任一 exit 4 指路）；``AGENT_RETRIEVAL_E2E_{BASE_URL,API_KEY,MODEL}``
（判定/docgen 网关，可选——缺配置时判定面/docgen 走「未配置降级」并在输出明示，
不影响 exit 0）。装配与断言全在 ``_common.py``（唯一出处）。

用法（仓库根）::

    .venv/Scripts/python.exe -m experiments.scenario_checks.full_stack_zh_skill

退出码：``0`` = 过（S-D4 组件存在性断言全过）/ ``4`` = 缺配置；运行期故障非零码显式失败。
"""

import sys

from experiments.scenario_checks import _common

GRID_ID = "full_stack_zh_skill"


def main() -> int:
    _common.require_env(_common.REDIS_ENV_VARS, _common.EMBEDDING_ENV_VARS)
    _common.check_grid(_common.run_full_stack_grid(GRID_ID))
    return _common.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

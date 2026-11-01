"""金版本零退化门控的 pytest 壳：CI 全量测试即携带检索质量红线。

门控本体在 ``experiments.golden.gate``（亦可用 CLI 独立运行 / ``--update`` 重固化）；
此处只做两件事：当前读数对 ``golden.json`` 零违规；对抗负例零误绑。fixture 缺失
（未生成 fixtures 的开发环境）时 fail 而非 skip——CI 必须有这条红线。
"""

from __future__ import annotations

from experiments.golden import gate


def test_golden_arms_do_not_regress() -> None:
    current = gate.run_gate()
    golden = gate.load_golden()
    problems = gate.violations(current, golden)
    assert not problems, "\n".join(problems)

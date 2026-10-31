import json, sys
work = sys.argv[1]
valid_yaml = sys.argv[2]
entries = json.loads(sys.argv[3])

import rhetra.runtime.secrets as secrets_mod
_calls = []
_orig = secrets_mod.resolve_secrets
def _counting(refs):
    _calls.append(list(refs))
    return _orig(refs)
secrets_mod.resolve_secrets = _counting

from rhetra import run_flow
from rhetra.adapters.offline_kit import MODEL_SCRIPTS, TRANSFORM_FNS

MODEL_SCRIPTS["models.json:offline-shim"] = lambda inputs: {"content": valid_yaml}
TRANSFORM_FNS["t-op-impl"] = lambda p: {"text": p["text"]}

result = run_flow(task="出一条最小流程", input={"text": "hi"},
                  resources={"entries": entries}, work_dir=work, mode="offline")
assert result.status == "completed", result.raw
assert result.evidence.verify() == "finalized"
loaded = [m for m in ("openai", "langchain_openai", "rhetra.adapters.runtime_kit",
                      "rhetra.adapters.real_model", "rhetra.adapters.real_http_tool")
          if m in sys.modules]
assert not loaded, loaded
assert _calls == [], _calls
print("ok")

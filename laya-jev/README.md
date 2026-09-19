# Jev integration

Jev is TypeSafe's System One model. This client uses the same
`predict(state, questions)` seam as Laya so eval and benchmark can
score three engines on one fixture set.

Put the key in repo-root `jev.key` (gitignored, first line is the key)
or in `laya-jev/.env`. Default model is `jev-latest`.

```text
jev.key          → apikey_...
laya-jev/.env    → TYPESAFE_API_KEY=apikey_...
```

```powershell
pip install -e laya-jev/
python laya-eval/eval_laya.py --compare
python laya-benchmark/benchmark.py --compare
```

Questions: `choice`, `noul`, `score`. Fixture `boolean` becomes `noul`.
`allowed_values` become `criteria` when criteria is missing.

```python
from jev_client import JevClient

with JevClient() as jev:
    result = jev.predict(
        {"request": "What is 2 + 2?"},
        {
            "model_tier": {
                "type": "choice",
                "instructions": "Least-capable tier that can do this.",
                "criteria": {
                    "light": "Single-step, deterministic.",
                    "frontier": "Multi-step or ambiguous.",
                },
            }
        },
    )
    print(result["answers"]["model_tier"]["choice"])
```

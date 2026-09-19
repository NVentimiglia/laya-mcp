# Untuned results: Laya vs Jev vs Ollama

Public weights (`convaiinnovations/laya`). Jev `jev-latest`. Ollama
`gemma4:latest` with thinking off. Sources: `laya-eval/eval_results.json`
and `laya-benchmark/benchmark_results.json` (2026-09-19).

```powershell
python laya-eval/eval_laya.py --compare
python laya-benchmark/benchmark.py --compare
```

## Eval (64 fixtures, 80 keys)

| Category | Keys | Laya | Jev | Ollama |
|---|---|---|---|---|
| routing | 4 | 2/4 | 3/4 | 3/4 |
| guardrails | 9 | 7/9 | 9/9 | 9/9 |
| triage | 7 | 3/7 | 7/7 | 5/7 |
| pr_review | 6 | 2/6 | 6/6 | 6/6 |
| moderation | 4 | 4/4 | 4/4 | 4/4 |
| finance_sie | 10 | 0/10 | 10/10 | 8/10 |
| real_estate | 10 | 2/10 | 10/10 | 9/10 |
| math | 10 | 0/10 | 10/10 | 7/10 |
| college | 10 | 2/10 | 10/10 | 10/10 |
| language | 10 | 4/10 | 10/10 | 10/10 |
| Total | 80 | 26/80 (32.5%) | 79/80 (98.8%) | 71/80 (88.8%) |

| Metric | Laya | Jev | Ollama |
|---|---|---|---|
| Median latency | 26 ms | 138 ms | 2,193 ms |
| Mean latency | 132 ms | 148 ms | 2,217 ms |
| Empty outputs | 0 of 64 | 0 of 64 | 0 of 64 |

Laya's mean includes the first call, which warms up CUDA.

50 MCQ keys: Laya 8/50, Jev 50/50, Ollama 44/50. Laya predicted A on
46 of 50.

| Band | Keys | Laya | Jev | Ollama |
|---|---|---|---|---|
| easy | 31 | 4/31 | 31/31 | 28/31 |
| medium | 18 | 4/18 | 18/18 | 15/18 |
| hard | 1 | 0/1 | 1/1 | 1/1 |

## Benchmark

6 fixtures × 5 passes (65 keys). Every engine answered every call.

| Metric | Laya | Jev | Ollama |
|---|---|---|---|
| Key accuracy | 69.2% (45/65) | 70.8% (46/65) | 92.3% (60/65) |
| Median latency | 24 ms | 138 ms | 2,304 ms |
| Mean latency | 61 ms | 149 ms | 2,303 ms |
| Slowest call | 1,006 ms | 353 ms | 2,541 ms |

| Fixture | Laya | Jev | Ollama |
|---|---|---|---|
| auth_patch | 10/15 | 0/15 | 15/15 |
| sql_injection | 0/15 | 11/15 | 10/15 |
| billing_email | 15/15 | 15/15 | 15/15 |
| jailbreak_prompt | 10/10 | 10/10 | 10/10 |
| simple_routing | 5/5 | 5/5 | 5/5 |
| complex_routing | 5/5 | 5/5 | 5/5 |

Laya calls `sql_injection` safe. Jev calls the `auth_patch` fix risky
and asks for changes; the fixture expects safe and merge.

## Changes from the first published run

The first run of these tables had harness bugs. Fixed on 2026-09-19:

- Ollama returned empty text on 59 of 64 eval calls. gemma4 is a
  thinking model and spent its 120-token budget on hidden reasoning.
  The harness now sends `think: false` and allows 200 tokens.
- The Ollama decision prompt listed no labels, so gemma4 invented
  labels such as `low` for `light`. It now gets the same instructions
  and labels as Laya and Jev.
- The scorer read the string `"false"` as true. Only real booleans,
  probabilities, and `"true"`/`"false"` now count as boolean answers.
- Every Jev benchmark call failed with a TLS recursion error. The
  benchmark ran before the certifi client fix.

Ollama moved from 10.0% to 88.8% on the eval and from 38.5% to 92.3%
on the benchmark. Laya did not change. Jev moved from 78/80 to 79/80
on the eval; the API is not deterministic.

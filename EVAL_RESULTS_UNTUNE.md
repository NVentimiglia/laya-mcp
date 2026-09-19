# Untuned results: Laya vs Jev vs Ollama

Public weights (`convaiinnovations/laya`). Jev `jev-latest`. Ollama
`gemma4:latest`. Sources: `laya-eval/eval_results.json` and
`laya-benchmark/benchmark_results.json` (2026-09-19).

## Eval (64 fixtures, 80 keys)

| Category | Keys | Laya | Jev | Ollama |
|---|---|---|---|---|
| routing | 4 | 2/4 | 3/4 | 0/4 |
| guardrails | 9 | 7/9 | 9/9 | 2/9 |
| triage | 7 | 3/7 | 7/7 | 0/7 |
| pr_review | 6 | 2/6 | 6/6 | 0/6 |
| moderation | 4 | 4/4 | 4/4 | 4/4 |
| finance_sie | 10 | 0/10 | 9/10 | 0/10 |
| real_estate | 10 | 2/10 | 10/10 | 0/10 |
| math | 10 | 0/10 | 10/10 | 1/10 |
| college | 10 | 2/10 | 10/10 | 0/10 |
| language | 10 | 4/10 | 10/10 | 1/10 |
| Total | 80 | 26/80 (32.5%) | 78/80 (97.5%) | 8/80 (10.0%) |

| Metric | Laya | Jev | Ollama |
|---|---|---|---|
| Mean latency | 44 ms | 135 ms | 3,050 ms |
| Median latency | 28 ms | 131 ms | 2,905 ms |
| Empty predicted | 0 of 64 | 0 of 64 | 59 of 64 |

50 MCQ keys: Laya 8/50, Jev 48/50, Ollama 2/50. Laya predicted A on
46 of 50.

| Band | Keys | Laya | Jev | Ollama |
|---|---|---|---|---|
| easy | 31 | 4/31 | 31/31 | 2/31 |
| medium | 18 | 4/18 | 17/18 | 0/18 |
| hard | 1 | 0/1 | 1/1 | 0/1 |

## Benchmark

6 fixtures × 5 passes (65 keys). Jev returned empty on all 30 calls.
Ollama returned empty on 20 of 30.

| Metric | Laya | Jev | Ollama |
|---|---|---|---|
| Key accuracy | 69.2% (45/65) | — | 38.5% (25/65) |
| Mean latency | 62 ms | — | 3,019 ms |
| Median latency | 28 ms | — | 3,362 ms |
| Slowest call | 996 ms | — | 3,480 ms |
| Empty outputs | 0 of 30 | 30 of 30 | 20 of 30 |

| Fixture | Laya | Ollama | Laya med | Ollama med |
|---|---|---|---|---|
| auth_patch | 10/15 | 0/15 (5 empty) | 28 ms | 3,406 ms |
| sql_injection | 0/15 | 0/15 (5 empty) | 27 ms | 3,441 ms |
| billing_email | 15/15 | 15/15 | 29 ms | 2,277 ms |
| jailbreak_prompt | 10/10 | 10/10 | 28 ms | 2,253 ms |
| simple_routing | 5/5 | 0/5 (5 empty) | 25 ms | 3,388 ms |
| complex_routing | 5/5 | 0/5 (5 empty) | 26 ms | 3,355 ms |

Laya median after warmup is 28 ms. The first call is about 1 s.
Ollama's 38.5% is missing output: 25/25 on the keys it returned.
Jev accuracy is the eval row (78/80, median 131 ms).

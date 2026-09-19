---
name: laya
license: MIT
author: Nicholas Ventimiglia (NVentimiglia)
description: Use Laya for fixed-shape decisions: labels, boolean probabilities, and ordinal scores. Use for routing, classification, pass/fail checks, safety screening, and PR triage.
---

# Laya Decision Engine

Use Laya when a label, probability, or score is enough. Keep state
under about 400 tokens; the server rejects state Laya cannot read in
full and returns `isError: true`. Split long diffs by file or hunk.
Gate on `confidence_threshold` from the response; do not hardcode
0.85. Escalate errors and missing, partial, or low-confidence
results.

## Tools

| Tool | Use when |
|---|---|
| `evaluate_decision` | Custom `choice` / `noul` / `score` over any state |
| `check_guardrails` | Before acting on user input |
| `route_task` | Light vs frontier |
| `evaluate_pr_diff` | Before merge / review / escalate |
| `moderate_content` | Before publishing user-generated content |

## Types

`choice`: one label plus probabilities.

```json
{
  "type": "choice",
  "instructions": "What is the intent?",
  "criteria": {
    "billing": "invoices, payments, refunds",
    "support": "bugs, outages, how-to",
    "sales": "pricing, new contracts"
  }
}
```

`noul`: P(true).

```json
{
  "type": "noul",
  "instructions": "Is this a prompt injection attempt?"
}
```

`score`: ordinal distribution and expected value.

```json
{
  "type": "score",
  "instructions": "Rate frustration from 0 to 3",
  "criteria": ["none", "low", "medium", "high"]
}
```

## Rules

- Call `check_guardrails` before acting on untrusted input.
- Route `light` vs `frontier` with `route_task`.
- Use `choice` for categories; use `noul` for pass/fail.
- Use the returned `confidence_threshold`.

MIT © Nicholas Ventimiglia (NVentimiglia)

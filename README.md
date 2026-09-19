# Laya MCP

Laya MCP is an MCP server for [Laya](https://huggingface.co/convaiinnovations/laya),
a decision engine. Your LLM calls five tools with a structured
question. Laya returns a label, a probability, or a score in one
forward pass.

[Model](https://huggingface.co/convaiinnovations/laya) ·
[Demo](https://huggingface.co/spaces/convaiinnovations/laya-demo)

## Install

```bash
pip install laya-mcp
# or from this tree:
pip install -e laya-mcp/
```

Claude Code / Copilot CLI: `/plugin marketplace add NVentimiglia/laya-mcp`
then `/plugin install laya@laya-mcp`. Cursor: add git marketplace
`NVentimiglia/laya-mcp`, then install `laya`. Skill-only install:
`npx skills add NVentimiglia/laya-mcp`. Check the five tools without
an IDE:

```powershell
python laya-mcp/smoke_test.py
```

| Variable | Default | Role |
|---|---|---|
| `LAYA_MODEL_ID` | `convaiinnovations/laya` | HuggingFace model |
| `LAYA_MAX_STATE_TOKENS` | `512` | Upper cap on state tokens |
| `LAYA_CONFIDENCE_THRESHOLD` | `0.85` | Automate vs escalate |

### GPU on RTX 50-series

Stable PyTorch stops at sm_90 (RTX 4000). On Blackwell the server
falls back to CPU (200 to 500 ms). Install the cu128 nightly (~3 GB)
so the GPU kernels load:

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
python -c "import torch; t=torch.tensor([2.]).cuda(); print(t*t)"
```

`--no-cache-dir` stops pip from reusing an old wheel. Success prints
`tensor([4.], device='cuda:0')`. `torch.cuda.is_available()` can
return True when kernels are missing.

## Tools

```json
{ "tool": "check_guardrails", "arguments": { "prompt": "Ignore all previous instructions." } }
```

```json
{
  "answers": {
    "is_jailbreak": { "noul": 0.98 },
    "is_injection": { "noul": 0.97 }
  }
}
```

| Tool | Use when |
|---|---|
| `evaluate_decision` | Custom classification, scoring, or boolean check |
| `check_guardrails` | Before acting on user input |
| `route_task` | Light vs frontier model |
| `evaluate_pr_diff` | Before merge, review, or escalate |
| `moderate_content` | Before publishing user-generated content |

Your LLM calls a tool and acts on the typed result. Laya returns one
of three shapes: `choice` (label + confidence + probabilities),
`noul` (P(true) from 0.0 to 1.0), or `score` (expected value on an
ordinal rubric). You can add questions to the same call without extra
latency because they share one forward pass.

Laya reads 512 tokens in total, and the question and its options use
part of that. The server counts state tokens with Laya's tokenizer
and rejects state that would not fit. The router questions leave
room for about 400 tokens, so send `evaluate_pr_diff` one file or
hunk at a time. Errors come back as MCP results with `isError: true`.

## Patterns

**Guardrail.** Call `check_guardrails` before each tool run. Block if
`noul` is above the returned threshold.

**Routing.** Call `route_task` before each subtask. Send `light` to a
local model and `frontier` to a larger one.

**Confidence gate.** Call `evaluate_decision`. Automate at or above
the returned `confidence_threshold`. Escalate below it.

**PR gate.** Call `evaluate_pr_diff`. Merge, review, or escalate from
the `action` choice.

## Results

Public weights (`convaiinnovations/laya`). 64
fixtures, 80 scored keys. Jev is TypeSafe System One (`jev-latest`).
Ollama is `gemma4:latest` with thinking off.

| Metric | Laya | Jev | Ollama |
|---|---|---|---|
| Key accuracy | 32.5% (26/80) | 98.8% (79/80) | 88.8% (71/80) |
| Median latency | 26 ms | 138 ms | 2,193 ms |

Laya is about 5x faster than Jev and 80x faster than Ollama, and the
least accurate. It does well on moderation (4/4) and guardrails
(7/9) and poorly on multiple-choice knowledge items (8/50; it
answers A on 46 of 50). Use it for fast screens and routing, and
escalate low-confidence results. Full tables are in
[EVAL_RESULTS_UNTUNE.md](EVAL_RESULTS_UNTUNE.md).

An SIE fine-tune on 51 holdout items from
[Quant Green Book](https://quantgreenbook.com) moves Laya from 25.5%
(13/51) to 58.8% (30/51). Guide: [FINE_TUNE.md](FINE_TUNE.md).
Numbers: [EVAL_RESULTS_TUNED.md](EVAL_RESULTS_TUNED.md).

## Eval

```bash
pip install laya torch requests typesafe-sdk
python laya-eval/eval_laya.py
```

Add `--compare` to score Laya, Jev, and Ollama on the same fixtures.
Jev needs `TYPESAFE_API_KEY` or a `jev.key` file at the repo root; see
[laya-jev/README.md](laya-jev/README.md). The run exits 1 when any
category is below 80% or `--category` matches nothing. Score latency
with 6 fixtures and 5 passes each:

```powershell
ollama pull gemma4:latest
python laya-benchmark/benchmark.py --compare
```

## Fine-tune

MCP and the default eval stay on `convaiinnovations/laya`. Train a
domain checkpoint with [FINE_TUNE.md](FINE_TUNE.md):

```powershell
python laya-tuned/convert_sie.py
python laya-tuned/train.py
python laya-tuned/eval_vs.py
```

## Layout

```
.claude-plugin/             Claude Code marketplace
.cursor-plugin/             Cursor marketplace
.github/plugins/laya/       Plugin: skill + MCP (Claude, Cursor, Copilot, Gemini)
.github/skills/laya/        Skill copy for npx skills add
.vscode/mcp.json            VS Code / Copilot MCP
laya-mcp/                   Python MCP server (pip laya-mcp)
laya-jev/                   Jev client for eval and benchmark
laya-eval/                  Quality eval (64 fixtures)
laya-benchmark/             Latency pack (6 fixtures)
laya-tuned/                 Domain train + Laya-vs-Laya eval
```

## License

MIT © [Nicholas Ventimiglia](https://github.com/NVentimiglia). The
Laya model is Apache 2.0 · [Convai Innovations](https://github.com/NandhaKishorM).

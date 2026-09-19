# Fine-Tuning Laya on Custom Domain Data

Laya ships a pre-trained checkpoint (`convaiinnovations/laya`).
Fine-tuning specialises it on one domain's question schemas.

`pip install laya` (0.1.6) **does not** expose `agent.finetune()`
or `agent.save()`. This repo trains with `laya-tuned/train.py`:
copy the public snapshot, run the same `proper_reward` rule the
checkpoint was trained with, overwrite `model.safetensors`.

Upstream notebook (different API than the pip package):
**[Fine-Tune on Custom Data (Colab T4)](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_colab.ipynb)**

---

## When to fine-tune

Fine-tune when:
- The domain has specialised vocabulary (medical, legal, finance)
- `choice` labels differ from general routing/triage patterns
- You need higher accuracy on one narrow task
- You have at least ~200 labelled examples per question type

The base checkpoint is enough for routing, guardrails, and
moderation.

---

## Data format

Train rows: `state`, typed `questions`, `answers`.

```json
[
  {
    "state": {
      "question": "<stem>",
      "choices": {"A": "<A>", "B": "<B>", "C": "<C>", "D": "<D>"}
    },
    "questions": {
      "answer": {
        "type": "choice",
        "instructions": "Select the single best answer. Return only the letter.",
        "criteria": {
          "A": "<A>",
          "B": "<B>",
          "C": "<C>",
          "D": "<D>"
        }
      }
    },
    "answers": {
      "answer": { "choice": "B" }
    }
  }
]
```

Rules:
- `state` — string or JSON, keep under 512 tokens
- `questions` — same schema as production
- `answers` — Laya output shape (`choice`, `noul`, or `score`)
- Balance labels; do not let one class exceed ~70%
- Hold out ~10% of ids. Do not train on the holdout.

---

## SIE run in this repo

SIE questions are from [Quant Green Book](https://quantgreenbook.com).
Thank you for the dataset. Source markdown stays **outside** git.
Convert, train, and score Laya against Laya:

```powershell
python laya-tuned/convert_sie.py
python laya-tuned/train.py
python laya-tuned/eval_vs.py
```

| Output | Git |
|---|---|
| `laya-tuned/train_sie.json` | ignored |
| `laya-tuned/eval_holdout.json` | ignored |
| `laya-tuned/checkpoints/laya-sie` | ignored |

`convert_sie.py` drops stems that match `finance_sie_*` in
`laya-eval/fixtures.json`, skips flashcards, and writes a 10%
holdout (seed 42, stratified A–D).

`train.py` loads `convaiinnovations/laya`, writes only to
`laya-tuned/checkpoints/laya-sie`. Leave `LAYA_MODEL_ID` on the
public model.

`eval_vs.py` is Laya-only: public weights vs `laya-sie` on the
SIE holdout. No Jev, no Ollama.

Latest numbers: [EVAL_RESULTS_TUNED.md](EVAL_RESULTS_TUNED.md).
Do not cite them next to the default 64-fixture eval.

---

## Deploying a fine-tuned checkpoint

MCP, eval, and benchmark read `LAYA_MODEL_ID` or `--model-id`.
Leave MCP on `convaiinnovations/laya`. Score the SIE checkpoint
with `eval_vs.py` or:

```powershell
python laya-eval/eval_laya.py --fixtures laya-tuned/eval_holdout.json --model-id laya-tuned/checkpoints/laya-sie --out laya-eval/eval_results_tuned.json
```

That last command is Laya-only if you omit `--compare`. Prefer
`eval_vs.py` so untuned and tuned share one report.

---

## Tips

| Concern | Recommendation |
|---|---|
| Not enough data | Start with 200 examples |
| Label imbalance | Oversample minority classes |
| Catastrophic forgetting | LR 2e-5, 2–3 epochs |
| Overfitting | Hold out 10%; stop when val loss plateaus |
| GPU (RTX 50-series) | Nightly cu128; `train.py` falls back to CPU |
| No `agent.finetune` | Use `laya-tuned/train.py` |

---

## Resources

- [Laya GitHub](https://github.com/NandhaKishorM/laya)
- [Fine-Tune Notebook (Colab)](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_colab.ipynb)
- [HuggingFace Model Card](https://huggingface.co/convaiinnovations/laya)
- [SIE isolation notes](laya-tuned/TODO.md)
- [Quant Green Book](https://quantgreenbook.com) — SIE dataset

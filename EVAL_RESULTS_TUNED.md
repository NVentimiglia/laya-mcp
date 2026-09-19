# Tuned results: Laya untuned vs Laya SIE-tuned

SIE fine-tune on 453 questions from
[Quant Green Book](https://quantgreenbook.com). 51-item holdout.
Laya only. Source: `laya-eval/eval_results_tuned.json` (2026-09-19).

```powershell
python laya-tuned/convert_sie.py
python laya-tuned/train.py
python laya-tuned/eval_vs.py
```

Checkpoint: `laya-tuned/checkpoints/laya-sie`. Base:
`convaiinnovations/laya`.

| Metric | Laya untuned | Laya tuned |
|---|---|---|
| Key accuracy | 25.5% (13/51) | **58.8% (30/51)** |
| Median latency | 24 ms | 24 ms |
| Predicted letter mix | 45 A / 3 B / 1 C / 2 D | 17 A / 22 B / 11 C / 1 D |

| True letter | Holdout n | Untuned | Tuned |
|---|---|---|---|
| A | 11 | 11/11 | 9/11 |
| B | 23 | 2/23 | 16/23 |
| C | 15 | 0/15 | 5/15 |
| D | 2 | 0/2 | 0/2 |
| Total | 51 | 13/51 (25.5%) | 30/51 (58.8%) |

Train: 453 rows (95 A / 206 B / 135 C / 17 D). 3 epochs, batch 4,
lr 2e-5, seed 42, about 45 s on GPU. Two seeded runs gave the same
loss and accuracy.

The first published run reported 60.8% (31/51). That run had no
seed, so batch order changed on every run, and one training stem was
cut short by a parser bug. Both are fixed; 58.8% is the reproducible
number.

Guide: [FINE_TUNE.md](FINE_TUNE.md).

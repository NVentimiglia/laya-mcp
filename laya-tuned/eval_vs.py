"""Laya-only eval: public checkpoint vs SIE fine-tune on the same holdout.

No Jev. No Ollama. Loads one agent at a time so 12 GB cards fit.

  python laya-tuned/eval_vs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_DIR = ROOT / "laya-eval"
DEFAULT_HOLD = HERE / "eval_holdout.json"
DEFAULT_OUT = EVAL_DIR / "eval_results_tuned.json"
UNTUNED_ID = "convaiinnovations/laya"
TUNED_ID = HERE / "checkpoints" / "laya-sie"

sys.path.insert(0, str(EVAL_DIR))

from eval_laya import (  # noqa: E402
    PASS_THRESHOLD,
    EngineStats,
    _categories_passed,
    _extract_laya_answers,
    normalize_questions,
)


def _load_agent(model_id: str) -> Any:
    import laya

    path = Path(model_id)
    if path.exists():
        model_id = str(path.resolve())
    print(f"Loading {model_id}…", flush=True)
    agent = laya.load(model_id)
    print("Ready.", flush=True)
    return agent


def _display_id(model_id: str) -> str:
    """Repo-relative path for local checkpoints, so reports hold no machine paths."""
    path = Path(model_id).resolve()
    if not path.exists():
        return model_id
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _score_model(
    name: str,
    model_id: str,
    fixtures: list[dict[str, Any]],
) -> EngineStats:
    import torch

    stats = EngineStats(name)
    agent = _load_agent(model_id)
    try:
        for fx in fixtures:
            questions = normalize_questions(fx["questions"])
            state = fx.get("input", fx.get("state"))
            t0 = time.perf_counter()
            try:
                result = agent.predict(state, questions)
                predicted = _extract_laya_answers(result)
                err = None
            except Exception as exc:  # noqa: BLE001
                predicted, err = {}, str(exc)
            latency_ms = (time.perf_counter() - t0) * 1000
            stats.add(
                fx["id"], fx.get("category", "sie"), fx.get("difficulty"),
                predicted, fx["ground_truth"], latency_ms, err,
            )
            last = stats.records[-1]
            mark = "OK" if last["correct"] == last["total"] else "NO"
            print(
                f"  {fx['id']:<24} {name:<14} {mark} "
                f"{last['correct']}/{last['total']} {last['latency_ms']:6.0f}ms",
                flush=True,
            )
    finally:
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return stats


def _print_engine(eng: EngineStats) -> None:
    print(f"\n{'='*62}", flush=True)
    print(f"ACCURACY — {eng.name}", flush=True)
    print(f"{'='*62}", flush=True)
    for cat, s in sorted(eng.cat.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        flag = "PASS" if acc >= PASS_THRESHOLD else "FAIL"
        print(
            f"  {flag} {cat:<24} {s['correct']:>3}/{s['total']:<3}  {acc*100:5.1f}%",
            flush=True,
        )
    print(
        f"\n  Overall: {eng.overall_correct}/{eng.overall_total}  "
        f"{eng.accuracy*100:.1f}%  avg {eng.avg_latency_ms:.0f} ms/call",
        flush=True,
    )


def run(
    fixtures_path: Path,
    out_path: Path,
    untuned_id: str,
    tuned_id: str,
) -> int:
    fixtures: list[dict[str, Any]] = json.loads(
        fixtures_path.read_text(encoding="utf-8")
    )
    print(f"Laya untuned vs tuned  n={len(fixtures)}", flush=True)
    base = _score_model("Laya-untuned", untuned_id, fixtures)
    tuned = _score_model("Laya-tuned", tuned_id, fixtures)
    _print_engine(base)
    _print_engine(tuned)
    print(f"\n{'='*62}", flush=True)
    print("LAYA UNTUNED VS TUNED", flush=True)
    print(f"{'='*62}", flush=True)
    print(f"  {'Metric':<22}{'untuned':>14}{'tuned':>14}", flush=True)
    print(
        f"  {'Overall accuracy':<22}"
        f"{base.accuracy*100:>13.1f}%"
        f"{tuned.accuracy*100:>13.1f}%",
        flush=True,
    )
    print(
        f"  {'Avg latency (ms)':<22}"
        f"{base.avg_latency_ms:>14.1f}"
        f"{tuned.avg_latency_ms:>14.1f}",
        flush=True,
    )
    delta = (tuned.accuracy - base.accuracy) * 100
    print(f"  {'Delta (pp)':<22}{delta:>28.1f}", flush=True)

    report = {
        "meta": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suite": "laya-vs",
            "fixtures": len(fixtures),
            "untuned_id": untuned_id,
            "tuned_id": _display_id(tuned_id),
        },
        "laya_untuned": {
            "model": base.name,
            "overall": {
                "correct": base.overall_correct,
                "total": base.overall_total,
                "accuracy": round(base.accuracy, 4),
            },
            "avg_latency_ms": round(base.avg_latency_ms, 2),
            "fixtures": base.records,
        },
        "laya_tuned": {
            "model": tuned.name,
            "overall": {
                "correct": tuned.overall_correct,
                "total": tuned.overall_total,
                "accuracy": round(tuned.accuracy, 4),
            },
            "avg_latency_ms": round(tuned.avg_latency_ms, 2),
            "fixtures": tuned.records,
        },
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {out_path}", flush=True)
    return 0 if _categories_passed(tuned) else 1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Laya untuned vs SIE-tuned")
    p.add_argument("--fixtures", type=Path, default=DEFAULT_HOLD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--untuned", default=UNTUNED_ID)
    p.add_argument("--tuned", default=str(TUNED_ID))
    args = p.parse_args(argv)
    if not args.fixtures.is_file():
        raise SystemExit(
            f"Missing {args.fixtures}. Run: python laya-tuned/convert_sie.py"
        )
    tuned_path = Path(args.tuned)
    if not tuned_path.exists():
        raise SystemExit(
            f"No checkpoint at {tuned_path.resolve()}. "
            "Run: python laya-tuned/train.py"
        )
    sys.exit(run(args.fixtures, args.out, args.untuned, str(tuned_path)))


if __name__ == "__main__":
    main()

"""
Laya Quality Evaluation Harness — Laya / Jev / Ollama

Runs fixtures through Laya, TypeSafe Jev, and/or a local Ollama model,
scores each against ground truth, and prints a side-by-side comparison.

Fixture schemas supported
-------------------------
Decision fixtures:
  id, category, description, input (obj), questions, ground_truth

MCQ fixtures:
  id, category, difficulty, description,
  input {question, choices {A,B,C,D}}, questions, ground_truth {answer: "X"}

Usage
-----
  pip install laya torch requests typesafe-sdk
  python eval_laya.py                          # Laya only
  python eval_laya.py --compare                # Laya + Jev + Ollama
  python eval_laya.py --jev                    # Laya + Jev
  python eval_laya.py --ollama                 # Laya + Ollama gemma4:latest
  python eval_laya.py --jev-only --ollama-only # skip Laya
  python eval_laya.py --category math          # filter by category

Jev requires TYPESAFE_API_KEY. Default model: jev-latest.

Exit code
---------
  0  every category >= 80% for every engine that ran
  1  a category is below 80%, or no engine ran
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_JEV_DIR = Path(__file__).resolve().parent.parent / "laya-jev"
if str(_JEV_DIR) not in sys.path:
    sys.path.insert(0, str(_JEV_DIR))

from jev_client import (  # noqa: E402
    DEFAULT_MODEL as DEFAULT_JEV_MODEL,
    JevClient,
    jev_available,
    normalize_questions,
)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_LAYA_MODEL_ID = "convaiinnovations/laya"


def _state_text(input_field: Any) -> str:
    if isinstance(input_field, str):
        return input_field
    return json.dumps(input_field)


def _is_mcq(fx: dict[str, Any]) -> bool:
    """True when the fixture is a multiple-choice question with A/B/C/D choices."""
    inp = fx.get("input", {})
    return isinstance(inp, dict) and "choices" in inp


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_answer(predicted: Any, expected: Any) -> bool:
    if predicted is None:
        return False
    if isinstance(expected, bool):
        if isinstance(predicted, float):
            return (predicted > 0.5) == expected
        return bool(predicted) == expected
    if isinstance(expected, str) and expected.lower() in ("true", "false"):
        expected_bool = expected.lower() == "true"
        if isinstance(predicted, float):
            return (predicted > 0.5) == expected_bool
        if isinstance(predicted, bool):
            return predicted == expected_bool
        return str(predicted).lower() == expected.lower()
    return str(predicted).strip().lower() == str(expected).strip().lower()


def _extract_laya_answers(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    answers = result.get("answers")
    if not isinstance(answers, dict):
        return {}
    flat: dict[str, Any] = {}
    for k, v in answers.items():
        if isinstance(v, dict):
            if "choice" in v:
                flat[k] = v["choice"]
            elif "noul" in v:
                flat[k] = v["noul"]
            elif "score" in v:
                flat[k] = v["score"]
            else:
                flat[k] = v
        else:
            flat[k] = v
    return flat


def _score_fixture(predicted: dict[str, Any], ground_truth: dict[str, Any]) -> tuple[int, int]:
    correct = sum(1 for k, v in ground_truth.items() if _score_answer(predicted.get(k), v))
    return correct, len(ground_truth)


# ---------------------------------------------------------------------------
# Ollama prompt builders
# ---------------------------------------------------------------------------


def _ollama_prompt_mcq(fx: dict[str, Any]) -> str:
    """Single-letter answer prompt for MCQ fixtures."""
    inp = fx["input"]
    q = inp.get("question", "")
    choices = inp.get("choices", {})
    lines = [q, ""]
    for letter, text in sorted(choices.items()):
        lines.append(f"{letter}. {text}")
    lines += [
        "",
        'Reply with ONLY a JSON object: {"answer":"X"} where X is A, B, C, or D.',
        "Do not include any explanation.",
    ]
    return "\n".join(lines)


def _ollama_prompt_decision(fx: dict[str, Any]) -> str:
    """Structured JSON prompt for decision fixtures."""
    state = _state_text(fx.get("input", fx.get("state")))
    gt = fx["ground_truth"]
    keys = list(gt.keys())

    schema_parts = []
    for key in keys:
        expected = gt[key]
        if isinstance(expected, bool) or (isinstance(expected, str) and expected.lower() in ("true", "false")):
            schema_parts.append(f'"{key}": true|false')
        else:
            schema_parts.append(f'"{key}": "..."')

    schema = "{" + ", ".join(schema_parts) + "}"

    return (
        f"Evaluate the following input and return ONLY a JSON object with this exact schema:\n"
        f"{schema}\n\n"
        f"Input:\n{state}\n\n"
        f"Reply with only the JSON object. No explanation."
    )


def _parse_ollama_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from Ollama free text."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Try to extract just a letter answer as fallback
        m = re.search(r'"answer"\s*:\s*"([A-Da-d])"', text)
        if m:
            return {"answer": m.group(1).upper()}
        return {}


# ---------------------------------------------------------------------------
# Ollama runner
# ---------------------------------------------------------------------------


def _ollama_available() -> bool:
    try:
        import requests  # noqa: PLC0415
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ollama_call(prompt: str, model: str) -> tuple[dict[str, Any], float]:
    """
    Call Ollama using /api/chat with format=json for reliable structured output.
    Falls back to /api/generate if /api/chat is unavailable.
    """
    import requests  # noqa: PLC0415

    t0 = time.perf_counter()

    # Try /api/chat with format:json first — much more reliable for structured output
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "seed": 42, "num_predict": 120},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.json().get("message", {}).get("content", "")
        return _parse_ollama_json(text), latency_ms
    except Exception as exc:
        print(
            f"WARNING: Ollama /api/chat failed ({type(exc).__name__}: {exc}); "
            "falling back to /api/generate",
            flush=True,
        )

    # Fallback: /api/generate (streaming=False)
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0, "seed": 42, "num_predict": 120}},
        timeout=120,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = resp.json().get("response", "")
    return _parse_ollama_json(text), latency_ms


# ---------------------------------------------------------------------------
# Per-engine accumulator
# ---------------------------------------------------------------------------


class EngineStats:
    def __init__(self, name: str):
        self.name = name
        self.cat: dict[str, dict[str, int]] = {}
        self.diff: dict[str, dict[str, int]] = {}
        self.records: list[dict[str, Any]] = []
        self.total_ms: float = 0.0
        self.calls: int = 0

    def add(self, fid: str, category: str, difficulty: str | None,
            predicted: dict[str, Any], ground_truth: dict[str, Any],
            latency_ms: float, error: str | None = None) -> tuple[int, int]:
        correct, total = _score_fixture(predicted, ground_truth)

        cs = self.cat.setdefault(category, {"correct": 0, "total": 0})
        cs["correct"] += correct
        cs["total"] += total

        if difficulty:
            ds = self.diff.setdefault(difficulty, {"correct": 0, "total": 0})
            ds["correct"] += correct
            ds["total"] += total

        self.total_ms += latency_ms
        self.calls += 1
        self.records.append({
            "id": fid, "category": category, "difficulty": difficulty,
            "ground_truth": ground_truth, "predicted": predicted,
            "correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0,
            "latency_ms": round(latency_ms, 2), "error": error,
        })
        return correct, total

    @property
    def overall_correct(self) -> int:
        return sum(s["correct"] for s in self.cat.values())

    @property
    def overall_total(self) -> int:
        return sum(s["total"] for s in self.cat.values())

    @property
    def accuracy(self) -> float:
        return self.overall_correct / self.overall_total if self.overall_total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------


PASS_THRESHOLD = 0.80


def _categories_passed(eng: EngineStats) -> bool:
    if not eng.cat:
        return False
    return all(
        (s["correct"] / s["total"] if s["total"] else 0.0) >= PASS_THRESHOLD
        for s in eng.cat.values()
    )


def _report_key(name: str) -> str:
    lower = name.lower()
    if lower.startswith("laya"):
        return "laya"
    if lower.startswith("jev"):
        return "jev"
    return "ollama"


def run_eval(
    fixtures_path: Path,
    out_path: Path,
    run_laya: bool,
    run_ollama: bool,
    run_jev: bool,
    ollama_model: str,
    jev_model: str,
    category_filter: str | None,
    laya_model_id: str = DEFAULT_LAYA_MODEL_ID,
) -> int:
    fixtures: list[dict[str, Any]] = json.loads(fixtures_path.read_text(encoding="utf-8"))

    if category_filter:
        fixtures = [f for f in fixtures if f["category"] == category_filter]
        print(f"Filtered to category '{category_filter}': {len(fixtures)} fixtures", flush=True)

    if not fixtures:
        print("No fixtures to run.", flush=True)
        return 0

    # --- Laya setup ---
    laya_stats = EngineStats("Laya")
    agent = None
    if run_laya:
        try:
            import laya  # noqa: PLC0415
            print(f"Loading Laya model ({laya_model_id})…", flush=True)
            agent = laya.load(laya_model_id)
            print("Laya model ready.", flush=True)
        except ImportError:
            print("WARNING: laya not installed — skipping Laya. Run: pip install laya torch", flush=True)
            run_laya = False

    # --- Ollama setup ---
    ollama_stats = EngineStats(f"Ollama ({ollama_model})")
    if run_ollama and not _ollama_available():
        print(f"WARNING: Ollama not reachable at {OLLAMA_BASE} — skipping Ollama.", flush=True)
        run_ollama = False

    # --- Jev setup ---
    jev_stats = EngineStats(f"Jev ({jev_model})")
    jev_client: JevClient | None = None
    if run_jev:
        if not jev_available():
            print(
                "WARNING: TYPESAFE_API_KEY is not set — skipping Jev.",
                flush=True,
            )
            run_jev = False
        else:
            try:
                jev_client = JevClient(model=jev_model)
                print(f"Jev client ready ({jev_model}).", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: Jev client failed — skipping Jev: {exc}", flush=True)
                run_jev = False

    if not (run_laya or run_jev or run_ollama):
        print("No engines available to run.", flush=True)
        return 1

    print(f"\nRunning {len(fixtures)} fixtures"
          + (" [Laya]" if run_laya else "")
          + (f" [Jev {jev_model}]" if run_jev else "")
          + (f" [Ollama {ollama_model}]" if run_ollama else "")
          + "\n", flush=True)

    try:
        return _run_fixtures(
            fixtures, out_path, category_filter,
            run_laya, agent, laya_stats,
            run_jev, jev_client, jev_stats,
            run_ollama, ollama_model, ollama_stats,
            laya_model_id,
        )
    finally:
        if jev_client is not None:
            jev_client.close()


def _run_fixtures(
    fixtures: list[dict[str, Any]],
    out_path: Path,
    category_filter: str | None,
    run_laya: bool,
    agent: Any,
    laya_stats: EngineStats,
    run_jev: bool,
    jev_client: JevClient | None,
    jev_stats: EngineStats,
    run_ollama: bool,
    ollama_model: str,
    ollama_stats: EngineStats,
    laya_model_id: str = DEFAULT_LAYA_MODEL_ID,
) -> int:
    # --- Fixture loop ---
    for fx in fixtures:
        fid = fx["id"]
        category = fx["category"]
        difficulty = fx.get("difficulty")
        ground_truth: dict[str, Any] = fx["ground_truth"]
        mcq = _is_mcq(fx)

        laya_correct = laya_total = 0
        jev_correct = jev_total = 0
        ollama_correct = ollama_total = 0
        raw_input = fx.get("input", fx.get("state"))
        questions = normalize_questions(fx["questions"])

        # Laya — same state shape as MCP tools (do not unwrap one-key dicts)
        if run_laya and agent is not None:
            state = raw_input
            t0 = time.perf_counter()
            try:
                result = agent.predict(state, questions)
                latency_ms = (time.perf_counter() - t0) * 1000
                predicted = _extract_laya_answers(result)
                err = None
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000
                predicted = {}
                err = str(exc)
            laya_correct, laya_total = laya_stats.add(
                fid, category, difficulty, predicted, ground_truth, latency_ms, err)

        # Jev — keep structured state; TypeSafe wants named fields
        if run_jev and jev_client is not None:
            jev_state = raw_input if raw_input is not None else ""
            t0 = time.perf_counter()
            try:
                result_j = jev_client.predict(jev_state, questions)
                latency_ms_j = (time.perf_counter() - t0) * 1000
                predicted_j = _extract_laya_answers(result_j)
                err_j = None
            except Exception as exc:  # noqa: BLE001
                latency_ms_j = (time.perf_counter() - t0) * 1000
                predicted_j, err_j = {}, str(exc)
            jev_correct, jev_total = jev_stats.add(
                fid, category, difficulty, predicted_j, ground_truth,
                latency_ms_j, err_j)

        # Ollama
        if run_ollama:
            prompt = _ollama_prompt_mcq(fx) if mcq else _ollama_prompt_decision(fx)
            try:
                predicted_o, latency_ms_o = _ollama_call(prompt, ollama_model)
                err_o = None
            except Exception as exc:  # noqa: BLE001
                predicted_o, latency_ms_o, err_o = {}, 0.0, str(exc)
            ollama_correct, ollama_total = ollama_stats.add(
                fid, category, difficulty, predicted_o, ground_truth, latency_ms_o, err_o)

        # Print row
        parts = [f"  {fid:<42}"]
        if run_laya:
            mark = "✓" if laya_correct == laya_total else ("✗" if laya_correct == 0 else "~")
            parts.append(f"Laya {mark}{laya_correct}/{laya_total} {laya_stats.records[-1]['latency_ms']:6.0f}ms")
        if run_jev:
            mark = "✓" if jev_correct == jev_total else ("✗" if jev_correct == 0 else "~")
            parts.append(f"Jev {mark}{jev_correct}/{jev_total} {jev_stats.records[-1]['latency_ms']:6.0f}ms")
        if run_ollama:
            mark = "✓" if ollama_correct == ollama_total else ("✗" if ollama_correct == 0 else "~")
            parts.append(f"Ollama {mark}{ollama_correct}/{ollama_total} {ollama_stats.records[-1]['latency_ms']:6.0f}ms")
        if difficulty:
            parts.append(f"[{difficulty}]")
        print("  ".join(parts), flush=True)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    engines = []
    if run_laya:
        engines.append(laya_stats)
    if run_jev:
        engines.append(jev_stats)
    if run_ollama:
        engines.append(ollama_stats)

    for eng in engines:
        print(f"\n{'='*62}", flush=True)
        print(f"ACCURACY BY CATEGORY — {eng.name}", flush=True)
        print(f"{'='*62}", flush=True)
        for cat, s in sorted(eng.cat.items()):
            acc = s["correct"] / s["total"] if s["total"] else 0
            flag = "✓" if acc >= 0.80 else "✗"
            print(f"  {flag} {cat:<24} {s['correct']:>3}/{s['total']:<3}  {acc*100:5.1f}%", flush=True)

        if eng.diff:
            print(f"\n  By difficulty:", flush=True)
            for diff in ("easy", "medium", "hard"):
                s = eng.diff.get(diff)
                if not s:
                    continue
                acc = s["correct"] / s["total"] if s["total"] else 0
                flag = "✓" if acc >= 0.80 else "✗"
                print(f"    {flag} {diff:<20} {s['correct']:>3}/{s['total']:<3}  {acc*100:5.1f}%", flush=True)

        print(f"\n  Overall: {eng.overall_correct}/{eng.overall_total}  "
              f"{eng.accuracy*100:.1f}%  avg {eng.avg_latency_ms:.0f} ms/call", flush=True)

    if len(engines) >= 2:
        width = 22 + 16 * len(engines)
        print(f"\n{'='*width}", flush=True)
        print("COMPARISON", flush=True)
        print(f"{'='*width}", flush=True)
        header = f"  {'Metric':<22}" + "".join(f"{eng.name:>16}" for eng in engines)
        print(header, flush=True)
        print(f"  {'-'*(width-2)}", flush=True)
        acc = "".join(f"{eng.accuracy*100:>15.1f}%" for eng in engines)
        print(f"  {'Overall accuracy':<22}{acc}", flush=True)
        lat = "".join(f"{eng.avg_latency_ms:>16.1f}" for eng in engines)
        print(f"  {'Avg latency (ms)':<22}{lat}", flush=True)
        kinds = []
        for eng in engines:
            kinds.append(
                "structured JSON" if _report_key(eng.name) != "ollama"
                else "free text"
            )
        kind = "".join(f"{k:>16}" for k in kinds)
        print(f"  {'Output type':<22}{kind}", flush=True)
        conf = "".join(
            f"{('yes' if _report_key(eng.name) != 'ollama' else 'no'):>16}"
            for eng in engines
        )
        print(f"  {'Calibrated confidence':<22}{conf}", flush=True)

    # ---------------------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------------------
    report: dict[str, Any] = {
        "meta": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "laya_model_id": laya_model_id,
            "fixtures": len(fixtures),
            "category_filter": category_filter,
        },
    }
    for eng in engines:
        key = _report_key(eng.name)
        report[key] = {
            "model": eng.name,
            "overall": {"correct": eng.overall_correct, "total": eng.overall_total,
                        "accuracy": round(eng.accuracy, 4)},
            "avg_latency_ms": round(eng.avg_latency_ms, 2),
            "by_category": {
                cat: {"correct": s["correct"], "total": s["total"],
                      "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0}
                for cat, s in eng.cat.items()
            },
            "by_difficulty": {
                diff: {"correct": s["correct"], "total": s["total"],
                       "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0}
                for diff, s in eng.diff.items()
            },
            "fixtures": eng.records,
        }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {out_path}", flush=True)

    if not engines:
        print("No engines ran.", flush=True)
        return 1

    all_passed = all(_categories_passed(eng) for eng in engines)
    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Laya quality evaluation harness")
    p.add_argument("--fixtures", type=Path,
                   default=Path(__file__).parent / "fixtures.json")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "eval_results.json")
    p.add_argument("--ollama", action="store_true",
                   help="Also run fixtures through Ollama and compare")
    p.add_argument("--ollama-only", action="store_true",
                   help="Run Ollama, skip Laya unless --compare")
    p.add_argument("--jev", action="store_true",
                   help="Also run fixtures through Jev (needs TYPESAFE_API_KEY)")
    p.add_argument("--jev-only", action="store_true",
                   help="Run Jev, skip Laya unless --compare")
    p.add_argument("--compare", action="store_true",
                   help="Run Laya + Jev + Ollama (three-way)")
    p.add_argument("--model", default=DEFAULT_OLLAMA_MODEL,
                   help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    p.add_argument("--jev-model", default=DEFAULT_JEV_MODEL,
                   help=f"Jev model (default: {DEFAULT_JEV_MODEL})")
    p.add_argument(
        "--model-id",
        default=os.environ.get("LAYA_MODEL_ID", DEFAULT_LAYA_MODEL_ID),
        help="Laya checkpoint or HuggingFace id "
             f"(default: LAYA_MODEL_ID or {DEFAULT_LAYA_MODEL_ID})",
    )
    p.add_argument("--category", default=None,
                   help="Run only fixtures matching this category")
    args = p.parse_args(argv)

    run_laya = args.compare or not (args.ollama_only or args.jev_only)
    run_ollama = args.compare or args.ollama or args.ollama_only
    run_jev = args.compare or args.jev or args.jev_only

    sys.exit(run_eval(
        fixtures_path=args.fixtures,
        out_path=args.out,
        run_laya=run_laya,
        run_ollama=run_ollama,
        run_jev=run_jev,
        ollama_model=args.model,
        jev_model=args.jev_model,
        category_filter=args.category,
        laya_model_id=args.model_id,
    ))


if __name__ == "__main__":
    main()

"""
Laya vs. Jev vs. Ollama Benchmark — deterministic, streaming-aware.

Measures
--------
  - Time to First Token (TTFT) via streaming (Ollama)
  - Tokens per second (generation throughput)
  - Peak memory delta (RSS)
  - Structured-output accuracy against a ground-truth fixture
  - Speedup ratio and correctness comparison

Usage
-----
  pip install requests psutil typesafe-sdk
  ollama pull gemma4:latest
  set TYPESAFE_API_KEY=...
  python benchmark.py --compare
  python benchmark.py [--n 10] [--model gemma4:latest] [--laya-only]

Results are written to laya-benchmark/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import psutil
import requests

_ROOT = Path(__file__).resolve().parent.parent
for _dir in (_ROOT / "laya-jev", _ROOT / "laya-eval"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

# Scoring and Ollama helpers live in eval_laya so both tools score alike.
from eval_laya import (  # noqa: E402
    DEFAULT_LAYA_MODEL_ID,
    DEFAULT_OLLAMA_MODEL as DEFAULT_MODEL,
    OLLAMA_BASE,
    OLLAMA_OPTIONS,
    _extract_laya_answers,
    _ollama_available,
    _parse_ollama_json,
    _score_fixture,
)
from jev_client import (  # noqa: E402
    DEFAULT_MODEL as DEFAULT_JEV_MODEL,
    JevClient,
    jev_available,
)


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f'No fixtures in {path}')
    return rows


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    fixture_id: str
    latency_ms: float
    ttft_ms: float | None        # None when streaming not used
    tokens_per_sec: float | None
    mem_delta_mb: float
    raw_output: str
    parsed: dict[str, Any]
    correct_keys: int
    total_keys: int

    @property
    def accuracy(self) -> float:
        return self.correct_keys / self.total_keys if self.total_keys else 0.0


@dataclass
class BenchmarkRun:
    engine: str                        # "laya", "jev", or "ollama"
    model: str
    calls: list[CallResult] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls) / max(len(self.calls), 1)

    @property
    def avg_ttft_ms(self) -> float | None:
        ttfts = [c.ttft_ms for c in self.calls if c.ttft_ms is not None]
        return sum(ttfts) / len(ttfts) if ttfts else None

    @property
    def avg_tps(self) -> float | None:
        tps_vals = [c.tokens_per_sec for c in self.calls if c.tokens_per_sec is not None]
        return sum(tps_vals) / len(tps_vals) if tps_vals else None

    @property
    def accuracy(self) -> float:
        ck = sum(c.correct_keys for c in self.calls)
        tk = sum(c.total_keys for c in self.calls)
        return ck / tk if tk else 0.0

    @property
    def avg_mem_delta_mb(self) -> float:
        return sum(c.mem_delta_mb for c in self.calls) / max(len(self.calls), 1)


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Laya benchmark
# ---------------------------------------------------------------------------


def _laya_call_direct(fixture: dict[str, Any], agent: Any) -> CallResult:
    """Call Laya in-process. A failed call scores zero instead of ending the run."""
    questions = fixture["questions"]
    state = fixture["state"]

    mem_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        result = agent.predict(state, questions)
        err = None
    except Exception as exc:  # noqa: BLE001
        result, err = {"answers": {}}, str(exc)
        print(f"  ERROR on {fixture['id']}: {exc}", flush=True)
    t1 = time.perf_counter()
    mem_after = _rss_mb()

    parsed = _extract_laya_answers(result)
    ck, tk = _score_fixture(parsed, fixture["ground_truth"])

    return CallResult(
        fixture_id=fixture["id"],
        latency_ms=(t1 - t0) * 1000,
        ttft_ms=None,
        tokens_per_sec=None,
        mem_delta_mb=mem_after - mem_before,
        raw_output=json.dumps(result) if err is None else err,
        parsed=parsed,
        correct_keys=ck,
        total_keys=tk,
    )


def benchmark_laya(
    fixtures: list[dict[str, Any]],
    n: int = 5,
    verbose: bool = True,
    model_id: str = DEFAULT_LAYA_MODEL_ID,
) -> BenchmarkRun:
    """Run Laya benchmark using the direct Python API."""
    try:
        import laya as _laya  # noqa: PLC0415
    except ImportError:
        print("  [SKIP] laya package not installed. Run: pip install laya", flush=True)
        return BenchmarkRun(engine="laya", model=model_id)

    print(f"\n{'='*60}", flush=True)
    print(f"Benchmarking Laya (direct) — {n} passes over {len(fixtures)} fixtures", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"  Loading model ({model_id})…", flush=True)
    agent = _laya.load(model_id)
    print("  Model loaded.", flush=True)

    run = BenchmarkRun(engine="laya", model=model_id)

    for pass_num in range(n):
        for fixture in fixtures:
            result = _laya_call_direct(fixture, agent)
            run.calls.append(result)
            if verbose:
                acc_str = f"{result.correct_keys}/{result.total_keys}"
                print(
                    f"  pass {pass_num+1} | {fixture['id']:<20} | "
                    f"{result.latency_ms:8.1f} ms | acc {acc_str}",
                    flush=True,
                )

    print(f"\n  Laya avg latency : {run.avg_latency_ms:.1f} ms", flush=True)
    print(f"  Laya accuracy    : {run.accuracy*100:.1f}%", flush=True)
    return run


# ---------------------------------------------------------------------------
# Jev benchmark (TypeSafe System One)
# ---------------------------------------------------------------------------


def benchmark_jev(
    fixtures: list[dict[str, Any]],
    n: int = 5,
    model: str = DEFAULT_JEV_MODEL,
    verbose: bool = True,
) -> BenchmarkRun:
    """Run Jev with the same fixtures and question schemas as Laya."""
    if not jev_available():
        print("  [SKIP] TYPESAFE_API_KEY is not set.", flush=True)
        return BenchmarkRun(engine="jev", model=model)

    print(f"\n{'='*60}", flush=True)
    print(
        f"Benchmarking Jev {model} — {n} passes over {len(fixtures)} fixtures",
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    run = BenchmarkRun(engine="jev", model=model)
    try:
        client = JevClient(model=model)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] Jev client failed: {exc}", flush=True)
        return run

    try:
        for pass_num in range(n):
            for fixture in fixtures:
                questions = fixture["questions"]
                mem_before = _rss_mb()
                t0 = time.perf_counter()
                try:
                    result = client.predict(fixture["state"], questions)
                    err = None
                except Exception as exc:  # noqa: BLE001
                    result, err = {"answers": {}}, str(exc)
                    print(f"  ERROR on {fixture['id']}: {exc}", flush=True)
                t1 = time.perf_counter()
                mem_after = _rss_mb()

                parsed = _extract_laya_answers(result)
                ck, tk = _score_fixture(parsed, fixture["ground_truth"])
                call = CallResult(
                    fixture_id=fixture["id"],
                    latency_ms=(t1 - t0) * 1000,
                    ttft_ms=None,
                    tokens_per_sec=None,
                    mem_delta_mb=mem_after - mem_before,
                    raw_output=json.dumps(result) if err is None else err,
                    parsed=parsed,
                    correct_keys=ck,
                    total_keys=tk,
                )
                run.calls.append(call)
                if verbose:
                    print(
                        f"  pass {pass_num+1} | {fixture['id']:<20} | "
                        f"{call.latency_ms:8.1f} ms | acc {ck}/{tk}",
                        flush=True,
                    )
    finally:
        client.close()

    print(f"\n  Jev avg latency : {run.avg_latency_ms:.1f} ms", flush=True)
    print(f"  Jev accuracy    : {run.accuracy*100:.1f}%", flush=True)
    return run


# ---------------------------------------------------------------------------
# Ollama benchmark (streaming)
# ---------------------------------------------------------------------------


def _ollama_stream(model: str, prompt: str) -> tuple[str, float, float, float]:
    """
    Stream a single Ollama generation.

    Returns
    -------
    (full_text, ttft_ms, tokens_per_sec, total_latency_ms)
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": OLLAMA_OPTIONS,
    }

    t0 = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []
    eval_count = 0
    eval_duration_ns = 0

    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk = json.loads(raw_line)
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()
            if not chunk.get("done", False):
                chunks.append(chunk.get("response", ""))
            else:
                eval_count = chunk.get("eval_count", 0)
                eval_duration_ns = chunk.get("eval_duration", 0)

    t1 = time.perf_counter()
    full_text = "".join(chunks)
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else (t1 - t0) * 1000
    total_ms = (t1 - t0) * 1000
    tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

    return full_text, ttft_ms, tps, total_ms


def benchmark_ollama(
    fixtures: list[dict[str, Any]],
    n: int = 5,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> BenchmarkRun:
    """Run Ollama benchmark with streaming TTFT measurement."""
    if not _ollama_available():
        print(f"  [SKIP] Ollama not reachable at {OLLAMA_BASE}.", flush=True)
        return BenchmarkRun(engine="ollama", model=model)

    print(f"\n{'='*60}", flush=True)
    print(f"Benchmarking Ollama {model} — {n} passes over {len(fixtures)} fixtures", flush=True)
    print(f"{'='*60}", flush=True)

    run = BenchmarkRun(engine="ollama", model=model)

    for pass_num in range(n):
        for fixture in fixtures:
            prompt_template = fixture["ollama_prompt"]
            state = fixture["state"]
            state_str = state if isinstance(state, str) else json.dumps(state)
            prompt = prompt_template.replace("{state}", state_str)

            mem_before = _rss_mb()
            try:
                text, ttft_ms, tps, total_ms = _ollama_stream(model, prompt)
            except Exception as exc:
                print(f"  ERROR on {fixture['id']}: {exc}", flush=True)
                tk = len(fixture["ground_truth"])
                result = CallResult(
                    fixture_id=fixture["id"],
                    latency_ms=0.0,
                    ttft_ms=None,
                    tokens_per_sec=None,
                    mem_delta_mb=0.0,
                    raw_output=str(exc),
                    parsed={},
                    correct_keys=0,
                    total_keys=tk,
                )
                run.calls.append(result)
                continue
            mem_after = _rss_mb()

            parsed = _parse_ollama_json(text)
            ck, tk = _score_fixture(parsed, fixture["ground_truth"])

            result = CallResult(
                fixture_id=fixture["id"],
                latency_ms=total_ms,
                ttft_ms=ttft_ms,
                tokens_per_sec=tps,
                mem_delta_mb=mem_after - mem_before,
                raw_output=text,
                parsed=parsed,
                correct_keys=ck,
                total_keys=tk,
            )
            run.calls.append(result)

            if verbose:
                acc_str = f"{ck}/{tk}"
                print(
                    f"  pass {pass_num+1} | {fixture['id']:<20} | "
                    f"{total_ms:8.1f} ms | TTFT {ttft_ms:6.1f} ms | "
                    f"{tps:5.1f} t/s | acc {acc_str}",
                    flush=True,
                )

    ttft_str = f"{run.avg_ttft_ms:.1f} ms" if run.avg_ttft_ms else "N/A"
    tps_str = f"{run.avg_tps:.1f} t/s" if run.avg_tps else "N/A"
    print(f"\n  Ollama avg latency : {run.avg_latency_ms:.1f} ms", flush=True)
    print(f"  Ollama avg TTFT    : {ttft_str}", flush=True)
    print(f"  Ollama avg TPS     : {tps_str}", flush=True)
    print(f"  Ollama accuracy    : {run.accuracy*100:.1f}%", flush=True)
    return run


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def _col_name(run: BenchmarkRun) -> str:
    if run.engine == "jev":
        return "Jev"
    if run.engine == "laya":
        return "Laya"
    return "Ollama"


def print_comparison(*runs: BenchmarkRun) -> None:
    active = [r for r in runs if r.calls]
    if len(active) < 2:
        print(
            "\n[Comparison skipped — need at least two engines with results]",
            flush=True,
        )
        return

    width = 30 + 16 * len(active)
    print(f"\n{'='*width}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*width}", flush=True)
    header = f"{'Metric':<30}" + "".join(f"{_col_name(r):>16}" for r in active)
    print(header, flush=True)
    print(f"{'-'*width}", flush=True)

    lat = "".join(f"{r.avg_latency_ms:>16.1f}" for r in active)
    print(f"{'Avg latency (ms)':<30}{lat}", flush=True)

    ttft = ""
    for r in active:
        if r.engine == "ollama" and r.avg_ttft_ms:
            ttft += f"{r.avg_ttft_ms:>16.1f}"
        elif r.engine == "ollama":
            ttft += f"{'N/A':>16}"
        else:
            ttft += f"{'N/A (1-pass)':>16}"
    print(f"{'Avg TTFT (ms)':<30}{ttft}", flush=True)

    tps = ""
    for r in active:
        if r.engine == "ollama" and r.avg_tps:
            tps += f"{r.avg_tps:>16.1f}"
        elif r.engine == "ollama":
            tps += f"{'N/A':>16}"
        else:
            tps += f"{'N/A (typed)':>16}"
    print(f"{'Avg tokens/sec':<30}{tps}", flush=True)

    mem = "".join(f"{r.avg_mem_delta_mb:>16.1f}" for r in active)
    print(f"{'Avg memory delta (MB)':<30}{mem}", flush=True)

    acc = "".join(f"{r.accuracy*100:>15.1f}%" for r in active)
    print(f"{'Accuracy':<30}{acc}", flush=True)

    by_engine = {r.engine: r for r in active}
    if "laya" in by_engine and "ollama" in by_engine:
        speedup = by_engine["ollama"].avg_latency_ms / max(
            by_engine["laya"].avg_latency_ms, 0.1
        )
        print(f"{'Speedup Laya vs Ollama':<30} {speedup:>15.1f}x", flush=True)
    if "jev" in by_engine and "ollama" in by_engine:
        speedup_j = by_engine["ollama"].avg_latency_ms / max(
            by_engine["jev"].avg_latency_ms, 0.1
        )
        print(f"{'Speedup Jev vs Ollama':<30} {speedup_j:>15.1f}x", flush=True)

    kind = "".join(
        f"{('structured JSON' if r.engine != 'ollama' else 'free text'):>16}"
        for r in active
    )
    print(f"{'Output type':<30}{kind}", flush=True)
    conf = "".join(
        f"{('yes' if r.engine != 'ollama' else 'no'):>16}" for r in active
    )
    print(f"{'Calibrated confidence':<30}{conf}", flush=True)
    print(f"{'='*width}", flush=True)


# ---------------------------------------------------------------------------
# Per-fixture accuracy table
# ---------------------------------------------------------------------------


def _acc_str(calls: list[CallResult]) -> str:
    if not calls:
        return "—"
    total = sum(c.total_keys for c in calls)
    if not total:
        return "—"
    return f"{sum(c.correct_keys for c in calls) / total * 100:.0f}%"


def print_fixture_table(
    fixtures: list[dict[str, Any]], *runs: BenchmarkRun
) -> None:
    active = [r for r in runs if r.calls]
    if not active:
        return

    grouped: list[dict[str, list[CallResult]]] = []
    for run in active:
        by_fx: dict[str, list[CallResult]] = {}
        for c in run.calls:
            by_fx.setdefault(c.fixture_id, []).append(c)
        grouped.append(by_fx)

    width = 52 + 12 * len(active)
    print(f"\n{'='*width}", flush=True)
    print("PER-FIXTURE RESULTS", flush=True)
    print(f"{'='*width}", flush=True)
    header = f"{'Fixture':<22} {'Description':<28}" + "".join(
        f"{_col_name(r) + ' acc':>12}" for r in active
    )
    print(header, flush=True)
    print(f"{'-'*width}", flush=True)

    for fx in fixtures:
        fid = fx["id"]
        desc = fx["description"][:27]
        cells = "".join(f"{_acc_str(g.get(fid, [])):>12}" for g in grouped)
        print(f"  {fid:<20} {desc:<28}{cells}", flush=True)

    print(f"{'='*width}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Laya vs Jev vs Ollama benchmark")
    p.add_argument("--n", type=int, default=5, help="Number of passes per fixture (default 5)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (default: gemma4:latest)")
    p.add_argument("--jev-model", default=DEFAULT_JEV_MODEL, help="Jev model (default: jev-latest)")
    p.add_argument(
        "--model-id",
        default=os.environ.get("LAYA_MODEL_ID", DEFAULT_LAYA_MODEL_ID),
        help="Laya checkpoint or HuggingFace id "
             f"(default: LAYA_MODEL_ID or {DEFAULT_LAYA_MODEL_ID})",
    )
    p.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).parent / "fixtures.json",
        help="JSON fixture pack (default: laya-benchmark/fixtures.json)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "benchmark_results.json",
        help="Output JSON path",
    )
    p.add_argument("--laya-only", action="store_true", help="Skip Ollama and Jev")
    p.add_argument("--ollama-only", action="store_true", help="Skip Laya and Jev")
    p.add_argument("--jev-only", action="store_true", help="Skip Laya and Ollama")
    p.add_argument("--jev", action="store_true", help="Include Jev (needs TYPESAFE_API_KEY)")
    p.add_argument("--compare", action="store_true", help="Run Laya + Jev + Ollama")
    p.add_argument("--quiet", action="store_true", help="Suppress per-call output")
    return p.parse_args(argv)


def _serialize_run(run: BenchmarkRun) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model": run.model,
        "avg_latency_ms": round(run.avg_latency_ms, 2),
        "avg_mem_delta_mb": round(run.avg_mem_delta_mb, 2),
        "accuracy": round(run.accuracy, 4),
        "calls": [asdict(c) for c in run.calls],
    }
    if run.engine == "ollama":
        data["avg_ttft_ms"] = (
            round(run.avg_ttft_ms, 2) if run.avg_ttft_ms else None
        )
        data["avg_tokens_per_sec"] = (
            round(run.avg_tps, 2) if run.avg_tps else None
        )
    return data


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    verbose = not args.quiet
    fixtures = load_fixtures(args.fixtures)

    run_laya = args.compare or args.laya_only or not (
        args.ollama_only or args.jev_only
    )
    run_ollama = args.compare or args.ollama_only or not (
        args.laya_only or args.jev_only
    )
    run_jev = args.compare or args.jev or args.jev_only

    print("Laya vs. Jev vs. Ollama Benchmark", flush=True)
    print(
        f"Fixtures: {len(fixtures)} | Passes: {args.n} | "
        f"Laya: {args.model_id} | Ollama: {args.model} | Jev: {args.jev_model}",
        flush=True,
    )

    laya_run = BenchmarkRun(engine="laya", model=args.model_id)
    jev_run = BenchmarkRun(engine="jev", model=args.jev_model)
    ollama_run = BenchmarkRun(engine="ollama", model=args.model)

    if run_laya:
        laya_run = benchmark_laya(
            fixtures, n=args.n, verbose=verbose, model_id=args.model_id,
        )
    if run_jev:
        jev_run = benchmark_jev(
            fixtures, n=args.n, model=args.jev_model, verbose=verbose,
        )
    if run_ollama:
        ollama_run = benchmark_ollama(
            fixtures, n=args.n, model=args.model, verbose=verbose,
        )

    print_comparison(laya_run, jev_run, ollama_run)
    print_fixture_table(fixtures, laya_run, jev_run, ollama_run)

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "laya_model_id": args.model_id,
            "n_passes": args.n,
            "fixtures": len(fixtures),
            "ollama_model": args.model,
            "jev_model": args.jev_model,
        },
        "laya": _serialize_run(laya_run),
        "jev": _serialize_run(jev_run),
        "ollama": _serialize_run(ollama_run),
    }

    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

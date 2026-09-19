"""Convert SIE markdown into gitignored Laya train/holdout JSON.

Dataset: https://quantgreenbook.com (thank you). Does not copy
source files into git. Writes:
  laya-tuned/train_sie.json
  laya-tuned/eval_holdout.json

  python laya-tuned/convert_sie.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SOURCE = Path(r"D:\Projects\QuantStudy\Repository\SIE\questions")
DEFAULT_FIXTURES = ROOT / "laya-eval" / "fixtures.json"

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_OPT = re.compile(r"^([A-D])\)\s*(.*)$", re.M)

DIFF_BAND = {1: "easy", 2: "easy", 3: "medium", 4: "medium"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_front_matter(block: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in block.splitlines():
        if ":" not in raw:
            continue
        key, val = raw.split(":", 1)
        meta[key.strip()] = val.strip().strip("'").strip('"')
    return meta


def parse_sie_markdown(text: str) -> dict[str, Any] | None:
    m = _FRONT.match(text.strip())
    if not m:
        return None
    meta = parse_front_matter(m.group(1))
    body = m.group(2).strip()
    if meta.get("type") not in (None, "", "multiple_choice"):
        return None
    opts = {k: v.strip() for k, v in _OPT.findall(body)}
    if set(opts) != {"A", "B", "C", "D"}:
        return None
    stem = body[: body.find("A)")].strip()
    stem = re.sub(r"\*\*Explanation:\*\*.*", "", stem, flags=re.S).strip()
    if not stem:
        return None
    answer = meta.get("correct_answer", "").strip().upper()
    if answer not in opts:
        return None
    try:
        difficulty_n = int(meta.get("difficulty") or "3")
    except ValueError:
        difficulty_n = 3
    band = DIFF_BAND.get(difficulty_n, "hard")
    criteria = {k: opts[k] for k in ("A", "B", "C", "D")}
    questions = {
        "answer": {
            "type": "choice",
            "instructions": "Select the single best answer. Return only the letter.",
            "criteria": criteria,
        }
    }
    return {
        "id": meta.get("id") or "",
        "topic": meta.get("topic") or "SIE",
        "difficulty": band,
        "stem": stem,
        "choices": criteria,
        "answer": answer,
        "questions": questions,
    }


def to_train_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": {"question": row["stem"], "choices": row["choices"]},
        "questions": row["questions"],
        "answers": {"answer": {"choice": row["answer"]}},
    }


def to_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "category": "sie",
        "difficulty": row["difficulty"],
        "description": row["topic"],
        "input": {"question": row["stem"], "choices": row["choices"]},
        "questions": row["questions"],
        "ground_truth": {"answer": row["answer"]},
    }


def poison_stems(fixtures_path: Path) -> set[str]:
    if not fixtures_path.is_file():
        return set()
    rows = json.loads(fixtures_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for fx in rows:
        if fx.get("category") != "finance_sie":
            continue
        q = (fx.get("input") or {}).get("question") or ""
        if q:
            out.add(_norm(q))
    return out


def split_holdout(
    rows: list[dict[str, Any]],
    frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_letter[row["answer"]].append(row)
    rng = random.Random(seed)
    holdout: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for letter in ("A", "B", "C", "D"):
        group = list(by_letter.get(letter, []))
        rng.shuffle(group)
        n_hold = max(1, int(round(len(group) * frac))) if group else 0
        holdout.extend(group[:n_hold])
        train.extend(group[n_hold:])
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def convert(
    source: Path,
    fixtures_path: Path,
    train_path: Path,
    holdout_path: Path,
    frac: float,
    seed: int,
) -> dict[str, int]:
    blocked = poison_stems(fixtures_path)
    parsed: list[dict[str, Any]] = []
    skipped = 0
    poisoned = 0
    for path in sorted(source.glob("sie_*.md")):
        row = parse_sie_markdown(path.read_text(encoding="utf-8"))
        if row is None or not row["id"]:
            skipped += 1
            continue
        if _norm(row["stem"]) in blocked:
            poisoned += 1
            continue
        parsed.append(row)
    train, holdout = split_holdout(parsed, frac, seed)
    hold_ids = {r["id"] for r in holdout}
    train = [r for r in train if r["id"] not in hold_ids]
    train_path.write_text(
        json.dumps([to_train_row(r) for r in train], indent=2),
        encoding="utf-8",
    )
    holdout_path.write_text(
        json.dumps([to_eval_row(r) for r in holdout], indent=2),
        encoding="utf-8",
    )
    letters = {k: sum(1 for r in train if r["answer"] == k) for k in "ABCD"}
    return {
        "parsed": len(parsed),
        "train": len(train),
        "holdout": len(holdout),
        "skipped": skipped,
        "poisoned": poisoned,
        **{f"train_{k}": v for k, v in letters.items()},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert SIE markdown to Laya JSON")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument("--train-out", type=Path, default=HERE / "train_sie.json")
    p.add_argument("--holdout-out", type=Path, default=HERE / "eval_holdout.json")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if not args.source.is_dir():
        raise SystemExit(f"SIE source not found: {args.source}")
    stats = convert(
        args.source, args.fixtures, args.train_out, args.holdout_out,
        args.holdout_frac, args.seed,
    )
    print(
        f"parsed {stats['parsed']}  train {stats['train']}  "
        f"holdout {stats['holdout']}  skipped {stats['skipped']}  "
        f"poison-drop {stats['poisoned']}  "
        f"A/B/C/D {stats['train_A']}/{stats['train_B']}/"
        f"{stats['train_C']}/{stats['train_D']}",
        flush=True,
    )
    print(f"train -> {args.train_out}", flush=True)
    print(f"holdout -> {args.holdout_out}", flush=True)


if __name__ == "__main__":
    main()

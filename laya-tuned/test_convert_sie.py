"""Parser tests for SIE markdown → Laya train/holdout JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convert_sie import parse_sie_markdown, poison_stems, split_holdout, to_eval_row, to_train_row

SAMPLE = """---
type: multiple_choice
id: sie_test_01
topic: Equities
correct_answer: B
difficulty: 2
---
What is a common stock?

A) A debt instrument
B) An equity security
C) A municipal bond
D) A futures contract
"""


def test_parse_four_choice_mcq():
    row = parse_sie_markdown(SAMPLE)
    assert row is not None
    assert row["id"] == "sie_test_01"
    assert row["answer"] == "B"
    assert row["difficulty"] == "easy"
    assert row["choices"]["A"].startswith("A debt")
    train = to_train_row(row)
    assert train["answers"]["answer"]["choice"] == "B"
    assert train["questions"]["answer"]["type"] == "choice"
    ev = to_eval_row(row)
    assert ev["ground_truth"]["answer"] == "B"
    assert ev["category"] == "sie"


def test_parse_rejects_missing_option():
    bad = SAMPLE.replace("D) A futures contract\n", "")
    assert parse_sie_markdown(bad) is None


def test_split_holdout_is_disjoint_and_stratified():
    rows = []
    for i, letter in enumerate("ABCD" * 10):
        rows.append({"id": f"sie_{i}", "answer": letter})
    train, hold = split_holdout(rows, 0.10, 42)
    train_ids = {r["id"] for r in train}
    hold_ids = {r["id"] for r in hold}
    assert not train_ids & hold_ids
    assert len(hold) == 4
    assert {r["answer"] for r in hold} == set("ABCD")


def test_poison_stems_reads_finance_sie_only(tmp_path: Path):
    fixtures = [
        {
            "id": "finance_sie_01",
            "category": "finance_sie",
            "input": {"question": "Which order type is a market order?"},
        },
        {
            "id": "routing_01",
            "category": "routing",
            "input": {"prompt": "not a stem"},
        },
    ]
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    stems = poison_stems(path)
    assert "which order type is a market order?" in stems
    assert len(stems) == 1

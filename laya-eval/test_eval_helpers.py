"""Unit tests for eval harness helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_laya import (
    EngineStats,
    _categories_passed,
    _extract_laya_answers,
    _state_text,
)


def test_extract_none_answers():
    assert _extract_laya_answers({"answers": None}) == {}
    assert _extract_laya_answers(None) == {}
    assert _extract_laya_answers("raw") == {}


def test_extract_choice_and_noul():
    flat = _extract_laya_answers({
        "answers": {
            "intent": {"choice": "billing", "confidence": 0.9},
            "risk": {"noul": 0.2},
        }
    })
    assert flat == {"intent": "billing", "risk": 0.2}


def test_state_text_keeps_one_key_dict():
    payload = {"request": "What is 2 + 2?"}
    text = _state_text(payload)
    assert "request" in text
    assert "2 + 2" in text


def test_categories_passed_empty_fails():
    assert _categories_passed(EngineStats("Laya")) is False


def test_categories_passed_uses_per_category_bar():
    eng = EngineStats("Laya")
    eng.add("a", "routing", None, {"model": "light"}, {"model": "light"}, 1.0)
    eng.add("b", "math", None, {"answer": "A"}, {"answer": "B"}, 1.0)
    assert eng.accuracy > 0
    assert _categories_passed(eng) is False


def test_tuned_fixtures_schema_and_disjoint():
    here = Path(__file__).resolve().parent
    rows = json.loads((here / "fixtures_tuned.json").read_text(encoding="utf-8"))
    default_ids = {
        fx["id"]
        for fx in json.loads((here / "fixtures.json").read_text(encoding="utf-8"))
    }
    assert isinstance(rows, list) and rows
    for fx in rows:
        assert fx["id"] not in default_ids
        assert fx.get("category")
        assert isinstance(fx.get("questions"), dict) and fx["questions"]
        assert isinstance(fx.get("ground_truth"), dict) and fx["ground_truth"]
        assert "input" in fx


def test_score_string_false_is_false():
    # bool("false") is True; the scorer must not use bool() on strings.
    from eval_laya import _score_answer
    assert _score_answer("false", False) is True
    assert _score_answer("false", True) is False
    assert _score_answer("true", True) is True


def test_score_label_is_not_boolean():
    from eval_laya import _score_answer
    assert _score_answer("safe", True) is False
    assert _score_answer("safe", False) is False


def test_score_noul_probability():
    from eval_laya import _score_answer
    assert _score_answer(0.9, True) is True
    assert _score_answer(0.1, False) is True
    assert _score_answer(0.1, "true") is False


def test_score_label_case_and_space():
    from eval_laya import _score_answer
    assert _score_answer(" Light ", "light") is True


def test_ollama_decision_prompt_lists_labels():
    from eval_laya import _ollama_prompt_decision
    fx = {
        "input": {"request": "What is 2 + 2?"},
        "questions": {
            "tier": {
                "type": "choice",
                "instructions": "Pick a tier.",
                "criteria": {"light": "Simple.", "frontier": "Hard."},
            },
            "urgent": {"type": "boolean", "instructions": "Urgent?"},
        },
        "ground_truth": {"tier": "light", "urgent": False},
    }
    prompt = _ollama_prompt_decision(fx)
    assert '"tier": "light|frontier"' in prompt
    assert '"urgent": true|false' in prompt
    assert "light: Simple." in prompt


def test_empty_category_filter_fails(tmp_path):
    from eval_laya import run_eval
    fixtures = tmp_path / "fx.json"
    fixtures.write_text(json.dumps([{"id": "a", "category": "math"}]), encoding="utf-8")
    rc = run_eval(fixtures, tmp_path / "out.json", False, False, False,
                  "m", "j", "no_such_category")
    assert rc == 1

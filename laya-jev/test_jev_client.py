"""Unit tests for Jev question normalisation and answer flattening."""

from __future__ import annotations

from types import SimpleNamespace
import os

import pytest

from jev_client import (
    _apply_typesafe_key,
    _make_http_client,
    flatten_answers,
    normalize_questions,
)


def test_boolean_becomes_noul():
    out = normalize_questions(
        {"jailbreak": {"type": "boolean", "instructions": "Is this a jailbreak?"}}
    )
    assert out["jailbreak"]["type"] == "noul"
    assert out["jailbreak"]["instructions"] == "Is this a jailbreak?"
    assert "criteria" not in out["jailbreak"]


def test_allowed_values_become_criteria():
    out = normalize_questions(
        {
            "department": {
                "type": "choice",
                "instructions": "Route the ticket.",
                "allowed_values": ["billing", "technical"],
            }
        }
    )
    assert out["department"]["criteria"] == {
        "billing": "billing",
        "technical": "technical",
    }
    assert "allowed_values" not in out["department"]


def test_existing_criteria_kept():
    criteria = {"safe": "No risk", "unsafe": "Clear vuln"}
    out = normalize_questions(
        {
            "safety": {
                "type": "choice",
                "instructions": "Is the diff safe?",
                "allowed_values": ["safe", "unsafe"],
                "criteria": criteria,
            }
        }
    )
    assert out["safety"]["criteria"] == criteria


def test_choice_without_options_raises():
    with pytest.raises(ValueError, match="needs criteria"):
        normalize_questions(
            {"x": {"type": "choice", "instructions": "Pick one"}}
        )


def test_extra_keys_stripped():
    out = normalize_questions(
        {
            "tone": {
                "type": "noul",
                "instructions": "Urgent?",
                "weight": 2,
                "notes": "drop me",
            }
        }
    )
    assert set(out["tone"]) == {"type", "instructions"}


def test_flatten_choice_and_noul():
    result = SimpleNamespace(
        model="jev-1.13.0",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 12}),
        answers={
            "dept": SimpleNamespace(
                model_dump=lambda: {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.9,
                    "legend": None,
                }
            ),
            "urgent": SimpleNamespace(
                model_dump=lambda: {"type": "noul", "noul": 0.81}
            ),
        },
    )
    flat = flatten_answers(result)
    assert flat["model"] == "jev-1.13.0"
    assert flat["answers"]["dept"]["choice"] == "billing"
    assert flat["answers"]["urgent"]["noul"] == 0.81
    assert "legend" not in flat["answers"]["dept"]


def test_http_client_closes_cleanly():
    client = _make_http_client(timeout=5)
    client.close()


def test_jev_key_file_sets_env_when_missing(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    _apply_typesafe_key("  apikey_test_not_real  \n")
    assert os.environ["TYPESAFE_API_KEY"] == "apikey_test_not_real"


def test_jev_key_file_does_not_override_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "already-set")
    _apply_typesafe_key("apikey_from_file")
    assert os.environ["TYPESAFE_API_KEY"] == "already-set"

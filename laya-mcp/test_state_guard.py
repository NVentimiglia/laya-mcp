"""
Unit tests for the state size guard in laya_mcp_server.

These tests run without a loaded Laya model by patching `laya.load`
before import and testing guard logic, validation, and error paths.

Usage
-----
  pip install pytest
  pytest laya-mcp/test_state_guard.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

_laya_stub = types.ModuleType("laya")
_laya_stub.load = MagicMock(return_value=MagicMock())
_laya_stub.guard_questions = MagicMock(return_value={})
_laya_stub.router_questions = MagicMock(return_value={})
_laya_stub.moderation_questions = MagicMock(return_value={})
sys.modules["laya"] = _laya_stub

import laya_mcp_server as srv  # noqa: E402


def _long_state(n_words: int) -> str:
    return " ".join(["word"] * n_words)


def _parse_err(contents) -> dict:
    return json.loads(contents[0].text)


def _run(coro):
    return asyncio.run(coro)


def test_token_estimate_empty():
    assert srv._token_estimate("") == 0


def test_token_estimate_single_word():
    assert srv._token_estimate("hello") == 1


def test_token_estimate_ten_words():
    assert srv._token_estimate(" ".join(["w"] * 10)) == 13


def test_guard_passes_short_string():
    srv._guard_state_size(_long_state(100))


def test_guard_raises_long_string():
    with pytest.raises(ValueError, match="exceeds the"):
        srv._guard_state_size(_long_state(400))


def test_guard_raises_at_exact_boundary():
    srv._guard_state_size(_long_state(394))
    with pytest.raises(ValueError):
        srv._guard_state_size(_long_state(395))


def test_guard_passes_short_dict():
    srv._guard_state_size({"key": "short value"})


def test_guard_raises_long_dict():
    big_dict = {"body": " ".join(["word"] * 400)}
    with pytest.raises(ValueError, match="exceeds the"):
        srv._guard_state_size(big_dict)


def test_guard_encodes_list_state():
    srv._guard_state_size(["short", "list"])
    with pytest.raises(ValueError, match="exceeds the"):
        srv._guard_state_size(["word"] * 400)


@pytest.mark.parametrize("tool,args", [
    ("evaluate_decision", {
        "state": _long_state(400),
        "questions": {"q": {"type": "noul", "instructions": "test"}},
    }),
    ("evaluate_pr_diff", {"diff": _long_state(400)}),
    ("check_guardrails", {"prompt": _long_state(400)}),
    ("route_task", {"request": _long_state(400)}),
    ("moderate_content", {"text": _long_state(400)}),
])
def test_oversized_state_returns_error(tool, args):
    result = _run(srv.call_tool(tool, args))
    parsed = _parse_err(result)
    assert "exceeds" in parsed["error"]


def test_unknown_tool_returns_error():
    result = _run(srv.call_tool("nonexistent_tool", {}))
    parsed = _parse_err(result)
    assert "Unknown tool" in parsed["error"]


def test_missing_required_args_returns_error():
    result = _run(srv.call_tool("evaluate_decision", {"questions": {}}))
    parsed = _parse_err(result)
    assert "Missing required argument(s): state" in parsed["error"]


def test_import_does_not_load_model():
    _laya_stub.load.assert_not_called()


def test_enrich_adds_confidence_summary():
    result = {
        "answers": {
            "intent": {"choice": "billing", "confidence": 0.94},
            "churn": {"noul": 0.88},
        }
    }
    enriched = srv._enrich(result)
    assert enriched["confidence_summary"] == {"intent": 0.94}
    assert enriched["confidence_threshold"] == srv.DEFAULT_CONFIDENCE_THRESHOLD


def test_enrich_no_summary_when_no_confidence():
    result = {"answers": {"churn": {"noul": 0.88}}}
    enriched = srv._enrich(result)
    assert "confidence_summary" not in enriched
    assert enriched["confidence_threshold"] == srv.DEFAULT_CONFIDENCE_THRESHOLD


def test_enrich_none_answers():
    enriched = srv._enrich({"answers": None})
    assert "confidence_summary" not in enriched
    assert enriched["confidence_threshold"] == srv.DEFAULT_CONFIDENCE_THRESHOLD


def test_enrich_does_not_mutate_original():
    result = {"answers": {"intent": {"choice": "billing", "confidence": 0.94}}}
    original_keys = set(result.keys())
    srv._enrich(result)
    assert set(result.keys()) == original_keys


def test_enrich_passthrough_non_dict():
    assert srv._enrich("raw string") == "raw string"
    assert srv._enrich(None) is None


def test_list_tools_names():
    tools = _run(srv.list_tools())
    assert {t.name for t in tools} == {
        "evaluate_decision",
        "evaluate_pr_diff",
        "check_guardrails",
        "route_task",
        "moderate_content",
    }

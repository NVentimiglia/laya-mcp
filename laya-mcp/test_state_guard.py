"""
Unit tests for the state size guard in laya_mcp_server.

These tests run without a loaded Laya model. A stub `laya` module
stands in before import, and a fake agent with a whitespace tokenizer
replaces the real one.

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
from mcp import types as mcp_types

_laya_stub = types.ModuleType("laya")
_laya_stub.load = MagicMock(return_value=MagicMock())
_laya_stub.guard_questions = MagicMock(return_value={"q": {"type": "noul"}})
_laya_stub.router_questions = MagicMock(return_value={"q": {"type": "noul"}})
_laya_stub.moderation_questions = MagicMock(return_value={"q": {"type": "noul"}})
sys.modules["laya"] = _laya_stub

import laya_mcp_server as srv  # noqa: E402

ROOM = 100
NOUL = {"q": {"type": "noul", "instructions": "test"}}


class FakeAgent:
    """Whitespace tokenizer; predict echoes a fixed answer."""

    def tok(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}

    def predict(self, state, questions):
        return {"answers": {"q": {"noul": 0.1, "confidence": 0.9}}}


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    monkeypatch.setattr(srv, "agent", FakeAgent())
    monkeypatch.setattr(srv, "_state_room", lambda questions: ROOM)


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _run(coro):
    return asyncio.run(coro)


def test_state_tokens_uses_tokenizer():
    assert srv._state_tokens(_words(7)) == 7


def test_state_tokens_counts_json_like_laya():
    # Laya serializes dicts with json.dumps before tokenizing.
    assert srv._state_tokens({"a": "b c"}) == len(json.dumps({"a": "b c"}).split())


def test_guard_passes_at_room():
    srv._guard_state_size(_words(ROOM), NOUL)


def test_guard_raises_past_room():
    with pytest.raises(ValueError, match="would drop the rest"):
        srv._guard_state_size(_words(ROOM + 1), NOUL)


def test_guard_catches_text_without_spaces(monkeypatch):
    # The old word-count estimate let unspaced text through.
    class CharTok(FakeAgent):
        def tok(self, text, add_special_tokens=False):
            return {"input_ids": list(text)}

    monkeypatch.setattr(srv, "agent", CharTok())
    with pytest.raises(ValueError):
        srv._guard_state_size("x" * (ROOM + 1), NOUL)


@pytest.mark.parametrize("tool,args", [
    ("evaluate_decision", {"state": _words(ROOM + 1), "questions": NOUL}),
    ("evaluate_pr_diff", {"diff": _words(ROOM + 1)}),
    ("check_guardrails", {"prompt": _words(ROOM + 1)}),
    ("route_task", {"request": _words(ROOM + 1)}),
    ("moderate_content", {"text": _words(ROOM + 1)}),
])
def test_oversized_state_raises(tool, args):
    with pytest.raises(ValueError, match="would drop the rest"):
        _run(srv.call_tool(tool, args))


def test_unknown_tool_raises():
    with pytest.raises(ValueError, match="Unknown tool"):
        _run(srv.call_tool("nonexistent_tool", {}))


def test_missing_required_args_raises():
    with pytest.raises(ValueError, match="Missing required argument"):
        _run(srv.call_tool("evaluate_decision", {"questions": {}}))


def test_empty_questions_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _run(srv.call_tool("evaluate_decision", {"state": "hi", "questions": {}}))


def test_unloaded_model_raises(monkeypatch):
    monkeypatch.setattr(srv, "agent", None)
    with pytest.raises(RuntimeError, match="not loaded"):
        _run(srv.call_tool("route_task", {"request": "hi"}))


def test_happy_path_returns_enriched_answers():
    result = _run(srv.call_tool("evaluate_decision", {"state": "hi", "questions": NOUL}))
    data = json.loads(result[0].text)
    assert data["answers"]["q"]["noul"] == 0.1
    assert data["confidence_summary"] == {"q": 0.9}


def _sdk_call(name: str, args: dict) -> mcp_types.CallToolResult:
    handler = srv.app.request_handlers[mcp_types.CallToolRequest]
    req = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=name, arguments=args),
    )
    return _run(handler(req)).root


def test_sdk_marks_errors_is_error():
    result = _sdk_call("route_task", {"request": _words(ROOM + 1)})
    assert result.isError is True
    assert "would drop the rest" in result.content[0].text


def test_sdk_success_not_is_error():
    result = _sdk_call("route_task", {"request": "What is 2 + 2?"})
    assert result.isError is False


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

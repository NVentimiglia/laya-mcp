"""
Laya MCP Server — non-autoregressive System 1 decisions over stdio.

Logs go to stderr. stdout is JSON-RPC only.

Copyright 2026 Nicholas Ventimiglia (NVentimiglia). MIT license.

  pip install "mcp[cli]" laya torch
  python laya_mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

import laya

MODEL_ID = os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya")
MAX_STATE_TOKENS = int(os.environ.get("LAYA_MAX_STATE_TOKENS", "512"))
DEFAULT_CONFIDENCE_THRESHOLD = float(
    os.environ.get("LAYA_CONFIDENCE_THRESHOLD", "0.85")
)

app = Server("laya-decision-engine")
agent: Any = None

PR_QUESTIONS: dict[str, Any] = {
    "safety": {
        "type": "choice",
        "instructions": "Is this PR diff safe to merge?",
        "criteria": {
            "safe": "No security concerns, no logic regressions.",
            "risky": "Possible regression or subtle bug; warrants review.",
            "unsafe": "Clear vulnerability, credential leak, or breaking change.",
        },
    },
    "risk_level": {
        "type": "score",
        "instructions": "Risk this PR introduces bugs or regressions?",
        "criteria": ["negligible", "low", "medium", "high", "critical"],
    },
    "needs_review": {
        "type": "noul",
        "instructions": "Does this PR need human review before merge?",
    },
    "action": {
        "type": "choice",
        "instructions": "What action should be taken on this PR?",
        "criteria": {
            "merge": "Safe to merge immediately.",
            "request_changes": "Changes needed before merging.",
            "escalate": "Needs senior or security review.",
        },
    },
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_engine() -> None:
    global agent
    if agent is None:
        _log(f"Loading Laya model '{MODEL_ID}' into memory…")
        agent = laya.load(MODEL_ID)
        _log("Laya model ready.")


def _token_estimate(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _guard_state_size(state: Any) -> None:
    raw = state if isinstance(state, str) else json.dumps(state)
    estimate = _token_estimate(raw)
    if estimate > MAX_STATE_TOKENS:
        raise ValueError(
            f"State is approximately {estimate} tokens, which exceeds the "
            f"{MAX_STATE_TOKENS}-token limit. Trim the state before calling Laya."
        )


def _require(args: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in args or args[k] is None]
    if missing:
        raise ValueError("Missing required argument(s): " + ", ".join(missing))


def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def _err(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": message}, indent=2))]


def _enrich(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    answers = result.get("answers")
    if not isinstance(answers, dict):
        answers = {}
    summary = {
        key: val["confidence"]
        for key, val in answers.items()
        if isinstance(val, dict) and "confidence" in val
    }
    result = dict(result)
    result["confidence_threshold"] = DEFAULT_CONFIDENCE_THRESHOLD
    if summary:
        result["confidence_summary"] = summary
    return result


async def _predict(state: Any, questions: dict[str, Any]) -> list[TextContent]:
    _guard_state_size(state)
    if agent is None:
        raise RuntimeError("Laya model is not loaded. Call load_engine() first.")
    result = await asyncio.to_thread(agent.predict, state, questions)
    return _ok(_enrich(result))


TOOLS = [
    Tool(
        name="evaluate_decision",
        description=(
            "Typed System-1 decisions (choice / score / noul) in one forward pass. "
            "Keep state under 512 tokens."
        ),
        inputSchema={
            "type": "object",
            "required": ["state", "questions"],
            "properties": {
                "state": {
                    "oneOf": [{"type": "string"}, {"type": "object"}],
                    "description": "Text or JSON to evaluate. Keep under 512 tokens.",
                },
                "questions": {
                    "type": "object",
                    "description": "Map of question key → {type, instructions, criteria?}.",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["type", "instructions"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["choice", "score", "noul"],
                            },
                            "instructions": {"type": "string"},
                            "criteria": {
                                "description": (
                                    "choice: label → description. "
                                    "score: ordered level names."
                                ),
                            },
                        },
                    },
                },
            },
        },
    ),
    Tool(
        name="evaluate_pr_diff",
        description="PR safety, risk, and merge-action verdict.",
        inputSchema={
            "type": "object",
            "required": ["diff"],
            "properties": {
                "diff": {"type": "string", "description": "Unified diff text."},
                "context": {"type": "string", "description": "Optional PR notes."},
            },
        },
    ),
    Tool(
        name="check_guardrails",
        description="Jailbreak / injection / safety screen.",
        inputSchema={
            "type": "object",
            "required": ["prompt"],
            "properties": {"prompt": {"type": "string"}},
        },
    ),
    Tool(
        name="route_task",
        description="Route a request to light vs frontier.",
        inputSchema={
            "type": "object",
            "required": ["request"],
            "properties": {"request": {"type": "string"}},
        },
    ),
    Tool(
        name="moderate_content",
        description="Toxicity, harassment, and threat screen.",
        inputSchema={
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    args = arguments or {}
    try:
        if name == "evaluate_decision":
            _require(args, "state", "questions")
            return await _predict(args["state"], args["questions"])
        if name == "evaluate_pr_diff":
            _require(args, "diff")
            diff = args["diff"]
            context = args.get("context") or ""
            state = (
                f"PR diff:\n{diff}\n\nContext:\n{context}"
                if context
                else f"PR diff:\n{diff}"
            )
            return await _predict(state, PR_QUESTIONS)
        if name == "check_guardrails":
            _require(args, "prompt")
            return await _predict({"prompt": args["prompt"]}, laya.guard_questions())
        if name == "route_task":
            _require(args, "request")
            return await _predict({"request": args["request"]}, laya.router_questions())
        if name == "moderate_content":
            _require(args, "text")
            return await _predict({"post": args["text"]}, laya.moderation_questions())
        raise ValueError(f"Unknown tool: {name!r}")
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:
        return _err(f"Laya inference error: {type(exc).__name__}: {exc}")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream, app.create_initialization_options()
        )


def main_sync() -> None:
    load_engine()
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()

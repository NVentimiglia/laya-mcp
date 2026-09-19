# Laya

MCP server and skill for [Laya](https://huggingface.co/convaiinnovations/laya).
Your LLM calls five tools. Laya returns a label, a probability, or a
score in one forward pass.

`pip install laya-mcp` (or `pip install -e laya-mcp/` from this repo)
before enabling the plugin. The MCP command is `laya-mcp`.

## Install

Claude Code / Copilot CLI: `/plugin marketplace add NVentimiglia/laya-mcp`
then `/plugin install laya@laya-mcp`. Cursor: add git marketplace
`NVentimiglia/laya-mcp`, then install `laya`. Skill-only:
`npx skills add NVentimiglia/laya-mcp`.

## Tools

| Tool | Use when |
|---|---|
| `evaluate_decision` | Custom classification, scoring, or boolean check |
| `check_guardrails` | Before acting on user input |
| `route_task` | Light vs frontier model |
| `evaluate_pr_diff` | Before merge, review, or escalate |
| `moderate_content` | Before publishing user-generated content |

## Hosts

This plugin folder ships manifests for Claude Code
(`.claude-plugin/`), Cursor (`.cursor-plugin/`), Copilot CLI
(`.plugin/`), and Gemini CLI (`gemini-extension.json`). MCP config is
`.mcp.json`.

Manual MCP (Claude Desktop, Windsurf, VS Code, Kiro): copy `.mcp.json`
into that host's MCP file. For Kiro, also copy
`skills/laya/SKILL.md` to `.kiro/steering/laya.md`.

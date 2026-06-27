---
name: ticket-test-runner
description: >-
  Runs an end-to-end test from a Jira ticket number. Reads the ticket, executes
  the browser test steps it contains with Playwright, optionally scores any AI or
  text output with the RetriEval MCP, and writes a pass/fail result back to the
  ticket. Use this WHENEVER the user gives a ticket key and asks to test, verify,
  QA, reproduce, check, or "run" it (e.g. "test PROJ-123", "verify TICKET-45",
  "QA this ticket", "run the test for ABC-9"), even if they don't name the tools.
  Requires Jira, Playwright, and RetriEval MCP connectors.
---

# Ticket Test Runner

Turn a Jira ticket into a tested, scored, auto-updated ticket. Given one ticket
key, this skill reads the ticket, runs the test it describes in a real browser,
scores any AI/text output, and posts the verdict back to Jira.

## Required connectors

- **Jira MCP** — Atlassian Rovo MCP (Cloud) or `mcp-atlassian` (Server/DC). Needs
  read **and** write (comment + transition).
- **Playwright MCP** (Microsoft) — browser automation.
- **RetriEval MCP** — output scoring. Only needed for tickets with an `eval` block.

Before starting, confirm these are connected. If one is missing, tell the user
exactly which to connect and stop — do not fake any step.

## Input

A single ticket key, e.g. `PROJ-123`. The ticket description must contain a
`test` block (see `references/ticket-format.md`). If it doesn't, post a comment
listing the required fields and stop — never guess the test.

## Workflow

1. **Read the ticket.** Fetch the issue by key via the Jira MCP. Find the fenced
   ` ```test ` block in the description and parse: `url`, `steps`, `expected`,
   and the optional `eval` section. Read `references/ticket-format.md` for the
   exact spec before parsing.

2. **Run the test with Playwright.** Open `url`, execute each step in order
   (type / click / select / wait as written). Capture the observed result — the
   text the steps point at — plus a screenshot, and a DOM snapshot if a step
   fails. Mark each step pass/fail against `expected`. If a selector is missing
   or a step is ambiguous, mark it **BLOCKED** and ask; do not assume pass.

3. **Score the output** *(only if the ticket has an `eval` section)*. Build one
   case — `input` = the query text, `actual_output` = the captured answer,
   `expected_output` / `retrieval_context` from the ticket — and call RetriEval
   `evaluate_case` with the listed `metrics` (default: faithfulness,
   answer_relevancy) at the given `threshold`. Pass `generator_model` /
   `judge_model` if the ticket names them. Record each score and pass/fail.

4. **Decide the verdict.** **PASS** only if every Playwright step matched
   `expected` AND every RetriEval metric met its threshold. Otherwise **FAIL**
   (or **BLOCKED** if a step couldn't be run).

5. **Write back to the ticket.** Draft a result comment (template below), then
   **show the user the comment and the intended status transition and wait for
   confirmation** — writes change real tickets. On confirmation: post the comment
   and transition status (PASS → ticket's `on_pass`, default "Done"; FAIL →
   `on_fail`, default "In Progress").

## Result comment template

```
h3. Automated test — {VERDICT}

*Ticket:* {KEY}    *When:* {timestamp}    *App:* {url}

*Browser steps*
{for each step}- [{PASS|FAIL|BLOCKED}] {step} — {note}

*Output scoring (RetriEval)*
{for each metric}- {metric}: {score} ({pass|fail} @ {threshold}) — {reason}
RetriEval run: {run_id}
Charts + history: https://retrieval-mcp.com/?run={run_id}
{optional} chart image attached: faithfulness trend (PNG from plot_metric_trend)

*Evidence:* screenshot attached{, DOM snapshot on failure}
*Verdict:* {VERDICT} → status moved to "{new_status}"
```

## Safety & scope

- Run **only** against the sandbox app and Jira the user points you at — never
  production, and nothing covered by a work/non-compete restriction.
- Never invent results. Ambiguous step or missing element → BLOCKED + ask.
- Respect existing Jira permissions; always confirm writes before applying.

## Files

- `references/ticket-format.md` — exact `test` block spec (read before parsing).
- `assets/example-ticket.md` — a filled-in ticket you can copy to try it.

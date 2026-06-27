# Ticket `test` block format

Put a fenced block labelled `test` in the ticket description. It's YAML-style.
The skill parses this to know what to run — so the ticket carries everything the
test needs.

## Fields

| Field | Required | Meaning |
|-------|----------|---------|
| `url` | yes | Page to open. Sandbox/staging only. |
| `steps` | yes | Ordered browser actions (see verbs below). |
| `expected` | yes | What proves the step/test passed (plain text). |
| `eval` | no | Score an AI/text answer with RetriEval. |
| `on_pass` | no | Jira status to move to on PASS (default `Done`). |
| `on_fail` | no | Jira status to move to on FAIL (default `In Progress`). |

## Step verbs

- `goto: <url>` — navigate
- `type: "<selector>" "<text>"` — type into an element
- `click: "<selector>"` — click
- `select: "<selector>" "<value>"` — choose an option
- `wait_for: "<selector>"` — wait until visible
- `expect_text: "<selector>" contains "<text>"` — inline assertion

Selectors are CSS or accessible names (Playwright MCP resolves both).

## `eval` section (optional)

```
eval:
  metrics: [faithfulness, answer_relevancy]   # any RetriEval metric
  threshold: 0.7
  capture: "<selector>"           # element whose text is the answer under test
  retrieval_context:              # what the answer should be grounded in
    - "Our refund policy allows returns within 30 days."
  expected_output: "Returns are accepted within 30 days."
  generator_model: "GPT-4o"       # optional, for the dashboard
  judge_model: "claude-sonnet-4-6"
```

If `eval` is present, the text from `capture` becomes `actual_output`, and the
skill calls RetriEval `evaluate_case`. PASS requires every metric ≥ `threshold`.

## Minimal valid block

```test
url: https://sandbox.example.com/chat
steps:
  - goto: https://sandbox.example.com/chat
  - type: "#question" "What is your refund policy?"
  - click: "#send"
  - wait_for: ".answer"
expected: the answer mentions a 30-day window
```

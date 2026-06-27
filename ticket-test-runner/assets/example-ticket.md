# Example ticket — SUPPORT-101

**Summary:** Support chatbot must answer the refund-policy question correctly and
stay grounded in the policy doc.

**Description:**

The chatbot should answer refund questions using only the policy text, and must
mention the 30-day window without inventing other terms.

```test
url: https://sandbox.example.com/chat
steps:
  - goto: https://sandbox.example.com/chat
  - type: "#question" "Can I get a refund after three weeks?"
  - click: "#send"
  - wait_for: ".answer"
  - expect_text: ".answer" contains "30 days"
expected: the answer confirms refunds within 30 days and nothing contradicting the policy
eval:
  metrics: [faithfulness, answer_relevancy]
  threshold: 0.7
  capture: ".answer"
  retrieval_context:
    - "Our refund policy allows returns within 30 days of delivery."
    - "Final-sale items are not eligible for refunds."
  expected_output: "Yes — refunds are accepted within 30 days of delivery."
  generator_model: "GPT-4o"
  judge_model: "claude-sonnet-4-6"
on_pass: Done
on_fail: In Progress
```

**How to run:** tell the agent — *"test SUPPORT-101"*.

Expected behavior: the skill reads this ticket, drives the chat in a browser,
asserts the answer mentions 30 days, scores it with RetriEval (faithfulness +
answer_relevancy ≥ 0.7), then comments the verdict and moves the ticket — after
you confirm.

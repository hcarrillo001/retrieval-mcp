# Dev.to / Hashnode post outline — RetriEval

Working title options:
- "I built a local-first eval MCP — and what it taught me about LLM-as-judge"
- "Testing RAG by stage: separating retriever failures from generator failures"
- "Evals as an MCP: letting an agent grade its own work"

Target length: 1,200–1,800 words. One demo GIF up top. Code blocks > prose.

## 1. Hook (the problem) — ~150 words
- "The answer is bad" is not an actionable bug report. In a RAG system, was it the
  retriever or the generator? Most eval setups blur the two.
- Frame: you're a QE who tests AI systems; you wanted eval *infrastructure* an agent
  could call, not another library you wire by hand.

## 2. What I built — ~150 words
- RetriEval: a local-first LLM-eval MCP. One paragraph + the 4-step flow diagram
  (ticket → browser → score → ticket). Link the repo + live dashboard early.

## 3. The core idea: stage-separated RAG eval — ~350 words (the centerpiece)
- Show the demo table: one case where the retriever missed (recall 0, faithfulness 1)
  and one where the generator hallucinated (recall 1, faithfulness 0).
- Explain why this matters: independent metrics point at the actual culprit.
- This is the section that signals seniority — spend the most words here.

## 4. LLM-as-judge, honestly — ~250 words
- Reason-before-score, why it cuts variance. Position bias in pairwise. The
  reliability question (judge vs human agreement) — name the limitation; don't oversell.
- Swappable judge: Claude or local DeepSeek for $0.

## 5. Authoring metrics from plain language — ~200 words
- "Describe the metric, it writes the rubric." Show author_metric in/out.

## 6. Making it an MCP (and why that's the point) — ~250 words
- Tools as the interface; the agent decides when to call them.
- The orchestration demo: Cursor + Playwright + RetriEval + Jira, end to end.
- Hosted vs local, the deploy story in two lines (Railway + Supabase + Vercel).

## 7. What I'd do differently / what's hard — ~150 words
- Capturing retrieved context for faithfulness. Judge cost/latency. Calibration.
- Honesty here reads as credibility — interviewers love a real "limitations" section.

## 8. Close + CTA — ~80 words
- Repo, dashboard, "try it: pip install retrieval-mcp". Invite issues/PRs.

---

## Distribution checklist
- [ ] Cross-post Dev.to + Hashnode (canonical URL set to one).
- [ ] 15s demo GIF (the Cursor loop or the dashboard) at the top.
- [ ] Share in r/LLMDevs, r/MachineLearning (rules permitting), and on LinkedIn with
      a 3-line "why I built this."
- [ ] Pin the repo on your GitHub profile; put the post link in the repo README.
- [ ] Add the line to your resume: "Built RetriEval, an open-source LLM-eval MCP with
      stage-separated RAG metrics and an agentic test-to-ticket pipeline."

## Interview framing (have these ready)
- "Why an MCP and not a library?" → agent-callable, client-agnostic, reusable.
- "How do you trust the judge?" → reason-before-score, calibration vs golden, kappa.
- "Retriever vs generator?" → the demo table; what each metric isolates.
- "How would this scale on a team?" → CI gate (Demo 1) + manual review (Demo 2),
  shared Supabase history, dashboard.

"""
Orchestrates the full per-PDF compliance pipeline: Stage A -> Stage B -> Stage C.

Not yet implemented.

Intended role: given an uploaded drawing PDF, for each page run
  Stage A: classifier_agent.classify_page  (drawing type)
  Stage B: feature_agent (compliance-relevant fact extraction, once implemented)
  Stage C: matcher.match_clauses + judge.judge_fact (once implemented)
then aggregate the per-page verdicts into a single pass/fail/flag report for
the whole document. This module is the single entry point the rest of the
dissertation's evaluation harness should call.
"""

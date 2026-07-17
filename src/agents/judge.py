"""
Stage C (part 2) of the pipeline: compliance judgement.

Given a fact (from feature_agent) and its candidate regulation excerpts (from
matcher.py), judges whether the fact satisfies, violates, or cannot be
conclusively checked against those excerpts. A candidate excerpt is never
assumed relevant just because matcher.py found it by keyword overlap --
this stage can and should say "no_matching_clause" if none of the candidates
actually address the fact, rather than forcing a verdict.

Output is one Verdict per fact: pass / fail / flag / no_matching_clause, with
a natural-language justification citing the specific excerpt relied on (if
any). pipeline.py will aggregate these into the final per-page and
per-document compliance report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agents.matcher import RegulationChunk, find_candidate_clauses, load_regulation_chunks
from src.openrouter_client import OpenRouterClient

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-20260507"

VALID_VERDICTS = {"pass", "fail", "flag", "no_matching_clause"}


@dataclass
class Verdict:
    """The compliance judgement for one extracted fact."""

    parameter_name: str
    value: str
    unit: str | None
    verdict: str  # "pass" | "fail" | "flag" | "no_matching_clause"
    reasoning: str
    cited_source_document: str | None
    cited_clause_text: str | None
    page_number: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vprint(verbose: bool, *lines: str) -> None:
    if not verbose:
        return
    for line in lines:
        print(line)


def _build_judge_prompt(fact: dict[str, Any], candidates: list[RegulationChunk]) -> str:
    if candidates:
        candidate_block = "\n\n".join(
            f'[{i}] (from {c.source_document}):\n"{c.text}"' for i, c in enumerate(candidates)
        )
    else:
        candidate_block = "(no candidate excerpts were found for this fact)"

    return f"""You are checking one fact extracted from an architectural drawing against candidate
excerpts from building-regulation documents. The candidates were found by keyword search and may or
may not actually be relevant -- do not assume they apply just because they were retrieved.

FACT (from the drawing):
- parameter: {fact['parameter_name']}
- value: {fact['value']} {fact.get('unit') or ''}
- as stated on the drawing: "{fact.get('source_quote', '')}"

CANDIDATE REGULATION EXCERPTS:
{candidate_block}

Decide exactly one verdict:
- "pass": a candidate excerpt states a specific requirement, and this fact's value satisfies it.
- "fail": a candidate excerpt states a specific requirement, and this fact's value violates it.
- "flag": a candidate excerpt seems related but doesn't give enough information to conclusively
  decide pass or fail (ambiguous wording, missing context, or it's only tangentially about this).
- "no_matching_clause": none of the candidate excerpts actually address this specific fact -- use
  this rather than forcing a pass/fail/flag onto an unrelated excerpt.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{
  "verdict": "pass",
  "reasoning": "<one or two sentences, citing the excerpt number in brackets if you used one>",
  "cited_excerpt_index": 0
}}
"cited_excerpt_index" should be null if verdict is "no_matching_clause" or no excerpt was actually used."""


def _parse_judge_response(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge did not return valid JSON: {raw_text!r}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Judge response was not a JSON object: {raw_text!r}")
    return data


def judge_fact(
    fact: dict[str, Any],
    candidates: list[RegulationChunk],
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Verdict:
    """Judge one fact against its candidate regulation excerpts."""
    prompt = _build_judge_prompt(fact, candidates)
    raw_response = client.complete(model=model, prompt=prompt)
    _vprint(verbose, f"    [Judge] {fact['parameter_name']}={fact['value']} -> " + raw_response.replace("\n", " "))

    data = _parse_judge_response(raw_response)
    verdict = data.get("verdict", "flag")
    if verdict not in VALID_VERDICTS:
        verdict = "flag"

    cited_idx = data.get("cited_excerpt_index")
    cited_doc: str | None = None
    cited_text: str | None = None
    if isinstance(cited_idx, int) and 0 <= cited_idx < len(candidates):
        cited_doc = candidates[cited_idx].source_document
        cited_text = candidates[cited_idx].text

    return Verdict(
        parameter_name=fact["parameter_name"],
        value=fact["value"],
        unit=fact.get("unit"),
        verdict=verdict,
        reasoning=data.get("reasoning", ""),
        cited_source_document=cited_doc,
        cited_clause_text=cited_text,
        page_number=fact.get("page_number", 0),
    )


def judge_features_for_pdf(
    pdf_path: str,
    client: OpenRouterClient | None = None,
    model: str = DEFAULT_MODEL,
    output_dir: str = "outputs",
    regulations_path: str = "knowledge_base/regulations_combined.txt",
    limit: int | None = None,
    verbose: bool = False,
) -> list[Verdict]:
    """Run Stage C over every fact already saved in features.json for a PDF.

    Requires feature_agent.extract_features_for_pdf to have already been run
    (so features.json exists under output_dir/<pdf-stem>/). `limit` caps how
    many facts get judged, useful for testing before paying for a full run.
    """
    client = client or OpenRouterClient()
    pdf_out_dir = Path(output_dir) / Path(pdf_path).stem

    features_path = pdf_out_dir / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found -- run feature_agent on this PDF first (Stage B before Stage C)."
        )
    facts = json.loads(features_path.read_text(encoding="utf-8"))
    if limit is not None:
        facts = facts[:limit]

    chunks = load_regulation_chunks(regulations_path)
    _vprint(verbose, f"[Station 0] Loaded {len(facts)} fact(s) and {len(chunks)} regulation chunk(s)")

    verdicts: list[Verdict] = []
    for fact in facts:
        candidates = find_candidate_clauses(fact, chunks)
        _vprint(
            verbose,
            f"  [Match] {fact['parameter_name']}={fact['value']} -> {len(candidates)} candidate(s)",
        )
        verdict = judge_fact(fact, candidates, client, model=model, verbose=verbose)
        verdicts.append(verdict)

    verdicts_path = pdf_out_dir / "verdicts.json"
    verdicts_path.write_text(json.dumps([v.to_dict() for v in verdicts], indent=2))
    _vprint(verbose, f"\n[Station 1] Saved '{verdicts_path}' ({len(verdicts)} verdict(s))")

    counts = {key: 0 for key in VALID_VERDICTS}
    for v in verdicts:
        counts[v.verdict] += 1
    summary = {
        "total": len(verdicts),
        "counts": counts,
        "failures": [v.to_dict() for v in verdicts if v.verdict == "fail"],
        "flagged": [v.to_dict() for v in verdicts if v.verdict == "flag"],
    }
    summary_path = pdf_out_dir / "verdicts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _vprint(verbose, f"[Station 1] Saved '{summary_path}' -- counts: {counts}")

    return verdicts


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print("Usage: python -m src.agents.judge <path-to-pdf> [limit]")
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) == 3 else None

    verdicts = judge_features_for_pdf(pdf_path, limit=limit, verbose=True)

    by_verdict: dict[str, int] = {}
    for v in verdicts:
        by_verdict[v.verdict] = by_verdict.get(v.verdict, 0) + 1
    print(f"\n=== SUMMARY: {len(verdicts)} verdict(s) -- {by_verdict} ===")
    for v in verdicts:
        print(json.dumps(v.to_dict()))

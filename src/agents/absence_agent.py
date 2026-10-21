"""
Stage D of the pipeline: absence detection.

Stages A-C only ever evaluate facts that are already on the drawing (Stage C
judges whether a *present* value satisfies a regulation; it has no way to
notice that a required item was never shown at all). This stage asks a
different question: given this drawing's type, what does the knowledge base
say should be specified for it, and which of those items has no corresponding
extracted fact anywhere on the drawing?

Cost-aware by design: one retrieval + one LLM call per unique
(page_class, subtype) group in the drawing, not per page, per fact, or per
regulation chunk -- grouping facts first keeps this roughly as cheap as a
single Stage C judgement, not hundreds of them.

Honesty matches the rest of the pipeline: the knowledge base currently only
has real clause text for a few of the ~15 NI Technical Booklets (see
knowledge_base/regulations_combined.txt). For a drawing area the knowledge
base doesn't actually cover, this stage should say so ("not_applicable")
rather than inventing a citation -- the same philosophy behind Stage C's
"no_matching_clause".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agents.matcher import (
    RegulationChunk,
    find_candidate_clauses_for_terms,
    load_regulation_chunks,
)
from src.openrouter_client import OpenRouterClient

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-20260507"

VALID_CONFIDENCES = {"likely_missing", "possibly_missing", "not_applicable"}

_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "be", "is",
    "are", "with", "at", "by", "this", "that", "as", "it", "its", "from",
}


@dataclass
class MissingItem:
    """One requirement that appears to have no corresponding fact on the drawing."""

    page_class: str
    page_subtype: str
    requirement: str
    expected_parameter_hint: str | None
    confidence: str  # "likely_missing" | "possibly_missing" | "not_applicable"
    reasoning: str
    cited_source_document: str | None
    cited_clause_text: str | None
    page_numbers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vprint(verbose: bool, *lines: str) -> None:
    if not verbose:
        return
    for line in lines:
        print(line)


def _group_terms(page_class: str, subtype: str, covered_params: list[str]) -> set[str]:
    import re

    term_source = " ".join([page_class, subtype] + [p.replace("_", " ") for p in covered_params])
    words = set(re.findall(r"[a-z0-9]+", term_source.lower()))
    return words - _STOPWORDS


def _build_absence_prompt(
    page_class: str,
    subtype: str,
    covered_params: list[str],
    candidates: list[RegulationChunk],
) -> str:
    covered_block = ", ".join(covered_params) if covered_params else "(none extracted)"

    if candidates:
        candidate_block = "\n\n".join(
            f'[{i}] (from {c.source_document}):\n"{c.text}"' for i, c in enumerate(candidates)
        )
    else:
        candidate_block = "(no candidate excerpts were found for this drawing type)"

    return f"""You are checking an architectural drawing for required items that appear to be
MISSING entirely -- not wrong values on something present, but things the regulations expect
this drawing type to specify that have no corresponding entry at all.

DRAWING TYPE: {page_class} / {subtype}

PARAMETERS ALREADY EXTRACTED FROM THIS DRAWING (already covered -- do NOT flag these as missing,
even if the wording differs slightly from how a clause phrases it):
{covered_block}

CANDIDATE REGULATION EXCERPTS for this drawing type (found by keyword search -- may or may not
actually be relevant or applicable; do not assume they apply just because they were retrieved):
{candidate_block}

For each excerpt that states a specific, checkable requirement for THIS drawing type (not a
generic administrative rule) and has no match among the already-extracted parameters, list it as
a missing item. If an excerpt doesn't actually apply to this drawing type, or the knowledge base
genuinely has nothing relevant to check this drawing type against, do not force a result --
return an empty list or mark confidence "not_applicable".

Respond with ONLY a JSON array, no markdown fences, no commentary. Each element:
{{
  "requirement": "<short name of the missing requirement, e.g. 'fire door rating'>",
  "expected_parameter_hint": "<what a parameter_name would look like if it had been extracted, or null>",
  "confidence": "likely_missing" | "possibly_missing" | "not_applicable",
  "reasoning": "<one or two sentences, citing the excerpt number in brackets if you used one>",
  "cited_excerpt_index": 0
}}
"cited_excerpt_index" should be null if no excerpt was actually used (e.g. confidence is
"not_applicable")."""


def _parse_absence_response(raw_text: str) -> list[dict[str, Any]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Absence check did not return valid JSON: {raw_text!r}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Absence check response was not a JSON array: {raw_text!r}")
    return data


def detect_missing_items_for_group(
    page_class: str,
    subtype: str,
    covered_params: list[str],
    page_numbers: list[int],
    chunks: list[RegulationChunk],
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> list[MissingItem]:
    """Run the absence check for one (page_class, subtype) group."""
    terms = _group_terms(page_class, subtype, covered_params)
    candidates = find_candidate_clauses_for_terms(terms, chunks, top_k=8)
    _vprint(verbose, f"  [Match] {page_class}/{subtype} -> {len(candidates)} candidate(s)")

    prompt = _build_absence_prompt(page_class, subtype, covered_params, candidates)
    raw_response = client.complete(model=model, prompt=prompt)
    _vprint(verbose, f"  [Absence] {page_class}/{subtype} -> " + raw_response.replace("\n", " "))

    items = _parse_absence_response(raw_response)
    results: list[MissingItem] = []
    for item in items:
        confidence = item.get("confidence", "possibly_missing")
        if confidence not in VALID_CONFIDENCES:
            confidence = "possibly_missing"
        if confidence == "not_applicable":
            continue  # nothing to report -- an honest "no finding", not a result

        cited_idx = item.get("cited_excerpt_index")
        cited_doc: str | None = None
        cited_text: str | None = None
        if isinstance(cited_idx, int) and 0 <= cited_idx < len(candidates):
            cited_doc = candidates[cited_idx].source_document
            cited_text = candidates[cited_idx].text

        results.append(
            MissingItem(
                page_class=page_class,
                page_subtype=subtype,
                requirement=item.get("requirement", ""),
                expected_parameter_hint=item.get("expected_parameter_hint"),
                confidence=confidence,
                reasoning=item.get("reasoning", ""),
                cited_source_document=cited_doc,
                cited_clause_text=cited_text,
                page_numbers=page_numbers,
            )
        )
    return results


def detect_missing_items_for_pdf(
    pdf_path: str,
    client: OpenRouterClient | None = None,
    model: str = DEFAULT_MODEL,
    output_dir: str = "outputs",
    regulations_path: str = "knowledge_base/regulations_combined.txt",
    verbose: bool = False,
) -> list[MissingItem]:
    """Run Stage D over a PDF's already-saved classification.json and features.json.

    Requires classifier_agent.classify_pdf and feature_agent.extract_features_for_pdf
    to have already run (Stage A and B before Stage D). Does not require Stage C.
    """
    client = client or OpenRouterClient()
    pdf_out_dir = Path(output_dir) / Path(pdf_path).stem

    classification_path = pdf_out_dir / "classification.json"
    features_path = pdf_out_dir / "features.json"
    if not classification_path.exists():
        raise FileNotFoundError(
            f"{classification_path} not found -- run classifier_agent on this PDF first (Stage A before Stage D)."
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found -- run feature_agent on this PDF first (Stage B before Stage D)."
        )

    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    facts = json.loads(features_path.read_text(encoding="utf-8"))

    # Group pages by (page_class, subtype); collect the parameter names already
    # extracted and the page numbers covered by each group.
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for page in classification:
        key = (page.get("page_class", ""), page.get("subtype", ""))
        group = groups.setdefault(key, {"page_numbers": [], "covered_params": set()})
        group["page_numbers"].append(page.get("page_number"))
    for fact in facts:
        key = (fact.get("page_class", ""), fact.get("page_subtype", ""))
        group = groups.setdefault(key, {"page_numbers": [], "covered_params": set()})
        group["covered_params"].add(fact.get("parameter_name", ""))

    chunks = load_regulation_chunks(regulations_path)
    _vprint(
        verbose,
        f"[Stage D] {len(groups)} drawing-type group(s), {len(chunks)} regulation chunk(s) loaded",
    )

    results: list[MissingItem] = []
    for (page_class, subtype), group in groups.items():
        results.extend(
            detect_missing_items_for_group(
                page_class=page_class,
                subtype=subtype,
                covered_params=sorted(group["covered_params"]),
                page_numbers=sorted(p for p in group["page_numbers"] if p is not None),
                chunks=chunks,
                client=client,
                model=model,
                verbose=verbose,
            )
        )

    missing_items_path = pdf_out_dir / "missing_items.json"
    missing_items_path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    _vprint(verbose, f"\n[Stage D] Saved '{missing_items_path}' ({len(results)} missing item(s))")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.agents.absence_agent <path-to-pdf>")
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    missing_items = detect_missing_items_for_pdf(pdf_path, verbose=True)

    by_confidence: dict[str, int] = {}
    for m in missing_items:
        by_confidence[m.confidence] = by_confidence.get(m.confidence, 0) + 1
    print(f"\n=== SUMMARY: {len(missing_items)} missing item(s) -- {by_confidence} ===")
    for m in missing_items:
        print(json.dumps(m.to_dict()))

"""
Stage A of the pipeline: fused vector + vision page classification.

For a single PDF page, combines the raw vector text/geometry (vector_reader)
with a rendered image of the page (image_renderer) and asks the primary LLM
(google/gemini-3.1-flash-lite-20260507 by default) to classify the page's drawing type.
Downstream stages (feature extraction, clause matching, judgement) use this
classification to decide which extraction schema and which KB clauses apply.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.extraction.image_renderer import render_page_to_base64
from src.extraction.vector_reader import PageVectorData, extract_pdf_vector_data
from src.openrouter_client import OpenRouterClient

DEFAULT_CLASSIFIER_MODEL = "google/gemini-3.1-flash-lite-20260507"

# Taxonomy the model must classify each page into.
CLASS_TAXONOMY: dict[str, list[str]] = {
    "Architectural": ["Floor Plan", "Elevation", "Section", "Site Plan"],
    "Engineering": ["Structural Plan", "Electrical Plan", "Mechanical Plan"],
    "Finishes": ["Finishes Plan", "Reflected Ceiling Plan"],
    "Safety": ["Fire Strategy Plan"],
    "Not a drawing": [],
}

# Substring keywords (checked case-insensitively) that a source filename might
# contain, mapped to the subtype they imply. Used only to compute the QA-only
# `filename_mismatch` flag -- never to influence the classification itself.
FILENAME_KEYWORD_TO_SUBTYPE: dict[str, str] = {
    "elevation": "Elevation",
    "section": "Section",
    "site plan": "Site Plan",
    "structural": "Structural Plan",
    "electrical": "Electrical Plan",
    "mechanical": "Mechanical Plan",
    "finishes": "Finishes Plan",
    "reflected ceiling": "Reflected Ceiling Plan",
    "fire": "Fire Strategy Plan",
}

# Classifications below this confidence are flagged for human review rather
# than silently passed downstream. Kept as a module constant so the policy
# is visible and easy to tune from one place.
CONFIDENCE_THRESHOLD = 0.6


@dataclass
class ClassificationResult:
    """Structured output of the classifier agent for one detected drawing on a page."""

    page_class: str
    subtype: str | None
    confidence: float
    reasoning: str
    # QA-only signal: True if the source filename's implied subtype disagrees
    # with the model's returned subtype. Never fed back into classification.
    filename_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def needs_review(result: ClassificationResult, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Return True if `result`'s confidence is below `threshold` and should be human-reviewed."""
    return result.confidence < threshold


def _vprint(verbose: bool, *lines: str) -> None:
    if not verbose:
        return
    for line in lines:
        print(line)


def _build_prompt(page_data: PageVectorData) -> str:
    taxonomy_lines = []
    for page_class, subtypes in CLASS_TAXONOMY.items():
        if subtypes:
            taxonomy_lines.append(f'- "{page_class}": {", ".join(subtypes)}')
        else:
            taxonomy_lines.append(f'- "{page_class}"')
    taxonomy_block = "\n".join(taxonomy_lines)

    text_lines = [
        f'  - "{s.text}" at {tuple(round(c, 1) for c in s.bbox)}' for s in page_data.text_spans
    ]
    text_block = "\n".join(text_lines) if text_lines else "  (no extracted text)"

    return f"""You are classifying one page of a CAD-exported architectural drawing set.

You are given:
1. An image render of the page (for visual layout: title blocks, symbols, hatching, north arrows, line weights).
2. The page's extracted vector text, with each string's bounding box (x0, y0, x1, y1) in PDF points.

Classify the page into one class and subtype from this taxonomy:
{taxonomy_block}

Use "Not a drawing" for cover sheets, schedules-only pages, notes-only pages, or anything that is not
itself a drawing view.

Most pages contain a single distinct drawing and should return exactly one classification. However, if
this page genuinely combines multiple distinct drawing types on one sheet (for example, a combined
Finishes Plan + Reflected Ceiling Plan on one sheet), return one classification object per distinct
drawing type present. Do not split a single drawing into multiple classifications just because it
contains multiple views of the same type (e.g. two floor plans of different storeys on one sheet is
still one "Floor Plan" classification, not two).

Page size: {page_data.width:.1f} x {page_data.height:.1f} points.

Extracted text spans ({len(page_data.text_spans)} total):
{text_block}

Respond with ONLY a JSON object, no markdown fences, no commentary, matching exactly this shape:
{{
  "classifications": [
    {{
      "class": "<one of the class names above>",
      "subtype": "<one of the subtypes above, or null if the class has none / is Not a drawing>",
      "confidence": <float between 0 and 1>,
      "reasoning": "<one or two sentences citing the specific visual or textual evidence used>"
    }}
  ]
}}
Return a single-element "classifications" array for pages with one drawing type; return multiple
elements only when the page genuinely contains multiple distinct drawing types."""


def _parse_model_output(raw_text: str) -> list[dict[str, Any]]:
    """Parse the model's raw response into a list of classification dicts."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Classifier did not return valid JSON: {raw_text!r}") from exc

    if isinstance(data, list):
        # Tolerate a bare-array response.
        return data

    if not isinstance(data, dict):
        raise ValueError(f"Classifier response was not a JSON object: {raw_text!r}")

    classifications = data.get("classifications", [])
    if not isinstance(classifications, list):
        raise ValueError(f"'classifications' was not a JSON array: {raw_text!r}")

    return classifications


def _filename_mismatch(filename_hint: str | None, subtype: str | None) -> tuple[bool, str | None]:
    """QA-only check: does the filename's implied subtype disagree with the model's subtype?

    Returns (mismatch, note). Never used to alter the classification itself.
    """
    if not filename_hint:
        return False, None

    lower = filename_hint.lower()
    matched = {kw: sub for kw, sub in FILENAME_KEYWORD_TO_SUBTYPE.items() if kw in lower}
    if not matched:
        return False, None

    expected_subtypes = set(matched.values())
    if subtype in expected_subtypes:
        return False, None

    return True, (
        f"filename '{filename_hint}' suggests {sorted(expected_subtypes)} "
        f"but model returned subtype={subtype!r}"
    )


def _to_classification_results(
    raw_items: list[dict[str, Any]],
    filename_hint: str | None,
) -> list[ClassificationResult]:
    results: list[ClassificationResult] = []
    for item in raw_items:
        subtype = item.get("subtype")
        mismatch, mismatch_note = _filename_mismatch(filename_hint, subtype)

        reasoning = item.get("reasoning", "")
        if mismatch_note:
            reasoning = f"{reasoning} [QA: {mismatch_note}]"

        results.append(
            ClassificationResult(
                page_class=item["class"],
                subtype=subtype,
                confidence=float(item["confidence"]),
                reasoning=reasoning,
                filename_mismatch=mismatch,
            )
        )
    return results


def classify_page(
    page_data: PageVectorData,
    page_image_base64: str,
    client: OpenRouterClient,
    model: str = DEFAULT_CLASSIFIER_MODEL,
    filename_hint: str | None = None,
    verbose: bool = False,
) -> list[ClassificationResult]:
    """Classify a single page given its pre-extracted vector data and rendered image.

    Returns a list because a single page can genuinely contain more than one
    distinct drawing type (e.g. a combined Finishes + RCP sheet); most pages
    will return a list of length 1. `filename_hint` (the source PDF's
    filename) is used only to compute the QA-only `filename_mismatch` flag on
    each result -- it never affects the classification itself.
    """
    prompt = _build_prompt(page_data)
    _vprint(
        verbose,
        f"  [Station 2] Sending to model '{model}' -- prompt is {len(prompt)} chars, "
        f"embedding all {len(page_data.text_spans)} text spans, plus 1 image.",
    )

    raw_response = client.complete(
        model=model,
        prompt=prompt,
        image_base64=page_image_base64,
    )
    _vprint(
        verbose,
        "  [Station 2] Raw model response:",
        "  " + raw_response.replace("\n", "\n  "),
    )

    raw_items = _parse_model_output(raw_response)
    _vprint(verbose, f"  [Station 3] Parsed {len(raw_items)} classification object(s) from response.")

    results = _to_classification_results(raw_items, filename_hint)
    for result in results:
        _vprint(
            verbose,
            f"  [Station 4] class={result.page_class!r} subtype={result.subtype!r} "
            f"confidence={result.confidence} filename_mismatch={result.filename_mismatch}",
        )
    return results


def classify_pdf(
    pdf_path: str,
    client: OpenRouterClient | None = None,
    model: str = DEFAULT_CLASSIFIER_MODEL,
    verbose: bool = False,
    output_dir: str | None = "outputs",
) -> list[tuple[int, ClassificationResult, bool]]:
    """Convenience entry point: classify every page of a PDF end to end.

    Returns a flat list of (page_number, result, needs_review_flag) tuples --
    flat because a page can yield more than one ClassificationResult. Low-
    confidence results are surfaced via the needs_review flag rather than
    silently passed downstream.

    If `output_dir` is set (default "outputs"), every page's rendered image,
    raw vector extraction, and the final classification results are written
    to disk under `output_dir/<pdf-stem>/` -- so nothing is API-call-once,
    look-once. Pass output_dir=None to keep everything in memory only.
    """
    client = client or OpenRouterClient()
    filename_hint = Path(pdf_path).name

    pdf_out_dir: Path | None = None
    if output_dir is not None:
        pdf_out_dir = Path(output_dir) / Path(pdf_path).stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)
        _vprint(verbose, f"[Station 0] Saving artifacts to '{pdf_out_dir}'")

    _vprint(verbose, f"[Station 0] Opening '{filename_hint}'")
    pages = extract_pdf_vector_data(pdf_path)
    _vprint(verbose, f"[Station 0] Found {len(pages)} page(s)")

    results: list[tuple[int, ClassificationResult, bool]] = []
    for page_data in pages:
        _vprint(
            verbose,
            "",
            f"=== Page {page_data.page_number} ===",
            f"  [Station 1] Vector scan: {len(page_data.text_spans)} text span(s), "
            f"{len(page_data.vector_paths)} vector path(s), "
            f"page size {page_data.width:.1f} x {page_data.height:.1f} pts",
        )

        image_b64 = render_page_to_base64(pdf_path, page_data.page_number)
        _vprint(verbose, f"  [Station 1] Rendered image: {len(image_b64)} base64 chars")

        if pdf_out_dir is not None:
            page_tag = f"page_{page_data.page_number:03d}"

            vector_path = pdf_out_dir / f"{page_tag}_vector.json"
            vector_path.write_text(json.dumps(page_data.to_dict(), indent=2))

            image_path = pdf_out_dir / f"{page_tag}.png"
            image_path.write_bytes(base64.b64decode(image_b64))

            _vprint(
                verbose,
                f"  [Station 1] Saved '{vector_path.name}' and '{image_path.name}'",
            )

        page_results = classify_page(
            page_data, image_b64, client, model=model, filename_hint=filename_hint, verbose=verbose
        )

        for result in page_results:
            flagged = needs_review(result)
            _vprint(
                verbose,
                f"  [Station 5] confidence={result.confidence} threshold={CONFIDENCE_THRESHOLD} "
                f"-> needs_review={flagged}",
                f"  [Station 6] FINAL: {json.dumps(result.to_dict())}",
            )
            results.append((page_data.page_number, result, flagged))

    if pdf_out_dir is not None:
        results_path = pdf_out_dir / "classification.json"
        results_path.write_text(
            json.dumps(
                [
                    {"page_number": page_number, "needs_review": flagged, **result.to_dict()}
                    for page_number, result, flagged in results
                ],
                indent=2,
            )
        )
        _vprint(verbose, f"\n[Station 6] Saved '{results_path}'")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.agents.classifier_agent <path-to-pdf>")
        raise SystemExit(1)

    all_results = classify_pdf(sys.argv[1], verbose=True)

    print("\n=== SUMMARY ===")
    for page_number, result, flagged in all_results:
        marker = " [NEEDS REVIEW]" if flagged else ""
        print(f"Page {page_number}{marker}: {json.dumps(result.to_dict())}")

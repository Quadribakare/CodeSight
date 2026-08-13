"""
Stage B of the pipeline: fused vector + vision compliance-relevant fact extraction.

Given a page's classification (class + subtype, from classifier_agent), its
extracted vector text (vector_reader), and its rendered image, this stage
extracts every piece of information that states or implies a specific,
checkable requirement or measurement -- whether it's a tagged/dimensioned
value on the drawing itself (door widths, room areas, ceiling heights) or a
requirement written as a sentence in specification/leader notes (U-values,
spacing requirements, minimum heights). Both kinds are "facts" here; this
stage does not know or care what the regulations actually require -- it only
reads what the drawing says, with a pointer back to exactly where each fact
came from so the judge stage (judge.py) can cite evidence. Matching facts
against the regulation knowledge base is matcher.py's job, not this one.

Reuses Stage A's already-saved artifacts (classification.json,
page_NNN_vector.json, page_NNN.png under outputs/<pdf-stem>/) rather than
re-classifying or re-extracting, since classify_pdf must already have been
run for a PDF before Stage B can run on it.
"""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.openrouter_client import OpenRouterClient

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-20260507"

# Each notes block is an independent, text-only API call -- network-bound,
# so a few can run at once instead of paying every call's latency serially.
DEFAULT_MAX_WORKERS = 8

VALID_FACT_TYPES = {"tagged_dimension", "note_derived"}


@dataclass
class ExtractedFact:
    """One compliance-relevant fact read off a classified page, with evidence."""

    parameter_name: str
    value: str
    unit: str | None
    fact_type: str  # "tagged_dimension" | "note_derived"
    source_quote: str
    source_bbox: tuple[float, float, float, float] | None
    page_number: int
    page_class: str
    page_subtype: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vprint(verbose: bool, *lines: str) -> None:
    if not verbose:
        return
    for line in lines:
        print(line)


def _load_page_output(pdf_output_dir: Path, page_number: int) -> dict[str, Any]:
    """Load a page's saved vector extraction (text spans, geometry, size)."""
    vector_path = pdf_output_dir / f"page_{page_number:03d}_vector.json"
    if not vector_path.exists():
        raise FileNotFoundError(
            f"{vector_path} not found -- run classifier_agent on this PDF first "
            f"(Stage A must run before Stage B)."
        )
    return json.loads(vector_path.read_text(encoding="utf-8"))


def _load_page_image_base64(pdf_output_dir: Path, page_number: int) -> str:
    image_path = pdf_output_dir / f"page_{page_number:03d}.png"
    if not image_path.exists():
        raise FileNotFoundError(
            f"{image_path} not found -- run classifier_agent on this PDF first "
            f"(Stage A must run before Stage B)."
        )
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _load_classifications(pdf_output_dir: Path) -> list[dict[str, Any]]:
    classification_path = pdf_output_dir / "classification.json"
    if not classification_path.exists():
        raise FileNotFoundError(
            f"{classification_path} not found -- run classifier_agent on this PDF first "
            f"(Stage A must run before Stage B)."
        )
    return json.loads(classification_path.read_text(encoding="utf-8"))


def _build_prompt(page_data: dict[str, Any], page_class: str, subtype: str | None) -> str:
    text_lines = [
        f'  [{i}] "{s["text"]}" at {tuple(round(c, 1) for c in s["bbox"])}'
        for i, s in enumerate(page_data["text_spans"])
    ]
    text_block = "\n".join(text_lines) if text_lines else "  (no extracted text)"

    return f"""You are extracting compliance-relevant facts from one page of an architectural drawing.

This page has already been classified as: class="{page_class}", subtype="{subtype}".

You are given:
1. An image of the page.
2. The page's extracted text, numbered [0], [1], [2]..., with each string's bounding box
   (x0, y0, x1, y1) in PDF points.

Extract every piece of information on this page that states or implies a specific, checkable
requirement relevant to building-regulation compliance. A "value" does NOT have to be numeric -- it
can be a numeric measurement, a material grade, a named standard, or a specific stated obligation.
This includes ALL of:
- Tagged or dimensioned values on the drawing itself (e.g. a door width written next to a door tag,
  a room area, a ceiling height, a stair riser/going dimension, an escape route length).
- Numeric requirements written as sentences in specification/leader notes (e.g. a stated U-value,
  a wall tie spacing requirement, a minimum DPC height, a minimum bearing depth, a fire rating).
- Non-numeric but still specific and checkable requirements in notes (e.g. "structural timbers to be
  graded C16 or C24 per BS 5268: Part 2 1991", "workmanship to comply with BS 8000 Parts 1 to 16",
  "contractor must notify building control before excavation of foundation trenches").

Do NOT extract vague, non-checkable boilerplate that no regulation clause could be matched against --
e.g. generic liability/insurance disclaimers, or instructions with no specific checkable content
("verify all dimensions on site"). If in doubt, ask: could this plausibly be compared against a
specific regulation clause and judged pass/fail? If yes, extract it; if it's just administrative
process or a disclaimer, skip it.

Do not estimate, infer, or guess a value from visual proportions, scale, or drawing geometry alone --
only extract what is explicitly written down somewhere in the text.

Page size: {page_data["width"]:.1f} x {page_data["height"]:.1f} points.

Extracted text spans ({len(page_data["text_spans"])} total):
{text_block}

For each fact, report:
- "parameter_name": a short snake_case description of what this is, e.g. "external_wall_u_value_achieved",
  "wall_tie_horizontal_spacing", "door_clear_width", "structural_timber_grade", "workmanship_standard"
- "value": the stated value, exactly as written, as a string. Numeric (e.g. "0.17", "900", "150") OR
  non-numeric (e.g. "C16 or C24", "BS 5268: Part 2 1991", "BS 8000 Parts 1 to 16")
- "unit": the unit if any, e.g. "W/m2K", "mm", "m2" (or null if there is no unit, which is normal for
  non-numeric facts like a grade or standard reference)
- "fact_type": "tagged_dimension" if it's a labelled measurement on the drawing itself, or
  "note_derived" if it's stated in a specification/leader note sentence
- "source_quote": the exact verbatim text this fact was read from
- "source_span_indices": the span number(s) from the numbered list above that this quote came from
  (a list, since a note-derived fact's sentence may span more than one extracted span)

Respond with ONLY a JSON array, no markdown fences, no commentary, matching exactly this shape:
[
  {{
    "parameter_name": "...",
    "value": "...",
    "unit": "..." ,
    "fact_type": "tagged_dimension",
    "source_quote": "...",
    "source_span_indices": [0]
  }},
  {{
    "parameter_name": "structural_timber_grade",
    "value": "C16 or C24",
    "unit": null,
    "fact_type": "note_derived",
    "source_quote": "All structural timbers to be clearly marked KD (Kiln Dried) C16 Grade or C24 Grade as per BS 5268: Part 2 1991",
    "source_span_indices": [173, 174]
  }}
]
Return an empty array if this page has no explicitly stated, checkable facts."""


def _parse_model_output(raw_text: str) -> list[dict[str, Any]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Feature extraction did not return valid JSON: {raw_text!r}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Feature extraction response was not a JSON array: {raw_text!r}")
    return data


def _union_bbox(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def _to_facts(
    raw_items: list[dict[str, Any]],
    page_data: dict[str, Any],
    page_number: int,
    page_class: str,
    subtype: str | None,
) -> list[ExtractedFact]:
    spans = page_data["text_spans"]
    facts: list[ExtractedFact] = []
    for item in raw_items:
        fact_type = item.get("fact_type", "note_derived")
        if fact_type not in VALID_FACT_TYPES:
            fact_type = "note_derived"

        span_indices = [i for i in item.get("source_span_indices", []) if 0 <= i < len(spans)]
        bbox = _union_bbox([tuple(spans[i]["bbox"]) for i in span_indices]) if span_indices else None

        facts.append(
            ExtractedFact(
                parameter_name=item["parameter_name"],
                value=str(item["value"]),
                unit=item.get("unit"),
                fact_type=fact_type,
                source_quote=item.get("source_quote", ""),
                source_bbox=bbox,
                page_number=page_number,
                page_class=page_class,
                page_subtype=subtype,
            )
        )
    return facts


def extract_page_features(
    page_data: dict[str, Any],
    page_image_base64: str,
    page_number: int,
    page_class: str,
    subtype: str | None,
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> list[ExtractedFact]:
    """Extract compliance-relevant facts from one already-classified page.

    `page_data` is a page's vector extraction as a plain dict (the same shape
    `vector_reader.PageVectorData.to_dict()` produces, or the parsed contents
    of a saved page_NNN_vector.json).
    """
    prompt = _build_prompt(page_data, page_class, subtype)
    _vprint(
        verbose,
        f"  [Station 2] Sending to model '{model}' -- prompt is {len(prompt)} chars, "
        f"embedding all {len(page_data['text_spans'])} text spans, plus 1 image.",
    )

    raw_response = client.complete(model=model, prompt=prompt, image_base64=page_image_base64)
    _vprint(verbose, "  [Station 2] Raw model response:", "  " + raw_response.replace("\n", "\n  "))

    raw_items = _parse_model_output(raw_response)
    _vprint(verbose, f"  [Station 3] Parsed {len(raw_items)} fact(s) from response.")

    facts = _to_facts(raw_items, page_data, page_number, page_class, subtype)
    for fact in facts:
        _vprint(
            verbose,
            f"  [Station 4] {fact.parameter_name}={fact.value}{fact.unit or ''} "
            f"({fact.fact_type}) <- {fact.source_quote!r}",
        )
    return facts


@dataclass
class NoteBlock:
    """One heading + its body content, detected by formatting, not by heading wording."""

    heading: str
    heading_span_index: int
    content_span_indices: list[int]
    combined_text: str


def _looks_like_heading(text: str) -> bool:
    """Content-agnostic heading detector: short, mostly-uppercase, ends with a colon.

    Deliberately does not check the actual words, since a different drawing
    set (different firm, different discipline) may use entirely different
    heading names -- this only relies on the formatting convention of a
    section header, which is common across specification-notes blocks.
    """
    stripped = text.strip()
    if not stripped.endswith(":") or len(stripped) > 60:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.8


def find_note_blocks(page_data: dict[str, Any]) -> list[NoteBlock]:
    """Group a page's text spans into heading + body blocks, using formatting only.

    A block runs from just after one detected heading up to (not including)
    the next detected heading. Headings with no body text (nothing before the
    next heading) are skipped.
    """
    spans = page_data["text_spans"]
    heading_indices = [i for i, s in enumerate(spans) if _looks_like_heading(s["text"])]

    # PDF text objects are not guaranteed to be stored in reading order -- we
    # already saw a heading's own body appear before the heading in the raw
    # span list. So the *next detected heading* is not a safe stopping point:
    # unrelated scattered content elsewhere on the page can sit between them
    # in raw order. Instead, only keep grouping spans that stay in the same
    # column as the heading (x0 within tolerance) and keep flowing downward
    # (y0 non-decreasing) -- stop the block the moment that breaks, even if
    # the next detected heading is still further away in index order.
    x_tolerance = 30.0
    y_backtrack_tolerance = 5.0

    blocks: list[NoteBlock] = []
    for pos, heading_idx in enumerate(heading_indices):
        heading_span = spans[heading_idx]
        heading_x0 = heading_span["bbox"][0]
        limit = heading_indices[pos + 1] if pos + 1 < len(heading_indices) else len(spans)

        content_indices: list[int] = []
        prev_y0 = heading_span["bbox"][1]
        for i in range(heading_idx + 1, limit):
            bbox = spans[i]["bbox"]
            same_line = abs(bbox[1] - prev_y0) < 3.0
            # Word-wrap fragments can land at an odd x0 on the same line (e.g.
            # "walls" split into "...for exter" + "lls" at a different x0) --
            # only enforce the column/direction check when moving to a new line.
            if not same_line:
                if abs(bbox[0] - heading_x0) > x_tolerance:
                    break
                if bbox[1] < prev_y0 - y_backtrack_tolerance:
                    break
            content_indices.append(i)
            prev_y0 = bbox[1]

        combined_text = " ".join(spans[i]["text"] for i in content_indices).strip()
        if combined_text:
            blocks.append(
                NoteBlock(
                    heading=heading_span["text"],
                    heading_span_index=heading_idx,
                    content_span_indices=content_indices,
                    combined_text=combined_text,
                )
            )
    return blocks


def _build_notes_block_prompt(block: NoteBlock, page_data: dict[str, Any]) -> str:
    spans = page_data["text_spans"]
    lines = [f'  [{i}] "{spans[i]["text"]}"' for i in block.content_span_indices]
    text_block = "\n".join(lines)

    return f"""You are reading ONE section of specification notes from an architectural drawing. This
section may contain several distinct checkable requirements -- your job is to list EVERY one of them,
not just the first one or two. Dense sentences with multiple numbers (e.g. a wall build-up describing
thickness, material grade, and a U-value all in one paragraph) must be split into one fact per value.

Section heading: "{block.heading}"

Numbered content for this section ({len(block.content_span_indices)} span(s)):
{text_block}

For each distinct checkable requirement, report:
- "parameter_name": short snake_case description. Make it specific enough to distinguish it from
  other facts in THIS SAME section -- e.g. if this section names more than one standard or more than
  one measurement, do not call them all the same generic name like "standard_compliance"; name each by
  what it actually specifies (e.g. "tie_material_standard" vs "tie_product_standard").
- "value": the stated value, exactly as written, as a string. Can be numeric or non-numeric (a grade,
  a named standard, a material name with a specific spec).
- "unit": the unit if any (e.g. "mm", "W/m2K"), or null
- "source_quote": the exact verbatim text this came from
- "source_span_indices": which of the numbered lines above this came from (a list)

Skip anything that is purely procedural/administrative with no specific checkable value (e.g. "verify
on site", generic liability wording). If truly nothing checkable is in this section, return [].

Respond with ONLY a JSON array, no markdown fences, no commentary:
[
  {{"parameter_name": "...", "value": "...", "unit": "...", "source_quote": "...", "source_span_indices": [0]}}
]"""


def _slugify(text: str) -> str:
    """Turn a heading (whatever it says, in any drawing) into a name-safe prefix."""
    stripped = text.strip().rstrip(":")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stripped).strip("_").lower()
    return slug or "section"


def _namespace_and_disambiguate(facts: list[ExtractedFact], heading: str) -> None:
    """Guarantee unique parameter_names within one block, using only mechanical steps.

    1. Prefix every name with the block's own (generically-detected) heading,
       so the same generic name in two different sections never collides.
    2. Count occurrences of the (now-prefixed) name within this same block's
       results and append _2, _3... to any repeats. Both steps are driven
       purely by structure already found in the page, not by any hardcoded
       heading or parameter name.
    """
    prefix = _slugify(heading)
    seen_counts: dict[str, int] = {}
    for fact in facts:
        fact.parameter_name = f"{prefix}__{fact.parameter_name}"
        seen_counts[fact.parameter_name] = seen_counts.get(fact.parameter_name, 0) + 1
        if seen_counts[fact.parameter_name] > 1:
            fact.parameter_name = f"{fact.parameter_name}_{seen_counts[fact.parameter_name]}"


def extract_notes_block_facts(
    block: NoteBlock,
    page_data: dict[str, Any],
    page_number: int,
    page_class: str,
    subtype: str | None,
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> list[ExtractedFact]:
    """Focused, text-only extraction pass over a single heading's body content.

    No image is sent -- this section's content is already plain text, so
    there's nothing visual to interpret, and skipping the image keeps each of
    these many small calls cheaper.
    """
    prompt = _build_notes_block_prompt(block, page_data)
    raw_response = client.complete(model=model, prompt=prompt)
    _vprint(
        verbose,
        f"    [Notes block '{block.heading}'] response: " + raw_response.replace("\n", " ")[:200],
    )

    raw_items = _parse_model_output(raw_response)
    facts = _to_facts(raw_items, page_data, page_number, page_class, subtype)
    for fact in facts:
        fact.fact_type = "note_derived"
    _namespace_and_disambiguate(facts, block.heading)
    return facts


def _fact_key(fact: ExtractedFact) -> tuple[str, str]:
    return (" ".join(fact.parameter_name.lower().split()), " ".join(fact.value.lower().split()))


def _merge_facts(general: list[ExtractedFact], notes: list[ExtractedFact]) -> list[ExtractedFact]:
    """Combine the general pass and the per-block notes pass, dropping exact duplicates.

    A duplicate is the same (parameter_name, value) pair found by both
    passes -- the general pass's copy is kept in that case, since it's first.
    Near-duplicates with slightly different parameter_name wording are not
    caught by this and may appear twice; that's a known limitation.
    """
    merged = list(general)
    seen = {_fact_key(f) for f in general}
    for fact in notes:
        key = _fact_key(fact)
        if key not in seen:
            merged.append(fact)
            seen.add(key)
    return merged


def extract_page_features_thorough(
    page_data: dict[str, Any],
    page_image_base64: str,
    page_number: int,
    page_class: str,
    subtype: str | None,
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[ExtractedFact]:
    """Run both extraction passes for one page and converge them into one fact list.

    Pass 1 (extract_page_features): one call over the whole page (image +
    all text), same as before. Pass 2: one additional focused, text-only call
    per detected heading/body block, each asked to exhaustively enumerate
    every value in just that section -- addressing dense paragraphs the
    single whole-page pass tends to under-report. Results are merged,
    dropping exact duplicates.

    Notes-block calls run concurrently (up to `max_workers` at a time) since
    each is independent -- set max_workers=1 to force one-at-a-time.
    """
    general_facts = extract_page_features(
        page_data, page_image_base64, page_number, page_class, subtype, client, model=model, verbose=verbose
    )

    blocks = find_note_blocks(page_data)
    _vprint(verbose, f"  [Notes] Detected {len(blocks)} heading/body block(s) for focused extraction")

    def _extract_one(block: NoteBlock) -> list[ExtractedFact]:
        return extract_notes_block_facts(
            block, page_data, page_number, page_class, subtype, client, model=model, verbose=verbose
        )

    notes_facts: list[ExtractedFact] = []
    if blocks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for block_facts in executor.map(_extract_one, blocks):
                notes_facts.extend(block_facts)

    merged = _merge_facts(general_facts, notes_facts)
    _vprint(
        verbose,
        f"  [Converge] {len(general_facts)} general + {len(notes_facts)} notes-block fact(s) "
        f"-> {len(merged)} after dedupe",
    )
    return merged


def extract_features_for_pdf(
    pdf_path: str,
    client: OpenRouterClient | None = None,
    model: str = DEFAULT_MODEL,
    output_dir: str = "outputs",
    verbose: bool = False,
) -> list[ExtractedFact]:
    """Run Stage B over every classified page of a PDF, reusing Stage A's saved output.

    Requires classifier_agent.classify_pdf to have already been run on this PDF
    (so classification.json, page_NNN_vector.json, and page_NNN.png already
    exist under output_dir/<pdf-stem>/) -- this does not re-classify or
    re-extract, to avoid paying for Stage A's API calls twice.
    """
    client = client or OpenRouterClient()
    pdf_out_dir = Path(output_dir) / Path(pdf_path).stem

    classifications = _load_classifications(pdf_out_dir)
    _vprint(verbose, f"[Station 0] Loaded {len(classifications)} classification(s) from '{pdf_out_dir}'")

    all_facts: list[ExtractedFact] = []
    page_data_cache: dict[int, dict[str, Any]] = {}
    image_cache: dict[int, str] = {}

    for entry in classifications:
        page_number = entry["page_number"]
        page_class = entry["page_class"]
        subtype = entry.get("subtype")

        if page_class == "Not a drawing":
            _vprint(verbose, f"\n=== Page {page_number}: skipped (Not a drawing) ===")
            continue

        _vprint(verbose, f"\n=== Page {page_number}: {page_class} / {subtype} ===")

        if page_number not in page_data_cache:
            page_data_cache[page_number] = _load_page_output(pdf_out_dir, page_number)
            image_cache[page_number] = _load_page_image_base64(pdf_out_dir, page_number)

        facts = extract_page_features_thorough(
            page_data_cache[page_number],
            image_cache[page_number],
            page_number,
            page_class,
            subtype,
            client,
            model=model,
            verbose=verbose,
        )
        all_facts.extend(facts)

    facts_path = pdf_out_dir / "features.json"
    facts_path.write_text(json.dumps([f.to_dict() for f in all_facts], indent=2))
    _vprint(verbose, f"\n[Station 5] Saved '{facts_path}' ({len(all_facts)} fact(s) total)")

    return all_facts


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.agents.feature_agent <path-to-pdf>")
        raise SystemExit(1)

    facts = extract_features_for_pdf(sys.argv[1], verbose=True)

    print(f"\n=== SUMMARY: {len(facts)} fact(s) extracted ===")
    for fact in facts:
        print(json.dumps(fact.to_dict()))

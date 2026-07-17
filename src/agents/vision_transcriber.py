"""
Vision-based exhaustive text transcription, as a completeness check alongside vector_reader.py.

Some CAD exports produce inconsistent or incomplete PDF text layers (text
buried in nested Form XObjects, hidden Optional Content Group layers, or
genuinely rasterized stamps). Rather than debugging each export's internals,
this module renders every page as a grid of overlapping high-DPI tiles (see
image_renderer.render_page_tiles) and asks a vision model to transcribe every
piece of text it can see, tile by tile.

This gives two complete, independent, unreconciled views of the same page's
text: everything vision could read off the image, and everything
vector_reader could read from the PDF's text objects. Neither is edited to
match the other -- both are kept in full so nothing is silently dropped or
merged away, and a human can compare them directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.extraction.image_renderer import PageTile, plan_page_tiles, render_page_tiles
from src.openrouter_client import OpenRouterClient

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-20260507"

VALID_CATEGORIES = {
    "room_label",
    "tag",
    "dimension",
    "leader_note",
    "specification_note",
    "title_block",
    "other",
}


@dataclass
class TranscribedItem:
    """One piece of text found by the vision model on one tile."""

    text: str
    approximate_location: str
    category: str
    tile_row: int
    tile_col: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vprint(verbose: bool, *lines: str) -> None:
    if not verbose:
        return
    for line in lines:
        print(line)


def _build_tile_prompt() -> str:
    categories = ", ".join(sorted(VALID_CATEGORIES))
    return f"""You are looking at one tile (a cropped section) of a large architectural drawing sheet.
Adjacent tiles overlap this one, so some text near the edges may be cut off or repeated -- that's fine,
just transcribe whatever is legible in this tile.

Transcribe every piece of text visible in this image, exhaustively. Include: room labels, door/window/
glazing tags, dimension callouts, leader notes, specification notes, title block fields, revision table
entries, stamps, and section markers. Do not skip anything because it looks minor.

For each piece of text, report:
- "text": the exact text, verbatim
- "approximate_location": a short human-readable description of where it sits in this tile (e.g.
  "top-left", "center, near a door symbol", "bottom margin")
- "category": one of {categories}

Respond with ONLY a JSON array, no markdown fences, no commentary:
[
  {{"text": "...", "approximate_location": "...", "category": "..."}}
]
Return an empty array if this tile has no legible text."""


def _parse_tile_response(raw_text: str, tile: PageTile) -> list[TranscribedItem]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tile transcription did not return valid JSON: {raw_text!r}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Tile transcription response was not a JSON array: {raw_text!r}")

    items = []
    for entry in data:
        category = entry.get("category", "other")
        if category not in VALID_CATEGORIES:
            category = "other"
        items.append(
            TranscribedItem(
                text=entry["text"],
                approximate_location=entry.get("approximate_location", ""),
                category=category,
                tile_row=tile.row,
                tile_col=tile.col,
            )
        )
    return items


def transcribe_tile(
    tile: PageTile,
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
) -> list[TranscribedItem]:
    """Send one rendered tile to the vision model and parse its transcription."""
    raw_response = client.complete(
        model=model,
        prompt=_build_tile_prompt(),
        image_base64=tile.png_base64,
    )
    return _parse_tile_response(raw_response, tile)


def _dedupe_items(items: list[TranscribedItem]) -> list[TranscribedItem]:
    """Drop exact-duplicate items caused by overlapping tiles picking up the same text twice.

    Only collapses identical (case/whitespace-insensitive) text -- never
    merges or rewrites content, so nothing distinct gets lost.
    """
    seen: dict[str, TranscribedItem] = {}
    for item in items:
        key = " ".join(item.text.lower().split())
        if key not in seen:
            seen[key] = item
    return list(seen.values())


def transcribe_page(
    pdf_path: str,
    page_number: int,
    client: OpenRouterClient | None = None,
    model: str = DEFAULT_MODEL,
    dpi: int = 300,
    tile_px: int = 2400,
    overlap_px: int = 250,
    verbose: bool = False,
    output_dir: str | None = "outputs",
) -> list[TranscribedItem]:
    """Tile-render one page and transcribe every tile, deduping only exact overlap-caused repeats.

    If `output_dir` is set (default "outputs"), the full transcription is
    saved to `output_dir/<pdf-stem>/page_<NNN>_vision_text.json` -- a
    complete, standalone record of everything vision read off this page,
    independent of vector_reader's text-layer extraction.
    """
    client = client or OpenRouterClient()

    plan = plan_page_tiles(pdf_path, page_number, dpi, tile_px, overlap_px)
    _vprint(verbose, f"[Tiling] {len(plan)} tile(s) planned for page {page_number} at {dpi} DPI")

    tiles = render_page_tiles(pdf_path, page_number, dpi, tile_px, overlap_px)

    all_items: list[TranscribedItem] = []
    for tile in tiles:
        items = transcribe_tile(tile, client, model)
        _vprint(verbose, f"  [Tile r{tile.row}c{tile.col}] {len(items)} item(s) transcribed")
        all_items.extend(items)

    deduped = _dedupe_items(all_items)
    _vprint(
        verbose,
        f"[Dedupe] {len(all_items)} raw item(s) -> {len(deduped)} after removing exact overlap duplicates",
    )

    if output_dir is not None:
        pdf_out_dir = Path(output_dir) / Path(pdf_path).stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = pdf_out_dir / f"page_{page_number:03d}_vision_text.json"
        out_path.write_text(
            json.dumps(
                {
                    "page_number": page_number,
                    "tile_count": len(tiles),
                    "item_count": len(deduped),
                    "items": [item.to_dict() for item in deduped],
                },
                indent=2,
            )
        )
        _vprint(verbose, f"[Save] Wrote '{out_path}'")

    return deduped


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m src.agents.vision_transcriber <path-to-pdf> <page-number>")
        raise SystemExit(1)

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])

    plan = plan_page_tiles(pdf_path, page_number)
    print(
        f"Page {page_number}: {len(plan)} tile(s) planned at 300 DPI. Proceeding will make "
        f"{len(plan)} vision API call(s) for transcription."
    )

    items = transcribe_page(pdf_path, page_number, verbose=True)

    print(f"\n=== SUMMARY: {len(items)} item(s) transcribed from vision ===")
    for item in items:
        print(json.dumps(item.to_dict()))

"""
Extracts vector text and geometry from CAD-exported architectural drawing PDFs.

Since the source PDFs are vector (no raster scans), every character and every
drawn line/rect/curve is a first-class object with real coordinates. This
module reads those objects directly via PyMuPDF rather than relying on OCR,
giving exact positions for downstream fact extraction and clause matching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import fitz  # PyMuPDF


@dataclass
class TextSpan:
    """A single run of text with consistent font/size, positioned on the page."""

    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font: str
    size: float


@dataclass
class VectorPath:
    """A single drawn path (line, rectangle, curve, or other) with its bounding box."""

    kind: str  # "line" | "rect" | "curve" | "other"
    points: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    color: tuple[float, ...] | None
    width: float | None


@dataclass
class PageVectorData:
    """All extracted vector text and geometry for a single PDF page."""

    page_number: int  # 0-indexed
    width: float
    height: float
    text_spans: list[TextSpan] = field(default_factory=list)
    vector_paths: list[VectorPath] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_text_spans(page: fitz.Page) -> list[TextSpan]:
    spans: list[TextSpan] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                spans.append(
                    TextSpan(
                        text=text,
                        bbox=tuple(span["bbox"]),
                        font=span.get("font", ""),
                        size=span.get("size", 0.0),
                    )
                )
    return spans


def _path_kind(item: tuple) -> str:
    op = item[0]
    if op == "l":
        return "line"
    if op == "re":
        return "rect"
    if op in ("c", "qu"):
        return "curve"
    return "other"


def _path_points(item: tuple) -> list[tuple[float, float]]:
    op, *operands = item
    points: list[tuple[float, float]] = []
    if op == "re":
        rect = operands[0]
        points = [
            (rect.x0, rect.y0),
            (rect.x1, rect.y0),
            (rect.x1, rect.y1),
            (rect.x0, rect.y1),
        ]
    else:
        for operand in operands:
            if hasattr(operand, "x") and hasattr(operand, "y"):
                points.append((operand.x, operand.y))
    return points


def _extract_vector_paths(page: fitz.Page) -> list[VectorPath]:
    paths: list[VectorPath] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            points = _path_points(item)
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            paths.append(
                VectorPath(
                    kind=_path_kind(item),
                    points=points,
                    bbox=bbox,
                    color=drawing.get("color"),
                    width=drawing.get("width"),
                )
            )
    return paths


def extract_pdf_vector_data(pdf_path: str) -> list[PageVectorData]:
    """Read every page of `pdf_path` and return its text spans and vector paths."""
    pages: list[PageVectorData] = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc):
            pages.append(
                PageVectorData(
                    page_number=page_number,
                    width=page.rect.width,
                    height=page.rect.height,
                    text_spans=_extract_text_spans(page),
                    vector_paths=_extract_vector_paths(page),
                )
            )
    return pages


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python vector_reader.py <path-to-pdf>")
        raise SystemExit(1)

    result = extract_pdf_vector_data(sys.argv[1])
    print(json.dumps([p.to_dict() for p in result], indent=2))

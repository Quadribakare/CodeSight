"""
Renders PDF pages to PNG images for the vision half of the multimodal pipeline.

The classifier and feature-extraction agents reason over both the raw vector
data (from vector_reader.py) and a rendered image of the same page, so the
LLM can use visual layout cues (title blocks, symbols, hatching, north arrows)
that are hard to recover from text/geometry alone.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

DEFAULT_DPI = 200

# Defaults for tiling large sheets (A1/A0) so small text stays legible for a
# vision model instead of being downsampled away by rendering the whole
# sheet into one image.
DEFAULT_TILE_DPI = 300
DEFAULT_TILE_PX = 2400
DEFAULT_TILE_OVERLAP_PX = 250


@dataclass
class PageTile:
    """One rendered tile of a (possibly large) page, in page-pixel-grid position."""

    row: int
    col: int
    bbox_pts: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    png_base64: str


def render_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = DEFAULT_DPI,
) -> list[str]:
    """Render every page of `pdf_path` to a PNG at `dpi` and save under `output_dir`.

    Returns the list of written PNG file paths, in page order.
    """
    pdf_stem = Path(pdf_path).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72  # PyMuPDF's default page unit is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    written_paths: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix)
            out_path = out_dir / f"{pdf_stem}_page{page_number:03d}.png"
            pix.save(out_path)
            written_paths.append(str(out_path))
    return written_paths


def render_page_to_png_bytes(pdf_path: str, page_number: int, dpi: int = DEFAULT_DPI) -> bytes:
    """Render a single page (0-indexed) of `pdf_path` to PNG bytes, without writing to disk."""
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")


def render_page_to_base64(pdf_path: str, page_number: int, dpi: int = DEFAULT_DPI) -> str:
    """Render a single page (0-indexed) of `pdf_path` and return it as a base64 PNG string."""
    png_bytes = render_page_to_png_bytes(pdf_path, page_number, dpi)
    return base64.b64encode(png_bytes).decode("ascii")


def _tile_starts(total_px: float, tile_px: int, overlap_px: int) -> list[int]:
    """Pixel offsets where each tile along one axis should start.

    Tiles are `tile_px` wide/tall and overlap the previous tile by
    `overlap_px`, so text sitting right on a tile boundary still appears
    whole in at least one tile. Always covers the full page, including a
    final tile flush with the far edge even if that means extra overlap.
    """
    if total_px <= tile_px:
        return [0]

    step = tile_px - overlap_px
    starts = list(range(0, math.ceil(total_px - tile_px) + 1, step))
    last_start = total_px - tile_px
    if not starts or starts[-1] < last_start:
        starts.append(last_start)
    return starts


def plan_page_tiles(
    pdf_path: str,
    page_number: int,
    dpi: int = DEFAULT_TILE_DPI,
    tile_px: int = DEFAULT_TILE_PX,
    overlap_px: int = DEFAULT_TILE_OVERLAP_PX,
) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """Compute the tile grid for a page without rendering anything (cheap, local, no API cost).

    Returns a list of (row, col, bbox_pts) -- useful for reporting how many
    tiles a page will need before actually paying to render/transcribe them.
    """
    zoom = dpi / 72
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        page_w_px = page.rect.width * zoom
        page_h_px = page.rect.height * zoom

    col_starts = _tile_starts(page_w_px, tile_px, overlap_px)
    row_starts = _tile_starts(page_h_px, tile_px, overlap_px)

    tiles = []
    for row, y0_px in enumerate(row_starts):
        for col, x0_px in enumerate(col_starts):
            x1_px = min(x0_px + tile_px, page_w_px)
            y1_px = min(y0_px + tile_px, page_h_px)
            bbox_pts = (x0_px / zoom, y0_px / zoom, x1_px / zoom, y1_px / zoom)
            tiles.append((row, col, bbox_pts))
    return tiles


def render_page_tiles(
    pdf_path: str,
    page_number: int,
    dpi: int = DEFAULT_TILE_DPI,
    tile_px: int = DEFAULT_TILE_PX,
    overlap_px: int = DEFAULT_TILE_OVERLAP_PX,
) -> list[PageTile]:
    """Render one page as a grid of overlapping tiles at `dpi`.

    Large sheets (A1/A0) rendered whole into a single image get downsampled
    by the time they reach a vision model, making small text illegible. This
    renders each tile separately at full `dpi` so nothing gets shrunk away.
    """
    zoom = dpi / 72
    plan = plan_page_tiles(pdf_path, page_number, dpi, tile_px, overlap_px)

    tiles: list[PageTile] = []
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        matrix = fitz.Matrix(zoom, zoom)
        for row, col, bbox_pts in plan:
            clip = fitz.Rect(*bbox_pts)
            pix = page.get_pixmap(matrix=matrix, clip=clip)
            png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            tiles.append(PageTile(row=row, col=col, bbox_pts=bbox_pts, png_base64=png_b64))
    return tiles


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python image_renderer.py <path-to-pdf> <output-dir>")
        raise SystemExit(1)

    paths = render_pdf_to_images(sys.argv[1], sys.argv[2])
    print(f"Wrote {len(paths)} page image(s) to {sys.argv[2]}")

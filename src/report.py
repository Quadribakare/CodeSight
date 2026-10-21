"""
Renders Stage D's missing_items.json as a downloadable PDF.

Deliberately styled after the layout of a real Building Control "Notice of
Rejection" (letterhead block, deficiency table, appeal-style footer) because
that's the format a reviewer already knows how to read -- but with no real
council name, logo, or branding anywhere, and an explicit disclaimer at both
the top and bottom. This is a pre-check report from this pipeline, not a
building-control decision, and must never be mistaken for one.

Uses fpdf2 rather than reportlab: it's pure-Python with no compiled/binary
dependencies, which matters on Vercel's serverless Python runtime where the
deployed function has a hard size limit (reportlab + the Pillow it pulls in
pushed the bundle over that limit and broke the whole app -- see the commit
that replaced it).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fpdf import FPDF

DISCLAIMER = (
    "This is an automatically generated pre-check from a research pipeline. "
    "It is NOT an official Building Control decision, is not issued by any "
    "local authority, and does not replace statutory review. Findings may be "
    "incomplete or incorrect -- verify everything against the actual "
    "regulations before relying on it."
)

_CONFIDENCE_LABELS = {
    "likely_missing": "Likely missing",
    "possibly_missing": "Possibly missing",
}

# fpdf2's built-in core fonts (no bundled Unicode font, to keep this
# dependency-free and small) only support latin-1. Clause text pulled from
# real scanned regulation PDFs routinely has smart quotes/en-dashes/etc
# outside that range, so anything going on the page gets normalized first
# instead of letting FPDFUnicodeEncodingException take the whole request down.
_CHAR_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "--", "…": "...", " ": " ",
}


def _sanitize(text: str) -> str:
    for src, dest in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, dest)
    return text.encode("latin-1", errors="replace").decode("latin-1")


_MAX_CLAUSE_CHARS = 450


def _truncate_clause(text: str) -> str:
    if len(text) <= _MAX_CLAUSE_CHARS:
        return text
    return text[:_MAX_CLAUSE_CHARS].rsplit(" ", 1)[0] + "..."


def _disclaimer_box(pdf: FPDF, text: str, *, text_color: tuple[int, int, int]) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(253, 236, 236)
    pdf.set_draw_color(229, 160, 160)
    pdf.set_text_color(*text_color)
    pdf.set_font("Helvetica", size=8.5)
    pdf.multi_cell(0, 4.5, text, border=1, fill=True)
    pdf.set_text_color(0, 0, 0)


def build_missing_items_pdf(filename: str, missing_items: list[dict[str, Any]]) -> bytes:
    """Build the Stage D report PDF and return its raw bytes."""
    pdf = FPDF(format="A4")
    pdf.set_margin(20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "Compliance Pre-Check -- Possible Missing Items", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _disclaimer_box(pdf, DISCLAIMER, text_color=(122, 31, 31))
    pdf.ln(4)

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Drawing: {_sanitize(filename)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT")

    reportable = [m for m in missing_items if m.get("confidence") in _CONFIDENCE_LABELS]
    by_group: dict[tuple[str, str], int] = {}
    for m in reportable:
        key = (m.get("page_class", ""), m.get("page_subtype", ""))
        by_group[key] = by_group.get(key, 0) + 1
    if by_group:
        summary = "; ".join(f"{cls}/{sub}: {n} item(s)" for (cls, sub), n in by_group.items())
    else:
        summary = "No likely-missing items found."
    pdf.multi_cell(0, 6, _sanitize(f"Summary: {summary}"))
    pdf.ln(4)

    if reportable:
        with pdf.table(
            col_widths=(28, 32, 24, 66),
            text_align="LEFT",
            line_height=4.5,
        ) as table:
            header = table.row()
            for h in ["Drawing area", "Requirement", "Confidence", "Cited clause excerpt"]:
                header.cell(h)
            for m in reportable:
                area = f"{m.get('page_class', '')}/{m.get('page_subtype', '')}"
                requirement = m.get("requirement", "")
                confidence = _CONFIDENCE_LABELS.get(m.get("confidence", ""), m.get("confidence", ""))
                clause = _truncate_clause(m.get("cited_clause_text") or "--")
                doc_name = m.get("cited_source_document")
                clause_cell = clause + (f" ({doc_name})" if doc_name else "")
                row = table.row()
                row.cell(_sanitize(area))
                row.cell(_sanitize(requirement))
                row.cell(_sanitize(confidence))
                row.cell(_sanitize(clause_cell))
    else:
        pdf.multi_cell(0, 6, "No likely-missing items were identified by this pre-check.")

    pdf.ln(6)
    _disclaimer_box(pdf, DISCLAIMER, text_color=(120, 120, 120))
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", size=7.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "Not affiliated with, and not issued by, any Building Control authority.")

    return bytes(pdf.output())

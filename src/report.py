"""
Renders Stage D's missing_items.json as a downloadable PDF.

Deliberately styled after the layout of a real Building Control "Notice of
Rejection" (letterhead block, deficiency table, appeal-style footer) because
that's the format a reviewer already knows how to read -- but with no real
council name, logo, or branding anywhere, and an explicit disclaimer at both
the top and bottom. This is a pre-check report from this pipeline, not a
building-control decision, and must never be mistaken for one.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], alignment=TA_CENTER, fontSize=16,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontSize=8.5,
            textColor=colors.HexColor("#7a1f1f"),
            backColor=colors.HexColor("#fdecec"),
            borderColor=colors.HexColor("#e5a0a0"),
            borderWidth=0.75,
            borderPadding=6,
            leading=11,
        ),
        "body": base["Normal"],
        "cell": ParagraphStyle("Cell", parent=base["Normal"], fontSize=8.5, leading=11),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"], fontSize=7.5, textColor=colors.grey,
        ),
    }


def build_missing_items_pdf(filename: str, missing_items: list[dict[str, Any]]) -> bytes:
    """Build the Stage D report PDF and return its raw bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    styles = _styles()
    story: list[Any] = []

    story.append(Paragraph("Compliance Pre-Check — Possible Missing Items", styles["title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(DISCLAIMER, styles["disclaimer"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph(f"<b>Drawing:</b> {filename}", styles["body"]))
    story.append(Paragraph(f"<b>Generated:</b> {date.today().isoformat()}", styles["body"]))

    reportable = [m for m in missing_items if m.get("confidence") in _CONFIDENCE_LABELS]
    by_group: dict[tuple[str, str], int] = {}
    for m in reportable:
        key = (m.get("page_class", ""), m.get("page_subtype", ""))
        by_group[key] = by_group.get(key, 0) + 1
    if by_group:
        summary = "; ".join(f"{cls}/{sub}: {n} item(s)" for (cls, sub), n in by_group.items())
    else:
        summary = "No likely-missing items found."
    story.append(Paragraph(f"<b>Summary:</b> {summary}", styles["body"]))
    story.append(Spacer(1, 14))

    if reportable:
        header = ["Drawing area", "Requirement", "Confidence", "Cited clause excerpt"]
        rows = [[Paragraph(f"<b>{h}</b>", styles["cell"]) for h in header]]
        for m in reportable:
            area = f"{m.get('page_class', '')}/{m.get('page_subtype', '')}"
            requirement = m.get("requirement", "")
            confidence = _CONFIDENCE_LABELS.get(m.get("confidence", ""), m.get("confidence", ""))
            clause = m.get("cited_clause_text") or "—"
            doc_name = m.get("cited_source_document")
            clause_cell = f"{clause}" + (f"<br/><i>({doc_name})</i>" if doc_name else "")
            rows.append(
                [
                    Paragraph(area, styles["cell"]),
                    Paragraph(requirement, styles["cell"]),
                    Paragraph(confidence, styles["cell"]),
                    Paragraph(clause_cell, styles["cell"]),
                ]
            )

        table = Table(rows, colWidths=[3.2 * cm, 3.6 * cm, 2.6 * cm, 7.6 * cm], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e1a2b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f5fa")]),
                ]
            )
        )
        story.append(table)
    else:
        story.append(Paragraph("No likely-missing items were identified by this pre-check.", styles["body"]))

    story.append(Spacer(1, 18))
    story.append(Paragraph(DISCLAIMER, styles["footer"]))
    story.append(
        Paragraph(
            "Not affiliated with, and not issued by, any Building Control authority.",
            styles["footer"],
        )
    )

    doc.build(story)
    return buf.getvalue()

"""
Writes up Stage C's verdicts.json and Stage D's missing_items.json as one
report -- the sections for the report page, and a downloadable PDF.

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

import re
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


def humanize(name: str | None) -> str:
    """'electrical_notes__conduit_material' -> 'Electrical notes: conduit material'."""
    if not name:
        return ""
    text = name.replace("__", ": ").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _reader_text(reasoning: str | None) -> str:
    """Strip the judge's prompt-internal excerpt numbers ("Excerpt [1]",
    "of [0]", "[4]"), which point at nothing a reader of the report can see."""
    text = reasoning or ""
    text = re.sub(r"\b[Ee]xcerpt\s*\[\d+\]\s*(?=paragraph\b)", "", text)
    text = re.sub(r"\b[Ee]xcerpt\s*\[\d+\]", "the regulation", text)
    text = re.sub(r"\s+(?:of|in|from)\s+\[\d+\](?:\s*(?:,|and)\s*\[\d+\])*", "", text)
    text = re.sub(r"\s*\[\d+\]", "", text).strip()
    return text[:1].upper() + text[1:]


def _value_text(v: dict[str, Any]) -> str:
    value = str(v.get("value") or "").strip()
    unit = (v.get("unit") or "").strip()
    return f"{value} {unit}".strip() if unit else value


def build_report_sections(
    verdicts: list[dict[str, Any]] | None,
    missing_items: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Turn Stage C verdicts and Stage D missing items into the sections of
    one written report, shared by the report page and the PDF so the two
    always say the same thing.

    Only what needs action gets its own entry: values that break a clause
    (Stage C "fail"), requirements the drawing doesn't show (Stage D), and
    values a clause touches but doesn't settle (Stage C "flag"). Passes and
    facts with no applicable clause are reported as a count, not listed.
    """
    verdicts = verdicts or []
    failures = [
        {
            "title": humanize(v.get("parameter_name")),
            "drawing_shows": _value_text(v),
            "requirement": _reader_text(v.get("reasoning")),
            "source": v.get("cited_source_document"),
            "clause": v.get("cited_clause_text"),
        }
        for v in verdicts if v.get("verdict") == "fail"
    ]
    to_confirm = [
        {
            "title": humanize(v.get("parameter_name")),
            "drawing_shows": _value_text(v),
            "requirement": _reader_text(v.get("reasoning")),
            "source": v.get("cited_source_document"),
            "clause": v.get("cited_clause_text"),
        }
        for v in verdicts if v.get("verdict") == "flag"
    ]
    missing = None
    if missing_items is not None:
        missing = [
            {
                "title": humanize(m.get("requirement")),
                "certainty": _CONFIDENCE_LABELS[m["confidence"]],
                "area": " / ".join(p for p in (m.get("page_class"), m.get("page_subtype")) if p),
                "requirement": _reader_text(m.get("reasoning")),
                "source": m.get("cited_source_document"),
                "clause": m.get("cited_clause_text"),
            }
            for m in missing_items if m.get("confidence") in _CONFIDENCE_LABELS
        ]
    return {
        "has_c": bool(verdicts),
        "has_d": missing is not None,
        "failures": failures,
        "missing": missing,
        "to_confirm": to_confirm,
        "n_checked": len(verdicts),
        "n_passed": sum(1 for v in verdicts if v.get("verdict") == "pass"),
        "n_no_clause": sum(1 for v in verdicts if v.get("verdict") == "no_matching_clause"),
    }


def _section_heading(pdf: FPDF, text: str) -> None:
    pdf.ln(3)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style="B", size=12.5)
    pdf.set_text_color(60, 30, 110)
    pdf.multi_cell(0, 7, _sanitize(text))
    pdf.set_draw_color(200, 190, 220)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _labelled(pdf: FPDF, label: str, text: str) -> None:
    # write() rather than multi_cell(markdown=True): sanitized text turns
    # em-dashes into "--", which fpdf2's markdown would read as strikethrough
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style="B", size=9.5)
    pdf.write(5, _sanitize(f"{label}: "))
    pdf.set_font("Helvetica", size=9.5)
    pdf.write(5, _sanitize(text))
    pdf.ln(5)


def _entry(pdf: FPDF, number: int, item: dict[str, Any], lines: list[tuple[str, str]]) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style="B", size=10.5)
    pdf.multi_cell(0, 6, _sanitize(f"{number}. {item['title']}"))
    for label, text in lines:
        if text:
            _labelled(pdf, label, text)
    if item.get("clause"):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", style="I", size=8)
        pdf.set_text_color(90, 90, 90)
        excerpt = " ".join(_truncate_clause(item["clause"]).split())
        pdf.multi_cell(0, 4, _sanitize(f"Regulation text (excerpt): {excerpt}"))
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def build_report_pdf(filename: str, report: dict[str, Any]) -> bytes:
    """Build the Stage C + D report PDF from build_report_sections() output
    and return its raw bytes."""
    pdf = FPDF(format="A4")
    pdf.set_margin(20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "Compliance Pre-Check Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _disclaimer_box(pdf, DISCLAIMER, text_color=(122, 31, 31))
    pdf.ln(4)

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Drawing: {_sanitize(filename)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT")

    failures, missing, to_confirm = report["failures"], report["missing"], report["to_confirm"]

    _section_heading(pdf, "Stage C -- Where the drawing does not meet the regulations")
    if not report["has_c"]:
        _labelled(pdf, "Not checked", "the regulation check has not been run for this drawing.")
    elif not failures:
        _labelled(pdf, "None found", "no value on the drawing breaks a regulation clause the check could match.")
    for i, item in enumerate(failures, 1):
        _entry(pdf, i, item, [
            ("Drawing shows", item["drawing_shows"]),
            ("What the regulations want", item["requirement"]),
            ("Source", item.get("source") or ""),
        ])

    _section_heading(pdf, "Stage D -- What the drawing is missing")
    if missing is None:
        _labelled(pdf, "Not checked", "the missing-items check has not been run for this drawing.")
    elif not missing:
        _labelled(pdf, "None found", "nothing the regulations expect for this type of drawing appears to be missing.")
    for i, item in enumerate(missing or [], 1):
        _entry(pdf, i, item, [
            ("Drawing area", item["area"]),
            ("What the regulations want", item["requirement"]),
            ("How sure", item["certainty"]),
            ("Source", item.get("source") or ""),
        ])

    if to_confirm:
        _section_heading(pdf, "Points to confirm")
        for i, item in enumerate(to_confirm, 1):
            _entry(pdf, i, item, [
                ("Drawing shows", item["drawing_shows"]),
                ("Why it needs a person to check", item["requirement"]),
                ("Source", item.get("source") or ""),
            ])

    if report["has_c"]:
        pdf.ln(2)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", size=9.5)
        pdf.multi_cell(0, 5, _sanitize(
            f"Of the {report['n_checked']} values read from the drawing, {report['n_passed']} met the "
            f"clause they were checked against and {report['n_no_clause']} had no clause that applies to them."
        ))

    pdf.ln(6)
    _disclaimer_box(pdf, DISCLAIMER, text_color=(120, 120, 120))
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", size=7.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "Not affiliated with, and not issued by, any Building Control authority.")

    return bytes(pdf.output())

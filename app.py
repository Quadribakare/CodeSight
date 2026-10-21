"""
Flask demo UI for the compliance-checking pipeline.

Lets you pick or upload a drawing PDF, trigger each pipeline stage
independently (Stage A classification, Stage B fact extraction, Stage C
matching + judgement), and view the results of whichever stages have run
so far. This is a demo/presentation layer only -- it calls the same
functions the CLI scripts use (classify_pdf, extract_features_for_pdf,
judge_features_for_pdf), nothing pipeline-specific lives here.

All reading/writing of the uploaded PDF and each stage's output goes
through src/storage.py, which is a real local disk when this runs as a
normal server and Vercel Blob storage (mirrored through /tmp) when it runs
as a Vercel Function -- see that module for why.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from flask import Flask, Response, redirect, render_template, request, url_for

from src import storage
from src.agents.absence_agent import detect_missing_items_for_pdf
from src.agents.classifier_agent import classify_pdf
from src.agents.feature_agent import extract_features_for_pdf
from src.agents.judge import judge_features_for_pdf
from src.report import build_missing_items_pdf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB, large drawing sheets


def _decode_filename(filename: str) -> str:
    """Undo Vercel's Python runtime handing routes a still-percent-encoded
    path segment instead of a decoded one (confirmed by direct inspection --
    a URL ending in "...(C01)%20Sections.pdf" arrived at the view function
    with the literal "%20" still in it, rather than a space). A normal WSGI
    server decodes this before Flask ever sees it, so only redo it on Vercel
    -- otherwise an already-decoded filename that happens to contain a literal
    "%" would get mangled by decoding it a second time.
    """
    return unquote(filename) if storage.IS_VERCEL else filename


def _stem(filename: str) -> str:
    return Path(filename).stem


def _pdf_status(filename: str) -> dict:
    return {"filename": filename, **storage.stage_status(_stem(filename))}


@app.route("/")
def index():
    pdfs = [_pdf_status(f) for f in storage.list_pdfs()]
    return render_template("index.html", pdfs=pdfs)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf_file")
    if file and file.filename.lower().endswith(".pdf"):
        storage.save_uploaded_pdf(file.filename, file)
        # A re-upload under an existing filename must not leave the previous
        # file's Stage A/B/C results looking like they belong to this one.
        storage.clear_outputs(_stem(file.filename))
        return redirect(url_for("dashboard", filename=file.filename))
    return redirect(url_for("index"))


@app.route("/pdf/<filename>")
def dashboard(filename: str):
    filename = _decode_filename(filename)
    if storage.ensure_local_pdf(filename) is None:
        return redirect(url_for("index"))

    stem = _stem(filename)
    storage.prepare_output_root(stem)
    classification = storage.read_json(stem, "classification.json")
    features = storage.read_json(stem, "features.json")
    verdicts = storage.read_json(stem, "verdicts.json")
    missing_items = storage.read_json(stem, "missing_items.json")

    verdict_counts = {}
    if verdicts:
        for v in verdicts:
            verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

    return render_template(
        "dashboard.html",
        filename=filename,
        classification=classification,
        features=features,
        verdicts=verdicts,
        verdict_counts=verdict_counts,
        missing_items=missing_items,
    )


@app.route("/pdf/<filename>/clear", methods=["POST"])
def clear_results(filename: str):
    filename = _decode_filename(filename)
    storage.clear_outputs(_stem(filename))
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/delete", methods=["POST"])
def delete_pdf(filename: str):
    filename = _decode_filename(filename)
    storage.delete_pdf(filename)
    return redirect(url_for("index"))


@app.route("/pdf/<filename>/run/all", methods=["POST"])
def run_all_stages(filename: str):
    filename = _decode_filename(filename)
    pdf_path = storage.ensure_local_pdf(filename)
    if pdf_path is None:
        return redirect(url_for("index"))
    stem = _stem(filename)
    out_root = storage.prepare_output_root(stem)
    # Sync after each stage, not just at the end, so a failure or a timeout
    # partway through (a large drawing's Stage C alone can take a while)
    # doesn't throw away whatever earlier stages already finished.
    classify_pdf(str(pdf_path), output_dir=str(out_root))
    storage.sync_output_dir(stem)
    extract_features_for_pdf(str(pdf_path), output_dir=str(out_root))
    storage.sync_output_dir(stem)
    judge_features_for_pdf(str(pdf_path), output_dir=str(out_root))
    storage.sync_output_dir(stem)
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/run/a", methods=["POST"])
def run_stage_a(filename: str):
    filename = _decode_filename(filename)
    pdf_path = storage.ensure_local_pdf(filename)
    if pdf_path is None:
        return redirect(url_for("index"))
    stem = _stem(filename)
    out_root = storage.prepare_output_root(stem)
    classify_pdf(str(pdf_path), output_dir=str(out_root))
    storage.sync_output_dir(stem)
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/run/b", methods=["POST"])
def run_stage_b(filename: str):
    filename = _decode_filename(filename)
    pdf_path = storage.ensure_local_pdf(filename)
    if pdf_path is None:
        return redirect(url_for("index"))
    stem = _stem(filename)
    out_root = storage.prepare_output_root(stem)
    extract_features_for_pdf(str(pdf_path), output_dir=str(out_root))
    storage.sync_output_dir(stem)
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/run/c", methods=["POST"])
def run_stage_c(filename: str):
    filename = _decode_filename(filename)
    limit_raw = request.form.get("limit", "").strip()
    limit = int(limit_raw) if limit_raw else None
    pdf_path = storage.ensure_local_pdf(filename)
    if pdf_path is None:
        return redirect(url_for("index"))
    stem = _stem(filename)
    out_root = storage.prepare_output_root(stem)
    judge_features_for_pdf(str(pdf_path), output_dir=str(out_root), limit=limit)
    storage.sync_output_dir(stem)
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/run/d", methods=["POST"])
def run_stage_d(filename: str):
    filename = _decode_filename(filename)
    pdf_path = storage.ensure_local_pdf(filename)
    if pdf_path is None:
        return redirect(url_for("index"))
    stem = _stem(filename)
    out_root = storage.prepare_output_root(stem)
    detect_missing_items_for_pdf(str(pdf_path), output_dir=str(out_root))
    storage.sync_output_dir(stem)
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/report.pdf")
def download_report(filename: str):
    filename = _decode_filename(filename)
    stem = _stem(filename)
    storage.prepare_output_root(stem)
    missing_items = storage.read_json(stem, "missing_items.json")
    if missing_items is None:
        return redirect(url_for("dashboard", filename=filename))
    pdf_bytes = build_missing_items_pdf(filename, missing_items)
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{stem}-missing-items.pdf"'},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

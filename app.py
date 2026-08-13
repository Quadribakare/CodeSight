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

from flask import Flask, redirect, render_template, request, url_for

from src import storage
from src.agents.classifier_agent import classify_pdf
from src.agents.feature_agent import extract_features_for_pdf
from src.agents.judge import judge_features_for_pdf

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


@app.route("/debug/inspect")
def debug_inspect():
    # Temporary: check actual stored file size + classification reasoning.
    # Remove after.
    filename = request.args.get("filename", "")
    pathname = storage.DATA_PREFIX + filename
    matches = [b for b in storage._blob_list(pathname) if b["pathname"] == pathname]
    stem = _stem(filename)
    storage.prepare_output_root(stem)
    classification = storage.read_json(stem, "classification.json")
    vector = storage.read_json(stem, "page_000_vector.json")
    image_path = storage.LOCAL_OUTPUT_DIR / stem / "page_000.png"

    # Fresh download + fresh fitz read, bypassing any stale saved output.
    import fitz

    from src.extraction.vector_reader import extract_pdf_vector_data

    (storage.LOCAL_DATA_DIR / filename).unlink(missing_ok=True)
    fresh_pdf_path = storage.ensure_local_pdf(filename)
    fresh_downloaded_bytes = fresh_pdf_path.stat().st_size if fresh_pdf_path else None
    fresh_header = fresh_pdf_path.read_bytes()[:8].hex() if fresh_pdf_path else None
    fresh_page_count = None
    fresh_span_count = None
    fresh_error = None
    try:
        if fresh_pdf_path:
            with fitz.open(str(fresh_pdf_path)) as doc:
                fresh_page_count = doc.page_count
            fresh_pages = extract_pdf_vector_data(str(fresh_pdf_path))
            fresh_span_count = sum(len(p.text_spans) for p in fresh_pages)
    except Exception as exc:
        fresh_error = repr(exc)

    return {
        "blob_size_bytes": matches[0].get("size") if matches else None,
        "blob_uploaded_at": matches[0].get("uploadedAt") if matches else None,
        "classification": classification,
        "fresh_downloaded_bytes": fresh_downloaded_bytes,
        "fresh_header_hex": fresh_header,
        "fresh_page_count": fresh_page_count,
        "fresh_span_count": fresh_span_count,
        "fresh_error": fresh_error,
        "vector_text_span_count": len(vector.get("text_spans", [])) if vector else None,
        "vector_first_5_spans": (vector.get("text_spans", [])[:5]) if vector else None,
        "rendered_image_bytes": image_path.stat().st_size if image_path.exists() else None,
    }


@app.route("/debug/image")
def debug_image():
    filename = request.args.get("filename", "")
    stem = _stem(filename)
    storage.prepare_output_root(stem)
    image_path = storage.LOCAL_OUTPUT_DIR / stem / "page_000.png"
    if not image_path.exists():
        return "no image", 404
    from flask import send_file

    return send_file(image_path, mimetype="image/png")


@app.route("/")
def index():
    pdfs = [_pdf_status(f) for f in storage.list_pdfs()]
    return render_template("index.html", pdfs=pdfs)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf_file")
    if file and file.filename.lower().endswith(".pdf"):
        storage.save_uploaded_pdf(file.filename, file)
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
    )


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)

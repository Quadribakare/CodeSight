"""
Flask demo UI for the compliance-checking pipeline.

Lets you pick or upload a drawing PDF, trigger each pipeline stage
independently (Stage A classification, Stage B fact extraction, Stage C
matching + judgement), and view the results of whichever stages have run
so far. This is a demo/presentation layer only -- it calls the same
functions the CLI scripts use (classify_pdf, extract_features_for_pdf,
judge_features_for_pdf), nothing pipeline-specific lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

from src.agents.classifier_agent import classify_pdf
from src.agents.feature_agent import extract_features_for_pdf
from src.agents.judge import judge_features_for_pdf

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB, large drawing sheets


def _list_pdfs() -> list[str]:
    DATA_DIR.mkdir(exist_ok=True)
    return sorted(p.name for p in DATA_DIR.glob("*.pdf"))


def _stem(filename: str) -> str:
    return Path(filename).stem


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _pdf_status(filename: str) -> dict:
    stem = _stem(filename)
    out_dir = OUTPUT_DIR / stem
    return {
        "filename": filename,
        "has_a": (out_dir / "classification.json").exists(),
        "has_b": (out_dir / "features.json").exists(),
        "has_c": (out_dir / "verdicts.json").exists(),
    }


@app.route("/")
def index():
    pdfs = [_pdf_status(f) for f in _list_pdfs()]
    return render_template("index.html", pdfs=pdfs)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf_file")
    if file and file.filename.lower().endswith(".pdf"):
        DATA_DIR.mkdir(exist_ok=True)
        file.save(DATA_DIR / file.filename)
        return redirect(url_for("dashboard", filename=file.filename))
    return redirect(url_for("index"))


@app.route("/pdf/<filename>")
def dashboard(filename: str):
    pdf_path = DATA_DIR / filename
    if not pdf_path.exists():
        return redirect(url_for("index"))

    out_dir = OUTPUT_DIR / _stem(filename)

    classification = _load_json(out_dir / "classification.json")
    features = _load_json(out_dir / "features.json")
    verdicts = _load_json(out_dir / "verdicts.json")

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
    classify_pdf(str(DATA_DIR / filename), output_dir=str(OUTPUT_DIR))
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/run/b", methods=["POST"])
def run_stage_b(filename: str):
    extract_features_for_pdf(str(DATA_DIR / filename), output_dir=str(OUTPUT_DIR))
    return redirect(url_for("dashboard", filename=filename))


@app.route("/pdf/<filename>/run/c", methods=["POST"])
def run_stage_c(filename: str):
    limit_raw = request.form.get("limit", "").strip()
    limit = int(limit_raw) if limit_raw else None
    judge_features_for_pdf(str(DATA_DIR / filename), output_dir=str(OUTPUT_DIR), limit=limit)
    return redirect(url_for("dashboard", filename=filename))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

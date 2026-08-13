"""
Persistence layer for the Flask demo app -- local disk when run normally,
Vercel Blob storage when deployed as a Vercel Function.

Why this exists: app.py and the three pipeline stages (classify_pdf,
extract_features_for_pdf, judge_features_for_pdf) were written assuming a
normal always-on server with a real, persistent disk under data/ and
outputs/. Vercel Functions don't give you that -- the deployed filesystem is
read-only except /tmp, and /tmp itself is not guaranteed to survive between
requests (Stage A, B, C are separate HTTP requests from separate button
clicks in the UI, and Vercel is free to route each to a fresh container).

Rather than rewrite the pipeline stages themselves, this module gives them
what they already expect -- a real local directory to read/write -- and
transparently mirrors that directory to Vercel Blob storage around each
stage run, only when actually running on Vercel. Locally (`python app.py`),
every function below is a thin pass-through to plain filesystem calls,
identical to the app's original behaviour.

Requires a Blob store connected to the Vercel project (adds the
BLOB_READ_WRITE_TOKEN env var automatically) -- see README for setup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

# Vercel only exposes the `VERCEL` system env var when "Enable access to
# System Environment Variables" is turned on in project settings -- easy to
# forget, and this app doesn't need anything else from that gated set. So
# detect Vercel by the one var this app actually depends on instead:
# BLOB_READ_WRITE_TOKEN, added automatically the moment a Blob store is
# connected, regardless of that toggle.
IS_VERCEL = bool(os.environ.get("VERCEL")) or bool(os.environ.get("BLOB_READ_WRITE_TOKEN"))

BASE_DIR = Path(__file__).resolve().parent.parent

# Read-only on Vercel (whatever data/*.pdf was committed to git, e.g. the
# two sample drawings); on a normal server this is also where uploads land.
BUNDLED_DATA_DIR = BASE_DIR / "data"

# Writable scratch space: /tmp on Vercel (ephemeral, request-local), the
# same data/ and outputs/ folders everywhere else.
LOCAL_DATA_DIR = Path("/tmp/data") if IS_VERCEL else BUNDLED_DATA_DIR
LOCAL_OUTPUT_DIR = Path("/tmp/outputs") if IS_VERCEL else BASE_DIR / "outputs"

DATA_PREFIX = "data/"
OUTPUT_PREFIX = "outputs/"

if IS_VERCEL:
    if not os.environ.get("BLOB_READ_WRITE_TOKEN"):
        raise RuntimeError(
            "Running on Vercel but BLOB_READ_WRITE_TOKEN is not set -- create a "
            "Blob store (Project > Storage > Create Database > Blob) and connect "
            "it to this project, then redeploy."
        )
    import vercel_blob


def _download_blob(url: str, dest_path: Path) -> None:
    """Download a blob's content straight to `dest_path`.

    Deliberately not using vercel_blob's own download_file(): it only
    accepts paths relative to the current working directory and silently
    mangles absolute paths (it does os.getcwd() + path.lstrip('/')), so
    downloading into an absolute /tmp path fails an internal assertion.
    """
    resp = requests.get(f"{url}?download=1", timeout=30)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)


def _blob_list(prefix: str) -> list[dict]:
    """All blob entries whose pathname starts with `prefix` (paginated).

    Passes `prefix` through to the API as a hint, but always re-filters
    client-side too -- so this stays correct even if a given API version
    ignores it and returns the whole store.
    """
    items: list[dict] = []
    cursor: str | None = None
    while True:
        options: dict[str, str] = {"prefix": prefix, "limit": "1000"}
        if cursor:
            options["cursor"] = cursor
        resp = vercel_blob.list(options)
        items.extend(b for b in resp.get("blobs", []) if b["pathname"].startswith(prefix))
        if not resp.get("hasMore"):
            break
        cursor = resp.get("cursor")
    return items


def list_pdfs() -> list[str]:
    """Every PDF available to run the pipeline on: bundled samples plus uploads."""
    names = {p.name for p in BUNDLED_DATA_DIR.glob("*.pdf")} if BUNDLED_DATA_DIR.exists() else set()
    if IS_VERCEL:
        names |= {
            b["pathname"][len(DATA_PREFIX):]
            for b in _blob_list(DATA_PREFIX)
            if b["pathname"].lower().endswith(".pdf")
        }
    return sorted(names)


def save_uploaded_pdf(filename: str, file_storage) -> None:
    """Persist an uploaded PDF wherever writes actually stick."""
    if not IS_VERCEL:
        BUNDLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_storage.save(BUNDLED_DATA_DIR / filename)
        return
    vercel_blob.put(DATA_PREFIX + filename, file_storage.read())


def ensure_local_pdf(filename: str) -> Path | None:
    """Return a real filesystem path to `filename` (for fitz to open), or None
    if it doesn't exist anywhere. Downloads from Blob storage into /tmp first
    if this is a Vercel request that hasn't seen this PDF yet."""
    bundled_path = BUNDLED_DATA_DIR / filename
    if bundled_path.exists():
        return bundled_path

    local_path = LOCAL_DATA_DIR / filename
    if local_path.exists():
        return local_path
    if not IS_VERCEL:
        return None

    pathname = DATA_PREFIX + filename
    matches = [b for b in _blob_list(pathname) if b["pathname"] == pathname]
    if not matches:
        return None
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _download_blob(matches[0]["url"], local_path)
    return local_path


def prepare_output_root(stem: str) -> Path:
    """Ensure LOCAL_OUTPUT_DIR/<stem>/ exists, pre-populated (on Vercel) with
    whatever this PDF's earlier stage runs already saved to Blob storage.

    Returns the *root* output dir -- the same `output_dir` value
    classify_pdf/extract_features_for_pdf/judge_features_for_pdf all take,
    since each of them appends the PDF's stem itself.
    """
    stem_dir = LOCAL_OUTPUT_DIR / stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    if IS_VERCEL:
        prefix = f"{OUTPUT_PREFIX}{stem}/"
        for blob in _blob_list(prefix):
            rel_name = blob["pathname"][len(prefix):]
            dest = stem_dir / rel_name
            if not dest.exists():
                _download_blob(blob["url"], dest)
    return LOCAL_OUTPUT_DIR


def sync_output_dir(stem: str) -> None:
    """Push whatever a pipeline stage just wrote in /tmp back up to Blob
    storage. No-op off Vercel, where it's already on real disk."""
    if not IS_VERCEL:
        return
    stem_dir = LOCAL_OUTPUT_DIR / stem
    if not stem_dir.exists():
        return
    for path in stem_dir.iterdir():
        if path.is_file():
            vercel_blob.put(f"{OUTPUT_PREFIX}{stem}/{path.name}", path.read_bytes())


def read_json(stem: str, name: str) -> Any:
    """Read one output JSON file (e.g. 'classification.json') for `stem`.

    Assumes prepare_output_root(stem) has already been called this request.
    """
    path = LOCAL_OUTPUT_DIR / stem / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stage_status(stem: str) -> dict:
    """Cheap has_a/has_b/has_c check -- lists filenames only, no downloads."""
    if IS_VERCEL:
        names = {b["pathname"].rsplit("/", 1)[-1] for b in _blob_list(f"{OUTPUT_PREFIX}{stem}/")}
    else:
        out_dir = LOCAL_OUTPUT_DIR / stem
        names = {p.name for p in out_dir.glob("*")} if out_dir.exists() else set()
    return {
        "has_a": "classification.json" in names,
        "has_b": "features.json" in names,
        "has_c": "verdicts.json" in names,
    }

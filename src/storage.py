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
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

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


def _quote(name: str) -> str:
    """Percent-encode one filename/stem for vercel_blob.put() only.

    put() builds its request URL by pasting the pathname straight into an
    f-string instead of encoding it, so a real drawing filename with spaces
    or parentheses (e.g. "25-112-A250(C01) Sections.pdf" -- typical of
    CAD-exported sheets) would otherwise go out as an invalid URL. Vercel's
    server decodes the query param on arrival, so what actually gets stored
    is the original literal string -- this is a write-time-only workaround.

    Don't use this for list()'s `prefix` or for comparing against a
    `pathname` a list() call already returned: those already go through
    requests' own (correct, single-pass) query encoding, and the pathnames
    list() gives back are the literal stored strings, not encoded ones.
    Pre-encoding on that side double-encodes and breaks lookups instead.
    """
    return quote(name, safe="")


_DOWNLOAD_RETRY_DELAYS_SEC = (0, 0.5, 1, 2, 3, 5, 8)  # ~19.5s total budget


def _download_blob(url: str, dest_path: Path) -> None:
    """Download a blob's content straight to `dest_path`.

    Deliberately not using vercel_blob's own download_file(): it only
    accepts paths relative to the current working directory and silently
    mangles absolute paths (it does os.getcwd() + path.lstrip('/')), so
    downloading into an absolute /tmp path fails an internal assertion.

    Retries a few times on 404: blob URLs sit behind a CDN, and deleting a
    file then re-uploading to the same pathname within a second or two can
    hit a brief window where the edge still serves a cached "not found" for
    that URL before the new object has propagated -- confirmed by hitting
    exactly this right after testing the delete button on a real file.
    """
    resp = None
    for delay in _DOWNLOAD_RETRY_DELAYS_SEC:
        if delay:
            time.sleep(delay)
        resp = requests.get(f"{url}?download=1", timeout=30)
        if resp.status_code != 404:
            break
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
        for b in _blob_list(DATA_PREFIX):
            name = b["pathname"][len(DATA_PREFIX):]
            if name.lower().endswith(".pdf"):
                names.add(name)
    return sorted(names)


def save_uploaded_pdf(filename: str, file_storage) -> None:
    """Persist an uploaded PDF wherever writes actually stick."""
    if not IS_VERCEL:
        BUNDLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_storage.save(BUNDLED_DATA_DIR / filename)
        return
    data = file_storage.read()
    vercel_blob.put(DATA_PREFIX + _quote(filename), data, {"allowOverwrite": "true"})
    # Also cache locally right away, so the redirect that immediately follows
    # an upload can serve this same file straight from /tmp instead of
    # round-tripping through Blob's CDN -- which can briefly 404 right after
    # a delete followed by a re-upload to the same pathname -- if that
    # request happens to land on this same warm container.
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_DIR / filename).write_bytes(data)


def delete_pdf(filename: str) -> None:
    """Permanently remove `filename` itself plus all its saved Stage A/B/C
    output -- both local disk (or the /tmp working copy on Vercel) and the
    Blob-stored copies. Used by the "Delete" button on the drawings list.
    """
    clear_outputs(Path(filename).stem)

    if not IS_VERCEL:
        (BUNDLED_DATA_DIR / filename).unlink(missing_ok=True)
        return

    (LOCAL_DATA_DIR / filename).unlink(missing_ok=True)
    pathname = DATA_PREFIX + filename
    matches = [b for b in _blob_list(pathname) if b["pathname"] == pathname]
    if matches:
        vercel_blob.delete([b["url"] for b in matches])


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
            vercel_blob.put(
                f"{OUTPUT_PREFIX}{_quote(stem)}/{_quote(path.name)}",
                path.read_bytes(),
                {"allowOverwrite": "true"},
            )


def clear_outputs(stem: str) -> None:
    """Delete every saved Stage A/B/C artifact for `stem` -- both the /tmp
    working copy and the Blob-stored copies. Used when a PDF is re-uploaded
    under a name that already has results (so the dashboard doesn't keep
    showing a previous file's classification/facts/verdicts as if they
    belonged to the new one) and by the manual "Clear results" button.
    """
    stem_dir = LOCAL_OUTPUT_DIR / stem
    if stem_dir.exists():
        shutil.rmtree(stem_dir)
    if IS_VERCEL:
        # _blob_list's prefix goes through requests' own (correct) query
        # encoding -- do NOT _quote() it here, same reasoning as everywhere
        # else in this module that lists/reads rather than writes.
        prefix = f"{OUTPUT_PREFIX}{stem}/"
        urls = [b["url"] for b in _blob_list(prefix)]
        if urls:
            vercel_blob.delete(urls)


def read_json(stem: str, name: str) -> Any:
    """Read one output JSON file (e.g. 'classification.json') for `stem`.

    Assumes prepare_output_root(stem) has already been called this request.
    """
    path = LOCAL_OUTPUT_DIR / stem / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stage_status(stem: str) -> dict:
    """Cheap has_a/has_b/has_c/has_d check -- lists filenames only, no downloads."""
    if IS_VERCEL:
        prefix = f"{OUTPUT_PREFIX}{stem}/"
        names = {b["pathname"][len(prefix):] for b in _blob_list(prefix)}
    else:
        out_dir = LOCAL_OUTPUT_DIR / stem
        names = {p.name for p in out_dir.glob("*")} if out_dir.exists() else set()
    return {
        "has_a": "classification.json" in names,
        "has_b": "features.json" in names,
        "has_c": "verdicts.json" in names,
        "has_d": "missing_items.json" in names,
    }

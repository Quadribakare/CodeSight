# Architectural Drawing Compliance Pipeline

MSc dissertation project: a staged multimodal pipeline that reads CAD-exported
architectural drawing PDFs (vector text + geometry, no raster images), classifies
each page, extracts compliance-relevant facts, matches them against a knowledge
base of building-regulation clauses, and produces a pass/fail/flag report.

**Scope:** drawings only. Administrative documents (rejection letters, approval
notices) are used externally as evaluation ground truth and are never processed
by the pipeline itself.

## Pipeline stages

1. **Stage A — Classification** (`src/agents/classifier_agent.py`): classifies
   each page into a drawing type (Architectural / Engineering / Finishes /
   Safety / Not a drawing), fusing vector text/geometry with a rendered image.
2. **Stage B — Feature extraction** (`src/agents/feature_agent.py`): extracts
   compliance-relevant facts from each classified page — both labelled
   dimensions on the drawing and requirements written in specification notes
   (U-values, material grades, named standards, spacing requirements).
3. **Stage C — Clause matching & judgement** (`src/agents/matcher.py` +
   `src/agents/judge.py`): finds candidate excerpts from the regulation
   knowledge base for each fact, then judges pass / fail / flag /
   no_matching_clause with cited evidence.

Each stage reads the previous stage's saved output rather than re-running it,
so re-running Stage C doesn't re-pay for Stage A or B's API calls.
`src/pipeline.py` (not yet built) would eventually chain all three into one
command; for now each stage is run separately (see below), which the web demo
also reflects with three separate buttons rather than one combined action.

A supplementary module, `src/agents/vision_transcriber.py`, tiles a page into
high-DPI images and transcribes it independently of the PDF's text layer —
useful as an occasional audit tool on an unfamiliar drawing set, not part of
the default pipeline (it costs ~20 API calls per page vs. ~1).

## Project layout

```
data/                     # drawing PDFs (gitignored, not committed)
knowledge_base/
  regulations_combined.txt   # extracted regulation text, 3 source documents combined
knowledge_base_schema/
  clause_schema.json       # JSON Schema for a structured clause record (not yet populated --
                            # matcher.py currently searches raw text chunks as an interim approach)
src/
  extraction/
    vector_reader.py       # PyMuPDF text + geometry extraction (no AI)
    image_renderer.py      # PyMuPDF page-to-image rendering, incl. tiled high-DPI rendering
  agents/
    classifier_agent.py    # Stage A
    feature_agent.py       # Stage B
    matcher.py             # Stage C: candidate clause retrieval (keyword-based, no AI)
    judge.py               # Stage C: compliance judgement
    vision_transcriber.py  # supplementary vision-based transcription (see above)
  pipeline.py              # orchestrator (stub -- not yet built)
  openrouter_client.py     # OpenRouter (OpenAI-compatible) API wrapper
app.py                     # Flask demo web UI
templates/, static/        # Flask demo UI's HTML/CSS/JS
tests/
outputs/                   # all per-PDF results land here, one subfolder per PDF (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in OPENROUTER_API_KEY
```

Put one or more drawing PDFs in `data/` before running anything.

## Model

- `google/gemini-3.1-flash-lite-20260507`, used for every stage (classification,
  feature extraction, clause matching, compliance judgement) via OpenRouter.

## Running the pipeline

Each stage is a separate command. Run them in order — each one requires the
previous stage's output to already exist for that PDF.

```bash
# Stage A -- classify every page (fast, ~1 API call per page)
python -m src.agents.classifier_agent "data\your-drawing.pdf"

# Stage B -- extract compliance-relevant facts (requires Stage A's output)
python -m src.agents.feature_agent "data\your-drawing.pdf"

# Stage C -- match facts against regulations and judge pass/fail/flag
# (requires Stage B's output; add a number to test on a subset first, e.g. `5`)
python -m src.agents.judge "data\your-drawing.pdf" [optional-limit]
```

All output lands in `outputs\<pdf-name>\`:
`classification.json`, `page_NNN_vector.json`, `page_NNN.png` (Stage A),
`features.json` (Stage B), `verdicts.json` + `verdicts_summary.json` (Stage C).

Optional supplementary check (expensive — see note above):
```bash
python -m src.agents.vision_transcriber "data\your-drawing.pdf" <page-number>
```

## Running the demo web UI

```bash
python app.py
```
Then open **http://127.0.0.1:5000/** in a browser. From there:
- Upload a PDF, or click one already listed under `data/`.
- On the dashboard, click **Run Stage A**, then **Run Stage B**, then
  **Run Stage C** — each button calls the same functions as the CLI commands
  above and displays the results as tables once done.
- In the Stage C results, click the pass / fail / flag / no-match badges to
  filter the verdict table to just that group.
## Running the pipeline

Each stage is run manually, in order — there is no orchestrator yet tying
them together automatically. Every stage depends on the previous stage's
output already existing on disk under `outputs/<pdf-stem>/`.

### Stage A — Classification

```bash
python -m src.agents.classifier_agent <path-to-pdf>
```

Reads the PDF directly. Produces, under `outputs/<pdf-stem>/`:
- `page_NNN_vector.json` — raw extracted text and geometry, per page
- `page_NNN.png` — rendered image, per page
- `classification.json` — final classification results, all pages

### Stage B — Fact Extraction

```bash
python -m src.agents.feature_agent <path-to-pdf>
```

Requires Stage A to have already run on this PDF. Produces:
- `features.json` — every extracted fact, with source quote and coordinates

### Stage C — Regulation Matching and Judgement

```bash
python -m src.agents.judge <path-to-pdf> [limit]
```

Requires Stage B to have already run on this PDF. `limit` is optional — caps
how many facts get judged, useful for a cheap sanity check before paying for
a full run (e.g. `judge.py drawing.pdf 6` to test on the first 6 facts only).
Produces:
- `verdicts.json` — one verdict (pass/fail/flag/no_matching_clause) per fact
- `verdicts_summary.json` — counts, plus the failures and flags pulled out

### Example — full run on one drawing

```bash
python -m src.agents.classifier_agent data/25-102-A100.pdf
python -m src.agents.feature_agent data/25-102-A100.pdf
python -m src.agents.judge data/25-102-A100.pdf
```

### Matcher.py standalone (debugging only, not part of the normal flow)

```bash
python -m src.agents.matcher
```

Loads and prints a per-document chunk count from the regulation corpus. Can
optionally take one fact as a JSON string argument to test retrieval against
it directly, without running the full pipeline.
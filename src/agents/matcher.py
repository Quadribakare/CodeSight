"""
Stage C (part 1) of the pipeline: clause matching against the knowledge base.

Given a fact produced by feature_agent, searches knowledge_base/regulations_combined.txt
for candidate excerpts that might be relevant to it. Splits the combined regulations
text into paragraph-level chunks (tagged by which source document each came from),
then ranks chunks by simple keyword overlap with the fact's parameter name, unit,
and source quote -- no AI call needed for this retrieval step, since it's cheap
and fast enough to do mechanically. The candidates returned here are not
guaranteed relevant -- judge.py makes the actual pass/fail/flag call, including
deciding a candidate isn't actually relevant.

This does not yet use knowledge_base_schema/clause_schema.json's structured
clause records, since the regulations haven't been split into individual
clause records yet -- this works directly off the raw combined text as an
interim step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REGULATIONS_PATH = "knowledge_base/regulations_combined.txt"

_DOC_HEADER_PATTERN = re.compile(r"={80}\nSOURCE DOCUMENT: (.+?)\nPAGES: \d+\n={80}\n\n")

_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "be", "is",
    "are", "with", "at", "by", "this", "that", "as", "it", "its", "from",
}

_MIN_CHUNK_LENGTH = 40


@dataclass
class RegulationChunk:
    """One paragraph-level excerpt from the combined regulations text."""

    text: str
    source_document: str
    chunk_index: int


def load_regulation_chunks(path: str = DEFAULT_REGULATIONS_PATH) -> list[RegulationChunk]:
    """Split the combined regulations text into paragraph chunks, tagged by source document."""
    full_text = Path(path).read_text(encoding="utf-8")
    parts = _DOC_HEADER_PATTERN.split(full_text)

    chunks: list[RegulationChunk] = []
    # parts alternates: [preamble, doc_name_1, doc_text_1, doc_name_2, doc_text_2, ...]
    for i in range(1, len(parts), 2):
        doc_name = parts[i].strip()
        doc_text = parts[i + 1] if i + 1 < len(parts) else ""
        paragraphs = re.split(r"\n\s*\n", doc_text)
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) >= _MIN_CHUNK_LENGTH:
                chunks.append(RegulationChunk(text=para, source_document=doc_name, chunk_index=idx))
    return chunks


def _fact_terms(fact: dict[str, Any]) -> set[str]:
    term_source = " ".join(
        [
            fact.get("parameter_name", "").replace("_", " "),
            fact.get("unit") or "",
            fact.get("source_quote", ""),
        ]
    )
    words = set(re.findall(r"[a-z0-9]+", term_source.lower()))
    return words - _STOPWORDS


def find_candidate_clauses_for_terms(
    terms: set[str],
    chunks: list[RegulationChunk],
    top_k: int = 5,
) -> list[RegulationChunk]:
    """Rank regulation chunks by keyword overlap with an arbitrary term set.

    Shared primitive behind find_candidate_clauses (per-fact queries, Stage C)
    and absence_agent's per-drawing-type queries (Stage D). Returns an empty
    list if nothing scores above zero overlap -- that's a valid, honest
    result, not an error: the caller should report "no match" rather than
    being handed an irrelevant chunk to force a verdict from.
    """
    if not terms:
        return []

    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        overlap = len(terms & chunk_words)
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda pair: -pair[0])
    return [chunk for _, chunk in scored[:top_k]]


def find_candidate_clauses(
    fact: dict[str, Any],
    chunks: list[RegulationChunk],
    top_k: int = 5,
) -> list[RegulationChunk]:
    """Rank regulation chunks by keyword overlap with a fact, returning the top_k best matches."""
    return find_candidate_clauses_for_terms(_fact_terms(fact), chunks, top_k)


if __name__ == "__main__":
    import sys

    chunks = load_regulation_chunks()
    print(f"Loaded {len(chunks)} regulation chunk(s) from '{DEFAULT_REGULATIONS_PATH}'")

    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.source_document] = by_doc.get(c.source_document, 0) + 1
    for doc, count in by_doc.items():
        print(f"  {doc}: {count} chunk(s)")

    if len(sys.argv) == 2:
        import json

        fact = json.loads(sys.argv[1])
        candidates = find_candidate_clauses(fact, chunks)
        print(f"\nTop {len(candidates)} candidate(s) for fact {fact}:")
        for c in candidates:
            print(f"  [{c.source_document} #{c.chunk_index}] {c.text[:150]}")

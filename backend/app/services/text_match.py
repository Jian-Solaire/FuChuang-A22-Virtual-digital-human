"""Shared text normalization helpers for mock retrieval."""

from __future__ import annotations

import re


_NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_text(text: str) -> str:
    return _NON_WORD_RE.sub(" ", text.lower()).strip()


def token_score(query: str, text: str) -> float:
    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)
    query_terms = [term for term in normalized_query.split() if term]
    if not query_terms or not normalized_text:
        return 0.0
    matches = sum(1 for term in query_terms if term in normalized_text)
    if matches == 0:
        return 0.0
    return round(matches / len(query_terms), 4)

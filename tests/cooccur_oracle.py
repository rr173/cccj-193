"""Independent brute-force oracle for windowed phrase co-occurrence.

Deliberately disjoint from every production module: it imports nothing from
``diskfm``, never uses an FM index, never calls the production sorting,
grouping, window-join or paging code, and never reuses the production
request model.  Expected values are derived straight from the original
strings:

* every hit of every phrase is enumerated position by position with
  ``str.find`` restarting one code point after each hit (overlap allowed);
* ``doc_id`` order is Python's Unicode code point order;
* for each requested pair and the same document, *all* combinations of one
  hit of A and one hit of B are enumerated and kept iff
  ``abs(a_offset - b_offset) <= window`` -- no sweep, no pointer, no
  binary search;
* both coordinate kinds are recomputed as ``len(text[:pos].encode("utf-8"))``.

The oracle also enumerates duplicates explicitly rather than counting them,
so it independently proves the "each combination exactly once" rule.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def brute_hits(text: str, phrase: str) -> List[int]:
    """All overlapping code-point offsets of ``phrase`` in ``text``."""
    out: List[int] = []
    start = 0
    while True:
        pos = text.find(phrase, start)
        if pos < 0:
            break
        out.append(pos)
        start = pos + 1
    return out


def _byte_offsets(text: str, positions: List[int]) -> Dict[int, int]:
    """Independent UTF-8 byte offset for each requested code-point offset."""
    return {p: len(text[:p].encode("utf-8")) for p in positions}


def expected_summary(docs: List[Tuple[str, str]],
                     patterns: List[Tuple[str, str]],
                     pairs: List[Tuple[str, str]],
                     window: int) -> List[dict]:
    """Per (pair, doc) non-zero totals in request pair order, then doc_id.

    ``patterns`` is a list of ``(pattern_id, phrase)``; ``pairs`` a list of
    ``(pattern_id_a, pattern_id_b)``.
    """
    phrase_of = dict(patterns)
    hits = {}
    for doc_id, text in docs:
        for pid, phrase in patterns:
            hits[(doc_id, pid)] = brute_hits(text, phrase)
    doc_ids = sorted({d for d, _t in docs})
    rows: List[dict] = []
    for a_id, b_id in pairs:
        for doc_id in doc_ids:
            a_hits = hits[(doc_id, a_id)]
            b_hits = hits[(doc_id, b_id)]
            count = 0
            for a in a_hits:
                for b in b_hits:
                    if abs(a - b) <= window:
                        count += 1
            if count:
                rows.append({
                    "pattern_a": a_id,
                    "pattern_b": b_id,
                    "doc_id": doc_id,
                    "cooccurrence_pairs": count,
                })
    return rows


def expected_pairs(docs: List[Tuple[str, str]],
                   patterns: List[Tuple[str, str]],
                   pair: Tuple[str, str], window: int) -> List[dict]:
    """Every position pair for one requested pair, in production order.

    Order: doc_id code point order, then ``(a_cp, b_cp)``.  Exactly one row
    per distinct ``(doc_id, a_cp, b_cp)`` combination; duplicates are
    asserted away rather than deduplicated silently.
    """
    a_id, b_id = pair
    phrase_of = dict(patterns)
    texts = dict(docs)
    rows: List[dict] = []
    for doc_id in sorted(texts):
        text = texts[doc_id]
        a_hits = brute_hits(text, phrase_of[a_id])
        b_hits = brute_hits(text, phrase_of[b_id])
        a_bytes = _byte_offsets(text, a_hits)
        b_bytes = _byte_offsets(text, b_hits)
        combos = set()
        for a in a_hits:
            for b in b_hits:
                if abs(a - b) <= window:
                    key = (a, b)
                    if key in combos:  # pragma: no cover - oracle invariant
                        raise AssertionError("duplicate combination")
                    combos.add(key)
                    rows.append({
                        "pattern_a": a_id,
                        "pattern_b": b_id,
                        "doc_id": doc_id,
                        "a_codepoint_offset": a,
                        "a_utf8_byte_offset": a_bytes[a],
                        "b_codepoint_offset": b,
                        "b_utf8_byte_offset": b_bytes[b],
                    })
    rows.sort(key=lambda r: (r["doc_id"],
                             r["a_codepoint_offset"],
                             r["b_codepoint_offset"]))
    return rows


def expected_pair_counts(docs: List[Tuple[str, str]],
                         patterns: List[Tuple[str, str]],
                         pairs: List[Tuple[str, str]],
                         window: int) -> List[int]:
    """Total position pairs per requested pair, in request order."""
    out = []
    for pair in pairs:
        out.append(len(expected_pairs(docs, patterns, pair, window)))
    return out

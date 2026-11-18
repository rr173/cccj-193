"""Independent brute-force oracle for the similarity catalog.

Deliberately disjoint from every diskfm similarity implementation module:
shingles are generated per document with plain Python sets over the
*original* strings, candidate pairs are enumerated exhaustively, and
intersection/union sizes are computed with integer set algebra.  Nothing
here imports the catalog builder, dictionary, posting or candidate code,
so a bug in any of those cannot leak into the expected results.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple


def shingle_set(text: str, q: int) -> frozenset:
    """Distinct q-codepoint shingles of one document (set semantics)."""
    if len(text) < q:
        return frozenset()
    return frozenset(text[i:i + q] for i in range(len(text) - q + 1))


def shingle_multiset(text: str, q: int) -> Dict[str, int]:
    """Occurrence count of every distinct shingle (for explain checks)."""
    counts: Dict[str, int] = {}
    if len(text) < q:
        return counts
    for i in range(len(text) - q + 1):
        gram = text[i:i + q]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def expected_query(docs: List[Tuple[str, str]], q: int, source_id: str,
                   threshold: Fraction) -> List[dict]:
    """Expected ordered result rows for one query.

    Order: similarity (exact Fraction) descending, then target doc_id in
    Unicode code point order.  The source document never appears; pairs
    below the threshold (including zero-share pairs) are excluded.
    """
    texts = dict(docs)
    source = shingle_set(texts[source_id], q)
    rows = []
    if not source:
        return rows
    for doc_id, text in docs:
        if doc_id == source_id:
            continue
        target = shingle_set(text, q)
        if not target:
            continue
        inter = len(source & target)
        if inter == 0:
            continue
        union = len(source | target)
        sim = Fraction(inter, union)
        if sim >= threshold:
            rows.append({
                "source_doc_id": source_id,
                "target_doc_id": doc_id,
                "intersection_count": inter,
                "union_count": union,
                "similarity": f"{sim.numerator}/{sim.denominator}",
                "similarity_num": sim.numerator,
                "similarity_den": sim.denominator,
                "source_shingle_count": len(source),
                "target_shingle_count": len(target),
            })
    rows.sort(key=lambda r: (-Fraction(r["similarity_num"],
                                         r["similarity_den"]),
                             r["target_doc_id"]))
    return rows


def expected_explain(docs: List[Tuple[str, str]], q: int, source_id: str,
                     target_id: str) -> List[dict]:
    """Expected shared shingles in code point order, with per-doc counts."""
    texts = dict(docs)
    s_counts = shingle_multiset(texts[source_id], q)
    t_counts = shingle_multiset(texts[target_id], q)
    out = []
    for gram in sorted(set(s_counts) & set(t_counts)):
        out.append({
            "qgram": gram,
            "qgram_utf8_hex": gram.encode("utf-8").hex(),
            "count_in_source": s_counts[gram],
            "count_in_target": t_counts[gram],
        })
    return out


def expected_posting_records(docs: List[Tuple[str, str]], q: int,
                             source_id: str) -> int:
    """Total posting entries a query for source_id must stream: the sum of
    document frequencies over the source document's distinct shingles."""
    texts = dict(docs)
    source = shingle_set(texts[source_id], q)
    if not source:
        return 0
    doc_sets = [(doc_id, shingle_set(text, q)) for doc_id, text in docs]
    total = 0
    for gram in source:
        for _doc_id, s in doc_sets:
            if gram in s:
                total += 1
    return total

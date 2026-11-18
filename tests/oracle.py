"""Independent brute-force oracle, deliberately disjoint from the FM code.

The expected result set is produced by calling ``str.find`` directly on the
*original* strings, restarting one codepoint after each hit so overlapping
occurrences are counted.  No BWT, rank or suffix-array machinery is touched
here -- this module imports none of the index implementation.  Both the
count multiset and the two coordinate kinds (codepoint offset and UTF-8
byte offset) are derived independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ExpectedHit:
    doc_id: str
    codepoint_offset: int
    utf8_byte_offset: int


def brute_count(docs: List[Tuple[str, str]], query: str) -> int:
    """Overlapping count via repeated str.find(start = previous + 1)."""
    total = 0
    for _doc_id, text in docs:
        start = 0
        while True:
            pos = text.find(query, start)
            if pos < 0:
                break
            total += 1
            start = pos + 1  # overlap allowed
    return total


def brute_locate(docs: List[Tuple[str, str]], query: str
                 ) -> List[ExpectedHit]:
    """All hits with both coordinate kinds computed from raw strings."""
    hits: List[ExpectedHit] = []
    for doc_id, text in docs:
        start = 0
        while True:
            pos = text.find(query, start)
            if pos < 0:
                break
            # Byte offset: length of the UTF-8 prefix strictly before pos.
            byte_off = len(text[:pos].encode("utf-8"))
            hits.append(ExpectedHit(doc_id, pos, byte_off))
            start = pos + 1
    return hits


def brute_multiset(docs: List[Tuple[str, str]], query: str
                   ) -> Dict[Tuple[str, int, int], int]:
    out: Dict[Tuple[str, int, int], int] = {}
    for h in brute_locate(docs, query):
        key = (h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
        out[key] = out.get(key, 0) + 1
    return out

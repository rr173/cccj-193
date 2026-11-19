"""Independent per-codepoint provenance oracle for release testing.

This oracle deliberately imports nothing from the release implementation.
It rebuilds, for one document, the full list of released characters together
with a per-character provenance tag, by applying each round's edits directly
to Python strings.  Bidirectional intervals, UTF-8 byte boundaries and lost
fragments are then derived from those lists -- never from the production
lineage format or interval composition code.

Provenance of a released codepoint is one of:

* ``("orig", baseline_index)``       -- a baseline original character;
* ``("gen", round_id, edit_id, rel)`` -- a generated character, where ``rel``
  is its offset inside the edit's replacement string at introduction time.

Later edits that truncate a generated region preserve the introducing
round/edit and the *original* relative offset, exactly as the specification
requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Tag = Tuple  # ("orig", i) or ("gen", round_id, edit_id, rel)


@dataclass
class OracleDoc:
    text: str
    tags: List[Tag]


def _utf8_prefix_byte_length(text: str, cp_offset: int) -> int:
    return len(text[:cp_offset].encode("utf-8"))


def apply_rounds(baseline: str, rounds: List[dict]) -> OracleDoc:
    """Apply rounds sequentially.

    Each round is ``{"round_id", "edits": [{"edit_id","start","end",
    "replacement"}]}`` with coordinates relative to that round's input.
    """
    chars = list(baseline)
    tags: List[Tag] = [("orig", i) for i in range(len(baseline))]
    for round_spec in rounds:
        rid = round_spec["round_id"]
        edits = sorted(round_spec["edits"], key=lambda e: e["start"])
        new_chars: List[str] = []
        new_tags: List[Tag] = []
        cursor = 0
        for e in edits:
            lo, hi = e["start"], e["end"]
            # Surviving head keeps characters and their provenance.
            new_chars.extend(chars[cursor:lo])
            new_tags.extend(tags[cursor:lo])
            # Replacement gets fresh generated tags with relative offsets.
            for rel, ch in enumerate(e["replacement"]):
                new_chars.append(ch)
                new_tags.append(("gen", rid, e["edit_id"], rel))
            cursor = hi
        new_chars.extend(chars[cursor:])
        new_tags.extend(tags[cursor:])
        chars, tags = new_chars, new_tags
    return OracleDoc(text="".join(chars), tags=tags)


def traceback(doc: OracleDoc, start: int, end: int) -> List[dict]:
    """Release -> provenance, derived purely from the per-codepoint tags."""
    baseline = _baseline_text(doc)
    pieces: List[dict] = []
    i = start
    while i < end:
        tag = doc.tags[i]
        j = i + 1
        # Extend while the next character carries the identical tag.
        while j < end and doc.tags[j] == tag:
            j += 1
        rel_bounds = {
            "codepoints": [i, j],
            "utf8_bytes": [_utf8_prefix_byte_length(doc.text, i),
                           _utf8_prefix_byte_length(doc.text, j)],
        }
        if tag[0] == "orig":
            base_indices = [doc.tags[x][1] for x in range(i, j)]
            pieces.append({
                "source": "baseline",
                "release": rel_bounds,
                "baseline": {
                    "codepoints": [min(base_indices),
                                   max(base_indices) + 1],
                    "utf8_bytes": [
                        _utf8_prefix_byte_length(
                            baseline, min(base_indices)),
                        _utf8_prefix_byte_length(
                            baseline, max(base_indices) + 1)],
                }})
        else:
            _, rid, eid, _rel0 = tag
            rels = [doc.tags[x][3] for x in range(i, j)]
            pieces.append({
                "source": "generated",
                "release": rel_bounds,
                "generation": {
                    "round_id": rid, "edit_id": eid,
                    "relative": {"codepoints": [min(rels),
                                                 max(rels) + 1]}},
            })
        i = j
    return pieces


def provenance_at(doc: OracleDoc, start: int, end: int) -> List[Tuple]:
    """Per-codepoint provenance tuple over a release interval.

    Segmentation-independent: returns one canonical tuple per released
    codepoint, which is what differential tests compare against.
    """
    out = []
    for i in range(start, end):
        tag = doc.tags[i]
        if tag[0] == "orig":
            out.append(("orig", tag[1]))
        else:
            out.append(("gen", tag[1], tag[2], tag[3]))
    return out


def _baseline_text(doc: OracleDoc) -> str:
    # Reconstruct baseline ordering from the surviving original tags.
    pairs = sorted(
        (tag[1], doc.text[i])
        for i, tag in enumerate(doc.tags) if tag[0] == "orig")
    return "".join(ch for _, ch in pairs)


def forward_map(baseline: str, doc: OracleDoc, start: int,
                end: int) -> dict:
    """Baseline -> release mapping plus lost baseline sub-intervals."""
    # baseline index -> release index, one entry per surviving character.
    base_to_release: Dict[int, int] = {}
    for rel_idx, tag in enumerate(doc.tags):
        if tag[0] == "orig":
            base_to_release[tag[1]] = rel_idx
    mapped: List[dict] = []
    lost: List[dict] = []
    cursor = start
    i = start
    while i < end:
        if i in base_to_release:
            j = i
            while (j + 1 < end and (j + 1) in base_to_release
                   and base_to_release[j + 1] == base_to_release[j] + 1):
                j += 1
            r_lo = base_to_release[i]
            r_hi = base_to_release[j] + 1
            mapped.append({
                "baseline": {
                    "codepoints": [i, j + 1],
                    "utf8_bytes": [
                        _utf8_prefix_byte_length(baseline, i),
                        _utf8_prefix_byte_length(baseline, j + 1)]},
                "release": {
                    "codepoints": [r_lo, r_hi],
                    "utf8_bytes": [
                        _utf8_prefix_byte_length(doc.text, r_lo),
                        _utf8_prefix_byte_length(doc.text, r_hi)]},
            })
            cursor = j + 1
            i = j + 1
        else:
            i += 1
    # Lost fragments: baseline runs with no released counterpart.
    k = start
    while k < end:
        if k not in base_to_release:
            j = k
            while j + 1 < end and (j + 1) not in base_to_release:
                j += 1
            lost.append({
                "baseline": {
                    "codepoints": [k, j + 1],
                    "utf8_bytes": [
                        _utf8_prefix_byte_length(baseline, k),
                        _utf8_prefix_byte_length(baseline, j + 1)]},
            })
            k = j + 1
        else:
            k += 1
    return {"mapped": mapped, "lost": lost}

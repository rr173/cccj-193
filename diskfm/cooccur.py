"""Windowed multi-phrase co-occurrence over a sealed diskfm index.

For a set of phrases (each with a unique ``pattern_id``) and a set of
*pairs* of pattern ids, report, per document and per pair, how many
hit-position pairs fall within a non-negative code-point window -- and
list those positions with both code-point and UTF-8 byte offsets.  This is
a phrase-relationship analysis, not a re-grouping of near-duplicate
documents (see :mod:`diskfm.simquery`) and not a thin loop that calls
``locate`` per phrase and merges lists in memory.

Semantics
---------
Offsets are code-point offsets within a single document.  A phrase is a
non-empty literal Unicode string matched by its exact stored code-point
sequence: no tokenisation, no normalisation, no case folding; occurrences
may overlap.  For a requested pair ``(A, B)`` of *distinct* pattern ids,
one hit of A at ``a`` and one hit of B at ``b`` in the same document form
a position pair iff ``abs(a - b) <= window``.  Every distinct combination
counts once.  Equal text under two different ids is two different analysis
items.  A pattern is never paired with itself and pairs never cross a
document boundary.  Documents with zero co-occurrences are not emitted.

Why suffix order is not joinable, and what this module does
------------------------------------------------------------
``locate`` visits matching rows in *suffix* order, which interleaves
documents according to suffix comparisons.  It is neither ``doc_id`` order
nor position order inside a document, so two hit streams cannot be joined
directly.  The engine therefore:

1. streams every hit of each phrase once via
   :meth:`DiskFMIndex.iter_raw_hits` (a single walk of the backward-search
   interval, no cursor, no page seam) and pushes the complete records
   ``(doc_index, codepoint_offset)`` through a bounded external sort;
2. writes each phrase's sorted stream to a fixed-width ``hits.bin`` plus a
   per-document group table.  The sort key is the complete position record,
   so distinct hits never compare equal and no hit is dropped or repeated;
3. k-way merges the group tables in document order and, for each document
   and requested pair, runs a two-pointer sweep over the two sorted slices.
   The sweep emits each combination with ``abs(a-b) <= W`` exactly once,
   accumulates the exact count in O(1) memory, and streams the pairs to a
   second bounded external sort (documents are never joined in RAM).

Closed interval arithmetic (``[a-W, a+W]``) makes the window boundary
exact: ``W`` equal to the difference is included, ``W == 0`` accepts only
equal offsets, and adjacent/overlapping hits are handled without special
cases.  Since each document's join runs over complete sorted slices, pairs
can be neither missed nor duplicated at a spill-run seam.

Bounded resources
-----------------
Every sort buffer holds at most ``chunk_items`` records and ``chunk_bytes``
serialised bytes, spilling a sorted run when either limit is hit; merges
hold one head record per run with bounded fan-in.  Materialised hit files
are fixed-width and read through a hard LRU page cap.  Result pages use a
paged offset index, so RAM does not grow with the output.  All hits remain
reachable: no truncation or approximation of hot documents.

Cursors bind the index identity, a digest of the complete normalised
request (pattern ids in order, phrase texts, pair order, window) and the
logical analysis digest.  A cursor from another index, or after changing
any analysis condition, is rejected.  Analyses are published atomically
into a content-addressed cache directory so paging can resume in another
process with the same index and request; output is deterministic.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import heapq
import json
import os
import shutil
import struct
import tempfile
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import format as fmt
from .format import CursorError, IndexCorruptError
from .pager import FilePager
from .vsort import VarCodec, VarExternalSorter

ANALYSIS_FORMAT_VERSION = 1

# Request limits: a cursor binds the whole request, so it must be finite.
MAX_PATTERNS = 4096
MAX_PAIRS = 1 << 20
MAX_PHRASE_BYTES = 1 << 20
MAX_WINDOW = 2 ** 32 - 1

DEFAULT_CHUNK_ITEMS = 20000
DEFAULT_CHUNK_BYTES = 1 << 22
DEFAULT_MAX_PAGES = 16
DEFAULT_PAGE_RECORDS = 4096

_CONTAINER_VERSION = 1
_CONTAINER_HEADER = struct.Struct("<4sIQ")   # magic, version, record count
_CONTAINER_HEADER_SIZE = _CONTAINER_HEADER.size
_FRAME_LEN = struct.Struct("<I")
_U64 = struct.Struct("<Q")

_HITS_MAGIC = b"CCHT"
_GROUPS_MAGIC = b"CCGR"
_PAIRS_MAGIC = b"CCPR"
_SUMMARY_MAGIC = b"CCSM"

# Materialised hit: packed key (doc_index<<32 | codepoint_offset), byte off.
_HIT_RECORD = struct.Struct("<QQ")
_HIT_RECORD_SIZE = _HIT_RECORD.size
# One phrase's hits inside one document: doc_index, first record, count.
_GROUP_RECORD = struct.Struct("<IQQ")
_GROUP_RECORD_SIZE = _GROUP_RECORD.size
_CP_MASK = 0xFFFFFFFF

_MANIFEST_NAME = "manifest.json"
_OPEN_MARKER = "OPEN"
_STAGING_SUFFIX = ".staging"


class CooccurError(ValueError):
    """A structurally invalid co-occurrence request (not index corruption)."""


# --------------------------------------------------------------------- #
# Request model
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    phrase: str


def _check_pattern_id(pid) -> str:
    if not isinstance(pid, str):
        raise CooccurError("pattern_id must be a string")
    if pid == "":
        raise CooccurError("pattern_id must be non-empty")
    if len(pid.encode("utf-8")) > MAX_PHRASE_BYTES:
        raise CooccurError("pattern_id too long")
    return pid


def _check_phrase(phrase) -> str:
    if not isinstance(phrase, str):
        raise CooccurError("phrase must be a string")
    if phrase == "":
        raise CooccurError("phrase must be non-empty")
    for ch in phrase:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise CooccurError(
                "phrase contains an isolated surrogate code point "
                f"U+{cp:04X}")
    if len(phrase.encode("utf-8")) > MAX_PHRASE_BYTES:
        raise CooccurError("phrase too long")
    return phrase


@dataclass(frozen=True)
class CooccurRequest:
    patterns: Tuple[Pattern, ...]
    pairs: Tuple[Tuple[str, str], ...]  # request order, both directions kept
    window: int

    def normalised(self) -> dict:
        return {
            "patterns": [{"pattern_id": p.pattern_id, "phrase": p.phrase}
                         for p in self.patterns],
            "pairs": [list(p) for p in self.pairs],
            "window": self.window,
        }

    def digest(self) -> str:
        payload = json.dumps(self.normalised(), sort_keys=True,
                             ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def pair_slots(self) -> Tuple[Tuple[int, int], ...]:
        index = {p.pattern_id: i for i, p in enumerate(self.patterns)}
        return tuple((index[a], index[b]) for a, b in self.pairs)


def build_request(patterns: Sequence, pairs: Sequence,
                  window: int) -> CooccurRequest:
    """Validate a raw request and return its canonical form."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise CooccurError("window must be a non-negative integer")
    if window < 0:
        raise CooccurError("window must be non-negative")
    if window > MAX_WINDOW:
        raise CooccurError("window is out of range")
    if patterns is None or len(patterns) == 0:
        raise CooccurError("at least one pattern is required")
    if len(patterns) > MAX_PATTERNS:
        raise CooccurError("too many patterns")
    if pairs is None or len(pairs) == 0:
        raise CooccurError("at least one pattern pair is required")
    if len(pairs) > MAX_PAIRS:
        raise CooccurError("too many pattern pairs")

    seen: Dict[str, Pattern] = {}
    ordered: List[Pattern] = []
    for entry in patterns:
        if isinstance(entry, Pattern):
            pid, phrase = entry.pattern_id, entry.phrase
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            pid, phrase = entry
        elif isinstance(entry, dict):
            if set(entry) != {"pattern_id", "phrase"}:
                raise CooccurError(
                    "each pattern object needs exactly pattern_id and phrase")
            pid, phrase = entry["pattern_id"], entry["phrase"]
        else:
            raise CooccurError(
                "pattern must be (pattern_id, phrase) or an object")
        _check_pattern_id(pid)
        _check_phrase(phrase)
        if pid in seen:
            raise CooccurError(f"duplicate pattern_id: {pid!r}")
        pat = Pattern(pid, phrase)
        seen[pid] = pat
        ordered.append(pat)

    norm_pairs: List[Tuple[str, str]] = []
    seen_pairs = set()
    for entry in pairs:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise CooccurError("pair must be (pattern_id, pattern_id)")
        a, b = entry
        if not isinstance(a, str) or not isinstance(b, str):
            raise CooccurError("pair members must be pattern_id strings")
        if a not in seen:
            raise CooccurError(f"pair references unknown pattern_id: {a!r}")
        if b not in seen:
            raise CooccurError(f"pair references unknown pattern_id: {b!r}")
        if a == b:
            raise CooccurError(
                f"pattern {a!r} must not be paired with itself")
        if (a, b) in seen_pairs:
            raise CooccurError(f"duplicate pattern pair: {(a, b)!r}")
        seen_pairs.add((a, b))
        norm_pairs.append((a, b))

    return CooccurRequest(tuple(ordered), tuple(norm_pairs), window)


def request_from_dict(obj) -> CooccurRequest:
    if not isinstance(obj, dict):
        raise CooccurError("request must be a JSON object")
    try:
        patterns = obj["patterns"]
        pairs = obj["pairs"]
        window = obj["window"]
    except KeyError as exc:
        raise CooccurError(f"request missing field: {exc}") from exc
    if not isinstance(patterns, list) or not isinstance(pairs, list):
        raise CooccurError("patterns and pairs must be arrays")
    return build_request(patterns, pairs, window)


# --------------------------------------------------------------------- #
# External-sort codecs
# --------------------------------------------------------------------- #
class HitCodec(VarCodec):
    """Sort record ``(doc_index, codepoint_offset)``: the joinable stream."""

    _STRUCT = struct.Struct("<QI")

    def encode(self, record) -> bytes:
        return self._STRUCT.pack(record[0], record[1])

    def decode(self, payload: bytes):
        if len(payload) != self._STRUCT.size:
            raise IndexCorruptError("hit sort record has wrong length")
        return self._STRUCT.unpack(payload)


class PairCodec(VarCodec):
    """Sort record ``(slot, doc_id_bytes, a_cp, a_byte, b_cp, b_byte)``.

    The complete doc_id participates in the key, so the merged order is
    exactly ``(request pair order, doc_id code point order, a_cp, b_cp)``.
    """

    def encode(self, record) -> bytes:
        slot, doc_id, acp, aby, bcp, bby = record
        return (struct.pack("<II", slot, len(doc_id)) + doc_id
                + struct.pack("<QQQQ", acp, aby, bcp, bby))

    def decode(self, payload: bytes):
        if len(payload) < 8:
            raise IndexCorruptError("pair sort record too short")
        slot, id_len = struct.unpack_from("<II", payload, 0)
        if len(payload) != 8 + id_len + 32:
            raise IndexCorruptError("pair sort record has wrong length")
        off = 8 + id_len
        doc_id = payload[8:off]
        acp, aby, bcp, bby = struct.unpack_from("<QQQQ", payload, off)
        return (slot, doc_id, acp, aby, bcp, bby)


class SummaryCodec(VarCodec):
    """Sort record ``(slot, doc_id_bytes, count)``."""

    def encode(self, record) -> bytes:
        slot, doc_id, count = record
        return (struct.pack("<II", slot, len(doc_id)) + doc_id
                + struct.pack("<Q", count))

    def decode(self, payload: bytes):
        if len(payload) < 8:
            raise IndexCorruptError("summary record too short")
        slot, id_len = struct.unpack_from("<II", payload, 0)
        if len(payload) != 8 + id_len + 8:
            raise IndexCorruptError("summary record has wrong length")
        off = 8 + id_len
        doc_id = payload[8:off]
        (count,) = struct.unpack_from("<Q", payload, off)
        return (slot, doc_id, count)


# --------------------------------------------------------------------- #
# Fixed-width files
# --------------------------------------------------------------------- #
class _FixedWriter:
    def __init__(self, path: str, magic: bytes, record: struct.Struct):
        self.path = path
        self.magic = magic
        self.record = record
        self.count = 0
        self.bytes_written = _CONTAINER_HEADER_SIZE
        self._f = open(path + ".tmp", "wb")
        self._f.write(_CONTAINER_HEADER.pack(magic, _CONTAINER_VERSION, 0))

    def add(self, *values) -> None:
        self._f.write(self.record.pack(*values))
        self.bytes_written += self.record.size
        self.count += 1

    def finish(self) -> int:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.seek(0)
        self._f.write(_CONTAINER_HEADER.pack(
            self.magic, _CONTAINER_VERSION, self.count))
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        os.replace(self._f.name, self.path)
        return self.count

    def abort(self) -> None:
        if not self._f.closed:
            self._f.close()
        try:
            os.unlink(self._f.name)
        except FileNotFoundError:
            pass


def _open_container(path: str, magic: bytes) -> Tuple[int, "object"]:
    f = open(path, "rb")
    head = f.read(_CONTAINER_HEADER_SIZE)
    if len(head) != _CONTAINER_HEADER_SIZE:
        f.close()
        raise IndexCorruptError("result file truncated in its header")
    file_magic, version, count = _CONTAINER_HEADER.unpack(head)
    if file_magic != magic:
        f.close()
        raise IndexCorruptError("result file magic mismatch")
    if version != _CONTAINER_VERSION:
        f.close()
        raise IndexCorruptError("result file version mismatch")
    return count, f


def _iter_fixed(path: str, magic: bytes, record: struct.Struct
                ) -> Iterator[tuple]:
    count, f = _open_container(path, magic)
    try:
        for _ in range(count):
            raw = f.read(record.size)
            if len(raw) != record.size:
                raise IndexCorruptError("fixed record file is truncated")
            yield record.unpack(raw)
        if f.read(1):
            raise IndexCorruptError("fixed record file has trailing bytes")
    finally:
        f.close()


class _FixedFile:
    """Random paged access to a fixed-width hit or group file."""

    def __init__(self, path: str, record: struct.Struct, count: int,
                 max_pages: int, page_records: int):
        self.path = path
        self.count = count
        self.record = record
        self._pager = FilePager(path, record_size=record.size,
                                header_size=_CONTAINER_HEADER_SIZE,
                                page_records=page_records,
                                max_pages=max_pages)

    def get(self, index: int) -> tuple:
        try:
            return self.record.unpack(self._pager.read_record(index))
        except IndexError as exc:
            raise IndexCorruptError("read past end of fixed record file") \
                from exc

    def stats(self) -> dict:
        return self._pager.stats()

    def close(self) -> None:
        self._pager.close()


def _hit_key(doc_index: int, cp_offset: int) -> int:
    return (doc_index << 32) | cp_offset


class _ContainerWriter:
    """Frame-delimited record stream plus a raw u64 offset index."""

    def __init__(self, path: str, magic: bytes, codec: VarCodec):
        self.path = path
        self.magic = magic
        self.codec = codec
        self.count = 0
        self.bytes_written = _CONTAINER_HEADER_SIZE
        self._f = open(path + ".tmp", "wb")
        self._f.write(_CONTAINER_HEADER.pack(magic, _CONTAINER_VERSION, 0))
        self._off = open(path + ".off.tmp", "wb")

    def add(self, record) -> None:
        payload = self.codec.encode(record)
        offset = self._f.tell()
        self._off.write(_U64.pack(offset))
        self._f.write(_FRAME_LEN.pack(len(payload)))
        self._f.write(payload)
        self.bytes_written += _FRAME_LEN.size + len(payload) + _U64.size
        self.count += 1

    def add_many(self, records: Iterator) -> None:
        for record in records:
            self.add(record)

    def finish(self) -> int:
        for f in (self._off, self._f):
            f.flush()
            os.fsync(f.fileno())
        self._f.seek(0)
        self._f.write(_CONTAINER_HEADER.pack(
            self.magic, _CONTAINER_VERSION, self.count))
        self._f.flush()
        os.fsync(self._f.fileno())
        self._off.close()
        self._f.close()
        os.replace(self._f.name, self.path)
        os.replace(self._off.name, self.path + ".off")
        return self.count

    def abort(self) -> None:
        for f in (self._off, self._f):
            if not f.closed:
                f.close()
        for path in (self._f.name, self._off.name):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class _Container:
    """Read side: full sequential scan or O(1) random paging."""

    def __init__(self, path: str, magic: bytes, codec: VarCodec,
                 max_pages: int, page_records: int):
        self.path = path
        self.codec = codec
        self.count, self._f = _open_container(path, magic)
        self._f.seek(0, os.SEEK_END)
        self.size = self._f.tell()
        off_path = path + ".off"
        try:
            off_size = os.path.getsize(off_path)
        except OSError as exc:
            self._f.close()
            raise IndexCorruptError("result offset index missing") from exc
        if off_size != self.count * _U64.size:
            self._f.close()
            raise IndexCorruptError("result offset index has wrong size")
        self._off = FilePager(off_path, record_size=_U64.size,
                              header_size=0, page_records=page_records,
                              max_pages=max_pages)

    def _record_at(self, index: int):
        if not 0 <= index < self.count:
            raise IndexCorruptError("result record index out of range")
        try:
            offset = self._off.get_u64(index)
        except IndexError as exc:
            raise IndexCorruptError("result offset index truncated") from exc
        if offset < _CONTAINER_HEADER_SIZE:
            raise IndexCorruptError("result offset out of range")
        self._f.seek(offset)
        raw = self._f.read(_FRAME_LEN.size)
        if len(raw) != _FRAME_LEN.size:
            raise IndexCorruptError("result frame truncated")
        (length,) = _FRAME_LEN.unpack(raw)
        start = offset + _FRAME_LEN.size
        if start + length > self.size:
            raise IndexCorruptError("result frame runs past end of file")
        payload = self._f.read(length)
        if len(payload) != length:
            raise IndexCorruptError("result frame truncated")
        if index + 1 < self.count:
            next_offset = self._off.get_u64(index + 1)
            if next_offset != start + length:
                raise IndexCorruptError("result offset index inconsistent")
        elif start + length != self.size:
            raise IndexCorruptError("result file has trailing bytes")
        return self.codec.decode(payload)

    def record(self, index: int):
        return self._record_at(index)

    def records(self, start: int = 0, limit: Optional[int] = None
                ) -> Iterator:
        """Page-sized random reads; bounded memory, no re-scan."""
        if start >= self.count:
            return
        stop = self.count if limit is None else min(start + limit, self.count)
        for i in range(start, stop):
            yield self._record_at(i)

    def scan(self) -> Iterator:
        """Single sequential pass, validating the frame chain."""
        self._f.seek(_CONTAINER_HEADER_SIZE)
        for _ in range(self.count):
            raw = self._f.read(_FRAME_LEN.size)
            if len(raw) != _FRAME_LEN.size:
                raise IndexCorruptError("result frame truncated")
            (length,) = _FRAME_LEN.unpack(raw)
            payload = self._f.read(length)
            if len(payload) != length:
                raise IndexCorruptError("result frame truncated")
            yield self.codec.decode(payload)
        if self._f.read(1):
            raise IndexCorruptError("result file has trailing bytes")

    def stats(self) -> dict:
        return self._off.stats()

    def close(self) -> None:
        self._off.close()
        self._f.close()


# --------------------------------------------------------------------- #
# Analysis construction
# --------------------------------------------------------------------- #
class _AnalysisBuilder:
    def __init__(self, index, request: CooccurRequest, workdir: str,
                 chunk_items: int, chunk_bytes: int):
        self.index = index
        self.request = request
        self.workdir = workdir
        self.chunk_items = chunk_items
        self.chunk_bytes = chunk_bytes
        self.pair_slots = request.pair_slots()
        self.stats: Dict[str, object] = {
            "hits_read": 0,
            "hit_sort_runs": 0,
            "hit_sort_bytes": 0,
            "hit_sort_merge_passes": 0,
            "candidate_pairs": 0,
            "position_pairs": 0,
        }

    def _sorter(self, codec: VarCodec) -> VarExternalSorter:
        return VarExternalSorter(
            self.workdir, codec, chunk_items=self.chunk_items,
            chunk_bytes=self.chunk_bytes)

    # -- per phrase: stream hits -> external sort -> fixed file ----------
    def _materialise_hits(self) -> List[Dict[str, object]]:
        phrases: List[Dict[str, object]] = []
        for slot, pattern in enumerate(self.request.patterns):
            sorter = self._sorter(HitCodec())
            hits_here = 0
            try:
                for doc_index, cp_offset in self.index.iter_raw_hits(
                        pattern.phrase):
                    sorter.add((doc_index, cp_offset))
                    hits_here += 1
                result = sorter.finalize()
                hits_path = os.path.join(self.workdir, f"hits_{slot}.bin")
                groups_path = hits_path + ".groups"
                hit_writer = _FixedWriter(hits_path, _HITS_MAGIC, _HIT_RECORD)
                group_writer = _FixedWriter(groups_path, _GROUPS_MAGIC,
                                            _GROUP_RECORD)
                count = 0
                groups = 0
                try:
                    count, groups = self._emit_sorted_hits(
                        result, hit_writer, group_writer)
                    hit_writer.finish()
                    group_writer.finish()
                except BaseException:
                    hit_writer.abort()
                    group_writer.abort()
                    raise
                result.cleanup()
            except BaseException:
                sorter.cleanup_discarded()
                raise
            if count != hits_here:
                raise IndexCorruptError("hit materialisation lost records")
            self.stats["hits_read"] += hits_here
            self.stats["hit_sort_runs"] += sorter.chunks_spilled
            self.stats["hit_sort_bytes"] += sorter.bytes_spilled
            self.stats["hit_sort_merge_passes"] += sorter.merge_passes
            phrases.append({
                "slot": slot,
                "pattern_id": pattern.pattern_id,
                "hits": count,
                "groups": groups,
                "hits_path": hits_path,
                "groups_path": groups_path,
            })
        self.stats["hits_per_phrase"] = [p["hits"] for p in phrases]
        return phrases

    def _emit_sorted_hits(self, sorted_hits, hit_writer, group_writer
                          ) -> Tuple[int, int]:
        """Write one phrase's hits, converting offsets with one byte pass
        per document.

        The iteration is pull-driven with a single look-ahead record, so a
        document with millions of overlapping hits costs no more working
        memory than one with a single hit.
        """
        count = 0
        groups = 0
        it = iter(sorted_hits)
        pending = next(it, None)
        while pending is not None:
            doc_index = pending[0]
            group_start = count
            cursor = self.index.document_byte_cursor(doc_index)
            while pending is not None and pending[0] == doc_index:
                cp_offset = pending[1]
                byte_off = cursor.byte_offset(cp_offset)
                hit_writer.add(_hit_key(doc_index, cp_offset), byte_off)
                count += 1
                pending = next(it, None)
            if pending is not None and pending[0] < doc_index:
                raise IndexCorruptError(
                    "hit stream is not sorted by document")
            group_writer.add(doc_index, group_start, count - group_start)
            groups += 1
        return count, groups

    # -- merge per-phrase group tables in document order -----------------
    def _merge_groups(self, phrases: List[Dict[str, object]]):
        iters = [_iter_fixed(p["groups_path"], _GROUPS_MAGIC, _GROUP_RECORD)
                 for p in phrases]
        heap: List[Tuple[int, int, tuple]] = []
        for slot, it in enumerate(iters):
            first = next(it, None)
            if first is not None:
                heapq.heappush(heap, (first[0], slot, first))
        while heap:
            doc = heap[0][0]
            entries: List[Tuple[int, int, int]] = []
            while heap and heap[0][0] == doc:
                _doc, slot, (_d, start, count) = heapq.heappop(heap)
                entries.append((slot, start, count))
                nxt = next(iters[slot], None)
                if nxt is not None:
                    heapq.heappush(heap, (nxt[0], slot, nxt))
            entries.sort()
            yield doc, entries

    # -- exact window join for one pair ----------------------------------
    @staticmethod
    def _join_slices(a_file: _FixedFile, a_start: int, a_end: int,
                     b_file: _FixedFile, b_start: int, b_end: int,
                     window: int, emit: Callable[[int, int, int, int], None]
                     ) -> int:
        """Emit every position pair with ``abs(a_cp - b_cp) <= window``.

        Both slices are sorted by code-point offset.  For each A hit we
        advance two monotone pointers into the B slice so exactly the
        closed window ``[a_cp - window, a_cp + window]`` is visited; each
        (A, B) combination is therefore produced once and only once.
        """
        j_lo = b_start
        j_hi = b_start
        produced = 0
        for i in range(a_start, a_end):
            a_key, a_byte = a_file.get(i)
            a_cp = a_key & _CP_MASK
            low = a_cp - window
            high = a_cp + window
            while j_lo < b_end and (b_file.get(j_lo)[0] & _CP_MASK) < low:
                j_lo += 1
            if j_hi < j_lo:
                j_hi = j_lo
            while j_hi < b_end and (b_file.get(j_hi)[0] & _CP_MASK) <= high:
                j_hi += 1
            for j in range(j_lo, j_hi):
                b_key, b_byte = b_file.get(j)
                emit(a_cp, a_byte, b_key & _CP_MASK, b_byte)
            produced += j_hi - j_lo
        return produced

    def run(self):
        phrases = self._materialise_hits()
        hit_files = [
            _FixedFile(p["hits_path"], _HIT_RECORD, p["hits"],
                       DEFAULT_MAX_PAGES, DEFAULT_PAGE_RECORDS)
            for p in phrases]

        pair_sorter = self._sorter(PairCodec())
        summary_sorter = self._sorter(SummaryCodec())
        candidate_pairs = 0
        documents_with_hits = 0
        groups_seen = 0
        try:
            for doc_index, entries in self._merge_groups(phrases):
                documents_with_hits += 1
                groups_seen += len(entries)
                by_slot = {slot: (start, start + count)
                           for slot, start, count in entries}
                doc_id = self.index._read_doc_id(doc_index).encode("utf-8")
                for pair_index, (sa, sb) in enumerate(self.pair_slots):
                    if sa not in by_slot or sb not in by_slot:
                        continue
                    a_start, a_end = by_slot[sa]
                    b_start, b_end = by_slot[sb]
                    holder = {"slot": pair_index, "doc": doc_id}

                    def emit(acp, aby, bcp, bby, holder=holder):
                        pair_sorter.add((holder["slot"], holder["doc"],
                                         acp, aby, bcp, bby))

                    produced = self._join_slices(
                        hit_files[sa], a_start, a_end,
                        hit_files[sb], b_start, b_end,
                        self.request.window, emit)
                    if produced:
                        candidate_pairs += produced
                        summary_sorter.add((pair_index, doc_id, produced))
            pair_result = pair_sorter.finalize()
            summary_result = summary_sorter.finalize()
        except BaseException:
            pair_sorter.cleanup_discarded()
            summary_sorter.cleanup_discarded()
            raise

        self.stats["pair_sort_runs"] = pair_sorter.chunks_spilled
        self.stats["pair_sort_bytes"] = pair_sorter.bytes_spilled
        self.stats["pair_sort_merge_passes"] = pair_sorter.merge_passes
        self.stats["summary_sort_runs"] = summary_sorter.chunks_spilled
        self.stats["summary_sort_bytes"] = summary_sorter.bytes_spilled
        self.stats["summary_sort_merge_passes"] = summary_sorter.merge_passes
        self.stats["documents_with_hits"] = documents_with_hits
        self.stats["phrase_document_groups"] = groups_seen
        self.stats["candidate_pairs"] = candidate_pairs

        # Publish ordered streams.  The pair stream carries a running hash
        # and per-slot counts, so the analysis digest is a real function of
        # the emitted content.
        summary_path = os.path.join(self.workdir, "summary.bin")
        pairs_path = os.path.join(self.workdir, "pairs.bin")
        summary_writer = _ContainerWriter(summary_path, _SUMMARY_MAGIC,
                                          SummaryCodec())
        try:
            summary_writer.add_many(summary_result)
            summary_rows = summary_writer.finish()
        except BaseException:
            summary_writer.abort()
            raise
        finally:
            summary_result.cleanup()

        n_slots = len(self.request.pairs)
        pair_counts = [0] * n_slots
        summary_totals = [0] * n_slots
        pair_writer = _ContainerWriter(pairs_path, _PAIRS_MAGIC,
                                       PairCodec())
        content_hash = hashlib.sha256()
        try:
            current_slot = -1
            for slot, doc_id, acp, aby, bcp, bby in pair_result:
                if slot < current_slot:
                    raise IndexCorruptError("pair result not ordered")
                current_slot = slot
                pair_writer.add((slot, doc_id, acp, aby, bcp, bby))
                content_hash.update(
                    struct.pack("<I", slot) + struct.pack("<I", len(doc_id))
                    + doc_id + struct.pack("<QQQQ", acp, aby, bcp, bby))
                pair_counts[slot] += 1
            position_pairs = pair_writer.finish()
        except BaseException:
            pair_writer.abort()
            raise
        finally:
            pair_result.cleanup()

        if position_pairs != candidate_pairs:
            raise IndexCorruptError(
                "enumerated position pairs disagree with the exact count")

        for slot, _doc_id, count in _iter_fixed_summary(summary_path):
            summary_totals[slot] += count
        if summary_totals != pair_counts:
            raise IndexCorruptError(
                "summary totals disagree with enumerated position pairs")

        for f in hit_files:
            f.close()

        self.stats["position_pairs"] = position_pairs
        self.stats["summary_rows"] = summary_rows
        self.stats["pair_counts"] = pair_counts
        self.stats["content_sha256"] = content_hash.hexdigest()
        self.stats["hit_file_bytes"] = sum(
            os.path.getsize(p["hits_path"]) + os.path.getsize(p["groups_path"])
            for p in phrases)
        self.stats["result_file_bytes"] = (
            os.path.getsize(summary_path) + os.path.getsize(summary_path + ".off")
            + os.path.getsize(pairs_path) + os.path.getsize(pairs_path + ".off"))
        # Measured temporary disk footprint: every materialised hit file,
        # every spilled sort run, and the published result containers.
        self.stats["temp_disk_bytes"] = (
            self.stats["hit_file_bytes"] + self.stats["result_file_bytes"]
            + self.stats["hit_sort_bytes"] + self.stats["pair_sort_bytes"]
            + self.stats["summary_sort_bytes"])
        return summary_rows, position_pairs, pair_counts


def _iter_fixed_summary(path: str) -> Iterator[Tuple[int, bytes, int]]:
    count, f = _open_container(path, _SUMMARY_MAGIC)
    try:
        for _ in range(count):
            raw = f.read(_FRAME_LEN.size)
            if len(raw) != _FRAME_LEN.size:
                raise IndexCorruptError("summary frame truncated")
            (length,) = _FRAME_LEN.unpack(raw)
            payload = f.read(length)
            if len(payload) != length:
                raise IndexCorruptError("summary frame truncated")
            yield SummaryCodec().decode(payload)
        if f.read(1):
            raise IndexCorruptError("summary file has trailing bytes")
    finally:
        f.close()


# --------------------------------------------------------------------- #
# Analysis publication (content-addressed, atomically published)
# --------------------------------------------------------------------- #
class CooccurAnalysis:
    """A computed or reopened co-occurrence analysis.

    Files live in ``directory`` and are immutable once published.  Paging
    is a random read through a bounded LRU over the result's offset index;
    nothing proportional to the result size is retained.  Cursors are
    stateless tokens; the same cursor resumes in another process that opens
    the same index identity and the same request (the analysis is
    deterministically recomputed or reopened from its cache).
    """

    def __init__(self, index, request: CooccurRequest, directory: str,
                 manifest: dict, max_pages: int = DEFAULT_MAX_PAGES,
                 page_records: int = DEFAULT_PAGE_RECORDS):
        self.index = index
        self.request = request
        self.directory = directory
        self.manifest = manifest
        self.identity = index.identity
        self.request_digest = manifest["request_sha256"]
        self.analysis_digest = manifest["analysis_sha256"]
        self.summary_rows = manifest["summary_rows"]
        self.position_pairs = manifest["position_pairs"]
        self.pair_counts = list(manifest["pair_counts"])
        self.pair_slots = request.pair_slots()
        self._max_pages = max_pages
        self._page_records = page_records
        self._summary = _Container(
            os.path.join(directory, "summary.bin"), _SUMMARY_MAGIC,
            SummaryCodec(), max_pages, page_records)
        self._pairs = _Container(
            os.path.join(directory, "pairs.bin"), _PAIRS_MAGIC,
            PairCodec(), max_pages, page_records)
        # slot -> (start, count) in the pair stream.  The pair stream is
        # written in request pair order and slot counts are recorded in
        # the manifest, so the starts are plain prefix sums -- no scan and
        # no hash map decides ordering.
        self._pair_bounds: List[Tuple[int, int]] = []
        running = 0
        for count in self.pair_counts:
            self._pair_bounds.append((running, count))
            running += count
        if running != self.position_pairs:
            raise IndexCorruptError(
                "pair counts disagree with the manifest total")
        self._page_stats = {
            "summary_pages_served": 0,
            "pair_pages_served": 0,
            "summary_records_read": 0,
            "pair_records_read": 0,
        }

    # -- validation / reopening ------------------------------------------
    @staticmethod
    def _load_manifest(directory: str) -> dict:
        marker = os.path.join(directory, _OPEN_MARKER)
        if not os.path.exists(marker):
            raise IndexCorruptError(
                "no OPEN marker: co-occurrence analysis is incomplete")
        try:
            with open(os.path.join(directory, _MANIFEST_NAME),
                      encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexCorruptError(
                "co-occurrence manifest missing or unreadable") from exc
        if manifest.get("analysis_format") != "diskfm-cooccur/1":
            raise IndexCorruptError("unknown co-occurrence analysis format")
        return manifest

    @staticmethod
    def _verify_files(directory: str, manifest: dict) -> None:
        for name, entry in manifest["files"].items():
            fmt.verify_file(os.path.join(directory, name), entry)

    # -- pair addressing --------------------------------------------------
    def _slot_for(self, pair) -> int:
        if isinstance(pair, int):
            if not 0 <= pair < len(self.request.pairs):
                raise CooccurError("pair index out of range")
            return pair
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise CooccurError("pair must be (pattern_id, pattern_id)")
        try:
            return self.request.pairs.index((pair[0], pair[1]))
        except ValueError:
            raise CooccurError("pair was not part of this request") from None

    # -- paging -----------------------------------------------------------
    def summary_page(self, page_size: int, cursor: Optional[str] = None
                     ) -> Tuple[List[dict], Optional[str]]:
        """One page of ``(pair, doc_id, count)`` rows, non-zero only."""
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        if cursor is None:
            start = 0
        else:
            start = self._consume_cursor(cursor, "summary", None)
        rows: List[dict] = []
        for slot, doc_id, count in self._summary.records(start, page_size):
            self._page_stats["summary_records_read"] += 1
            a_id, b_id = self.request.pairs[slot]
            rows.append({
                "pattern_a": a_id,
                "pattern_b": b_id,
                "doc_id": doc_id.decode("utf-8"),
                "cooccurrence_pairs": count,
            })
        self._page_stats["summary_pages_served"] += 1
        next_cursor = None
        if start + len(rows) < self.summary_rows:
            next_cursor = self._make_cursor("summary", None,
                                            start + len(rows))
        return rows, next_cursor

    def iter_summary(self) -> Iterator[dict]:
        for slot, doc_id, count in self._summary.scan():
            a_id, b_id = self.request.pairs[slot]
            yield {
                "pattern_a": a_id,
                "pattern_b": b_id,
                "doc_id": doc_id.decode("utf-8"),
                "cooccurrence_pairs": count,
            }

    def pair_summary(self, pair) -> List[dict]:
        slot = self._slot_for(pair)
        a_id, b_id = self.request.pairs[slot]
        out = []
        for row in self.iter_summary():
            if row["pattern_a"] == a_id and row["pattern_b"] == b_id:
                out.append(row)
        return out

    def pairs_page(self, pair, page_size: int, cursor: Optional[str] = None
                   ) -> Tuple[List[dict], Optional[str]]:
        """One page of position pairs for a requested phrase pair."""
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        slot = self._slot_for(pair)
        if cursor is None:
            local = 0
        else:
            local = self._consume_cursor(cursor, "pairs", slot)
        start_record, total = self._pair_bounds[slot]
        rows: List[dict] = []
        remaining = total - local
        take = min(page_size, remaining)
        a_id, b_id = self.request.pairs[slot]
        for index in range(start_record + local, start_record + local + take):
            _slot, doc_id, acp, aby, bcp, bby = self._pairs.record(index)
            rows.append({
                "pattern_a": a_id,
                "pattern_b": b_id,
                "doc_id": doc_id.decode("utf-8"),
                "a_codepoint_offset": acp,
                "a_utf8_byte_offset": aby,
                "b_codepoint_offset": bcp,
                "b_utf8_byte_offset": bby,
            })
            self._page_stats["pair_records_read"] += 1
        self._page_stats["pair_pages_served"] += 1
        next_cursor = None
        if local + take < total:
            next_cursor = self._make_cursor("pairs", slot, local + take)
        return rows, next_cursor

    def iter_pairs(self, pair) -> Iterator[dict]:
        slot = self._slot_for(pair)
        start_record, total = self._pair_bounds[slot]
        a_id, b_id = self.request.pairs[slot]
        for index in range(start_record, start_record + total):
            _slot, doc_id, acp, aby, bcp, bby = self._pairs.record(index)
            yield {
                "pattern_a": a_id,
                "pattern_b": b_id,
                "doc_id": doc_id.decode("utf-8"),
                "a_codepoint_offset": acp,
                "a_utf8_byte_offset": aby,
                "b_codepoint_offset": bcp,
                "b_utf8_byte_offset": bby,
            }

    def pair_count(self, pair) -> int:
        return self.pair_counts[self._slot_for(pair)]

    # -- cursors ----------------------------------------------------------
    def _binding(self) -> dict:
        return {
            "v": ANALYSIS_FORMAT_VERSION,
            "id": self.identity,
            "req": self.request_digest,
            "ana": self.analysis_digest,
            "win": self.request.window,
        }

    def _make_cursor(self, part: str, slot: Optional[int], offset: int) -> str:
        token = dict(self._binding())
        token.update({"part": part, "slot": slot, "off": offset})
        payload = json.dumps(token, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest().encode("ascii")
        return base64.urlsafe_b64encode(digest + payload).decode("ascii")

    def _consume_cursor(self, cursor: str, part: str,
                        slot: Optional[int]) -> int:
        try:
            if not isinstance(cursor, str):
                raise ValueError("cursor must be a string")
            raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
            digest, payload = raw[:64], raw[64:]
            if hashlib.sha256(payload).hexdigest().encode("ascii") != digest:
                raise ValueError("digest mismatch")
            token = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError,
                binascii.Error) as exc:
            raise CursorError("malformed or tampered co-occurrence cursor") \
                from exc
        expected = self._binding()
        for key, label in (("v", "format"), ("id", "index identity"),
                           ("req", "request"), ("ana", "analysis"),
                           ("win", "window")):
            if token.get(key) != expected[key]:
                raise CursorError(f"cursor bound to a different {label}")
        if token.get("part") != part:
            raise CursorError(
                "cursor was issued for a different result part")
        if token.get("slot") != slot:
            raise CursorError(
                "cursor was issued for a different phrase pair")
        offset = token.get("off")
        if not isinstance(offset, int) or isinstance(offset, bool) or \
                offset < 0:
            raise CursorError("cursor offset invalid")
        limit = self.summary_rows if part == "summary" else \
            self.pair_counts[slot]
        if offset > limit:
            raise CursorError("cursor offset past the end of results")
        return offset

    # -- stats ------------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        analysis = dict(self.manifest.get("stats", {}))
        analysis["summary_rows"] = self.summary_rows
        analysis["position_pairs"] = self.position_pairs
        analysis["pair_counts"] = list(self.pair_counts)
        analysis["temp_disk_bytes"] = analysis.get("temp_disk_bytes", 0)
        paging = dict(self._page_stats)
        paging["summary_pager"] = self._summary.stats()
        paging["pairs_pager"] = self._pairs.stats()
        return {"analysis": analysis, "paging": paging}

    def close(self) -> None:
        self._summary.close()
        self._pairs.close()

    def __enter__(self) -> "CooccurAnalysis":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------- #
# Cache layout: <root>/<index identity>/<request digest>/
#
# The cache lives *outside* the sealed index directory: a read-only query
# must never modify the index it reads.  Callers may point it elsewhere with
# ``DISKFM_COOCCUR_CACHE``; the default is a per-user temp directory keyed
# by index identity, so two distinct indexes never share a result.
# --------------------------------------------------------------------- #
def _default_cache_root() -> str:
    return os.path.join(tempfile.gettempdir(), "diskfm-cooccur-cache")


def _cache_root(index) -> str:
    root = os.environ.get("DISKFM_COOCCUR_CACHE")
    return root if root else _default_cache_root()


def _analysis_dir(index, request: CooccurRequest) -> str:
    return os.path.join(_cache_root(index), index.identity, request.digest())


def _file_manifest(directory: str, names: Sequence[str]) -> Dict[str, dict]:
    return {name: fmt.hash_file(os.path.join(directory, name))
            for name in names}


def _publish_manifest(directory: str, manifest: dict) -> None:
    """Write the manifest and OPEN marker into staging, then rename.

    ``os.replace`` is atomic for a directory rename onto a *nonexistent*
    target.  If a valid target already exists (another process won the
    race) we discard our staging tree, because both trees are byte-identical
    for the same index identity and request.
    """
    staging = directory + _STAGING_SUFFIX
    fmt.write_manifest(staging, manifest)
    marker = os.path.join(staging, _OPEN_MARKER)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
        f.flush()
        os.fsync(f.fileno())
    parent = os.path.dirname(directory)
    os.makedirs(parent, exist_ok=True)
    if not os.path.exists(directory):
        os.replace(staging, directory)
        return
    # A published analysis already exists.  Move it aside first so the
    # rename cannot fail on a non-empty target, then discard the old tree
    # only after the new one is in place.
    backup = directory + ".old." + uuid.uuid4().hex
    os.replace(directory, backup)
    try:
        os.replace(staging, directory)
    except BaseException:
        os.replace(backup, directory)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def compute_analysis(index, request: CooccurRequest, *,
                     chunk_items: int = DEFAULT_CHUNK_ITEMS,
                     chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                     refresh: bool = False,
                     cache_root: Optional[str] = None) -> CooccurAnalysis:
    """Compute (or reopen) the analysis for ``request`` over ``index``.

    The cache key is ``(index identity, request digest)``, so two byte-
    identical indexes built separately never share results, and a changed
    request never reuses the old token.  Publication is atomic: everything
    is written to ``<dir>.staging``, hashed, marked OPEN (fsync), and
    renamed.  A failure removes the staging tree; any previously published
    analysis and the source index are untouched.
    """
    if not isinstance(request, CooccurRequest):
        raise TypeError("request must be a CooccurRequest")
    # The index's completed/OPEN state, manifest schema, file hashes and
    # identity were validated when it was opened.  Refuse a closed index so
    # a cursor/analysis can never be resumed against stale pager handles.
    for attr in ("_rank_pager", "_content_f"):
        fh = getattr(index, attr, None)
        if fh is None or getattr(fh, "closed", False) or \
                getattr(getattr(fh, "_f", None), "closed", False):
            raise IndexCorruptError(
                "index handle is closed; reopen it before analysing")
    if cache_root is None and os.environ.get("DISKFM_COOCCUR_CACHE"):
        cache_root = os.environ["DISKFM_COOCCUR_CACHE"]
    if cache_root is None:
        cache_root = _default_cache_root()
    directory = os.path.join(cache_root, index.identity, request.digest())
    staging = directory + _STAGING_SUFFIX
    if not refresh and os.path.exists(
            os.path.join(directory, _OPEN_MARKER)):
        return _reopen_analysis(index, request, directory)
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="cooccur_", dir=staging)
    try:
        builder = _AnalysisBuilder(index, request, workdir,
                                   chunk_items, chunk_bytes)
        summary_rows, position_pairs, pair_counts = builder.run()
        names = ["summary.bin", "pairs.bin"]
        files = _file_manifest(workdir, names)
        for name in names:
            os.replace(os.path.join(workdir, name),
                       os.path.join(staging, name))
            os.replace(os.path.join(workdir, name + ".off"),
                       os.path.join(staging, name + ".off"))
            files[name + ".off"] = fmt.hash_file(
                os.path.join(staging, name + ".off"))
        stats = dict(builder.stats)
        analysis_sha = hashlib.sha256(
            json.dumps({"request": request.digest(),
                        "content": stats["content_sha256"],
                        "pairs": position_pairs,
                        "summary": summary_rows},
                       sort_keys=True).encode("utf-8")).hexdigest()
        manifest = {
            "analysis_format": "diskfm-cooccur/1",
            "analysis_format_version": ANALYSIS_FORMAT_VERSION,
            "source_index_identity": index.identity,
            "source_content_hash": index.manifest.get("content_hash"),
            "request": request.normalised(),
            "request_sha256": request.digest(),
            "analysis_sha256": analysis_sha,
            "summary_rows": summary_rows,
            "position_pairs": position_pairs,
            "pair_counts": pair_counts,
            "parameters": {
                "chunk_items": chunk_items,
                "chunk_bytes": chunk_bytes,
            },
            "stats": stats,
            "files": files,
        }
        _publish_manifest(directory, manifest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return _reopen_analysis(index, request, directory)


def _reopen_analysis(index, request: CooccurRequest,
                     directory: str) -> CooccurAnalysis:
    manifest = CooccurAnalysis._load_manifest(directory)
    if manifest.get("source_index_identity") != index.identity:
        raise IndexCorruptError(
            "analysis was built for a different index identity")
    if manifest.get("request_sha256") != request.digest():
        raise IndexCorruptError(
            "analysis request digest does not match this request")
    CooccurAnalysis._verify_files(directory, manifest)
    return CooccurAnalysis(index, request, directory, manifest)


def analyse(index, patterns: Sequence, pairs: Sequence, window: int, *,
            chunk_items: int = DEFAULT_CHUNK_ITEMS,
            chunk_bytes: int = DEFAULT_CHUNK_BYTES,
            refresh: bool = False,
            cache_root: Optional[str] = None) -> CooccurAnalysis:
    """Validate a request and return a paged :class:`CooccurAnalysis`.

    This is the single analysis entry point; :meth:`DiskFMIndex.cooccur`
    and the CLI both call it.  Validation covers request structure and the
    request's pattern ids, phrases, pairs and window; the index's completion
    marker, manifest hashes and identity are validated when it was opened.
    """
    request = build_request(patterns, pairs, window)
    return compute_analysis(index, request, chunk_items=chunk_items,
                            chunk_bytes=chunk_bytes, refresh=refresh,
                            cache_root=cache_root)

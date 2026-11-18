"""Read side of the disk FM-index: open, count, locate, document text.

Ordering of the deterministic suffix order (built on integer symbols):

* user symbols are compared by their exact Unicode code point value
  (0..U+10FFFF); no case folding, no normalisation;
* internal document-end sentinels compare *after* every legal user code
  point because they start at 0x110000;
* among sentinels, lower document index sorts first.

Because each document ends in a unique sentinel that no query string can
contain, a match interval can never span a document boundary: a query q
matches only BWT rows whose suffix begins with q inside one document.
During LF recovery the walk is globally exact on the cyclic symbol stream;
when BWT[cur_t] is document j's sentinel we stop one step early and use
``p0 = (sentinel_pos(j) + 1 + t) mod N`` instead of taking the
boundary-crossing LF step, reconstructing the match inside the same doc.

Cursors are opaque tokens binding (index identity, exact query, interval,
next page offset).  Reusing a cursor on another index or with another query
raises :class:`~diskfm.format.CursorError`.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import format as fmt
from .pager import FilePager


@dataclass
class Hit:
    doc_id: str
    codepoint_offset: int
    utf8_byte_offset: int


class DiskFMIndex:
    """All random access goes through bounded LRUs; nothing loads wholesale."""

    def __init__(self, index_dir: str, max_pages: int = 16,
                 page_records: int = 4096):
        self.index_dir = index_dir
        self._stats = {
            "rank_calls": 0,
            "rank_page_misses": 0,
            "rank_page_hits": 0,
            "lf_steps": 0,
            "sample_rows_read": 0,
            "bytesample_pages_read": 0,
            "content_bytes_read": 0,
        }
        if not os.path.isdir(index_dir):
            raise FileNotFoundError(index_dir)
        if not os.path.exists(os.path.join(index_dir, fmt.OPEN_MARKER)):
            raise fmt.IndexCorruptError(
                "no OPEN marker: index build was interrupted; not readable")
        with open(os.path.join(index_dir, fmt.MANIFEST_NAME),
                  encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("format_version") != fmt.FORMAT_VERSION:
            raise fmt.IndexCorruptError("manifest format version mismatch")
        self._verify_files()
        geo = self.manifest["geometry"]
        self.n: int = geo["n_symbols"]
        self.n_docs: int = geo["n_documents"]
        self.sample_interval: int = geo["sample_interval"]
        self.byte_sample_interval: int = geo["byte_sample_interval"]
        self.identity: str = self.manifest["identity"]
        self._open_pagers(page_records, max_pages)
        self._validate_rankdir_geometry()
        self._validate_doctable_geometry()

    def _open_pagers(self, page_records: int, max_pages: int) -> None:
        self._rank_pager = FilePager(
            self._path("rank.bin"), record_size=4,
            header_size=fmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        # Symbol directory: fixed 32-byte records, paged like every other
        # random-access structure so its RAM does not grow with alphabet.
        self._rankdir_pager = FilePager(
            self._path("rankdir.bin"),
            record_size=fmt.RANKDIR_RECORD_SIZE,
            header_size=fmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._sample_pager = FilePager(
            self._path("sa_sample.bin"),
            record_size=fmt.SAMPLE_RECORD_SIZE,
            header_size=fmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._bytesample_pager = FilePager(
            self._path("bytesample.bin"), record_size=8,
            header_size=fmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        # Document coordinate records are fixed-width and also paged.
        self._doctable_pager = FilePager(
            self._path("doctable.bin"),
            record_size=fmt.DOCTABLE_RECORD_SIZE,
            header_size=fmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._content_f = open(self._path("content.bin"), "rb")
        self._doctable_f = open(self._path("doctable.bin"), "rb")

    # ------------------------------------------------------------------ #
    # Opening / integrity
    # ------------------------------------------------------------------ #
    def _path(self, name: str) -> str:
        return os.path.join(self.index_dir, name)

    def _verify_files(self) -> None:
        for name, entry in self.manifest["files"].items():
            fmt.verify_file(self._path(name), entry)
        # The manifest itself and OPEN marker existence complete the binding.
        if not os.path.exists(self._path(fmt.MANIFEST_NAME)):
            raise fmt.IndexCorruptError("manifest missing")

    @property
    def _rankdir_count(self) -> int:
        entry = self.manifest["files"]["rankdir.bin"]
        body = entry["bytes"] - fmt.HEADER_SIZE
        if body % fmt.RANKDIR_RECORD_SIZE != 0:
            raise fmt.IndexCorruptError("rank directory has ragged records")
        return body // fmt.RANKDIR_RECORD_SIZE

    def _rankdir_record(self, index: int) -> Tuple[int, int, int, int]:
        return fmt.RANKDIR_RECORD.unpack(
            self._rankdir_pager.read_record(index))

    def _validate_rankdir_geometry(self) -> None:
        # Stream the directory through its LRU pager; the cache cap, not the
        # alphabet size, bounds the resident memory of this check.
        total = 0
        prev_symbol = None
        for i in range(self._rankdir_count):
            symbol, _off, count, cumulative = self._rankdir_record(i)
            if prev_symbol is not None and symbol <= prev_symbol:
                raise fmt.IndexCorruptError("rankdir symbols not sorted")
            if cumulative != total:
                raise fmt.IndexCorruptError("rankdir cumulative counts wrong")
            if symbol > 0xFFFFFFFF or count == 0:
                raise fmt.IndexCorruptError("rankdir record out of range")
            prev_symbol = symbol
            total += count
        if total != self.n:
            raise fmt.IndexCorruptError(
                f"rankdir totals {total} occurrences but N={self.n}")

    def _doc_record(self, doc_idx: int) -> dict:
        if not 0 <= doc_idx < self.n_docs:
            raise fmt.IndexCorruptError(
                f"document index {doc_idx} out of range 0..{self.n_docs - 1}")
        (sentinel_pos, t_start, content_start, t_len, id_blob_offset,
         id_len, byte_cp_start) = fmt.DOCTABLE_RECORD.unpack(
             self._doctable_pager.read_record(doc_idx))
        if t_start + t_len != sentinel_pos:
            raise fmt.IndexCorruptError(
                f"document {doc_idx} geometry inconsistent")
        return {
            "sentinel_pos": sentinel_pos,
            "t_start": t_start,
            "content_start": content_start,
            "t_len": t_len,
            "id_blob_offset": id_blob_offset,
            "id_len": id_len,
            "byte_cp_start": byte_cp_start,
        }

    def _read_doc_id(self, doc_idx: int) -> str:
        rec = self._doc_record(doc_idx)
        abs_off = (fmt.HEADER_SIZE
                   + self.n_docs * fmt.DOCTABLE_RECORD_SIZE
                   + rec["id_blob_offset"])
        self._doctable_f.seek(abs_off)
        raw = self._doctable_f.read(rec["id_len"])
        if len(raw) != rec["id_len"]:
            raise fmt.IndexCorruptError("doc id blob truncated")
        return raw.decode("utf-8")

    def _validate_doctable_geometry(self) -> None:
        # Validate every fixed record and account for the whole id blob via
        # paged reads; nothing is retained beyond the LRU window.
        id_total = 0
        expected_offsets = 0
        for d in range(self.n_docs):
            rec = self._doc_record(d)
            if rec["byte_cp_start"] > 0xFFFFFFFF:
                raise fmt.IndexCorruptError("byte checkpoint index too large")
            if rec["id_blob_offset"] != expected_offsets:
                raise fmt.IndexCorruptError(
                    "doc id blob offsets are not contiguous")
            expected_offsets += rec["id_len"]
            id_total += rec["id_len"]
        expected = (fmt.HEADER_SIZE
                    + self.n_docs * fmt.DOCTABLE_RECORD_SIZE + id_total)
        actual = self.manifest["files"]["doctable.bin"]["bytes"]
        if expected != actual:
            raise fmt.IndexCorruptError(
                f"doctable length {actual} does not match geometry {expected}")

    def close(self) -> None:
        self._rank_pager.close()
        self._rankdir_pager.close()
        self._sample_pager.close()
        self._bytesample_pager.close()
        self._doctable_pager.close()
        self._content_f.close()
        self._doctable_f.close()

    def __enter__(self) -> "DiskFMIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Rank / C / LF -- the on-disk FM primitives
    # ------------------------------------------------------------------ #
    def _dir_index(self, symbol: int) -> Optional[int]:
        """Paged binary search over the sorted rank-directory symbols."""
        lo, hi = 0, self._rankdir_count
        while lo < hi:
            mid = (lo + hi) // 2
            if self._rankdir_record(mid)[0] < symbol:
                lo = mid + 1
            else:
                hi = mid
        if lo < self._rankdir_count and \
                self._rankdir_record(lo)[0] == symbol:
            return lo
        return None

    def rank(self, symbol: int, i: int) -> int:
        """Occurrences of ``symbol`` in BWT[0:i] (i excluded).

        The occurrence list of ``symbol`` is a sorted u32 array on disk;
        rank is a paged binary search, never a corpus scan.
        """
        self._stats["rank_calls"] += 1
        if i <= 0:
            return 0
        dir_idx = self._dir_index(symbol)
        if dir_idx is None:
            return 0
        _sym, offset, count, _cumulative = self._rankdir_record(dir_idx)
        if i >= self.n:
            return count
        first = (offset - fmt.HEADER_SIZE) // 4
        lo, hi = 0, count
        # Paged binary search; only log2(count) records are touched.
        while lo < hi:
            mid = (lo + hi) // 2
            v = self._rank_pager.get_u32(first + mid)
            if v < i:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _c(self, symbol: int) -> Optional[int]:
        """C(symbol): count of symbols strictly smaller, via the directory."""
        dir_idx = self._dir_index(symbol)
        if dir_idx is None:
            return None
        return self._rankdir_record(dir_idx)[3]

    def lf(self, row: int) -> int:
        """One LF step: LF(i) = C(BWT[i]) + rank(BWT[i], i)."""
        bwt_symbol = self._bwt_at(row)
        base = self._c(bwt_symbol)
        if base is None:
            raise fmt.IndexCorruptError("BWT symbol absent from rank directory")
        return base + self.rank(bwt_symbol, row)

    def _bwt_at(self, row: int) -> int:
        # BWT rows are needed one at a time during LF; read through the same
        # bounded pager mechanism rather than decoding the whole BWT.
        if not hasattr(self, "_bwt_pager"):
            self._bwt_pager = FilePager(
                self._path("bwt.bin"), record_size=4,
                header_size=fmt.HEADER_SIZE,
                page_records=self._rank_pager.page_records,
                max_pages=self._rank_pager.max_pages)
        return self._bwt_pager.get_u32(row)

    # ------------------------------------------------------------------ #
    # Backward search
    # ------------------------------------------------------------------ #
    def _interval(self, query: str) -> Tuple[int, int]:
        fmt.validate_query(query)
        if self.n == 0:
            return (0, 0)
        lo, hi = 0, self.n
        for ch in reversed(query):
            c = ord(ch)
            base = self._c(c)
            if base is None:
                return (0, 0)
            lo = base + self.rank(c, lo)
            hi = base + self.rank(c, hi)
            if lo == hi:
                return (0, 0)
        return lo, hi

    def count(self, query: str) -> int:
        """Overlapping occurrence count via interval width; nothing enumerated."""
        lo, hi = self._interval(query)
        return hi - lo

    # ------------------------------------------------------------------ #
    # Coordinate recovery
    # ------------------------------------------------------------------ #
    def _sample_count(self) -> int:
        entry = self.manifest["files"]["sa_sample.bin"]
        body = entry["bytes"] - fmt.HEADER_SIZE
        if body % fmt.SAMPLE_RECORD_SIZE != 0:
            raise fmt.IndexCorruptError("sa_sample has ragged records")
        return body // fmt.SAMPLE_RECORD_SIZE

    def _sample_at(self, sample_index: int) -> Tuple[int, int]:
        raw = self._sample_pager.read_record(sample_index)
        row, pos = fmt.SAMPLE_RECORD.unpack(raw)
        self._stats["sample_rows_read"] += 1
        return row, pos

    def _row_to_position(self, row: int) -> int:
        """LF-walk to a sampled row, then reconstruct the T position.

        Invariant after t LF steps: SA[cur_t] = (p0 - t) mod N, where p0 is
        the T position of the starting row.  Hence, once a sampled row is
        reached, p0 = (sample_pos + t) mod N.  LF is globally well-defined
        on the cyclic stream, so walking *through* a sentinel row would stay
        exact; as an optimisation we stop one step earlier.  When
        BWT[cur_t] is a sentinel S_j, necessarily
        SA[cur_t] = sentinel_pos(j) + 1 (the first symbol of the next
        document), hence p0 = (sentinel_pos(j) + 1 + t) mod N.  This never
        fabricates a cross-document match: backward search cannot cross a
        sentinel because no query contains a code point >= 0x110000.
        """
        sample_count = self._sample_count()
        t = 0
        cur = row
        while cur % self.sample_interval != 0:
            sym = self._bwt_at(cur)
            if sym >= fmt.MIN_SENTINEL:
                doc_idx = sym - fmt.MIN_SENTINEL
                sentinel_pos = self._doc_record(doc_idx)["sentinel_pos"]
                return (sentinel_pos + 1 + t) % self.n
            cur = self.lf(cur)
            t += 1
            self._stats["lf_steps"] += 1
        # cur is a sampled SA row: binary search its sample slot.
        lo, hi = 0, sample_count
        while lo < hi:
            mid = (lo + hi) // 2
            sample_row, _ = self._sample_at(mid)
            if sample_row < cur:
                lo = mid + 1
            else:
                hi = mid
        if lo >= sample_count:
            raise fmt.IndexCorruptError("sampled row missing during locate")
        sample_row, sample_pos = self._sample_at(lo)
        if sample_row != cur:
            raise fmt.IndexCorruptError("SA sample lookup failed")
        return (sample_pos + t) % self.n

    def _doc_for_position(self, pos: int) -> int:
        """Paged binary search over t_start values (records are doc-ordered)."""
        lo, hi = 0, self.n_docs
        while lo < hi:
            mid = (lo + hi) // 2
            if self._doc_record(mid)["t_start"] <= pos:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            raise fmt.IndexCorruptError(f"position {pos} outside any document")
        rec = self._doc_record(idx)
        if pos > rec["sentinel_pos"]:
            raise fmt.IndexCorruptError("position landed beyond document end")
        return idx

    def _byte_offset(self, doc_idx: int, cp_offset: int) -> int:
        """Return the UTF-8 byte offset of codepoint ``cp_offset`` inside
        the document's own byte string (so the first byte is always 0)."""
        doc = self._doc_record(doc_idx)
        j = self.byte_sample_interval
        k = cp_offset // j
        cp_start = doc["byte_cp_start"]
        base_cp = k * j
        base_byte = self._bytesample_pager.get_u64(cp_start + k)
        extra = cp_offset - base_cp
        if extra == 0:
            return base_byte
        # Three distinct offset spaces stay explicit here:
        #   base_byte        : bytes inside this document (what we return)
        #   rel_seek         : body-relative position in content.bin
        #   abs_seek/abs_end : real file offsets including the 8-byte header
        rel_seek = doc["content_start"] + base_byte
        abs_seek = fmt.HEADER_SIZE + rel_seek
        abs_end = (fmt.HEADER_SIZE + doc["content_start"]
                   + self._doc_byte_len(doc_idx))
        # Stream at most `extra` codepoints through an incremental decoder:
        # we never overshoot a character boundary or enter the next doc.
        self._content_f.seek(abs_seek)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        decoded = ""
        while len(decoded) < extra:
            if self._content_f.tell() >= abs_end:
                raise fmt.IndexCorruptError(
                    "byte-offset recovery ran past document end")
            one = self._content_f.read(1)
            self._stats["content_bytes_read"] += 1
            try:
                decoded += decoder.decode(one)
            except UnicodeDecodeError as exc:
                raise fmt.IndexCorruptError(
                    "invalid UTF-8 inside indexed document") from exc
        return base_byte + len(decoded[:extra].encode("utf-8"))

    def _doc_byte_len(self, doc_idx: int) -> int:
        rec = self._doc_record(doc_idx)
        if doc_idx + 1 < self.n_docs:
            return self._doc_record(doc_idx + 1)["content_start"] - \
                rec["content_start"]
        body_len = (self.manifest["files"]["content.bin"]["bytes"]
                    - fmt.HEADER_SIZE)
        return body_len - rec["content_start"]

    def locate_page(self, query: str, page_size: int,
                    cursor: Optional[str] = None) -> Tuple[List[Hit], Optional[str]]:
        """Return one page of hits and a cursor for the next page (or None)."""
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        lo, hi = self._interval(query)
        width = hi - lo
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if cursor is None:
            offset = 0
        else:
            offset = self._consume_cursor(cursor, query, lo, hi, query_hash)
        if offset >= width:
            return [], None
        end = min(offset + page_size, width)
        hits: List[Hit] = []
        for row in range(lo + offset, lo + end):
            pos = self._row_to_position(row)
            doc_idx = self._doc_for_position(pos)
            doc = self._doc_record(doc_idx)
            cp_off = pos - doc["t_start"]
            if cp_off > doc["t_len"]:
                raise fmt.IndexCorruptError("hit outside document bounds")
            byte_off = self._byte_offset(doc_idx, cp_off)
            hits.append(Hit(self._read_doc_id(doc_idx), cp_off, byte_off))
        next_offset = end
        next_cursor = None
        if next_offset < width:
            next_cursor = self._make_cursor(
                query_hash, lo, hi, next_offset)
        return hits, next_cursor

    def locate_all(self, query: str, page_size: int = 64) -> List[Hit]:
        """Convenience: page through everything (each page recovered alone)."""
        out: List[Hit] = []
        cursor = None
        while True:
            page, cursor = self.locate_page(query, page_size, cursor)
            out.extend(page)
            if cursor is None:
                break
        return out

    # ------------------------------------------------------------------ #
    # Cursors: bound to identity + query + interval + position
    # ------------------------------------------------------------------ #
    def _make_cursor(self, query_hash: str, lo: int, hi: int,
                     offset: int) -> str:
        payload = json.dumps({
            "id": self.identity,
            "q": query_hash,
            "lo": lo,
            "hi": hi,
            "off": offset,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return digest + payload.hex()

    def _consume_cursor(self, cursor: str, query: str, lo: int, hi: int,
                        query_hash: str) -> int:
        try:
            digest = cursor[:16]
            payload = bytes.fromhex(cursor[16:])
            if hashlib.sha256(payload).hexdigest()[:16] != digest:
                raise ValueError
            token = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise fmt.CursorError("malformed or tampered cursor") from exc
        if token.get("id") != self.identity:
            raise fmt.CursorError(
                "cursor belongs to a different index identity")
        if token.get("q") != query_hash:
            raise fmt.CursorError(
                "cursor was issued for a different query fragment")
        if token.get("lo") != lo or token.get("hi") != hi:
            raise fmt.CursorError(
                "cursor interval no longer matches this query")
        offset = token.get("off")
        if not isinstance(offset, int) or offset < 0:
            raise fmt.CursorError("cursor position invalid")
        if offset > (hi - lo):
            raise fmt.CursorError("cursor position past end of interval")
        return offset

    # ------------------------------------------------------------------ #
    # Original text access (raw strings preserved byte-for-byte)
    # ------------------------------------------------------------------ #
    def document_text(self, doc_id: str) -> str:
        # Paged scan of document ids; no full doctable/id-blob retention.
        for idx in range(self.n_docs):
            if self._read_doc_id(idx) == doc_id:
                rec = self._doc_record(idx)
                length = self._doc_byte_len(idx)
                self._content_f.seek(
                    fmt.HEADER_SIZE + rec["content_start"])
                raw = self._content_f.read(length)
                self._stats["content_bytes_read"] += len(raw)
                return raw.decode("utf-8")
        raise KeyError(doc_id)

    def stats(self) -> Dict[str, int]:
        s = dict(self._stats)
        s["rank_page_misses"] = self._rank_pager.page_misses
        s["rank_page_hits"] = self._rank_pager.page_hits
        s["bytesample_pages_read"] = self._bytesample_pager.page_misses
        s["sample_pager"] = self._sample_pager.stats()
        s["rank_pager"] = self._rank_pager.stats()
        s["rankdir_pager"] = self._rankdir_pager.stats()
        s["doctable_pager"] = self._doctable_pager.stats()
        s["bytesample_pager"] = self._bytesample_pager.stats()
        if hasattr(self, "_bwt_pager"):
            s["bwt_pager"] = self._bwt_pager.stats()
        return s

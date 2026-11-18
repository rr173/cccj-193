"""Read side of the similarity catalog: open, query, explain, validate.

A catalog is fully self-contained: it stores qgram relations derived from
the source corpus, never the text itself, so **no query path can read the
original documents back** and none ever re-shingles a document.

Query plan for ``query(source_doc_id, threshold, page_size)``:

1. Read only the source document's own qgram-id list from ``docqgrams.bin``
   (one contiguous slice, paged).
2. For each of those qgram ids, stream its posting list from
   ``postings.bin`` (paged, bounded LRU) and emit the candidate doc indexes
   into a bounded **external counting merge** -- hot template shingles with
   huge posting lists spill to disk instead of accumulating in RAM.  No
   candidate counts are kept in an in-memory hash table.
3. Merging the counter yields ``(candidate, intersection)`` pairs.  The
   union is ``s + t - intersection`` with per-document shingle counts read
   from ``docs.bin`` (paged).  The threshold test is exact integer
   arithmetic ``inter * den >= num * union`` -- binary floating point never
   decides a boundary.
4. Passing results are sorted by ``(-similarity, target_doc_id)`` with the
   similarity kept as an exact :class:`fractions.Fraction`, again through a
   bounded external sort, then paged.  There is no candidate cap: every
   document sharing at least one shingle is counted and scored.

Cursors are opaque tokens binding (catalog identity, source doc_id, exact
threshold, offset) with a tamper-evident digest; reuse across a different
catalog, source document or threshold raises
:class:`~diskfm.format.CursorError`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Tuple

from . import simformat as sfmt
from .format import CursorError, IndexCorruptError
from .pager import FilePager
from .vsort import ResultCodec, U64Codec, VarExternalSorter

# Query-time memory knobs (separate from the build-time ones).
_COUNTER_CHUNK_ITEMS = 20000
_COUNTER_CHUNK_BYTES = 1 << 22
_RESULT_CHUNK_ITEMS = 20000
_RESULT_CHUNK_BYTES = 1 << 22


def parse_threshold(value) -> Fraction:
    """Parse a threshold into an exact Fraction in the interval (0, 1].

    Floats are converted through their decimal ``repr`` (``0.1`` means
    exactly 1/10, not its binary approximation); strings accept both
    decimal (``"0.75"``) and ratio (``"3/4"``) forms.  Boundary decisions
    therefore never depend on binary floating point.
    """
    if isinstance(value, bool):
        raise TypeError("threshold must not be a bool")
    if isinstance(value, Fraction):
        frac = value
    elif isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, float):
        frac = Fraction(str(value))
    elif isinstance(value, str):
        try:
            frac = Fraction(value.strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"unparsable threshold: {value!r}") from exc
    else:
        raise TypeError(f"unsupported threshold type: {type(value).__name__}")
    if not (Fraction(0) < frac <= Fraction(1)):
        raise ValueError(
            f"threshold must be > 0 and <= 1, got {value!r}")
    return frac


def _result_sort_key(record):
    inter, union, _t_shingles, doc_id_bytes = record
    # Exact rational order, descending similarity, then ascending doc_id
    # (UTF-8 byte order == Unicode code point order).
    return (-Fraction(inter, union), doc_id_bytes)


class SimilarityCatalog:
    """An opened similarity catalog; all random access is paged/bounded."""

    def __init__(self, catalog_dir: str, max_pages: int = 16,
                 page_records: int = 4096,
                 counter_chunk_items: int = _COUNTER_CHUNK_ITEMS,
                 counter_chunk_bytes: int = _COUNTER_CHUNK_BYTES,
                 result_chunk_items: int = _RESULT_CHUNK_ITEMS,
                 result_chunk_bytes: int = _RESULT_CHUNK_BYTES):
        self.catalog_dir = catalog_dir
        self._counter_chunk_items = counter_chunk_items
        self._counter_chunk_bytes = counter_chunk_bytes
        self._result_chunk_items = result_chunk_items
        self._result_chunk_bytes = result_chunk_bytes
        if not os.path.isdir(catalog_dir):
            raise FileNotFoundError(catalog_dir)
        sfmt.check_open_marker(catalog_dir)
        with open(os.path.join(catalog_dir, sfmt.MANIFEST_NAME),
                  encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("format_version") != sfmt.SIM_FORMAT_VERSION:
            raise IndexCorruptError("catalog format version mismatch")
        self._check_manifest_schema()
        for name, entry in self.manifest["files"].items():
            sfmt.verify_file(os.path.join(catalog_dir, name), entry)

        self.q: int = self.manifest["parameters"]["q"]
        self.n_docs: int = self.manifest["documents"]["count"]
        self.n_qgrams: int = self.manifest["dictionary"]["qgram_count"]
        self.n_postings: int = self.manifest["postings"]["entry_count"]
        self.identity: str = self.manifest["catalog_identity"]

        self._qgramdir_pager = FilePager(
            self._path("qgramdir.bin"), record_size=sfmt.QGRAMDIR_RECORD_SIZE,
            header_size=sfmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._postings_pager = FilePager(
            self._path("postings.bin"), record_size=sfmt.POSTING_RECORD_SIZE,
            header_size=sfmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._docs_pager = FilePager(
            self._path("docs.bin"), record_size=sfmt.DOCS_RECORD_SIZE,
            header_size=sfmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._docqgrams_pager = FilePager(
            self._path("docqgrams.bin"),
            record_size=sfmt.DOCQGRAM_RECORD_SIZE,
            header_size=sfmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._qgrams_f = open(self._path("qgrams.bin"), "rb")
        self._docs_f = open(self._path("docs.bin"), "rb")
        self._last_query_stats: Optional[Dict[str, object]] = None
        self._validate_geometry()

    # ------------------------------------------------------------------ #
    # Opening / integrity
    # ------------------------------------------------------------------ #
    def _path(self, name: str) -> str:
        return os.path.join(self.catalog_dir, name)

    def _check_manifest_schema(self) -> None:
        m = self.manifest
        try:
            src = m["source"]
            params = m["parameters"]
            docs = m["documents"]
            rel = m["relations"]
            dictionary = m["dictionary"]
            postings = m["postings"]
            files = m["files"]
        except KeyError as exc:
            raise IndexCorruptError(
                f"catalog manifest missing section: {exc}") from exc
        if not isinstance(src.get("index_identity"), str) \
                or not isinstance(src.get("content_hash"), str):
            raise IndexCorruptError("manifest source identity incomplete")
        if params.get("tokenization") != sfmt.TOKENIZATION_RULE:
            raise IndexCorruptError("unknown tokenization rule declaration")
        if not isinstance(params.get("q"), int) or params["q"] < 1:
            raise IndexCorruptError("manifest q is not a positive integer")
        for section, keys in (
                (docs, ("count", "with_shingles", "without_shingles",
                        "digest")),
                (rel, ("raw", "deduped")),
                (dictionary, ("qgram_count",)),
                (postings, ("entry_count",))):
            for k in keys:
                if not isinstance(section.get(k), (int, str)):
                    raise IndexCorruptError(
                        f"manifest field {k} missing or mistyped")
        if docs["with_shingles"] + docs["without_shingles"] != docs["count"]:
            raise IndexCorruptError("manifest document counts inconsistent")
        if rel["deduped"] > rel["raw"]:
            raise IndexCorruptError("manifest relation counts impossible")
        expected_files = {"qgrams.bin", "qgramdir.bin", "postings.bin",
                          "docqgrams.bin", "docs.bin"}
        if set(files) != expected_files:
            raise IndexCorruptError("manifest file set unexpected")

    def _file_body(self, name: str) -> int:
        return self.manifest["files"][name]["bytes"] - sfmt.HEADER_SIZE

    # File name -> expected kind tag in the 8-byte header.
    _FILE_KINDS = {
        "qgrams.bin": sfmt.KIND_QGRAMS,
        "qgramdir.bin": sfmt.KIND_QGRAMDIR,
        "postings.bin": sfmt.KIND_POSTINGS,
        "docqgrams.bin": sfmt.KIND_DOCQGRAMS,
        "docs.bin": sfmt.KIND_DOCS,
    }

    def _check_file_headers(self) -> None:
        # Beyond the manifest hashes, every data file must carry its own
        # magic/version/kind header.
        for name, kind in self._FILE_KINDS.items():
            with open(self._path(name), "rb") as f:
                sfmt.read_header(f, kind)

    def _validate_geometry(self) -> None:
        self._check_file_headers()
        # File-size geometry must agree with the manifest counts exactly.
        if self._file_body("qgramdir.bin") != \
                self.n_qgrams * sfmt.QGRAMDIR_RECORD_SIZE:
            raise IndexCorruptError("qgramdir size does not match manifest")
        if self._file_body("postings.bin") != \
                self.n_postings * sfmt.POSTING_RECORD_SIZE:
            raise IndexCorruptError("postings size does not match manifest")
        if self._file_body("docqgrams.bin") != \
                self.manifest["relations"]["deduped"] * \
                sfmt.DOCQGRAM_RECORD_SIZE:
            raise IndexCorruptError("docqgrams size does not match manifest")
        docs_body = self._file_body("docs.bin")
        if docs_body < self.n_docs * sfmt.DOCS_RECORD_SIZE:
            raise IndexCorruptError("docs.bin shorter than its records")

        # Stream the qgram directory through its bounded pager: posting
        # slices must tile postings.bin, blob slices must tile qgrams.bin
        # contiguously, and every posting list must be non-empty.
        blob_expected = 0
        posting_expected = 0
        qgrams_body = self._file_body("qgrams.bin")
        prev_key: Optional[bytes] = None
        for i in range(self.n_qgrams):
            (blob_off, byte_len, reserved, post_off, post_count) = \
                self._qgramdir_record(i)
            if reserved != 0:
                raise IndexCorruptError("qgramdir reserved field non-zero")
            if blob_off != blob_expected:
                raise IndexCorruptError("qgram blob offsets not contiguous")
            if byte_len == 0:
                raise IndexCorruptError("empty qgram in dictionary")
            blob_expected += byte_len
            if post_off != posting_expected:
                raise IndexCorruptError("posting offsets not contiguous")
            if post_count < 1:
                raise IndexCorruptError("empty posting list")
            posting_expected += post_count
            # Dictionary order must be strictly increasing in UTF-8 byte
            # (== code point) order: this is the structural proof that no
            # two distinct qgrams were ever merged into one entry.
            key = self._read_qgram(blob_off, byte_len)
            if prev_key is not None and key <= prev_key:
                raise IndexCorruptError(
                    "qgram dictionary not strictly ordered")
            prev_key = key
        if blob_expected != qgrams_body:
            raise IndexCorruptError("qgram blob length mismatch")
        if posting_expected != self.n_postings:
            raise IndexCorruptError("posting total does not match manifest")

        # Stream the docs table: forward slices must tile docqgrams.bin,
        # id blob offsets must be contiguous, and the no-shingles flag must
        # agree with both the shingle count and the recorded text length.
        dq_expected = 0
        id_expected = 0
        shingle_sum = 0
        dq_total = self.manifest["relations"]["deduped"]
        for d in range(self.n_docs):
            rec = self._doc_record(d)
            if rec["docqgrams_offset"] != dq_expected:
                raise IndexCorruptError("docqgrams offsets not contiguous")
            dq_expected += rec["shingle_count"]
            shingle_sum += rec["shingle_count"]
            if rec["id_blob_offset"] != id_expected:
                raise IndexCorruptError("doc id blob offsets not contiguous")
            id_expected += rec["id_len"]
            flagged = bool(rec["flags"] & sfmt.FLAG_NO_SHINGLES)
            if rec["flags"] & ~sfmt.FLAG_NO_SHINGLES:
                raise IndexCorruptError("unknown docs.bin flag bits")
            if flagged != (rec["shingle_count"] == 0):
                raise IndexCorruptError("no-shingles flag/count disagree")
            if flagged != (rec["text_len"] < self.q):
                raise IndexCorruptError("no-shingles flag/length disagree")
        if dq_expected != dq_total or shingle_sum != dq_total:
            raise IndexCorruptError("docqgrams total does not match manifest")
        if docs_body != self.n_docs * sfmt.DOCS_RECORD_SIZE + id_expected:
            raise IndexCorruptError("docs.bin id blob length mismatch")

    def close(self) -> None:
        self._qgramdir_pager.close()
        self._postings_pager.close()
        self._docs_pager.close()
        self._docqgrams_pager.close()
        self._qgrams_f.close()
        self._docs_f.close()

    def __enter__(self) -> "SimilarityCatalog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Low-level paged accessors
    # ------------------------------------------------------------------ #
    def _qgramdir_record(self, qgram_id: int) -> Tuple[int, int, int, int, int]:
        return sfmt.QGRAMDIR_RECORD.unpack(
            self._qgramdir_pager.read_record(qgram_id))

    def _read_qgram(self, blob_offset: int, byte_len: int) -> bytes:
        self._qgrams_f.seek(sfmt.HEADER_SIZE + blob_offset)
        raw = self._qgrams_f.read(byte_len)
        if len(raw) != byte_len:
            raise IndexCorruptError("qgram blob truncated")
        return raw

    def _posting_entry(self, index: int) -> Tuple[int, int]:
        return sfmt.POSTING_RECORD.unpack(
            self._postings_pager.read_record(index))

    def _doc_record(self, doc_idx: int) -> Dict[str, int]:
        if not 0 <= doc_idx < self.n_docs:
            raise IndexCorruptError(f"doc index {doc_idx} out of range")
        (shingle_count, dq_offset, text_len, id_off, id_len, flags) = \
            sfmt.DOCS_RECORD.unpack(self._docs_pager.read_record(doc_idx))
        return {
            "shingle_count": shingle_count,
            "docqgrams_offset": dq_offset,
            "text_len": text_len,
            "id_blob_offset": id_off,
            "id_len": id_len,
            "flags": flags,
        }

    def _doc_id_bytes_at(self, rec: Dict[str, int]) -> bytes:
        abs_off = (sfmt.HEADER_SIZE + self.n_docs * sfmt.DOCS_RECORD_SIZE
                   + rec["id_blob_offset"])
        self._docs_f.seek(abs_off)
        raw = self._docs_f.read(rec["id_len"])
        if len(raw) != rec["id_len"]:
            raise IndexCorruptError("doc id blob truncated")
        return raw

    def _doc_id(self, doc_idx: int) -> str:
        return self._doc_id_bytes_at(self._doc_record(doc_idx)).decode("utf-8")

    def _find_doc(self, doc_id: str) -> int:
        """Paged scan of the doc-id table (never the text, never shingles)."""
        if not isinstance(doc_id, str):
            raise TypeError("doc_id must be a str")
        target = doc_id.encode("utf-8")
        for idx in range(self.n_docs):
            rec = self._doc_record(idx)
            if rec["id_len"] == len(target) and \
                    self._doc_id_bytes_at(rec) == target:
                return idx
        raise KeyError(doc_id)

    def _iter_doc_qgrams(self, rec: Dict[str, int]) -> Iterator[int]:
        """Stream one document's sorted distinct qgram ids (paged)."""
        start = rec["docqgrams_offset"]
        for i in range(start, start + rec["shingle_count"]):
            yield sfmt.DOCQGRAM_RECORD.unpack(
                self._docqgrams_pager.read_record(i))[0]

    def _iter_posting(self, qgram_id: int) -> Iterator[Tuple[int, int]]:
        """Stream one qgram's (doc_index, term_freq) entries (paged)."""
        _blob_off, _blen, _res, post_off, post_count = \
            self._qgramdir_record(qgram_id)
        for i in range(post_off, post_off + post_count):
            yield self._posting_entry(i)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def _compute_results(self, source_doc_id: str, thr: Fraction,
                         workdir: str):
        """Run candidate counting + exact scoring; returns (sorted result
        handle, stats).  The caller must clean up the result handle."""
        stats: Dict[str, object] = {
            "source_doc_id": source_doc_id,
            "threshold": f"{thr.numerator}/{thr.denominator}",
            "posting_lists_read": 0,
            "posting_records_read": 0,
            "candidate_relations": 0,
            "candidates": 0,
            "exact_scored": 0,
            "results": 0,
        }
        src_idx = self._find_doc(source_doc_id)
        src_rec = self._doc_record(src_idx)
        s_count = src_rec["shingle_count"]
        stats["source_doc_index"] = src_idx
        stats["source_shingle_count"] = s_count
        counter = VarExternalSorter(
            workdir, U64Codec(), chunk_items=self._counter_chunk_items,
            chunk_bytes=self._counter_chunk_bytes)
        result_sorter = VarExternalSorter(
            workdir, ResultCodec(), chunk_items=self._result_chunk_items,
            chunk_bytes=self._result_chunk_bytes, key=_result_sort_key)
        try:
            if s_count > 0:
                # Only the source document's own qgram relations and the
                # postings of exactly those qgrams are ever read.
                for qid in self._iter_doc_qgrams(src_rec):
                    stats["posting_lists_read"] += 1
                    for doc_idx, _tf in self._iter_posting(qid):
                        stats["posting_records_read"] += 1
                        if doc_idx != src_idx:
                            counter.add(doc_idx)
                stats["candidate_relations"] = counter.items_seen
                counted = counter.finalize()
                with counted:
                    prev_doc: Optional[int] = None
                    inter = 0
                    for doc_idx in counted:
                        if doc_idx != prev_doc:
                            if prev_doc is not None:
                                self._score_candidate(
                                    prev_doc, inter, s_count, thr,
                                    result_sorter, stats)
                            prev_doc = doc_idx
                            inter = 1
                        else:
                            inter += 1
                    if prev_doc is not None:
                        self._score_candidate(
                            prev_doc, inter, s_count, thr,
                            result_sorter, stats)
                counted.cleanup()
            stats["counter_chunks_spilled"] = counter.chunks_spilled
            stats["counter_bytes_spilled"] = counter.bytes_spilled
            stats["counter_merge_passes"] = counter.merge_passes
            results = result_sorter.finalize()
            stats["result_chunks_spilled"] = result_sorter.chunks_spilled
            stats["result_bytes_spilled"] = result_sorter.bytes_spilled
        except BaseException:
            counter.cleanup_discarded()
            result_sorter.cleanup_discarded()
            raise
        stats["results"] = result_sorter.items_seen
        self._last_query_stats = stats
        return results, stats

    def _score_candidate(self, doc_idx: int, inter: int, s_count: int,
                         thr: Fraction, result_sorter: VarExternalSorter,
                         stats: Dict[str, object]) -> None:
        stats["candidates"] += 1
        stats["exact_scored"] += 1
        t_count = self._doc_record(doc_idx)["shingle_count"]
        union = s_count + t_count - inter
        # Exact rational comparison: inter/union >= num/den, no floats.
        if inter * thr.denominator >= thr.numerator * union:
            doc_id_bytes = self._doc_id_bytes_at(self._doc_record(doc_idx))
            result_sorter.add((inter, union, t_count, doc_id_bytes))

    @staticmethod
    def _make_row(source_doc_id: str, s_count: int, inter: int, union: int,
                  t_shingles: int, doc_id_bytes: bytes) -> dict:
        sim = Fraction(inter, union)
        return {
            "source_doc_id": source_doc_id,
            "target_doc_id": doc_id_bytes.decode("utf-8"),
            "intersection_count": inter,
            "union_count": union,
            "similarity": f"{sim.numerator}/{sim.denominator}",
            "similarity_num": sim.numerator,
            "similarity_den": sim.denominator,
            "source_shingle_count": s_count,
            "target_shingle_count": t_shingles,
        }

    def _rows_from_results(self, results, source_doc_id: str, s_count: int,
                           offset: int, page_size: int
                           ) -> Tuple[List[dict], bool]:
        rows: List[dict] = []
        has_more = False
        with results:
            for idx, (inter, union, t_shingles, doc_id_bytes) in \
                    enumerate(results):
                if idx < offset:
                    continue
                if len(rows) >= page_size:
                    has_more = True
                    break
                rows.append(self._make_row(
                    source_doc_id, s_count, inter, union, t_shingles,
                    doc_id_bytes))
        return rows, has_more

    def query_page(self, source_doc_id: str, threshold, page_size: int,
                   cursor: Optional[str] = None
                   ) -> Tuple[List[dict], Optional[str]]:
        """One stable page of near-duplicates, ordered by similarity desc
        (exact rational) then target doc_id code point order."""
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        thr = parse_threshold(threshold)
        if cursor is None:
            offset = 0
        else:
            offset = self._consume_cursor(cursor, source_doc_id, thr)
        workdir = tempfile.mkdtemp(prefix="diskfm_simq_")
        try:
            results, stats = self._compute_results(
                source_doc_id, thr, workdir)
            try:
                src_rec = self._doc_record(stats["source_doc_index"])
                rows, has_more = self._rows_from_results(
                    results, source_doc_id, src_rec["shingle_count"],
                    offset, page_size)
            finally:
                results.cleanup()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        next_cursor = None
        if has_more:
            next_cursor = self._make_cursor(
                source_doc_id, thr, offset + page_size)
        return rows, next_cursor

    def iter_results(self, source_doc_id: str, threshold
                     ) -> Iterator[dict]:
        """Stream the full ordered result set with a single computation."""
        thr = parse_threshold(threshold)
        workdir = tempfile.mkdtemp(prefix="diskfm_simq_")
        try:
            results, stats = self._compute_results(
                source_doc_id, thr, workdir)
            try:
                src_rec = self._doc_record(stats["source_doc_index"])
                s_count = src_rec["shingle_count"]
                with results:
                    for inter, union, t_shingles, doc_id_bytes in results:
                        yield self._make_row(
                            source_doc_id, s_count, inter, union,
                            t_shingles, doc_id_bytes)
            finally:
                results.cleanup()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def query_all(self, source_doc_id: str, threshold,
                  page_size: int = 64) -> List[dict]:
        """All result rows from one computation (page_size only chunks the
        internal iteration; the order is identical to paged querying)."""
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        return list(self.iter_results(source_doc_id, threshold))

    # ------------------------------------------------------------------ #
    # Explain: shared shingles between two documents
    # ------------------------------------------------------------------ #
    def explain(self, source_doc_id: str, target_doc_id: str) -> List[dict]:
        """The deterministically ordered shingles shared by two documents.

        Order is dictionary order (UTF-8 byte order == code point order of
        the qgram).  Each entry carries the qgram both as a JSON-safe
        string and as hex UTF-8 (so NUL and non-ASCII are unambiguous),
        plus its occurrence count inside each of the two documents.  This
        operation is independent of -- and never influences -- query
        ordering.
        """
        s_idx = self._find_doc(source_doc_id)
        t_idx = self._find_doc(target_doc_id)
        s_rec = self._doc_record(s_idx)
        t_rec = self._doc_record(t_idx)
        shared: List[dict] = []
        if s_rec["shingle_count"] == 0 or t_rec["shingle_count"] == 0:
            self._last_explain_stats = {
                "shared_count": 0,
                "dictionary_reads": 0,
                "posting_probes": 0,
            }
            return shared
        s_iter = self._iter_doc_qgrams(s_rec)
        t_iter = self._iter_doc_qgrams(t_rec)
        s_cur = next(s_iter, None)
        t_cur = next(t_iter, None)
        dict_reads = 0
        probes = 0
        while s_cur is not None and t_cur is not None:
            if s_cur < t_cur:
                s_cur = next(s_iter, None)
            elif t_cur < s_cur:
                t_cur = next(t_iter, None)
            else:
                blob_off, byte_len, _r, post_off, post_count = \
                    self._qgramdir_record(s_cur)
                qgram_bytes = self._read_qgram(blob_off, byte_len)
                dict_reads += 1
                tf_s = self._posting_tf(post_off, post_count, s_idx)
                tf_t = self._posting_tf(post_off, post_count, t_idx)
                probes += 2
                shared.append({
                    "qgram": qgram_bytes.decode("utf-8"),
                    "qgram_utf8_hex": qgram_bytes.hex(),
                    "count_in_source": tf_s,
                    "count_in_target": tf_t,
                })
                s_cur = next(s_iter, None)
                t_cur = next(t_iter, None)
        self._last_explain_stats = {
            "shared_count": len(shared),
            "dictionary_reads": dict_reads,
            "posting_probes": probes,
        }
        return shared

    def _posting_tf(self, post_off: int, post_count: int,
                    doc_idx: int) -> int:
        """Paged binary search for doc_idx inside one sorted posting list."""
        lo, hi = 0, post_count
        while lo < hi:
            mid = (lo + hi) // 2
            d, tf = self._posting_entry(post_off + mid)
            if d < doc_idx:
                lo = mid + 1
            else:
                hi = mid
        if lo < post_count:
            d, tf = self._posting_entry(post_off + lo)
            if d == doc_idx:
                return tf
        raise IndexCorruptError(
            "posting list does not contain a document its forward "
            "relation claims")

    # ------------------------------------------------------------------ #
    # Cursors
    # ------------------------------------------------------------------ #
    def _make_cursor(self, source_doc_id: str, thr: Fraction,
                     offset: int) -> str:
        payload = json.dumps({
            "id": self.identity,
            "src": source_doc_id,
            "th": f"{thr.numerator}/{thr.denominator}",
            "off": offset,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return digest + payload.hex()

    def _consume_cursor(self, cursor: str, source_doc_id: str,
                        thr: Fraction) -> int:
        try:
            digest = cursor[:16]
            payload = bytes.fromhex(cursor[16:])
            if hashlib.sha256(payload).hexdigest()[:16] != digest:
                raise ValueError
            token = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CursorError("malformed or tampered cursor") from exc
        if token.get("id") != self.identity:
            raise CursorError("cursor belongs to a different catalog")
        if token.get("src") != source_doc_id:
            raise CursorError("cursor was issued for a different source doc")
        if token.get("th") != f"{thr.numerator}/{thr.denominator}":
            raise CursorError("cursor was issued for a different threshold")
        offset = token.get("off")
        if not isinstance(offset, int) or offset < 0:
            raise CursorError("cursor position invalid")
        return offset

    # ------------------------------------------------------------------ #
    # Document listing (diagnostics)
    # ------------------------------------------------------------------ #
    def docs_page(self, page_size: int, cursor: Optional[str] = None
                  ) -> Tuple[List[dict], Optional[str]]:
        """Page through per-document diagnostics; short documents are
        explicitly flagged, never silently treated as zero-similarity."""
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        if cursor is None:
            offset = 0
        else:
            offset = self._consume_docs_cursor(cursor)
        rows: List[dict] = []
        end = min(offset + page_size, self.n_docs)
        for idx in range(offset, end):
            rec = self._doc_record(idx)
            rows.append({
                "doc_id": self._doc_id_bytes_at(rec).decode("utf-8"),
                "shingle_count": rec["shingle_count"],
                "text_len_codepoints": rec["text_len"],
                "comparable": not bool(rec["flags"] & sfmt.FLAG_NO_SHINGLES),
                "flags": rec["flags"],
            })
        next_cursor = None
        if end < self.n_docs:
            payload = json.dumps({"id": self.identity, "off": end},
                                 sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()[:16]
            next_cursor = digest + payload.hex()
        return rows, next_cursor

    def _consume_docs_cursor(self, cursor: str) -> int:
        try:
            digest = cursor[:16]
            payload = bytes.fromhex(cursor[16:])
            if hashlib.sha256(payload).hexdigest()[:16] != digest:
                raise ValueError
            token = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CursorError("malformed or tampered cursor") from exc
        if token.get("id") != self.identity:
            raise CursorError("cursor belongs to a different catalog")
        offset = token.get("off")
        if not isinstance(offset, int) or offset < 0:
            raise CursorError("cursor position invalid")
        return offset

    # ------------------------------------------------------------------ #
    # Deep validation and statistics
    # ------------------------------------------------------------------ #
    def validate(self) -> Dict[str, object]:
        """Re-check everything, including posting/forward sortedness.

        Opening already re-verified every file hash and all directory
        geometry; this additionally streams every posting list and every
        forward relation to prove ordering and referential bounds, and
        confirms doc_id uniqueness.  Returns a report of real counts.
        """
        report: Dict[str, object] = {}
        # Posting lists: strictly increasing doc indexes, in range, tf >= 1.
        entries = 0
        for qid in range(self.n_qgrams):
            _bo, _bl, _r, post_off, post_count = self._qgramdir_record(qid)
            prev = -1
            for i in range(post_off, post_off + post_count):
                d, tf = self._posting_entry(i)
                if d <= prev:
                    raise IndexCorruptError("posting list not sorted")
                if d >= self.n_docs:
                    raise IndexCorruptError("posting doc index out of range")
                if tf < 1:
                    raise IndexCorruptError("posting term frequency < 1")
                prev = d
                entries += 1
        report["posting_entries_checked"] = entries
        if entries != self.manifest["relations"]["deduped"]:
            raise IndexCorruptError("posting entry total mismatch")

        # Forward relations: strictly increasing qgram ids per document.
        fwd = 0
        seen_ids = set()
        for d in range(self.n_docs):
            rec = self._doc_record(d)
            doc_id = self._doc_id_bytes_at(rec)
            if doc_id in seen_ids:
                raise IndexCorruptError("duplicate doc_id in catalog")
            seen_ids.add(doc_id)
            prev = -1
            for qid in self._iter_doc_qgrams(rec):
                if qid <= prev:
                    raise IndexCorruptError("doc qgram ids not sorted")
                if qid >= self.n_qgrams:
                    raise IndexCorruptError("qgram id out of range")
                prev = qid
                fwd += 1
        report["forward_relations_checked"] = fwd
        report["unique_doc_ids"] = len(seen_ids)
        if fwd != entries:
            raise IndexCorruptError(
                "forward relations and postings disagree")
        report["documents"] = self.n_docs
        report["qgrams"] = self.n_qgrams
        report["ok"] = True
        return report

    def stats(self) -> Dict[str, object]:
        out = {
            "manifest": self.manifest,
            "pagers": {
                "qgramdir": self._qgramdir_pager.stats(),
                "postings": self._postings_pager.stats(),
                "docs": self._docs_pager.stats(),
                "docqgrams": self._docqgrams_pager.stats(),
            },
        }
        if self._last_query_stats is not None:
            out["last_query"] = self._last_query_stats
        if hasattr(self, "_last_explain_stats"):
            out["last_explain"] = self._last_explain_stats
        return out

"""Read side of a desensitisation release package.

A :class:`ReleasePackage` opens without any ancestor files: the package is
self-contained.  All random access goes through bounded
:class:`~diskfm.pager.FilePager` instances and fixed-width records, so a
small one-document range query touches only that document's docs record, the
relevant lineage shard pages and width-block checkpoints -- it never replays
rounds, scans the corpus body or reads other documents' lineage.

Two migration directions are supported, both over non-empty half-open
codepoint intervals of one document and both reporting codepoint *and*
UTF-8 byte boundaries on each side:

* :meth:`forward_map` -- baseline original -> selected release.  Surviving
  baseline fragments map to their release landing intervals; every baseline
  sub-interval that lost its counterpart is reported with baseline
  coordinates only (even when a request endpoint sits inside a deleted run).
* :meth:`traceback` -- release -> baseline/generated provenance.  Each
  released fragment is classified as baseline original (with baseline
  coordinates) or generated (introducing round, edit and relative interval
  inside that replacement).  Generated fragments never receive fabricated
  baseline coordinates; later truncation keeps the introducing round and the
  original relative offsets; adjacent segments from different sources stay
  distinct even when the text happens to be identical.

:meth:`export_jsonl` streams ``{"doc_id","text"}`` JSONL with bounded write
blocks (streaming JSON escaping), and :meth:`validate` independently
re-checks every file, structure, ledger, interval geometry and the package
identity -- a corrupt or empty mapping is an error, never a silent result.
"""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple

from . import format as fmt
from . import pubformat as pfmt
from .pager import FilePager


class ReleasePackage:
    def __init__(self, package_dir: str, max_pages: int = 16,
                 page_records: int = 4096, verify_on_open: bool = False):
        """Open a package without scanning its contents.

        By default opening checks the OPEN marker, manifest, format version
        and each declared file's presence and declared length, but does not
        hash or read any body/lineage bytes -- a small one-document query
        must not pull the whole corpus.  Pass ``verify_on_open=True`` (used
        by the standalone ``verify`` command's deep path) or call
        :meth:`validate` for the full content-hash and structural audit.
        """
        self.package_dir = package_dir
        pfmt.check_open_marker(package_dir)
        with open(os.path.join(package_dir, pfmt.MANIFEST_NAME),
                  encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("format_version") != pfmt.PUB_FORMAT_VERSION:
            raise pfmt.ReleaseCorruptError(
                "release manifest format version mismatch")
        if verify_on_open:
            self._verify_files()
        else:
            self._check_files_present_and_sized()
        self.n_docs = self.manifest["geometry"]["documents"]
        self.max_pages = max_pages
        self.page_records = page_records
        self._docs_pager = FilePager(
            self._path("docs.bin"), pfmt.DOCS_RECORD_SIZE,
            pfmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._docs_f = open(self._path("docs.bin"), "rb")
        self._texts_f = open(self._path("texts.bin"), "rb")
        self._lineage_f = open(self._path("lineage.bin"), "rb")
        self._width_f = open(self._path("width.bin"), "rb")
        # Bounded LRU over lineage *segments* keyed by their absolute record
        # index inside lineage.bin (the per-shard headers are skipped in the
        # index arithmetic, so shards with different offsets still page).
        self._seg_cache: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()
        self._seg_cache_cap = max_pages * 8
        self._lineage_page_hits = 0
        self._lineage_page_misses = 0
        self.stats = {
            "docs_pages": 0, "lineage_pages": 0,
            "lineage_records_read": 0, "width_checkpoint_reads": 0,
            "width_packed_reads": 0, "text_bytes_read": 0,
            "documents_touched": 0,
        }

    def _path(self, name: str) -> str:
        return os.path.join(self.package_dir, name)

    def _verify_files(self) -> None:
        for name, entry in self.manifest["files"].items():
            try:
                pfmt.verify_file(self._path(name), entry)
            except fmt.IndexCorruptError as exc:
                raise pfmt.ReleaseCorruptError(str(exc)) from exc

    def _check_files_present_and_sized(self) -> None:
        # Cheap opening guard: every declared file exists and matches its
        # declared length (a truncation or missing part is caught), without
        # streaming and hashing the whole package.
        for name, entry in self.manifest["files"].items():
            path = self._path(name)
            if not os.path.exists(path):
                raise pfmt.ReleaseCorruptError(
                    f"missing package file: {name}")
            if os.path.getsize(path) != entry["bytes"]:
                raise pfmt.ReleaseCorruptError(
                    f"{name} length disagrees with the manifest")
        for rnd in self.manifest["rounds"]:
            if not os.path.exists(self._path(rnd["ledger"])):
                raise pfmt.ReleaseCorruptError(
                    f"missing ledger: {rnd['ledger']}")

    def close(self) -> None:
        self._docs_pager.close()
        for f in (self._docs_f, self._texts_f, self._lineage_f,
                  self._width_f):
            f.close()

    def __enter__(self) -> "ReleasePackage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def identity(self) -> str:
        return self.manifest["package_identity"]

    @property
    def root_index_identity(self) -> str:
        return self.manifest["chain"]["root_index_identity"]

    # ------------------------------------------------------------------ #
    # Document table
    # ------------------------------------------------------------------ #
    _DOC_KEYS = ("text_offset", "text_bytes", "orig_cp_len", "pub_cp_len",
                 "lineage_offset", "lineage_bytes", "id_blob_offset",
                 "orig_width_block_offset", "pub_width_block_offset")

    def _doc_record(self, doc_idx: int) -> dict:
        if not 0 <= doc_idx < self.n_docs:
            raise pfmt.ReleaseCorruptError(
                f"document index {doc_idx} out of range 0..{self.n_docs - 1}")
        vals = pfmt.DOCS_RECORD.unpack(
            self._docs_pager.read_record(doc_idx))
        self.stats["docs_pages"] = self._docs_pager.page_misses
        return dict(zip(self._DOC_KEYS, vals))

    def _doc_id_at(self, doc_idx: int) -> str:
        rec = self._doc_record(doc_idx)
        blob_start = (self.n_docs * pfmt.DOCS_RECORD_SIZE
                      + rec["id_blob_offset"])
        if doc_idx + 1 < self.n_docs:
            next_off = self._doc_record(doc_idx + 1)["id_blob_offset"]
            id_len = next_off - rec["id_blob_offset"]
        else:
            total = (self.manifest["files"]["docs.bin"]["bytes"]
                     - pfmt.HEADER_SIZE
                     - self.n_docs * pfmt.DOCS_RECORD_SIZE)
            id_len = total - rec["id_blob_offset"]
        self._docs_f.seek(pfmt.HEADER_SIZE + blob_start)
        raw = self._docs_f.read(id_len)
        if len(raw) != id_len:
            raise pfmt.ReleaseCorruptError("doc id blob truncated")
        return raw.decode("utf-8")

    def find_doc(self, doc_id: str) -> Optional[int]:
        target = doc_id.encode("utf-8")
        for idx in range(self.n_docs):
            if self._doc_id_at(idx).encode("utf-8") == target:
                return idx
        return None

    def require_doc(self, doc_id: str) -> int:
        idx = self.find_doc(doc_id)
        if idx is None:
            raise KeyError(f"unknown doc_id: {doc_id}")
        return idx

    # ------------------------------------------------------------------ #
    # Paged lineage access (bounded; never whole-shard retention)
    # ------------------------------------------------------------------ #
    def _read_shard_header(self, rec: dict) -> dict:
        if rec["lineage_bytes"] < pfmt.SHARD_HEADER_SIZE:
            raise pfmt.ReleaseCorruptError("lineage shard too small")
        self._lineage_f.seek(
            pfmt.HEADER_SIZE + rec["lineage_offset"])
        head = self._lineage_f.read(pfmt.SHARD_HEADER_SIZE)
        if len(head) != pfmt.SHARD_HEADER_SIZE:
            raise pfmt.ReleaseCorruptError("lineage shard header truncated")
        count, sample_count, shard_pub_len, sample_offset, reserved = \
            pfmt.SHARD_HEADER.unpack(head)
        if reserved != 0:
            raise pfmt.ReleaseCorruptError("non-zero shard reserved field")
        expected_size = (pfmt.SHARD_HEADER_SIZE
                         + count * pfmt.SEGMENT_RECORD_SIZE
                         + sample_count * pfmt.SHARD_SAMPLE_RECORD_SIZE)
        if expected_size != rec["lineage_bytes"]:
            raise pfmt.ReleaseCorruptError(
                "lineage shard size disagrees with its records and samples")
        expected_sample_offset = (pfmt.SHARD_HEADER_SIZE
                                  + count * pfmt.SEGMENT_RECORD_SIZE)
        if sample_offset != expected_sample_offset:
            raise pfmt.ReleaseCorruptError(
                "lineage sample table offset inconsistent")
        if shard_pub_len != rec["pub_cp_len"]:
            raise pfmt.ReleaseCorruptError(
                "lineage shard published length disagrees with docs record")
        return {"segment_count": count, "sample_count": sample_count,
                "sample_offset": sample_offset}

    def _shard_segment_count(self, rec: dict) -> int:
        return self._read_shard_header(rec)["segment_count"]

    def _read_sample(self, rec: dict, sample_offset: int,
                     sample_index: int) -> Tuple[int, int]:
        pos = (pfmt.HEADER_SIZE + rec["lineage_offset"] + sample_offset
               + sample_index * pfmt.SHARD_SAMPLE_RECORD_SIZE)
        self._lineage_f.seek(pos)
        raw = self._lineage_f.read(pfmt.SHARD_SAMPLE_RECORD_SIZE)
        if len(raw) != pfmt.SHARD_SAMPLE_RECORD_SIZE:
            raise pfmt.ReleaseCorruptError("lineage sample record truncated")
        return pfmt.SHARD_SAMPLE_RECORD.unpack(raw)

    def _read_segment(self, rec: dict, seg_index: int) -> dict:
        """Read one fixed-width segment through the bounded LRU.

        Shard headers make per-shard record offsets non-uniform, so the cache
        is keyed by ``(lineage_offset, segment index)`` and every read uses an
        explicit absolute file position.
        """
        key = (rec["lineage_offset"], seg_index)
        cached = self._seg_cache.get(key)
        if cached is not None:
            self._seg_cache.move_to_end(key)
            self._lineage_page_hits += 1
            return cached
        self._lineage_page_misses += 1
        abs_pos = (pfmt.HEADER_SIZE + rec["lineage_offset"]
                   + pfmt.SHARD_HEADER_SIZE
                   + seg_index * pfmt.SEGMENT_RECORD_SIZE)
        self._lineage_f.seek(abs_pos)
        raw = self._lineage_f.read(pfmt.SEGMENT_RECORD_SIZE)
        if len(raw) != pfmt.SEGMENT_RECORD_SIZE:
            raise pfmt.ReleaseCorruptError("lineage segment record truncated")
        self.stats["lineage_records_read"] += 1
        kind, ref1, ref2, _reserved, a, b, c, d = \
            pfmt.SEGMENT_RECORD.unpack(raw)
        if c >= d or b <= a or d - c != b - a:
            raise pfmt.ReleaseCorruptError("lineage segment geometry invalid")
        if kind == pfmt.SEG_KIND_ORIG:
            seg = {"kind": "orig", "orig_lo": a, "orig_hi": b,
                   "pub_lo": c, "pub_hi": d}
        elif kind == pfmt.SEG_KIND_GEN:
            seg = {"kind": "gen", "round_ref": ref1, "edit_ref": ref2,
                   "rel_lo": a, "rel_hi": b,
                   "pub_lo": c, "pub_hi": d}
        else:
            raise pfmt.ReleaseCorruptError(f"unknown segment kind {kind}")
        self._seg_cache[key] = seg
        self._seg_cache.move_to_end(key)
        while len(self._seg_cache) > self._seg_cache_cap:
            self._seg_cache.popitem(last=False)
        return seg

    def _segment_pub_lo(self, rec: dict, seg_index: int) -> int:
        return self._read_segment(rec, seg_index)["pub_lo"]

    def _locate_pub_segment(self, rec: dict, count: int,
                            pub_offset: int) -> int:
        """Paged binary search: index of the segment covering pub_offset."""
        lo, hi = 0, count
        while lo < hi:
            mid = (lo + hi) // 2
            if self._segment_pub_lo(rec, mid) <= pub_offset:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            raise pfmt.ReleaseCorruptError(
                "published offset precedes the first lineage segment")
        seg = self._read_segment(rec, idx)
        if pub_offset >= seg["pub_hi"]:
            raise pfmt.ReleaseCorruptError(
                "lineage shard has a coverage gap at the requested offset")
        return idx

    # ------------------------------------------------------------------ #
    # UTF-8 byte boundaries via retained width blocks (bounded reads)
    # ------------------------------------------------------------------ #
    def _width_byte_offset(self, block_offset: int, cp_offset: int) -> int:
        def read_at(abs_off, n):
            self._width_f.seek(abs_off)
            data = self._width_f.read(n)
            if len(data) != n:
                raise pfmt.ReleaseCorruptError("width block read truncated")
            if abs_off != pfmt.HEADER_SIZE + block_offset:
                self.stats["width_packed_reads"] += 1
            return data
        self.stats["width_checkpoint_reads"] += 1
        return pfmt.width_byte_offset(
            lambda rel: pfmt.HEADER_SIZE + block_offset + rel,
            read_at, cp_offset)

    def _orig_bytes(self, rec: dict, cp_offset: int) -> int:
        return self._width_byte_offset(
            rec["orig_width_block_offset"], cp_offset)

    def _pub_bytes(self, rec: dict, cp_offset: int) -> int:
        return self._width_byte_offset(
            rec["pub_width_block_offset"], cp_offset)

    # ------------------------------------------------------------------ #
    # Direction 1: baseline original -> release
    # ------------------------------------------------------------------ #
    def forward_map(self, doc_id: str, start: int, end: int) -> dict:
        self._check_interval(start, end)
        doc_idx = self.require_doc(doc_id)
        rec = self._doc_record(doc_idx)
        self.stats["documents_touched"] += 1
        if end > rec["orig_cp_len"]:
            raise pfmt.ReleaseCorruptError(
                "requested baseline interval exceeds the document length")
        header = self._read_shard_header(rec)
        count = header["segment_count"]
        # Bounded entry point: binary search the sparse baseline->segment
        # sample table (O(log samples) reads), then walk only the local
        # segments until the segment's baseline start reaches `end`.  A small
        # range query therefore never scans the whole shard or corpus.
        k_start = start // pfmt.SHARD_SAMPLE_INTERVAL
        threshold, seg_idx = self._read_sample(
            rec, header["sample_offset"], k_start)
        mapped: List[dict] = []
        lost: List[dict] = []
        cursor = start
        i = seg_idx
        while i < count:
            seg = self._read_segment(rec, i)
            i += 1
            if seg["kind"] != "orig":
                continue
            o_lo, o_hi = seg["orig_lo"], seg["orig_hi"]
            if o_lo >= end:
                break
            if o_hi <= start:
                continue
            if o_lo > cursor:
                lost.append(self._lost_piece(rec, cursor, min(o_lo, end)))
            hit_lo = max(o_lo, start)
            hit_hi = min(o_hi, end)
            shift = hit_lo - o_lo
            span = hit_hi - hit_lo
            p_lo = seg["pub_lo"] + shift
            p_hi = p_lo + span
            mapped.append({
                "baseline": self._bounds(
                    rec, hit_lo, hit_hi, published=False),
                "release": self._bounds(rec, p_lo, p_hi, published=True),
            })
            cursor = max(cursor, hit_hi)
        if cursor < end:
            lost.append(self._lost_piece(rec, cursor, end))
        if not mapped and not lost:
            raise pfmt.ReleaseCorruptError(
                "forward map produced no mapping and no lost fragment; "
                "an empty result cannot represent a valid interval")
        return {"doc_id": doc_id, "mapped": mapped, "lost": lost}

    def _lost_piece(self, rec: dict, lo: int, hi: int) -> dict:
        return {"baseline": self._bounds(rec, lo, hi, published=False)}

    def _bounds(self, rec: dict, lo: int, hi: int,
                published: bool) -> dict:
        if published:
            b_lo, b_hi = self._pub_bytes(rec, lo), self._pub_bytes(rec, hi)
        else:
            b_lo, b_hi = self._orig_bytes(rec, lo), self._orig_bytes(rec, hi)
        return {"codepoints": [lo, hi], "utf8_bytes": [b_lo, b_hi]}

    # ------------------------------------------------------------------ #
    # Direction 2: release -> provenance
    # ------------------------------------------------------------------ #
    def traceback(self, doc_id: str, start: int, end: int) -> dict:
        self._check_interval(start, end)
        doc_idx = self.require_doc(doc_id)
        rec = self._doc_record(doc_idx)
        self.stats["documents_touched"] += 1
        if end > rec["pub_cp_len"]:
            raise pfmt.ReleaseCorruptError(
                "requested release interval exceeds the published length")
        count = self._shard_segment_count(rec)
        first = self._locate_pub_segment(rec, count, start)
        pieces: List[dict] = []
        cursor = start
        i = first
        while cursor < end:
            seg = self._read_segment(rec, i)
            if seg["pub_lo"] > cursor:
                raise pfmt.ReleaseCorruptError(
                    "lineage shard has a coverage gap inside the request")
            hit_lo = max(seg["pub_lo"], cursor)
            hit_hi = min(seg["pub_hi"], end)
            rel = self._bounds(rec, hit_lo, hit_hi, published=True)
            if seg["kind"] == "orig":
                shift = hit_lo - seg["pub_lo"]
                span = hit_hi - hit_lo
                base = self._bounds(
                    rec, seg["orig_lo"] + shift,
                    seg["orig_lo"] + shift + span, published=False)
                pieces.append({"source": "baseline", "release": rel,
                               "baseline": base})
            else:
                shift = hit_lo - seg["pub_lo"]
                span = hit_hi - hit_lo
                round_id = self.manifest["round_dictionary"][
                    seg["round_ref"]]
                edit_id = self.manifest["edit_dictionary"][seg["edit_ref"]]
                rel_bounds = {
                    "codepoints": [seg["rel_lo"] + shift,
                                   seg["rel_lo"] + shift + span],
                }
                pieces.append({
                    "source": "generated", "release": rel,
                    "generation": {
                        "round_id": round_id, "edit_id": edit_id,
                        "relative": rel_bounds}})
            cursor = hit_hi
            i += 1
        if not pieces:
            raise pfmt.ReleaseCorruptError(
                "traceback produced no provenance pieces")
        return {"pieces": pieces}

    @staticmethod
    def _check_interval(start: int, end: int) -> None:
        if not (isinstance(start, int) and isinstance(end, int)
                and not isinstance(start, bool) and not isinstance(end, bool)):
            raise ValueError("interval endpoints must be integers")
        if start < 0 or end <= start:
            raise ValueError(
                "interval must be a non-empty half-open codepoint range")

    # ------------------------------------------------------------------ #
    # Streaming JSONL export (bounded buffers, streaming escaping)
    # ------------------------------------------------------------------ #
    def export_jsonl(self, out: BinaryIO, block_bytes: int = 1 << 16) -> int:
        written = 0
        for doc_idx in range(self.n_docs):
            doc_id = self._doc_id_at(doc_idx)
            rec = self._doc_record(doc_idx)
            parts = [b'{"doc_id":',
                     json.dumps(doc_id, ensure_ascii=False).encode("utf-8"),
                     b',"text":"']
            for chunk in self._iter_text_chunks(rec, block_bytes):
                parts.append(self._escape_json_chunk(chunk))
            parts.append(b'"}\n')
            for piece in parts:
                out.write(piece)
                written += len(piece)
        return written

    def _iter_text_chunks(self, rec: dict, block_bytes: int
                          ) -> Iterator[bytes]:
        abs_start = pfmt.HEADER_SIZE + rec["text_offset"]
        remaining = rec["text_bytes"]
        self._texts_f.seek(abs_start)
        while remaining:
            take = min(block_bytes, remaining)
            raw = self._texts_f.read(take)
            if len(raw) != take:
                raise pfmt.ReleaseCorruptError("texts.bin truncated")
            self.stats["text_bytes_read"] += take
            remaining -= take
            yield raw

    @staticmethod
    def _escape_json_chunk(raw: bytes) -> bytes:
        # Escape on the UTF-8 byte stream: every JSON-special ASCII byte is
        # a single-byte codepoint, so replacing bytes never corrupts
        # multi-byte characters.  NUL and controls are emitted as \u00XX.
        out = bytearray()
        table = {0x22: b'\\"', 0x5C: b"\\\\", 0x0A: b"\\n",
                 0x0D: b"\\r", 0x08: b"\\b", 0x0C: b"\\f",
                 0x09: b"\\t"}
        for b in raw:
            if b in table:
                out += table[b]
            elif b < 0x20:
                out += ("\\u%04x" % b).encode("ascii")
            else:
                out.append(b)
        return bytes(out)

    def document_text(self, doc_id: str) -> str:
        doc_idx = self.require_doc(doc_id)
        rec = self._doc_record(doc_idx)
        self._texts_f.seek(pfmt.HEADER_SIZE + rec["text_offset"])
        raw = self._texts_f.read(rec["text_bytes"])
        if len(raw) != rec["text_bytes"]:
            raise pfmt.ReleaseCorruptError("texts.bin truncated")
        self.stats["text_bytes_read"] += len(raw)
        return raw.decode("utf-8")

    # ------------------------------------------------------------------ #
    # Whole-package validation (independent, structural)
    # ------------------------------------------------------------------ #
    def validate(self) -> dict:
        # Deep, independent audit: full content hashes first, then identity,
        # then every document/lineage/width/ledger structure.  This is the
        # only operation that must read the entire package.
        self._verify_files()
        report = {"documents": 0, "segments": 0, "edits": 0,
                  "rounds": len(self.manifest["rounds"])}
        if self._compute_identity() != self.manifest["package_identity"]:
            raise pfmt.ReleaseCorruptError(
                "package identity mismatch; lineage or files tampered")
        prev_id_blob_end = 0
        text_cursor = 0
        for d in range(self.n_docs):
            rec = self._doc_record(d)
            if rec["id_blob_offset"] != prev_id_blob_end:
                raise pfmt.ReleaseCorruptError(
                    "doc id blob offsets are not contiguous")
            prev_id_blob_end += self._doc_id_length(d, rec)
            if rec["text_offset"] != text_cursor:
                raise pfmt.ReleaseCorruptError(
                    "texts.bin document slices are not back-to-back")
            text_cursor += rec["text_bytes"]
            self._validate_shard(d, rec, report)
            self._validate_width_blocks(rec)
            report["documents"] += 1
        if text_cursor != (self.manifest["files"]["texts.bin"]["bytes"]
                           - pfmt.HEADER_SIZE):
            raise pfmt.ReleaseCorruptError("texts.bin size disagrees with docs")
        self._validate_ledgers(report)
        return report

    def _doc_id_length(self, doc_idx: int, rec: dict) -> int:
        if doc_idx + 1 < self.n_docs:
            return (self._doc_record(doc_idx + 1)["id_blob_offset"]
                    - rec["id_blob_offset"])
        total = (self.manifest["files"]["docs.bin"]["bytes"]
                 - pfmt.HEADER_SIZE
                 - self.n_docs * pfmt.DOCS_RECORD_SIZE)
        return total - rec["id_blob_offset"]

    def _validate_shard(self, doc_idx: int, rec: dict, report: dict) -> None:
        header = self._read_shard_header(rec)
        count = header["segment_count"]
        if count == 0 and rec["pub_cp_len"] > 0:
            raise pfmt.ReleaseCorruptError(
                f"document {doc_idx} has text but no lineage coverage")
        pub_cursor = 0
        last_orig_hi = 0
        orig_segment_indices: List[int] = []
        for i in range(count):
            seg = self._read_segment(rec, i)
            if seg["pub_lo"] != pub_cursor:
                raise pfmt.ReleaseCorruptError(
                    "lineage segments do not tile the published text")
            pub_cursor = seg["pub_hi"]
            if seg["kind"] == "orig":
                if seg["orig_lo"] < last_orig_hi:
                    raise pfmt.ReleaseCorruptError(
                        "baseline lineage intervals overlap or go backwards")
                last_orig_hi = max(last_orig_hi, seg["orig_hi"])
                if seg["orig_hi"] > rec["orig_cp_len"]:
                    raise pfmt.ReleaseCorruptError(
                        "baseline interval exceeds the source document length")
                orig_segment_indices.append(i)
            else:
                if not (0 <= seg["round_ref"]
                        < len(self.manifest["round_dictionary"])):
                    raise pfmt.ReleaseCorruptError(
                        "generated segment round reference out of range")
                if not (0 <= seg["edit_ref"]
                        < len(self.manifest["edit_dictionary"])):
                    raise pfmt.ReleaseCorruptError(
                        "generated segment edit reference out of range")
                if seg["rel_lo"] < 0 or seg["rel_hi"] <= seg["rel_lo"]:
                    raise pfmt.ReleaseCorruptError(
                        "generated segment relative interval invalid")
            report["segments"] += 1
        self._validate_samples(rec, header, orig_segment_indices)
        if pub_cursor != rec["pub_cp_len"]:
            raise pfmt.ReleaseCorruptError(
                "lineage published coverage disagrees with docs record")
        if last_orig_hi > rec["orig_cp_len"]:
            raise pfmt.ReleaseCorruptError(
                "baseline coverage exceeds the source document length")

    def _validate_samples(self, rec: dict, header: dict,
                          orig_segment_indices: List[int]) -> None:
        interval = pfmt.SHARD_SAMPLE_INTERVAL
        n_expected = rec["orig_cp_len"] // interval + 1
        if header["sample_count"] != n_expected:
            raise pfmt.ReleaseCorruptError(
                "lineage sample table count inconsistent with baseline length")
        prev_seg = 0
        for k in range(n_expected):
            threshold, seg_index = self._read_sample(
                rec, header["sample_offset"], k)
            if threshold != k * interval:
                raise pfmt.ReleaseCorruptError(
                    "lineage sample threshold out of sequence")
            if seg_index < prev_seg or seg_index > header["segment_count"]:
                raise pfmt.ReleaseCorruptError(
                    "lineage sample segment index non-monotonic or out of "
                    "range")
            # The pointed segment must be the first original segment whose
            # span extends past the threshold (or a post-gap sentinel index).
            if seg_index < header["segment_count"]:
                pointed = self._read_segment(rec, seg_index)
                if pointed["kind"] == "orig":
                    if pointed["orig_hi"] <= threshold:
                        raise pfmt.ReleaseCorruptError(
                            "lineage sample points at a segment ending before "
                            "its threshold")
                    if (seg_index > 0 and seg_index - 1
                            in orig_segment_indices):
                        earlier = self._read_segment(rec, seg_index - 1)
                        if (earlier["kind"] == "orig"
                                and earlier["orig_hi"] > threshold):
                            raise pfmt.ReleaseCorruptError(
                                "lineage sample skips an eligible segment")
            prev_seg = seg_index

    def _validate_width_blocks(self, rec: dict) -> None:
        for key, cp_len in (
                ("orig_width_block_offset", rec["orig_cp_len"]),
                ("pub_width_block_offset", rec["pub_cp_len"])):
            off = rec[key]
            self._width_f.seek(pfmt.HEADER_SIZE + off)
            head = self._width_f.read(pfmt.WIDTH_BLOCK_HEADER_SIZE)
            if len(head) != pfmt.WIDTH_BLOCK_HEADER_SIZE:
                raise pfmt.ReleaseCorruptError("width block header missing")
            stored_cp, packed_bytes, cp_count = \
                pfmt.WIDTH_BLOCK_HEADER.unpack(head)
            if stored_cp != cp_len:
                raise pfmt.ReleaseCorruptError(
                    "width block codepoint count disagrees with docs record")
            expected_cp_count = (
                cp_len // pfmt.WIDTH_CHECKPOINT_INTERVAL + 1)
            if cp_count != expected_cp_count:
                raise pfmt.ReleaseCorruptError(
                    "width block checkpoint count inconsistent")
            # End-of-block byte offset must equal the stored text byte length.
            end_off = self._width_byte_offset(off, cp_len)
            if key == "pub_width_block_offset":
                if end_off != rec["text_bytes"]:
                    raise pfmt.ReleaseCorruptError(
                        "released width totals disagree with texts.bin slice")

    def _validate_ledgers(self, report: dict) -> None:
        seen_edits = set()
        for entry in self.manifest["rounds"]:
            path = self._path(entry["ledger"])
            if not os.path.exists(path):
                raise pfmt.ReleaseCorruptError(
                    f"missing ledger {entry['ledger']}")
            digest = hashlib_sha256_file(path)
            rows = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    key = (rec["round_id"], rec["edit_id"])
                    if key in seen_edits:
                        raise pfmt.ReleaseCorruptError(
                            "duplicate edit recorded in ledgers")
                    seen_edits.add(key)
                    if rec["consumed_end"] <= rec["consumed_start"]:
                        raise pfmt.ReleaseCorruptError(
                            "ledger consumed interval must be non-empty")
                    if rec["generated_codepoints"] < 0:
                        raise pfmt.ReleaseCorruptError(
                            "ledger generated length must be non-negative")
                    rows += 1
            if rows != entry["edits"]:
                raise pfmt.ReleaseCorruptError(
                    "ledger row count disagrees with the round manifest")
            report["edits"] += rows

    def _compute_identity(self) -> str:
        view = {k: v for k, v in self.manifest.items()
                if k != "package_identity"}
        return pfmt.compute_package_identity(view)

    def runtime_stats(self) -> dict:
        s = dict(self.stats)
        s["docs_pager"] = self._docs_pager.stats()
        s["lineage_segment_cache"] = {
            "hits": self._lineage_page_hits,
            "misses": self._lineage_page_misses,
            "resident_segments": len(self._seg_cache),
            "capacity": self._seg_cache_cap,
        }
        return s


def hashlib_sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

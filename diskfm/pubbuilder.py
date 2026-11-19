"""External-memory construction of a desensitisation release package.

One round = one human-approved, unordered JSONL edit list applied to one
sealed input (a diskfm index or a previous release package, both opened
read-only through :mod:`diskfm.pubsource`).  The pipeline never searches for
matches itself: it executes the listed edits and nothing else.

Stages
======

1. **Parse + reject whole rounds** (``_ingest_edit_list``): the manifest
   header is checked against the direct input identity; every record is
   validated (fields, non-empty half-open codepoint intervals inside the
   document's current length, no lone surrogates in the replacement, per
   record / identifier caps) and spilled through two bounded external sorts
   -- one keyed by edit id (duplicate ids reject the round) and one by
   ``(doc_index, lo)`` (overlaps and ordering are decided during the merge,
   adjacency is legal).  The input is read strictly once and no output file
   is created before every record has been accepted.
2. **Streaming application + lineage composition** (``_apply_round``): the
   sorted edits are merged with the input's per-document codepoint and
   lineage streams in one forward pass per document.  Baseline-original
   segments survive via equal-length interval translation; generated
   segments keep their introducing round/edit and *relative* offsets and are
   split or trimmed when a later edit truncates them; holes left by deletions
   are preserved as gaps (adjacent segments from different sources are never
   coalesced, and an identical replacement string is still tagged generated).
3. **Bounded body / width / lineage output**: released UTF-8 bytes stream to
   ``texts.bin`` in bounded write blocks; per-codepoint widths of both the
   baseline and released text go to packed 2-bit width blocks with u64
   checkpoints; lineage is one fixed-width segment record per compressed run.
4. **Ledger + identity + atomic publish**: a body-free canonical ledger
   records round/edit/document/consumed interval/generated length; the
   package identity binds the chain root, every round's logical edit digest,
   the identifier dictionaries, geometry and all file hashes.  Everything is
   built inside ``<target>.staging``; on any error the staging tree is
   removed and an existing target is never touched.  OPEN is fsynced last and
   the directory is atomically renamed only after full validation.

Budget
======

Every large structure is streamed or paged: a whole document is never
decoded at once (blocked incremental UTF-8 reader), a hot document's edit
intervals are not resident (external sort + bounded merge heap), lineage is
never one record per codepoint (long unchanged copies stay one segment), and
all random access uses :class:`~diskfm.pager.FilePager` with a fixed page
cap.  Explicit caps reject oversized edit records and identifiers instead of
letting them dominate the working set.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Tuple

from . import format as fmt
from . import pubformat as pfmt
from .pubsource import IndexSource, ReleaseSource, open_source
from .vsort import VarExternalSorter, VarCodec, VarSortResult

# Explicit, configurable caps (round rejected when crossed).
DEFAULT_MAX_EDIT_BYTES = 1 << 20       # one JSONL record
DEFAULT_MAX_REPLACEMENT_CP = 1 << 24   # codepoints of one replacement
DEFAULT_MAX_INTERVAL_CP = 1 << 31
DEFAULT_MAX_ID_BYTES = 4096
DEFAULT_MAX_EDITS = 1 << 31
DEFAULT_CHUNK_ITEMS = 20000
DEFAULT_CHUNK_BYTES = 1 << 22
DEFAULT_BLOCK_BYTES = 1 << 16
DEFAULT_WRITE_BYTES = 1 << 16

# Spill record framing inside the external sorts (all little-endian).
# edit-id sort:  u32 doc_idx, u32 lo, u32 hi, u32 edit_id_len, id_bytes,
#                u32 repl_len_cp, u32 repl_utf8_len, repl_utf8_bytes
# interval sort: same layout (ordered by doc_idx, lo during the merge).
_TAIL_U32 = struct.Struct("<IIIII")  # doc_idx, lo, hi, repl_cp, repl_utf8


class EditRecordCodec(VarCodec):
    """Framed edit record; replacement bytes travel only in temp spills."""

    def encode(self, record) -> bytes:
        doc_idx, lo, hi, edit_id, repl_cp, repl_bytes = record
        idb = edit_id.encode("utf-8")
        return (_TAIL_U32.pack(doc_idx, lo, hi, repl_cp, len(repl_bytes))
                + struct.pack("<I", len(idb)) + idb + repl_bytes)

    def decode(self, payload: bytes):
        if len(payload) < _TAIL_U32.size + 4:
            raise pfmt.PublishRejected("malformed spilled edit record")
        doc_idx, lo, hi, repl_cp, repl_utf8_len = _TAIL_U32.unpack(
            payload[: _TAIL_U32.size])
        at = _TAIL_U32.size
        (id_len,) = struct.unpack("<I", payload[at:at + 4])
        at += 4
        if len(payload) < at + id_len + repl_utf8_len:
            raise pfmt.PublishRejected("truncated spilled edit record")
        edit_id = payload[at:at + id_len].decode("utf-8")
        at += id_len
        repl_bytes = payload[at:at + repl_utf8_len]
        return (doc_idx, lo, hi, edit_id, repl_cp, repl_bytes)


def _id_sort_key(record):
    return (record[3], record[0], record[1])


def _interval_sort_key(record):
    return (record[0], record[1], record[3])


class WidthBlockWriter:
    """Builds one packed 2-bit width block with u64 byte checkpoints.

    Fed one width (1..4) at a time; flushes to the file on close.  The byte
    buffer holds at most one checkpoint interval (64 codepoints -> 16 bytes),
    so RAM is bounded independently of document length.
    """

    def __init__(self, f: BinaryIO):
        self._f = f
        self._cp_len = 0
        self._byte_total = 0
        self._checkpoints: List[int] = []
        self._packed = bytearray()
        self._bitbuf = 0
        self._bitbuf_len = 0

    def add(self, utf8_width: int) -> None:
        if not 1 <= utf8_width <= 4:
            raise pfmt.PublishRejected("illegal UTF-8 width in source")
        if self._cp_len % pfmt.WIDTH_CHECKPOINT_INTERVAL == 0:
            self._checkpoints.append(self._byte_total)
        self._bitbuf |= (utf8_width - 1) << self._bitbuf_len
        self._bitbuf_len += 2
        self._byte_total += utf8_width
        self._cp_len += 1
        if self._bitbuf_len >= 8:
            self._packed.append(self._bitbuf & 0xFF)
            self._bitbuf >>= 8
            self._bitbuf_len -= 8

    def finish(self) -> Tuple[int, int]:
        """Write header + checkpoints + packed stream; return (offset,len)."""
        offset = self._f.tell()
        # A terminal checkpoint at a multiple of the interval lets an
        # end-of-block offset resolve directly: checkpoint count is always
        # cp_len // interval + 1 (including the single zero checkpoint of an
        # empty block).
        if self._cp_len == 0:
            self._checkpoints.append(0)
        elif self._cp_len % pfmt.WIDTH_CHECKPOINT_INTERVAL == 0:
            self._checkpoints.append(self._byte_total)
        if self._bitbuf_len:
            self._packed.append(self._bitbuf & 0xFF)
            self._bitbuf = 0
            self._bitbuf_len = 0
        self._f.write(pfmt.WIDTH_BLOCK_HEADER.pack(
            self._cp_len, len(self._packed), len(self._checkpoints)))
        for cp in self._checkpoints:
            self._f.write(struct.pack("<Q", cp))
        self._f.write(self._packed)
        return offset, self._f.tell() - offset


@dataclass
class PublishStats:
    documents: int = 0
    edited_documents: int = 0
    deleted_documents: int = 0
    empty_output_documents: int = 0
    edits_applied: int = 0
    segments_written: int = 0
    spill_chunks: int = 0
    spill_bytes: int = 0
    source_bytes_read: int = 0
    texts_bytes_written: int = 0
    peak_rss_kb: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _peak_rss_kb() -> Optional[int]:
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


class ReleaseBuilder:
    """Apply one approved edit round and publish a self-contained package."""

    def __init__(self, source_path: str, edit_list_path: str,
                 target_dir: str, *,
                 chunk_items: int = DEFAULT_CHUNK_ITEMS,
                 chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                 block_bytes: int = DEFAULT_BLOCK_BYTES,
                 write_bytes: int = DEFAULT_WRITE_BYTES,
                 max_edit_bytes: int = DEFAULT_MAX_EDIT_BYTES,
                 max_replacement_cp: int = DEFAULT_MAX_REPLACEMENT_CP,
                 max_interval_cp: int = DEFAULT_MAX_INTERVAL_CP,
                 max_id_bytes: int = DEFAULT_MAX_ID_BYTES,
                 max_edits: int = DEFAULT_MAX_EDITS,
                 max_pages: int = 16, page_records: int = 4096):
        if os.path.exists(target_dir):
            raise FileExistsError(
                f"release target already exists: {target_dir}")
        for name, val in (("chunk_items", chunk_items),
                          ("chunk_bytes", chunk_bytes),
                          ("block_bytes", block_bytes),
                          ("write_bytes", write_bytes),
                          ("max_edit_bytes", max_edit_bytes),
                          ("max_replacement_cp", max_replacement_cp),
                          ("max_interval_cp", max_interval_cp),
                          ("max_id_bytes", max_id_bytes),
                          ("max_edits", max_edits)):
            if not isinstance(val, int) or isinstance(val, bool) or val < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.source_path = source_path
        self.edit_list_path = edit_list_path
        self.target_dir = target_dir
        self.staging = target_dir + pfmt.STAGING_SUFFIX
        if os.path.exists(self.staging):
            shutil.rmtree(self.staging)
        os.makedirs(self.staging)
        os.makedirs(os.path.join(self.staging, pfmt.LEDGER_DIR))
        self.chunk_items = chunk_items
        self.chunk_bytes = chunk_bytes
        self.block_bytes = block_bytes
        self.write_bytes = write_bytes
        self.max_edit_bytes = max_edit_bytes
        self.max_replacement_cp = max_replacement_cp
        self.max_interval_cp = max_interval_cp
        self.max_id_bytes = max_id_bytes
        self.max_edits = max_edits
        self.max_pages = max_pages
        self.page_records = page_records
        self.stats = PublishStats()
        self._built = False
        self.source = open_source(
            source_path, block_bytes=block_bytes, max_pages=max_pages,
            page_records=page_records)

    # ------------------------------------------------------------------ #
    # Stage 1: parse, validate and externally sort the edit list
    # ------------------------------------------------------------------ #
    def _read_header(self) -> Tuple[str, str]:
        with open(self.edit_list_path, "rb") as f:
            first = f.readline()
        if not first:
            raise pfmt.PublishRejected("edit list is empty: missing header")
        try:
            header = json.loads(first.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise pfmt.PublishRejected(
                "edit list header is not valid JSON") from exc
        if not isinstance(header, dict):
            raise pfmt.PublishRejected("edit list header must be an object")
        expect = header.get("expected_input_identity")
        round_id = header.get("round_id")
        if not isinstance(expect, str) or not expect:
            raise pfmt.PublishRejected(
                "header.expected_input_identity must be a non-empty string")
        if not isinstance(round_id, str) or not round_id:
            raise pfmt.PublishRejected(
                "header.round_id must be a non-empty string")
        if len(round_id.encode("utf-8")) > self.max_id_bytes:
            raise pfmt.PublishRejected("round_id exceeds identifier cap")
        if expect != self.source.input_identity:
            raise pfmt.PublishRejected(
                "expected_input_identity does not match the direct input")
        return expect, round_id

    @staticmethod
    def _canonical_record_hash(rec: dict) -> bytes:
        payload = json.dumps(
            rec, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(
            struct.pack("<Q", len(payload)) + payload).digest()

    def _ingest_edit_list(self, round_id: str
                          ) -> Tuple[VarSortResult, VarSortResult,
                                     str, int, Dict[str, int]]:
        n_docs = self.source.doc_count()
        id_sorter = VarExternalSorter(
            self.staging, EditRecordCodec(),
            chunk_items=self.chunk_items, chunk_bytes=self.chunk_bytes,
            key=_id_sort_key)
        iv_sorter = VarExternalSorter(
            self.staging, EditRecordCodec(),
            chunk_items=self.chunk_items, chunk_bytes=self.chunk_bytes,
            key=_interval_sort_key)
        # doc_id -> index, built lazily from the (bounded-paged) source.
        doc_index_cache: Dict[str, int] = {}
        edit_ids: Dict[str, None] = {}
        rounds_used = self._prior_round_ids()
        if round_id in rounds_used:
            raise pfmt.PublishRejected(
                "round_id already used somewhere on the release chain")
        # Logical-content digest of the round.  The header (bound input
        # identity + round id) enters directly; the per-record contribution
        # is the XOR of each canonical record's hash, so the value is
        # independent of record order while still sensitive to any field.
        header_hash = hashlib.sha256(
            json.dumps({"expected_input_identity":
                        self.source.input_identity, "round_id": round_id},
                       sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).digest()
        record_xor = bytearray(32)
        record_count = 0
        with open(self.edit_list_path, "rb") as f:
            f.readline()  # header already validated
            for line_no, raw_line in enumerate(f, start=2):
                if not raw_line.strip():
                    continue  # blank lines permitted between records
                if len(raw_line) > self.max_edit_bytes:
                    raise pfmt.PublishRejected(
                        f"edit record on line {line_no} exceeds record cap")
                try:
                    rec = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise pfmt.PublishRejected(
                        f"edit record on line {line_no} is not valid JSON"
                    ) from exc
                self._validate_record_shape(rec)
                edit_id = rec["edit_id"]
                doc_id = rec["doc_id"]
                lo, hi = rec["start"], rec["end"]
                replacement = rec["replacement"]
                if len(edit_id.encode("utf-8")) > self.max_id_bytes:
                    raise pfmt.PublishRejected(
                        f"edit_id on line {line_no} exceeds identifier cap")
                if len(doc_id.encode("utf-8")) > self.max_id_bytes:
                    raise pfmt.PublishRejected(
                        f"doc_id on line {line_no} exceeds identifier cap")
                if edit_id in edit_ids:
                    raise pfmt.PublishRejected(
                        f"duplicate edit_id on line {line_no}; round rejected")
                edit_ids[edit_id] = None
                if edit_id == round_id:
                    raise pfmt.PublishRejected(
                        "edit_id must not equal the round identifier")
                fmt.validate_text_str(edit_id)
                fmt.validate_text_str(doc_id)
                fmt.validate_text_str(replacement)
                doc_idx = self._resolve_doc_index(
                    doc_id, doc_index_cache)
                if doc_idx is None:
                    raise pfmt.PublishRejected(
                        f"unknown doc_id on line {line_no}; round rejected")
                cur_len = self._current_doc_length(doc_idx)
                if not (isinstance(lo, int) and isinstance(hi, int)
                        and not isinstance(lo, bool)
                        and not isinstance(hi, bool)):
                    raise pfmt.PublishRejected(
                        f"line {line_no}: interval endpoints must be integers")
                if lo >= hi:
                    raise pfmt.PublishRejected(
                        f"line {line_no}: interval must be non-empty and "
                        "half-open (start < end)")
                if lo < 0 or hi > cur_len or hi - lo > self.max_interval_cp:
                    raise pfmt.PublishRejected(
                        f"line {line_no}: interval {lo},{hi} out of bounds "
                        f"for document length {cur_len}")
                repl_cp = len(replacement)
                if repl_cp > self.max_replacement_cp:
                    raise pfmt.PublishRejected(
                        f"line {line_no}: replacement exceeds length cap")
                repl_bytes = replacement.encode("utf-8")
                record_count += 1
                if record_count > self.max_edits:
                    raise pfmt.PublishRejected("edit count exceeds cap")
                spill = (doc_idx, lo, hi, edit_id, repl_cp, repl_bytes)
                id_sorter.add(spill)
                iv_sorter.add(spill)
                # Order-free per-record contribution (XOR over 32 bytes).
                h = self._canonical_record_hash(rec)
                for i in range(32):
                    record_xor[i] ^= h[i]
        if record_count == 0:
            raise pfmt.PublishRejected(
                "edit list contains no edit records; round rejected")
        logical = hashlib.sha256()
        logical.update(header_hash)
        logical.update(struct.pack("<Q", record_count))
        logical.update(bytes(record_xor))
        edit_digest = logical.hexdigest()
        id_result = id_sorter.finalize()
        iv_result = iv_sorter.finalize()
        self.stats.spill_chunks = (id_sorter.chunks_spilled
                                   + iv_sorter.chunks_spilled)
        self.stats.spill_bytes = (id_sorter.bytes_spilled
                                  + iv_sorter.bytes_spilled)
        self._check_duplicate_ids(id_result)
        self._check_overlaps(iv_result, n_docs)
        self._round_edit_ids = sorted(edit_ids)
        return (id_result, iv_result, edit_digest,
                record_count, doc_index_cache)

    def _validate_record_shape(self, rec) -> None:
        if not isinstance(rec, dict):
            raise pfmt.PublishRejected("edit record must be a JSON object")
        required = {"edit_id", "doc_id", "start", "end", "replacement"}
        if set(rec.keys()) != required:
            raise pfmt.PublishRejected(
                "edit record must have exactly the keys "
                "edit_id, doc_id, start, end, replacement")
        for key in ("edit_id", "doc_id", "replacement"):
            if not isinstance(rec[key], str):
                raise pfmt.PublishRejected(
                    f"edit record field {key} must be a string")
        for key in ("start", "end"):
            if not isinstance(rec[key], int) or isinstance(rec[key], bool):
                raise pfmt.PublishRejected(
                    f"edit record field {key} must be an integer")

    def _resolve_doc_index(self, doc_id: str,
                           cache: Dict[str, int]) -> Optional[int]:
        if doc_id in cache:
            return cache[doc_id]
        n = self.source.doc_count()
        for idx in range(n):
            if self.source.doc_id(idx) == doc_id:
                cache[doc_id] = idx
                return idx
        return None

    def _current_doc_length(self, doc_idx: int) -> int:
        if isinstance(self.source, IndexSource):
            return self.source._idx._doc_record(doc_idx)["t_len"]
        return self.source._doc_record(doc_idx)["pub_cp_len"]

    def _prior_round_ids(self) -> Dict[str, None]:
        if isinstance(self.source, IndexSource):
            return {}
        return {entry["round_id"]: None
                for entry in self.source.manifest["rounds"]}

    def _check_duplicate_ids(self, id_result: VarSortResult) -> None:
        prev = None
        for rec in id_result:
            if prev is not None and rec[3] == prev[3]:
                raise pfmt.PublishRejected(
                    "duplicate edit_id detected in sorted spill; "
                    "round rejected")
            prev = rec

    def _check_overlaps(self, iv_result: VarSortResult,
                        n_docs: int) -> None:
        prev_doc = -1
        prev_hi = 0
        for rec in iv_result:
            doc_idx, lo, hi = rec[0], rec[1], rec[2]
            if doc_idx != prev_doc:
                prev_doc = doc_idx
                prev_hi = 0
            if lo < prev_hi:
                raise pfmt.PublishRejected(
                    f"overlapping intervals in document index {doc_idx}; "
                    "adjacent intervals are legal, overlaps reject the round")
            prev_hi = hi

    # ------------------------------------------------------------------ #
    # Stage 2/3: streaming application, lineage composition, file output
    # ------------------------------------------------------------------ #
    def _apply_round(self, iv_result: VarSortResult,
                     round_id: str) -> Tuple[List[dict], List[str],
                                             List[str], dict]:
        texts_path = os.path.join(self.staging, "texts.bin")
        docs_path = os.path.join(self.staging, "docs.bin")
        width_path = os.path.join(self.staging, "width.bin")
        lineage_path = os.path.join(self.staging, "lineage.bin")
        prior_rounds = 0 if isinstance(self.source, IndexSource) \
            else len(self.source.manifest["rounds"])
        serial = prior_rounds + 1
        ledger_rel = os.path.join(
            pfmt.LEDGER_DIR, f"{serial:06d}.jsonl")
        ledger_path = os.path.join(self.staging, ledger_rel)
        # Carry every prior body-free ledger so the package stays
        # self-contained through the whole chain.
        if isinstance(self.source, ReleaseSource):
            for entry in self.source.manifest["rounds"]:
                src = os.path.join(self.source.package_dir, entry["ledger"])
                dst = os.path.join(self.staging, entry["ledger"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
        round_ids: List[str] = list(self._prior_round_ids().keys())
        round_ids.append(round_id)
        round_dict: Dict[str, int] = {rid: i for i, rid in enumerate(round_ids)}
        # The edit dictionary is chain-wide and index-stable: inherit every
        # prior edit id *at its existing list position*, then append this
        # round's new ids in sorted order.  Prior lineage segment indices
        # therefore stay valid and the dictionary stays order-independent.
        if isinstance(self.source, ReleaseSource):
            edit_id_list = list(self.source.manifest["edit_dictionary"])
        else:
            edit_id_list = []
        edit_dict: Dict[str, int] = {
            eid: i for i, eid in enumerate(edit_id_list)}
        for eid in sorted(self._round_edit_ids):
            if eid not in edit_dict:
                edit_dict[eid] = len(edit_id_list)
                edit_id_list.append(eid)
        # Translation tables for inherited lineage segments (package input).
        if isinstance(self.source, ReleaseSource):
            old_edit_by_index = list(
                self.source.manifest["edit_dictionary"])
            old_round_by_index = list(
                self.source.manifest["round_dictionary"])
        else:
            old_edit_by_index = None
            old_round_by_index = None
        self._new_edit_index = edit_dict
        self._new_round_index = round_dict
        self._current_round_id = round_id
        docs_records: List[Tuple[int, ...]] = []
        id_blob = bytearray()
        ledger_rows: List[dict] = []
        self.stats.documents = self.source.doc_count()
        with open(texts_path, "wb") as texts_f, \
                open(docs_path, "wb") as docs_f, \
                open(width_path, "wb") as width_f, \
                open(lineage_path, "wb") as lin_f, \
                open(ledger_path, "w", encoding="utf-8") as ledger_f:
            pfmt.write_header(texts_f, pfmt.KIND_TEXTS)
            pfmt.write_header(docs_f, pfmt.KIND_DOCS)
            pfmt.write_header(width_f, pfmt.KIND_WIDTH)
            pfmt.write_header(lin_f, pfmt.KIND_LINEAGE)
            iv_iter = iter(iv_result)
            next_edit = next(iv_iter, None)
            for doc_idx, doc_id, chars, segments, orig_widths in \
                    self.source.iter_documents():
                doc_edits = []
                while next_edit is not None and next_edit[0] == doc_idx:
                    _d, lo, hi, edit_id, _rc, repl_bytes = next_edit
                    doc_edits.append((lo, hi, edit_id, repl_bytes))
                    next_edit = next(iv_iter, None)
                if next_edit is not None and next_edit[0] < doc_idx:
                    raise pfmt.PublishRejected(
                        "edit stream out of document order")
                docs_records, id_blob = self._process_document(
                    doc_idx, doc_id, chars, segments, orig_widths,
                    doc_edits, round_id, round_dict, edit_dict,
                    texts_f, width_f, lin_f, ledger_f, ledger_rows,
                    docs_records, id_blob,
                    old_edit_by_index, old_round_by_index)
            if next_edit is not None:
                raise pfmt.PublishRejected(
                    "edit references a document beyond the input set")
            # Fixed-width docs records first, then the id blob.
            for packed in docs_records:
                docs_f.write(packed)
            docs_f.write(bytes(id_blob))
        return ledger_rows, round_ids, edit_id_list, {
            "texts": os.path.getsize(texts_path),
            "docs": os.path.getsize(docs_path),
            "width": os.path.getsize(width_path),
            "lineage": os.path.getsize(lineage_path),
        }

    def _process_document(self, doc_idx, doc_id, chars, segments,
                          orig_widths, doc_edits, round_id, round_dict,
                          edit_dict, texts_f, width_f, lin_f, ledger_f,
                          ledger_rows, docs_records, id_blob,
                          old_edit_by_index=None, old_round_by_index=None):
        text_offset = texts_f.tell() - pfmt.HEADER_SIZE
        orig_block_writer = WidthBlockWriter(width_f)
        pub_block_writer = WidthBlockWriter(width_f)
        # Baseline width block: consume the source's retained widths once.
        orig_cp_len = 0
        for w in orig_widths:
            orig_block_writer.add(w)
            orig_cp_len += 1
        orig_width_off, _ = orig_block_writer.finish()
        orig_width_off -= pfmt.HEADER_SIZE

        lin_offset = lin_f.tell() - pfmt.HEADER_SIZE
        seg_count = 0
        lin_f.write(b"\x00" * pfmt.SHARD_HEADER_SIZE)  # patched later
        # Compact per-segment metadata retained only for the current document
        # (segment count is edit-driven and compressed, never per-codepoint);
        # it builds the sparse baseline->segment sample table afterwards.
        seg_meta: List[Tuple[int, int, int]] = []  # kind, orig_lo, orig_hi
        pub_cp_len = 0
        char_iter = iter(chars)
        seg_iter = iter(segments)
        pushback = None
        input_cursor = 0

        def emit_bytes(utf8_bytes: bytes) -> None:
            for i in range(0, len(utf8_bytes), self.write_bytes):
                piece = utf8_bytes[i:i + self.write_bytes]
                texts_f.write(piece)
                self.stats.texts_bytes_written += len(piece)

        def next_segment():
            nonlocal pushback
            if pushback is not None:
                seg = pushback
                pushback = None
                return seg
            return next(seg_iter, None)

        def take_char(allow_end=False):
            try:
                return next(char_iter)
            except StopIteration:
                if allow_end:
                    return None
                raise pfmt.PublishRejected(
                    "edit interval consumed past the end of the source "
                    "document; round rejected")

        def copy_span(end):
            """Copy input [input_cursor, end) (end=None => to document end).

            Characters stream straight to texts.bin; the lineage segments
            tiling the same span are trimmed and re-emitted with unchanged
            source attribution.  The two streams advance in lockstep by
            codepoint count, so neither whole text nor all intervals reside
            in memory.
            """
            nonlocal pub_cp_len, seg_count, pushback
            span_len = None if end is None else end - input_cursor
            out_lo = pub_cp_len
            copied = 0
            # 1) stream the surviving characters (bounded block reader).
            while span_len is None or copied < span_len:
                ch = take_char(allow_end=(span_len is None))
                if ch is None:
                    break
                emit_bytes(ch.encode("utf-8"))
                pub_block_writer.add(len(ch.encode("utf-8")))
                copied += 1
            # 2) emit the lineage segments tiling the same codepoint range.
            seg_written = 0
            while seg_written < copied:
                seg = next_segment()
                if seg is None:
                    raise pfmt.PublishRejected(
                        "input lineage ends before its text; round rejected")
                s_lo, s_hi = seg["pub_lo"], seg["pub_hi"]
                take = min(s_hi - s_lo, copied - seg_written)
                meta = self._emit_trimmed_segment(
                    seg, s_lo, s_lo + take, out_lo + seg_written, lin_f,
                    old_edit_by_index, old_round_by_index)
                seg_meta.append(meta)
                seg_count += 1
                seg_written += take
                if take < s_hi - s_lo:
                    tail = dict(seg)
                    tail["pub_lo"] = s_lo + take
                    if seg["kind"] == "orig":
                        tail["orig_lo"] = seg["orig_lo"] + take
                    else:
                        tail["rel_lo"] = seg["rel_lo"] + take
                    pushback = tail
            if copied != seg_written:
                raise pfmt.PublishRejected(
                    "input text and lineage codepoint counts disagree; "
                    "round rejected")
            pub_cp_len += copied

        def discard_span(lo, hi):
            """Consume [lo, hi) input characters and their lineage coverage."""
            nonlocal pushback
            for _ in range(hi - lo):
                take_char()
            remaining = hi - lo
            while remaining > 0:
                seg = next_segment()
                if seg is None:
                    raise pfmt.PublishRejected(
                        "input lineage ends before its text; round rejected")
                s_lo, s_hi = seg["pub_lo"], seg["pub_hi"]
                take = min(s_hi - s_lo, remaining)
                remaining -= take
                if take < s_hi - s_lo:
                    tail = dict(seg)
                    tail["pub_lo"] = s_lo + take
                    if seg["kind"] == "orig":
                        tail["orig_lo"] = seg["orig_lo"] + take
                    else:
                        tail["rel_lo"] = seg["rel_lo"] + take
                    pushback = tail

        for lo, hi, edit_id, repl_bytes in doc_edits:
            copy_span(lo)
            discard_span(lo, hi)
            repl_text = repl_bytes.decode("utf-8")
            repl_len = len(repl_text)
            if repl_len > 0:
                # A pure deletion (empty replacement) emits no lineage
                # segment: the gap is represented by the surviving segments
                # no longer being adjacent.  The ledger row below still
                # records the consumed interval and zero generated length.
                lin_f.write(pfmt.SEGMENT_RECORD.pack(
                    pfmt.SEG_KIND_GEN, round_dict[round_id],
                    edit_dict[edit_id], 0, 0, repl_len,
                    pub_cp_len, pub_cp_len + repl_len))
                seg_count += 1
                seg_meta.append((pfmt.SEG_KIND_GEN, -1, -1))
                emit_bytes(repl_bytes)
                for ch in repl_text:
                    pub_block_writer.add(len(ch.encode("utf-8")))
                pub_cp_len += repl_len
            ledger_rows.append({
                "round_id": round_id, "edit_id": edit_id,
                "doc_id": doc_id, "consumed_start": lo,
                "consumed_end": hi, "generated_codepoints": repl_len})
            ledger_f.write(json.dumps(
                ledger_rows[-1], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False) + "\n")
            self.stats.edits_applied += 1
            input_cursor = hi
        # Tail copy to the document end (also covers documents with no edits).
        copy_span(None)
        # Any leftover lineage segment would mean text/lineage desync.
        if next_segment() is not None:
            raise pfmt.PublishRejected(
                "input lineage outlives its text; round rejected")
        pub_width_off, _ = pub_block_writer.finish()
        pub_width_off -= pfmt.HEADER_SIZE
        # Sparse baseline->segment sample table for bounded forward queries.
        # Sample k holds the first segment whose original span could contain
        # baseline position k*interval (or the segment after a deleted run).
        samples: List[Tuple[int, int]] = []
        sample_interval = pfmt.SHARD_SAMPLE_INTERVAL
        n_thresholds = (orig_cp_len // sample_interval) + 1
        si = 0
        for k in range(n_thresholds):
            threshold = k * sample_interval
            while si < len(seg_meta):
                kind, o_lo, o_hi = seg_meta[si]
                if kind == pfmt.SEG_KIND_ORIG and o_hi > threshold:
                    break
                si += 1
            samples.append((threshold, min(si, len(seg_meta))))
        sample_offset = (pfmt.SHARD_HEADER_SIZE
                         + seg_count * pfmt.SEGMENT_RECORD_SIZE)
        for threshold, seg_index in samples:
            lin_f.write(pfmt.SHARD_SAMPLE_RECORD.pack(threshold, seg_index))
        lin_end = lin_f.tell() - pfmt.HEADER_SIZE
        lin_f.seek(pfmt.HEADER_SIZE + lin_offset)
        lin_f.write(pfmt.SHARD_HEADER.pack(
            seg_count, len(samples), pub_cp_len, sample_offset, 0))
        lin_f.seek(pfmt.HEADER_SIZE + lin_end)
        text_bytes = (texts_f.tell() - pfmt.HEADER_SIZE) - text_offset
        id_offset = len(id_blob)
        id_blob += doc_id.encode("utf-8")
        docs_records.append(pfmt.DOCS_RECORD.pack(
            text_offset, text_bytes, orig_cp_len, pub_cp_len,
            lin_offset, lin_end - lin_offset, id_offset,
            orig_width_off, pub_width_off))
        if doc_edits:
            self.stats.edited_documents += 1
        if pub_cp_len == 0:
            self.stats.empty_output_documents += 1
            if orig_cp_len > 0:
                self.stats.deleted_documents += 1
        self.stats.segments_written += seg_count
        return docs_records, id_blob

    def _emit_trimmed_segment(self, seg, in_lo, in_hi, out_lo, lin_f,
                              old_edit_dict=None, old_round_dict=None):
        length = in_hi - in_lo
        out_hi = out_lo + length
        if seg["kind"] == "orig":
            shift = in_lo - seg["pub_lo"]
            base_lo = seg["orig_lo"] + shift
            lin_f.write(pfmt.SEGMENT_RECORD.pack(
                pfmt.SEG_KIND_ORIG, 0, 0, 0,
                base_lo, base_lo + length, out_lo, out_hi))
            return (pfmt.SEG_KIND_ORIG, base_lo, base_lo + length)
        else:
            shift = in_lo - seg["pub_lo"]
            rel_lo = seg["rel_lo"] + shift
            if old_edit_dict is not None and old_round_dict is not None:
                # Inherited generated segment: translate both dictionary
                # indices from the source package's dictionaries to the
                # chain-wide dictionaries of the package being built.
                edit_id = old_edit_dict[seg["edit_ref"]]
                round_id_value = old_round_dict[seg["round_ref"]]
                edit_ref = self._new_edit_index[edit_id]
                round_ref = self._new_round_index[round_id_value]
            else:
                edit_ref = seg["edit_ref"]
                round_ref = seg["round_ref"]
            lin_f.write(pfmt.SEGMENT_RECORD.pack(
                pfmt.SEG_KIND_GEN, round_ref, edit_ref, 0,
                rel_lo, rel_lo + length, out_lo, out_hi))
            return (pfmt.SEG_KIND_GEN, -1, -1)

    # ------------------------------------------------------------------ #
    # Stage 4: ledger naming, identity, atomic publish
    # ------------------------------------------------------------------ #
    def build(self) -> Dict[str, object]:
        if self._built:
            raise RuntimeError("build() may only be called once")
        self._built = True
        try:
            manifest = self._build_inner()
        except BaseException:
            self.source.close()
            shutil.rmtree(self.staging, ignore_errors=True)
            raise
        self.source.close()
        return manifest

    def _build_inner(self) -> Dict[str, object]:
        _expect, round_id = self._read_header()
        id_result, iv_result, edit_digest, n_edits, _cache = \
            self._ingest_edit_list(round_id)
        try:
            with iv_result:
                ledger_rows, round_ids, edit_ids, sizes = \
                    self._apply_round(iv_result, round_id)
        finally:
            id_result.cleanup()
            iv_result.cleanup()
        # Temp spill files are gone before the staging tree is published.
        for name in os.listdir(self.staging):
            if name.startswith("vsort_"):  # pragma: no cover - safety
                os.unlink(os.path.join(self.staging, name))
        return self._finalize(round_id, edit_digest, n_edits,
                              round_ids, edit_ids, ledger_rows, sizes)

    def _finalize(self, round_id, edit_digest, n_edits, round_ids,
                  edit_ids, ledger_rows, sizes) -> Dict[str, object]:
        self.stats.peak_rss_kb = _peak_rss_kb()
        self.stats.source_bytes_read = self.source.stats.get(
            "source_bytes_read", 0)
        files = {}
        for root, _dirs, names in os.walk(self.staging):
            for name in names:
                if name == pfmt.OPEN_MARKER:
                    continue
                p = os.path.join(root, name)
                rel = os.path.relpath(p, self.staging)
                files[rel] = pfmt.hash_file(p)
        if isinstance(self.source, IndexSource):
            chain_root = self.source.root_index_identity
            prior = []
            serial = 1
        else:
            chain_root = self.source.root_index_identity
            prior = self.source.manifest["rounds"]
            serial = len(prior) + 1
        ledger_rel = os.path.join(
            pfmt.LEDGER_DIR, f"{serial:06d}.jsonl")
        rounds = list(prior) + [{
            "round_id": round_id,
            "serial": serial,
            "edit_digest": edit_digest,
            "edits": n_edits,
            "ledger": ledger_rel,
        }]
        manifest = {
            "format_version": pfmt.PUB_FORMAT_VERSION,
            "chain": {
                "root_index_identity": chain_root,
                "direct_input_identity": self.source.input_identity,
                "direct_input_kind": self.source.input_kind,
            },
            "round_id": round_id,
            "rounds": rounds,
            "round_dictionary": round_ids,
            "edit_dictionary": edit_ids,
            "edit_digest": edit_digest,
            "geometry": {
                "documents": self.stats.documents,
                "edits_applied": self.stats.edits_applied,
                "segments": self.stats.segments_written,
            },
            "limits": {
                "max_edit_bytes": self.max_edit_bytes,
                "max_replacement_codepoints": self.max_replacement_cp,
                "max_interval_codepoints": self.max_interval_cp,
                "max_identifier_bytes": self.max_id_bytes,
                "max_edits": self.max_edits,
            },
            "build_stats": self.stats.as_dict(),
            "files": files,
        }
        manifest["package_identity"] = pfmt.compute_package_identity(
            manifest)
        fmt.write_manifest(self.staging, manifest)
        marker = os.path.join(self.staging, pfmt.OPEN_MARKER)
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.staging, self.target_dir)
        return manifest

    def _compute_identity(self, manifest_without_id: dict) -> str:
        return pfmt.compute_package_identity(manifest_without_id)

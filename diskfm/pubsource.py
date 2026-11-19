"""Read-only, bounded-memory source adapters for release construction.

A release round consumes one of two sealed inputs, always read-only:

* a sealed diskfm index (chain root): :class:`IndexSource` streams
  ``(doc_id, codepoints, baseline lineage)`` directly from ``content.bin``
  through an incremental UTF-8 decoder -- never decoding a whole document and
  never touching BWT/rank structures;
* a previous release package: :class:`ReleaseSource` streams the same triple
  from ``texts.bin`` plus the paged lineage shards of
  :mod:`diskfm.pubformat`, so a later round can compose the prior lineage
  with its own edits without the original index being present.

Both adapters expose the same narrow interface used by
:mod:`diskfm.pubbuilder`:

* :meth:`doc_count`, :attr:`root_index_identity`, :attr:`input_identity`;
* :meth:`iter_documents` -- ``(doc_index, doc_id, cp_iter, lineage_iter,``
  ``orig_widths_iter)`` per document, where every iterator yields one item at
  a time and holds only bounded buffers (one read block, one incremental
  decoder, one segment window).  A package source's lineage iterator yields
  segment dicts; an index source synthesises one whole-document baseline
  segment so the downstream join code is identical.
"""

from __future__ import annotations

import codecs
import json
import os
import struct
from typing import Dict, Iterator, Tuple

from . import format as fmt
from . import pubformat as pfmt
from .pager import FilePager
from .query import DiskFMIndex


class _BlockCharStream:
    """Forward codepoint stream over a UTF-8 byte range, block by block.

    Holds at most ``block_bytes`` raw bytes plus one incremental decoder.
    Multi-byte characters straddling a block boundary are held in the
    decoder's internal state, never split or mis-decoded.
    """

    def __init__(self, f, abs_start: int, abs_end: int,
                 block_bytes: int, stats: Dict[str, int],
                 error_cls=fmt.IndexCorruptError):
        self._f = f
        self._pos = abs_start
        self._end = abs_end
        self._block_bytes = max(16, block_bytes)
        self._stats = stats
        self._error_cls = error_cls
        self._decoder = codecs.getincrementaldecoder("utf-8")(
            errors="strict")
        self._pending = ""
        self._eof = False

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        while not self._pending:
            if self._eof:
                if self._pending:
                    break
                raise StopIteration
            take = min(self._block_bytes, self._end - self._pos)
            if take <= 0:
                self._eof = True
                tail = self._decoder.decode(b"", final=True)
                if tail:
                    self._pending = tail
                    break
                raise StopIteration
            self._f.seek(self._pos)
            raw = self._f.read(take)
            self._stats["source_bytes_read"] = self._stats.get(
                "source_bytes_read", 0) + len(raw)
            if len(raw) != take:
                raise self._error_cls("source body truncated mid-document")
            self._pos += take
            if self._pos >= self._end:
                self._eof = True
                try:
                    decoded = self._decoder.decode(raw, final=True)
                except UnicodeDecodeError as exc:
                    raise self._error_cls(
                        "invalid UTF-8 inside source document") from exc
            else:
                try:
                    decoded = self._decoder.decode(raw, final=False)
                except UnicodeDecodeError as exc:
                    raise self._error_cls(
                        "invalid UTF-8 inside source document") from exc
            self._pending = decoded
        ch = self._pending[0]
        self._pending = self._pending[1:]
        if 0xD800 <= ord(ch) <= 0xDFFF:
            raise self._error_cls(
                "source text contains an isolated surrogate code point")
        return ch


class _WidthBlockStream:
    """Sequential, bounded reader of one packed 2-bit width block.

    Reads one packed byte (four codepoint widths) at a time after the block
    header and its checkpoints, so RAM is O(1) however long the document is.
    Checkpoints are skipped over, not retained.
    """

    def __init__(self, f, block_abs_offset: int):
        self._f = f
        self._offset = block_abs_offset
        f.seek(block_abs_offset)
        head = f.read(pfmt.WIDTH_BLOCK_HEADER_SIZE)
        if len(head) != pfmt.WIDTH_BLOCK_HEADER_SIZE:
            raise pfmt.ReleaseCorruptError("width block header truncated")
        self.cp_len, packed_bytes, cp_count = \
            pfmt.WIDTH_BLOCK_HEADER.unpack(head)
        self._packed_abs = (block_abs_offset
                            + pfmt.WIDTH_BLOCK_HEADER_SIZE + 8 * cp_count)
        self._packed_bytes = packed_bytes
        self._emitted = 0
        self._byte_index = 0
        self._bitbuf = 0
        self._bitbuf_len = 0

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        if self._emitted >= self.cp_len:
            raise StopIteration
        if self._bitbuf_len < 2:
            if self._byte_index >= self._packed_bytes:
                raise pfmt.ReleaseCorruptError("packed width stream exhausted")
            self._f.seek(self._packed_abs + self._byte_index)
            raw = self._f.read(1)
            if len(raw) != 1:
                raise pfmt.ReleaseCorruptError("packed width stream truncated")
            self._bitbuf |= raw[0] << self._bitbuf_len
            self._bitbuf_len += 8
            self._byte_index += 1
        width = (self._bitbuf & 0x3) + 1
        self._bitbuf >>= 2
        self._bitbuf_len -= 2
        self._emitted += 1
        return width


class _SegmentStream:
    """Forward paged stream over one lineage shard's fixed-width records."""

    def __init__(self, f, shard_offset: int, segment_count: int,
                 page_records: int = 4096):
        self._f = f
        self._offset = shard_offset + pfmt.SHARD_HEADER_SIZE
        self._remaining = segment_count
        self._page_records = page_records
        self._idx = 0

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        if self._remaining <= 0:
            raise StopIteration
        self._f.seek(self._offset)
        raw = self._f.read(pfmt.SEGMENT_RECORD_SIZE)
        if len(raw) != pfmt.SEGMENT_RECORD_SIZE:
            raise pfmt.ReleaseCorruptError("lineage segment record truncated")
        kind, ref1, ref2, _reserved, a, b, c, d = \
            pfmt.SEGMENT_RECORD.unpack(raw)
        self._offset += pfmt.SEGMENT_RECORD_SIZE
        self._remaining -= 1
        if kind == pfmt.SEG_KIND_ORIG:
            seg = {"kind": "orig", "seg_index": self._idx,
                   "orig_lo": a, "orig_hi": b,
                   "pub_lo": c, "pub_hi": d}
        elif kind == pfmt.SEG_KIND_GEN:
            seg = {"kind": "gen", "seg_index": self._idx,
                   "round_ref": ref1, "edit_ref": ref2,
                   "rel_lo": a, "rel_hi": b,
                   "pub_lo": c, "pub_hi": d}
        else:
            raise pfmt.ReleaseCorruptError(
                f"unknown lineage segment kind {kind}")
        if c >= d or b <= a or d - c != b - a:
            raise pfmt.ReleaseCorruptError("lineage segment geometry invalid")
        self._idx += 1
        return seg


class _Utf8WidthStream:
    """Forward stream of per-codepoint UTF-8 widths over a byte range.

    Independent of :class:`_BlockCharStream` (its own file position and
    incremental decoder), so the retained baseline width block and the
    emitted text can be produced without either iterator consuming the other.
    """

    def __init__(self, f, abs_start: int, abs_end: int, block_bytes: int,
                 stats: Dict[str, int]):
        self._inner = _BlockCharStream(
            f, abs_start, abs_end, block_bytes, stats)

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        ch = next(self._inner)
        return len(ch.encode("utf-8"))


class IndexSource:
    """Sealed diskfm index presented as a streaming release input."""

    kind = "index"

    def __init__(self, index_dir: str, block_bytes: int = 1 << 16):
        self.index_dir = index_dir
        self.block_bytes = block_bytes
        self._idx = DiskFMIndex(index_dir)
        self._content_f = open(
            os.path.join(index_dir, "content.bin"), "rb")
        # Separate handle for the retained-width pass so the two forward
        # block readers never move each other's file position.  Bytes read by
        # the auxiliary width pass are counted separately.
        self._width_f = open(
            os.path.join(index_dir, "content.bin"), "rb")
        self.stats = {"source_bytes_read": 0,
                      "source_width_bytes_read": 0}
        self._width_stats = {"source_bytes_read": 0}

    @property
    def root_index_identity(self) -> str:
        return self._idx.identity

    @property
    def input_identity(self) -> str:
        # The chain root's logical identity is the sealed index identity.
        return self._idx.identity

    @property
    def input_kind(self) -> str:
        return "index"

    def doc_count(self) -> int:
        return self._idx.n_docs

    def doc_id(self, doc_idx: int) -> str:
        return self._idx._read_doc_id(doc_idx)

    def close(self) -> None:
        self._content_f.close()
        self._width_f.close()
        self._idx.close()

    def __enter__(self) -> "IndexSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def iter_documents(self) -> Iterator[Tuple[int, str, Iterator[str],
                                               Iterator[dict],
                                               Iterator[int]]]:
        for doc_idx in range(self._idx.n_docs):
            doc_id = self._idx._read_doc_id(doc_idx)
            rec = self._idx._doc_record(doc_idx)
            abs_start = fmt.HEADER_SIZE + rec["content_start"]
            abs_end = abs_start + self._idx._doc_byte_len(doc_idx)
            cp_len = rec["t_len"]
            chars = _BlockCharStream(
                self._content_f, abs_start, abs_end, self.block_bytes,
                self.stats)
            widths = _Utf8WidthStream(
                self._width_f, abs_start, abs_end, self.block_bytes,
                self._width_stats)

            def synthetic_segments(length=cp_len):
                if length:
                    yield {"kind": "orig", "seg_index": 0,
                           "orig_lo": 0, "orig_hi": length,
                           "pub_lo": 0, "pub_hi": length}

            yield (doc_idx, doc_id, chars, synthetic_segments(), widths)


class ReleaseSource:
    """Previous release package presented as a streaming release input."""

    kind = "release"

    def __init__(self, package_dir: str, block_bytes: int = 1 << 16,
                 max_pages: int = 16, page_records: int = 4096):
        self.package_dir = package_dir
        self.block_bytes = block_bytes
        pfmt.check_open_marker(package_dir)
        with open(os.path.join(package_dir, pfmt.MANIFEST_NAME),
                  encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("format_version") != pfmt.PUB_FORMAT_VERSION:
            raise pfmt.ReleaseCorruptError(
                "release manifest format version mismatch")
        self._verify_files()
        self._n_docs = self.manifest["geometry"]["documents"]
        self._docs_pager = FilePager(
            os.path.join(package_dir, "docs.bin"),
            record_size=pfmt.DOCS_RECORD_SIZE,
            header_size=pfmt.HEADER_SIZE, page_records=page_records,
            max_pages=max_pages)
        self._docs_f = open(os.path.join(package_dir, "docs.bin"), "rb")
        self._texts_f = open(os.path.join(package_dir, "texts.bin"), "rb")
        self._lineage_f = open(
            os.path.join(package_dir, "lineage.bin"), "rb")
        self._width_f = open(os.path.join(package_dir, "width.bin"), "rb")
        self.stats = {"source_bytes_read": 0}

    def _verify_files(self) -> None:
        for name, entry in self.manifest["files"].items():
            pfmt.verify_file(os.path.join(self.package_dir, name), entry)
        ledger_dir = os.path.join(self.package_dir, pfmt.LEDGER_DIR)
        if not os.path.isdir(ledger_dir):
            raise pfmt.ReleaseCorruptError("ledger directory missing")
        for entry in self.manifest["rounds"]:
            lp = os.path.join(self.package_dir, entry["ledger"])
            if not os.path.exists(lp):
                raise pfmt.ReleaseCorruptError(
                    f"ledger file missing: {entry['ledger']}")

    @property
    def root_index_identity(self) -> str:
        return self.manifest["chain"]["root_index_identity"]

    @property
    def input_identity(self) -> str:
        return self.manifest["package_identity"]

    @property
    def input_kind(self) -> str:
        return "release"

    def doc_count(self) -> int:
        return self._n_docs

    def _doc_record(self, doc_idx: int) -> dict:
        if not 0 <= doc_idx < self._n_docs:
            raise pfmt.ReleaseCorruptError(
                f"document index {doc_idx} out of range")
        vals = pfmt.DOCS_RECORD.unpack(
            self._docs_pager.read_record(doc_idx))
        keys = ("text_offset", "text_bytes", "orig_cp_len", "pub_cp_len",
                "lineage_offset", "lineage_bytes", "id_blob_offset",
                "orig_width_block_offset", "pub_width_block_offset")
        return dict(zip(keys, vals))

    def doc_id(self, doc_idx: int) -> str:
        rec = self._doc_record(doc_idx)
        id_blob_abs = (pfmt.HEADER_SIZE
                       + self._n_docs * pfmt.DOCS_RECORD_SIZE
                       + rec["id_blob_offset"])
        next_rec = self._doc_record(doc_idx + 1) if doc_idx + 1 < self._n_docs \
            else None
        if next_rec is None:
            id_end = (self.manifest["files"]["docs.bin"]["bytes"]
                      - pfmt.HEADER_SIZE
                      - self._n_docs * pfmt.DOCS_RECORD_SIZE)
        else:
            id_end = next_rec["id_blob_offset"]
        id_len = id_end - rec["id_blob_offset"]
        self._docs_f.seek(id_blob_abs)
        raw = self._docs_f.read(id_len)
        if len(raw) != id_len:
            raise pfmt.ReleaseCorruptError("doc id blob truncated")
        return raw.decode("utf-8")

    def close(self) -> None:
        self._docs_pager.close()
        for f in (self._docs_f, self._texts_f, self._lineage_f,
                  self._width_f):
            f.close()

    def __enter__(self) -> "ReleaseSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def iter_documents(self) -> Iterator[Tuple[int, str, Iterator[str],
                                               Iterator[dict],
                                               Iterator[int]]]:
        for doc_idx in range(self._n_docs):
            doc_id = self.doc_id(doc_idx)
            rec = self._doc_record(doc_idx)
            abs_start = pfmt.HEADER_SIZE + rec["text_offset"]
            abs_end = abs_start + rec["text_bytes"]
            chars = _BlockCharStream(
                self._texts_f, abs_start, abs_end, self.block_bytes,
                self.stats, error_cls=pfmt.ReleaseCorruptError)
            # Read the shard header to learn its segment count.
            self._lineage_f.seek(pfmt.HEADER_SIZE + rec["lineage_offset"])
            head = self._lineage_f.read(pfmt.SHARD_HEADER_SIZE)
            if len(head) != pfmt.SHARD_HEADER_SIZE:
                raise pfmt.ReleaseCorruptError("lineage shard header missing")
            segment_count, _sample_count, shard_pub_len, _soff, _res = \
                pfmt.SHARD_HEADER.unpack(head)
            if shard_pub_len != rec["pub_cp_len"]:
                raise pfmt.ReleaseCorruptError(
                    "lineage shard length disagrees with docs record")
            segments = _SegmentStream(
                self._lineage_f,
                pfmt.HEADER_SIZE + rec["lineage_offset"], segment_count)
            # Baseline widths are retained compactly from the prior package:
            # the old characters themselves are never read back.
            widths = _WidthBlockStream(
                self._width_f,
                pfmt.HEADER_SIZE + rec["orig_width_block_offset"])
            yield doc_idx, doc_id, chars, segments, widths


def open_source(path: str, block_bytes: int = 1 << 16,
                max_pages: int = 16, page_records: int = 4096):
    """Open a sealed index or a completed release package read-only."""
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    if os.path.exists(os.path.join(path, pfmt.OPEN_MARKER)) and \
            os.path.exists(os.path.join(path, pfmt.MANIFEST_NAME)):
        with open(os.path.join(path, pfmt.MANIFEST_NAME),
                  encoding="utf-8") as f:
            head = json.load(f)
        if head.get("package_identity") is not None:
            return ReleaseSource(
                path, block_bytes=block_bytes, max_pages=max_pages,
                page_records=page_records)
    return IndexSource(path, block_bytes=block_bytes)

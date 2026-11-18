"""On-disk format of the document-level near-duplicate similarity catalog.

Format version: 1.

A similarity catalog is an *independent, re-openable* directory built from a
sealed diskfm index.  It contains no original document text -- only qgram
(shingle) relations derived from it -- so querying it can never read the
corpus back.  Every data file starts with an 8-byte little-endian header
``[u16 magic=0x5349 ('SI'), u16 version, u16 file_kind, u16 reserved=0]``.

Catalog directory contents (all binary, little-endian unless stated):

==========================  ================================================
``manifest.json``           UTF-8 JSON: format version, catalog identity,
                            bound source-index identity + content hash,
                            parameters (q and the tokenization rule
                            declaration), document-set digest, geometry,
                            build statistics, and SHA-256 + byte length of
                            every data file.
``OPEN``                    completion marker; written last (fsync, then
                            atomic rename of the staging directory).  Absent
                            => the build was interrupted and the catalog
                            must never be opened.
``qgrams.bin``              qgram dictionary blob: the exact UTF-8 bytes of
                            every distinct qgram, concatenated in dictionary
                            order.  Dictionary order is byte-lexicographic
                            on UTF-8, which coincides with Unicode code
                            point order.  Qgram identity is the *full
                            length-delimited byte string* -- no hashes are
                            used anywhere, so two different qgrams can never
                            silently merge.
``qgramdir.bin``            fixed 32-byte records, one per dictionary entry
                            (record i describes qgram id i):
                            ``u64 blob_offset`` (body-relative, into
                            qgrams.bin), ``u32 byte_len``, ``u32 reserved``,
                            ``u64 posting_offset`` (entry index into
                            postings.bin), ``u64 posting_count``.
``postings.bin``            per-qgram posting lists, concatenated in
                            dictionary order.  Each entry is 8 bytes:
                            ``u32 doc_index, u32 term_freq`` where term_freq
                            is the occurrence count of the qgram inside that
                            document (kept for ``explain``; Jaccard counting
                            always counts a document once per qgram).
                            Entries inside one posting list are sorted by
                            doc_index.
``docqgrams.bin``           forward relations: for each document (in
                            document order) its sorted distinct qgram ids as
                            u32 values.  The slice belonging to document d
                            is given by its docs.bin record.
``docs.bin``                fixed 40-byte records, one per document, in the
                            source index's document order:
                            ``u64 shingle_count`` (distinct qgrams),
                            ``u64 docqgrams_offset`` (entry index into
                            docqgrams.bin), ``u64 text_len_codepoints``,
                            ``u64 id_blob_offset``, ``u32 id_len``,
                            ``u32 flags``.  Followed by the concatenated
                            UTF-8 doc_id blob.  ``flags & FLAG_NO_SHINGLES``
                            marks documents shorter than q codepoints: they
                            have no comparable shingles and are flagged
                            explicitly instead of being silently treated as
                            zero-similarity.
==========================  ================================================

The manifest binds all files together (SHA-256 + length each) and records
the source index identity (its instance UUID and content hash) plus a
document-set digest recomputed over the streamed ``(doc_id, text)`` pairs.
Rebuilding from the same source index and parameters reproduces identical
data files and digests; only the per-build catalog UUID differs.
"""

from __future__ import annotations

import os
import struct
from typing import Dict

from . import format as fmt  # reuse hash_file / verify_file / write_manifest

SIM_FORMAT_VERSION = 1
SIM_MAGIC = 0x5349  # 'SI'
HEADER_SIZE = fmt.HEADER_SIZE
HEADER_STRUCT = fmt.HEADER_STRUCT

# File kind tags.
KIND_QGRAMS = 1
KIND_QGRAMDIR = 2
KIND_POSTINGS = 3
KIND_DOCQGRAMS = 4
KIND_DOCS = 5

MANIFEST_NAME = fmt.MANIFEST_NAME
OPEN_MARKER = fmt.OPEN_MARKER
STAGING_SUFFIX = fmt.STAGING_SUFFIX

QGRAMDIR_RECORD = struct.Struct("<QIIQQ")  # 32 bytes
QGRAMDIR_RECORD_SIZE = QGRAMDIR_RECORD.size
POSTING_RECORD = struct.Struct("<II")  # 8 bytes: doc_index, term_freq
POSTING_RECORD_SIZE = POSTING_RECORD.size
DOCQGRAM_RECORD = struct.Struct("<I")  # 4 bytes: qgram_id
DOCQGRAM_RECORD_SIZE = DOCQGRAM_RECORD.size
DOCS_RECORD = struct.Struct("<QQQQII")  # 40 bytes
DOCS_RECORD_SIZE = DOCS_RECORD.size

# docs.bin flag bits.
FLAG_NO_SHINGLES = 0x1  # document shorter than q: nothing comparable

# Declarative tokenization rule recorded in the manifest.  Shingles are
# maximal-runs of exactly q consecutive Unicode code points over the raw
# document string: no tokenization, no normalisation, no case folding, and
# shingles never cross a document boundary.
TOKENIZATION_RULE = "unicode-codepoint-qgram/v1"

# Re-export the shared error types so callers have one obvious import.
IndexCorruptError = fmt.IndexCorruptError
CursorError = fmt.CursorError


def write_header(f, kind: int) -> None:
    f.write(HEADER_STRUCT.pack(SIM_MAGIC, SIM_FORMAT_VERSION, kind, 0))


def read_header(f, expected_kind: int) -> None:
    raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise IndexCorruptError("file truncated inside header")
    magic, version, kind, reserved = HEADER_STRUCT.unpack(raw)
    if magic != SIM_MAGIC:
        raise IndexCorruptError("bad magic; not a diskfm similarity file")
    if version != SIM_FORMAT_VERSION:
        raise IndexCorruptError(
            f"unsupported similarity format version {version}; "
            f"expected {SIM_FORMAT_VERSION}")
    if kind != expected_kind:
        raise IndexCorruptError(
            f"file kind {kind} does not match expected {expected_kind}")
    if reserved != 0:
        raise IndexCorruptError("non-zero reserved header field")


def hash_file(path: str) -> Dict[str, object]:
    return fmt.hash_file(path)


def verify_file(path: str, entry: Dict[str, object]) -> None:
    fmt.verify_file(path, entry)


def write_manifest(catalog_dir: str, manifest: Dict[str, object]) -> None:
    fmt.write_manifest(catalog_dir, manifest)


def check_open_marker(catalog_dir: str) -> None:
    if not os.path.exists(os.path.join(catalog_dir, OPEN_MARKER)):
        raise IndexCorruptError(
            "no OPEN marker: catalog build was interrupted; not readable")

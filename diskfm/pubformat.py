"""On-disk format of a desensitisation *release package* (version 1).

A release package is a self-contained, independently transportable directory
produced by applying one human-approved edit round to either a sealed diskfm
index (the chain root) or a previous release package.  It deliberately
contains **no removed text, no source index and no intermediate package**:

* the only body text ever stored is the current released text
  (``texts.bin``);
* the lineage keeps compact per-codepoint UTF-8 widths and random-access
  samples of the *baseline* text instead of the baseline characters
  (``width.bin``);
* the edit ledgers record round / edit / document / consumed input interval
  / generated length only -- never the consumed characters or the
  replacement value (``ledgers/*.jsonl``).

Every binary file starts with an 8-byte little-endian header
``[u16 magic=0x5055 ('PU'), u16 version, u16 file_kind, u16 reserved=0]``.

Package contents
================

==========================  ================================================
``manifest.json``           UTF-8 JSON: format version, deterministic
                            package identity, chain-root index identity,
                            direct-input binding, round chain, logical edit
                            digest, identifier dictionaries, geometry,
                            build statistics, SHA-256 + byte length of every
                            packaged file.
``OPEN``                    completion marker; written and fsynced last
                            inside the staging directory, which is then
                            atomically renamed over *no* existing target.
``texts.bin``               released UTF-8 document bodies back to back, in
                            source document order.  Deleted documents occupy
                            zero bytes; documents turned into the empty
                            string are still listed in ``docs.bin``.
``docs.bin``                fixed 72-byte records (nine u64 fields), one per
                            document in source order, followed by the
                            concatenated UTF-8 doc_id blob.  Accessed through
                            a bounded pager; never loaded wholesale.
``width.bin``               per-document width blocks, two per document:
                            baseline codepoint widths (retained so lost
                            fragments can report UTF-8 byte boundaries
                            without keeping the old characters) and released
                            codepoint widths.  Each block is a packed 2-bit
                            width stream plus u64 checkpoints for random
                            access in O(1) bounded reads.
``lineage.bin``             per-document lineage shards located through
                            ``docs.bin``.  A shard is a 24-byte header
                            followed by fixed 40-byte segment records that
                            tile the released codepoint space, then a sparse
                            baseline->segment sample table.  Each segment is
                            either *baseline original* (baseline interval
                            + released interval, equal lengths) or
                            *generated* (round, edit, relative interval
                            inside that edit's replacement, released
                            interval).  Long unchanged copies are one
                            compressed segment, never per-codepoint records.
``ledgers/NNNNNN.jsonl``    one canonical, body-free ledger file per round,
                            chain order.  Records: round, edit id, doc_id,
                            consumed half-open interval on that round's
                            input, generated codepoint length.
==========================  ================================================

Identity
========

``package_identity`` is a deterministic SHA-256 over the chain root, the
round chain, the per-round logical edit digests, the identifier
dictionaries, the geometry and every packaged file's hash -- therefore it
covers the lineage and cannot be reproduced from output strings alone.
Shuffling edit-list record order or changing spill chunk sizes changes
neither any output byte nor the identity: every digest input is taken from
canonically ordered logical content.
"""

from __future__ import annotations

import os
import struct
from typing import Dict

from . import format as fmt

PUB_FORMAT_VERSION = 1
PUB_MAGIC = 0x5055  # 'PU'
HEADER_SIZE = fmt.HEADER_SIZE
HEADER_STRUCT = fmt.HEADER_STRUCT

# Binary file kind tags.
KIND_TEXTS = 1
KIND_DOCS = 2
KIND_WIDTH = 3
KIND_LINEAGE = 4

MANIFEST_NAME = fmt.MANIFEST_NAME
OPEN_MARKER = fmt.OPEN_MARKER
STAGING_SUFFIX = ".staging"
LEDGER_DIR = "ledgers"

# docs.bin: nine little-endian u64 fields = 72 bytes.
#   text_offset, text_bytes, orig_cp_len, pub_cp_len,
#   lineage_offset, lineage_bytes, id_blob_offset,
#   orig_width_block_offset, pub_width_block_offset
DOCS_RECORD = struct.Struct("<QQQQQQQQQ")
DOCS_RECORD_SIZE = DOCS_RECORD.size

# lineage.bin shard header: u32 segment_count, u32 sample_count, u64
# pub_cp_len, u32 sample_offset (shard-relative), u32 reserved.
SHARD_HEADER = struct.Struct("<IIQII")
SHARD_HEADER_SIZE = SHARD_HEADER.size
# Baseline codepoint distance between sparse baseline->segment samples.
SHARD_SAMPLE_INTERVAL = 64
# Sample table record: u32 baseline_codepoint_threshold, u32 segment_index.
SHARD_SAMPLE_RECORD = struct.Struct("<II")
SHARD_SAMPLE_RECORD_SIZE = SHARD_SAMPLE_RECORD.size

# Segment record: u32 kind, u32 ref1, u32 ref2, u32 reserved, four u64.
SEG_KIND_ORIG = 1
SEG_KIND_GEN = 2
SEGMENT_RECORD = struct.Struct("<IIIIQQQQ")
SEGMENT_RECORD_SIZE = SEGMENT_RECORD.size

# Original segment: ref1 = segment index inside its shard;
#   a,b = baseline [lo,hi); c,d = released [lo,hi); lengths equal.
# Generated segment: ref1 = round dictionary index, ref2 = edit dictionary
#   index; a,b = relative [lo,hi) inside the replacement; c,d = released
#   [lo,hi); relative length equals released length.

# width.bin block header: u64 cp_len, u64 packed_bytes, u64 checkpoint_count.
WIDTH_BLOCK_HEADER = struct.Struct("<QQQ")
WIDTH_BLOCK_HEADER_SIZE = WIDTH_BLOCK_HEADER.size
WIDTH_CHECKPOINT_INTERVAL = 64  # codepoints between u64 byte checkpoints

IndexCorruptError = fmt.IndexCorruptError


class PublishRejected(ValueError):
    """Raised when a whole edit round must be rejected before publication.

    Messages carry coordinates and identifiers only -- never consumed text
    fragments or replacement values.
    """


class ReleaseCorruptError(IndexCorruptError):
    """Raised when a release package is missing files or fails validation."""


def write_header(f, kind: int) -> None:
    f.write(HEADER_STRUCT.pack(PUB_MAGIC, PUB_FORMAT_VERSION, kind, 0))


def read_header(f, expected_kind: int) -> None:
    raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise ReleaseCorruptError("file truncated inside header")
    magic, version, kind, reserved = HEADER_STRUCT.unpack(raw)
    if magic != PUB_MAGIC:
        raise ReleaseCorruptError("bad magic; not a diskfm release file")
    if version != PUB_FORMAT_VERSION:
        raise ReleaseCorruptError(
            f"unsupported release format version {version}; "
            f"expected {PUB_FORMAT_VERSION}")
    if kind != expected_kind:
        raise ReleaseCorruptError(
            f"file kind {kind} does not match expected {expected_kind}")
    if reserved != 0:
        raise ReleaseCorruptError("non-zero reserved header field")


def hash_file(path: str) -> Dict[str, object]:
    return fmt.hash_file(path)


def verify_file(path: str, entry: Dict[str, object]) -> None:
    fmt.verify_file(path, entry)


def check_open_marker(package_dir: str) -> None:
    if not os.path.exists(os.path.join(package_dir, OPEN_MARKER)):
        raise ReleaseCorruptError(
            "no OPEN marker: release build was interrupted; not readable")


def compute_package_identity(manifest_without_id: dict) -> str:
    """Deterministic identity covering lineage, chain and every file hash.

    The output strings alone cannot reproduce this: the round chain, logical
    edit digests, identifier dictionaries, geometry and per-file SHA-256
    entries all enter the hash.  Spill chunk sizes and record ordering never
    appear, so reordering an edit list or re-chunking yields the same value.
    """
    import hashlib
    import json
    h = hashlib.sha256()
    h.update(b"diskfm-release-v1\n")
    h.update(json.dumps(
        manifest_without_id["chain"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    h.update(b"\nrounds\n")
    for entry in manifest_without_id["rounds"]:
        h.update(json.dumps(
            entry, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    h.update(b"\ndictionaries\n")
    h.update(json.dumps(
        manifest_without_id["round_dictionary"], sort_keys=True,
        separators=(",", ":")).encode("utf-8"))
    h.update(json.dumps(
        manifest_without_id["edit_dictionary"], sort_keys=True,
        separators=(",", ":")).encode("utf-8"))
    h.update(b"\ngeometry\n")
    h.update(json.dumps(
        manifest_without_id["geometry"], sort_keys=True,
        separators=(",", ":")).encode("utf-8"))
    h.update(b"\nfiles\n")
    for rel in sorted(manifest_without_id["files"]):
        h.update(rel.encode("utf-8"))
        h.update(json.dumps(
            manifest_without_id["files"][rel],
            sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def width_byte_offset(block_seek, read_at, cp_offset: int) -> int:
    """Random UTF-8 byte offset inside a width block.

    ``block_seek`` returns the absolute file offset of a byte relative to the
    block start; ``read_at`` returns ``n`` bytes at that absolute offset.
    At most one checkpoint (u64) and 16 packed bytes (64 codepoints x 2 bits)
    are read, regardless of document length.
    """
    head = read_at(block_seek(0), WIDTH_BLOCK_HEADER_SIZE)
    if len(head) != WIDTH_BLOCK_HEADER_SIZE:
        raise ReleaseCorruptError("width block header truncated")
    cp_len, packed_bytes, cp_count = WIDTH_BLOCK_HEADER.unpack(head)
    if not 0 <= cp_offset <= cp_len:
        raise ReleaseCorruptError(
            f"codepoint offset {cp_offset} outside width block of {cp_len}")
    if cp_offset == 0:
        # Empty block, or the start of any block: zero prefix bytes.
        return 0
    k = cp_offset // WIDTH_CHECKPOINT_INTERVAL
    if k >= cp_count:
        raise ReleaseCorruptError("width checkpoint index out of range")
    cp_raw = read_at(block_seek(WIDTH_BLOCK_HEADER_SIZE + 8 * k), 8)
    if len(cp_raw) != 8:
        raise ReleaseCorruptError("width checkpoint truncated")
    (base_bytes,) = struct.unpack("<Q", cp_raw)
    extra = cp_offset - k * WIDTH_CHECKPOINT_INTERVAL
    if extra == 0:
        return base_bytes
    packed_start = WIDTH_BLOCK_HEADER_SIZE + 8 * cp_count
    bit_start = (k * WIDTH_CHECKPOINT_INTERVAL) * 2
    byte_start = bit_start // 8
    need_bytes = (extra * 2 + (bit_start % 8) + 7) // 8
    raw = read_at(block_seek(packed_start + byte_start), need_bytes)
    if len(raw) != need_bytes:
        raise ReleaseCorruptError("packed width stream truncated")
    bitbuf = 0
    for i, b in enumerate(raw):
        bitbuf |= b << (8 * i)
    bitbuf >>= (bit_start % 8)
    total = 0
    for j in range(extra):
        total += ((bitbuf >> (2 * j)) & 0x3) + 1
    return base_bytes + total

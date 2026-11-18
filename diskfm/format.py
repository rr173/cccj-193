"""On-disk format constants, header helpers and manifest binding.

Format version: 1.

Every data file starts with an 8-byte little-endian header
``[u16 magic=0x464D ('FM'), u16 version, u16 file_kind, u16 reserved=0]``.

Index directory contents (all binary, little-endian unless stated):

==========================  ================================================
``manifest.json``           UTF-8 JSON: identity, geometry, file hashes.
``OPEN``                    completion marker; absent => index unreadable.
``text.bin``                global symbol stream T as u32, user codepoints
                            in [0, 0x10FFFF] plus internal sentinels
                            >= 0x110000 (one per document).
``content.bin``             exact original document strings back to back,
                            raw UTF-8 bytes (NUL/newlines untouched).
``bytesample.bin``          u64 per-document byte-offset checkpoints for
                            converting codepoint offsets into UTF-8 byte
                            offsets without re-encoding whole documents.
``doctable.bin``            fixed 48-byte records: u64 sentinel position in
                            T, u64 start in T, u64 start byte in content,
                            u64 length in codepoints, u64 offset of the
                            doc_id inside the trailing id blob, u32 id byte
                            length, u32 first checkpoint index in
                            bytesample.bin.  Followed by the doc_id byte
                            blob; both parts are accessed through bounded
                            paging, never loaded wholesale.
``bwt.bin``                 BWT symbols as u32, one per T position.
``rank.bin``                occurrence lists: for every BWT symbol c,
                            sorted u32 row indices where BWT[row] == c.
                            Prefix count = binary search.  Rank is therefore
                            a real on-disk structure, not a materialised
                            full SA or re-scan of the corpus.
``rankdir.bin``             32-byte records: u64 symbol, u64 absolute
                            offset in rank.bin, u64 count, u64 cumulative
                            count C(c) of symbols strictly smaller; read
                            through a bounded LRU pager.
``sa_sample.bin``           pairs u32 SA row, u64 T position, one per
                            sampled row (every sample_interval rows).
==========================  ================================================

The manifest binds all of the above together: geometry (N, doc count,
alphabet size, sampling), the index identity embedded in cursors, and a
SHA-256 + byte length for every file.  A truncated/edited file fails the
hash check; a missing OPEN marker means a build was interrupted and the
index must never be opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Dict

FORMAT_VERSION = 1
MAGIC = 0x464D  # 'FM'
HEADER_SIZE = 8
HEADER_STRUCT = struct.Struct("<HHHH")

# File kind tags.
KIND_TEXT = 1
KIND_CONTENT = 2
KIND_DOCTABLE = 3
KIND_BWT = 4
KIND_RANK = 5
KIND_RANKDIR = 6
KIND_SA_SAMPLE = 7
KIND_BYTESAMPLE = 8

MANIFEST_NAME = "manifest.json"
OPEN_MARKER = "OPEN"
STAGING_SUFFIX = ".staging"

# Unicode range that may legitimately appear in user text.
MAX_USER_CODEPOINT = 0x10FFFF
MIN_SENTINEL = MAX_USER_CODEPOINT + 1  # 0x110000: first internal symbol

DOCTABLE_RECORD = struct.Struct("<QQQQQII")  # 48 bytes
DOCTABLE_RECORD_SIZE = DOCTABLE_RECORD.size
RANKDIR_RECORD = struct.Struct("<QQQQ")  # 32 bytes
RANKDIR_RECORD_SIZE = RANKDIR_RECORD.size
SAMPLE_RECORD = struct.Struct("<IQ")  # 12 bytes
SAMPLE_RECORD_SIZE = SAMPLE_RECORD.size


class IndexCorruptError(RuntimeError):
    """Raised when an index directory is truncated, inconsistent or tampered."""


class CursorError(ValueError):
    """Raised when a paging cursor does not belong to this query/index."""


def write_header(f, kind: int) -> None:
    f.write(HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, kind, 0))


def read_header(f, expected_kind: int) -> None:
    raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise IndexCorruptError("file truncated inside header")
    magic, version, kind, reserved = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise IndexCorruptError("bad magic; not a diskfm file")
    if version != FORMAT_VERSION:
        raise IndexCorruptError(
            f"unsupported format version {version}; expected {FORMAT_VERSION}")
    if kind != expected_kind:
        raise IndexCorruptError(
            f"file kind {kind} does not match expected {expected_kind}")
    if reserved != 0:
        raise IndexCorruptError("non-zero reserved header field")


def hash_file(path: str) -> Dict[str, object]:
    """Stream-hash a file; returns sha256 hex digest and byte length."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def verify_file(path: str, entry: Dict[str, object]) -> None:
    if not os.path.exists(path):
        raise IndexCorruptError(f"missing index file: {os.path.basename(path)}")
    actual = hash_file(path)
    if actual["bytes"] != entry["bytes"]:
        raise IndexCorruptError(
            f"{os.path.basename(path)} has wrong length "
            f"({actual['bytes']} != {entry['bytes']}); index truncated/edited")
    if actual["sha256"] != entry["sha256"]:
        raise IndexCorruptError(
            f"{os.path.basename(path)} checksum mismatch; index corrupt")


def write_manifest(index_dir: str, manifest: Dict[str, object]) -> None:
    tmp = os.path.join(index_dir, MANIFEST_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, sort_keys=True, indent=1)
        f.write("\n")
    os.replace(tmp, os.path.join(index_dir, MANIFEST_NAME))


def validate_text_str(s: str) -> None:
    """Reject lone surrogate code points; everything else Unicode is legal."""
    for ch in s:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise ValueError(
                "input text contains an isolated surrogate code point "
                f"U+{cp:04X}; such data cannot be represented as Unicode text")


def validate_query(q: str) -> None:
    if not isinstance(q, str):
        raise TypeError("query must be a str")
    if len(q) == 0:
        raise ValueError("query must be non-empty")
    validate_text_str(q)

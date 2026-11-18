"""External merge sort for variable-length records.

The fixed-width :mod:`diskfm.extsort` handles the FM-index build; the
similarity catalog additionally needs to sort records whose keys are whole
qgram byte strings of arbitrary length (route: *length-prefixed full UTF-8
qgram as the external sort key*).  This module provides that, with the same
bounded-memory contract:

1. The in-memory buffer holds at most ``chunk_items`` records **and** at
   most ``chunk_bytes`` serialised bytes; crossing either limit spills one
   sorted run to a temporary file.  Memory therefore stays bounded even
   when individual keys are large (big ``q``).
2. Sorted runs are k-way merged with :func:`heapq.merge`, holding one head
   record per run: merge RAM is O(runs), not O(data).
3. Records on disk are framed as ``u32 payload_len + payload``; the codec
   only converts a record to/from its payload bytes.

Sorting order is the natural Python ordering of the decoded records (or a
caller-supplied ``key`` function applied to decoded records).  For qgram
relations the record is ``(qgram_utf8_bytes, doc_index)``: byte order of
UTF-8 coincides with Unicode code point order, and because the *complete*
byte string is the key, two distinct qgrams can never compare equal -- no
hash, no collision, no silent merge.
"""

from __future__ import annotations

import heapq
import os
import struct
import tempfile
from typing import Callable, Iterator, List, Optional

from .format import IndexCorruptError

_LEN = struct.Struct("<I")


class VarCodec:
    """Record <-> payload-bytes interface for :class:`VarExternalSorter`."""

    def encode(self, record) -> bytes:
        raise NotImplementedError

    def decode(self, payload: bytes):
        raise NotImplementedError


class U64Codec(VarCodec):
    """Record: one non-negative int < 2**64 (fixed 8-byte payload)."""

    _STRUCT = struct.Struct("<Q")

    def encode(self, record) -> bytes:
        return self._STRUCT.pack(record)

    def decode(self, payload: bytes):
        if len(payload) != 8:
            raise IndexCorruptError("u64 record has wrong length")
        return self._STRUCT.unpack(payload)[0]


class QGramRelCodec(VarCodec):
    """Record ``(qgram_utf8: bytes, doc_index: int)``.

    Payload: the qgram bytes followed by one little-endian u32 doc index.
    Natural tuple ordering sorts by qgram bytes first (code point order),
    then by document index -- exactly the grouping the dictionary builder
    needs.
    """

    _TAIL = struct.Struct("<I")

    def encode(self, record) -> bytes:
        qgram, doc_index = record
        return qgram + self._TAIL.pack(doc_index)

    def decode(self, payload: bytes):
        if len(payload) < 4:
            raise IndexCorruptError("qgram relation record too short")
        return payload[:-4], self._TAIL.unpack(payload[-4:])[0]


class ResultCodec(VarCodec):
    """Record ``(intersection, union, target_shingles, target_id_bytes)``.

    Used to sort query results; the caller supplies the exact ordering key
    (descending Jaccard ratio as a Fraction, then target doc_id bytes).
    """

    _HEAD = struct.Struct("<QQQ")

    def encode(self, record) -> bytes:
        inter, union, t_shingles, doc_id_bytes = record
        return self._HEAD.pack(inter, union, t_shingles) + doc_id_bytes

    def decode(self, payload: bytes):
        if len(payload) < self._HEAD.size:
            raise IndexCorruptError("result record too short")
        inter, union, t_shingles = self._HEAD.unpack(
            payload[: self._HEAD.size])
        return inter, union, t_shingles, payload[self._HEAD.size:]


def _record_stream(f, codec: VarCodec, read_size: int = 1 << 16) -> Iterator:
    """Yield decoded records from a framed run file, refilling a small
    byte buffer; a ragged tail is corruption, never silent data loss."""
    buf = bytearray()
    while True:
        chunk = f.read(read_size)
        if chunk:
            buf += chunk
        while len(buf) >= 4:
            (n,) = _LEN.unpack_from(buf, 0)
            if len(buf) < 4 + n:
                break
            yield codec.decode(bytes(buf[4:4 + n]))
            del buf[:4 + n]
        if not chunk:
            if buf:
                raise IndexCorruptError("ragged temporary run file")
            return


class VarSortResult:
    """Handle to a finished variable-length external sort."""

    def __init__(self, chunk_paths: List[str], codec: VarCodec,
                 key: Optional[Callable] = None):
        self._chunk_paths = chunk_paths
        self._codec = codec
        self._key = key
        self._closed = False

    def __iter__(self) -> Iterator:
        if self._closed:
            raise RuntimeError("variable sort result already cleaned up")
        streams = [open(p, "rb") for p in self._chunk_paths]
        try:
            iters = [_record_stream(f, self._codec) for f in streams]
            # One head record per run in the merge heap.
            yield from heapq.merge(*iters, key=self._key)
        finally:
            for f in streams:
                f.close()

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        for path in self._chunk_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def __enter__(self) -> "VarSortResult":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


class VarExternalSorter:
    """Accumulate variable-length records, spill sorted runs, merge.

    The final merge and any intermediate rounds keep at most
    ``merge_fan_in`` run files open at once: when more runs exist, they are
    merged in groups into intermediate runs first, so file-descriptor
    usage stays bounded no matter how small ``chunk_items`` is.
    """

    def __init__(self, tmpdir: str, codec: VarCodec,
                 chunk_items: int = 20000, chunk_bytes: int = 1 << 22,
                 key: Optional[Callable] = None, merge_fan_in: int = 128):
        if chunk_items < 1 or chunk_bytes < 64:
            raise ValueError("bad chunk geometry")
        if merge_fan_in < 2:
            raise ValueError("merge_fan_in must be >= 2")
        self.tmpdir = tmpdir
        self.codec = codec
        self.chunk_items = chunk_items
        self.chunk_bytes = chunk_bytes
        self._key = key
        self.merge_fan_in = merge_fan_in
        self._buf: List = []
        self._buf_bytes = 0
        self._chunks: List[str] = []
        self.items_seen = 0
        self.chunks_spilled = 0
        self.bytes_spilled = 0
        self.merge_passes = 0

    def add(self, record) -> None:
        payload = self.codec.encode(record)
        self._buf.append(payload)
        self._buf_bytes += len(payload) + 4
        self.items_seen += 1
        if (len(self._buf) >= self.chunk_items
                or self._buf_bytes >= self.chunk_bytes):
            self._spill()

    def _spill(self) -> None:
        if not self._buf:
            return
        # Always sort *decoded* records: raw payload order is not the record
        # order (one qgram may be a prefix of another, and the little-endian
        # trailer is not order-preserving), so framing bytes must never
        # participate in comparisons.
        records = [self.codec.decode(p) for p in self._buf]
        records.sort(key=self._key)  # key=None -> natural tuple order
        fd, path = tempfile.mkstemp(prefix="vsort_", dir=self.tmpdir)
        with os.fdopen(fd, "wb") as f:
            for record in records:
                payload = self.codec.encode(record)
                f.write(_LEN.pack(len(payload)))
                f.write(payload)
        self.bytes_spilled += os.path.getsize(path)
        self._chunks.append(path)
        self.chunks_spilled += 1
        self._buf = []
        self._buf_bytes = 0

    def _merge_group_to_file(self, paths: List[str], out_path: str) -> None:
        """Merge one group of sorted runs into a single sorted run file."""
        streams = [open(p, "rb") for p in paths]
        try:
            iters = [_record_stream(f, self.codec) for f in streams]
            with open(out_path, "wb") as out:
                for record in heapq.merge(*iters, key=self._key):
                    payload = self.codec.encode(record)
                    out.write(_LEN.pack(len(payload)))
                    out.write(payload)
        finally:
            for f in streams:
                f.close()
        for p in paths:
            os.unlink(p)

    def finalize(self) -> VarSortResult:
        if self._buf:
            self._spill()
        if not self._chunks:
            fd, path = tempfile.mkstemp(prefix="vsort_", dir=self.tmpdir)
            os.close(fd)
            self._chunks.append(path)
        # Bounded fan-in: collapse runs in groups until one final merge
        # (performed lazily by VarSortResult) needs no more than
        # merge_fan_in open files.
        while len(self._chunks) > self.merge_fan_in:
            self.merge_passes += 1
            collapsed: List[str] = []
            for i in range(0, len(self._chunks), self.merge_fan_in):
                group = self._chunks[i:i + self.merge_fan_in]
                if len(group) == 1:
                    collapsed.append(group[0])
                    continue
                fd, path = tempfile.mkstemp(prefix="vsort_mp_",
                                            dir=self.tmpdir)
                os.close(fd)
                self._merge_group_to_file(group, path)
                self.bytes_spilled += os.path.getsize(path)
                collapsed.append(path)
            self._chunks = collapsed
        if len(self._chunks) > 1:
            self.merge_passes += 1
        return VarSortResult(self._chunks, self.codec, self._key)

    def cleanup_discarded(self) -> None:
        self._buf = []
        self._buf_bytes = 0
        for path in self._chunks:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._chunks = []

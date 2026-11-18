"""External merge sort over fixed-width unsigned-integer records.

The whole package must keep its working-set size bounded while the corpus
grows.  This module is the only sorting primitive used while the suffix
order is constructed.  It implements a standard two-phase external merge
sort:

1. Repeatedly pull at most ``chunk_items`` records into a Python list, sort
   it in memory and flush the sorted run to a temporary file (a "chunk").
2. k-way merge the chunks with :mod:`heapq`, streaming the sorted result
   through a generator.  The heap holds one record per chunk, so merge RAM
   does not grow with the corpus either.

Records are plain Python integers in memory (or 3-tuples for the wide
layout) and are serialised with a pluggable fixed-width codec:

* :class:`Packed64Codec` -- a composite key packed into one 64-bit
  little-endian integer.  This is the small sort key used instead of
  copying whole suffix strings around.
* :class:`Triple32Codec` -- ``(key0, key1, payload)`` as three little-endian
  32-bit unsigned integers (12 bytes), the general prefix-doubling record
  used when the packed fields would not fit in 64 bits.

Nothing here ever materialises a suffix string; positions/ranks are the
only payloads.
"""

from __future__ import annotations

import heapq
import os
import struct
import tempfile
from typing import Iterable, Iterator, List, Sequence, Tuple


def pack_parts(bits: Sequence[int], vals: Sequence[int]) -> int:
    """Pack ``vals`` into one word using the given field bit widths (<=64)."""
    if sum(bits) > 64 or len(bits) != len(vals):
        raise ValueError("packed fields must fit within 64 bits")
    out = 0
    for width, value in zip(bits, vals):
        if not (0 <= value < (1 << width)):
            raise ValueError("packed value out of range for its field width")
        out = (out << width) | value
    return out


def unpack_parts(bits: Sequence[int], word: int) -> Tuple[int, ...]:
    out = []
    for width in reversed(bits):
        out.append(word & ((1 << width) - 1))
        word >>= width
    return tuple(reversed(out))


def fits_packed64(bits: Iterable[int], vals: Iterable[int]) -> bool:
    bits = list(bits)
    vals = list(vals)
    return sum(bits) <= 64 and all(0 <= v < (1 << b) for b, v in zip(bits, vals))


def bits_needed(max_value: int) -> int:
    """Bits required to store any value in [0, max_value] (>= 1)."""
    return max(1, (max_value + 1).bit_length())


class _Codec:
    itemsize = 0

    def encode_many(self, records: List) -> bytes:
        raise NotImplementedError

    def decode_many(self, raw: bytes) -> List:
        raise NotImplementedError


class Packed64Codec(_Codec):
    """One 64-bit little-endian word per record (record is an int)."""

    itemsize = 8
    _struct = struct.Struct("<Q")

    def encode_many(self, records: List[int]) -> bytes:
        return b"".join(self._struct.pack(r) for r in records)

    def decode_many(self, raw: bytes) -> List[int]:
        n = len(raw) // 8
        return [v for (v,) in self._struct.iter_unpack(raw[: n * 8])]


class Triple32Codec(_Codec):
    """Three 32-bit little-endian words per record (record is a 3-tuple)."""

    itemsize = 12
    _struct = struct.Struct("<III")

    def encode_many(self, records: List[tuple]) -> bytes:
        return b"".join(self._struct.pack(*r) for r in records)

    def decode_many(self, raw: bytes) -> List[tuple]:
        n = len(raw) // 12
        return list(self._struct.iter_unpack(raw[: n * 12]))


def _chunk_stream(f, codec: _Codec, refill_records: int) -> Iterator:
    """Yield records from a chunk file, refilling a small byte buffer."""
    buf_size = refill_records * codec.itemsize
    while True:
        raw = f.read(buf_size)
        if not raw:
            return
        for rec in codec.decode_many(raw):
            yield rec
        if len(raw) < buf_size:
            return


class ExternalSortResult:
    """Handle to a finished external sort; iterate to read the sorted run."""

    def __init__(self, chunk_paths: List[str], codec: _Codec, tmpdir: str):
        self._chunk_paths = chunk_paths
        self._codec = codec
        self._tmpdir = tmpdir
        self._closed = False

    @property
    def codec(self) -> _Codec:
        return self._codec

    def __iter__(self) -> Iterator:
        if self._closed:
            raise RuntimeError("external sort result already cleaned up")
        # One open file per chunk; the merge heap only holds one head record
        # per chunk, so merge RAM is O(number of chunks) not O(corpus).
        streams = [open(p, "rb") for p in self._chunk_paths]
        try:
            iters = [_chunk_stream(f, self._codec, 4096) for f in streams]
            yield from heapq.merge(*iters)
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

    def __enter__(self) -> "ExternalSortResult":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


class ExternalSorter:
    """Accumulate records, spill sorted chunks, then merge on demand."""

    def __init__(self, tmpdir: str, codec: _Codec = None,
                 chunk_items: int = 20000):
        self.tmpdir = tmpdir
        self.codec = codec or Packed64Codec()
        self.chunk_items = chunk_items
        self._buf: List = []
        self._chunks: List[str] = []
        self.items_seen = 0
        self.chunks_spilled = 0
        self.bytes_spilled = 0
        self.merge_passes = 0

    def add(self, record) -> None:
        self._buf.append(record)
        self.items_seen += 1
        if len(self._buf) >= self.chunk_items:
            self._spill()

    def add_many(self, records: Iterable) -> None:
        for r in records:
            self.add(r)

    def _spill(self) -> None:
        if not self._buf:
            return
        self._buf.sort()
        fd, path = tempfile.mkstemp(prefix="esort_", dir=self.tmpdir)
        with os.fdopen(fd, "wb") as f:
            # Encode in bounded batches so the encoded blob stays small.
            batch = 4096
            for i in range(0, len(self._buf), batch):
                f.write(self.codec.encode_many(self._buf[i : i + batch]))
        self.bytes_spilled += os.path.getsize(path)
        self._chunks.append(path)
        self.chunks_spilled += 1
        self._buf = []

    def finalize(self) -> ExternalSortResult:
        """Flush the last buffer and return a mergeable sorted result."""
        if self._buf:
            self._spill()
        if not self._chunks:
            fd, path = tempfile.mkstemp(prefix="esort_", dir=self.tmpdir)
            os.close(fd)
            self._chunks.append(path)
        if len(self._chunks) > 1:
            self.merge_passes = 1
        return ExternalSortResult(self._chunks, self.codec, self.tmpdir)

    def cleanup_discarded(self) -> None:
        """Drop everything: used when a sort's output is no longer needed."""
        self._buf = []
        for path in self._chunks:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._chunks = []


def external_sorted(records: Iterable, tmpdir: str,
                    codec: _Codec = None, chunk_items: int = 20000
                    ) -> ExternalSortResult:
    """Convenience wrapper: push an iterable through one external sort."""
    sorter = ExternalSorter(tmpdir, codec, chunk_items)
    sorter.add_many(records)
    return sorter.finalize()

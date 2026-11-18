"""Bounded LRU paging over fixed-width on-disk arrays.

Both the builder (random access to the position-indexed rank file) and the
query path (binary search inside the on-disk occurrence lists) need random
reads over arrays far larger than RAM.  This pager maps a file into fixed
size pages kept in an LRU set of at most ``max_pages`` entries: resident
memory is therefore capped at ``max_pages * page_bytes`` regardless of the
file size, and every miss is counted so tests can show that locate does not
read whole structures.
"""

from __future__ import annotations

import struct
from collections import OrderedDict
from typing import Dict


class FilePager:
    def __init__(self, path: str, record_size: int, header_size: int,
                 page_records: int = 4096, max_pages: int = 16):
        if record_size < 1 or page_records < 1 or max_pages < 2:
            raise ValueError("bad pager geometry")
        self.path = path
        self.record_size = record_size
        self.header_size = header_size
        self.page_records = page_records
        self.page_bytes = page_records * record_size
        self.max_pages = max_pages
        self._f = open(path, "rb")
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self.page_hits = 0
        self.page_misses = 0
        self.bytes_read = 0

    def _load_page(self, page_no: int) -> bytes:
        page = self._cache.get(page_no)
        if page is not None:
            self._cache.move_to_end(page_no)
            self.page_hits += 1
            return page
        self.page_misses += 1
        self._f.seek(self.header_size + page_no * self.page_bytes)
        page = self._f.read(self.page_bytes)
        self.bytes_read += len(page)
        self._cache[page_no] = page
        self._cache.move_to_end(page_no)
        while len(self._cache) > self.max_pages:
            self._cache.popitem(last=False)
        return page

    def read_record(self, index: int) -> bytes:
        """Return the raw record at logical index ``index``."""
        if index < 0:
            raise IndexError("negative record index")
        page_no, off = divmod(index, self.page_records)
        page = self._load_page(page_no)
        start = off * self.record_size
        raw = page[start : start + self.record_size]
        if len(raw) != self.record_size:
            raise IndexError("read past end of paged file")
        return raw

    def get_u32(self, index: int) -> int:
        return struct.unpack("<I", self.read_record(index))[0]

    def get_u64(self, index: int) -> int:
        return struct.unpack("<Q", self.read_record(index))[0]

    def stats(self) -> Dict[str, int]:
        return {
            "page_hits": self.page_hits,
            "page_misses": self.page_misses,
            "bytes_read": self.bytes_read,
            "resident_pages": len(self._cache),
            "resident_bytes": len(self._cache) * self.page_bytes,
        }

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "FilePager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

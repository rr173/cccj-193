"""External-memory construction of the disk FM-index.

Build route (everything streamed through :mod:`diskfm.extsort`, never a
full suffix array in RAM and never a whole suffix copied as a sort key):

1. Ingest JSONL once.  The global symbol stream
   ``T = doc0 + S0 + doc1 + S1 + ...`` is written as u32 symbols to
   ``text.bin``; original UTF-8 bytes are stored verbatim in
   ``content.bin`` together with per-document coordinate and byte-offset
   tables.  ``Sd = 0x110000 + d`` are internal sentinels, strictly above
   the largest legal Unicode code point (U+10FFFF), so they can never
   collide with user data.
2. Prefix doubling with two external sorts per round:

   * round 0 sorts packed ``(T[p], p)`` records to get length-1 ranks;
   * each later round emits ``(rank[p], rank[(p+k) mod N], p)`` triples,
     sorts them externally, re-numbers distinct pairs, then externally
     sorts ``(p, new_rank)`` to restore the position-indexed rank file.

   As soon as every rank is distinct the sorted positions *are* the suffix
   array; construction stops (early on repetitive text).
3. One streaming SA pass emits the BWT (cyclic rule
   ``BWT[row] = T[(pos-1) mod N]``) and the sampled SA positions.
4. One external sort of ``(symbol, row)`` pairs builds the on-disk
   occurrence lists that back every rank query.

RAM is governed by one knob, ``chunk_items``: at most that many records
sit in the in-memory sort buffer, every merge keeps one head record per
chunk, and all random access goes through bounded pread buffers -- RSS does
not grow with the corpus.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import shutil
import struct
import uuid
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from . import format as fmt
from .extsort import (
    ExternalSorter,
    Packed64Codec,
    Triple32Codec,
    _Codec,
    bits_needed,
    pack_parts,
    unpack_parts,
)
from .pager import FilePager

# Rough bytes per buffered record when translating a memory budget.
_BYTES_PER_RECORD = 40
# Codepoint distance between byte-offset checkpoints inside a document.
BYTE_SAMPLE_INTERVAL = 64


@dataclass
class BuildStats:
    documents: int = 0
    symbols: int = 0
    doubling_rounds: int = 0
    external_sorts: int = 0
    chunks_spilled: int = 0
    bytes_spilled: int = 0
    widest_codec: str = "Packed64"
    early_stop: bool = False

    def as_dict(self) -> Dict[str, int]:
        return {
            "documents": self.documents,
            "symbols": self.symbols,
            "doubling_rounds": self.doubling_rounds,
            "external_sorts": self.external_sorts,
            "chunks_spilled": self.chunks_spilled,
            "bytes_spilled": self.bytes_spilled,
            "widest_codec": self.widest_codec,
            "early_stop": self.early_stop,
        }


class _U32Random:
    """Random u32 reads through a bounded LRU pager (RAM stays corpus-independent)."""

    def __init__(self, path: str, max_pages: int = 8, header_size: int = 0):
        self._pager = FilePager(path, record_size=4,
                                header_size=header_size,
                                page_records=4096, max_pages=max_pages)

    def get(self, index: int) -> int:
        return self._pager.get_u32(index)

    def close(self) -> None:
        self._pager.close()


def _iter_jsonl(path: str) -> Iterator[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict) or "doc_id" not in obj or "text" not in obj:
                raise ValueError(
                    f"{path}:{line_no}: every line must be a JSON object "
                    "with doc_id and text")
            doc_id = obj["doc_id"]
            text = obj["text"]
            if not isinstance(doc_id, str) or not isinstance(text, str):
                raise ValueError(
                    f"{path}:{line_no}: doc_id and text must both be strings")
            yield doc_id, text


@dataclass
class _DocRecord:
    sentinel_pos: int
    t_start: int
    content_start: int
    t_len: int
    byte_cp_start: int  # index of this doc's first checkpoint in bytesample.bin
    id_blob_offset: int  # offset inside the trailing doc_id blob
    id_blob: bytes


class IndexBuilder:
    """Stream an unordered JSONL corpus into a closed, reopenable index."""

    def __init__(self, index_dir: str, chunk_items: int = 20000,
                 sample_interval: int = 32,
                 byte_sample_interval: int = BYTE_SAMPLE_INTERVAL,
                 force_wide: bool = False):
        if sample_interval < 1:
            raise ValueError("sample_interval must be >= 1")
        if byte_sample_interval < 1:
            raise ValueError("byte_sample_interval must be >= 1")
        if chunk_items < 1:
            raise ValueError("chunk_items must be >= 1")
        self.index_dir = index_dir
        self.staging = index_dir + fmt.STAGING_SUFFIX
        self.chunk_items = chunk_items
        self.sample_interval = sample_interval
        self.byte_sample_interval = byte_sample_interval
        self.force_wide = force_wide
        self.stats = BuildStats()
        if os.path.exists(index_dir):
            raise FileExistsError(f"index directory already exists: {index_dir}")
        if os.path.exists(self.staging):
            shutil.rmtree(self.staging)
        os.makedirs(self.staging)
        self._open_outputs()

    # ------------------------------------------------------------------ #
    # Phase 1: ingestion
    # ------------------------------------------------------------------ #
    def _open_outputs(self) -> None:
        self.text_f = open(os.path.join(self.staging, "text.bin"), "wb")
        fmt.write_header(self.text_f, fmt.KIND_TEXT)
        self.content_f = open(os.path.join(self.staging, "content.bin"), "wb")
        fmt.write_header(self.content_f, fmt.KIND_CONTENT)
        self.bytesample_f = open(
            os.path.join(self.staging, "bytesample.bin"), "wb")
        fmt.write_header(self.bytesample_f, fmt.KIND_BYTESAMPLE)
        self._docs: List[_DocRecord] = []
        self._seen_ids: set = set()
        self._n = 0
        self._id_blob_cursor = 0
        # Per-build instance identity: cursors bind to this exact index,
        # so two independently built directories never share a token even
        # when their contents are byte-identical.
        self._instance_id = uuid.uuid4().hex
        # Content fingerprint is separate and still recorded for audits.
        self._identity = hashlib.sha256()
        self._sym_buf = array.array("I")
        self._byte_cp_count = 0

    def _flush_symbols(self) -> None:
        if self._sym_buf:
            self._sym_buf.tofile(self.text_f)
            self._sym_buf = array.array("I")

    def add_document(self, doc_id: str, text: str) -> None:
        fmt.validate_text_str(text)
        if doc_id in self._seen_ids:
            raise ValueError(f"duplicate doc_id: {doc_id!r}")
        self._seen_ids.add(doc_id)
        d = len(self._docs)
        sentinel = fmt.MIN_SENTINEL + d
        if sentinel > 0xFFFFFFFF or self._n + len(text) + 1 > 0xFFFFFFFF:
            raise ValueError("corpus exceeds the u32 geometry of this format")

        id_blob = doc_id.encode("utf-8")
        raw = text.encode("utf-8")
        t_start = self._n
        # Store body-relative offsets (the 8-byte file header is excluded);
        # readers add fmt.HEADER_SIZE when seeking the real file.
        content_start = self.content_f.tell() - fmt.HEADER_SIZE
        self.content_f.write(raw)
        self._identity.update(raw)
        self._identity.update(b"\x00DOC\x00")
        self._identity.update(id_blob)

        # Byte-offset checkpoints: codepoint-local offsets at multiples of
        # byte_sample_interval, plus one at offset 0.  Always u64.
        byte_cp_start = self._byte_cp_count
        cp_index = 0
        byte_acc = 0
        checkpoint = array.array("Q", [0])
        for ch in text:
            if cp_index and cp_index % self.byte_sample_interval == 0:
                checkpoint.append(byte_acc)
            byte_acc += len(ch.encode("utf-8"))
            cp_index += 1
        checkpoint.tofile(self.bytesample_f)
        self._byte_cp_count += len(checkpoint)

        for ch in text:
            self._sym_buf.append(ord(ch))
            if len(self._sym_buf) >= 8192:
                self._flush_symbols()
        self._flush_symbols()
        self._sym_buf.append(sentinel)
        self._flush_symbols()

        self._docs.append(
            _DocRecord(sentinel_pos=self._n + len(text), t_start=t_start,
                       content_start=content_start, t_len=len(text),
                       byte_cp_start=byte_cp_start,
                       id_blob_offset=self._id_blob_cursor,
                       id_blob=id_blob))
        self._id_blob_cursor += len(id_blob)
        self._n += len(text) + 1
        self.stats.documents += 1
        self.stats.symbols = self._n

    def ingest_jsonl(self, path: str) -> None:
        for doc_id, text in _iter_jsonl(path):
            self.add_document(doc_id, text)

    # ------------------------------------------------------------------ #
    # External sort helpers with bookkeeping
    # ------------------------------------------------------------------ #
    def _new_sorter(self, codec: _Codec) -> ExternalSorter:
        s = ExternalSorter(self.staging, codec, self.chunk_items)
        return s

    def _track(self, sorter: ExternalSorter) -> None:
        self.stats.external_sorts += 1
        self.stats.chunks_spilled += sorter.chunks_spilled
        self.stats.bytes_spilled += sorter.bytes_spilled
        if isinstance(sorter.codec, Triple32Codec):
            self.stats.widest_codec = "Triple32"

    # ------------------------------------------------------------------ #
    # Phase 2: prefix doubling
    # ------------------------------------------------------------------ #
    def _build_initial_ranks(self) -> Tuple[str, int]:
        """Round 0: sort (symbol, position); write position-indexed rank file."""
        n = self._n
        max_symbol = fmt.MIN_SENTINEL + max(0, len(self._docs) - 1)
        sym_bits = bits_needed(max_symbol)
        pos_bits = bits_needed(max(0, n - 1))
        packed = (not self.force_wide) and (sym_bits + pos_bits) <= 64
        codec: _Codec = Packed64Codec() if packed else Triple32Codec()
        if not packed:
            self.stats.widest_codec = "Triple32"
        sorter = self._new_sorter(codec)
        text_rand = _U32Random(os.path.join(self.staging, "text.bin"),
                                 header_size=fmt.HEADER_SIZE)
        try:
            for p in range(n):
                c = text_rand.get(p)
                if packed:
                    sorter.add(pack_parts([sym_bits, pos_bits], [c, p]))
                else:
                    sorter.add((c, p, 0))
        finally:
            text_rand.close()
        result = sorter.finalize()
        self._track(sorter)

        rank_sorted_path = os.path.join(self.staging, "r0_sorted.bin")
        # Merge (symbol, p) -> assign dense ranks, emit (p, rank) packed.
        rank_bits = bits_needed(max(0, n - 1))
        inner = Packed64Codec()
        inner_sorter = self._new_sorter(inner)
        prev_key = None
        rank = -1
        with result, open(rank_sorted_path, "wb") as out:
            for rec in result:
                if packed:
                    c, p = unpack_parts([sym_bits, pos_bits], rec)
                else:
                    c, p, _ = rec
                if prev_key is None or c != prev_key:
                    rank += 1
                    prev_key = c
                out.write(struct.pack("<II", p, rank))
                inner_sorter.add(pack_parts([pos_bits, rank_bits], [p, rank]))
        result.cleanup()
        # Second external sort of the round: restore position order.
        pos_result = inner_sorter.finalize()
        self._track(inner_sorter)
        rank_path = os.path.join(self.staging, "rank_0.bin")
        distinct = rank + 1
        with pos_result, open(rank_path, "wb") as rf:
            for word in pos_result:
                p, r = unpack_parts([pos_bits, rank_bits], word)
                rf.write(struct.pack("<I", r))
        pos_result.cleanup()
        os.unlink(rank_sorted_path)
        return rank_path, distinct

    def _doubling_codec(self, rank_bits: int, n: int) -> Tuple[_Codec, bool]:
        pos_bits = bits_needed(max(0, n - 1))
        triple_bits = [rank_bits, rank_bits, pos_bits]
        if not self.force_wide and sum(triple_bits) <= 64:
            return Packed64Codec(), True
        return Triple32Codec(), False

    def build(self) -> Dict[str, object]:
        # All ingestion writers must hit disk before random reads start.
        self.text_f.flush()
        self.content_f.flush()
        self.bytesample_f.flush()
        if self._n == 0:
            return self._finalize_empty()
        n = self._n
        pos_bits = bits_needed(n - 1)

        rank_path, distinct = self._build_initial_ranks()
        round_no = 0
        k = 1
        sa_path = os.path.join(self.staging, "sa.bin")

        while distinct < n:
            round_no += 1
            if k >= n:
                raise fmt.IndexCorruptError(
                    "prefix doubling failed to separate all suffixes")
            rank_bits = bits_needed(distinct - 1)
            codec, packed = self._doubling_codec(rank_bits, n)
            sorter = self._new_sorter(codec)
            rank_rand = _U32Random(rank_path)
            try:
                for p in range(n):
                    r1 = rank_rand.get(p)
                    r2 = rank_rand.get((p + k) % n)
                    if packed:
                        sorter.add(pack_parts(
                            [rank_bits, rank_bits, pos_bits], [r1, r2, p]))
                    else:
                        sorter.add((r1, r2, p))
            finally:
                rank_rand.close()
            result = sorter.finalize()
            self._track(sorter)

            new_rank_path = os.path.join(self.staging, f"rank_{round_no}.bin")
            # Sort B of the round: (p, new rank) back into position order.
            inner = Packed64Codec()
            inner_sorter = self._new_sorter(inner)
            prev_pair = None
            new_rank = -1
            sa_order = array.array("I")
            with result:
                for rec in result:
                    if packed:
                        r1, r2, p = unpack_parts(
                            [rank_bits, rank_bits, pos_bits], rec)
                    else:
                        r1, r2, p = rec
                    pair = (r1, r2)
                    if prev_pair is None or pair != prev_pair:
                        new_rank += 1
                        prev_pair = pair
                    inner_sorter.add(pack_parts(
                        [pos_bits, bits_needed(n - 1)], [p, new_rank]))
                    sa_order.append(p)
            result.cleanup()
            new_distinct = new_rank + 1

            if new_distinct == n:
                # The merge order already is the suffix array: keep it and
                # skip the position-order restoration sort entirely.
                with open(sa_path, "wb") as sf:
                    sa_order.tofile(sf)
                inner_sorter.cleanup_discarded()
                self.stats.early_stop = True
                distinct = new_distinct
                break

            pos_result = inner_sorter.finalize()
            self._track(inner_sorter)
            with pos_result, open(new_rank_path, "wb") as rf:
                for word in pos_result:
                    _p, r = unpack_parts(
                        [pos_bits, bits_needed(n - 1)], word)
                    rf.write(struct.pack("<I", r))
            pos_result.cleanup()
            os.unlink(rank_path)
            rank_path = new_rank_path
            distinct = new_distinct
            k *= 2

        self.stats.doubling_rounds = round_no
        if distinct != n:
            raise fmt.IndexCorruptError("suffix order construction incomplete")
        if not os.path.exists(sa_path):
            # Round-0 ranks already distinct (N <= alphabet): rebuild SA order.
            self._materialize_sa_from_ranks(rank_path, sa_path)
        self._emit_bwt_samples_and_rank(sa_path)
        for p in (rank_path, sa_path):
            if os.path.exists(p):
                os.unlink(p)
        return self._finalize()

    def _materialize_sa_from_ranks(self, rank_path: str, sa_path: str) -> None:
        """Fallback: distinct position ranks already define suffix order."""
        n = self._n
        pos_bits = bits_needed(n - 1)
        rank_bits = bits_needed(n - 1)
        sorter = self._new_sorter(Packed64Codec())
        rank_rand = _U32Random(rank_path)
        try:
            for p in range(n):
                r = rank_rand.get(p)
                sorter.add(pack_parts([rank_bits, pos_bits], [r, p]))
        finally:
            rank_rand.close()
        result = sorter.finalize()
        self._track(sorter)
        with result, open(sa_path, "wb") as sf:
            out = array.array("I")
            for word in result:
                _r, p = unpack_parts([rank_bits, pos_bits], word)
                out.append(p)
                if len(out) >= 8192:
                    out.tofile(sf)
                    out = array.array("I")
            out.tofile(sf)
        result.cleanup()

    # ------------------------------------------------------------------ #
    # Phase 3: BWT + sampled SA + on-disk occurrence lists
    # ------------------------------------------------------------------ #
    def _emit_bwt_samples_and_rank(self, sa_path: str) -> None:
        n = self._n
        text_rand = _U32Random(os.path.join(self.staging, "text.bin"),
                                 header_size=fmt.HEADER_SIZE)
        bwt_path = os.path.join(self.staging, "bwt.bin")
        sample_path = os.path.join(self.staging, "sa_sample.bin")
        # (symbol, row) pairs are themselves external-sorted below, so the
        # rank structure is built with bounded RAM too.
        sym_bits = bits_needed(fmt.MIN_SENTINEL + len(self._docs) - 1)
        row_bits = bits_needed(n - 1)
        packed = (sym_bits + row_bits) <= 64
        occ_sorter = self._new_sorter(Packed64Codec() if packed else
                                      Triple32Codec())
        if not packed:
            self.stats.widest_codec = "Triple32"
        try:
            with open(sa_path, "rb") as sa_f, \
                    open(bwt_path, "wb") as bwt_f, \
                    open(sample_path, "wb") as sample_f:
                fmt.write_header(bwt_f, fmt.KIND_BWT)
                fmt.write_header(sample_f, fmt.KIND_SA_SAMPLE)
                bwt_buf = array.array("I")
                row = 0
                while True:
                    block = sa_f.read(8192 * 4)
                    if not block:
                        break
                    positions = array.array("I")
                    positions.frombytes(block)
                    for pos in positions:
                        bwt_sym = text_rand.get((pos - 1) % n)
                        bwt_buf.append(bwt_sym)
                        if packed:
                            occ_sorter.add(pack_parts(
                                [sym_bits, row_bits], [bwt_sym, row]))
                        else:
                            occ_sorter.add((bwt_sym, row, 0))
                        if row % self.sample_interval == 0:
                            sample_f.write(
                                fmt.SAMPLE_RECORD.pack(row, pos))
                        row += 1
                        if len(bwt_buf) >= 8192:
                            bwt_buf.tofile(bwt_f)
                            bwt_buf = array.array("I")
                bwt_buf.tofile(bwt_f)
                if row != n:
                    raise fmt.IndexCorruptError("SA length mismatch")
        finally:
            text_rand.close()

        occ_result = occ_sorter.finalize()
        self._track(occ_sorter)
        rank_path = os.path.join(self.staging, "rank.bin")
        rankdir_path = os.path.join(self.staging, "rankdir.bin")
        with occ_result, open(rank_path, "wb") as rf, \
                open(rankdir_path, "wb") as df:
            fmt.write_header(rf, fmt.KIND_RANK)
            fmt.write_header(df, fmt.KIND_RANKDIR)
            current_symbol = None
            offset = 0
            count = 0
            cumulative = 0  # total occurrences of smaller symbols
            row_buf = array.array("I")
            dir_records = []

            def _flush_symbol():
                nonlocal row_buf, offset, count, cumulative
                if current_symbol is None:
                    return
                row_buf.tofile(rf)
                # 4th field: C(current_symbol) = strictly-smaller total.
                dir_records.append(fmt.RANKDIR_RECORD.pack(
                    current_symbol, fmt.HEADER_SIZE + offset, count,
                    cumulative))
                offset += count * 4
                cumulative += count
                count = 0
                row_buf = array.array("I")

            for rec in occ_result:
                if packed:
                    symbol, r = unpack_parts([sym_bits, row_bits], rec)
                else:
                    symbol, r, _ = rec
                if current_symbol is None:
                    current_symbol = symbol
                elif symbol != current_symbol:
                    _flush_symbol()
                    current_symbol = symbol
                row_buf.append(r)
                count += 1
                if len(row_buf) >= 8192:
                    row_buf.tofile(rf)
                    row_buf = array.array("I")
            _flush_symbol()
            for rec_bytes in dir_records:
                df.write(rec_bytes)
        occ_result.cleanup()

    # ------------------------------------------------------------------ #
    # Finalize: doc table, manifest, completion marker
    # ------------------------------------------------------------------ #
    def _write_doctable(self) -> None:
        path = os.path.join(self.staging, "doctable.bin")
        id_blob = b"".join(d.id_blob for d in self._docs)
        with open(path, "wb") as f:
            fmt.write_header(f, fmt.KIND_DOCTABLE)
            for d in self._docs:
                f.write(fmt.DOCTABLE_RECORD.pack(
                    d.sentinel_pos, d.t_start, d.content_start, d.t_len,
                    d.id_blob_offset, len(d.id_blob), d.byte_cp_start))
            f.write(id_blob)
            if self._id_blob_cursor != len(id_blob):
                raise fmt.IndexCorruptError("doc id blob length mismatch")

    def _finalize_empty(self) -> Dict[str, object]:
        # Empty corpus: legal but trivially matches nothing.
        self.text_f.close()
        self.content_f.close()
        self.bytesample_f.close()
        for name, kind in (("bwt.bin", fmt.KIND_BWT),
                           ("rank.bin", fmt.KIND_RANK),
                           ("rankdir.bin", fmt.KIND_RANKDIR),
                           ("sa_sample.bin", fmt.KIND_SA_SAMPLE)):
            with open(os.path.join(self.staging, name), "wb") as f:
                fmt.write_header(f, kind)
        return self._finalize()

    def _finalize(self) -> Dict[str, object]:
        for f in (self.text_f, self.content_f, self.bytesample_f):
            if not f.closed:
                f.close()
        self._write_doctable()

        files = {}
        for name in sorted(os.listdir(self.staging)):
            p = os.path.join(self.staging, name)
            if os.path.isfile(p):
                files[name] = fmt.hash_file(p)
        identity = self._identity.hexdigest()
        manifest = {
            "format_version": fmt.FORMAT_VERSION,
            "identity": self._instance_id,
            "content_hash": identity,
            "geometry": {
                "n_symbols": self._n,
                "n_documents": len(self._docs),
                "sample_interval": self.sample_interval,
                "byte_sample_interval": self.byte_sample_interval,
                "min_sentinel": fmt.MIN_SENTINEL,
            },
            "build_stats": self.stats.as_dict(),
            "files": files,
        }
        fmt.write_manifest(self.staging, manifest)
        # Completion marker last, with fsync: its absence means "unfinished".
        marker = os.path.join(self.staging, fmt.OPEN_MARKER)
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.staging, self.index_dir)
        return manifest

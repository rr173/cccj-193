"""External-memory construction of the document similarity catalog.

Input is a *sealed* diskfm index (its OPEN marker, manifest, per-file
hashes, doc-table geometry and content hash are all validated on open; the
source directory is only ever read, never modified).  Output is an
independent, re-openable catalog directory holding a disk-resident inverted
qgram index:

1. **Ingest** -- stream ``(doc_id, text)`` from the source index once.  For
   every document emit one ``(qgram_utf8, doc_index)`` record per q-gram
   occurrence into a bounded variable-length external sort.  The full
   length-delimited UTF-8 qgram is the sort key: no hashing, so distinct
   qgrams can never silently merge.  While streaming, recompute the source
   builder's content hash and compare it against the sealed manifest, and
   compute an independent document-set digest.
2. **Dedupe + dictionary + postings** -- one merge pass over the sorted
   relations groups identical ``(qgram, doc)`` pairs: each distinct qgram
   (in UTF-8 byte order = code point order) is appended to ``qgrams.bin``
   and gets a ``qgramdir.bin`` record; each distinct ``(qgram, doc)`` pair
   appends one ``(doc, term_freq)`` posting entry to ``postings.bin`` and
   one forward relation ``(doc_index, qgram_id)`` to a second external
   sort.  Duplicate shingles inside one document raise that document's
   term_freq but are counted exactly once for Jaccard.
3. **Forward relations + doc stats** -- merging the second sort yields each
   document's sorted distinct qgram ids, streamed to ``docqgrams.bin``;
   per-document statistics go to ``docs.bin`` (shingle count, offset, text
   length, id blob offset, flags).  Documents shorter than q are flagged
   ``FLAG_NO_SHINGLES`` -- explicitly not comparable, never silently zero.

Memory stays bounded: both sorts spill after ``chunk_items`` records or
``chunk_bytes`` serialised bytes, merges hold one head record per run, and
postings/dictionary/docqgrams are written sequentially as the merge
streams.  The catalog is published atomically: everything is built inside
``<catalog>.staging``, the OPEN marker is fsynced last, and the staging
directory is renamed over no existing target.  Any failure removes the
staging tree; a pre-existing valid catalog is never touched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from . import format as fmt
from . import simformat as sfmt
from .extsort import pack_parts, unpack_parts
from .query import DiskFMIndex
from .vsort import QGramRelCodec, U64Codec, VarExternalSorter

_META_FRAME = struct.Struct("<QQ")  # text_len_codepoints, raw_relation_count
_POSTING_BUF_RECORDS = 8192
_DOCQGRAM_BUF_RECORDS = 8192


def _peak_rss_kb() -> Optional[int]:
    """Peak resident set size (VmHWM) of this process; None if unavailable."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


@dataclass
class SimBuildStats:
    documents: int = 0
    documents_with_shingles: int = 0
    documents_without_shingles: int = 0
    raw_relations: int = 0
    deduped_relations: int = 0
    dictionary_qgrams: int = 0
    posting_entries: int = 0
    relation_sort_chunks: int = 0
    relation_sort_bytes: int = 0
    relation_merge_passes: int = 0
    forward_sort_chunks: int = 0
    forward_sort_bytes: int = 0
    forward_merge_passes: int = 0
    source_content_bytes_read: int = 0
    peak_rss_kb: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        out = {
            "documents": self.documents,
            "documents_with_shingles": self.documents_with_shingles,
            "documents_without_shingles": self.documents_without_shingles,
            "raw_relations": self.raw_relations,
            "deduped_relations": self.deduped_relations,
            "dictionary_qgrams": self.dictionary_qgrams,
            "posting_entries": self.posting_entries,
            "external_sort_runs": (self.relation_sort_chunks
                                   + self.forward_sort_chunks),
            "temp_bytes": (self.relation_sort_bytes
                           + self.forward_sort_bytes),
            "relation_sort": {
                "chunks_spilled": self.relation_sort_chunks,
                "bytes_spilled": self.relation_sort_bytes,
                "merge_passes": self.relation_merge_passes,
            },
            "forward_sort": {
                "chunks_spilled": self.forward_sort_chunks,
                "bytes_spilled": self.forward_sort_bytes,
                "merge_passes": self.forward_merge_passes,
            },
            "source_content_bytes_read": self.source_content_bytes_read,
            "peak_rss_kb": self.peak_rss_kb,
        }
        return out


class SimilarityBuilder:
    """Build a qgram-posting similarity catalog from a sealed diskfm index."""

    def __init__(self, source_index_dir: str, catalog_dir: str, q: int,
                 chunk_items: int = 20000, chunk_bytes: int = 1 << 22,
                 expect_identity: Optional[str] = None):
        if not isinstance(q, int) or isinstance(q, bool) or q < 1:
            raise ValueError("q must be a positive integer")
        if chunk_items < 1:
            raise ValueError("chunk_items must be >= 1")
        if chunk_bytes < 64:
            raise ValueError("chunk_bytes must be >= 64")
        self.q = q
        self.catalog_dir = catalog_dir
        self.staging = catalog_dir + sfmt.STAGING_SUFFIX
        self.chunk_items = chunk_items
        self.chunk_bytes = chunk_bytes
        self.stats = SimBuildStats()
        # Opening the source validates its OPEN marker, manifest, every
        # file hash and the doc-table/rank geometry; it is opened read-only
        # and never modified.
        self.source = DiskFMIndex(source_index_dir)
        if expect_identity is not None and \
                self.source.identity != expect_identity:
            self.source.close()
            raise ValueError(
                "source index identity does not match the expected identity")
        if os.path.exists(catalog_dir):
            self.source.close()
            raise FileExistsError(
                f"catalog directory already exists: {catalog_dir}")
        if os.path.exists(self.staging):
            shutil.rmtree(self.staging)
        os.makedirs(self.staging)
        self._built = False

    # ------------------------------------------------------------------ #
    # Phase 1: stream documents, emit raw (qgram, doc) relations
    # ------------------------------------------------------------------ #
    def _ingest(self, sorter: VarExternalSorter, meta_path: str) -> None:
        q = self.q
        # Recomputation of the source builder's content hash proves the
        # streamed documents match the sealed manifest exactly.
        source_hash = hashlib.sha256()
        # Independent digest of the (doc_id, text) stream, recorded in the
        # catalog manifest as the document-set identity.
        doc_digest = hashlib.sha256()
        with open(meta_path, "wb") as meta_f:
            for doc_idx, (doc_id, text) in enumerate(
                    self.source.iter_documents()):
                id_blob = doc_id.encode("utf-8")
                raw = text.encode("utf-8")
                source_hash.update(raw)
                source_hash.update(b"\x00DOC\x00")
                source_hash.update(id_blob)
                doc_digest.update(struct.pack("<I", len(id_blob)))
                doc_digest.update(id_blob)
                doc_digest.update(struct.pack("<Q", len(raw)))
                doc_digest.update(raw)

                text_len = len(text)
                raw_count = 0
                if text_len >= q:
                    raw_count = text_len - q + 1
                    for i in range(raw_count):
                        sorter.add((text[i:i + q].encode("utf-8"), doc_idx))
                meta_f.write(struct.pack("<I", len(id_blob)))
                meta_f.write(id_blob)
                meta_f.write(_META_FRAME.pack(text_len, raw_count))
                self.stats.documents += 1
        if source_hash.hexdigest() != self.source.manifest["content_hash"]:
            raise fmt.IndexCorruptError(
                "streamed documents do not match the source index content "
                "hash; source index is corrupt or was modified")
        self._document_digest = doc_digest.hexdigest()
        self.stats.source_content_bytes_read = \
            self.source.stats()["content_bytes_read"]

    # ------------------------------------------------------------------ #
    # Phase 2: dedupe -> dictionary + postings + forward relations
    # ------------------------------------------------------------------ #
    def _emit_dictionary_and_postings(self, rel_result,
                                      fwd_sorter: VarExternalSorter) -> None:
        qgrams_path = os.path.join(self.staging, "qgrams.bin")
        qgramdir_path = os.path.join(self.staging, "qgramdir.bin")
        postings_path = os.path.join(self.staging, "postings.bin")
        deduped = 0
        qgram_id = 0
        posting_offset = 0  # entries already written to postings.bin
        with open(qgrams_path, "wb") as qgrams_f, \
                open(qgramdir_path, "wb") as qgramdir_f, \
                open(postings_path, "wb") as postings_f:
            sfmt.write_header(qgrams_f, sfmt.KIND_QGRAMS)
            sfmt.write_header(qgramdir_f, sfmt.KIND_QGRAMDIR)
            sfmt.write_header(postings_f, sfmt.KIND_POSTINGS)
            posting_buf = bytearray()
            blob_offset = 0
            prev_key: Optional[bytes] = None
            group_count = 0        # posting entries in the current group
            cur_doc: Optional[int] = None
            cur_tf = 0

            def write_posting(doc: int, tf: int) -> None:
                nonlocal posting_offset
                posting_buf.extend(sfmt.POSTING_RECORD.pack(doc, tf))
                posting_offset += 1
                if len(posting_buf) >= (_POSTING_BUF_RECORDS
                                        * sfmt.POSTING_RECORD_SIZE):
                    postings_f.write(posting_buf)
                    posting_buf.clear()

            def close_group() -> None:
                nonlocal group_count, cur_doc, cur_tf, deduped, qgram_id
                # Flush the last (doc, tf) pair of the open group; only now
                # does group_count include every entry of this posting list,
                # so the directory slice (start, count) is exact.
                write_posting(cur_doc, cur_tf)
                fwd_sorter.add(pack_parts([32, 32], [cur_doc, qgram_id]))
                deduped += 1
                group_count += 1
                qgramdir_f.write(sfmt.QGRAMDIR_RECORD.pack(
                    blob_offset, len(prev_key), 0,
                    posting_offset - group_count, group_count))
                group_count = 0
                cur_doc = None
                cur_tf = 0

            for key, doc_idx in rel_result:
                if prev_key is None:
                    prev_key = key
                    qgrams_f.write(key)
                elif key != prev_key:
                    close_group()
                    blob_offset += len(prev_key)
                    qgram_id += 1
                    prev_key = key
                    qgrams_f.write(key)
                if cur_doc is None:
                    cur_doc, cur_tf = doc_idx, 1
                elif doc_idx == cur_doc:
                    cur_tf += 1  # repeated shingle inside one document
                else:
                    write_posting(cur_doc, cur_tf)
                    fwd_sorter.add(pack_parts([32, 32], [cur_doc, qgram_id]))
                    deduped += 1
                    group_count += 1
                    cur_doc, cur_tf = doc_idx, 1
            if prev_key is not None:
                close_group()
            if posting_buf:
                postings_f.write(posting_buf)
        self.stats.deduped_relations = deduped
        self.stats.dictionary_qgrams = qgram_id + (1 if prev_key is not None
                                                   else 0)
        self.stats.posting_entries = posting_offset
        if deduped != posting_offset:
            raise fmt.IndexCorruptError(
                "posting entry count does not match deduped relations")

    # ------------------------------------------------------------------ #
    # Phase 3: forward relations -> docqgrams.bin + docs.bin
    # ------------------------------------------------------------------ #
    def _emit_docqgrams_and_docs(self, fwd_result, meta_path: str) -> None:
        docqgrams_path = os.path.join(self.staging, "docqgrams.bin")
        docs_path = os.path.join(self.staging, "docs.bin")
        n_docs = self.stats.documents
        dq_written = 0
        with_shingles = 0
        with open(docqgrams_path, "wb") as dq_f, \
                open(docs_path, "wb") as docs_f, \
                open(meta_path, "rb") as meta_f:
            sfmt.write_header(dq_f, sfmt.KIND_DOCQGRAMS)
            sfmt.write_header(docs_f, sfmt.KIND_DOCS)
            dq_buf = bytearray()
            id_blob_len = 0
            rel_iter = iter(fwd_result)
            current = next(rel_iter, None)
            for doc_idx in range(n_docs):
                (id_len,) = struct.unpack("<I", meta_f.read(4))
                meta_f.seek(id_len, os.SEEK_CUR)  # id bytes re-read below
                text_len, _raw = _META_FRAME.unpack(
                    meta_f.read(_META_FRAME.size))
                offset = dq_written
                count = 0
                while current is not None:
                    word = current
                    d, qid = unpack_parts([32, 32], word)
                    if d != doc_idx:
                        break
                    dq_buf.extend(sfmt.DOCQGRAM_RECORD.pack(qid))
                    dq_written += 1
                    count += 1
                    if len(dq_buf) >= (_DOCQGRAM_BUF_RECORDS
                                       * sfmt.DOCQGRAM_RECORD_SIZE):
                        dq_f.write(dq_buf)
                        dq_buf.clear()
                    current = next(rel_iter, None)
                flags = sfmt.FLAG_NO_SHINGLES if text_len < self.q else 0
                if (count == 0) != (text_len < self.q):
                    raise fmt.IndexCorruptError(
                        "shingle count inconsistent with document length")
                if count:
                    with_shingles += 1
                docs_f.write(sfmt.DOCS_RECORD.pack(
                    count, offset, text_len, id_blob_len, id_len, flags))
                id_blob_len += id_len
            if current is not None:
                raise fmt.IndexCorruptError(
                    "forward relations outlive the document table")
            if dq_buf:
                dq_f.write(dq_buf)
            # Second streaming pass over the meta file appends the id blob
            # without ever holding all doc ids in memory.
            meta_f.seek(0)
            for _ in range(n_docs):
                (id_len,) = struct.unpack("<I", meta_f.read(4))
                id_bytes = meta_f.read(id_len)
                docs_f.write(id_bytes)
                meta_f.seek(_META_FRAME.size, os.SEEK_CUR)
        if dq_written != self.stats.deduped_relations:
            raise fmt.IndexCorruptError(
                "forward relation count does not match deduped relations")
        self.stats.documents_with_shingles = with_shingles
        self.stats.documents_without_shingles = n_docs - with_shingles

    # ------------------------------------------------------------------ #
    # Build driver + atomic publish
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
        meta_path = os.path.join(self.staging, "docs_meta.tmp")
        rel_sorter = VarExternalSorter(
            self.staging, QGramRelCodec(),
            chunk_items=self.chunk_items, chunk_bytes=self.chunk_bytes)
        self._ingest(rel_sorter, meta_path)
        self.stats.raw_relations = rel_sorter.items_seen
        rel_result = rel_sorter.finalize()
        self.stats.relation_sort_chunks = rel_sorter.chunks_spilled
        self.stats.relation_sort_bytes = rel_sorter.bytes_spilled
        self.stats.relation_merge_passes = rel_sorter.merge_passes

        fwd_sorter = VarExternalSorter(
            self.staging, U64Codec(),
            chunk_items=self.chunk_items, chunk_bytes=self.chunk_bytes)
        with rel_result:
            self._emit_dictionary_and_postings(rel_result, fwd_sorter)
        rel_result.cleanup()
        fwd_result = fwd_sorter.finalize()
        self.stats.forward_sort_chunks = fwd_sorter.chunks_spilled
        self.stats.forward_sort_bytes = fwd_sorter.bytes_spilled
        self.stats.forward_merge_passes = fwd_sorter.merge_passes
        with fwd_result:
            self._emit_docqgrams_and_docs(fwd_result, meta_path)
        fwd_result.cleanup()
        os.unlink(meta_path)
        return self._finalize()

    def _finalize(self) -> Dict[str, object]:
        self.stats.peak_rss_kb = _peak_rss_kb()
        files = {}
        for name in sorted(os.listdir(self.staging)):
            p = os.path.join(self.staging, name)
            if os.path.isfile(p):
                files[name] = sfmt.hash_file(p)
        src_manifest = self.source.manifest
        manifest = {
            "format_version": sfmt.SIM_FORMAT_VERSION,
            "catalog_identity": uuid.uuid4().hex,
            "source": {
                "index_identity": src_manifest["identity"],
                "content_hash": src_manifest["content_hash"],
                "format_version": src_manifest["format_version"],
                "n_documents": src_manifest["geometry"]["n_documents"],
                "n_symbols": src_manifest["geometry"]["n_symbols"],
            },
            "parameters": {
                "q": self.q,
                "tokenization": sfmt.TOKENIZATION_RULE,
                "shingle_unit": "codepoint",
                "normalization": "none",
                "case_folding": "none",
                "whitespace_folding": "none",
                "cross_document_shingles": False,
            },
            "documents": {
                "count": self.stats.documents,
                "with_shingles": self.stats.documents_with_shingles,
                "without_shingles": self.stats.documents_without_shingles,
                "digest": self._document_digest,
            },
            "relations": {
                "raw": self.stats.raw_relations,
                "deduped": self.stats.deduped_relations,
            },
            "dictionary": {"qgram_count": self.stats.dictionary_qgrams},
            "postings": {"entry_count": self.stats.posting_entries},
            "build_stats": self.stats.as_dict(),
            "files": files,
        }
        sfmt.write_manifest(self.staging, manifest)
        # Completion marker last, fsynced: absence means "interrupted".
        marker = os.path.join(self.staging, sfmt.OPEN_MARKER)
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.staging, self.catalog_dir)
        return manifest

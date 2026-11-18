"""Full differential test suite for the disk FM-index.

Every property below is checked against the independent str.find oracle in
``oracle.py``: overlapping counts, both coordinate kinds, paging order,
cursors, integrity failures and external-memory behaviour (tiny chunk
buffers force real spill/merge paths, tiny sample intervals force long LF
walks and sentinel shortcuts).
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm import (  # noqa: E402
    CursorError,
    DiskFMIndex,
    IndexBuilder,
    IndexCorruptError,
)
from diskfm import format as fmt  # noqa: E402
from diskfm.extsort import (  # noqa: E402
    ExternalSorter,
    Packed64Codec,
    Triple32Codec,
    pack_parts,
    unpack_parts,
)
from diskfm.pager import FilePager  # noqa: E402

from oracle import brute_count, brute_locate, brute_multiset  # noqa: E402


# --------------------------------------------------------------------------- #
# External sort unit tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("codec_cls", [Packed64Codec, Triple32Codec])
def test_external_sort_matches_in_memory(codec_cls):
    with tempfile.TemporaryDirectory() as tmp:
        codec = codec_cls()
        rng = random.Random(1234)
        if codec_cls is Packed64Codec:
            data = [rng.randrange(1 << 64) for _ in range(5000)]
        else:
            data = [tuple(rng.randrange(1 << 32) for _ in range(3))
                    for _ in range(5000)]
        sorter = ExternalSorter(tmp, codec, chunk_items=317)
        sorter.add_many(data)
        assert sorter.chunks_spilled > 1  # really external
        result = sorter.finalize()
        with result:
            got = list(result)
        assert got == sorted(data)
        result.cleanup()


def test_invalid_construction_arguments_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        for kw in ({"sample_interval": 0},
                   {"byte_sample_interval": 0},
                   {"chunk_items": 0}):
            with pytest.raises(ValueError):
                IndexBuilder(os.path.join(tmp, repr(kw)), **kw)


def test_filepager_lru_is_bounded():
    import array as _array
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "paged.bin")
        n = 10000
        with open(path, "wb") as f:
            f.write(b"FM\x01\x00\x01\x00\x00\x00")
            _array.array("I", range(n)).tofile(f)
        pg = FilePager(path, record_size=4, header_size=8,
                       page_records=64, max_pages=3)
        for i in range(10):
            assert pg.get_u32(i * 64) == i * 64
        assert pg.stats()["resident_pages"] == 3
        for _ in range(100):
            pg.get_u32(9 * 64)
        assert pg.stats()["resident_pages"] == 3
        with pytest.raises(IndexError):
            pg.get_u32(n)
        pg.close()


def test_pack_unpack_roundtrip():
    bits = [11, 21, 32]
    word = pack_parts(bits, [123, 4567, 999999])
    assert unpack_parts(bits, word) == (123, 4567, 999999)
    with pytest.raises(ValueError):
        pack_parts(bits, [1 << 11, 0, 0])


def test_external_sort_many_way_merge():
    # Force far more chunks than the merge heap's usual width so the k-way
    # heap merge path is genuinely exercised on disk.
    with tempfile.TemporaryDirectory() as tmp:
        rng = random.Random(55)
        data = [rng.randrange(1 << 64) for _ in range(20000)]
        sorter = ExternalSorter(tmp, Packed64Codec(), chunk_items=37)
        sorter.add_many(data)
        assert sorter.chunks_spilled > 100
        result = sorter.finalize()
        with result:
            got = list(result)
        assert got == sorted(data)
        result.cleanup()


def test_jsonl_blank_lines_and_unordered_input():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "in.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(json.dumps({"doc_id": "z", "text": "zebra"}) + "\n")
            f.write("   \n")
            f.write(json.dumps({"doc_id": "a", "text": "ababa"}) + "\n")
            f.write("\n")
        idx_dir = os.path.join(tmp, "idx")
        b = IndexBuilder(idx_dir, chunk_items=4, sample_interval=2)
        b.ingest_jsonl(jp)
        b.build()
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("aba") == 2
            assert idx.count("eb") == 1
            assert idx.document_text("z") == "zebra"
            assert idx.document_text("a") == "ababa"


def test_malformed_jsonl_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "bad.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": "ok", "text": "x"}) + "\n")
            f.write("{not json\n")
        b = IndexBuilder(os.path.join(tmp, "idx"))
        with pytest.raises(ValueError):
            b.ingest_jsonl(jp)


def test_jsonl_missing_fields_and_wrong_types_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        for line in ('{"doc_id":"x"}\n',
                     '{"text":"y"}\n',
                     '{"doc_id":7,"text":"y"}\n',
                     '{"doc_id":"z","text":null}\n',
                     '[1,2,3]\n'):
            jp = os.path.join(tmp, "in.jsonl")
            with open(jp, "w", encoding="utf-8") as f:
                f.write(line)
            b = IndexBuilder(os.path.join(tmp, f"idx{line[:4]}"))
            with pytest.raises(ValueError):
                b.ingest_jsonl(jp)


def test_rankdir_cumulative_counts_match_symbol_histogram():
    docs = [("a", "banana"), ("b", "band"), ("c", "")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=4, sample_interval=2)
        with DiskFMIndex(idx_dir) as idx:
            # Independent histogram over the global symbol stream.
            hist = {}
            for _id, text in docs:
                for ch in text:
                    hist[ord(ch)] = hist.get(ord(ch), 0) + 1
            for d in range(len(docs)):
                hist[fmt.MIN_SENTINEL + d] = 1
            running = 0
            for i in range(idx._rankdir_count):
                symbol, _off, count, cumulative = idx._rankdir_record(i)
                assert count == hist[symbol]
                assert cumulative == running
                running += count
            assert running == idx.n
            # C(c) used by backward search equals the cumulative field.
            for symbol in hist:
                assert idx._c(symbol) == sum(
                    v for s, v in hist.items() if s < symbol)
            # Unknown symbols have neither a directory entry nor C value.
            assert idx._c(ord("z")) is None
            assert idx.rank(ord("z"), idx.n) == 0


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _build(tmp, docs, **kw):
    idx_dir = os.path.join(tmp, "idx")
    builder = IndexBuilder(idx_dir, **kw)
    for doc_id, text in docs:
        builder.add_document(doc_id, text)
    manifest = builder.build()
    return idx_dir, manifest


def _queries_for(docs, rng, extra=()):
    alphabet = sorted({c for _, t in docs for c in t} | set(extra))
    queries = list(extra)
    for _ in range(60):
        L = rng.randint(1, 5)
        if alphabet:
            queries.append("".join(rng.choice(alphabet) for _ in range(L)))
    # Real substrings sampled from documents too.
    for _id, text in docs:
        for _ in range(8):
            if len(text) >= 2:
                i = rng.randrange(len(text) - 1)
                j = rng.randint(i + 1, min(len(text), i + 6))
                queries.append(text[i:j])
    return [q for q in dict.fromkeys(queries) if q]


def test_many_small_documents_paged_directory_and_doctable():
    # Hundreds of documents force many rankdir symbols and doctable rows;
    # both structures must stay behind their bounded LRU pagers.
    rng = random.Random(31337)
    docs = [(f"doc{i:04d}",
             "".join(rng.choice("abcdefgh😀\n\x00")
                     for _ in range(rng.randint(0, 12))))
            for i in range(300)]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=64, sample_interval=3)
        with DiskFMIndex(idx_dir, max_pages=4, page_records=32) as idx:
            assert idx.n_docs == 300
            for q in ("a", "😀", "\x00", "\n", "abc", "fg", "zz"):
                assert idx.count(q) == brute_count(docs, q), q
                got = {}
                for h in idx.locate_all(q, page_size=7):
                    key = (h.doc_id, h.codepoint_offset,
                           h.utf8_byte_offset)
                    got[key] = got.get(key, 0) + 1
                assert got == brute_multiset(docs, q), q
            s = idx.stats()
            # Directory/doctable resident pages never exceed the LRU cap no
            # matter how many documents exist.
            assert s["rankdir_pager"]["resident_pages"] <= 4
            assert s["doctable_pager"]["resident_pages"] <= 4
            # Round-trip a late document id through the paged id blob.
            assert idx.document_text("doc0299") == dict(docs)["doc0299"]
            with pytest.raises(KeyError):
                idx.document_text("doc9999")


def test_built_suffix_order_equals_independent_suffix_array():
    docs = [("a", "mississippi"), ("b", "abracadabra"), ("c", ""),
            ("d", "aaaa"), ("e", "😀a😀a\x00")]
    for force_wide in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            idx_dir, _ = _build(tmp, docs, chunk_items=7,
                                sample_interval=3, force_wide=force_wide)
            symbols = []
            for d, text in docs:
                symbols.extend(ord(ch) for ch in text)
                symbols.append(fmt.MIN_SENTINEL + docs.index((d, text)))
            true_sa = sorted(range(len(symbols)),
                             key=lambda p: symbols[p:])
            with DiskFMIndex(idx_dir) as idx:
                # Every BWT row must recover exactly the true SA position;
                # this validates doubling, BWT, LF and samples end to end.
                bad = [r for r in range(idx.n)
                       if idx._row_to_position(r) != true_sa[r]]
                assert bad == []


# --------------------------------------------------------------------------- #
# The headline cases from the specification
# --------------------------------------------------------------------------- #
def test_ababa_overlapping_offsets():
    docs = [("d", "ababa")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=4, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("aba") == 2
            hits = idx.locate_all("aba", page_size=1)
            assert [(h.doc_id, h.codepoint_offset) for h in hits] == [
                ("d", 0), ("d", 2)]
            assert brute_count(docs, "aba") == 2


def test_no_cross_document_match():
    # "ab" followed by a document "a" must never form "aba".
    docs = [("p", "ab"), ("q", "a")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=3, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("aba") == 0
            assert idx.locate_all("aba") == []
            assert idx.count("ab") == 1
            assert idx.count("a") == 2


def test_emoji_nul_and_byte_offsets():
    text = "a😀b\x00c🙂"
    docs = [("e", text)]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=2, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            expect = brute_locate(docs, "😀")
            got = idx.locate_all("😀")
            assert [(h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
                    for h in got] == [
                        (h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
                        for h in expect]
            # Explicit UTF-8 prefix length check.
            for cp in range(len(text)):
                q = text[cp]
                if q in ("\x00",):
                    h = idx.locate_all(q)[0]
                    assert h.utf8_byte_offset == len(text[:cp].encode())
            assert idx.count("\x00") == 1
            # Every recovered byte offset equals the UTF-8 prefix length.
            for cp in range(len(text)):
                hits = idx.locate_all(text[cp])
                target = [h for h in hits if h.doc_id == "e"
                          and h.codepoint_offset == cp]
                assert target and target[0].utf8_byte_offset == \
                    len(text[:cp].encode("utf-8"))


def test_internal_sentinels_are_unreachable_in_queries():
    # Language-level guarantee: no Python str (and hence no JSONL text or
    # query) can hold a code point above U+10FFFF, while every internal
    # sentinel is >= 0x110000.  Backward search therefore can never match
    # across a document-end sentinel.
    with pytest.raises(ValueError):
        chr(fmt.MIN_SENTINEL)
    # Sentinels start immediately above the largest legal user codepoint.
    assert fmt.MIN_SENTINEL == fmt.MAX_USER_CODEPOINT + 1
    assert ord(chr(fmt.MAX_USER_CODEPOINT)) == fmt.MAX_USER_CODEPOINT


def test_max_codepoint_does_not_clash_with_sentinels():
    docs = [("u", chr(0x10FFFF)), ("v", chr(0x10FFFF) + "x"),
            ("w", "x" + chr(0x10FFFF))]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, manifest = _build(tmp, docs, chunk_items=2,
                                   sample_interval=1)
        assert manifest["geometry"]["min_sentinel"] == 0x110000
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count(chr(0x10FFFF)) == 3
            # The sentinel itself is not a legal user query symbol, and a
            # query built from the max codepoint still matches exactly.
            assert idx.count(chr(0x10FFFF) + "x") == 1
            assert idx.count("x" + chr(0x10FFFF)) == 1


def test_lone_surrogate_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        builder = IndexBuilder(os.path.join(tmp, "idx"))
        with pytest.raises(ValueError):
            builder.add_document("bad", "x\ud800y")
        with pytest.raises(ValueError):
            IndexBuilder(os.path.join(tmp, "idx2")).add_document(
                "bad2", "a\udfff")


def test_empty_documents_allowed():
    docs = [("a", ""), ("b", "x"), ("c", "")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=2, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("x") == 1
            assert idx.locate_all("x")[0].doc_id == "b"
            with pytest.raises(ValueError):
                idx.count("")


def test_newlines_and_nul_preserved():
    docs = [("n", "line1\nline2\r\n\tend"),
            ("z", "\x00\x00\x00")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=3, sample_interval=2)
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("\n") == 2
            assert idx.count("\x00") == 3
            assert idx.document_text("n") == docs[0][1]
            assert idx.document_text("z") == docs[1][1]


# --------------------------------------------------------------------------- #
# Broad differential fuzzing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample_interval,chunk_items,force_wide", [
    (1, 7, False), (2, 13, False), (5, 29, True), (32, 64, False),
])
def test_differential_random_corpus(sample_interval, chunk_items, force_wide):
    rng = random.Random(99 + sample_interval * 7 + chunk_items)
    docs = []
    alphabet = "ab"  # highly repetitive
    for d in range(12):
        L = rng.randint(0, 40)
        docs.append((f"doc{d}", "".join(rng.choice(alphabet)
                                        for _ in range(L))))
    # Duplicate text under distinct doc_ids must count separately.
    docs.append(("dup1", "ababab"))
    docs.append(("dup2", "ababab"))
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=chunk_items,
                            sample_interval=sample_interval,
                            force_wide=force_wide)
        with DiskFMIndex(idx_dir) as idx:
            for q in _queries_for(docs, rng, extra=("aba", "abab", "b")):
                assert idx.count(q) == brute_count(docs, q), q
                got_ms = {}
                for h in idx.locate_all(q, page_size=1):
                    key = (h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
                    got_ms[key] = got_ms.get(key, 0) + 1
                assert got_ms == brute_multiset(docs, q), q


def test_long_repetitive_string():
    docs = [("r", "ab" * 500), ("s", "a" * 300)]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=50, sample_interval=7)
        with DiskFMIndex(idx_dir) as idx:
            for q in ("ab", "aba", "bab", "a" * 10, "ab" * 20, "ab" * 500):
                assert idx.count(q) == brute_count(docs, q), q
                got_ms = {}
                for h in idx.locate_all(q, page_size=3):
                    key = (h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
                    got_ms[key] = got_ms.get(key, 0) + 1
                assert got_ms == brute_multiset(docs, q), q


def test_identical_documents_count_separately():
    docs = [("k1", "same"), ("k2", "same"), ("k3", "same")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=3, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("same") == 3
            ids = sorted(h.doc_id for h in idx.locate_all("same"))
            assert ids == ["k1", "k2", "k3"]


# --------------------------------------------------------------------------- #
# Paging and cursors
# --------------------------------------------------------------------------- #
def test_paging_page_size_one_exhausts():
    docs = [("p", "abababab")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=4, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            seen = []
            cursor = None
            pages = 0
            while True:
                page, cursor = idx.locate_page("aba", 1, cursor)
                assert len(page) <= 1
                seen.extend(page)
                pages += 1
                if cursor is None:
                    break
            assert pages == idx.count("aba")  # 3 overlapping hits
            assert [h.codepoint_offset for h in seen] == [0, 2, 4]
            # A None cursor means "no further page"; restarting without a
            # cursor deterministically replays the first page.
            restart, _ = idx.locate_page("aba", 1)
            assert [h.codepoint_offset for h in restart] == [0]


def test_cursor_rejects_different_query():
    docs = [("p", "abababa")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=4, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            _page, cursor = idx.locate_page("aba", 1)
            assert cursor is not None
            with pytest.raises(CursorError):
                idx.locate_page("bab", 1, cursor)
            with pytest.raises(CursorError):
                idx.locate_page("ab", 1, cursor)


def test_cursor_rejects_other_index():
    docs = [("p", "abababa")]
    with tempfile.TemporaryDirectory() as tmp:
        idx1 = os.path.join(tmp, "i1")
        idx2 = os.path.join(tmp, "i2")
        for target in (idx1, idx2):
            b = IndexBuilder(target, chunk_items=4, sample_interval=1)
            b.add_document(*docs[0])
            b.build()
        with DiskFMIndex(idx1) as a, DiskFMIndex(idx2) as other:
            _page, cursor = a.locate_page("aba", 1)
            with pytest.raises(CursorError):
                other.locate_page("aba", 1, cursor)


def test_tampered_cursor_rejected():
    docs = [("p", "abababa")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=4, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            _page, cursor = idx.locate_page("aba", 1)
            bad = cursor[:-1] + ("0" if cursor[-1] != "0" else "1")
            with pytest.raises(CursorError):
                idx.locate_page("aba", 1, bad)
            with pytest.raises(CursorError):
                idx.locate_page("aba", 1, "not-a-cursor")


def test_sampling_boundary_intervals():
    docs = [("b", "abcabc" * 30)]
    # sample_interval = 1 means every row is sampled (zero LF steps);
    # large intervals force the maximum LF walk and sentinel shortcut.
    for si in (1, 2, 3, 100):
        with tempfile.TemporaryDirectory() as tmp:
            idx_dir, _ = _build(tmp, docs, chunk_items=25,
                                sample_interval=si)
            with DiskFMIndex(idx_dir) as idx:
                q = "abc"
                hits = idx.locate_all(q, page_size=7)
                assert brute_multiset(docs, q) == {
                    (h.doc_id, h.codepoint_offset, h.utf8_byte_offset):
                    1 for h in hits}


# --------------------------------------------------------------------------- #
# Build ordering invariance and JSONL input
# --------------------------------------------------------------------------- #
def _write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(json.dumps({"doc_id": doc_id, "text": text},
                               ensure_ascii=False) + "\n")


def test_input_order_does_not_change_result_multiset():
    rng = random.Random(7)
    docs = [(f"d{i}", "".join(rng.choice("abc")
                               for _ in range(rng.randint(0, 25))))
            for i in range(15)]
    queries = _queries_for(docs, rng, extra=("abc", "ca", "b"))
    with tempfile.TemporaryDirectory() as tmp:
        orders = [docs, list(reversed(docs))]
        shuffled = docs[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
        results = []
        for k, order in enumerate(orders):
            jp = os.path.join(tmp, f"in{k}.jsonl")
            _write_jsonl(jp, order)
            idx_dir = os.path.join(tmp, f"idx{k}")
            b = IndexBuilder(idx_dir, chunk_items=17, sample_interval=3)
            b.ingest_jsonl(jp)
            b.build()
            with DiskFMIndex(idx_dir) as idx:
                results.append({q: idx.locate_all(q) for q in queries})
        for q in queries:
            multisets = []
            for res in results:
                ms = {}
                for h in res[q]:
                    key = (h.doc_id, h.codepoint_offset,
                           h.utf8_byte_offset)
                    ms[key] = ms.get(key, 0) + 1
                multisets.append(ms)
            assert all(m == multisets[0] for m in multisets), q


def test_duplicate_doc_id_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        b = IndexBuilder(os.path.join(tmp, "idx"))
        b.add_document("x", "a")
        with pytest.raises(ValueError):
            b.add_document("x", "b")


# --------------------------------------------------------------------------- #
# Integrity / truncation / interrupted build
# --------------------------------------------------------------------------- #
def _simple_index(tmp):
    docs = [("a", "mississippi"), ("b", "ssi"), ("c", "")]
    return _build(tmp, docs, chunk_items=5, sample_interval=2)


def test_interrupted_build_has_no_open_marker():
    with tempfile.TemporaryDirectory() as tmp:
        staging = os.path.join(tmp, "idx.staging")
        b = IndexBuilder(os.path.join(tmp, "idx"), chunk_items=4,
                         sample_interval=2)
        b.add_document("a", "mississippi")
        b.text_f.flush()
        # Simulate a crash before build(): staging exists with no marker.
        assert os.path.isdir(staging)
        assert not os.path.exists(os.path.join(staging, fmt.OPEN_MARKER))
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(staging)
        assert not os.path.exists(os.path.join(tmp, "idx"))


def test_truncated_file_fails_verification():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _simple_index(tmp)
        target = os.path.join(idx_dir, "rank.bin")
        with open(target, "r+b") as f:
            f.truncate(os.path.getsize(target) - 3)
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)


def test_corrupted_byte_fails_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _simple_index(tmp)
        target = os.path.join(idx_dir, "bwt.bin")
        with open(target, "r+b") as f:
            f.seek(fmt.HEADER_SIZE + 1)
            f.write(b"\xff")
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)


def test_missing_file_fails():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _simple_index(tmp)
        os.unlink(os.path.join(idx_dir, "rankdir.bin"))
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)


def test_manifest_tampering_fails():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _simple_index(tmp)
        mp = os.path.join(idx_dir, "manifest.json")
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        m["geometry"]["n_symbols"] += 1
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(m, f)
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)


def test_reopen_is_self_contained():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, manifest = _simple_index(tmp)
        # Reopening from a fresh object must give identical answers.
        with DiskFMIndex(idx_dir) as idx:
            before = {q: (idx.count(q),
                          [(h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
                           for h in idx.locate_all(q)])
                      for q in ("ssi", "i", "miss", "x")}
        with DiskFMIndex(idx_dir) as idx2:
            for q, (cnt, hits) in before.items():
                assert idx2.count(q) == cnt
                assert [(h.doc_id, h.codepoint_offset, h.utf8_byte_offset)
                        for h in idx2.locate_all(q)] == hits


def test_removing_open_marker_blocks_otherwise_complete_index():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _simple_index(tmp)
        os.unlink(os.path.join(idx_dir, fmt.OPEN_MARKER))
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)


def test_bad_header_magic_and_kind_fail():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _simple_index(tmp)
        bwt = os.path.join(idx_dir, "bwt.bin")
        with open(bwt, "r+b") as f:
            f.seek(0)
            f.write(b"\x00\x00")  # destroy magic
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)
        # A header whose kind tag lies about the file must also fail.
        idx_dir2, _ = _simple_index(os.path.join(tmp, "second"))
        rank = os.path.join(idx_dir2, "rank.bin")
        with open(rank, "r+b") as f:
            f.seek(2)
            f.write(struct.pack("<H", 999))
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir2)


def test_failed_build_leaves_no_readable_index():
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "idx")
        b = IndexBuilder(target, chunk_items=4, sample_interval=2)
        b.add_document("ok", "ababa")
        with pytest.raises(ValueError):
            b.add_document("ok", "duplicate-id-rejected")
        # No finalized directory may exist; staging alone is not openable.
        assert not os.path.exists(target)
        staging = target + ".staging"
        if os.path.exists(staging):
            with pytest.raises(IndexCorruptError):
                DiskFMIndex(staging)


# --------------------------------------------------------------------------- #
# Runtime accounting: bounded reads, no full-SA / full-corpus access
# --------------------------------------------------------------------------- #
def test_count_does_not_enumerate_or_decode_corpus():
    docs = [("big", "ab" * 400)]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=31, sample_interval=5)
        with DiskFMIndex(idx_dir) as idx:
            n = idx.n
            idx.count("ab")
            s = idx.stats()
            # count uses only rank calls over occurrence lists: no SA
            # samples, no document content, no LF walks.
            assert s["rank_calls"] > 0
            assert s["lf_steps"] == 0
            assert s["sample_rows_read"] == 0
            assert s["content_bytes_read"] == 0
            # Rank paging touches O(log count) pages per binary search.
            # Resident memory is fixed by the LRU geometry (max_pages),
            # never by corpus size; the large-corpus bounded-RSS proof lives
            # in test_large_corpus_bounded_memory.
            assert s["rank_pager"]["page_misses"] <= \
                len("ab") * (idx.n.bit_length() + 1)
            assert s["rank_pager"]["resident_pages"] <= \
                idx._rank_pager.max_pages
            assert n > 0


def test_locate_reads_only_pages_for_requested_page():
    docs = [("big", "abc" * 300)]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=37, sample_interval=4)
        with DiskFMIndex(idx_dir) as idx:
            page1, cursor = idx.locate_page("abc", page_size=5)
            assert len(page1) == 5
            s1 = idx.stats()
            page2, _ = idx.locate_page("abc", 5, cursor)
            assert len(page2) == 5
            # The two pages must be disjoint even though hits overlap.
            keys1 = {(h.doc_id, h.codepoint_offset) for h in page1}
            keys2 = {(h.doc_id, h.codepoint_offset) for h in page2}
            assert keys1.isdisjoint(keys2)
            # LF steps are bounded by the sample interval, not corpus size.
            assert s1["lf_steps"] <= 5 * 4
            # Page 1 never decoded whole documents.
            assert s1["content_bytes_read"] < len(docs[0][1].encode())


def test_lf_step_counter_is_real():
    docs = [("w", "abcdefgh" * 20)]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=11, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            idx.locate_all("abc", page_size=4)
            # Every row sampled -> coordinate recovery needs zero LF steps.
            assert idx.stats()["lf_steps"] == 0
        idx_dir2 = os.path.join(tmp, "idx2")
        b2 = IndexBuilder(idx_dir2, chunk_items=11, sample_interval=16)
        b2.add_document(*docs[0])
        b2.build()
        with DiskFMIndex(idx_dir2) as idx:
            idx.locate_all("abc", page_size=4)
            assert idx.stats()["lf_steps"] > 0


def test_rank_counter_increments_per_call():
    docs = [("c", "banana")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=3, sample_interval=1)
        with DiskFMIndex(idx_dir) as idx:
            before = idx.stats()["rank_calls"]
            idx.count("ana")  # 3 backward-search characters
            after = idx.stats()["rank_calls"]
            assert after - before >= 3


# --------------------------------------------------------------------------- #
# Unicode breadth
# --------------------------------------------------------------------------- #
def test_zwj_sequences_and_combining_marks():
    docs = [("e1", "a👨‍👩‍👧b"),
            ("e2", "x👨‍👩‍👧y👨‍👩‍👧"),
            ("e3", "é̂combining")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=3, sample_interval=2)
        with DiskFMIndex(idx_dir) as idx:
            for q in ("👨‍👩‍👧", "👨", "‍", "👧", "é̂",
                      "combining", "a👨", "👧b"):
                assert idx.count(q) == brute_count(docs, q), q
                got = sorted((h.doc_id, h.codepoint_offset,
                              h.utf8_byte_offset)
                             for h in idx.locate_all(q))
                exp = sorted((h.doc_id, h.codepoint_offset,
                              h.utf8_byte_offset)
                             for h in brute_locate(docs, q))
                assert got == exp, q
            for _id, text in docs:
                for cp in range(len(text)):
                    hits = [h for h in idx.locate_all(text[cp])
                            if h.doc_id == _id
                            and h.codepoint_offset == cp]
                    assert hits and hits[0].utf8_byte_offset == \
                        len(text[:cp].encode("utf-8"))


def test_mixed_multilingual_text():
    docs = [
        ("u1", "日本語テキストとsubstring"),
        ("u2", "éñÜÅøÞ  mixed 日本語"),
        ("u3", "𐍈𐌗𐍈 rune 𐍈 test"),  # 4-byte Gothic
        ("u4", "tab\tnewline\nnul\x00end"),
    ]
    rng = random.Random(2024)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir, _ = _build(tmp, docs, chunk_items=9, sample_interval=3)
        with DiskFMIndex(idx_dir) as idx:
            for q in _queries_for(docs, rng, extra=(
                    "日本語", "substring", "𐍈", "\n", "\x00", "語")):
                assert idx.count(q) == brute_count(docs, q), q
                got = {(h.doc_id, h.codepoint_offset, h.utf8_byte_offset):
                       1 for h in idx.locate_all(q)}
                assert got == brute_multiset(docs, q), q
            # Round-trip every original string byte-for-byte.
            for doc_id, text in docs:
                assert idx.document_text(doc_id) == text


def test_empty_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = os.path.join(tmp, "empty")
        b = IndexBuilder(idx_dir, chunk_items=4)
        manifest = b.build()
        assert manifest["geometry"]["n_symbols"] == 0
        with DiskFMIndex(idx_dir) as idx:
            assert idx.count("anything") == 0
            assert idx.locate_all("anything") == []
            with pytest.raises(ValueError):
                idx.count("")

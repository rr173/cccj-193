"""Differential test suite for the document similarity catalog.

Expected results come from the independent set-algebra oracle in
``sim_oracle.py``, which shares no code with the catalog implementation.
Tiny sort buffers force real external-merge spill paths; corruption,
interruption, cursor and determinism behaviour are covered alongside the
Jaccard semantics.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diskfm import (  # noqa: E402
    CursorError,
    IndexBuilder,
    IndexCorruptError,
    SimilarityBuilder,
    SimilarityCatalog,
)
from diskfm import simformat as sfmt  # noqa: E402
from diskfm.simquery import parse_threshold  # noqa: E402
from diskfm.vsort import QGramRelCodec, ResultCodec, VarExternalSorter  # noqa: E402

from sim_oracle import (  # noqa: E402
    expected_explain,
    expected_posting_records,
    expected_query,
    shingle_set,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_index(tmp, docs, name="idx", **kw):
    idx_dir = os.path.join(tmp, name)
    builder = IndexBuilder(idx_dir, **kw)
    for doc_id, text in docs:
        builder.add_document(doc_id, text)
    builder.build()
    return idx_dir


def _build_catalog(tmp, docs, q, index_kw=None, cat_kw=None, name="cat"):
    idx_dir = _build_index(tmp, docs, name=f"idx_{name}", **(index_kw or {}))
    cat_dir = os.path.join(tmp, name)
    builder = SimilarityBuilder(idx_dir, cat_dir, q=q, **(cat_kw or {}))
    manifest = builder.build()
    return cat_dir, manifest


def _rows(cat, source, threshold):
    return cat.query_all(source, threshold)


# --------------------------------------------------------------------------- #
# vsort unit tests (variable-length external sort)
# --------------------------------------------------------------------------- #
def test_vsort_matches_in_memory_with_prefix_keys():
    # Keys where one is a strict byte-prefix of another stress the framing:
    # raw payload order must never be used for comparisons.
    with tempfile.TemporaryDirectory() as tmp:
        records = [(k, i) for i, k in enumerate(
            [b"ab", b"a", b"abc", b"", b"ab", b"\x00", b"\x00a", b"a\x00",
             b"abc", b"\xff\xfe", b"\xff"])]
        sorter = VarExternalSorter(tmp, QGramRelCodec(), chunk_items=3,
                                   chunk_bytes=64)
        for r in records:
            sorter.add(r)
        assert sorter.chunks_spilled > 1
        result = sorter.finalize()
        with result:
            got = list(result)
        assert got == sorted(records)
        result.cleanup()


def test_vsort_bounded_fan_in_multi_pass():
    # More runs than the fan-in forces intermediate merge rounds; the
    # result is still a fully sorted stream and fd usage stays bounded.
    with tempfile.TemporaryDirectory() as tmp:
        rng = random.Random(4242)
        records = [(rng.randbytes(rng.randint(0, 6)), rng.randrange(100))
                   for _ in range(500)]
        sorter = VarExternalSorter(tmp, QGramRelCodec(), chunk_items=4,
                                   chunk_bytes=128, merge_fan_in=3)
        for r in records:
            sorter.add(r)
        assert sorter.chunks_spilled > 10
        result = sorter.finalize()
        assert sorter.merge_passes > 1  # intermediate rounds happened
        with result:
            got = list(result)
        assert got == sorted(records)
        result.cleanup()
        # No run files left behind.
        assert [f for f in os.listdir(tmp) if f.startswith("vsort")] == []


def test_vsort_key_function_and_chunk_bytes_spill():
    # The key path (used for exact rational result ordering) must spill by
    # byte budget too and merge identically to an in-memory sort.
    from diskfm.simquery import _result_sort_key
    with tempfile.TemporaryDirectory() as tmp:
        rng = random.Random(99)
        records = [(rng.randint(1, 50), rng.randint(1, 50),
                    rng.randint(0, 50), f"doc{rng.randint(0,30)}".encode())
                   for _ in range(400)]
        records = [(i, max(u, i), t, d) for (i, u, t, d) in records]
        sorter = VarExternalSorter(tmp, ResultCodec(), chunk_items=1000,
                                   chunk_bytes=512, key=_result_sort_key)
        for r in records:
            sorter.add(r)
        assert sorter.chunks_spilled > 1  # spilled on bytes, not items
        result = sorter.finalize()
        with result:
            got = list(result)
        assert got == sorted(records, key=_result_sort_key)
        result.cleanup()


def test_vsort_ragged_run_file_is_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        sorter = VarExternalSorter(tmp, QGramRelCodec(), chunk_items=2)
        sorter.add((b"x", 1))
        result = sorter.finalize()
        path = result._chunk_paths[0]
        with open(path, "ab") as f:
            f.write(b"\x05\x00")  # dangling partial frame
        with pytest.raises(IndexCorruptError):
            with result:
                list(result)
        result.cleanup()


# --------------------------------------------------------------------------- #
# Threshold parsing
# --------------------------------------------------------------------------- #
def test_threshold_parsing_exactness():
    assert parse_threshold("0.5") == Fraction(1, 2)
    assert parse_threshold("1/3") == Fraction(1, 3)
    assert parse_threshold(0.1) == Fraction(1, 10)  # decimal, not binary
    assert parse_threshold(1) == Fraction(1)
    assert parse_threshold(Fraction(2, 7)) == Fraction(2, 7)
    assert parse_threshold(" 0.25 ") == Fraction(1, 4)
    for bad in (0, -1, 1.5, "0", "-0.5", "2", "1/0", "abc", "0.1.2", ""):
        with pytest.raises((ValueError, ZeroDivisionError)):
            parse_threshold(bad)
    for bad_type in (None, [], True):
        with pytest.raises(TypeError):
            parse_threshold(bad_type)


# --------------------------------------------------------------------------- #
# Core Jaccard semantics
# --------------------------------------------------------------------------- #
def test_identical_documents_are_separate_results():
    docs = [("a", "the quick brown fox"), ("b", "the quick brown fox"),
            ("c", "the quick brown fox")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=3)
        with SimilarityCatalog(cat_dir) as cat:
            for source in ("a", "b", "c"):
                rows = _rows(cat, source, "0.9")
                assert [r["target_doc_id"] for r in rows] == sorted(
                    d for d in ("a", "b", "c") if d != source)
                for r in rows:
                    assert r["similarity"] == "1/1"
                    assert r["source_doc_id"] == source
                    # A document is never its own result.
                    assert r["target_doc_id"] != source
                assert rows == expected_query(docs, 3, source,
                                              Fraction(9, 10))
                # Threshold 1 keeps only exact duplicates.
                assert _rows(cat, source, "1") == rows


def test_extreme_chunk_geometry_and_wide_source():
    # chunk_items=1 forces every record into its own run (maximal merge
    # fan-in); the source index built with the wide codec is equally valid
    # input.
    docs = _template_docs(15)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(
            tmp, docs, q=4,
            index_kw={"force_wide": True, "chunk_items": 8,
                      "sample_interval": 2},
            cat_kw={"chunk_items": 1, "chunk_bytes": 64})
        with SimilarityCatalog(cat_dir, counter_chunk_items=1,
                               result_chunk_items=1,
                               result_chunk_bytes=64) as cat:
            rows = _rows(cat, "doc003", "0.4")
            assert rows == expected_query(docs, 4, "doc003",
                                          Fraction(2, 5))
            st = cat.stats()["last_query"]
            assert st["counter_chunks_spilled"] > 10
            assert cat.validate()["ok"]


def test_one_codepoint_changed():
    docs = [("x", "abcdefghij"), ("y", "abcdefghik"), ("z", "abcdefghij")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        with SimilarityCatalog(cat_dir) as cat:
            rows = _rows(cat, "x", "0.01")
            assert rows == expected_query(docs, 4, "x", Fraction(1, 100))
            by_id = {r["target_doc_id"]: r for r in rows}
            assert by_id["z"]["similarity"] == "1/1"
            # q=4 over length-10 texts: 7 shingles each; only the last
            # shingle contains the changed final codepoint.
            assert by_id["y"]["intersection_count"] == 6
            assert by_id["y"]["union_count"] == 8
            assert by_id["y"]["similarity"] == "3/4"


def test_repeated_shingles_counted_once():
    # "aaaaaa" with q=2 has one distinct shingle ("aa"), not five.
    docs = [("rep", "aaaaaa"), ("one", "aa"), ("two", "aabb")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, docs, q=2)
        assert m["relations"]["raw"] == 5 + 1 + 3
        assert m["relations"]["deduped"] == 1 + 1 + 3
        with SimilarityCatalog(cat_dir) as cat:
            rows = _rows(cat, "rep", "0.01")
            by_id = {r["target_doc_id"]: r for r in rows}
            assert by_id["one"]["similarity"] == "1/1"
            assert by_id["two"]["intersection_count"] == 1
            assert by_id["two"]["union_count"] == 3
            assert by_id["two"]["similarity"] == "1/3"
            assert rows == expected_query(docs, 2, "rep", Fraction(1, 100))


def test_q1_and_larger_q():
    docs = [("a", "xyx"), ("b", "xyz"), ("c", "zzz")]
    with tempfile.TemporaryDirectory() as tmp:
        for q in (1, 2, 3, 4):
            cat_dir, m = _build_catalog(tmp, docs, q=q, name=f"cat{q}")
            with SimilarityCatalog(cat_dir) as cat:
                for source in ("a", "b", "c"):
                    rows = _rows(cat, source, "0.000001")
                    assert rows == expected_query(
                        docs, q, source, Fraction(1, 10**6)), (q, source)
            if q > 3:
                # q=4: every document is shorter than q -> nothing comparable.
                assert m["documents"]["without_shingles"] == 3
                assert m["dictionary"]["qgram_count"] == 0


def test_short_documents_flagged_not_silent():
    docs = [("full", "abcdefg"), ("short", "abc"), ("empty", "")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, docs, q=5)
        assert m["documents"]["without_shingles"] == 2
        assert m["documents"]["with_shingles"] == 1
        with SimilarityCatalog(cat_dir) as cat:
            # Short documents produce no results and read no postings...
            assert _rows(cat, "short", "0.0001") == []
            assert cat.stats()["last_query"]["posting_records_read"] == 0
            assert _rows(cat, "empty", "0.0001") == []
            # ... and never appear as targets either.
            rows = _rows(cat, "full", "0.0001")
            assert rows == []
            # ... while docs_page flags them explicitly.
            page, cursor = cat.docs_page(10)
            flags = {r["doc_id"]: r["comparable"] for r in page}
            assert flags == {"full": True, "short": False, "empty": False}
            assert cursor is None
            short_row = [r for r in page if r["doc_id"] == "short"][0]
            assert short_row["text_len_codepoints"] == 3
            assert short_row["shingle_count"] == 0


def test_empty_documents_and_empty_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, [("e", ""), ("f", "")], q=1)
        assert m["documents"]["without_shingles"] == 2
        with SimilarityCatalog(cat_dir) as cat:
            assert _rows(cat, "e", "0.5") == []
            assert cat.validate()["ok"]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, [], q=3)
        assert m["documents"]["count"] == 0
        with SimilarityCatalog(cat_dir) as cat:
            assert cat.validate()["documents"] == 0
            with pytest.raises(KeyError):
                _rows(cat, "nobody", "0.5")


def test_emoji_nul_newline_shingles_and_explain_encoding():
    text_a = "😀\x00\n tail 😀\x00\n"
    text_b = "😀\x00\n tail 😀\x00\n!"
    docs = [("emoji", text_a), ("nul", text_b), ("other", "plain")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=3)
        with SimilarityCatalog(cat_dir) as cat:
            rows = _rows(cat, "emoji", "0.5")
            assert rows == expected_query(docs, 3, "emoji", Fraction(1, 2))
            shared = cat.explain("emoji", "nul")
            assert shared == expected_explain(docs, 3, "emoji", "nul")
            # NUL and non-ASCII survive as exact JSON strings plus hex.
            grams = {e["qgram"] for e in shared}
            assert "😀\x00\n" in grams
            entry = [e for e in shared if e["qgram"] == "😀\x00\n"][0]
            assert entry["qgram_utf8_hex"] == "😀\x00\n".encode().hex()
            assert entry["count_in_source"] == 2
            assert json.dumps(entry)  # JSON-safe representation


def test_unicode_doc_id_ordering_breaks_ties():
    ids = ["z", "é", "😀", "A", "a", "10"]
    docs = [(i, "same text everywhere") for i in ids]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        with SimilarityCatalog(cat_dir) as cat:
            rows = _rows(cat, "a", "0.5")
            # All similarities tie at 1/1: order is doc_id code point order.
            expect = sorted(i for i in ids if i != "a")
            assert [r["target_doc_id"] for r in rows] == expect
            assert expect == ["10", "A", "z", "é", "😀"]


def test_threshold_exactly_equal_vs_slightly_higher():
    # sim("s","t") = 1/2 exactly; sim("s","u") = 1/3 exactly.
    docs = [("s", "ab"), ("t", "a"), ("u", "ac")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=1)
        with SimilarityCatalog(cat_dir) as cat:
            assert [r["target_doc_id"] for r in _rows(cat, "s", "1/2")] == \
                ["t"]
            just_above_half = "500000000000000001/1000000000000000000"
            assert _rows(cat, "s", just_above_half) == []
            just_below_half = "499999999999999999/1000000000000000000"
            assert [r["target_doc_id"] for r in
                    _rows(cat, "s", just_below_half)] == ["t"]
            # 1/3 boundary: included exactly at 1/3, excluded just above.
            assert [r["target_doc_id"] for r in _rows(cat, "s", "1/3")] == \
                ["t", "u"]
            just_above_third = "333333333333333334/1000000000000000000"
            assert [r["target_doc_id"] for r in
                    _rows(cat, "s", just_above_third)] == ["t"]
            # Float 0.5 means exactly 1/2 (decimal), so it includes.
            assert [r["target_doc_id"] for r in _rows(cat, "s", 0.5)] == ["t"]
            assert _rows(cat, "s", "1") == []


def test_no_shared_shingles_makes_no_fake_candidates():
    docs = [("a", "abcdef"), ("b", "ghijkl"), ("c", "mnopqr")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        with SimilarityCatalog(cat_dir) as cat:
            assert _rows(cat, "a", "0.0001") == []
            st = cat.stats()["last_query"]
            # Postings of a's shingles were streamed (they contain only a
            # itself), but no candidate relation was fabricated.
            assert st["posting_records_read"] == expected_posting_records(
                docs, 2, "a")
            assert st["candidate_relations"] == 0
            assert st["candidates"] == 0
            assert st["results"] == 0


def test_unknown_source_doc_raises():
    docs = [("a", "abcdef")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        with SimilarityCatalog(cat_dir) as cat:
            with pytest.raises(KeyError):
                _rows(cat, "ghost", "0.5")
            with pytest.raises(KeyError):
                cat.explain("a", "ghost")


def test_invalid_arguments_rejected():
    docs = [("a", "abcdef")]
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        for bad_q in (0, -3, 1.5, "2", True):
            with pytest.raises((ValueError, TypeError)):
                SimilarityBuilder(idx_dir, os.path.join(tmp, "c"), q=bad_q)
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        with SimilarityCatalog(cat_dir) as cat:
            with pytest.raises(ValueError):
                cat.query_page("a", "0.5", 0)
            with pytest.raises(ValueError):
                cat.docs_page(0)


# --------------------------------------------------------------------------- #
# Hot templates: many candidates, no caps, spill paths
# --------------------------------------------------------------------------- #
def _template_docs(n, template_len=40, tail_len=8, seed=7):
    rng = random.Random(seed)
    template = "".join(rng.choice("abcdefgh") for _ in range(template_len))
    docs = []
    for i in range(n):
        tail = "".join(rng.choice("xyz") for _ in range(tail_len))
        docs.append((f"doc{i:03d}", template + tail + str(i)))
    return docs


def test_hot_template_all_candidates_reported():
    docs = _template_docs(60)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4,
                                    cat_kw={"chunk_items": 64,
                                            "chunk_bytes": 4096})
        with SimilarityCatalog(cat_dir, counter_chunk_items=50,
                               result_chunk_items=30,
                               result_chunk_bytes=2048) as cat:
            rows = _rows(cat, "doc000", "0.5")
            # Every other document shares the template: no silent caps.
            assert len(rows) == 59
            assert rows == expected_query(docs, 4, "doc000",
                                          Fraction(1, 2))
            st = cat.stats()["last_query"]
            # The counter really spilled and merged externally.
            assert st["counter_chunks_spilled"] > 1
            assert st["result_chunks_spilled"] > 1
            assert st["candidate_relations"] == st["posting_records_read"] - \
                st["posting_lists_read"]  # only the source itself skipped
            assert st["candidates"] == 59
            assert st["exact_scored"] == 59
            assert st["results"] == 59


def test_paging_exhausts_and_matches_unpaged():
    docs = _template_docs(40)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        with SimilarityCatalog(cat_dir) as cat:
            unpaged = _rows(cat, "doc005", "0.3")
            for page_size in (1, 2, 7, 64, 1000):
                collected = []
                cursor = None
                while True:
                    page, cursor = cat.query_page(
                        "doc005", "0.3", page_size, cursor)
                    collected.extend(page)
                    if cursor is None:
                        break
                assert collected == unpaged, page_size
                assert len(collected) == 39


def test_cursor_rejects_other_source_threshold_catalog_and_tampering():
    docs = _template_docs(12)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        cat2_dir, _ = _build_catalog(tmp, docs, q=4, name="cat2")
        with SimilarityCatalog(cat_dir) as cat, \
                SimilarityCatalog(cat2_dir) as cat2:
            page, cursor = cat.query_page("doc000", "0.3", 2)
            assert cursor is not None
            with pytest.raises(CursorError):
                cat.query_page("doc001", "0.3", 2, cursor)  # other source
            with pytest.raises(CursorError):
                cat.query_page("doc000", "0.4", 2, cursor)  # other threshold
            with pytest.raises(CursorError):
                cat2.query_page("doc000", "0.3", 2, cursor)  # other catalog
            with pytest.raises(CursorError):
                cat.query_page("doc000", "0.3", 2, cursor[:-2] + "00")
            with pytest.raises(CursorError):
                cat.query_page("doc000", "0.3", 2, "not-a-cursor")
            # The legitimate follow-up still works.
            page2, _ = cat.query_page("doc000", "0.3", 2, cursor)
            assert [r["target_doc_id"] for r in page2] == \
                [r["target_doc_id"]
                 for r in _rows(cat, "doc000", "0.3")[2:4]]


def test_docs_cursor_binding():
    docs = [(f"d{i}", f"text{i}") for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        cat2_dir, _ = _build_catalog(tmp, docs, q=2, name="cat2")
        with SimilarityCatalog(cat_dir) as cat, \
                SimilarityCatalog(cat2_dir) as cat2:
            page, cursor = cat.docs_page(2)
            assert len(page) == 2 and cursor is not None
            with pytest.raises(CursorError):
                cat2.docs_page(2, cursor)
            seen = list(page)
            while cursor is not None:
                page, cursor = cat.docs_page(2, cursor)
                seen.extend(page)
            assert [r["doc_id"] for r in seen] == [d for d, _ in docs]


# --------------------------------------------------------------------------- #
# Explain
# --------------------------------------------------------------------------- #
def test_explain_counts_and_order_and_no_side_effects():
    docs = [("a", "ababab"), ("b", "ababa"), ("c", "zz")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        with SimilarityCatalog(cat_dir) as cat:
            shared = cat.explain("a", "b")
            assert shared == expected_explain(docs, 2, "a", "b")
            by_gram = {e["qgram"]: e for e in shared}
            assert by_gram["ab"]["count_in_source"] == 3
            assert by_gram["ab"]["count_in_target"] == 2
            assert by_gram["ba"]["count_in_source"] == 2
            # Deterministic order: code point order of the qgrams.
            assert [e["qgram"] for e in shared] == ["ab", "ba"]
            # Explain does not perturb query results.
            before = _rows(cat, "a", "0.1")
            cat.explain("a", "b")
            assert _rows(cat, "a", "0.1") == before
            # No shingles -> empty explain.
            assert cat.explain("a", "c") == []
            assert cat.stats()["last_explain"]["shared_count"] == 0
            # Explaining a document against itself returns all its shingles.
            self_shared = cat.explain("a", "a")
            assert self_shared == expected_explain(docs, 2, "a", "a")
            assert len(self_shared) == 2  # {"ab", "ba"} distinct


def test_cursor_past_last_page_is_empty():
    docs = _template_docs(10)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        with SimilarityCatalog(cat_dir) as cat:
            total = _rows(cat, "doc000", "0.3")
            page, cursor = cat.query_page("doc000", "0.3", 100)
            assert len(page) == len(total) and cursor is None
            # Walking exactly to the end yields an empty page, no error.
            cursor = cat._make_cursor("doc000",
                                      parse_threshold("0.3"), len(total))
            page, cursor = cat.query_page("doc000", "0.3", 5, cursor)
            assert page == [] and cursor is None


# --------------------------------------------------------------------------- #
# Randomized differential run
# --------------------------------------------------------------------------- #
def test_randomized_differential():
    rng = random.Random(2024)
    alphabet = "ab😀\n\x00é"
    template = "".join(rng.choice(alphabet) for _ in range(15))
    docs = []
    for i in range(45):
        kind = i % 4
        if kind == 0:
            text = template + "".join(rng.choice(alphabet)
                                      for _ in range(rng.randint(0, 6)))
        elif kind == 1:
            text = "".join(rng.choice(alphabet)
                           for _ in range(rng.randint(0, 25)))
        elif kind == 2:
            text = template[: rng.randint(0, len(template))]
        else:
            text = ""
        docs.append((f"d{i:02d}", text))
    docs.append(("dup1", template))
    docs.append(("dup2", template))
    with tempfile.TemporaryDirectory() as tmp:
        for q in (1, 2, 5):
            cat_dir, _ = _build_catalog(
                tmp, docs, q=q, name=f"cat{q}",
                cat_kw={"chunk_items": 30, "chunk_bytes": 1500})
            with SimilarityCatalog(cat_dir, counter_chunk_items=40,
                                   result_chunk_items=25) as cat:
                for source, thr in (("d00", "0.2"), ("d01", "0.5"),
                                    ("d03", "0.9"), ("dup1", "0.75"),
                                    ("d11", "0.05")):
                    rows = _rows(cat, source, thr)
                    assert rows == expected_query(
                        docs, q, source, Fraction(thr)), (q, source, thr)
                    # Paged and unpaged agree.
                    collected = []
                    cursor = None
                    while True:
                        page, cursor = cat.query_page(source, thr, 3, cursor)
                        collected.extend(page)
                        if cursor is None:
                            break
                    assert collected == rows
                for a, b in (("d00", "d04"), ("dup1", "dup2"), ("d01", "d05")):
                    assert cat.explain(a, b) == expected_explain(
                        docs, q, a, b)
                assert cat.validate()["ok"]


# --------------------------------------------------------------------------- #
# Integrity, corruption, interruption, determinism
# --------------------------------------------------------------------------- #
def test_deterministic_rebuild_same_logical_output():
    docs = _template_docs(25)
    with tempfile.TemporaryDirectory() as tmp:
        cat1, m1 = _build_catalog(tmp, docs, q=4, name="c1")
        cat2, m2 = _build_catalog(tmp, docs, q=4, name="c2")
        # Data files and digests are byte-identical; only the per-build
        # catalog UUID differs.
        assert m1["files"] == m2["files"]
        assert m1["documents"] == m2["documents"]
        assert m1["relations"] == m2["relations"]
        assert m1["catalog_identity"] != m2["catalog_identity"]
        with SimilarityCatalog(cat1) as c1, SimilarityCatalog(cat2) as c2:
            assert _rows(c1, "doc003", "0.4") == _rows(c2, "doc003", "0.4")


def test_input_order_changes_bytes_but_not_results():
    docs = _template_docs(15)
    shuffled = docs[::2] + docs[1::2]
    with tempfile.TemporaryDirectory() as tmp:
        cat1, _ = _build_catalog(tmp, docs, q=4, name="c1")
        cat2, _ = _build_catalog(tmp, shuffled, q=4, name="c2")
        with SimilarityCatalog(cat1) as c1, SimilarityCatalog(cat2) as c2:
            for source in ("doc000", "doc007"):
                assert _rows(c1, source, "0.3") == _rows(c2, source, "0.3")


def _flip_last_byte(path):
    with open(path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 0xFF]))


def test_catalog_file_corruption_detected():
    docs = _template_docs(10)
    for victim in ("qgrams.bin", "qgramdir.bin", "postings.bin",
                   "docqgrams.bin", "docs.bin"):
        with tempfile.TemporaryDirectory() as tmp:
            cat_dir, _ = _build_catalog(tmp, docs, q=4)
            _flip_last_byte(os.path.join(cat_dir, victim))
            with pytest.raises(IndexCorruptError):
                SimilarityCatalog(cat_dir)


def test_catalog_truncation_and_missing_open_detected():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        with open(os.path.join(cat_dir, "postings.bin"), "r+b") as f:
            f.truncate(10)
        with pytest.raises(IndexCorruptError):
            SimilarityCatalog(cat_dir)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        os.unlink(os.path.join(cat_dir, "OPEN"))
        with pytest.raises(IndexCorruptError):
            SimilarityCatalog(cat_dir)


def test_manifest_tampering_detected():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        mpath = os.path.join(cat_dir, "manifest.json")
        manifest = json.load(open(mpath))
        manifest["files"]["postings.bin"]["sha256"] = "0" * 64
        json.dump(manifest, open(mpath, "w"))
        with pytest.raises(IndexCorruptError):
            SimilarityCatalog(cat_dir)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        mpath = os.path.join(cat_dir, "manifest.json")
        manifest = json.load(open(mpath))
        manifest["documents"]["count"] += 1
        json.dump(manifest, open(mpath, "w"))
        with pytest.raises(IndexCorruptError):
            SimilarityCatalog(cat_dir)


def test_bad_file_header_detected_despite_rehash():
    docs = _template_docs(6)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        # Rewrite the qgrams.bin header with the wrong kind tag, then
        # re-hash: only the per-file header check can catch this.
        path = os.path.join(cat_dir, "qgrams.bin")
        with open(path, "r+b") as f:
            f.seek(4)
            f.write((99).to_bytes(2, "little"))
        _refresh_manifest_hash(cat_dir, "qgrams.bin")
        with pytest.raises(IndexCorruptError):
            SimilarityCatalog(cat_dir)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        # Wrong magic entirely.
        path = os.path.join(cat_dir, "postings.bin")
        with open(path, "r+b") as f:
            f.write(b"XX")
        _refresh_manifest_hash(cat_dir, "postings.bin")
        with pytest.raises(IndexCorruptError):
            SimilarityCatalog(cat_dir)


def test_deep_validate_catches_sortedness_violations():
    docs = _template_docs(10)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, docs, q=4)
        # Swap two entries *inside* one multi-entry posting list, then
        # re-hash so the checksums pass: only deep validation notices.
        with SimilarityCatalog(cat_dir) as cat:
            target = None
            for qid in range(cat.n_qgrams):
                _bo, _bl, _r, post_off, post_count = cat._qgramdir_record(qid)
                if post_count >= 2:
                    target = post_off
                    break
        assert target is not None
        ppath = os.path.join(cat_dir, "postings.bin")
        with open(ppath, "r+b") as f:
            f.seek(sfmt.HEADER_SIZE + target * 8)
            e0 = f.read(8)
            e1 = f.read(8)
            f.seek(sfmt.HEADER_SIZE + target * 8)
            f.write(e1)
            f.write(e0)
        _refresh_manifest_hash(cat_dir, "postings.bin")
        with SimilarityCatalog(cat_dir) as cat:
            with pytest.raises(IndexCorruptError):
                cat.validate()
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, docs, q=4)
        # Swap the first two forward relations of a document that has at
        # least two (they are strictly increasing by construction).
        dpath = os.path.join(cat_dir, "docqgrams.bin")
        with open(dpath, "r+b") as f:
            f.seek(sfmt.HEADER_SIZE)
            r0 = f.read(4)
            r1 = f.read(4)
            assert r0 != r1
            f.seek(sfmt.HEADER_SIZE)
            f.write(r1)
            f.write(r0)
        _refresh_manifest_hash(cat_dir, "docqgrams.bin")
        with SimilarityCatalog(cat_dir) as cat:
            with pytest.raises(IndexCorruptError):
                cat.validate()


def _refresh_manifest_hash(cat_dir, name):
    mpath = os.path.join(cat_dir, "manifest.json")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["files"][name] = sfmt.hash_file(os.path.join(cat_dir, name))
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, sort_keys=True, indent=1)
        f.write("\n")


def test_interrupted_build_leaves_no_catalog():
    docs = _template_docs(10)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        cat_dir = os.path.join(tmp, "cat")
        builder = SimilarityBuilder(idx_dir, cat_dir, q=4, chunk_items=16)

        class _Boom(Exception):
            pass

        def boom(*a, **k):
            raise _Boom()

        builder._emit_dictionary_and_postings = boom
        with pytest.raises(_Boom):
            builder.build()
        assert not os.path.exists(cat_dir)
        assert not os.path.exists(cat_dir + sfmt.STAGING_SUFFIX)
        # Source index is untouched and still opens.
        from diskfm import DiskFMIndex
        with DiskFMIndex(idx_dir) as idx:
            assert idx.n_docs == len(docs)


def test_build_does_not_modify_source_index():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        from diskfm import DiskFMIndex
        with DiskFMIndex(idx_dir) as idx:
            before = dict(idx.manifest["files"])
            identity_before = idx.identity
        SimilarityBuilder(idx_dir, os.path.join(tmp, "cat"), q=4).build()
        with DiskFMIndex(idx_dir) as idx:
            assert dict(idx.manifest["files"]) == before
            assert idx.identity == identity_before


def test_stale_staging_dir_is_replaced_not_reused():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        cat_dir = os.path.join(tmp, "cat")
        staging = cat_dir + sfmt.STAGING_SUFFIX
        os.makedirs(staging)
        # Junk from a crashed earlier attempt must be discarded wholesale.
        with open(os.path.join(staging, "qgrams.bin"), "wb") as f:
            f.write(b"junk")
        manifest = SimilarityBuilder(idx_dir, cat_dir, q=4).build()
        assert manifest["dictionary"]["qgram_count"] > 0
        with SimilarityCatalog(cat_dir) as cat:
            assert cat.validate()["ok"]


def test_temp_space_failure_leaves_no_catalog():
    import diskfm.vsort as vsort_mod
    docs = _template_docs(20)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        cat_dir = os.path.join(tmp, "cat")
        real_tempfile = vsort_mod.tempfile

        class _FakeTempfile:
            @staticmethod
            def mkstemp(*a, **k):
                raise OSError(28, "No space left on device")

        vsort_mod.tempfile = _FakeTempfile
        try:
            builder = SimilarityBuilder(idx_dir, cat_dir, q=4,
                                        chunk_items=8, chunk_bytes=512)
            with pytest.raises(OSError):
                builder.build()
        finally:
            vsort_mod.tempfile = real_tempfile
        assert not os.path.exists(cat_dir)
        assert not os.path.exists(cat_dir + sfmt.STAGING_SUFFIX)


def test_existing_catalog_is_protected():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        cat_dir = os.path.join(tmp, "cat")
        SimilarityBuilder(idx_dir, cat_dir, q=4).build()
        with pytest.raises(FileExistsError):
            SimilarityBuilder(idx_dir, cat_dir, q=4)
        with pytest.raises(FileExistsError):
            SimilarityBuilder(idx_dir, cat_dir, q=9)
        # The pre-existing catalog is intact and valid.
        with SimilarityCatalog(cat_dir) as cat:
            assert cat.validate()["ok"]
            assert cat.q == 4


def test_broken_or_swapped_source_index_rejected_at_build():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        cat_dir = os.path.join(tmp, "cat")
        SimilarityBuilder(idx_dir, cat_dir, q=4).build()
        # Corrupt the source: new builds must fail, existing catalog safe.
        with open(os.path.join(idx_dir, "rank.bin"), "r+b") as f:
            f.truncate(12)
        with pytest.raises(IndexCorruptError):
            SimilarityBuilder(idx_dir, os.path.join(tmp, "cat2"), q=4)
        with SimilarityCatalog(cat_dir) as cat:
            assert cat.validate()["ok"]
    with tempfile.TemporaryDirectory() as tmp:
        idx_a = _build_index(tmp, docs, name="idxA")
        idx_b = _build_index(tmp, [("x", "different")], name="idxB")
        os.rename(idx_a, idx_a + "_moved")
        os.rename(idx_b, idx_a)  # swap: valid index, wrong identity
        from diskfm import DiskFMIndex
        with DiskFMIndex(idx_a) as idx:
            real_identity = idx.identity
        with pytest.raises(ValueError):
            SimilarityBuilder(idx_a, os.path.join(tmp, "cat"), q=4,
                              expect_identity="0" * 32)
        # Correct identity passes.
        SimilarityBuilder(idx_a, os.path.join(tmp, "cat"), q=4,
                          expect_identity=real_identity).build()


def test_max_codepoint_and_multilingual():
    docs = [("max", "a\U0010FFFFb\U0010FFFF"), ("mix", "héllo😀世界\x00"),
            ("max2", "a\U0010FFFFb\U0010FFFF")]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=2)
        with SimilarityCatalog(cat_dir) as cat:
            rows = _rows(cat, "max", "0.5")
            assert rows == expected_query(docs, 2, "max", Fraction(1, 2))
            assert [r["target_doc_id"] for r in rows] == ["max2"]
            shared = cat.explain("max", "max2")
            assert shared == expected_explain(docs, 2, "max", "max2")


def test_exotic_doc_ids_roundtrip():
    # NUL, newline, emoji and very long doc_ids are plain data.
    weird = ["a\x00b", "line\nbreak", "😀😀", "x" * 500, "普通名"]
    docs = [(w, "shared body text") for w in weird]
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=3)
        with SimilarityCatalog(cat_dir) as cat:
            for w in weird:
                rows = _rows(cat, w, "0.9")
                assert {r["target_doc_id"] for r in rows} == \
                    set(weird) - {w}
            page, _ = cat.docs_page(100)
            assert {r["doc_id"] for r in page} == set(weird)


def test_catalog_is_independent_of_source_index():
    # Once built, the catalog must answer everything on its own: deleting
    # the source index changes nothing (structural proof that queries can
    # never read the original text back).
    docs = _template_docs(12)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = _build_index(tmp, docs)
        cat_dir = os.path.join(tmp, "cat")
        SimilarityBuilder(idx_dir, cat_dir, q=4).build()
        import shutil
        shutil.rmtree(idx_dir)
        with SimilarityCatalog(cat_dir) as cat:
            rows = _rows(cat, "doc000", "0.5")
            assert rows == expected_query(docs, 4, "doc000",
                                          Fraction(1, 2))
            assert cat.explain("doc000", "doc001") == expected_explain(
                docs, 4, "doc000", "doc001")
            assert cat.validate()["ok"]


def test_tiny_pagers_still_correct():
    # Two-page LRU and four-record pages force constant page churn in every
    # catalog structure; results must be identical to the unpaged oracle.
    docs = _template_docs(30)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        with SimilarityCatalog(cat_dir, max_pages=2, page_records=4,
                               counter_chunk_items=7,
                               result_chunk_items=5,
                               result_chunk_bytes=512) as cat:
            rows = _rows(cat, "doc007", "0.4")
            assert rows == expected_query(docs, 4, "doc007",
                                          Fraction(2, 5))
            assert cat.validate()["ok"]
            shared = cat.explain("doc007", "doc008")
            assert shared == expected_explain(docs, 4, "doc007", "doc008")
            # Pagers really paged: misses far below the record counts.
            pagers = cat.stats()["pagers"]
            assert pagers["postings"]["resident_pages"] <= 2
            assert pagers["postings"]["page_misses"] > 1


def test_query_temp_dirs_are_cleaned_up():
    docs = _template_docs(8)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, _ = _build_catalog(tmp, docs, q=4)
        made = []
        import diskfm.simquery as sq
        real_mkdtemp = sq.tempfile.mkdtemp

        def spy(*a, **k):
            d = real_mkdtemp(*a, **k)
            made.append(d)
            return d

        sq.tempfile.mkdtemp = spy
        try:
            with SimilarityCatalog(cat_dir) as cat:
                _rows(cat, "doc000", "0.5")
                with pytest.raises(KeyError):
                    _rows(cat, "ghost", "0.5")
                cat.query_page("doc000", "0.5", 2)
        finally:
            sq.tempfile.mkdtemp = real_mkdtemp
        assert len(made) == 3
        for d in made:
            assert not os.path.exists(d)


def test_stats_are_real_counters():
    docs = _template_docs(15)
    with tempfile.TemporaryDirectory() as tmp:
        cat_dir, m = _build_catalog(tmp, docs, q=4,
                                    cat_kw={"chunk_items": 32,
                                            "chunk_bytes": 2048})
        bs = m["build_stats"]
        assert bs["raw_relations"] == m["relations"]["raw"]
        assert bs["deduped_relations"] == m["relations"]["deduped"]
        assert bs["external_sort_runs"] >= 2
        assert bs["temp_bytes"] > 0
        assert bs["source_content_bytes_read"] > 0
        assert bs["peak_rss_kb"] is None or bs["peak_rss_kb"] > 0
        with SimilarityCatalog(cat_dir) as cat:
            _rows(cat, "doc000", "0.5")
            st = cat.stats()["last_query"]
            assert st["posting_records_read"] == expected_posting_records(
                docs, 4, "doc000")
            assert st["posting_lists_read"] == len(
                shingle_set(dict(docs)["doc000"], 4))
            assert st["results"] == len(
                expected_query(docs, 4, "doc000", Fraction(1, 2)))
            s = cat.stats()
            for pager in ("qgramdir", "postings", "docs", "docqgrams"):
                assert "resident_bytes" in s["pagers"][pager]

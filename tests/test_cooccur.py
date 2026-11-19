"""Differential tests for windowed multi-phrase co-occurrence.

Expected values come from ``cooccur_oracle.py``, a brute-force enumerator
that shares no code with the implementation.  Tiny sort buffers force real
external-spill paths; the tests cover overlapping and adjacent hits, window
boundaries, emoji/NUL/newlines with both coordinate kinds, identical text
under distinct ids, reversed pair order, cross-document isolation, zero-hit
phrases, dense hot-document pairings, cursor binding, corruption, temp
space failure and stats-that-are-measured.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diskfm import (  # noqa: E402
    CooccurError,
    CursorError,
    DiskFMIndex,
    IndexBuilder,
    IndexCorruptError,
    build_request,
)
from diskfm import cooccur as co  # noqa: E402
from diskfm.cooccur import compute_analysis  # noqa: E402

from cooccur_oracle import (  # noqa: E402
    expected_pair_counts,
    expected_pairs,
    expected_summary,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _analysis(index, patterns, pairs, window, cache, **kw):
    return compute_analysis(
        index, build_request(patterns, pairs, window),
        cache_root=cache, **kw)


def _collect(analysis) -> list:
    return list(analysis.iter_summary())


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_request_rejects_structure_errors():
    good = [("A", "ab"), ("B", "ba")]
    with pytest.raises(CooccurError):
        build_request(good, [("A", "B")], -1)
    with pytest.raises(CooccurError):
        build_request(good, [("A", "B")], True)
    with pytest.raises(CooccurError):
        build_request([], [("A", "B")], 1)
    with pytest.raises(CooccurError):
        build_request(good, [], 1)
    with pytest.raises(CooccurError):
        build_request([("A", "ab"), ("A", "bc")], [("A", "B")], 1)
    with pytest.raises(CooccurError):
        build_request([("A", "")], [("A", "B")], 1)
    with pytest.raises(CooccurError):
        build_request([("A", "ab")], [("A", "A")], 1)
    with pytest.raises(CooccurError):
        build_request(good, [("A", "Z")], 1)
    with pytest.raises(CooccurError):
        build_request(good, [("A", "B"), ("A", "B")], 1)
    with pytest.raises(CooccurError):
        build_request(good, [("B", "A")], 2 ** 40)
    with pytest.raises(CooccurError):
        build_request([("A", "\ud800")], [("A", "B")], 1)
    # A valid request normalises identically regardless of input shape.
    a = build_request(good, [("B", "A")], 3)
    b = build_request([{"pattern_id": "A", "phrase": "ab"},
                       {"pattern_id": "B", "phrase": "ba"}],
                      [["B", "A"]], 3)
    assert a == b


def test_duplicate_text_distinct_ids_are_distinct_items():
    docs = [("d1", "xyxyxy")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, [("P", "xy"), ("Q", "xy")],
                           [("P", "Q"), ("Q", "P")], 0, cache)
            assert _collect(an) == expected_summary(
                docs, [("P", "xy"), ("Q", "xy")],
                [("P", "Q"), ("Q", "P")], 0)
            for pair in [("P", "Q"), ("Q", "P")]:
                assert list(an.iter_pairs(pair)) == expected_pairs(
                    docs, [("P", "xy"), ("Q", "xy")], pair, 0)
            assert an.pair_counts == [3, 3]


# --------------------------------------------------------------------------- #
# Exact boundary semantics
# --------------------------------------------------------------------------- #
def test_overlap_adjacency_and_window_boundaries():
    # "aaa" produces overlapping hits at 0,1,2; "b" at 3.
    text = "aaab"
    docs = [("d", text)]
    patterns = [("A", "aaa"), ("B", "b")]
    expected_zero = expected_pairs(docs, patterns, ("A", "B"), 0)
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 0, cache)
            assert list(an.iter_pairs(("A", "B"))) == expected_zero
            # diff at "A"=2,"B"=3 is exactly 1: included at window 1...
            an1 = _analysis(idx, patterns, [("A", "B")], 1, cache)
            p1 = list(an1.iter_pairs(("A", "B")))
            assert p1 == expected_pairs(docs, patterns, ("A", "B"), 1)
            # "aaa" has exactly one hit (offset 0); "b" is at 3, so the
            # difference is 3: window 2 excludes it, window 3 includes it.
            assert not any(r["a_codepoint_offset"] == 0 and
                           r["b_codepoint_offset"] == 3 for r in p1)
            p2 = list(_analysis(idx, patterns, [("A", "B")], 2,
                                cache).iter_pairs(("A", "B")))
            assert p2 == expected_pairs(docs, patterns, ("A", "B"), 2)
            assert not p2
            p3 = list(_analysis(idx, patterns, [("A", "B")], 3,
                                cache).iter_pairs(("A", "B")))
            assert p3 == expected_pairs(docs, patterns, ("A", "B"), 3)
            assert len(p3) == 1


def test_swapped_pair_order_is_independent_and_each_counted_once():
    docs = [("d", "abab")]
    patterns = [("A", "a"), ("B", "b")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B"), ("B", "A")], 1, cache)
            ab = list(an.iter_pairs(("A", "B")))
            ba = list(an.iter_pairs(("B", "A")))
            assert ab == expected_pairs(docs, patterns, ("A", "B"), 1)
            assert ba == expected_pairs(docs, patterns, ("B", "A"), 1)
            # (A@0,B@1) and (B@1,A@0) are different analysis items.
            assert len(ab) == len(ba) == 3
            # No combination is emitted twice inside one pair.
            keys = [(r["doc_id"], r["a_codepoint_offset"],
                     r["b_codepoint_offset"]) for r in ab]
            assert len(keys) == len(set(keys))


def test_multiple_documents_and_no_cross_document_join():
    # "a" ends doc1 and "b" starts doc2: suffixes would be adjacent but the
    # join must not connect them.
    docs = [("d1", "xxa"), ("d2", "byy")]
    patterns = [("A", "a"), ("B", "b")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 100, cache)
            assert list(an.iter_summary()) == []
            assert list(an.iter_pairs(("A", "B"))) == []


def test_no_hits_and_high_frequency_dense_pairs():
    # A hot document: 200 overlapping hits of "aa" against 200 of "ab"... we
    # instead use a long run so the pair count is exactly known.
    text = "a" * 120
    docs = [("hot", text)]
    patterns = [("A", "a"), ("B", "aa")]
    window = 2
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            # Phrase "zzz" has no hits: excluded, never a fake zero row.
            response = _analysis(
                idx, patterns + [("Z", "zzz")],
                [("A", "B"), ("A", "Z")], window, cache,
                chunk_items=7, chunk_bytes=128)
            assert list(response.iter_summary()) == expected_summary(
                docs, patterns + [("Z", "zzz")],
                [("A", "B"), ("A", "Z")], window)
            got = list(response.iter_pairs(("A", "B")))
            assert got == expected_pairs(docs, patterns, ("A", "B"), window)
            assert response.pair_counts[1] == 0
            # Real spill happened under these tiny buffers.
            stats = response.stats()["analysis"]
            assert stats["hit_sort_runs"] > 1
            assert stats["pair_sort_runs"] >= 1
            assert stats["position_pairs"] == len(got)


def test_dense_hot_document_exact_count_and_unique_pairs():
    # 150 hits of "aa" (offsets 0..148) and 150 of "ab" would require an
    # alternating text; instead use a mixed run where counts are known by
    # the oracle and the pair number is large enough to force spill.
    text = ("ab" * 60) + ("a" * 60)
    docs = [("hot", text), ("other", "abab" * 10)]
    patterns = [("A", "a"), ("B", "b"), ("C", "ab")]
    pairs = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")]
    window = 3
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, pairs, window, cache,
                           chunk_items=13, chunk_bytes=256)
            assert _collect(an) == expected_summary(
                docs, patterns, pairs, window)
            totals = []
            for pair in pairs:
                got = list(an.iter_pairs(pair))
                assert got == expected_pairs(docs, patterns, pair, window)
                assert len(got) == len({(r["doc_id"],
                                         r["a_codepoint_offset"],
                                         r["b_codepoint_offset"])
                                        for r in got})
                totals.append(len(got))
            assert an.pair_counts == totals == expected_pair_counts(
                docs, patterns, pairs, window)
            assert sum(totals) > 1000  # genuinely dense
            stats = an.stats()["analysis"]
            assert stats["position_pairs"] == sum(totals)
            assert stats["pair_sort_runs"] >= 2
            assert stats["temp_disk_bytes"] > 0


# --------------------------------------------------------------------------- #
# Coordinates: codepoint vs UTF-8 byte
# --------------------------------------------------------------------------- #
def test_emoji_nul_newline_coordinates():
    text = "😀a\x00b\nc😀a"
    docs = [("d", text)]
    patterns = [("A", "a"), ("B", "b"), ("E", "😀"), ("N", "\x00"),
                ("L", "\n")]
    pairs = [("A", "B"), ("E", "A"), ("N", "L"), ("E", "N")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            for window in (0, 1, 4):
                an = _analysis(idx, patterns, pairs, window, cache)
                assert _collect(an) == expected_summary(
                    docs, patterns, pairs, window)
                for pair in pairs:
                    assert list(an.iter_pairs(pair)) == expected_pairs(
                        docs, patterns, pair, window)
            # Explicit coordinate check across the emoji (4 UTF-8 bytes).
            an = _analysis(idx, [("A", "a"), ("E", "😀")],
                           [("E", "A")], 1, cache)
            row = list(an.iter_pairs(("E", "A")))[0]
            assert row["a_codepoint_offset"] == 0
            assert row["a_utf8_byte_offset"] == 0
            assert row["b_codepoint_offset"] == 1
            assert row["b_utf8_byte_offset"] == 4


def test_zero_window_long_phrases_and_multibyte():
    docs = [("z", "the quick 😀้ามี fox; the quick brown fox"),
            ("y", "😀้ามี😀้ามี")]
    patterns = [("Q", "the quick"), ("F", "brown fox"), ("M", "😀้ามี")]
    pairs = [("Q", "Q2"), ("Q", "F"), ("M", "M2")]
    patterns = [("Q", "the quick"), ("F", "brown fox"), ("M", "😀้ามี"),
                ("Q2", "the quick"), ("M2", "😀้ามี")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, pairs, 0, cache)
            assert _collect(an) == expected_summary(
                docs, patterns, pairs, 0)
            for pair in pairs:
                assert list(an.iter_pairs(pair)) == expected_pairs(
                    docs, patterns, pair, 0)
            assert an.pair_counts == expected_pair_counts(
                docs, patterns, pairs, 0)


# --------------------------------------------------------------------------- #
# Randomized differential
# --------------------------------------------------------------------------- #
def test_randomized_differential_against_oracle():
    rng = random.Random(20261119)
    alphabet = "ab😀\x00\n"
    docs = []
    for d in range(14):
        n = rng.randint(0, 45)
        docs.append((f"doc{d:02d}", "".join(rng.choice(alphabet)
                                            for _ in range(n))))
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            for trial in range(20):
                npats = rng.randint(2, 4)
                patterns = []
                for i in range(npats):
                    length = rng.randint(1, 3)
                    patterns.append((f"p{i}", "".join(
                        rng.choice(alphabet) for _ in range(length))))
                ids = [p[0] for p in patterns]
                pairs = []
                for a in ids:
                    for b in ids:
                        if a != b and rng.random() < 0.5:
                            pairs.append((a, b))
                if not pairs:
                    pairs = [(ids[0], ids[1])]
                window = rng.randint(0, 6)
                an = _analysis(idx, patterns, pairs, window, cache,
                               chunk_items=rng.randint(2, 20),
                               chunk_bytes=rng.choice([64, 256, 4096]))
                assert _collect(an) == expected_summary(
                    docs, patterns, pairs, window), (trial, patterns,
                                                     pairs, window)
                for pair in pairs:
                    assert list(an.iter_pairs(pair)) == expected_pairs(
                        docs, patterns, pair, window), (trial, pair)
                assert an.pair_counts == expected_pair_counts(
                    docs, patterns, pairs, window)


# --------------------------------------------------------------------------- #
# Paging and cursors
# --------------------------------------------------------------------------- #
def test_byte_offsets_are_one_pass_per_document():
    # A dense run of emoji (4 bytes each) with many overlapping hits: byte
    # offsets must be exact and the decoder must read each document's bytes
    # at most once, not once per hit.
    text = ("😀a" * 60) + "😀"
    docs = [("d", text), ("e", "😀😀😀")]
    patterns = [("A", "a"), ("E", "😀"), ("EA", "😀a")]
    pairs = [("A", "E"), ("E", "A"), ("EA", "A")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            before = idx.stats()["content_bytes_read"]
            an = _analysis(idx, patterns, pairs, 4, cache,
                           chunk_items=9, chunk_bytes=256)
            for pair in pairs:
                assert list(an.iter_pairs(pair)) == expected_pairs(
                    docs, patterns, pair, 4)
            total_bytes = sum(len(t.encode("utf-8")) for _d, t in docs)
            read = idx.stats()["content_bytes_read"] - before
            # One byte pass per phrase per document at most; never a prefix
            # walk per hit.  Without a cursor this would be far larger.
            hit_total = an.stats()["analysis"]["hits_read"]
            assert hit_total > 150
            assert read <= 3 * total_bytes


def test_summary_paging_exhausts_and_matches_unpaged():
    docs = [(f"doc{d:03d}", "aba" * (d + 1)) for d in range(25)]
    patterns = [("A", "aba"), ("B", "ba")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 2, cache)
            full = _collect(an)
            assert full == expected_summary(docs, patterns,
                                            [("A", "B")], 2)
            page_size = 7
            rows, cursor = an.summary_page(page_size)
            seen = list(rows)
            pages = 1
            while cursor is not None:
                rows, cursor = an.summary_page(page_size, cursor)
                seen.extend(rows)
                pages += 1
            assert seen == full
            assert pages == (len(full) + page_size - 1) // page_size


def test_pair_paging_exhausts_and_matches_unpaged():
    docs = [("d", "a" * 80)]
    patterns = [("A", "a"), ("B", "aa")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 4, cache,
                           chunk_items=19, chunk_bytes=512)
            full = list(an.iter_pairs(("A", "B")))
            assert full == expected_pairs(docs, patterns, ("A", "B"), 4)
            rows, cursor = an.pairs_page(("A", "B"), 17)
            seen = list(rows)
            while cursor is not None:
                rows, cursor = an.pairs_page(("A", "B"), 17, cursor)
                seen.extend(rows)
            assert seen == full


def test_cursor_rejects_changed_request_index_and_tampering():
    docs = [("a", "abab"), ("b", "baba")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_a = _build_index(tmp, docs, name="idxA")
        idx_b = _build_index(tmp, docs, name="idxB")
        with DiskFMIndex(idx_a) as ia, DiskFMIndex(idx_b) as ib:
            an = _analysis(ia, [("A", "a"), ("B", "b")], [("A", "B")], 5,
                           cache)
            _rows, cursor = an.pairs_page(("A", "B"), 1)
            assert cursor is not None
            # Same request, same index: resumes.
            assert an.pairs_page(("A", "B"), 1, cursor)[0]
            # Different index identity even though content is identical.
            other = _analysis(ib, [("A", "a"), ("B", "b")], [("A", "B")], 5,
                              cache)
            with pytest.raises(CursorError):
                other.pairs_page(("A", "B"), 1, cursor)
            # Different window.
            changed = _analysis(ia, [("A", "a"), ("B", "b")], [("A", "B")],
                                6, cache)
            with pytest.raises(CursorError):
                changed.pairs_page(("A", "B"), 1, cursor)
            # Different pair order (request digest changes).
            reordered = _analysis(ia, [("A", "a"), ("B", "b")],
                                  [("B", "A")], 5, cache)
            with pytest.raises(CursorError):
                reordered.pairs_page(("B", "A"), 1, cursor)
            # Changed phrase text for the same ids.
            changed_phrase = _analysis(ia, [("A", "ab"), ("B", "b")],
                                       [("A", "B")], 5, cache)
            with pytest.raises(CursorError):
                changed_phrase.pairs_page(("A", "B"), 1, cursor)
            # Wrong part.
            with pytest.raises(CursorError):
                an.summary_page(1, cursor)
            # A pair that is not in this request at all is a request error,
            # never a silent empty page.
            with pytest.raises(CooccurError):
                an.pairs_page(("B", "A"), 1, cursor)
            # Tampering.
            tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
            with pytest.raises(CursorError):
                an.pairs_page(("A", "B"), 1, tampered)
            with pytest.raises(CursorError):
                an.pairs_page(("A", "B"), 1, "not-a-cursor")


def test_cursor_resumes_across_processes():
    docs = [("p", "ab" * 40)]
    patterns = [("A", "a"), ("B", "b")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        req_path = os.path.join(tmp, "req.json")
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump({"patterns": [{"pattern_id": "A", "phrase": "a"},
                                    {"pattern_id": "B", "phrase": "b"}],
                       "pairs": [["A", "B"]], "window": 2}, f)
        env = dict(os.environ, DISKFM_COOCCUR_CACHE=cache)
        first = subprocess.run(
            [sys.executable, "-m", "diskfm.cooccurcli", "run", idx_dir,
             req_path, "--pair", "A", "B", "--pair-page-size", "5"],
            check=True, capture_output=True, text=True, env=env, cwd=ROOT)
        first_rows = [json.loads(line) for line in first.stdout.splitlines()]
        token_line = [l for l in first.stderr.splitlines()
                      if l.startswith("next_cursor=")]
        assert token_line
        cursor = token_line[0].split("=", 1)[1]
        second = subprocess.run(
            [sys.executable, "-m", "diskfm.cooccurcli", "run", idx_dir,
             req_path, "--pair", "A", "B", "--pair-page-size", "5",
             "--cursor", cursor],
            check=True, capture_output=True, text=True, env=env, cwd=ROOT)
        second_rows = [json.loads(line) for line in second.stdout.splitlines()]
        full = subprocess.run(
            [sys.executable, "-m", "diskfm.cooccurcli", "run", idx_dir,
             req_path, "--pair", "A", "B", "--all"],
            check=True, capture_output=True, text=True, env=env, cwd=ROOT)
        full_rows = [json.loads(line) for line in full.stdout.splitlines()]
        assert first_rows + second_rows == full_rows[:len(first_rows)
                                                      + len(second_rows)]
        assert len(full_rows) == 79


# --------------------------------------------------------------------------- #
# Determinism and stats
# --------------------------------------------------------------------------- #
def test_deterministic_repeat_runs():
    docs = [("a", "ababab"), ("b", "baba"), ("c", "zzz")]
    patterns = [("A", "ab"), ("B", "ba")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            first = _analysis(idx, patterns, [("A", "B")], 3, cache)
            snapshot = (_collect(first), list(first.iter_pairs(("A", "B"))),
                        first.pair_counts,
                        first.stats()["analysis"]["content_sha256"])
            first.close()
            second = _analysis(idx, patterns, [("A", "B")], 3, cache)
            assert (_collect(second),
                    list(second.iter_pairs(("A", "B"))),
                    second.pair_counts,
                    second.stats()["analysis"]["content_sha256"]) == snapshot
            # A forced recomputation gives the same logical result.
            third = compute_analysis(
                idx, build_request(patterns, [("A", "B")], 3),
                cache_root=cache, refresh=True)
            assert (_collect(third),
                    list(third.iter_pairs(("A", "B")))) == snapshot[:2]


def test_stats_are_measured_counters():
    docs = [("d", "a" * 50)]
    patterns = [("A", "a"), ("B", "aa")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 2, cache,
                           chunk_items=8, chunk_bytes=128)
            stats = an.stats()["analysis"]
            # Hits actually read are exactly the oracle's hit totals.
            assert stats["hits_read"] == 50 + 49
            assert stats["hits_per_phrase"] == [50, 49]
            assert stats["hit_sort_runs"] >= 2
            assert stats["pair_sort_runs"] >= 1
            assert stats["position_pairs"] == an.pair_counts[0]
            assert stats["candidate_pairs"] == stats["position_pairs"]
            assert stats["summary_rows"] == 1
            assert stats["temp_disk_bytes"] > 0
            # Paging counters move only with real page reads.
            an.summary_page(1)
            an.pairs_page(("A", "B"), 10)
            paging = an.stats()["paging"]
            assert paging["summary_pages_served"] == 1
            assert paging["pair_pages_served"] == 1
            assert paging["summary_records_read"] == 1
            assert paging["pair_records_read"] == 10
            assert paging["pairs_pager"]["resident_bytes"] <= \
                16 * 4096 * 8


# --------------------------------------------------------------------------- #
# Corruption, failure isolation, source safety
# --------------------------------------------------------------------------- #
def _cache_dir_for(tmp, idx_dir, request):
    identity = None
    with DiskFMIndex(idx_dir) as idx:
        identity = idx.identity
    return os.path.join(tmp, "cache", identity, request.digest())


def test_corrupt_cached_analysis_is_reported():
    docs = [("d", "aba" * 10)]
    patterns = [("A", "aba"), ("B", "ba")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 2, cache)
            directory = an.directory
            an.close()
            # Flip a byte in the published pair stream.
            target = os.path.join(directory, "pairs.bin")
            with open(target, "r+b") as f:
                f.seek(os.path.getsize(target) - 1)
                b = f.read(1)
                f.seek(-1, os.SEEK_END)
                f.write(bytes([b[0] ^ 0xFF]))
            with pytest.raises(IndexCorruptError):
                compute_analysis(idx, build_request(patterns, [("A", "B")], 2),
                                 cache_root=cache)


def test_truncated_analysis_file_is_reported():
    docs = [("d", "aba" * 10)]
    patterns = [("A", "aba"), ("B", "ba")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            an = _analysis(idx, patterns, [("A", "B")], 2, cache)
            directory = an.directory
            an.close()
            target = os.path.join(directory, "summary.bin")
            with open(target, "r+b") as f:
                f.truncate(max(co._CONTAINER_HEADER_SIZE, 20))
            with pytest.raises(IndexCorruptError):
                compute_analysis(
                    idx, build_request(patterns, [("A", "B")], 2),
                    cache_root=cache)


def test_temp_space_failure_leaves_no_published_analysis():
    docs = [("d", "aba" * 10)]
    patterns = [("A", "aba"), ("B", "ba")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        real_mkstemp = co.tempfile.mkstemp

        def boom(*a, **k):
            raise OSError(28, "No space left on device")

        with DiskFMIndex(idx_dir) as idx:
            request = build_request(patterns, [("A", "B")], 2)
            directory = _cache_dir_for(tmp, idx_dir, request)
            co.tempfile.mkstemp = boom
            try:
                with pytest.raises(OSError):
                    compute_analysis(idx, request, cache_root=cache)
            finally:
                co.tempfile.mkstemp = real_mkstemp
            assert not os.path.exists(directory)
            assert not os.path.exists(directory + co._STAGING_SUFFIX)
            # The index is untouched and a clean retry succeeds.
            an = compute_analysis(idx, request, cache_root=cache)
            assert an.pair_counts == expected_pair_counts(
                docs, patterns, [("A", "B")], 2)


def test_source_index_is_not_modified():
    docs = [("d", "abab")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            before = dict(idx.manifest["files"])
            identity = idx.identity
            _rows = _analysis(idx, [("A", "a"), ("B", "b")], [("A", "B")],
                              1, cache)
            assert dict(idx.manifest["files"]) == before
            assert idx.identity == identity
        # Nothing was written into the sealed index directory.
        assert sorted(os.listdir(idx_dir)) == sorted(
            ["manifest.json", "OPEN", "text.bin", "content.bin",
             "bytesample.bin", "doctable.bin", "bwt.bin", "rank.bin",
             "rankdir.bin", "sa_sample.bin"])


def test_broken_index_rejected_before_analysis():
    docs = [("d", "abab")]
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        idx_dir = _build_index(tmp, docs)
        os.unlink(os.path.join(idx_dir, "OPEN"))
        with pytest.raises(IndexCorruptError):
            DiskFMIndex(idx_dir)


# --------------------------------------------------------------------------- #
# Spill seam: results are identical with and without external spill
# --------------------------------------------------------------------------- #
def test_spill_vs_in_memory_identical():
    rng = random.Random(7)
    docs = [(f"d{i}", "".join(rng.choice("abc") for _ in range(60)))
            for i in range(8)]
    patterns = [("A", "a"), ("B", "b"), ("C", "c"), ("AB", "ab")]
    pairs = [("A", "B"), ("A", "AB"), ("C", "AB")]
    with tempfile.TemporaryDirectory() as tmp:
        cache_big = os.path.join(tmp, "big")
        cache_small = os.path.join(tmp, "small")
        idx_dir = _build_index(tmp, docs)
        with DiskFMIndex(idx_dir) as idx:
            big = _analysis(idx, patterns, pairs, 3, cache_big,
                            chunk_items=100000, chunk_bytes=1 << 22)
            small = _analysis(idx, patterns, pairs, 3, cache_small,
                              chunk_items=3, chunk_bytes=96)
            assert _collect(big) == _collect(small) == expected_summary(
                docs, patterns, pairs, 3)
            for pair in pairs:
                assert list(big.iter_pairs(pair)) == \
                    list(small.iter_pairs(pair)) == expected_pairs(
                        docs, patterns, pair, 3)
            assert small.stats()["analysis"]["hit_sort_runs"] > \
                big.stats()["analysis"]["hit_sort_runs"]
            big.close()
            small.close()

"""Functional and differential tests for the desensitisation release pipeline.

Covers the specification's acceptance scenario (two rounds over
``A😀秘密BC``), generated-fragment truncation, identical-text-but-replaced
identity separation, adjacent edits, whole-document deletion, unchanged
documents, NUL/newline/multi-byte edge cases, every whole-round rejection
condition, tamper/corruption detection, interrupted builds and deterministic
identity under record shuffling and chunk-size changes.

Bidirectional results are checked against ``pub_oracle``, an independent
per-codepoint provenance reconstruction that shares no production code.
"""

from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diskfm.builder import IndexBuilder  # noqa: E402
from diskfm.pubbuilder import ReleaseBuilder  # noqa: E402
from diskfm.pubformat import (PublishRejected, ReleaseCorruptError)  # noqa
from diskfm.pubquery import ReleasePackage  # noqa: E402
from diskfm.query import DiskFMIndex  # noqa: E402
import pytest  # noqa: E402
from pub_oracle import (apply_rounds, forward_map, provenance_at,
                        traceback)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def write_corpus(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(json.dumps({"doc_id": doc_id, "text": text},
                               ensure_ascii=False) + "\n")


def build_index(tmp, docs, name="idx", **kw):
    jp = os.path.join(tmp, "in.jsonl")
    write_corpus(jp, docs)
    idx = os.path.join(tmp, name)
    b = IndexBuilder(idx, chunk_items=kw.pop("chunk_items", 16),
                     sample_interval=kw.pop("sample_interval", 8))
    b.ingest_jsonl(jp)
    b.build()
    return idx


def write_edit_list(path, expected_identity, round_id, records):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(
            {"expected_input_identity": expected_identity,
             "round_id": round_id}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def publish(source, path, expected_identity, round_id, records,
            target=None, **kw):
    el = path
    write_edit_list(el, expected_identity, round_id, records)
    target = target or (path + ".pub")
    return ReleaseBuilder(source, el, target, **kw).build(), target


def index_identity(idx):
    with DiskFMIndex(idx) as ix:
        return ix.identity


# --------------------------------------------------------------------- #
# Specification acceptance scenario
# --------------------------------------------------------------------- #
def test_acceptance_two_rounds_and_migration():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d1", "A😀秘密BC")])
        root = index_identity(idx)
        m1, p1 = publish(
            idx, os.path.join(tmp, "r1"), root, "round-1",
            [{"edit_id": "e1", "doc_id": "d1", "start": 2, "end": 4,
              "replacement": "•"}], target=os.path.join(tmp, "pub1"))
        assert ReleasePackage(p1).document_text("d1") == "A😀•BC"
        m2, p2 = publish(
            p1, os.path.join(tmp, "r2"), m1["package_identity"], "round-2",
            [{"edit_id": "e2a", "doc_id": "d1", "start": 0, "end": 1,
              "replacement": "前置"},
             {"edit_id": "e2b", "doc_id": "d1", "start": 2, "end": 4,
              "replacement": ""}], target=os.path.join(tmp, "pub2"))
        with ReleasePackage(p2) as pkg:
            assert pkg.document_text("d1") == "前置😀C"
            r = pkg.forward_map("d1", 1, 6)
            mapped = r["mapped"]
            assert [m["baseline"]["codepoints"] for m in mapped] == [
                [1, 2], [5, 6]]
            assert [m["release"]["codepoints"] for m in mapped] == [
                [2, 3], [3, 4]]
            assert mapped[0]["release"]["utf8_bytes"] == [6, 10]
            assert mapped[1]["release"]["utf8_bytes"] == [10, 11]
            assert [lo["baseline"]["codepoints"]
                    for lo in r["lost"]] == [[2, 5]]
            # Lost fragment byte boundaries come from retained widths.
            assert r["lost"][0]["baseline"]["utf8_bytes"] == [5, 12]
            assert pkg.validate()["rounds"] == 2


def test_acceptance_matches_independent_oracle():
    baseline = "A😀秘密BC"
    rounds = [
        {"round_id": "round-1", "edits": [
            {"edit_id": "e1", "start": 2, "end": 4, "replacement": "•"}]},
        {"round_id": "round-2", "edits": [
            {"edit_id": "e2a", "start": 0, "end": 1,
             "replacement": "前置"},
            {"edit_id": "e2b", "start": 2, "end": 4,
             "replacement": ""}]},
    ]
    doc = apply_rounds(baseline, rounds)
    assert doc.text == "前置😀C"
    oracle_fwd = forward_map(baseline, doc, 1, 6)
    assert [m["release"]["codepoints"] for m in oracle_fwd["mapped"]] == [
        [2, 3], [3, 4]]
    assert [lo["baseline"]["codepoints"]
            for lo in oracle_fwd["lost"]] == [[2, 5]]


def test_generated_fragment_middle_deletion_keeps_origin():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "0123456789")])
        root = index_identity(idx)
        m1, p1 = publish(
            idx, tmp + "/r1", root, "r1",
            [{"edit_id": "g", "doc_id": "d", "start": 2, "end": 5,
              "replacement": "ABCDE"}], target=tmp + "/p1")
        m2, p2 = publish(
            p1, tmp + "/r2", m1["package_identity"], "r2",
            [{"edit_id": "cut", "doc_id": "d", "start": 4, "end": 6,
              "replacement": ""}], target=tmp + "/p2")
        with ReleasePackage(p2) as pkg:
            assert pkg.document_text("d") == "01ABE56789"
            tr = pkg.traceback("d", 0, 10)
            gens = [(p["generation"]["round_id"],
                     p["generation"]["edit_id"],
                     tuple(p["generation"]["relative"]["codepoints"]))
                    for p in tr["pieces"] if p["source"] == "generated"]
            assert ("r1", "g", (0, 2)) in gens
            assert ("r1", "g", (4, 5)) in gens


def test_identical_text_but_replaced_is_not_original():
    # Replace a fragment with the exact same string: body is unchanged but
    # provenance must still say "generated"; a package built on the other
    # version must reject the bound edit list.
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "hello world")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "same", "doc_id": "d", "start": 0, "end": 5,
              "replacement": "hello"}], target=tmp + "/pub")
        with ReleasePackage(p) as pkg:
            assert pkg.document_text("d") == "hello world"
            tr = pkg.traceback("d", 0, 5)
            assert all(x["source"] == "generated" for x in tr["pieces"])
            assert tr["pieces"][0]["generation"]["edit_id"] == "same"
        # An edit list bound to the index identity is rejected by the
        # package even though the visible text is identical.
        bad = os.path.join(tmp, "bad.jsonl")
        write_edit_list(bad, root, "rX",
                        [{"edit_id": "x", "doc_id": "d", "start": 0,
                          "end": 1, "replacement": "z"}])
        with pytest.raises(PublishRejected):
            ReleaseBuilder(p, bad, tmp + "/badpub").build()
        assert not os.path.exists(tmp + "/badpub")
        assert not os.path.exists(tmp + "/badpub.staging")


# --------------------------------------------------------------------- #
# Structural edge cases
# --------------------------------------------------------------------- #
def test_adjacent_edits_and_unchanged_and_empty_docs():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [
            ("a", "abcdef"), ("b", "keep"), ("c", ""), ("d", "nu\0l")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "a1", "doc_id": "a", "start": 0, "end": 2,
              "replacement": "X"},
             {"edit_id": "a2", "doc_id": "a", "start": 2, "end": 4,
              "replacement": ""}], target=tmp + "/pub")
        with ReleasePackage(p) as pkg:
            assert pkg.document_text("a") == "Xef"
            assert pkg.document_text("b") == "keep"
            assert pkg.document_text("c") == ""
            assert pkg.document_text("d") == "nu\0l"
            report = pkg.validate()
            assert report["documents"] == 4
            # NUL and newline round-trip through the streaming JSONL export.
            buf = io.BytesIO()
            pkg.export_jsonl(buf)
            rows = [json.loads(line) for line in
                    buf.getvalue().decode("utf-8").splitlines()]
            got = {r["doc_id"]: r["text"] for r in rows}
            assert got["d"] == "nu\0l"
            assert got["c"] == ""


def test_whole_document_deletion_is_retained_as_empty_doc():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("gone", "secret"), ("stay", "ok")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "del", "doc_id": "gone", "start": 0, "end": 6,
              "replacement": ""}], target=tmp + "/pub")
        with ReleasePackage(p) as pkg:
            assert pkg.document_text("gone") == ""
            assert pkg.document_text("stay") == "ok"
            assert pkg.require_doc("gone") is not None
            # The whole baseline interval is reported lost, not mapped.
            r = pkg.forward_map("gone", 0, 6)
            assert r["mapped"] == []
            assert [lo["baseline"]["codepoints"]
                    for lo in r["lost"]] == [[0, 6]]
            pkg.validate()


def test_multibyte_char_straddling_read_blocks():
    # Force the block size below one multi-byte character's encoding.
    with tempfile.TemporaryDirectory() as tmp:
        text = "x" * 5 + "😀" * 20 + "y" * 5
        idx = build_index(tmp, [("d", text)])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "e", "doc_id": "d", "start": 10, "end": 12,
              "replacement": "Z"}], target=tmp + "/pub",
            chunk_items=4, block_bytes=3, write_bytes=3)
        with ReleasePackage(p) as pkg:
            assert pkg.document_text("d") == text[:10] + "Z" + text[12:]
            pkg.validate()


# --------------------------------------------------------------------- #
# Whole-round rejection conditions
# --------------------------------------------------------------------- #
def _expect_reject(tmp, idx_or_pub, identity, round_id, records,
                   source=None):
    el = os.path.join(tmp, "reject.jsonl")
    write_edit_list(el, identity, round_id, records)
    target = os.path.join(tmp, "reject_out")
    with pytest.raises(PublishRejected):
        ReleaseBuilder(source or idx_or_pub, el, target,
                       chunk_items=4).build()
    assert not os.path.exists(target)
    assert not os.path.exists(target + ".staging")


def test_reject_overlap_duplicate_unknown_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdefgh")])
        root = index_identity(idx)
        _expect_reject(tmp, idx, root, "r", [
            {"edit_id": "e1", "doc_id": "d", "start": 0, "end": 3,
             "replacement": "x"},
            {"edit_id": "e2", "doc_id": "d", "start": 2, "end": 4,
             "replacement": "y"}])
        _expect_reject(tmp, idx, root, "r", [
            {"edit_id": "dup", "doc_id": "d", "start": 0, "end": 1,
             "replacement": "x"},
            {"edit_id": "dup", "doc_id": "d", "start": 2, "end": 3,
             "replacement": "y"}])
        _expect_reject(tmp, idx, root, "r", [
            {"edit_id": "e", "doc_id": "ghost", "start": 0, "end": 1,
             "replacement": "x"}])
        _expect_reject(tmp, idx, root, "r", [
            {"edit_id": "e", "doc_id": "d", "start": 0, "end": 99,
             "replacement": "x"}])
        _expect_reject(tmp, idx, root, "r", [
            {"edit_id": "e", "doc_id": "d", "start": 3, "end": 3,
             "replacement": "x"}])  # empty interval


def test_reject_identity_mismatch_and_reused_round():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abc")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r1", root, "r1",
            [{"edit_id": "e", "doc_id": "d", "start": 0, "end": 1,
              "replacement": "x"}], target=tmp + "/pub")
        # Wrong expected identity (points at the root, but input is the pub).
        _expect_reject(tmp, p, root, "r2",
                       [{"edit_id": "e", "doc_id": "d", "start": 0,
                         "end": 1, "replacement": "y"}], source=p)
        # Reusing a chain round id.
        _expect_reject(tmp, p, m["package_identity"], "r1",
                       [{"edit_id": "e2", "doc_id": "d", "start": 0,
                         "end": 1, "replacement": "y"}], source=p)


def test_diagnostics_never_echo_consumed_or_replacement_text():
    import io as _io
    from diskfm.pubformat import PublishRejected as _PR
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "TOPSECRET-tail")])
        root = index_identity(idx)
        secret = "TOPSECRET"
        marker = "UNIQUEMARKER"
        el = os.path.join(tmp, "r.jsonl")
        # Two defects at once: overlapping intervals carrying distinctive
        # replacement values that must never appear in the rejection text.
        with open(el, "w", encoding="utf-8") as f:
            f.write(json.dumps(
                {"expected_input_identity": root,
                 "round_id": "r1"}) + "\n")
            f.write(json.dumps(
                {"edit_id": "a", "doc_id": "d", "start": 0, "end": 3,
                 "replacement": marker}) + "\n")
            f.write(json.dumps(
                {"edit_id": "b", "doc_id": "d", "start": 2, "end": 4,
                 "replacement": marker + "X"}) + "\n")
        target = os.path.join(tmp, "out")
        try:
            ReleaseBuilder(idx, el, target).build()
            raise AssertionError("overlapping round should have been rejected")
        except _PR as exc:
            message = str(exc)
            assert marker not in message
            assert secret not in message
        assert not os.path.exists(target)
        assert not os.path.exists(target + ".staging")


def test_invalid_utf8_edit_list_is_rejected_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abc")])
        root = index_identity(idx)
        el = os.path.join(tmp, "broken.jsonl")
        with open(el, "wb") as f:
            f.write(json.dumps(
                {"expected_input_identity": root,
                 "round_id": "r1"}).encode("utf-8") + b"\n")
            f.write(b'{"edit_id":"e","doc_id":"d","start":0,"end":1,'
                    b'"replacement":"\xff\xfe bad utf8"}\n')
        target = os.path.join(tmp, "out")
        with pytest.raises(PublishRejected):
            ReleaseBuilder(idx, el, target).build()
        assert not os.path.exists(target)
        assert not os.path.exists(target + ".staging")


def test_nonempty_interval_never_yields_empty_result():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdef")])
        root = index_identity(idx)
        # Delete the whole document, then every forward query over its
        # baseline range must report lost fragments rather than a fake,
        # empty "valid" mapping.
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "del", "doc_id": "d", "start": 0, "end": 6,
              "replacement": ""}], target=tmp + "/pub")
        with ReleasePackage(p) as pkg:
            for start, end in ((0, 6), (1, 3), (5, 6)):
                r = pkg.forward_map("d", start, end)
                assert r["mapped"] == []
                assert r["lost"], "lost fragments must not be silent"


def test_open_does_not_read_package_contents():
    # A normal open plus a tiny local query must read only the small
    # directory and the touched document's lineage/width pages -- never the
    # released body or the lineage of every document.
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [
            ("big", "q" * 5000), ("small", "ab")], chunk_items=16)
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "e", "doc_id": "big", "start": 10, "end": 12,
              "replacement": "X"}], target=tmp + "/pub")
        pkg = ReleasePackage(p)  # no full hash pass on open
        try:
            body_size = m["files"]["texts.bin"]["bytes"]
            pkg.forward_map("big", 20, 24)
            stats = pkg.runtime_stats()
            # The whole body is far larger than anything a 4-codepoint
            # query may touch.
            assert stats["text_bytes_read"] == 0
            assert stats["lineage_records_read"] < 20
            # Other documents' lineage was never read.
            assert stats["documents_touched"] == 1
        finally:
            pkg.close()
        assert body_size > 1000


def test_size_corruption_caught_on_open_content_corruption_on_validate():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdefgh")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "e", "doc_id": "d", "start": 1, "end": 3,
              "replacement": "XY"}], target=tmp + "/pub")
        # Content tamper that preserves the file length: open succeeds
        # (no content read), but the deep validate must reject it.
        lin = os.path.join(p, "lineage.bin")
        with open(lin, "r+b") as f:
            f.seek(pf_header_end())
            b = f.read(1)
            f.seek(pf_header_end())
            f.write(bytes([b[0] ^ 0xFF]))
        ReleasePackage(p).close()  # opening does not hash contents
        with pytest.raises(ReleaseCorruptError):
            ReleasePackage(p).validate()
        # Restore, then truncate: now even opening must fail on length.
        with open(lin, "r+b") as f:
            f.seek(pf_header_end())
            f.write(b)
        with open(lin, "r+b") as f:
            f.truncate(os.path.getsize(lin) - 4)
        with pytest.raises(ReleaseCorruptError):
            ReleasePackage(p)


def test_explicit_caps_reject_oversized_records_and_identifiers():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdefghij")])
        root = index_identity(idx)
        # Identifier cap.
        el = os.path.join(tmp, "idcap.jsonl")
        write_edit_list(el, root, "r1", [{
            "edit_id": "x" * 100, "doc_id": "d",
            "start": 0, "end": 1, "replacement": "q"}])
        with pytest.raises(PublishRejected):
            ReleaseBuilder(idx, el, tmp + "/o1",
                           max_id_bytes=16).build()
        assert not os.path.exists(tmp + "/o1")
        # Record-size cap (replacement longer than the per-record budget).
        el2 = os.path.join(tmp, "reccap.jsonl")
        write_edit_list(el2, root, "r2", [{
            "edit_id": "e", "doc_id": "d",
            "start": 0, "end": 1, "replacement": "z" * 5000}])
        with pytest.raises(PublishRejected):
            ReleaseBuilder(idx, el2, tmp + "/o2",
                           max_edit_bytes=256).build()
        assert not os.path.exists(tmp + "/o2")
        # Edit-count cap.
        el3 = os.path.join(tmp, "countcap.jsonl")
        write_edit_list(el3, root, "r3", [
            {"edit_id": f"e{i}", "doc_id": "d",
             "start": i, "end": i + 1, "replacement": ""}
            for i in range(4)])
        with pytest.raises(PublishRejected):
            ReleaseBuilder(idx, el3, tmp + "/o3", max_edits=2).build()
        assert not os.path.exists(tmp + "/o3")


def test_forward_lost_when_endpoints_sit_inside_deleted_runs():
    # Request endpoints fall strictly inside removed baseline regions; the
    # lost pieces must still be reported with baseline coordinates only.
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "0123456789")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "del", "doc_id": "d", "start": 2, "end": 8,
              "replacement": ""}], target=tmp + "/pub")
        with ReleasePackage(p) as pkg:
            # [3,7) is wholly inside the removed [2,8): entirely lost.
            r = pkg.forward_map("d", 3, 7)
            assert r["mapped"] == []
            assert [lo["baseline"]["codepoints"]
                    for lo in r["lost"]] == [[3, 7]]
            # A window straddling a deleted boundary maps the live head and
            # reports the in-deletion tail.
            r2 = pkg.forward_map("d", 1, 4)
            assert [m2["baseline"]["codepoints"]
                    for m2 in r2["mapped"]] == [[1, 2]]
            assert [lo["baseline"]["codepoints"]
                    for lo in r2["lost"]] == [[2, 4]]


def test_reject_malformed_records_and_lone_surrogate():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abc")])
        root = index_identity(idx)
        el = os.path.join(tmp, "bad.jsonl")
        with open(el, "w", encoding="utf-8") as f:
            f.write(json.dumps(
                {"expected_input_identity": root,
                 "round_id": "r"}) + "\n")
            f.write(json.dumps({"edit_id": "e", "doc_id": "d",
                                "start": 0, "end": 1}) + "\n")  # missing field
        with pytest.raises(PublishRejected):
            ReleaseBuilder(idx, el, tmp + "/out").build()
        # Lone surrogate inside a replacement.
        el2 = os.path.join(tmp, "sur.jsonl")
        with open(el2, "w", encoding="utf-8", errors="surrogatepass") as f:
            header = json.dumps(
                {"expected_input_identity": root, "round_id": "r2"})
            f.write(header + "\n")
            rec = {"edit_id": "e", "doc_id": "d", "start": 0, "end": 1,
                   "replacement": "\ud800"}
            f.write(json.dumps(rec, ensure_ascii=False)
                    .encode("utf-8", errors="surrogatepass").decode(
                        "utf-8", errors="surrogatepass") + "\n")
        with pytest.raises(PublishRejected):
            ReleaseBuilder(idx, el2, tmp + "/out2").build()


# --------------------------------------------------------------------- #
# Determinism and self-containment
# --------------------------------------------------------------------- #
def test_shuffle_and_chunk_size_preserve_identity():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdefghij"), ("e", "klmnop")])
        root = index_identity(idx)
        records = [
            {"edit_id": "z", "doc_id": "d", "start": 1, "end": 3,
             "replacement": "QQ"},
            {"edit_id": "a", "doc_id": "e", "start": 0, "end": 2,
             "replacement": "RR"},
            {"edit_id": "m", "doc_id": "d", "start": 5, "end": 6,
             "replacement": ""},
        ]
        m1, p1 = publish(
            idx, tmp + "/rA", root, "r1", records,
            target=tmp + "/pubA", chunk_items=2, chunk_bytes=256)
        shuffled = list(records)
        random.Random(7).shuffle(shuffled)
        m2, p2 = publish(
            idx, tmp + "/rB", root, "r1", shuffled,
            target=tmp + "/pubB", chunk_items=50, chunk_bytes=1 << 20)
        assert m1["package_identity"] == m2["package_identity"]
        assert m1["edit_digest"] == m2["edit_digest"]
        with ReleasePackage(p1) as a, ReleasePackage(p2) as b:
            for doc in ("d", "e"):
                assert a.document_text(doc) == b.document_text(doc)
                assert (a.forward_map(doc, 0,
                                      a._doc_record(
                                          a.require_doc(doc))["orig_cp_len"])
                        == b.forward_map(
                            doc, 0,
                            b._doc_record(
                                b.require_doc(doc))["orig_cp_len"]))


def test_package_is_self_contained_after_ancestors_removed():
    with tempfile.TemporaryDirectory() as tmp:
        home = os.path.join(tmp, "home")
        os.makedirs(home)
        idx = build_index(home, [("d", "A😀秘密BC"), ("u", "plain text")])
        root = index_identity(idx)
        m1, p1 = publish(
            idx, home + "/r1", root, "r1",
            [{"edit_id": "e1", "doc_id": "d", "start": 2, "end": 4,
              "replacement": "•"}], target=home + "/pub1")
        m2, p2 = publish(
            p1, home + "/r2", m1["package_identity"], "r2",
            [{"edit_id": "e2", "doc_id": "d", "start": 0, "end": 1,
              "replacement": "前置"}], target=home + "/pub2")
        portable = os.path.join(tmp, "moved_pub2")
        shutil.move(p2, portable)
        shutil.rmtree(home)  # index, intermediate package and lists gone
        with ReleasePackage(portable) as pkg:
            assert pkg.document_text("d") == "前置😀•BC"
            pkg.forward_map("d", 0, 6)
            pkg.traceback("d", 0, pkg._doc_record(
                pkg.require_doc("d"))["pub_cp_len"])
            pkg.validate()
            buf = io.BytesIO()
            pkg.export_jsonl(buf)
            assert b"plain text" in buf.getvalue()
        # No removed-text or ancestor artifacts leaked into the package.
        names = set()
        for root_dir, _dirs, files in os.walk(portable):
            names.update(files)
        assert "content.bin" not in names
        assert not any(n.startswith("vsort_") for n in names)


# --------------------------------------------------------------------- #
# Tamper and interruption
# --------------------------------------------------------------------- #
def test_tampered_lineage_and_missing_files_are_detected():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdefgh")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r", root, "r1",
            [{"edit_id": "e", "doc_id": "d", "start": 1, "end": 3,
              "replacement": "XY"}], target=tmp + "/pub")
        # Corrupt a lineage byte.
        lin = os.path.join(p, "lineage.bin")
        with open(lin, "r+b") as f:
            f.seek(pf_header_end())
            b = f.read(1)
            f.seek(pf_header_end())
            f.write(bytes([b[0] ^ 0xFF]))
        with pytest.raises(ReleaseCorruptError):
            ReleasePackage(p).validate()
        # Remove a file and reopen: hash verification must fail.
        os.remove(lin)
        with pytest.raises(ReleaseCorruptError):
            ReleasePackage(p)


def pf_header_end():
    from diskfm import pubformat as pfmt
    return pfmt.HEADER_SIZE + pfmt.SHARD_HEADER_SIZE


def test_interrupted_build_leaves_no_openable_target():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abcdefgh")])
        root = index_identity(idx)
        el = os.path.join(tmp, "r.jsonl")
        write_edit_list(el, root, "r1",
                        [{"edit_id": "e", "doc_id": "d", "start": 0,
                          "end": 2, "replacement": "zz"}])
        target = os.path.join(tmp, "pub")
        builder = ReleaseBuilder(idx, el, target)
        # Simulate a crash mid-build: staging exists but has no OPEN marker.
        try:
            builder._read_header()
            id_res, iv_res, _digest, _n, _cache = \
                builder._ingest_edit_list("r1")
            assert os.path.isdir(target + ".staging")
            assert not os.path.exists(target)
        finally:
            builder.source.close()
            shutil.rmtree(target + ".staging", ignore_errors=True)
        assert not os.path.exists(target)
        # A normal build afterwards still succeeds.
        ReleaseBuilder(idx, el, target).build()
        assert ReleasePackage(target).document_text("d") == "zzcdefgh"


def test_existing_target_is_never_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        idx = build_index(tmp, [("d", "abc")])
        root = index_identity(idx)
        m, p = publish(
            idx, tmp + "/r1", root, "r1",
            [{"edit_id": "e", "doc_id": "d", "start": 0, "end": 1,
              "replacement": "x"}], target=tmp + "/pub")
        with pytest.raises(FileExistsError):
            ReleaseBuilder(idx, tmp + "/r1", p)
        # Existing package untouched and still valid.
        ReleasePackage(p).validate()


# --------------------------------------------------------------------- #
# Differential fuzz against the independent oracle
# --------------------------------------------------------------------- #
def test_random_rounds_match_oracle():
    rng = random.Random(1234)
    with tempfile.TemporaryDirectory() as tmp:
        alphabet = "ab😀c\nd\0e秘"
        baseline = "".join(rng.choice(alphabet) for _ in range(60))
        idx = build_index(tmp, [("d", baseline)], chunk_items=8)
        source = idx
        identity = index_identity(idx)
        oracle_rounds = []
        current = baseline
        package_dir = None
        for rnd in range(4):
            n = len(current)
            edits = []
            positions = sorted(rng.sample(range(n), min(6, n)))
            used = set()
            for k, pos in enumerate(positions):
                if pos in used:
                    continue
                span = rng.randint(1, min(4, n - pos))
                if any(pos < u < pos + span for u in used):
                    continue
                repl = "".join(rng.choice(alphabet)
                               for _ in range(rng.randint(0, 4)))
                edits.append({
                    "edit_id": f"e{rnd}_{k}", "doc_id": "d",
                    "start": pos, "end": pos + span,
                    "replacement": repl})
                used.update(range(pos, pos + span))
            if not edits:
                continue
            oracle_rounds.append(
                {"round_id": f"r{rnd}", "edits": edits})
            target = os.path.join(tmp, f"pub{rnd}")
            m, package_dir = publish(
                source, os.path.join(tmp, f"el{rnd}"), identity,
                f"r{rnd}", edits, target=target, chunk_items=8)
            source, identity = package_dir, m["package_identity"]
            # Recompute the expected current text via raw string edits.
            current = _apply_to_string(current, edits)
        with ReleasePackage(package_dir) as pkg:
            assert pkg.document_text("d") == current
            odoc = apply_rounds(baseline, oracle_rounds)
            assert odoc.text == current
            n = len(current)
            for start, end in ((0, n), (1, max(2, n - 1)),
                               (n // 2, n // 2 + 1)):
                if start >= end:
                    continue
                # Per-codepoint provenance is segmentation-independent.
                got = pkg.traceback("d", start, end)
                got_flat = _flatten_trace(got)
                want_flat = provenance_at(odoc, start, end)
                assert got_flat == want_flat
                gf = pkg.forward_map("d", 0, len(baseline))
                wf = forward_map(baseline, odoc, 0, len(baseline))
                assert _sorted_mapped(gf) == _sorted_mapped(wf)
                assert _sorted_lost(gf) == _sorted_lost(wf)
            pkg.validate()


def _flatten_trace(result):
    """Expand production trace pieces into one provenance tuple per codepoint."""
    flat = []
    for p in result["pieces"]:
        lo, hi = p["release"]["codepoints"]
        for i in range(lo, hi):
            if p["source"] == "baseline":
                shift = i - lo
                flat.append(("orig",
                             p["baseline"]["codepoints"][0] + shift))
            else:
                shift = i - lo
                flat.append((
                    "gen",
                    p["generation"]["round_id"],
                    p["generation"]["edit_id"],
                    p["generation"]["relative"]["codepoints"][0] + shift))
    return flat


def _apply_to_string(text, edits):
    out = []
    cursor = 0
    for e in sorted(edits, key=lambda x: x["start"]):
        out.append(text[cursor:e["start"]])
        out.append(e["replacement"])
        cursor = e["end"]
    out.append(text[cursor:])
    return "".join(out)


def _normalize_trace(result):
    return sorted(
        (p["source"],
         tuple(p["release"]["codepoints"]),
         tuple(p.get("baseline", {}).get("codepoints", [])),
         (p["generation"]["round_id"], p["generation"]["edit_id"],
          tuple(p["generation"]["relative"]["codepoints"]))
         if p["source"] == "generated" else None)
        for p in result["pieces"])


def _normalize_trace_w(pieces):
    return sorted(
        (p["source"], tuple(p["release"]["codepoints"]),
         tuple(p.get("baseline", {}).get("codepoints", [])),
         (p["generation"]["round_id"], p["generation"]["edit_id"],
          tuple(p["generation"]["relative"]["codepoints"]))
         if p["source"] == "generated" else None)
        for p in pieces)


def _sorted_mapped(r):
    return sorted(
        (tuple(m["baseline"]["codepoints"]),
         tuple(m["release"]["codepoints"]),
         tuple(m["baseline"]["utf8_bytes"]),
         tuple(m["release"]["utf8_bytes"])) for m in r["mapped"])


def _sorted_lost(r):
    return sorted(tuple(lo["baseline"]["codepoints"]) for lo in r["lost"])

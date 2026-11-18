"""Large-corpus, bounded-memory and CLI tests for the similarity catalog.

The corpus is generated at runtime (never committed): a hot repeated
template shared by most documents plus sparse random tails, so popular
shingles have long posting lists and the candidate counter must really
merge externally.  Build and query run in fresh subprocesses with tiny
sort buffers; peak RSS (VmHWM) and real I/O counters are recorded as
evidence that neither phase scales its memory with the corpus, and that
querying never reads the original text back (the catalog contains none).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_oracle import expected_posting_records, expected_query  # noqa: E402
from fractions import Fraction  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "scripts", "sim_mem_worker.py")


def _write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(json.dumps({"doc_id": doc_id, "text": text},
                               ensure_ascii=False) + "\n")


def _make_docs(n_docs, seed=11):
    """Hot template docs (near-duplicates) plus sparse random documents."""
    rng = random.Random(seed)
    # Non-repetitive 60-codepoint template: ~55 distinct 6-gram shingles.
    template = "".join(
        rng.choice("甲乙丙丁戊己庚辛壬癸合同条款模板权利义务契约")
        for _ in range(60))
    docs = []
    for i in range(n_docs):
        if i % 4 != 3:
            # Near-duplicate: shared template + small unique tail.
            tail = "".join(rng.choice("0123456789\n")
                           for _ in range(rng.randint(4, 12)))
            docs.append((f"contract{i:04d}", template + tail + str(i)))
        else:
            # Sparse: mostly disjoint random content.
            docs.append((f"contract{i:04d}", "".join(
                rng.choice("稀疏文档字符集😀\x00\n")
                for _ in range(rng.randint(20, 60)))))
    return docs


def _run(*args, check=True):
    return subprocess.run([sys.executable, "-I", *args],
                          check=check, capture_output=True, text=True)


def _build_index_and_catalog(tmp, tag, docs, q, chunk_items, chunk_bytes):
    jp = os.path.join(tmp, f"{tag}.jsonl")
    _write_jsonl(jp, docs)
    idx_dir = os.path.join(tmp, f"{tag}_idx")
    cat_dir = os.path.join(tmp, f"{tag}_cat")
    env_worker = os.path.join(ROOT, "scripts", "mem_worker.py")
    _run(env_worker, "build", jp, idx_dir, "--chunk-items", "2000")
    out = _run(WORKER, "build", idx_dir, cat_dir, "--q", str(q),
               "--chunk-items", str(chunk_items),
               "--chunk-bytes", str(chunk_bytes))
    return idx_dir, cat_dir, json.loads(out.stdout)


def test_large_catalog_bounded_memory_and_external_merge():
    q = 6
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, n_docs in (("small", 60), ("large", 240)):
            docs = _make_docs(n_docs)
            idx_dir, cat_dir, build_info = _build_index_and_catalog(
                tmp, tag, docs, q, chunk_items=800, chunk_bytes=16384)
            bs = build_info["build_stats"]
            # Real external merge: many runs, real temp bytes, and the
            # relation counts are honest (raw >= deduped > 0).
            assert bs["external_sort_runs"] > 4
            assert bs["temp_bytes"] > 0
            assert bs["raw_relations"] >= bs["deduped_relations"] > 0
            assert bs["peak_rss_kb"] > 0

            # Query a hot template document in a fresh process.
            out = _run(WORKER, "query", cat_dir, "contract0000",
                       "--threshold", "0.5", "--counter-chunk-items", "500",
                       "--result-chunk-items", "300",
                       "--result-chunk-bytes", "8192")
            query_info = json.loads(out.stdout)
            qs = query_info["query_stats"]
            # Every template document is found: no candidate cap.
            expected = expected_query(docs, q, "contract0000",
                                      Fraction(1, 2))
            assert query_info["results"] == len(expected)
            assert len(expected) >= n_docs // 2  # hot template majority
            assert query_info["first_rows"] == expected[:3]
            # The query streamed exactly the postings of the source
            # document's shingles -- nothing else.
            assert qs["posting_records_read"] == expected_posting_records(
                docs, q, "contract0000")
            assert qs["posting_lists_read"] == qs["source_shingle_count"]
            # Hot shingles forced the counter to spill and merge.
            assert qs["counter_chunks_spilled"] > 1
            # The catalog directory contains no original text at all.
            assert not os.path.exists(os.path.join(cat_dir, "content.bin"))
            runs.append((tag, docs, build_info, query_info))

        (tag_s, docs_s, b_small, q_small), (tag_l, docs_l, b_large, q_large) \
            = runs
        rel_ratio = (b_large["relations"]["deduped"]
                     / b_small["relations"]["deduped"])
        build_rss_ratio = b_large["peak_rss_kb"] / b_small["peak_rss_kb"]
        query_rss_ratio = q_large["peak_rss_kb"] / q_small["peak_rss_kb"]
        evidence = {
            "relation_ratio": rel_ratio,
            "build_peak_rss_small_kb": b_small["peak_rss_kb"],
            "build_peak_rss_large_kb": b_large["peak_rss_kb"],
            "build_rss_ratio": build_rss_ratio,
            "query_peak_rss_small_kb": q_small["peak_rss_kb"],
            "query_peak_rss_large_kb": q_large["peak_rss_kb"],
            "query_rss_ratio": query_rss_ratio,
            "large_build_stats": b_large["build_stats"],
            "large_query_stats": q_large["query_stats"],
            "large_query_pagers": q_large["pagers"],
            "large_file_bytes": b_large["file_bytes"],
        }
        evidence_path = os.path.join(tmp, "sim_memory_evidence.json")
        with open(evidence_path, "w") as f:
            json.dump(evidence, f, indent=2, sort_keys=True)
        print("\nSIM MEMORY EVIDENCE:", json.dumps(evidence, sort_keys=True))
        # Relations scale with the corpus (4x docs) ...
        assert 3.0 < rel_ratio < 5.0
        # ... while peak RSS of both phases stays essentially flat.
        assert build_rss_ratio < 1.5
        assert query_rss_ratio < 1.35
        # Query-side pager residency is capped by the LRU geometry, not by
        # the catalog size.
        for pager, rec_size in (("qgramdir", 32), ("postings", 8),
                                ("docs", 40), ("docqgrams", 4)):
            for info in (q_small, q_large):
                st = info["pagers"][pager]
                assert st["resident_pages"] <= 8
                assert st["resident_bytes"] <= 8 * 1024 * rec_size


def test_sim_cli_end_to_end():
    docs = _make_docs(30)
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "in.jsonl")
        _write_jsonl(jp, docs)
        idx_dir = os.path.join(tmp, "idx")
        cat_dir = os.path.join(tmp, "cat")
        env = dict(os.environ, PYTHONPATH=ROOT)

        def run(*args, check=True):
            return subprocess.run(
                [sys.executable, "-m", "diskfm.simcli", *args],
                cwd=ROOT, env=env, check=check, capture_output=True,
                text=True)

        # Build the source index first (existing product CLI).
        subprocess.run([sys.executable, "-m", "diskfm", "build", jp,
                        idx_dir, "--chunk-items", "2000"],
                       cwd=ROOT, env=env, check=True, capture_output=True)
        build = run("build", idx_dir, cat_dir, "--q", "6",
                    "--chunk-items", "500", "--chunk-bytes", "8192")
        manifest = json.loads(build.stdout)
        assert manifest["parameters"]["q"] == 6
        assert manifest["documents"]["count"] == 30
        assert manifest["source"]["index_identity"]

        q = run("query", cat_dir, "contract0000", "--threshold", "0.5",
                "--page-size", "5")
        rows = [json.loads(line) for line in q.stdout.splitlines()]
        assert len(rows) == 5
        assert "next_cursor=" in q.stderr
        cursor = q.stderr.strip().split("next_cursor=")[1]
        q2 = run("query", cat_dir, "contract0000", "--threshold", "0.5",
                 "--page-size", "5", "--cursor", cursor)
        rows2 = [json.loads(line) for line in q2.stdout.splitlines()]
        expected = expected_query(docs, 6, "contract0000", Fraction(1, 2))
        assert rows + rows2 == expected[:10]
        # --all streams the full ordered set in one go.
        qa = run("query", cat_dir, "contract0000", "--threshold", "0.5",
                 "--all")
        assert [json.loads(l) for l in qa.stdout.splitlines()] == expected

        e = run("explain", cat_dir, "contract0000", "contract0004")
        shared = [json.loads(line) for line in e.stdout.splitlines()]
        assert shared and all("qgram_utf8_hex" in s for s in shared)

        d = run("docs", cat_dir, "--all")
        doc_rows = [json.loads(line) for line in d.stdout.splitlines()]
        assert len(doc_rows) == 30
        assert all(r["comparable"] for r in doc_rows)

        v = run("validate", cat_dir)
        assert json.loads(v.stdout)["ok"] is True

        s = run("stats", cat_dir)
        stats = json.loads(s.stdout)
        assert stats["manifest"]["relations"]["deduped"] > 0

        # A corrupted catalog fails loudly on every read path.
        with open(os.path.join(cat_dir, "qgrams.bin"), "r+b") as f:
            f.truncate(6)
        bad = run("query", cat_dir, "contract0000", "--threshold", "0.5",
                  check=False)
        assert bad.returncode != 0
        assert "error" in bad.stderr.lower()
        badv = run("validate", cat_dir, check=False)
        assert badv.returncode != 0
        assert "CORRUPT" in badv.stderr

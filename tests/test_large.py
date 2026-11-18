"""Large-corpus, bounded-memory and CLI tests for the disk FM-index.

These are kept in a separate module so the fast differential suite stays
fast.  The corpus here is generated at runtime; construction runs with a
deliberately tiny in-memory sort buffer so every doubling round spills to
disk, and peak RSS of both build and query is measured from
``/proc/self/status`` (Linux) to demonstrate it does not scale with the
corpus.
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

from oracle import brute_count  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_big_jsonl(path, n_docs, doc_len, seed):
    rng = random.Random(seed)
    docs = []
    with open(path, "w", encoding="utf-8") as f:
        for d in range(n_docs):
            # Mix repetitive runs (stress prefix doubling) with randomness.
            if d % 3 == 0:
                text = ("ab" * (doc_len // 2))[:doc_len]
            else:
                text = "".join(rng.choice("abcd😀\n\x00")
                               for _ in range(doc_len))
            docs.append((f"doc{d:05d}", text))
            f.write(json.dumps({"doc_id": f"doc{d:05d}", "text": text},
                               ensure_ascii=False) + "\n")
    return docs


def test_large_corpus_bounded_memory():
    # Two corpus sizes: if RAM were proportional to the text, the 4x larger
    # build/query would need ~4x RSS.  External construction keeps peak RSS
    # nearly flat while on-disk artifacts grow linearly.  Both phases run in
    # fresh subprocesses whose VmHWM measures one clean phase each.
    worker = os.path.join(ROOT, "scripts", "mem_worker.py")
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, (n_docs, doc_len) in (("small", (60, 400)),
                                       ("large", (240, 400))):
            jp = os.path.join(tmp, f"{tag}.jsonl")
            docs = _write_big_jsonl(jp, n_docs, doc_len, seed=42)
            idx_dir = os.path.join(tmp, f"{tag}_idx")

            build_out = subprocess.run(
                [sys.executable, "-I", worker, "build", jp, idx_dir,
                 "--chunk-items", "2000"],
                check=True, capture_output=True, text=True)
            build_info = json.loads(build_out.stdout)
            assert build_info["build_stats"]["chunks_spilled"] > 5
            assert build_info["build_stats"]["external_sorts"] >= 4

            query_out = subprocess.run(
                [sys.executable, "-I", worker, "query", idx_dir, "ab",
                 "--page-size", "25"],
                check=True, capture_output=True, text=True)
            query_info = json.loads(query_out.stdout)
            assert query_info["count"] == brute_count(docs, "ab")
            cs = query_info["count_stats"]
            # count alone: no enumeration, no LF, no SA samples, no reads of
            # the original content.
            assert cs["lf_steps"] == 0
            assert cs["sample_rows_read"] == 0
            assert cs["content_bytes_read"] == 0
            assert cs["rank_calls"] > 0
            # One locate page recovered only its own 25 coordinates.
            qs = query_info["query_stats"]
            assert query_info["page_hits"] == 25
            assert qs["sample_rows_read"] <= 25 * (
                (query_info["n_symbols"] // 16).bit_length() + 1)
            # LF work for a 25-hit page is bounded by page size times the
            # sample interval, independent of the corpus length.
            assert qs["lf_steps"] <= 25 * 16
            runs.append((tag, build_info, query_info))

        (_, b_small, q_small), (_, b_large, q_large) = runs
        size_ratio = (b_large["n_symbols"] / b_small["n_symbols"])
        text_ratio = (b_large["file_bytes"]["text.bin"]
                      / b_small["file_bytes"]["text.bin"])
        build_rss_ratio = b_large["peak_rss_kb"] / b_small["peak_rss_kb"]
        query_rss_ratio = q_large["peak_rss_kb"] / q_small["peak_rss_kb"]
        evidence = {
            "size_ratio": size_ratio,
            "on_disk_text_ratio": text_ratio,
            "build_peak_rss_small_kb": b_small["peak_rss_kb"],
            "build_peak_rss_large_kb": b_large["peak_rss_kb"],
            "build_rss_ratio": build_rss_ratio,
            "query_peak_rss_small_kb": q_small["peak_rss_kb"],
            "query_peak_rss_large_kb": q_large["peak_rss_kb"],
            "query_rss_ratio": query_rss_ratio,
            "large_query_stats": q_large["query_stats"],
            "large_build_stats": b_large["build_stats"],
        }
        with open(os.path.join(tmp, "memory_evidence.json"), "w") as f:
            json.dump(evidence, f, indent=2, sort_keys=True)
        print("\nMEMORY EVIDENCE:", json.dumps(evidence, sort_keys=True))
        # Disk footprint scales linearly with the corpus ...
        assert 3.5 < text_ratio < 4.5
        # ... while peak RSS of either phase stays essentially flat:
        # query RSS within 25%, build RSS within 50% for a 4x corpus.
        assert query_rss_ratio < 1.25
        assert build_rss_ratio < 1.5
        # Structural proof: every query-side pager's resident footprint is
        # bounded by the same fixed LRU cap at both corpus sizes; at the
        # large size every pager actually sits at its cap, yet that cap is
        # independent of N (the small index simply has fewer pages to fill).
        caps = {"rank_pager": 8 * 1024 * 4,
                "sample_pager": 8 * 1024 * 12,
                "bwt_pager": 8 * 1024 * 4}
        for pager, cap in caps.items():
            large_res = q_large["query_stats"][pager]["resident_bytes"]
            small_res = q_small["query_stats"][pager]["resident_bytes"]
            assert large_res <= cap and small_res <= cap
            assert (q_large["query_stats"][pager]["resident_pages"] <= 8
                    and q_small["query_stats"][pager]["resident_pages"] <= 8)
        # And the resident query data is tiny versus the on-disk structures
        # of the large index (the full SA alone would be N*4 bytes).
        resident_total = sum(
            q_large["query_stats"][p]["resident_bytes"]
            for p in ("rank_pager", "sample_pager", "bwt_pager",
                      "bytesample_pager"))
        assert resident_total < b_large["file_bytes"]["rank.bin"]


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "in.jsonl")
        with open(jp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": "a", "text": "ababa"}) + "\n")
            f.write(json.dumps({"doc_id": "b", "text": "ab"}) + "\n")
            f.write(json.dumps({"doc_id": "c", "text": ""}) + "\n")
        idx_dir = os.path.join(tmp, "idx")
        env = dict(os.environ, PYTHONPATH=ROOT)

        def run(*args):
            # No -I here: isolated mode drops both cwd and PYTHONPATH, so
            # the package would be unimportable.  The test driver itself
            # still runs under -I; inputs are temp files we generated.
            return subprocess.run(
                [sys.executable, "-m", "diskfm", *args],
                cwd=ROOT, env=env, check=True, capture_output=True, text=True)

        build = run("build", jp, idx_dir, "--chunk-items", "4",
                    "--sample-interval", "2")
        assert '"external_sorts"' in build.stdout
        assert run("count", idx_dir, "aba").stdout.strip() == "2"
        assert run("count", idx_dir, "nope").stdout.strip() == "0"
        loc = run("locate", idx_dir, "ab", "--page-size", "1", "--all")
        rows = [json.loads(line) for line in loc.stdout.splitlines()]
        assert len(rows) == 3
        assert {(r["doc_id"], r["codepoint_offset"]) for r in rows} == {
            ("a", 0), ("a", 2), ("b", 0)}
        assert run("verify", idx_dir).stdout.strip() == "ok"
        # `get` prints the original document verbatim, not JSON.
        assert run("get", idx_dir, "a").stdout == "ababa"

        # Truncation is a hard failure, not a silent undercount.
        with open(os.path.join(idx_dir, "rank.bin"), "r+b") as f:
            f.truncate(10)
        bad = subprocess.run(
            [sys.executable, "-m", "diskfm", "count", idx_dir, "ab"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        # Truncation is a hard failure (non-zero exit), never a silent
        # undercount of matches.
        assert bad.returncode != 0
        assert bad.stdout.strip() == ""
        assert "error" in bad.stderr.lower()

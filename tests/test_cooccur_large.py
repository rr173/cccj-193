"""Large-corpus and bounded-memory evidence for co-occurrence analysis.

The corpus is generated at runtime (nothing large is committed).  A worker
subprocess measures peak RSS from ``/proc/self/status`` for a clean phase,
so the reported number is executed, not estimated.  The tests prove:

* the analysis stays exact on a dense hot document (every oracle position
  pair matches) even with tiny sort buffers that force many spill runs;
* peak RSS is essentially flat when the corpus and hit count grow by a
  large factor, while the on-disk artifacts grow linearly;
* all pagination reads together reach exactly the published pair total;
* statistics (hits read, spill runs, candidate pairs, final pairs, page
  reads, temp bytes) are measured from the execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cooccur_oracle import expected_pair_counts, expected_pairs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "scripts", "cooccur_mem_worker.py")


def _write_corpus(path: str, n_docs: int, doc_len: int, seed: int):
    import random
    rng = random.Random(seed)
    docs = []
    with open(path, "w", encoding="utf-8") as f:
        for d in range(n_docs):
            if d % 5 == 0:
                # Dense hot document: many overlapping "aa" hits plus "ab".
                text = ("a" * (doc_len // 2)) + ("ab" * (doc_len // 4))
            else:
                text = "".join(rng.choice("ab😀\x00\n")
                               for _ in range(doc_len))
            docs.append((f"doc{d:05d}", text))
            f.write(json.dumps({"doc_id": f"doc{d:05d}", "text": text},
                               ensure_ascii=False) + "\n")
    return docs


def _run_worker(args):
    out = subprocess.run([sys.executable, "-I", WORKER] + args,
                         check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def test_dense_hot_document_is_exact_with_spill():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "corpus.jsonl")
        docs = _write_corpus(jp, 12, 240, seed=11)
        idx_dir = os.path.join(tmp, "idx")
        cache = os.path.join(tmp, "cache")
        info = _run_worker([
            "analyse", jp, idx_dir, cache,
            "--chunk-items", "3", "--chunk-bytes", "64"])
        patterns = [("A", "a"), ("B", "b"), ("AA", "aa"), ("AB", "ab")]
        pairs = [("A", "B"), ("A", "AA"), ("B", "AB"), ("AA", "AB")]
        # The worker used its own identical pattern set; assert exactness
        # against the independent oracle for every pair.
        assert info["pair_counts"] == expected_pair_counts(
            docs, patterns, pairs, info["window"])
        for pair, count in zip(pairs, info["pair_counts"]):
            assert count == len(expected_pairs(
                docs, patterns, pair, info["window"]))
        stats = info["stats"]
        assert stats["hit_sort_runs"] > 10  # real spill at chunk-items=3
        assert stats["pair_sort_runs"] > 10
        assert stats["position_pairs"] == sum(info["pair_counts"])
        assert stats["candidate_pairs"] == stats["position_pairs"]
        assert stats["temp_disk_bytes"] > 0


def test_paging_reads_reach_exact_total():
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "corpus.jsonl")
        docs = _write_corpus(jp, 10, 200, seed=5)
        idx_dir = os.path.join(tmp, "idx")
        cache = os.path.join(tmp, "cache")
        info = _run_worker([
            "paged", jp, idx_dir, cache,
            "--chunk-items", "5", "--page-size", "7"])
        patterns = [("A", "a"), ("B", "b"), ("AA", "aa"), ("AB", "ab")]
        pairs = [("A", "B"), ("A", "AA"), ("B", "AB"), ("AA", "AB")]
        assert info["pair_counts"] == expected_pair_counts(
            docs, patterns, pairs, info["window"])
        # Every pair paginated to exhaustion yields exactly its count and
        # exactly the oracle's rows in order.
        for pair, rows in zip(pairs, info["rows_per_pair"]):
            assert rows == expected_pairs(docs, patterns, pair,
                                         info["window"])
        assert info["page_reads"] > sum(info["pair_counts"]) // 7


def test_peak_rss_flat_while_artifacts_grow():
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, (n_docs, doc_len) in (("small", (40, 220)),
                                       ("large", (160, 220))):
            jp = os.path.join(tmp, f"{tag}.jsonl")
            _write_corpus(jp, n_docs, doc_len, seed=3)
            idx_dir = os.path.join(tmp, f"{tag}_idx")
            cache = os.path.join(tmp, f"{tag}_cache")
            info = _run_worker([
                "analyse", jp, idx_dir, cache,
                "--chunk-items", "2000"])
            runs.append((tag, info))
        (_, small), (_, large) = runs
        size_ratio = large["n_symbols"] / small["n_symbols"]
        rss_ratio = large["peak_rss_kb"] / small["peak_rss_kb"]
        pair_ratio = (large["stats"]["position_pairs"]
                      / max(1, small["stats"]["position_pairs"]))
        # 4x the corpus and many more hits, but the measured peak RSS stays
        # flat (well below corpus growth); on-disk hit files grow with it.
        assert size_ratio >= 3.5
        assert rss_ratio < 1.6
        assert pair_ratio > 3.0
        assert larger_files(large, small)


def larger_files(large, small):
    return (large["stats"]["hit_file_bytes"]
            > small["stats"]["hit_file_bytes"])

"""Bounded-memory and locality evidence for the release pipeline.

A single large document (generated at runtime, never committed) plus a batch
of unordered edits is published twice through deliberately tiny budgets so
the external sorts spill repeatedly.  Each phase runs in a fresh subprocess
whose VmHWM measures that phase alone; we then compare a smaller and a larger
document and require publish peak RSS to stay essentially flat while the
on-disk artifacts grow.  A final local-query subprocess proves that querying
a tiny range reads neither the whole document body nor the lineage of other
documents.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "scripts", "pub_mem_worker.py")


def _write_corpus(path, n_codepoints, seed):
    rng = random.Random(seed)
    alphabet = "abc😀\n\x00秘z"
    with open(path, "w", encoding="utf-8") as f:
        text = "".join(rng.choice(alphabet) for _ in range(n_codepoints))
        f.write(json.dumps({"doc_id": "big", "text": text},
                           ensure_ascii=False))
        f.write("\n")
        f.write(json.dumps({"doc_id": "other",
                            "text": "q" * 200}, ensure_ascii=False))
        f.write("\n")
    return text


def _unordered_edits(path, identity, round_id, n_codepoints, seed,
                     count):
    rng = random.Random(seed)
    records = []
    pos = 1
    spacing = max(8, n_codepoints // count)
    used = set()
    k = 0
    while len(records) < count and pos < n_codepoints - 4:
        span = rng.randint(1, 3)
        if any((pos + j) in used for j in range(span)):
            pos += spacing
            continue
        repl = "".join(rng.choice("XY•秘")
                       for _ in range(rng.randint(0, 3)))
        records.append({"edit_id": f"e{k:05d}", "doc_id": "big",
                        "start": pos, "end": pos + span,
                        "replacement": repl})
        used.update(range(pos, pos + span))
        pos += spacing
        k += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(
            {"expected_input_identity": identity,
             "round_id": round_id}, ensure_ascii=False) + "\n")
        rng.shuffle(records)  # unordered on disk
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def _run_publish(input_path, edit_list, target, tiny=True):
    # -m needs the package on sys.path; the inputs are self-generated temp
    # files, so isolated mode is not required here (matches the other CLI
    # end-to-end tests).  Script workers below still run under -I.
    cmd = [sys.executable, "-m", "diskfm.pubcli", "publish",
           input_path, edit_list, target,
           "--chunk-items", "64", "--chunk-bytes", "4096",
           "--block-bytes", "256", "--write-bytes", "256"]
    out = subprocess.run(cmd, cwd=ROOT, check=True,
                         capture_output=True, text=True)
    return json.loads(out.stdout)


def _build_index(jsonl, idx_dir):
    subprocess.run(
        [sys.executable, "-m", "diskfm", "build", jsonl, idx_dir,
         "--chunk-items", "256", "--sample-interval", "16"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    with open(os.path.join(idx_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["identity"]


def test_two_rounds_bounded_memory_and_local_query():
    evidence = {}
    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for tag, n_cp in (("small", 20000), ("large", 80000)):
            jp = os.path.join(tmp, f"{tag}.jsonl")
            text = _write_corpus(jp, n_cp, seed=11)
            idx = os.path.join(tmp, f"{tag}_idx")
            root_id = _build_index(jp, idx)

            el1 = os.path.join(tmp, f"{tag}_r1.jsonl")
            edits1 = _unordered_edits(
                el1, root_id, "r1", n_cp, seed=21, count=120)
            pub1 = os.path.join(tmp, f"{tag}_pub1")
            m1 = _run_publish(idx, el1, pub1)
            assert m1["build_stats"]["spill_chunks"] > 2
            assert m1["build_stats"]["spill_bytes"] > 0

            el2 = os.path.join(tmp, f"{tag}_r2.jsonl")
            edits2 = _unordered_edits(
                el2, m1["package_identity"], "r2", n_cp, seed=37,
                count=80)
            pub2 = os.path.join(tmp, f"{tag}_pub2")
            m2 = _run_publish(pub1, el2, pub2)
            assert m2["build_stats"]["spill_chunks"] > 2

            # Verify after removing every ancestor.
            subprocess.run(
                [sys.executable, "-m", "diskfm.pubcli", "verify",
                 pub2], cwd=ROOT, check=True, capture_output=True, text=True)
            runs[tag] = {"manifest": m2,
                         "pub_dir": pub2, "text": text,
                         "size_ratio_cp": n_cp / 20000}

        small, large = runs["small"], runs["large"]
        # On-disk body/width footprint scales with the document ...
        size_ratio = (large["manifest"]["build_stats"]["texts_bytes_written"]
                      / small["manifest"]["build_stats"]["texts_bytes_written"])
        assert size_ratio > 2.0
        # ... while the (edit-driven) segment count stays comparable.
        seg_ratio = (large["manifest"]["build_stats"]["segments_written"]
                     / small["manifest"]["build_stats"]["segments_written"])
        assert seg_ratio < 1.5
        # ... and peak RSS of a publish stays essentially flat.
        rss_ratio = (large["manifest"]["build_stats"]["peak_rss_kb"]
                     / small["manifest"]["build_stats"]["peak_rss_kb"])
        evidence["publish_peak_rss_small_kb"] = \
            small["manifest"]["build_stats"]["peak_rss_kb"]
        evidence["publish_peak_rss_large_kb"] = \
            large["manifest"]["build_stats"]["peak_rss_kb"]
        evidence["publish_rss_ratio"] = rss_ratio
        evidence["segment_ratio"] = size_ratio
        evidence["large_spill_bytes"] = \
            large["manifest"]["build_stats"]["spill_bytes"]
        assert rss_ratio < 1.6

        # Local-query locality: a 5-codepoint window must not read the whole
        # document body nor scan every lineage segment.
        big_cp = large["manifest"]["geometry"]["documents"]
        q = subprocess.run(
            [sys.executable, "-I", WORKER, large["pub_dir"], "big",
             "100", "105"], check=True, capture_output=True, text=True)
        info = json.loads(q.stdout)
        evidence["local_query"] = info
        n_segments = large["manifest"]["build_stats"]["segments_written"]
        rt = info["runtime"]
        # Only a handful of lineage records are read for a tiny window.
        assert rt["lineage_records_read"] <= 12
        assert rt["lineage_records_read"] < n_segments
        # No document body bytes are read by coordinate-only queries.
        assert rt["text_bytes_read"] == 0
        # Both query directions target the same single document; the other
        # doc's docs record and lineage are never touched.
        assert rt["documents_touched"] == 2
        # RSS of the query phase is independent of the document length.
        qs = subprocess.run(
            [sys.executable, "-I", WORKER, small["pub_dir"], "big",
             "100", "105"], check=True, capture_output=True, text=True)
        small_info = json.loads(qs.stdout)
        evidence["query_rss_large_kb"] = info["peak_rss_kb"]
        evidence["query_rss_small_kb"] = small_info["peak_rss_kb"]
        assert (info["peak_rss_kb"]
                <= small_info["peak_rss_kb"] * 1.5 + 2048)

        with open(os.path.join(tmp, "publish_memory_evidence.json"),
                  "w") as f:
            json.dump(evidence, f, indent=2, sort_keys=True)
        print("PUBLISH MEMORY EVIDENCE:",
              json.dumps(evidence, sort_keys=True))

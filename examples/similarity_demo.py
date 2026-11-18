"""End-to-end similarity-catalog demo on a tiny generated corpus.

Creates a handful of contract-like documents (exact duplicates, a
one-codepoint variant, a template family, unrelated texts, and a document
too short to have shingles), builds a sealed diskfm index, builds a
similarity catalog on top of it, then runs query / explain / docs /
validate.  Everything is written under a temporary directory.

Run:  python3 examples/similarity_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm import IndexBuilder, SimilarityBuilder, SimilarityCatalog  # noqa: E402

TEMPLATE = "甲方应按照本合同约定向乙方支付款项，乙方应按时交付成果物。"


def make_docs():
    return [
        ("contract-001", TEMPLATE + "付款期限为三十日。"),
        ("contract-002", TEMPLATE + "付款期限为三十日。"),   # exact duplicate
        ("contract-003", TEMPLATE + "付款期限为九十日。"),   # small edit
        ("contract-004", TEMPLATE + "付款期限为三十日", ),   # missing 。
        ("notes", "会议纪要：下周再讨论付款期限。"),
        ("empty", ""),
        ("tiny", "短"),                                       # < q codepoints
    ]


def main() -> int:
    q = 6
    tmp = tempfile.mkdtemp(prefix="diskfm_sim_demo_")
    idx_dir = os.path.join(tmp, "corpus.idx")
    cat_dir = os.path.join(tmp, "corpus.sim")

    docs = make_docs()
    builder = IndexBuilder(idx_dir)
    for doc_id, text in docs:
        builder.add_document(doc_id, text)
    builder.build()
    print(f"[1] sealed substring index at {idx_dir} ({len(docs)} docs)")

    manifest = SimilarityBuilder(idx_dir, cat_dir, q=q).build()
    print(f"[2] similarity catalog at {cat_dir}")
    print("    documents:      ", manifest["documents"]["count"],
          f"(without shingles: {manifest['documents']['without_shingles']})")
    print("    relations:      ", manifest["relations"])
    print("    dictionary:     ", manifest["dictionary"])

    with SimilarityCatalog(cat_dir) as cat:
        print(f"\n[3] near-duplicates of contract-001 (threshold 0.6, q={q})")
        for row in cat.query_all("contract-001", "0.6"):
            print(f"    {row['target_doc_id']:<13} sim={row['similarity']:<5}"
                  f" inter={row['intersection_count']:<3}"
                  f" union={row['union_count']:<3}"
                  f" shingles={row['source_shingle_count']}/"
                  f"{row['target_shingle_count']}")

        print("\n[4] first shared shingles of contract-001 vs contract-003")
        for entry in cat.explain("contract-001", "contract-003")[:4]:
            print(f"    {entry['qgram']!r:<28} hex={entry['qgram_utf8_hex']}"
                  f"  counts={entry['count_in_source']}/"
                  f"{entry['count_in_target']}")

        print("\n[5] per-document diagnostics")
        rows, _ = cat.docs_page(100)
        for r in rows:
            print(f"    {r['doc_id']:<13} shingles={r['shingle_count']:<3}"
                  f" len={r['text_len_codepoints']:<3}"
                  f" comparable={r['comparable']}")

        print("\n[6] deep validation:", json.dumps(cat.validate(),
                                                   sort_keys=True))
    print(f"\nDemo artifacts left in {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

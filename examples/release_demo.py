"""End-to-end CLI-style example of the desensitisation release pipeline.

Builds a tiny in-memory corpus, seals a diskfm index, then publishes two
approved rounds: a replacement and a later replacement-plus-deletion.  It
prints the exported released text, a baseline->release migration (including
the lost fragment) and a release->provenance traceback.  Run with:

    python3 -I examples/release_demo.py

Everything is created inside a temporary directory and nothing is copied out
of the release package: the final queries run after the index and the
intermediate package are removed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm.builder import IndexBuilder
from diskfm.pubbuilder import ReleaseBuilder
from diskfm.pubquery import ReleasePackage
from diskfm.query import DiskFMIndex


def write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(json.dumps(
                {"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")


def write_edit_list(path, identity, round_id, records):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(
            {"expected_input_identity": identity,
             "round_id": round_id}, ensure_ascii=False) + "\n")
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    work = tempfile.mkdtemp(prefix="diskfm-release-demo-")
    try:
        corpus = os.path.join(work, "corpus.jsonl")
        write_jsonl(corpus, [("d1", "A😀秘密BC"), ("d2", "public note")])
        index_dir = os.path.join(work, "index")
        builder = IndexBuilder(index_dir, chunk_items=16)
        builder.ingest_jsonl(corpus)
        builder.build()
        with DiskFMIndex(index_dir) as sealed:
            root_identity = sealed.identity

        # Round 1: mask the two secret codepoints with one bullet.
        list1 = os.path.join(work, "round1.jsonl")
        write_edit_list(list1, root_identity, "round-1", [{
            "edit_id": "mask-secret", "doc_id": "d1",
            "start": 2, "end": 4, "replacement": "•"}])
        pub1 = os.path.join(work, "release-1")
        manifest1 = ReleaseBuilder(
            index_dir, list1, pub1, chunk_items=8).build()

        # Round 2, bound to the *first release*: prepend a marker and delete
        # the bullet and the following original character.
        list2 = os.path.join(work, "round2.jsonl")
        write_edit_list(list2, manifest1["package_identity"], "round-2", [
            {"edit_id": "prepend", "doc_id": "d1", "start": 0, "end": 1,
             "replacement": "前置"},
            {"edit_id": "drop-bullet", "doc_id": "d1", "start": 2,
             "end": 4, "replacement": ""}])
        pub2 = os.path.join(work, "release-2")
        ReleaseBuilder(pub1, list2, pub2, chunk_items=8).build()

        # The package must stand alone: remove every ancestor.
        shutil.rmtree(index_dir)
        shutil.rmtree(pub1)

        with ReleasePackage(pub2) as pkg:
            print("released d1:", pkg.document_text("d1"))
            print("full validation:", pkg.validate())
            print("baseline[1,6) -> release:")
            print(json.dumps(pkg.forward_map("d1", 1, 6),
                             ensure_ascii=False, indent=2, sort_keys=True))
            print("release provenance [0,4):")
            print(json.dumps(pkg.traceback("d1", 0, 4),
                             ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

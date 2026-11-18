"""Memory-measurement worker for the similarity catalog (fresh process).

Spawned by the large-corpus tests so the peak RSS reported from
``/proc/self/status`` (VmHWM) describes one clean build or query phase.
All output is a single JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm import SimilarityBuilder, SimilarityCatalog  # noqa: E402


def peak_rss_kb() -> int:
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM unavailable")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("index")
    b.add_argument("catalog")
    b.add_argument("--q", type=int, required=True)
    b.add_argument("--chunk-items", type=int, required=True)
    b.add_argument("--chunk-bytes", type=int, required=True)

    q = sub.add_parser("query")
    q.add_argument("catalog")
    q.add_argument("doc_id")
    q.add_argument("--threshold", required=True)
    q.add_argument("--counter-chunk-items", type=int, default=20000)
    q.add_argument("--result-chunk-items", type=int, default=20000)
    q.add_argument("--result-chunk-bytes", type=int, default=1 << 22)
    q.add_argument("--max-pages", type=int, default=8)
    q.add_argument("--page-records", type=int, default=1024)

    args = p.parse_args()
    if args.cmd == "build":
        builder = SimilarityBuilder(
            args.index, args.catalog, q=args.q,
            chunk_items=args.chunk_items, chunk_bytes=args.chunk_bytes)
        manifest = builder.build()
        out = {
            "phase": "build",
            "peak_rss_kb": peak_rss_kb(),
            "documents": manifest["documents"],
            "relations": manifest["relations"],
            "dictionary": manifest["dictionary"],
            "build_stats": manifest["build_stats"],
            "file_bytes": {k: v["bytes"]
                           for k, v in manifest["files"].items()},
        }
    else:
        with SimilarityCatalog(
                args.catalog, max_pages=args.max_pages,
                page_records=args.page_records,
                counter_chunk_items=args.counter_chunk_items,
                result_chunk_items=args.result_chunk_items,
                result_chunk_bytes=args.result_chunk_bytes) as cat:
            rows = cat.query_all(args.doc_id, args.threshold)
            out = {
                "phase": "query",
                "peak_rss_kb": peak_rss_kb(),
                "results": len(rows),
                "first_rows": rows[:3],
                "query_stats": cat.stats()["last_query"],
                "pagers": cat.stats()["pagers"],
            }
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

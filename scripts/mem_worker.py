"""Memory-measurement worker: build/query in a fresh process.

The test harness spawns this as a subprocess so the peak resident set size
reported from ``/proc/self/status`` (VmHWM) describes one clean phase only,
without the parent test runner's allocations mixed in.  All output is a
single JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm import DiskFMIndex, IndexBuilder  # noqa: E402


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
    b.add_argument("input")
    b.add_argument("index")
    b.add_argument("--chunk-items", type=int, required=True)
    b.add_argument("--sample-interval", type=int, default=16)

    q = sub.add_parser("query")
    q.add_argument("index")
    q.add_argument("query")
    q.add_argument("--page-size", type=int, default=25)
    q.add_argument("--max-pages", type=int, default=8)
    q.add_argument("--page-records", type=int, default=1024)

    args = p.parse_args()
    if args.cmd == "build":
        builder = IndexBuilder(
            args.index, chunk_items=args.chunk_items,
            sample_interval=args.sample_interval)
        builder.ingest_jsonl(args.input)
        manifest = builder.build()
        out = {
            "phase": "build",
            "peak_rss_kb": peak_rss_kb(),
            "n_symbols": manifest["geometry"]["n_symbols"],
            "build_stats": manifest["build_stats"],
            "file_bytes": {k: v["bytes"]
                           for k, v in manifest["files"].items()},
        }
    else:
        with DiskFMIndex(args.index, max_pages=args.max_pages,
                         page_records=args.page_records) as idx:
            count = idx.count(args.query)
            # Snapshot immediately after count: proves count alone does no
            # LF walks, SA reads or document decoding.
            count_stats = idx.stats()
            page, cursor = idx.locate_page(
                args.query, args.page_size)
            out = {
                "phase": "query",
                "peak_rss_kb": peak_rss_kb(),
                "count": count,
                "page_hits": len(page),
                "has_next_page": cursor is not None,
                "count_stats": count_stats,
                "query_stats": idx.stats(),
                "n_symbols": idx.n,
            }
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

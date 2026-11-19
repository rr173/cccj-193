"""Memory-measurement worker for co-occurrence analysis (fresh process).

Run as a subprocess so peak RSS (VmHWM) describes one clean phase.  Output
is a single JSON object on stdout; every statistic is measured, none is
estimated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm import DiskFMIndex, IndexBuilder  # noqa: E402
from diskfm.cooccur import build_request, compute_analysis  # noqa: E402

PATTERNS = [("A", "a"), ("B", "b"), ("AA", "aa"), ("AB", "ab")]
PAIRS = [("A", "B"), ("A", "AA"), ("B", "AB"), ("AA", "AB")]
WINDOW = 3


def peak_rss_kb() -> int:
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM unavailable")


def _build(jp, idx_dir):
    builder = IndexBuilder(idx_dir, chunk_items=4000, sample_interval=16)
    builder.ingest_jsonl(jp)
    manifest = builder.build()
    return manifest["geometry"]["n_symbols"]


def _open(idx_dir):
    return DiskFMIndex(idx_dir, max_pages=8, page_records=256)


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyse")
    a.add_argument("corpus")
    a.add_argument("index")
    a.add_argument("cache")
    a.add_argument("--chunk-items", type=int, default=2000)
    a.add_argument("--chunk-bytes", type=int, default=1 << 22)

    pg = sub.add_parser("paged")
    pg.add_argument("corpus")
    pg.add_argument("index")
    pg.add_argument("cache")
    pg.add_argument("--chunk-items", type=int, default=2000)
    pg.add_argument("--page-size", type=int, default=7)

    args = p.parse_args()
    if not os.path.exists(args.index):
        n_symbols = _build(args.corpus, args.index)
    else:
        with _open(args.index) as idx:
            n_symbols = idx.n
    request = build_request(PATTERNS, PAIRS, WINDOW)
    with _open(args.index) as idx:
        analysis = compute_analysis(
            idx, request, chunk_items=args.chunk_items,
            chunk_bytes=getattr(args, "chunk_bytes", 1 << 22),
            cache_root=args.cache)
        try:
            stats = dict(analysis.stats()["analysis"])
            if args.cmd == "analyse":
                # Touch every pair so the reported totals are the real ones.
                for pair in PAIRS:
                    list(analysis.iter_pairs(pair))
                    list(analysis.pair_summary(pair))
                out = {
                    "phase": "analyse",
                    "n_symbols": n_symbols,
                    "peak_rss_kb": peak_rss_kb(),
                    "window": WINDOW,
                    "pair_counts": list(analysis.pair_counts),
                    "stats": stats,
                }
            else:
                rows_per_pair = []
                page_reads = 0
                for pair in PAIRS:
                    rows = []
                    cursor = None
                    while True:
                        page, cursor = analysis.pairs_page(
                            pair, args.page_size, cursor)
                        rows.extend(page)
                        page_reads += 1
                        if cursor is None:
                            break
                    rows_per_pair.append(rows)
                out = {
                    "phase": "paged",
                    "n_symbols": n_symbols,
                    "peak_rss_kb": peak_rss_kb(),
                    "window": WINDOW,
                    "pair_counts": list(analysis.pair_counts),
                    "rows_per_pair": rows_per_pair,
                    "page_reads": page_reads,
                    "stats": stats,
                }
        finally:
            analysis.close()
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

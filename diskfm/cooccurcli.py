"""CLI for windowed multi-phrase co-occurrence analysis.

    python3 -m diskfm.cooccurcli run INDEX REQUEST.json \
        [--summary-page-size N] [--pair-page-size N] [--all] \
        [--cursor TOKEN] [--pair A B] [--stats]

The request JSON is::

    {"patterns": [{"pattern_id": "A", "phrase": "foo"}, ...],
     "pairs": [["A", "B"], ...],
     "window": 12}

The command prints one JSON object per line.  With ``--all`` it streams
every non-zero summary row and every position pair; otherwise it prints one
summary page (or one position-pair page when ``--pair`` is given) and
writes ``next_cursor=<token>`` to stderr.  Cursors bind the index identity,
the complete request and the analysis, so changing any of them invalidates
the token.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .cooccur import (DEFAULT_CHUNK_BYTES, DEFAULT_CHUNK_ITEMS, CooccurError,
                      compute_analysis, request_from_dict)
from .format import CursorError, IndexCorruptError
from .query import DiskFMIndex


def _load_request(path: str):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return request_from_dict(obj)


def _emit(row: dict) -> None:
    print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))


def _cmd_run(args) -> int:
    request = _load_request(args.request)
    with DiskFMIndex(args.index, max_pages=args.max_pages,
                     page_records=args.page_records) as idx:
        analysis = compute_analysis(
            idx, request, chunk_items=args.chunk_items,
            chunk_bytes=args.chunk_bytes, refresh=args.refresh,
            cache_root=args.cache_dir)
        try:
            if args.pair is not None:
                pair = (args.pair[0], args.pair[1])
                if args.all:
                    for row in analysis.iter_pairs(pair):
                        _emit(row)
                else:
                    rows, cursor = analysis.pairs_page(
                        pair, args.pair_page_size, args.cursor)
                    for row in rows:
                        _emit(row)
                    if cursor is not None:
                        sys.stderr.write(f"next_cursor={cursor}\n")
            elif args.all:
                for row in analysis.iter_summary():
                    _emit(row)
                if args.pairs:
                    for pair in request.pairs:
                        for row in analysis.iter_pairs(pair):
                            _emit(row)
            else:
                rows, cursor = analysis.summary_page(
                    args.summary_page_size, args.cursor)
                for row in rows:
                    _emit(row)
                if cursor is not None:
                    sys.stderr.write(f"next_cursor={cursor}\n")
            if args.stats:
                json.dump(analysis.stats(), sys.stderr, sort_keys=True,
                          indent=2)
                sys.stderr.write("\n")
        finally:
            analysis.close()
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog="diskfm-cooccur",
        description="windowed multi-phrase co-occurrence analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compute or reopen an analysis")
    r.add_argument("index")
    r.add_argument("request", help="request JSON file")
    r.add_argument("--summary-page-size", type=int, default=64)
    r.add_argument("--pair-page-size", type=int, default=64)
    r.add_argument("--cursor", default=None)
    r.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                   help="page position pairs for this pair")
    r.add_argument("--all", action="store_true",
                   help="stream all summary rows (and pairs with --pairs)")
    r.add_argument("--pairs", action="store_true",
                   help="with --all, also stream every position pair")
    r.add_argument("--chunk-items", type=int, default=DEFAULT_CHUNK_ITEMS)
    r.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    r.add_argument("--max-pages", type=int, default=16)
    r.add_argument("--page-records", type=int, default=4096)
    r.add_argument("--cache-dir", default=None,
                   help="analysis cache root (default: temp dir keyed by "
                        "index identity)")
    r.add_argument("--refresh", action="store_true",
                   help="recompute instead of reopening a cached analysis")
    r.add_argument("--stats", action="store_true")
    r.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (IndexCorruptError, CursorError, CooccurError, ValueError,
            KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

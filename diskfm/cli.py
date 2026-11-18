"""Command line interface: build / count / locate / verify / stats."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import List

from .builder import IndexBuilder
from .format import CursorError, IndexCorruptError
from .query import DiskFMIndex


def _cmd_build(args) -> int:
    builder = IndexBuilder(
        args.index, chunk_items=args.chunk_items,
        sample_interval=args.sample_interval,
        byte_sample_interval=args.byte_sample_interval,
        force_wide=args.force_wide)
    builder.ingest_jsonl(args.input)
    manifest = builder.build()
    json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_count(args) -> int:
    with DiskFMIndex(args.index) as idx:
        print(idx.count(args.query))
        if args.stats:
            json.dump(idx.stats(), sys.stderr, sort_keys=True, indent=2)
            sys.stderr.write("\n")
    return 0


def _cmd_locate(args) -> int:
    with DiskFMIndex(args.index) as idx:
        cursor = args.cursor
        pages = 0
        while True:
            hits, cursor = idx.locate_page(args.query, args.page_size, cursor)
            for h in hits:
                print(json.dumps(asdict(h), ensure_ascii=False,
                                 separators=(",", ":")))
            pages += 1
            if cursor is None or not args.all:
                break
        if cursor is not None and not args.all:
            sys.stderr.write(f"next_cursor={cursor}\n")
        if args.stats:
            json.dump({"pages": pages, **idx.stats()}, sys.stderr,
                      sort_keys=True, indent=2)
            sys.stderr.write("\n")
    return 0


def _cmd_verify(args) -> int:
    try:
        with DiskFMIndex(args.index):
            pass
    except IndexCorruptError as exc:
        print(f"CORRUPT: {exc}", file=sys.stderr)
        return 2
    print("ok")
    return 0


def _cmd_stats(args) -> int:
    with DiskFMIndex(args.index) as idx:
        out = {"manifest": idx.manifest, "runtime_counters": idx.stats()}
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2,
                  sort_keys=True)
        sys.stdout.write("\n")
    return 0


def _cmd_get(args) -> int:
    with DiskFMIndex(args.index) as idx:
        sys.stdout.write(idx.document_text(args.doc_id))
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(prog="diskfm",
                                description="external-memory substring index")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build an index from JSONL")
    b.add_argument("input")
    b.add_argument("index")
    b.add_argument("--chunk-items", type=int, default=20000,
                   help="max records held in one in-memory sort buffer")
    b.add_argument("--sample-interval", type=int, default=32)
    b.add_argument("--byte-sample-interval", type=int, default=64)
    b.add_argument("--force-wide", action="store_true",
                   help="force the 12-byte Triple32 external-sort codec")
    b.set_defaults(func=_cmd_build)

    c = sub.add_parser("count", help="count overlapping occurrences")
    c.add_argument("index")
    c.add_argument("query")
    c.add_argument("--stats", action="store_true")
    c.set_defaults(func=_cmd_count)

    l = sub.add_parser("locate", help="page through occurrence coordinates")
    l.add_argument("index")
    l.add_argument("query")
    l.add_argument("--page-size", type=int, default=64)
    l.add_argument("--cursor", default=None)
    l.add_argument("--all", action="store_true",
                   help="follow every page until exhausted")
    l.add_argument("--stats", action="store_true")
    l.set_defaults(func=_cmd_locate)

    v = sub.add_parser("verify", help="re-validate checksums and geometry")
    v.add_argument("index")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("stats", help="manifest and runtime counters")
    s.add_argument("index")
    s.set_defaults(func=_cmd_stats)

    g = sub.add_parser("get", help="print one original document verbatim")
    g.add_argument("index")
    g.add_argument("doc_id")
    g.set_defaults(func=_cmd_get)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (IndexCorruptError, CursorError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

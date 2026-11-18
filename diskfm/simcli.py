"""Command line interface for the document similarity catalog.

Independent of the substring-index CLI (``diskfm``): this builds, queries,
explains and validates qgram-posting similarity catalogs.

    python3 -m diskfm.simcli build INDEX_DIR CATALOG_DIR --q 5
    python3 -m diskfm.simcli query CATALOG_DIR DOC_ID --threshold 0.8
    python3 -m diskfm.simcli explain CATALOG_DIR DOC_ID TARGET_DOC_ID
    python3 -m diskfm.simcli docs CATALOG_DIR
    python3 -m diskfm.simcli validate CATALOG_DIR
    python3 -m diskfm.simcli stats CATALOG_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .format import CursorError, IndexCorruptError
from .simbuilder import SimilarityBuilder
from .simquery import SimilarityCatalog


def _cmd_build(args) -> int:
    builder = SimilarityBuilder(
        args.index, args.catalog, q=args.q,
        chunk_items=args.chunk_items, chunk_bytes=args.chunk_bytes,
        expect_identity=args.expect_identity)
    manifest = builder.build()
    json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2,
              sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_query(args) -> int:
    if args.all and args.cursor is not None:
        print("error: --all and --cursor are mutually exclusive",
              file=sys.stderr)
        return 2
    with SimilarityCatalog(args.catalog, max_pages=args.max_pages,
                           page_records=args.page_records) as cat:
        if args.all:
            for row in cat.iter_results(args.doc_id, args.threshold):
                print(json.dumps(row, ensure_ascii=False,
                                 separators=(",", ":")))
        else:
            rows, cursor = cat.query_page(
                args.doc_id, args.threshold, args.page_size, args.cursor)
            for row in rows:
                print(json.dumps(row, ensure_ascii=False,
                                 separators=(",", ":")))
            if cursor is not None:
                sys.stderr.write(f"next_cursor={cursor}\n")
        if args.stats:
            json.dump(cat.stats().get("last_query", {}), sys.stderr,
                      sort_keys=True, indent=2)
            sys.stderr.write("\n")
    return 0


def _cmd_explain(args) -> int:
    with SimilarityCatalog(args.catalog, max_pages=args.max_pages,
                           page_records=args.page_records) as cat:
        shared = cat.explain(args.doc_id, args.target_doc_id)
        for entry in shared:
            print(json.dumps(entry, ensure_ascii=False,
                             separators=(",", ":")))
        if args.stats:
            json.dump(cat.stats().get("last_explain", {}), sys.stderr,
                      sort_keys=True, indent=2)
            sys.stderr.write("\n")
    return 0


def _cmd_docs(args) -> int:
    with SimilarityCatalog(args.catalog) as cat:
        cursor = args.cursor
        while True:
            rows, cursor = cat.docs_page(args.page_size, cursor)
            for row in rows:
                print(json.dumps(row, ensure_ascii=False,
                                 separators=(",", ":")))
            if cursor is None or not args.all:
                break
        if cursor is not None and not args.all:
            sys.stderr.write(f"next_cursor={cursor}\n")
    return 0


def _cmd_validate(args) -> int:
    try:
        with SimilarityCatalog(args.catalog) as cat:
            report = cat.validate()
    except IndexCorruptError as exc:
        print(f"CORRUPT: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **report}, sort_keys=True))
    return 0


def _cmd_stats(args) -> int:
    with SimilarityCatalog(args.catalog) as cat:
        json.dump(cat.stats(), sys.stdout, ensure_ascii=False, indent=2,
                  sort_keys=True)
        sys.stdout.write("\n")
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog="diskfm-sim",
        description="document-level near-duplicate audit catalog")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a similarity catalog from an index")
    b.add_argument("index", help="sealed diskfm index directory (read-only)")
    b.add_argument("catalog", help="catalog directory to create")
    b.add_argument("--q", type=int, required=True,
                   help="shingle length in Unicode code points (>= 1)")
    b.add_argument("--chunk-items", type=int, default=20000,
                   help="max records held in one in-memory sort buffer")
    b.add_argument("--chunk-bytes", type=int, default=1 << 22,
                   help="max serialised bytes held in one sort buffer")
    b.add_argument("--expect-identity", default=None,
                   help="require this exact source index identity")
    b.set_defaults(func=_cmd_build)

    q = sub.add_parser("query", help="list near-duplicates of one document")
    q.add_argument("catalog")
    q.add_argument("doc_id")
    q.add_argument("--threshold", required=True,
                   help="Jaccard threshold in (0,1]; decimal or num/den")
    q.add_argument("--page-size", type=int, default=64)
    q.add_argument("--cursor", default=None)
    q.add_argument("--all", action="store_true",
                   help="stream every result in one computation")
    q.add_argument("--max-pages", type=int, default=16)
    q.add_argument("--page-records", type=int, default=4096)
    q.add_argument("--stats", action="store_true")
    q.set_defaults(func=_cmd_query)

    e = sub.add_parser("explain", help="show shingles shared by two documents")
    e.add_argument("catalog")
    e.add_argument("doc_id")
    e.add_argument("target_doc_id")
    e.add_argument("--max-pages", type=int, default=16)
    e.add_argument("--page-records", type=int, default=4096)
    e.add_argument("--stats", action="store_true")
    e.set_defaults(func=_cmd_explain)

    d = sub.add_parser("docs", help="page through per-document diagnostics")
    d.add_argument("catalog")
    d.add_argument("--page-size", type=int, default=64)
    d.add_argument("--cursor", default=None)
    d.add_argument("--all", action="store_true")
    d.set_defaults(func=_cmd_docs)

    v = sub.add_parser("validate", help="deep re-validation of the catalog")
    v.add_argument("catalog")
    v.set_defaults(func=_cmd_validate)

    s = sub.add_parser("stats", help="manifest and runtime counters")
    s.add_argument("catalog")
    s.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (IndexCorruptError, CursorError, ValueError, KeyError,
            FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

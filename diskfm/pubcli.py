"""Command line interface for the desensitisation release pipeline.

    python3 -m diskfm.pubcli publish INPUT EDIT_LIST TARGET [budget flags]
    python3 -m diskfm.pubcli export  PACKAGE [OUT_PATH]
    python3 -m diskfm.pubcli forward PACKAGE DOC_ID START END
    python3 -m diskfm.pubcli trace   PACKAGE DOC_ID START END
    python3 -m diskfm.pubcli text    PACKAGE DOC_ID
    python3 -m diskfm.pubcli verify  PACKAGE
    python3 -m diskfm.pubcli stats   PACKAGE

``INPUT`` is either a sealed diskfm index (chain root) or a completed
release package; it is only ever opened read-only.  Diagnostics never echo
consumed fragments or replacement values -- coordinates and identifiers only.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import pubformat as pfmt
from .pubbuilder import (DEFAULT_BLOCK_BYTES, DEFAULT_CHUNK_BYTES,
                         DEFAULT_CHUNK_ITEMS, DEFAULT_MAX_EDIT_BYTES,
                         DEFAULT_MAX_EDITS, DEFAULT_MAX_ID_BYTES,
                         DEFAULT_MAX_INTERVAL_CP,
                         DEFAULT_MAX_REPLACEMENT_CP, DEFAULT_WRITE_BYTES,
                         ReleaseBuilder)
from .pubquery import ReleasePackage


def _cmd_publish(args) -> int:
    builder = ReleaseBuilder(
        args.input, args.edit_list, args.target,
        chunk_items=args.chunk_items, chunk_bytes=args.chunk_bytes,
        block_bytes=args.block_bytes, write_bytes=args.write_bytes,
        max_edit_bytes=args.max_edit_bytes,
        max_replacement_cp=args.max_replacement_codepoints,
        max_interval_cp=args.max_interval_codepoints,
        max_id_bytes=args.max_identifier_bytes,
        max_edits=args.max_edits)
    manifest = builder.build()
    json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2,
              sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_export(args) -> int:
    with ReleasePackage(args.package) as pkg:
        out = sys.stdout.buffer if args.output is None else open(
            args.output, "wb")
        try:
            pkg.export_jsonl(out, block_bytes=args.block_bytes)
        finally:
            if args.output is not None:
                out.close()
    return 0


def _cmd_forward(args) -> int:
    with ReleasePackage(args.package) as pkg:
        result = pkg.forward_map(args.doc_id, args.start, args.end)
        json.dump(result, sys.stdout, ensure_ascii=False,
                  separators=(",", ":"), sort_keys=True)
        sys.stdout.write("\n")
        if args.stats:
            json.dump(pkg.runtime_stats(), sys.stderr, sort_keys=True,
                      indent=2)
            sys.stderr.write("\n")
    return 0


def _cmd_trace(args) -> int:
    with ReleasePackage(args.package) as pkg:
        result = pkg.traceback(args.doc_id, args.start, args.end)
        json.dump(result, sys.stdout, ensure_ascii=False,
                  separators=(",", ":"), sort_keys=True)
        sys.stdout.write("\n")
        if args.stats:
            json.dump(pkg.runtime_stats(), sys.stderr, sort_keys=True,
                      indent=2)
            sys.stderr.write("\n")
    return 0


def _cmd_text(args) -> int:
    with ReleasePackage(args.package) as pkg:
        sys.stdout.write(pkg.document_text(args.doc_id))
    return 0


def _cmd_verify(args) -> int:
    try:
        with ReleasePackage(args.package) as pkg:
            report = pkg.validate()
    except pfmt.ReleaseCorruptError as exc:
        print(f"CORRUPT: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **report}, sort_keys=True))
    return 0


def _cmd_stats(args) -> int:
    with ReleasePackage(args.package) as pkg:
        json.dump({"manifest": pkg.manifest,
                   "runtime": pkg.runtime_stats()},
                  sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog="diskfm-pub",
        description="desensitisation release publishing and lineage queries")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("publish", help="apply one approved edit round")
    b.add_argument("input", help="sealed index or previous release package")
    b.add_argument("edit_list", help="header + unordered JSONL edit records")
    b.add_argument("target", help="new release directory (must not exist)")
    b.add_argument("--chunk-items", type=int, default=DEFAULT_CHUNK_ITEMS)
    b.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    b.add_argument("--block-bytes", type=int, default=DEFAULT_BLOCK_BYTES)
    b.add_argument("--write-bytes", type=int, default=DEFAULT_WRITE_BYTES)
    b.add_argument("--max-edit-bytes", type=int,
                   default=DEFAULT_MAX_EDIT_BYTES)
    b.add_argument("--max-replacement-codepoints", type=int,
                   default=DEFAULT_MAX_REPLACEMENT_CP)
    b.add_argument("--max-interval-codepoints", type=int,
                   default=DEFAULT_MAX_INTERVAL_CP)
    b.add_argument("--max-identifier-bytes", type=int,
                   default=DEFAULT_MAX_ID_BYTES)
    b.add_argument("--max-edits", type=int, default=DEFAULT_MAX_EDITS)
    b.set_defaults(func=_cmd_publish)

    e = sub.add_parser("export", help="stream {doc_id,text} JSONL")
    e.add_argument("package")
    e.add_argument("output", nargs="?", default=None,
                   help="output path (default: stdout)")
    e.add_argument("--block-bytes", type=int, default=DEFAULT_BLOCK_BYTES)
    e.set_defaults(func=_cmd_export)

    fwd = sub.add_parser(
        "forward", help="baseline -> release mapping with lost fragments")
    fwd.add_argument("package")
    fwd.add_argument("doc_id")
    fwd.add_argument("start", type=int)
    fwd.add_argument("end", type=int)
    fwd.add_argument("--stats", action="store_true")
    fwd.set_defaults(func=_cmd_forward)

    tr = sub.add_parser(
        "trace", help="release -> baseline/generated provenance")
    tr.add_argument("package")
    tr.add_argument("doc_id")
    tr.add_argument("start", type=int)
    tr.add_argument("end", type=int)
    tr.add_argument("--stats", action="store_true")
    tr.set_defaults(func=_cmd_trace)

    t = sub.add_parser("text", help="print one released document verbatim")
    t.add_argument("package")
    t.add_argument("doc_id")
    t.set_defaults(func=_cmd_text)

    v = sub.add_parser("verify", help="validate every file and structure")
    v.add_argument("package")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("stats", help="manifest and runtime counters")
    s.add_argument("package")
    s.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (pfmt.PublishRejected, pfmt.ReleaseCorruptError, ValueError,
            KeyError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

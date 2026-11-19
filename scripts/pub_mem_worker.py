"""Subprocess worker for release-pipeline bounded-memory evidence.

Runs one isolated phase (a local lineage query) so its VmHWM measures only
that phase, and prints runtime counters proving a small range query does not
scan the whole document body or read other documents' lineage.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskfm.pubquery import ReleasePackage  # noqa: E402


def _peak_rss_kb():
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def main(argv):
    package, doc_id, start, end = argv[1], argv[2], int(argv[3]), int(argv[4])
    with ReleasePackage(package) as pkg:
        rec = pkg._doc_record(pkg.require_doc(doc_id))
        forward = pkg.forward_map(doc_id, start, end)
        trace = pkg.traceback(doc_id, start, end)
        info = {
            "peak_rss_kb": _peak_rss_kb(),
            "document_codepoints": rec["orig_cp_len"],
            "request": [start, end],
            "runtime": pkg.runtime_stats(),
            "mapped_pieces": len(forward["mapped"]),
            "lost_pieces": len(forward["lost"]),
            "trace_pieces": len(trace["pieces"]),
        }
    print(json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

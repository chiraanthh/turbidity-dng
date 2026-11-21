#!/usr/bin/env python
"""Estimate turbidity (NTU) from iPhone Bayer-RAW DNGs of the laser-through-cuvette setup.

    python predict.py IMG_0001.dng [IMG_0002.dng ...] [--json] [--models models/]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import load_bundle      # noqa: E402
from service import analyse        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dng", nargs="+", type=Path)
    ap.add_argument("--models", type=Path, default=Path(__file__).resolve().parent / "models")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    bundle = load_bundle(a.models)
    results = []
    for p in a.dng:
        try:
            r, _ = analyse(p, bundle)
        except Exception as e:                          # unreadable file, ProRAW, wrong CFA, ...
            r = dict(file=p.name, status="error", message=str(e))
        results.append(r)
        if not a.json:
            print(f"{r['file']}: [{r['status']}] {r['message']}")
            for w in r.get("qc_warnings", []):
                print(f"    warning: {w}")
    if a.json:
        print(json.dumps(results, indent=1))
    return 0 if all(r["status"] != "error" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Merge per-rank Chrome trace files into a single multi-pid trace.

Usage:
  python scripts/merge_traces.py outputs/traces/rank_*_trace.json -o outputs/traces/merged_trace.json
  python scripts/merge_traces.py outputs/traces/                    # auto-glob rank_*_trace.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def merge_traces(trace_paths: list[str], output_path: str) -> str:
    """Merge per-rank trace files, assigning each rank a unique pid."""
    all_events = []
    all_metadata = []
    for rank, path in enumerate(sorted(trace_paths)):
        with open(path) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        for ev in events:
            ev["pid"] = rank
        all_events.extend(events)
        if "_metadata" in data:
            all_metadata.append(data["_metadata"])

    merged = {
        "traceEvents": all_events,
        "displayTimeUnit": "ns",
        "_metadata": all_metadata[0] if all_metadata else None,
        "_all_metadata": all_metadata,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge per-rank Chrome traces")
    parser.add_argument("inputs", nargs="*", help="Trace files or directory containing rank_*_trace.json")
    parser.add_argument("-o", "--output", default="outputs/traces/merged_trace.json")
    args = parser.parse_args()

    paths = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "rank_*_trace.json"))))
        else:
            paths.append(inp)

    if not paths:
        # Fallback: look in outputs/traces/
        default_dir = "outputs/traces"
        paths = sorted(glob.glob(os.path.join(default_dir, "rank_*_trace.json")))
        if not paths:
            print("ERROR: No trace files found.", file=sys.stderr)
            sys.exit(1)

    output = merge_traces(paths, args.output)
    print(f"Merged {len(paths)} rank trace(s) → {output}")


if __name__ == "__main__":
    main()

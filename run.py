#!/usr/bin/env python3
"""CLI:  python run.py --pitch "..."   (or --pitch-file my_pitch.txt)

Exit codes: 0 verified result, 2 unverified, 3 provider error, 4 invalid input, 1 tool error.
The run (including failed runs) is saved as outputs/runs/<run_id>.json."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pitchcheck.llm import load_config  # noqa: E402
from pitchcheck.pipeline import CardCache, check_pitch, format_text, load_catalog  # noqa: E402
from pitchcheck.runs import save_run  # noqa: E402
from pitchcheck.textutil import setup_stdout  # noqa: E402

EXIT = {"verified": 0, "unverified": 2, "provider_error": 3, "invalid_input": 4}


def main():
    setup_stdout()
    ap = argparse.ArgumentParser(description="Check a new microdrama pitch against the catalog.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pitch", help="the pitch text")
    g.add_argument("--pitch-file", help="path to a UTF-8 text file holding the pitch")
    ap.add_argument("--config", default=None)
    ap.add_argument("--catalog", default=os.path.join(HERE, "data", "catalog.csv"))
    ap.add_argument("--mode", choices=["full", "shortlist"], default=None, help="full (default) compares every show")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "outputs", "runs"))
    a = ap.parse_args()

    if a.pitch is not None:
        pitch = a.pitch
    else:
        with open(a.pitch_file, encoding="utf-8-sig") as f:
            pitch = f.read()
    cfg = load_config(a.config)
    catalog = load_catalog(a.catalog)
    cache = CardCache(os.path.join(HERE, "outputs", "cards_cache.json"))
    res = check_pitch(pitch.strip(), catalog, cfg, cache, mode=a.mode, k=a.k)
    path = save_run(res, a.runs_dir)
    print(format_text(res, catalog))
    print(f"\n[run saved: {path}]")
    return EXIT.get(res["status"], 1)


if __name__ == "__main__":
    sys.exit(main())

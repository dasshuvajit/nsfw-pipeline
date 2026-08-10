#!/usr/bin/env python3
"""Reclaim disk from already-packaged art_series runs.

A packaged run keeps its sellable output in ``package/`` (public/ + gated/ are
COPIES of the keepers) — the raw candidate renders in ``base/`` and ``covers/``
are redundant once the package exists. This tool deletes those intermediates
for runs that (a) have a ``package/`` dir and (b) are older than ``--keep-days``.

NEVER touched, by design: ``manifest.json`` (the cross-run diversity memory —
deleting it breaks anti-repetition), ``prompts.json``, ``package/``, contact
sheets, or any run without a package (it may still be mid-pipeline or worth
re-packaging).

Safe by default: dry-run unless ``--apply`` is passed.

    python3 scripts/reclaim_disk.py                 # report only
    python3 scripts/reclaim_disk.py --apply         # actually delete
    python3 scripts/reclaim_disk.py --keep-days 3   # tighter window
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES_DIR = ROOT / "output" / "art_series"
# Intermediate dirs that are safe to drop once package/ exists.
RECLAIMABLE_SUBDIRS = ("base", "covers")


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run report)")
    ap.add_argument("--keep-days", type=float, default=7.0,
                    help="only reclaim runs older than this many days "
                         "(default 7 — recent runs may still be under review)")
    args = ap.parse_args()

    if not SERIES_DIR.is_dir():
        print(f"nothing to do — {SERIES_DIR} does not exist")
        return 0

    cutoff = time.time() - args.keep_days * 86400
    total = 0
    targets: list[tuple[Path, int]] = []
    for run in sorted(SERIES_DIR.iterdir()):
        if not run.is_dir() or not (run / "package").is_dir():
            continue                      # unpackaged / foreign dir — never touch
        if run.stat().st_mtime > cutoff:
            continue                      # too recent
        for sub in RECLAIMABLE_SUBDIRS:
            d = run / sub
            if d.is_dir():
                size = _dir_bytes(d)
                if size:
                    targets.append((d, size))
                    total += size

    if not targets:
        print("nothing reclaimable (no packaged runs older than "
              f"{args.keep_days:g} days with intermediates)")
        return 0

    for d, size in targets:
        rel = d.relative_to(ROOT)
        if args.apply:
            shutil.rmtree(d)
            print(f"  deleted  {rel}  ({_fmt(size)})")
        else:
            print(f"  would delete  {rel}  ({_fmt(size)})")
    verb = "reclaimed" if args.apply else "reclaimable (re-run with --apply)"
    print(f"\n{_fmt(total)} {verb} across {len(targets)} dir(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

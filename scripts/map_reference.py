#!/usr/bin/env python3
"""Map a chosen reference image to a character in the DB.

Use this after generating candidates with
``python scripts/render_set.py --ratio square ...``, picking the best
render visually, and copying it under ``characters/<id>/``. This script
just UPDATEs ``characters.reference_image_path`` — no cropping, no face
detection, no side files. The user owns the file.

What this does:

  1. Reads the ``characters`` row for ``--character-id``.
  2. Resolves ``--image`` to an absolute path and checks it exists.
  3. Converts to a path relative to the project root (so the DB value
     stays portable across machines) and UPDATEs the row.
  4. Prints the before → after diff.

Exit codes:
  0  success
  1  DB file not found
  2  character_id not found in DB
  3  --image file does not exist
  5  unexpected SQLite error

Example:
    python scripts/map_reference.py --character-id char_001 \\
           --image characters/char_001/reference.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

# Make src/ importable when running this file as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


def _load_db_path(override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    default = PROJECT_ROOT / "nsfw_pipeline.db"
    if not CONFIG_PATH.exists():
        return default
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    raw = (cfg.get("pipeline") or {}).get("db_path")
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _resolve_image(arg: str) -> Path:
    """Expand --image to an absolute path (resolved against PROJECT_ROOT)."""
    p = Path(arg).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _to_db_value(abs_path: Path) -> str:
    """Prefer a project-root-relative path, falling back to absolute."""
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--character-id",
        required=True,
        help="Character primary key (e.g. char_001).",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the chosen reference image (absolute or relative "
        "to the project root).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override DB path. Default: read from config/pipeline.yaml.",
    )
    args = parser.parse_args()

    db_path = _load_db_path(args.db_path)
    if not db_path.exists():
        print(
            f"ERROR: database not found at {db_path}.\n"
            f"Run `python scripts/init_db.py` first.",
            file=sys.stderr,
        )
        return 1

    image_path = _resolve_image(args.image)
    if not image_path.exists():
        print(
            f"ERROR: --image not found on disk: {image_path}",
            file=sys.stderr,
        )
        return 3

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        row = conn.execute(
            "SELECT id, reference_image_path FROM characters WHERE id = ?",
            (args.character_id,),
        ).fetchone()

        if row is None:
            print(
                f"ERROR: character {args.character_id!r} not found in DB.",
                file=sys.stderr,
            )
            return 2

        old_value = row["reference_image_path"]
        new_value = _to_db_value(image_path)

        conn.execute(
            "UPDATE characters SET reference_image_path = ? WHERE id = ?",
            (new_value, args.character_id),
        )
        conn.commit()
    except sqlite3.Error as exc:
        print(f"ERROR: SQLite error: {exc}", file=sys.stderr)
        return 5
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print()
    print("OK — reference image mapped.")
    print(f"  character:   {args.character_id}")
    print(f"  old path:    {old_value!r}")
    print(f"  new path:    {new_value!r}")
    print(f"  resolved to: {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

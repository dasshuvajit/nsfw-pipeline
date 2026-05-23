"""Re-export a finished series with its current selected set.

Reads the current selected image list + series metadata from the DB,
clears any stale ``NNN_`` prefixed export copies + manifest + metadata
in the destination, then runs ``Exporter.export()``. Preserves the
existing ``metadata.json`` title / description / tags when present
(LLM-generated copy from the original run isn't worth regenerating).

Usage:
    PYTHONPATH=. python scripts/reexport_series.py series_83453d626f45
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from src.export.exporter import Exporter

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "nsfw_pipeline.db"
OUTPUT_ROOT = ROOT / "output"
_DUP_PREFIX_RE = re.compile(r"^\d{3}_")


def _find_export_dir(series_id: str, content_level: str) -> Path:
    base = OUTPUT_ROOT / content_level / series_id
    matches = [p for p in base.glob("*/*") if p.is_dir()]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one llm_id/target_id under {base}; got {matches}"
        )
    return matches[0]


def _series_metadata(conn: sqlite3.Connection, series_id: str) -> dict:
    row = conn.execute(
        "SELECT content_level, mode, style_profile_id, llm_series_plan "
        "FROM series WHERE id = ?", (series_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Series not found: {series_id}")
    series_plan = json.loads(row["llm_series_plan"]) if row["llm_series_plan"] else {}
    return {
        "content_level": row["content_level"],
        "mode": row["mode"],
        "style_profile_id": row["style_profile_id"],
        "series_plan": series_plan,
    }


def _selected_images(conn: sqlite3.Connection, series_id: str) -> list[dict]:
    cur = conn.execute(
        """
        SELECT i.*, p.prompt_text, s.aspect_ratio
        FROM images i
        LEFT JOIN prompts p ON i.prompt_id = p.id
        LEFT JOIN scenes  s ON p.scene_id  = s.id
        WHERE i.series_id = ? AND i.selected = 1
        ORDER BY i.quality_score DESC
        """,
        (series_id,),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _load_existing_metadata(export_dir: Path) -> dict:
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _llm_id_from_export_dir(export_dir: Path) -> str:
    # output_root / content_level / series_id / llm_id / target_id
    # export_dir == .../<llm_id>/<target_id>
    return export_dir.parent.name


def _clear_stale_outputs(export_dir: Path) -> None:
    """Remove old NNN_-prefixed copies + manifest + metadata so the
    fresh export starts from a clean state. Raw renders (no prefix)
    are kept — those are the DB-tracked source files."""
    images_dir = export_dir / "images"
    removed = 0
    if images_dir.is_dir():
        for p in images_dir.iterdir():
            if p.is_file() and _DUP_PREFIX_RE.match(p.name):
                p.unlink()
                removed += 1
    for stale in ("metadata.json", "manifest.json"):
        sp = export_dir / stale
        if sp.exists():
            sp.unlink()
    print(f"Cleared {removed} prefixed copies + manifest/metadata")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series_id")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    sm = _series_metadata(conn, args.series_id)
    content_level = sm["content_level"]

    images = _selected_images(conn, args.series_id)
    if not images:
        raise SystemExit(f"No selected images for {args.series_id}")
    print(f"Series: {args.series_id}")
    print(f"Content level: {content_level}")
    print(f"Selected images: {len(images)}")
    print(f"Quality range: "
          f"{images[-1]['quality_score']:.3f} → {images[0]['quality_score']:.3f}\n")

    export_dir = _find_export_dir(args.series_id, content_level)
    print(f"Export dir: {export_dir}")
    llm_id = _llm_id_from_export_dir(export_dir)
    model_id = images[0]["model_id"]
    print(f"LLM: {llm_id}\nModel: {model_id}\n")

    # Preserve existing LLM-generated metadata when present.
    existing_meta = _load_existing_metadata(export_dir)
    metadata = {
        "title": existing_meta.get("title", ""),
        "description": existing_meta.get("description", ""),
        "tags": existing_meta.get("tags", []),
    }

    _clear_stale_outputs(export_dir)

    exporter = Exporter(output_root=OUTPUT_ROOT)
    out_dir = exporter.export(
        series_id=args.series_id,
        content_level=content_level,
        images=images,
        metadata=metadata,
        series_plan=sm["series_plan"],
        model_id=model_id,
        style_profile_id=sm["style_profile_id"],
        llm_id=llm_id,
        target_id=model_id,
        target_kind="model",
    )
    print(f"\nExport written: {out_dir}")
    n_prefixed = len(list((out_dir / 'images').glob('[0-9][0-9][0-9]_*')))
    print(f"Images written: {n_prefixed}")


if __name__ == "__main__":
    main()

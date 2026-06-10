"""Re-run the set builder on a completed series with the NSFW-aware scorer.

For a series whose raw renders are still on disk, this:
  1. Re-scores every raw render with ``content_level`` passed through
     to ``ImageScorer.score()`` (triggers the NSFW path at T3/T4).
  2. Runs ``SetBuilder.build()`` against the new scores.
  3. UPSERTs every scored image into the ``images`` table (existing
     rows get refreshed scores; previously-dropped renders get
     freshly inserted).
  4. Flips ``selected`` to match the SetBuilder result.
  5. Prints a before/after delta.

Usage:
    PYTHONPATH=. python scripts/rerun_set_builder.py series_83453d626f45
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from pathlib import Path

import yaml

from src.filter.set_builder import SetBuilder, SetTooSmall
from src.scoring.image_scorer import ImageScorer

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "nsfw_pipeline.db"
CONFIG = ROOT / "config" / "pipeline.yaml"

# Filename → prompt_NNN
_PROMPT_RE = re.compile(r"(prompt_\d{3})")
# Filename → leading NNN_ duplication prefix
_DUP_PREFIX_RE = re.compile(r"^\d{3}_")


def _series_root(series_id: str, content_level: str) -> Path:
    return ROOT / "output" / content_level / series_id


def _find_images_dir(series_id: str, content_level: str) -> Path:
    root = _series_root(series_id, content_level)
    matches = list(root.glob("*/*/images"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one images/ under {root}; got {matches}"
        )
    return matches[0]


def _unique_raw_renders(img_dir: Path) -> list[Path]:
    """Return one Path per unique render.

    The exporter writes an extra ``NNN_<orig>`` copy alongside the raw
    render. The original (un-prefixed) is what the DB stores, so always
    prefer it when both exist.
    """
    by_base: dict[str, list[Path]] = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() != ".png":
            continue
        base = _DUP_PREFIX_RE.sub("", p.name)
        by_base.setdefault(base, []).append(p)
    result: list[Path] = []
    for base, candidates in by_base.items():
        # Prefer the un-prefixed file when present.
        unprefixed = [c for c in candidates if c.name == base]
        result.append(unprefixed[0] if unprefixed else candidates[0])
    return sorted(result, key=lambda x: x.name)


def _prompt_id_for(path: Path, series_id: str) -> str | None:
    m = _PROMPT_RE.search(path.name)
    if not m:
        return None
    # The full prompt_id is the series-specific prefix + prompt_NNN.
    # We can match it back from the DB rather than reconstructing it.
    return m.group(1)


def _resolve_prompt_id(conn: sqlite3.Connection, series_id: str, suffix: str) -> str | None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT id, prompt_text FROM prompts WHERE id LIKE ? AND id LIKE ?",
        (f"%{suffix}", f"{series_id}%"),
    ).fetchone()
    return row[0] if row else None


def _prompt_text(conn: sqlite3.Connection, prompt_id: str) -> str | None:
    row = conn.execute(
        "SELECT prompt_text FROM prompts WHERE id = ?", (prompt_id,)
    ).fetchone()
    return row[0] if row else None


def _series_row(conn: sqlite3.Connection, series_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT content_level, status FROM series WHERE id = ?",
        (series_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Series not found: {series_id}")
    return row


def _existing_images(conn: sqlite3.Connection, series_id: str) -> dict[str, dict]:
    """Map file_path → existing image row dict."""
    cur = conn.execute(
        "SELECT * FROM images WHERE series_id = ?", (series_id,),
    )
    out = {}
    cols = [c[0] for c in cur.description]
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        out[d["file_path"]] = d
    return out


def _load_quality_cutoff() -> float:
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    return float(cfg.get("set_builder", {}).get("quality_cutoff", 0.55))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series_id")
    ap.add_argument("--apply", action="store_true",
                    help="Persist changes to the DB. Without this flag the "
                         "script prints what WOULD change but writes nothing.")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    series = _series_row(conn, args.series_id)
    content_level = series["content_level"]
    print(f"Series: {args.series_id}")
    print(f"Content level: {content_level}")
    print(f"Status: {series['status']}")

    img_dir = _find_images_dir(args.series_id, content_level)
    renders = _unique_raw_renders(img_dir)
    print(f"Raw renders on disk: {len(renders)}")

    existing = _existing_images(conn, args.series_id)
    print(f"Existing DB rows: {len(existing)}\n")

    quality_cutoff = _load_quality_cutoff()
    scorer = ImageScorer(use_hps_v2=False, use_image_reward=False)

    # Score every raw render with NSFW mode.
    scored: list[dict] = []
    for path in renders:
        suffix = _PROMPT_RE.search(path.name)
        if not suffix:
            print(f"  WARN: skipping unrecognised file {path.name}")
            continue
        prompt_id = _resolve_prompt_id(conn, args.series_id, suffix.group(1))
        if not prompt_id:
            print(f"  WARN: no prompt_id match for {path.name}")
            continue
        prompt_text = _prompt_text(conn, prompt_id)
        result = scorer.score(
            path, prompt=prompt_text, content_level=content_level,
        )

        existing_row = existing.get(str(path))
        img_id = existing_row["id"] if existing_row else uuid.uuid4().hex
        model_id = (
            existing_row["model_id"] if existing_row else "gonzalomo_chroma_v30"
        )
        width = existing_row["width"] if existing_row else 0
        height = existing_row["height"] if existing_row else 0
        seed = existing_row["seed"] if existing_row else 0

        scored.append({
            "id": img_id,
            "prompt_id": prompt_id,
            "file_path": str(path),
            "model_id": model_id,
            "width": width,
            "height": height,
            "seed": seed,
            "content_level": content_level,
            "quality_score": result["composite"],
            "aesthetic_score": result["aesthetic"],
            "blur_score": result["blur"],
            "face_confidence": result["face_confidence"],
            "hps_v2_score": result["hps_v2"],
            "image_reward_score": result["image_reward"],
            "quality_flags": result["flags"],
            "was_in_db": existing_row is not None,
            "was_selected": existing_row["selected"] if existing_row else 0,
        })

    print(f"Scored: {len(scored)} images\n")

    # Run SetBuilder on the rescored set.
    set_builder = SetBuilder(
        min_images=10, max_images=25, quality_cutoff=quality_cutoff,
    )
    try:
        selected = set_builder.build(scored, content_level)
    except SetTooSmall as exc:
        print(f"SetBuilder failed: {exc}")
        sys.exit(1)

    selected_ids = {img["id"] for img in selected}
    print(f"SetBuilder selected: {len(selected)}/{len(scored)} (cutoff {quality_cutoff})")
    print(f"Quality range: {selected[-1]['quality_score']:.3f} → {selected[0]['quality_score']:.3f}\n")

    # Delta report.
    newly_selected = [s for s in selected if not s["was_selected"]]
    newly_dropped = [s for s in scored if s["was_selected"] and s["id"] not in selected_ids]
    fresh_inserts = [s for s in selected if not s["was_in_db"]]

    print(f"Newly selected (was dropped or not in DB): {len(newly_selected)}")
    for s in newly_selected:
        print(
            f"  + {s['quality_score']:.3f}  "
            f"{Path(s['file_path']).name[-50:]}"
        )
    print(f"\nNewly dropped (was selected): {len(newly_dropped)}")
    for s in newly_dropped:
        print(
            f"  - {s['quality_score']:.3f}  "
            f"{Path(s['file_path']).name[-50:]}"
        )
    print(f"\nFresh DB inserts (renders not previously persisted): {len(fresh_inserts)}")

    if not args.apply:
        print("\n[dry-run] No DB changes written. Re-run with --apply to persist.")
        return

    # Write changes.
    cur = conn.cursor()
    for s in scored:
        sel = 1 if s["id"] in selected_ids else 0
        if s["was_in_db"]:
            cur.execute(
                """
                UPDATE images SET
                    quality_score = ?,
                    aesthetic_score = ?,
                    blur_score = ?,
                    face_confidence = ?,
                    hps_v2_score = ?,
                    image_reward_score = ?,
                    quality_flags = ?,
                    selected = ?
                WHERE id = ?
                """,
                (
                    s["quality_score"], s["aesthetic_score"], s["blur_score"],
                    s["face_confidence"], s["hps_v2_score"],
                    s["image_reward_score"], json.dumps(s["quality_flags"]),
                    sel, s["id"],
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO images (
                    id, prompt_id, series_id, model_id,
                    file_path, width, height, seed,
                    content_level, quality_score, aesthetic_score,
                    blur_score, face_confidence, hps_v2_score,
                    image_reward_score, quality_flags, selected
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    s["id"], s["prompt_id"], args.series_id, s["model_id"],
                    s["file_path"], s["width"], s["height"], s["seed"],
                    content_level, s["quality_score"], s["aesthetic_score"],
                    s["blur_score"], s["face_confidence"], s["hps_v2_score"],
                    s["image_reward_score"], json.dumps(s["quality_flags"]),
                    sel,
                ),
            )
    # Refresh actual_count on the series row.
    cur.execute(
        """
        UPDATE series SET actual_count = (
          SELECT COUNT(*) FROM images WHERE series_id = ? AND selected = 1
        ) WHERE id = ?
        """,
        (args.series_id, args.series_id),
    )
    conn.commit()
    print("\n[applied] DB updated.")


if __name__ == "__main__":
    main()

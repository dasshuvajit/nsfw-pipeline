"""Render a specific subset of scenes from an existing series.

Wraps ``engine.run_phase_b`` with an explicit ``scene_ids`` list — the
official ``render_prompts.py`` CLI only takes a single ``--scene-id``,
which is too narrow when rerunning a handful of scenes after a
selective prompt regen.

Skips Phase C (set build + export). The caller is expected to follow
up with ``rerun_set_builder.py`` + ``reexport_series.py``.

Usage:
    PYTHONPATH=. python scripts/render_scenes.py \\
        --series-id series_83453d626f45 \\
        --model gonzalomo_chroma_v30 \\
        --llm davidau_nemo_thinking_heretic_claude_opus \\
        --scene-ids series_83453d626f45_scene_006,...
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.engine import (  # noqa: E402
    EngineError,
    PipelineEngine,
    _load_config,
)


DB = PROJECT_ROOT / "nsfw_pipeline.db"


def _delete_image_rows(scene_ids: list[str], series_id: str) -> int:
    """Remove the existing image rows for these scenes — the new
    renders will INSERT fresh rows with different file_paths.
    """
    conn = sqlite3.connect(str(DB))
    try:
        placeholders = ",".join("?" * len(scene_ids))
        cur = conn.execute(
            f"DELETE FROM images WHERE series_id = ? "
            f"AND prompt_id IN ("
            f"  SELECT id FROM prompts WHERE scene_id IN ({placeholders})"
            f")",
            (series_id, *scene_ids),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def _persist_rendered(images: list[dict], series_id: str, content_level: str) -> int:
    """Insert the new image rows. Marked selected=1 by default — the
    follow-up rerun_set_builder pass re-decides selection based on
    fresh quality scores.
    """
    conn = sqlite3.connect(str(DB))
    try:
        for img in images:
            conn.execute(
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
                    img.get("id") or uuid.uuid4().hex,
                    img.get("prompt_id"),
                    series_id,
                    img.get("model_id"),
                    img["file_path"],
                    img.get("width", 0),
                    img.get("height", 0),
                    img.get("seed", 0),
                    content_level,
                    img.get("quality_score"),
                    img.get("aesthetic_score"),
                    img.get("blur_score"),
                    img.get("face_confidence"),
                    img.get("hps_v2_score"),
                    img.get("image_reward_score"),
                    json.dumps(img.get("quality_flags", [])),
                    1,
                ),
            )
        conn.commit()
        return len(images)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--scene-ids", required=True,
                    help="Comma-separated scene_ids to render")
    ap.add_argument("--keep-old-images", action="store_true",
                    help="Don't DELETE the existing image rows for "
                         "these scenes before rendering. By default "
                         "we remove them so set_builder picks the "
                         "fresh renders without ambiguity.")
    args = ap.parse_args()

    scene_ids = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
    if not scene_ids:
        print("ERROR: --scene-ids empty", file=sys.stderr)
        return 2

    print(f"Series: {args.series_id}")
    print(f"Model:  {args.model}")
    print(f"LLM:    {args.llm}")
    print(f"Scenes ({len(scene_ids)}):")
    for sid in scene_ids:
        print(f"   {sid}")
    print()

    if not args.keep_old_images:
        n = _delete_image_rows(scene_ids, args.series_id)
        print(f"Removed {n} existing image rows for these scenes\n")

    config = _load_config()
    engine = PipelineEngine(config=config, dry_run=False)

    try:
        rendered_images, _ctx_state = engine.run_phase_b(
            series_id=args.series_id,
            model_id=args.model,
            scene_ids=scene_ids,
            cli_llm_override=args.llm,
            target_kind="model",
        )
    except EngineError as exc:
        print(f"\nERROR: run_phase_b failed: {exc}", file=sys.stderr)
        return 3

    print(f"\nRendered {len(rendered_images)} images.")

    # run_phase_b internally scores via ImageScorer (with the NSFW
    # mode fix from earlier), so the scores are already in each dict.
    # It does NOT persist — that's Phase C's job; we do it minimally
    # here so rerun_set_builder.py can see them.
    conn = sqlite3.connect(str(DB))
    content_level = conn.execute(
        "SELECT content_level FROM series WHERE id = ?", (args.series_id,),
    ).fetchone()[0]
    conn.close()
    n_saved = _persist_rendered(rendered_images, args.series_id, content_level)
    print(f"Persisted {n_saved} new image rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

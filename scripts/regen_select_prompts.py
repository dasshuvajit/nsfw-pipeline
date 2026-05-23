"""Surgical re-roll of specific prompts within an existing series.

Why this exists: ``prepare_prompts.py --regen-prompts <model>`` is
binary — it deletes ALL prompts for the target and re-rolls all
scenes. When only a handful of scenes need to be re-rolled (e.g.
tier-purity leaks from facet-generation failures), this script does
the same work on a per-scene subset.

How it works:
  1. DELETE the specified prompt rows and (optionally) their matching
     scene_facet rows.
  2. Monkey-patch ``PipelineEngine._load_series_for_retarget`` to
     return ONLY the target scene_ids.
  3. Call ``engine.run_phase_a`` — the engine's per-scene loop
     regenerates facets (or loads from DB), composes prompts, and
     INSERTs them. Same vocab_version stamping, diversity tracker,
     deduplicator, and prompt_hash invariants as the canonical CLI.

Usage:
    PYTHONPATH=. python scripts/regen_select_prompts.py \\
        --series-id series_83453d626f45 \\
        --model gonzalomo_chroma_v30 \\
        --llm davidau_nemo_thinking_heretic_claude_opus \\
        --scene-ids series_83453d626f45_scene_006,... \\
        --also-delete-facets
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.content_level import ContentLevelLoader  # noqa: E402
from src.core.engine import (  # noqa: E402
    EngineError,
    PipelineEngine,
    _load_config,
    _load_style_profile,
)
from src.core.generation_context import build_context  # noqa: E402

DB = PROJECT_ROOT / "nsfw_pipeline.db"


def _delete_prompts(scene_ids: list[str], target_id: str, llm_id: str) -> int:
    conn = sqlite3.connect(str(DB))
    try:
        placeholders = ",".join("?" * len(scene_ids))
        cur = conn.execute(
            f"DELETE FROM prompts WHERE scene_id IN ({placeholders}) "
            f"AND model_id = ? AND llm_id = ?",
            (*scene_ids, target_id, llm_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def _delete_facets(scene_ids: list[str], family_id: str, llm_id: str) -> int:
    conn = sqlite3.connect(str(DB))
    try:
        placeholders = ",".join("?" * len(scene_ids))
        cur = conn.execute(
            f"DELETE FROM scene_facets WHERE scene_id IN ({placeholders}) "
            f"AND family = ? AND llm_id = ?",
            (*scene_ids, family_id, llm_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def _resolve_family(model_id: str) -> str:
    from src.memory.model_registry import ModelRegistryLoader
    loader = ModelRegistryLoader(commercial_mode=False)
    entry = loader.get_model(model_id)
    return entry.family


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--scene-ids", required=True,
                    help="Comma-separated scene_ids to re-roll")
    ap.add_argument("--also-delete-facets", action="store_true")
    args = ap.parse_args()

    scene_ids = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
    if not scene_ids:
        print("ERROR: --scene-ids empty", file=sys.stderr)
        return 2

    family_id = _resolve_family(args.model)
    print(f"Series:  {args.series_id}")
    print(f"Model:   {args.model}  (family={family_id})")
    print(f"LLM:     {args.llm}")
    print(f"Scenes:  {len(scene_ids)}")
    for sid in scene_ids:
        print(f"   {sid}")
    print()

    n_p = _delete_prompts(scene_ids, args.model, args.llm)
    print(f"Deleted {n_p} prompt rows")
    if args.also_delete_facets:
        n_f = _delete_facets(scene_ids, family_id, args.llm)
        print(f"Deleted {n_f} scene_facets rows")
    print()

    # Resolve the series's content_level / style_profile / mode so we
    # can build a baseline GenerationContext.
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT content_level, style_profile_id, mode "
        "FROM series WHERE id = ?", (args.series_id,),
    ).fetchone()
    conn.close()
    if not row:
        print(f"ERROR: series {args.series_id} not found", file=sys.stderr)
        return 2
    content_level = row["content_level"]
    style_profile_id = row["style_profile_id"]
    mode_name = row["mode"]
    print(f"Content level: {content_level}")
    print(f"Style profile: {style_profile_id}")
    print(f"Mode:          {mode_name}\n")

    config = _load_config()
    engine = PipelineEngine(config=config, dry_run=False, force_mode=mode_name)
    style_profile = _load_style_profile(engine.db_path, style_profile_id)
    content_rules = ContentLevelLoader(engine.db_path).load(content_level)

    # Baseline ctx (model-kind). Per-target ctx is rebuilt inside
    # run_phase_a's per-target loop.
    ctx = build_context(
        mode=mode_name,
        content_level=content_level,
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=engine.db_path,
        model_id=args.model,
        commercial_mode=False,
    )

    # Monkey-patch: filter scenes for re-target to ONLY the target ids,
    # AND rewrite prompt IDs at save-time to honour the scene's original
    # position in the full series. Without this, the engine assigns
    # `_prompt_000..NNN` based on `len(prompts_for_target)` which
    # collides with the existing prompts on the untouched scenes.
    from src.core import engine as engine_module
    original_loader = engine_module.PipelineEngine._load_series_for_retarget
    original_save = engine_module.PipelineEngine._save_prompts_for_target

    # Resolve the scene_id → original-position map from the full
    # series, BEFORE we filter. Position is the index returned by
    # the original loader (matches how the original prepare run
    # numbered them).
    full_plan, full_scenes = original_loader(engine, args.series_id)
    scene_id_to_pos = {s["id"]: idx for idx, s in enumerate(full_scenes)}

    def filtered_loader(self, series_id):  # noqa: ANN001
        wanted = set(scene_ids)
        filtered = [s for s in full_scenes if s["id"] in wanted]
        if len(filtered) != len(wanted):
            missing = sorted(wanted - {s["id"] for s in filtered})
            raise RuntimeError(f"scene_ids missing from series: {missing}")
        return full_plan, filtered

    def patched_save(self, *, series_id, ctx, prompts, llm_id, target_kind="model"):  # noqa: ANN001
        # Rewrite prompt[id] to match the scene's position in the
        # full series, not the position in the filtered list.
        prefix_re = "_prompt_"
        for p in prompts:
            sid = p.get("scene_id")
            if sid is None or sid not in scene_id_to_pos:
                continue
            pos = scene_id_to_pos[sid]
            old_id = p["id"]
            head, _, _tail = old_id.rpartition(prefix_re)
            new_id = f"{head}{prefix_re}{pos:03d}"
            if new_id != old_id:
                p["id"] = new_id
        return original_save(
            self, series_id=series_id, ctx=ctx, prompts=prompts,
            llm_id=llm_id, target_kind=target_kind,
        )

    engine_module.PipelineEngine._load_series_for_retarget = filtered_loader
    engine_module.PipelineEngine._save_prompts_for_target = patched_save

    try:
        result = engine.run_phase_a(
            ctx,
            models=[args.model],
            families=None,
            series_id_existing=args.series_id,
            regen_facets=None,
            regen_prompts=None,
            regen_family_prompts=None,
            style_profile=style_profile,
            style_profile_id=style_profile_id,
            cli_llm_override=args.llm,
        )
    except EngineError as exc:
        print(f"\nERROR: engine failed: {exc}", file=sys.stderr)
        return 3
    finally:
        engine_module.PipelineEngine._load_series_for_retarget = original_loader
        engine_module.PipelineEngine._save_prompts_for_target = original_save

    print(f"\nrun_phase_a result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

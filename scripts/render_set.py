#!/usr/bin/env python3
"""Manual Mode 1 (Character) set renderer — Phase 1.

Per ARCHITECTURE.md Sections 7 + 22 step 13, this is the first script
that exercises the full Phase 1 path end-to-end:

    DB character → 15 hand-written scenes → deterministic prompts →
    content-level sanitizer → per-scene aspect ratio → ComfyUI render →
    image scorer → on-disk export under output/{tier}/{series_id}/

It is **manual**: no LLM, no scheduler. IPAdapter is opt-in via
``--reference-image``: when provided, the renderer switches from the
``sdxl/base.json`` template to ``sdxl/ipadapter.json`` and locks every
render to the reference face/identity. The 15 scenes are hardcoded as
Python lists below — different poses, cameras, and lighting combos that
span the allowed_pose_types for each content level. They're meant to be
edited.

What this does, in order:

  1. Loads the character row from the SQLite DB (default: nsfw_pipeline.db).
  2. Loads the linked style_profile row.
  3. Loads ContentLevelRules for the requested tier (T1_suggestive | T2_implied
     | T3_artnude | T4_explicit).
  4. Picks the right pool of hand-written scenes for that level.
  5. Trims/repeats them to match --count.
  6. Builds prompts via PromptBuilder (deterministic comma-joined template).
  7. Sanitizes each prompt via PromptSanitizer (content-level boost/suppress).
  8. Picks an aspect ratio per scene via RatioSelector.
  9. Renders each prompt via ComfyUI (per-image retry from ComfyUIClient).
 10. Copies the rendered file into output/{level}/{series_id}/images/.
 11. Scores each saved image via ImageScorer (lazy-loads CLIP+insightface
     once at the end of the render loop, scores in batch).
 12. Persists series + scenes + prompts + images rows into the DB.
 13. Prints a summary: counts, render+score timing, quality histogram.

`--dry-run` short-circuits at step 8: prints every prompt + ratio it
*would* render and exits without touching ComfyUI or the DB. Use this to
inspect prompts before burning render time.

Pre-flight:
  - `python scripts/init_db.py` has been run.
  - `python scripts/bootstrap_character.py characters/<id>/identity.json`
    has populated the character.
  - `config/comfyui_workflows/sdxl/base.json` exists AND has
    `lora_loader_0` + `lora_loader_1` nodes (see PROJECT_GUIDE.md
    "Known Issues" — the seeded `realistic_soft` profile has 2 LoRAs).
  - For IPAdapter renders: `config/comfyui_workflows/sdxl/ipadapter.json`
    must also exist (see PROJECT_GUIDE.md → "Building the SDXL
    ipadapter.json workflow template").
  - ComfyUI is running at the URL in pipeline.yaml.
  - The aesthetic MLP weights are in models/aesthetic/.

Examples:
    python scripts/render_set.py --character char_001 --theme "studio portrait" \\
        --level T2_implied --count 15
    python scripts/render_set.py --character char_001 --level T3_artnude --count 5
    python scripts/render_set.py --character char_001 --level T2_implied --dry-run
    # IPAdapter — lock every render to a reference face
    python scripts/render_set.py --character char_001 --level T2_implied \\
        --reference-image characters/char_001/reference.png \\
        --ipadapter-weight 0.75
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Make src/ importable when running this file as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.content_level import (  # noqa: E402
    ContentLevelLoader,
    UnknownContentLevel,
)
from src.core.ratio_selector import (  # noqa: E402
    RATIO_TO_RESOLUTION,
    RatioPick,
    RatioSelector,
    model_resolution_overrides,
)
from src.memory.character_manager import (  # noqa: E402
    CharacterManager,
    CharacterNotFound,
)
from src.core.style_profile_adapter import (  # noqa: E402
    StyleProfileForWorkflow,
)
from src.memory.model_registry import (  # noqa: E402
    ModelNotFound,
    ModelRegistryEntry,
    ModelRegistryLoader,
)
from src.prompt.builder import (  # noqa: E402
    PromptBuilder,
    compute_prompt_hash,
)
from src.prompt.sanitizer import PromptSanitizer  # noqa: E402
from src.render.comfyui_client import (  # noqa: E402
    ComfyUIClient,
    ComfyUIError,
)
from src.render.workflow_builder import (  # noqa: E402
    WorkflowBuilder,
    WorkflowTemplateError,
    _REQUIRED_NODES_EXTERNAL,
    _assert_external_template_inputs,
    _resolve_template_path,
)
from src.scoring.image_scorer import (  # noqa: E402
    ImageScorer,
    ScorerError,
    ScorerModelError,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
WORKFLOW_DIR = PROJECT_ROOT / "config" / "comfyui_workflows"


# ─── HARDCODED SCENES ───────────────────────────────────────────────────────
#
# These are the Phase 1 hand-written scene baselines. Edit freely. Each
# scene is a dict of the same fields the `scenes` table stores; empty
# fields are dropped at prompt-build time. Two pools — one per content
# level — because the allowed pose vocabulary is different (Section 5).
#
# Goals when picking these:
#   - Cover all four aspect-ratio buckets at least once per pool.
#   - Span the allowed_pose_types from `content_level_rules` (so the
#     scene constraint enforcer in a future iteration would no-op).
#   - Vary lighting + environment so the IPAdapter consistency test
#     (next script) has something interesting to lock onto.
# ────────────────────────────────────────────────────────────────────────────

SOFT_SCENES: list[dict[str, str]] = [
    {
        "pose": "standing relaxed", "camera": "full body",
        "camera_angle": "eye-level", "lighting": "soft window light",
        "environment_detail": "minimalist studio backdrop",
        "expression": "soft smile", "mood_note": "elegant",
    },
    {
        "pose": "sitting elegantly", "camera": "medium shot",
        "camera_angle": "slight high angle", "lighting": "warm golden hour",
        "environment_detail": "wooden chair near a tall window",
        "expression": "calm gaze", "mood_note": "dreamy",
    },
    {
        "pose": "leaning gently", "camera": "three-quarter",
        "camera_angle": "eye-level", "lighting": "soft diffused",
        "environment_detail": "white-painted brick wall",
        "expression": "gentle", "mood_note": "aesthetic",
    },
    {
        "pose": "looking over shoulder", "camera": "close-up portrait",
        "camera_angle": "slight low angle", "lighting": "rim light",
        "environment_detail": "neutral studio gradient",
        "expression": "soft smile", "mood_note": "playful",
    },
    {
        "pose": "walking gracefully", "camera": "full body",
        "camera_angle": "tracking eye-level", "lighting": "natural daylight",
        "environment_detail": "quiet city sidewalk",
        "expression": "calm gaze", "mood_note": "elegant",
    },
    {
        "pose": "reclining softly", "camera": "wide shot",
        "camera_angle": "eye-level", "lighting": "warm lamp glow",
        "environment_detail": "linen-covered chaise lounge",
        "expression": "dreamy", "mood_note": "soft",
    },
    {
        "pose": "hand on hip", "camera": "upper body",
        "camera_angle": "eye-level", "lighting": "studio beauty light",
        "environment_detail": "seamless paper backdrop",
        "expression": "confident", "mood_note": "aesthetic",
    },
    {
        "pose": "arms crossed", "camera": "medium shot",
        "camera_angle": "eye-level", "lighting": "soft overcast",
        "environment_detail": "garden archway",
        "expression": "gentle", "mood_note": "elegant",
    },
    {
        "pose": "chin on hand", "camera": "close-up face",
        "camera_angle": "slight high angle", "lighting": "window soft light",
        "environment_detail": "wooden cafe table",
        "expression": "dreamy", "mood_note": "soft",
    },
    {
        "pose": "profile pose", "camera": "side profile",
        "camera_angle": "eye-level", "lighting": "single key light",
        "environment_detail": "dark gradient backdrop",
        "expression": "calm gaze", "mood_note": "aesthetic",
    },
    {
        "pose": "standing relaxed", "camera": "full body",
        "camera_angle": "low angle", "lighting": "backlit silhouette edge",
        "environment_detail": "sunset terrace",
        "expression": "soft smile", "mood_note": "dreamy",
    },
    {
        "pose": "sitting elegantly", "camera": "three-quarter",
        "camera_angle": "eye-level", "lighting": "soft diffused window",
        "environment_detail": "art gallery bench",
        "expression": "gentle", "mood_note": "elegant",
    },
    {
        "pose": "leaning gently", "camera": "medium shot",
        "camera_angle": "eye-level", "lighting": "ambient indoor",
        "environment_detail": "library bookshelf",
        "expression": "calm gaze", "mood_note": "playful",
    },
    {
        "pose": "looking over shoulder", "camera": "upper body",
        "camera_angle": "slight high angle", "lighting": "natural reflected light",
        "environment_detail": "white linen drapes",
        "expression": "soft smile", "mood_note": "soft",
    },
    {
        "pose": "walking gracefully", "camera": "full body",
        "camera_angle": "eye-level", "lighting": "morning haze",
        "environment_detail": "tree-lined avenue",
        "expression": "dreamy", "mood_note": "aesthetic",
    },
]

FULL_SCENES: list[dict[str, str]] = [
    {
        "pose": "standing confident", "camera": "full body",
        "camera_angle": "low angle", "lighting": "dramatic side key",
        "environment_detail": "dark cinematic backdrop",
        "expression": "intense gaze", "mood_note": "bold",
    },
    {
        "pose": "sitting bold", "camera": "medium shot",
        "camera_angle": "eye-level", "lighting": "warm rim light",
        "environment_detail": "velvet armchair",
        "expression": "provocative smile", "mood_note": "sensual",
    },
    {
        "pose": "reclining expressive", "camera": "wide shot",
        "camera_angle": "eye-level", "lighting": "moody low key",
        "environment_detail": "satin sheets on a low bed",
        "expression": "intense gaze", "mood_note": "intimate",
    },
    {
        "pose": "dynamic pose", "camera": "three-quarter",
        "camera_angle": "slight low angle", "lighting": "high contrast key",
        "environment_detail": "concrete loft wall",
        "expression": "bold expression", "mood_note": "intense",
    },
    {
        "pose": "arched back", "camera": "upper body",
        "camera_angle": "eye-level", "lighting": "sculpted side light",
        "environment_detail": "draped silk backdrop",
        "expression": "intense gaze", "mood_note": "passionate",
    },
    {
        "pose": "kneeling", "camera": "three-quarter",
        "camera_angle": "eye-level", "lighting": "warm tungsten",
        "environment_detail": "plush rug",
        "expression": "confident", "mood_note": "intimate",
    },
    {
        "pose": "stretching", "camera": "full body",
        "camera_angle": "eye-level", "lighting": "morning window beam",
        "environment_detail": "minimalist bedroom",
        "expression": "bold expression", "mood_note": "sensual",
    },
    {
        "pose": "turning away", "camera": "side profile",
        "camera_angle": "eye-level", "lighting": "single hard key",
        "environment_detail": "dark studio gradient",
        "expression": "intense gaze", "mood_note": "intimate",
    },
    {
        "pose": "looking back over shoulder", "camera": "close-up",
        "camera_angle": "slight low angle", "lighting": "rim accent",
        "environment_detail": "neutral cinematic backdrop",
        "expression": "provocative smile", "mood_note": "bold",
    },
    {
        "pose": "lounging", "camera": "wide shot",
        "camera_angle": "eye-level", "lighting": "warm bedroom lamp",
        "environment_detail": "linen-covered low platform",
        "expression": "intense gaze", "mood_note": "passionate",
    },
    {
        "pose": "standing confident", "camera": "full body",
        "camera_angle": "low angle", "lighting": "backlit silhouette",
        "environment_detail": "loft window at night",
        "expression": "bold expression", "mood_note": "intense",
    },
    {
        "pose": "sitting bold", "camera": "three-quarter",
        "camera_angle": "eye-level", "lighting": "moody side light",
        "environment_detail": "leather armchair",
        "expression": "provocative smile", "mood_note": "sensual",
    },
    {
        "pose": "reclining expressive", "camera": "medium shot",
        "camera_angle": "high angle", "lighting": "warm directional key",
        "environment_detail": "low velvet daybed",
        "expression": "intense gaze", "mood_note": "intimate",
    },
    {
        "pose": "arched back", "camera": "close-up",
        "camera_angle": "eye-level", "lighting": "single sharp key",
        "environment_detail": "dark backdrop with rim",
        "expression": "bold expression", "mood_note": "passionate",
    },
    {
        "pose": "kneeling", "camera": "upper body",
        "camera_angle": "slight high angle", "lighting": "warm soft fill",
        "environment_detail": "rug and pillows",
        "expression": "confident", "mood_note": "sensual",
    },
]


# ─── data classes ───────────────────────────────────────────────────────────


@dataclass
class _RenderRecord:
    """Per-scene bookkeeping that survives end-of-loop summarization."""

    index: int
    prompt_id: str
    scene_id: str
    scene: dict[str, Any]
    ratio: str
    resolution: tuple[int, int]
    prompt_text: str
    negative_prompt: str
    prompt_hash: str
    seed: int | None = None
    rendered_path: Path | None = None
    saved_path: Path | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None
    score: dict[str, Any] | None = None


# ─── config + DB helpers ────────────────────────────────────────────────────


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"ERROR: config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _resolve_db_path(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    raw = (cfg.get("pipeline") or {}).get("db_path") or "nsfw_pipeline.db"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _resolve_comfy_input_dir(cfg: dict[str, Any]) -> Path:
    """Pull comfyui.input_dir out of pipeline.yaml. Falls back to
    a sibling of comfyui.output_dir, since standard ComfyUI installs
    keep input/ and output/ next to each other.
    """
    comfy = cfg.get("comfyui") or {}
    raw = comfy.get("input_dir")
    if raw:
        return Path(raw).expanduser()
    raw_out = comfy.get("output_dir") or "~/AI/apps/ComfyUI/output"
    return Path(raw_out).expanduser().parent / "input"


def _stage_reference_image(reference_path: Path, comfy_input_dir: Path) -> str:
    """Copy a reference image into ComfyUI's input dir and return the
    basename that LoadImage can resolve.

    Idempotent: if the file is already inside comfy_input_dir we just
    return its name without re-copying. Mirrored on every render call so
    callers don't have to remember to pre-stage.
    """
    if not reference_path.exists():
        raise SystemExit(
            f"ERROR: --reference-image not found: {reference_path}"
        )
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    dest = comfy_input_dir / reference_path.name
    if dest.resolve() != reference_path.resolve():
        shutil.copy2(reference_path, dest)
    return dest.name


def _load_style_profile(db_path: Path, style_profile_id: str) -> dict[str, Any]:
    """Fetch a style profile from ``config/style_profiles.yaml``.

    ``db_path`` kept in the signature for caller parity — unused.
    Emits a flat dict shaped like the legacy ``style_profiles`` DB row
    so downstream adapters (``StyleProfileForWorkflow``) don't have to
    change.
    """
    from src.memory.style_profile_loader import (
        StyleProfileLoader, StyleProfileNotFound,
    )

    try:
        p = StyleProfileLoader().get_profile(style_profile_id)
    except StyleProfileNotFound:
        raise SystemExit(
            f"ERROR: style_profile {style_profile_id!r} not found in "
            f"config/style_profiles.yaml."
        )
    # Post-2026-04 refactor: profiles carry only aesthetic intent;
    # render tuning and lora_stack come from config/models/*.yaml.
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "base_style_keywords": p.base_style_keywords,
        "base_negative_prompt": p.base_negative_prompt,
        "palette_hint": p.palette_hint,
        "lighting_hint": p.lighting_hint,
        "suited_tiers": list(p.suited_tiers),
        "suited_families": list(p.suited_families),
        "lora_stack": None,
    }


def _resolve_model(
    db_path: Path,
    style_row: dict[str, Any],
    character_row: dict[str, Any],
    default_model_id: str,
    cli_override: str | None,
) -> ModelRegistryEntry:
    """Pick the model to render with.

    Precedence: ``--model`` CLI override → ``characters.model_id`` →
    ``pipeline.default_model_id``. The style profile only declares an
    aesthetic archetype now — it no longer binds a specific checkpoint.
    ``suited_families`` on the profile gates which model families are
    compatible; a mismatch exits 9.

    Exit codes:
        8 — ``--model`` id not found in registry or inactive
        9 — resolved model's family is not in the profile's
            ``suited_families`` list
    """
    loader = ModelRegistryLoader(db_path)

    resolved_id = (
        cli_override
        or character_row.get("model_id")
        or default_model_id
    )
    try:
        model = loader.get_model(resolved_id)
    except ModelNotFound as exc:
        # Exit 8: not found or inactive.
        raise SystemExit(
            f"EXIT 8: model {resolved_id!r} not found in registry or "
            f"inactive. {exc}"
        )

    suited = style_row.get("suited_families") or []
    if suited and model.family not in suited:
        raise SystemExit(
            f"EXIT 9: model {model.id!r} is in family {model.family!r} "
            f"but style_profile {style_row['id']!r} lists "
            f"suited_families={suited}. Pick a compatible model or "
            f"switch archetype."
        )
    return model


# ─── scene selection ────────────────────────────────────────────────────────


def _scenes_for_level(level: str, count: int) -> list[dict[str, Any]]:
    """Pick `count` scenes from the level pool, repeating if requested > pool.

    T1/T2 use the restrained (ex-SOFT) pool; T3/T4 the expressive (ex-FULL).
    """
    pool = SOFT_SCENES if level in ("T1_suggestive", "T2_implied") else FULL_SCENES
    if count <= 0:
        raise SystemExit(f"ERROR: --count must be > 0 (got {count})")
    if count <= len(pool):
        return [dict(s) for s in pool[:count]]
    # Repeat the pool to satisfy larger counts. Phase 1 doesn't need this
    # but we don't want to silently truncate either.
    out: list[dict[str, Any]] = []
    while len(out) < count:
        for scene in pool:
            out.append(dict(scene))
            if len(out) >= count:
                break
    return out


# ─── DB persistence ─────────────────────────────────────────────────────────


def _persist_run(
    db_path: Path,
    series_id: str,
    series_row: dict[str, Any],
    records: list[_RenderRecord],
) -> None:
    """Write series + scenes + prompts + images rows in one transaction."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO series (
                id, mode, content_level, character_id, style_profile_id,
                theme, mood, environment, variation_axes, target_count,
                actual_count, base_seed, status, llm_series_plan
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                series_id,
                series_row["mode"],
                series_row["content_level"],
                series_row["character_id"],
                series_row["style_profile_id"],
                series_row["theme"],
                series_row.get("mood"),
                series_row.get("environment"),
                None,                              # variation_axes
                series_row["target_count"],
                series_row["actual_count"],
                None,                              # base_seed (random per image)
                series_row["status"],
                None,                              # llm_series_plan
            ),
        )

        for rec in records:
            conn.execute(
                """
                INSERT INTO scenes (
                    id, series_id, variation_axis,
                    pose, camera, camera_angle, lighting, environment_detail,
                    mood_note, expression, aspect_ratio, resolution_w,
                    resolution_h, content_level, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.scene_id,
                    series_id,
                    "manual",                       # variation_axis: hand-written
                    rec.scene.get("pose"),
                    rec.scene.get("camera"),
                    rec.scene.get("camera_angle"),
                    rec.scene.get("lighting"),
                    rec.scene.get("environment_detail"),
                    rec.scene.get("mood_note"),
                    rec.scene.get("expression"),
                    rec.ratio,
                    rec.resolution[0],
                    rec.resolution[1],
                    series_row["content_level"],
                    "active",
                ),
            )
            conn.execute(
                """
                INSERT INTO prompts (
                    id, series_id, scene_id, prompt_text, negative_prompt,
                    prompt_hash, content_level, status, render_attempts
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.prompt_id,
                    series_id,
                    rec.scene_id,
                    rec.prompt_text,
                    rec.negative_prompt,
                    rec.prompt_hash,
                    series_row["content_level"],
                    "rendered" if rec.saved_path else "failed",
                    1,
                ),
            )
            if rec.saved_path:
                image_id = uuid.uuid4().hex
                score = rec.score or {}
                conn.execute(
                    """
                    INSERT INTO images (
                        id, prompt_id, series_id, model_id,
                        file_path, width, height,
                        seed, content_level, quality_score, aesthetic_score,
                        blur_score, face_confidence, quality_flags, selected
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        image_id,
                        rec.prompt_id,
                        series_id,
                        series_row.get("model_id"),
                        str(rec.saved_path),
                        rec.width or rec.resolution[0],
                        rec.height or rec.resolution[1],
                        rec.seed or 0,
                        series_row["content_level"],
                        score.get("composite"),
                        score.get("aesthetic"),
                        score.get("blur"),
                        score.get("face_confidence"),
                        json.dumps(score.get("flags", [])),
                        0,
                    ),
                )

        conn.commit()
    finally:
        conn.close()


# ─── summary ────────────────────────────────────────────────────────────────


def _print_summary(
    series_id: str,
    output_dir: Path,
    records: list[_RenderRecord],
    render_seconds: float,
    score_seconds: float,
) -> None:
    rendered = [r for r in records if r.saved_path]
    failed = [r for r in records if r.error]
    scored = [r for r in records if r.score]

    print()
    print("─" * 60)
    print(f"Series: {series_id}")
    print(f"Output: {output_dir}")
    print("─" * 60)
    print(f"  scenes total:    {len(records)}")
    print(f"  rendered:        {len(rendered)}")
    print(f"  failed:          {len(failed)}")
    print(f"  scored:          {len(scored)}")
    print(f"  render time:     {render_seconds:.1f}s")
    print(f"  score time:      {score_seconds:.1f}s")

    if rendered:
        avg_render = render_seconds / len(rendered)
        print(f"  avg render/img:  {avg_render:.1f}s")

    ratio_counts = Counter(r.ratio for r in records)
    print("\nAspect ratios picked:")
    for ratio, count in ratio_counts.most_common():
        print(f"  {ratio:<14} {count}")

    if scored:
        composites = [r.score["composite"] for r in scored]
        avg = sum(composites) / len(composites)
        buckets = Counter()
        for c in composites:
            if c >= 0.70:
                buckets["excellent (>=0.70)"] += 1
            elif c >= 0.55:
                buckets["passing  (>=0.55)"] += 1
            elif c >= 0.40:
                buckets["weak     (>=0.40)"] += 1
            else:
                buckets["bad      (<0.40)"] += 1
        print(f"\nQuality distribution (composite, avg={avg:.3f}):")
        for label in (
            "excellent (>=0.70)",
            "passing  (>=0.55)",
            "weak     (>=0.40)",
            "bad      (<0.40)",
        ):
            print(f"  {label}    {buckets.get(label, 0)}")

        flag_counts: Counter[str] = Counter()
        for r in scored:
            for flag in r.score.get("flags", []):
                flag_counts[flag] += 1
        if flag_counts:
            print("\nQuality flags:")
            for flag, count in flag_counts.most_common():
                print(f"  {flag:<18} {count}")

    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  scene {r.index}: {r.error}")


# ─── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--character", required=True, help="Character id (e.g. char_001)")
    parser.add_argument(
        "--theme",
        default="manual baseline",
        help="Free-form theme label written to series.theme. Default: %(default)s",
    )
    parser.add_argument(
        "--level",
        choices=("T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"),
        required=True,
        help="Content tier. Picks the matching scene pool + sanitizer rules. "
             "T1/T2 use the 'restrained' scene pool; T3/T4 use 'expressive'.",
    )
    parser.add_argument(
        "--count", type=int, default=15, help="Number of scenes to render. Default: %(default)s"
    )
    parser.add_argument(
        "--db-path", default=None, help="Override DB path. Default: pipeline.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts + ratios and print them, but do not call ComfyUI "
        "or write to disk/DB. Use to inspect prompts before burning render time.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip the ImageScorer pass (useful when iterating on render speed).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-image render timeout (seconds). Default: pipeline.yaml or 300",
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Path to a reference image. When provided, every render uses "
        "the IPAdapter workflow (sdxl/ipadapter.json) and locks identity "
        "to this face. Without this flag, plain sdxl/base.json is used.",
    )
    parser.add_argument(
        "--ipadapter-weight",
        type=float,
        default=0.7,
        help="IPAdapter weight (0.0-1.0). Higher = stronger reference lock, "
        "lower = more prompt freedom. Only used with --reference-image. "
        "Default: %(default)s (Section 7 sweet spot).",
    )
    parser.add_argument(
        "--ratio",
        choices=(
            "portrait_23", "portrait_916", "square", "landscape",
            "auto", "auto:deviantart", "auto:patreon",
        ),
        default=None,
        help="Aspect-ratio behaviour. One of the 4 explicit names forces "
        "that single ratio for every scene (bypasses the selector). "
        "'auto' / 'auto:deviantart' / 'auto:patreon' run the multi-axis "
        "RatioSelector with the matching audience bonus (or none for plain "
        "'auto'). Default: same as 'auto' (no audience bonus). "
        "See config/ratio_signals.yaml for the bonus tables.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the checkpoint used for this render. Takes a "
        "model id from config/models/ (e.g. 'juggernaut_ragnarok' — see "
        "`python scripts/list_models.py`). When set, the model YAML's "
        "default_sampler/scheduler/steps/cfg override the style "
        "profile's (because those numbers were measured for this "
        "checkpoint). Family mismatches (e.g. a flux model against "
        "an sdxl style profile) exit 9.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Path to an external ComfyUI workflow template "
        "(e.g. templates/chroma/chroma_done_properly.json). The "
        "template's baked-in checkpoint / LoRAs / sampler all run "
        "as authored; the pipeline injects only positive/negative "
        "prompt, seed, and resolution. IPAdapter is forced off "
        "(--reference-image is ignored with an INFO log). "
        "See docs/COMFYUI_WORKFLOWS.md § External templates.",
    )
    args = parser.parse_args()

    cfg = _load_config()
    db_path = _resolve_db_path(cfg, args.db_path)
    if not db_path.exists():
        print(
            f"ERROR: database not found at {db_path}.\n"
            f"Run `python scripts/init_db.py` first.",
            file=sys.stderr,
        )
        return 1

    # ----- 1. Character + style profile + content rules ---------------------
    try:
        character = CharacterManager(db_path).get_character(args.character)
    except CharacterNotFound as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    style_row = _load_style_profile(db_path, character["style_profile_id"])

    # Resolve the model to render with. Precedence: --model CLI override →
    # characters.model_id → pipeline.default_model_id. The capability
    # flags (supports_ipadapter, supports_lora) on the model YAML drive
    # the WorkflowBuilder guards that fire just below.
    default_model_id = (cfg.get("pipeline") or {}).get("default_model_id")
    if not default_model_id:
        print(
            "ERROR: pipeline.default_model_id is not set in "
            "config/pipeline.yaml.",
            file=sys.stderr,
        )
        return 1
    try:
        model_entry = _resolve_model(
            db_path, style_row, character, default_model_id, args.model
        )
    except SystemExit as exc:
        msg = str(exc)
        print(msg, file=sys.stderr)
        if msg.startswith("EXIT 8"):
            return 8
        if msg.startswith("EXIT 9"):
            return 9
        return 10
    style_for_wf = StyleProfileForWorkflow(style_row, model_entry)
    registry = ModelRegistryLoader(db_path)
    prompt_guide = registry.get_prompt_guide(model_entry.id)
    family = registry.get_family(model_entry.family)
    model_res = model_resolution_overrides(model_entry)

    # --template forces IPAdapter off (external templates own their
    # own graph; the pipeline's IPAdapter chain doesn't apply). Drop
    # --reference-image with an INFO log if both are set.
    if args.template and args.reference_image:
        print(
            f"INFO: --template is set, so --reference-image "
            f"{args.reference_image!r} is ignored (IPAdapter is disabled "
            f"for external templates).",
        )
        args.reference_image = None

    # Validate the external template BEFORE the render loop so a
    # malformed template fails fast instead of after minutes of setup.
    if args.template:
        try:
            _probe_builder = WorkflowBuilder(WORKFLOW_DIR)
            _probe_resolved = _resolve_template_path(
                args.template, _probe_builder.workflow_dir, PROJECT_ROOT,
            )
            _probe_workflow = _probe_builder._load(_probe_resolved)
            WorkflowBuilder._assert_required_nodes(
                _probe_workflow, "external", Path(_probe_resolved).name,
                _REQUIRED_NODES_EXTERNAL,
            )
            _assert_external_template_inputs(
                _probe_workflow, Path(_probe_resolved).name,
            )
        except WorkflowTemplateError as exc:
            print(f"ERROR: --template validation failed: {exc}", file=sys.stderr)
            return 9

    # Early fail: the pre-render family/capability guards that catch
    # the "caller passed --reference-image AND --model no-ipa-model"
    # case, and the "LoRA-bearing profile against a no-LoRA model"
    # case. These mirror WorkflowBuilder's internal guards but fire
    # before we spin up the render loop.
    if args.reference_image and not model_entry.supports_ipadapter:
        print(
            f"EXIT 9: --model {model_entry.id!r} does not support "
            f"IPAdapter (supports_ipadapter: false in its model YAML) "
            f"but --reference-image was also passed. Drop one of the "
            f"two, or pick a model with IPAdapter support (see "
            f"`python scripts/list_models.py`).",
            file=sys.stderr,
        )
        return 9
    if style_row.get("lora_stack") and not model_entry.supports_lora:
        print(
            f"EXIT 9: --model {model_entry.id!r} does not support "
            f"LoRAs (supports_lora: false in its model YAML) but "
            f"style profile {style_row['id']!r} declares a lora_stack. "
            f"Create a no-LoRA style profile for this model, or pick a "
            f"LoRA-compatible model (see "
            f"`python scripts/list_models.py`).",
            file=sys.stderr,
        )
        return 9

    try:
        rules = ContentLevelLoader(db_path).load(args.level)
    except UnknownContentLevel as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    # ----- 2. Scene pool + prompt build --------------------------------------
    scenes = _scenes_for_level(args.level, args.count)
    builder = PromptBuilder()
    sanitizer = PromptSanitizer.from_rules(rules)

    aspect_weights = cfg.get("aspect_ratio_weights")
    if not aspect_weights:
        print(
            "ERROR: pipeline.yaml is missing aspect_ratio_weights.",
            file=sys.stderr,
        )
        return 4
    selector = RatioSelector.from_config(
        cfg,
        ratio_signals_path=PROJECT_ROOT / "config" / "ratio_signals.yaml",
    )

    # --ratio: one of the 4 explicit ratios → forced; "auto[:audience]" →
    # selector with audience bonus; None → selector with no audience.
    forced_ratio: str | None = None
    selector_audience: str | None = None
    if args.ratio in {"portrait_23", "portrait_916", "square", "landscape"}:
        forced_ratio = args.ratio
        print(
            f"Forcing ratio={args.ratio} for all scenes "
            f"(--ratio override; RatioSelector bypassed)."
        )
    elif args.ratio in {"auto:deviantart", "auto:patreon"}:
        selector_audience = args.ratio.split(":", 1)[1]
        print(
            f"Aspect-ratio selection: auto with audience={selector_audience!r}."
        )
    # args.ratio in {None, "auto"} → plain auto, no audience bonus.

    # Reference image staging — only when --reference-image is set. We
    # validate the file exists before the render loop so the script bails
    # early on a typo, then copy it into ComfyUI's input/ directory so
    # the LoadImage node can resolve it by basename. In --dry-run we
    # validate but don't copy (matches the rest of dry-run's read-only
    # contract).
    ipadapter_basename: str | None = None
    if args.reference_image:
        ref_path = Path(args.reference_image).expanduser()
        if not ref_path.is_absolute():
            ref_path = (PROJECT_ROOT / ref_path).resolve()
        if not ref_path.exists():
            print(f"ERROR: --reference-image not found: {ref_path}", file=sys.stderr)
            return 7

        if args.dry_run:
            ipadapter_basename = ref_path.name
        else:
            comfy_input_dir = _resolve_comfy_input_dir(cfg)
            ipadapter_basename = _stage_reference_image(ref_path, comfy_input_dir)

    series_id = f"manual_{args.character}_{args.level}_{int(time.time())}"
    output_dir = PROJECT_ROOT / "output" / args.level / series_id / "images"

    # Model-aware negative prompt assembly happens per-scene below so the
    # Phase D conflict-axis filter sees each scene's actual positive
    # prompt (the strongest signal for collisions).
    supports_neg = (
        prompt_guide.supports_negative_prompt if prompt_guide
        else family.supports_negative_prompt
    )
    prompt_style = prompt_guide.prompt_style if prompt_guide else family.prompt_style
    trigger_words = prompt_guide.trigger_words if prompt_guide else None
    avoid_words = prompt_guide.avoid_words if prompt_guide else None
    guide_axes = (
        prompt_guide.negative_axes
        if prompt_guide and prompt_guide.negative_axes
        else None
    )

    records: list[_RenderRecord] = []
    for i, scene in enumerate(scenes):
        prompt_dict = builder.build_one(
            character, scene, style_row,
            family=family,
            trigger_words=trigger_words,
            avoid_words=avoid_words,
        )
        sanitized_text = sanitizer.sanitize_text(prompt_dict["prompt_text"])
        # Rehash so prompt_hash matches the stored text (sanitizer may
        # rewrite tokens; the builder's hash was computed pre-sanitize).
        sanitized_hash = compute_prompt_hash(sanitized_text)
        prompt_dict["negative_prompt"] = builder.assemble_negative_prompt(
            model_negative=(
                prompt_guide.base_negative_prompt
                if prompt_guide and not guide_axes
                else None
            ),
            model_negative_axes=guide_axes,
            style_negative=style_row.get("base_negative_prompt"),
            character_negative=character.get("negative_prompt"),
            supports_negative=supports_neg,
            negative_embeddings=(
                prompt_guide.negative_embeddings if prompt_guide else None
            ),
            conflict_terms=[
                character.get("base_prompt", ""),
                sanitized_text,
            ],
        )

        if forced_ratio is not None:
            res = model_res.get(forced_ratio, RATIO_TO_RESOLUTION.get(forced_ratio, (1024, 1024)))
            pick = RatioPick(
                ratio=forced_ratio,
                resolution=res,
                reason=f"cli-override (--ratio {forced_ratio})",
            )
        else:
            pick = selector.select(
                scene,
                content_level=args.level,
                model_checkpoint=style_for_wf.model_checkpoint,
                model_resolutions=model_res,
                family_id=model_entry.family,
                audience=selector_audience,
            )

        records.append(
            _RenderRecord(
                index=i,
                prompt_id=uuid.uuid4().hex,
                scene_id=uuid.uuid4().hex,
                scene=scene,
                ratio=pick.ratio,
                resolution=pick.resolution,
                prompt_text=sanitized_text,
                negative_prompt=prompt_dict["negative_prompt"],
                prompt_hash=sanitized_hash,
            )
        )

    # ----- 3. Dry-run short circuit ------------------------------------------
    if args.dry_run:
        print(f"DRY RUN — {len(records)} scenes for character={args.character} "
              f"level={args.level}")
        print(f"  series_id (would be): {series_id}")
        print(f"  output_dir (would be): {output_dir}")
        tier_label = (
            "tier 1 (CLI --model)" if model_source == "cli"
            else "tier 2 (character.model_id or pipeline.default_model_id)"
        )
        print(
            f"  model:               {model_entry.id} "
            f"[{tier_label}]"
        )
        print(f"    checkpoint:        {model_entry.filename}")
        print(f"    family:            {model_entry.family}")
        print(f"    prompt style:      {prompt_style}")
        print(
            f"    sampler/sched:     {style_for_wf.sampler}/"
            f"{style_for_wf.scheduler}"
        )
        print(
            f"    steps/cfg:         {style_for_wf.steps}/{style_for_wf.cfg}"
        )
        print(f"    neg prompt:        {'yes' if model_entry.supports_negative_prompt else 'no (Flux)'}")
        if trigger_words:
            print(f"    trigger words:     {', '.join(trigger_words[:5])}")
        if ipadapter_basename:
            print(f"  IPAdapter:           ENABLED — sdxl/ipadapter.json")
            print(f"  reference image:     {ipadapter_basename} (weight={args.ipadapter_weight})")
        else:
            print(f"  IPAdapter:           disabled — sdxl/base.json")
        print()
        for rec in records:
            print(f"[{rec.index:02d}] ratio={rec.ratio} ({rec.resolution[0]}x{rec.resolution[1]}) "
                  f"reason={rec.scene.get('pose','?')!r}")
            print(f"     prompt: {rec.prompt_text}")
            print()
        ratio_counts = Counter(r.ratio for r in records)
        print("Ratio distribution:")
        for ratio, count in ratio_counts.most_common():
            print(f"  {ratio:<14} {count}")
        return 0

    # ----- 4. ComfyUI render loop --------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    comfy_cfg = cfg.get("comfyui") or {}
    comfy_url = comfy_cfg.get("base_url", "http://127.0.0.1:8188")
    comfy_output_dir = comfy_cfg.get(
        "output_dir", str(Path.home() / "AI" / "apps" / "ComfyUI" / "output")
    )
    timeout = args.timeout or int(comfy_cfg.get("render_timeout_seconds", 300))

    wf_builder = WorkflowBuilder(WORKFLOW_DIR)
    client = ComfyUIClient(base_url=comfy_url, output_dir=comfy_output_dir)

    print(f"Rendering {len(records)} scenes")
    print(f"  character:    {args.character}")
    print(f"  style:        {style_row['id']}")
    print(f"  level:        {args.level}")
    print(f"  series_id:    {series_id}")
    print(f"  output:       {output_dir}")
    print(f"  comfy:        {comfy_url}")
    # Model resolution banner — shows which tier won and the sampler
    # settings WorkflowBuilder will actually inject. Having this in the
    # startup banner makes --model A/B runs easy to audit from logs.
    tier_label = (
        "tier 1 (CLI --model — registry sampler wins)"
        if model_source == "cli"
        else "tier 2 (character.model_id or pipeline.default_model_id "
             "— registry sampler wins)"
    )
    print(f"  model:        {model_entry.id}  [{tier_label}]")
    print(f"    checkpoint: {model_entry.filename}")
    print(f"    family:     {model_entry.family}")
    print(
        f"    sampler:    {style_for_wf.sampler}/{style_for_wf.scheduler} "
        f"steps={style_for_wf.steps} cfg={style_for_wf.cfg}"
    )
    print(
        f"    capability: ipadapter={model_entry.supports_ipadapter} "
        f"lora={model_entry.supports_lora}"
    )
    if args.template:
        print(f"  template:     {args.template} (external — IPAdapter disabled)")
    elif ipadapter_basename:
        print(f"  ipadapter:    {ipadapter_basename} (weight={args.ipadapter_weight})")
    else:
        print(f"  ipadapter:    (disabled — pass --reference-image to enable)")
    print()

    resolved_template = (
        _resolve_template_path(args.template, wf_builder.workflow_dir, PROJECT_ROOT)
        if args.template else None
    )

    render_start = time.time()
    for rec in records:
        try:
            if resolved_template:
                workflow = wf_builder.build_external(
                    external_template=resolved_template,
                    prompt_text=rec.prompt_text,
                    negative_prompt=rec.negative_prompt,
                    resolution=rec.resolution,
                )
            else:
                workflow = wf_builder.build(
                    prompt_text=rec.prompt_text,
                    negative_prompt=rec.negative_prompt,
                    style_profile=style_for_wf,
                    resolution=rec.resolution,
                    seed=None,  # random per image
                    ipadapter_image=ipadapter_basename,
                    ipadapter_weight=args.ipadapter_weight,
                )
            # Pull the seed back out so we can persist it.
            rec.seed = workflow["ksampler"]["inputs"]["seed"]

            print(f"[{rec.index + 1:02d}/{len(records)}] {rec.ratio} "
                  f"{rec.resolution[0]}x{rec.resolution[1]} seed={rec.seed} "
                  f"pose={rec.scene.get('pose','?')!r}")

            rendered = client.render_single_with_retry(
                workflow,
                max_attempts=int(comfy_cfg.get("max_retry_per_image", 3)),
                timeout=timeout,
            )
            if not rendered:
                rec.error = "ComfyUI returned no images"
                continue

            # Phase 1 batch_size is always 1 — there's only one image per render.
            first = rendered[0]
            rec.rendered_path = first.file_path
            dest = output_dir / f"{rec.index:02d}_{rec.scene_id[:8]}{first.file_path.suffix}"
            shutil.copy2(first.file_path, dest)
            rec.saved_path = dest
            rec.width = rec.resolution[0]
            rec.height = rec.resolution[1]

        except WorkflowTemplateError as exc:
            rec.error = f"WorkflowTemplateError: {exc}"
            print(f"     ERROR: {rec.error}", file=sys.stderr)
            # Template errors are fatal — they'll fire on every subsequent
            # scene too. Bail early.
            break
        except ComfyUIError as exc:
            rec.error = f"ComfyUIError: {exc}"
            print(f"     ERROR: {rec.error}", file=sys.stderr)
            continue

    render_seconds = time.time() - render_start

    # ----- 5. Score (lazy-loads CLIP+insightface once) -----------------------
    score_seconds = 0.0
    rendered_records = [r for r in records if r.saved_path]
    if rendered_records and not args.no_score:
        try:
            scorer = ImageScorer()
            score_start = time.time()
            for rec in rendered_records:
                try:
                    rec.score = scorer.score(rec.saved_path)
                except ScorerError as exc:
                    rec.score = {
                        "composite": 0.0, "aesthetic": 0.0, "blur": 0.0,
                        "face_confidence": 0.0,
                        "flags": ["scorer_error"],
                    }
                    print(f"     SCORER WARN: {exc}", file=sys.stderr)
            score_seconds = time.time() - score_start
        except ScorerModelError as exc:
            print(f"\nWARNING: scoring skipped — {exc}", file=sys.stderr)

    # ----- 6. Persist to DB --------------------------------------------------
    series_row = {
        "mode": "character",
        "content_level": args.level,
        "character_id": args.character,
        "style_profile_id": character["style_profile_id"],
        "theme": args.theme,
        "mood": None,
        "environment": None,
        "target_count": len(records),
        "actual_count": len(rendered_records),
        "status": "complete" if rendered_records else "failed",
        # Carry the resolved model_id down to the images INSERT so every
        # row in the audit trail knows which checkpoint produced it.
        "model_id": model_entry.id,
    }
    try:
        _persist_run(db_path, series_id, series_row, records)
    except sqlite3.Error as exc:
        print(f"\nERROR: DB persist failed: {exc}", file=sys.stderr)
        return 5

    # ----- 7. Update character usage counters --------------------------------
    if rendered_records:
        try:
            CharacterManager(db_path).update_usage(
                args.character, len(rendered_records)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: update_usage failed: {exc}", file=sys.stderr)

    _print_summary(
        series_id, output_dir, records, render_seconds, score_seconds
    )

    if not rendered_records:
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())

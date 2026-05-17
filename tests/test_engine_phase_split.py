"""Tests for the Phase 3 engine split — `run_phase_a / b / c` extracted
from `run_cycle`.

These tests exercise the public phase methods in isolation, without
requiring Ollama or ComfyUI. The LLM (SeriesPlanner / SceneGenerator
via mode.plan / mode.generate_scenes; SceneFacetGenerator) and the
ComfyUI client are mocked.

What this covers (per the plan + Phase 3 task list):

  * `run_phase_a` happy path — series + scenes + facets + prompts
    inserted with correct `model_id`.
  * `run_phase_a` multi-model — per-model prompts created;
    sibling-family models share scene_facets row.
  * `run_phase_a` re-target — `series_id_existing` skips planning
    and scene generation; reuses existing scenes.
  * `run_phase_a` `--regen-facets` — existing facets DELETEd before
    fresh generation.
  * `run_phase_a` `--regen-prompts` — existing prompts for that model
    DELETEd.
  * `run_phase_a` duplicate insert raises — when called twice for the
    same (series, model) without `--regen-prompts`.
  * `run_phase_b` missing prompts raises with helpful message.
  * `run_phase_b` missing series raises.
  * `_load_series_for_retarget` round-trips llm_series_plan JSON.
  * `_delete_prompts_for_target` returns count.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Fresh DB built from init_db.py SCHEMA_SQL."""
    db_path = tmp_path / "phase_split.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    return db_path


@pytest.fixture
def engine(fresh_db: Path):
    """A real PipelineEngine in dry_run mode pointed at fresh_db."""
    from src.core.engine import PipelineEngine
    minimal_config = {
        "execution": {"mode": "manual"},
        "compliance": {"commercial_mode": False},
        "comfyui": {
            "base_url": "http://127.0.0.1:8188",
            "output_dir": str(fresh_db.parent / "_comfy_out"),
            "input_dir": str(fresh_db.parent / "_comfy_in"),
            "render_timeout_seconds": 60,
            "max_retry_per_image": 1,
            "workflow_dir": "config/comfyui_workflows",
        },
        "set_builder": {"min_images": 1, "max_images": 5, "quality_cutoff": 0.0},
        "watermark": {"enabled": False},
        "postprocess": {"upscale_enabled": False, "face_refine_enabled": False},
        "scoring": {"use_hps_v2": False, "use_image_reward": False},
        "pipeline": {"output_dir": str(fresh_db.parent / "out")},
        "aspect_ratio_weights": {
            "T1_suggestive": {"portrait_23": 0.5, "square": 0.5},
            "T2_implied":    {"portrait_23": 0.5, "square": 0.5},
            "T3_artnude":    {"portrait_23": 0.7, "landscape": 0.3},
            "T4_explicit":   {"portrait_23": 0.7, "landscape": 0.3},
        },
        "mode_weights": {
            "character": 1.0, "theme": 0.0, "style": 0.0,
            "niche": 0.0, "variation": 0.0,
        },
    }
    return PipelineEngine(config=minimal_config, db_path=fresh_db, dry_run=True)


def _ctx_for(engine, model_id: str = "gonzalomo_photo_v70"):
    """Build a real GenerationContext via engine.build_context().

    Attaches a stub character so the prompt-builder's
    ``base_prompt`` requirement is satisfied. (Without a real
    character, character mode would synthesize from the series_plan,
    which is empty in these tests.)
    """
    from src.core.content_level import ContentLevelLoader
    from src.core.generation_context import build_context
    style_profile = {"id": "golden_hour_natural", "base_negative_prompt": ""}
    content_rules = ContentLevelLoader(engine.db_path).load("T2_implied")
    ctx = build_context(
        mode="character",
        content_level="T2_implied",
        execution_mode="manual",
        style_profile=style_profile,
        content_rules=content_rules,
        db_path=engine.db_path,
        model_id=model_id,
        commercial_mode=False,
    )
    ctx.character = {
        "base_prompt": "elara, 32yo, brunette, hazel eyes, athletic",
        "negative_prompt": "",
    }
    ctx.character_id = None  # leave NULL so series.character_id FK is happy
    return ctx


def _series_plan() -> dict:
    return {
        "theme": "rooftop golden hour",
        "mood": "calm",
        "environment": "urban rooftop",
        "variation_axes": ["pose", "camera"],
    }


def _scenes(n: int = 3) -> list[dict]:
    return [
        {
            "variation_axis": "pose",
            "pose": "standing relaxed",
            "camera": "85mm portrait",
            "camera_angle": "eye level",
            "lighting": "golden hour rim light",
            "environment_detail": f"rooftop balcony view {i}",
            "mood_note": "calm",
            "expression": "soft smile",
        }
        for i in range(n)
    ]


def _patch_mode_with_canned_data(engine, plan: dict, scenes: list[dict]):
    """Replace engine._mode_registry['character'] with a mock that
    returns canned plan + scenes (no LLM calls)."""
    mode = MagicMock()
    mode.plan.return_value = plan
    mode.generate_scenes.return_value = [dict(s) for s in scenes]
    # Mode subclass checks (NicheMode/CharacterMode/VariationMode) use
    # isinstance, so the mock won't be flagged as any of them — that's
    # fine for these tests (we exercise default character path).
    engine._mode_registry["character"] = mode
    return mode


def _patch_facet_gen_with_canned(facet_dict: dict):
    """Patch SceneFacetGenerator.generate at the engine import path."""
    return patch(
        "src.core.engine.SceneFacetGenerator.generate",
        return_value=facet_dict,
    )


def _patch_preflight():
    """Skip Phase A preflight (Ollama reachability) — no live services
    in unit tests. Render-time checks moved to ``_preflight_phase_b``
    which fires from ``run_phase_b`` only; tests in this file exercise
    ``run_phase_a`` so only the Phase-A patcher is needed here."""
    return patch(
        "src.core.engine.PipelineEngine._preflight_phase_a",
        return_value=None,
    )


def _patch_unload():
    return patch("src.core.engine.OllamaClient.unload_model", return_value=None)


# ── run_phase_a happy path ──────────────────────────────────────────


def test_run_phase_a_creates_series_scenes_facets_prompts(engine, fresh_db):
    plan = _series_plan()
    scenes = _scenes(3)
    canned_facet = {"camera_spec": "85mm f/1.4", "clothing": "ivory silk slip"}
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        result = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    assert result["status"] == "complete"
    assert result["scenes_created"] == 3
    assert result["facets_created"] == 3   # one per scene
    assert result["prompts_created"] == 3
    assert result["models_completed"] == ["gonzalomo_photo_v70"]

    # DB inspection
    conn = sqlite3.connect(str(fresh_db))
    series_id = result["series_id"]
    assert conn.execute(
        "SELECT COUNT(*) FROM series WHERE id = ?", (series_id,)
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM scenes WHERE series_id = ?", (series_id,)
    ).fetchone()[0] == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM scene_facets WHERE family = 'sdxl'"
    ).fetchone()[0] == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM prompts WHERE series_id = ? AND model_id = 'gonzalomo_photo_v70'",
        (series_id,)
    ).fetchone()[0] == 3


# ── run_phase_a multi-model fan-out ─────────────────────────────────


def test_run_phase_a_multi_model_creates_prompts_per_model(engine, fresh_db):
    """Two SDXL siblings share 1 facet row each but get distinct prompts."""
    plan = _series_plan()
    scenes = _scenes(2)
    canned_facet = {"camera_spec": "85mm f/1.4", "clothing": "silk slip"}
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        result = engine.run_phase_a(
            ctx,
            # Two sdxl-family models — should share scene_facets row
            # (PRIMARY KEY = scene_id + family).
            models=["gonzalomo_photo_v70", "juggernaut_ragnarok"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    assert result["status"] == "complete"
    assert result["scenes_created"] == 2
    assert result["facets_created"] == 2   # 2 scenes × 1 family (shared)
    assert result["prompts_created"] == 4   # 2 scenes × 2 models
    assert set(result["models_completed"]) == {"gonzalomo_photo_v70", "juggernaut_ragnarok"}

    conn = sqlite3.connect(str(fresh_db))
    series_id = result["series_id"]
    # Both models present in prompts table.
    rows = conn.execute(
        "SELECT model_id, COUNT(*) FROM prompts WHERE series_id = ? "
        "GROUP BY model_id ORDER BY model_id", (series_id,)
    ).fetchall()
    assert dict(rows) == {"juggernaut_ragnarok": 2, "gonzalomo_photo_v70": 2}
    # Only one facet row per scene (shared sdxl family).
    facet_count = conn.execute(
        "SELECT COUNT(*) FROM scene_facets WHERE family = 'sdxl'"
    ).fetchone()[0]
    assert facet_count == 2


def test_run_phase_a_cross_family_creates_separate_facets(engine, fresh_db):
    """sdxl + chroma have different prompt_styles → different facet rows."""
    plan = _series_plan()
    scenes = _scenes(2)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")

    # Different facet shapes per family — but for this test the SDXL
    # mock returns SDXL-shape, chroma returns chroma-shape. Easiest:
    # make the patch return-value include all expected fields; the
    # repo only saves the relevant ones per its _FACET_FIELDS list.
    rich_facet = {
        "camera_spec": "85mm", "clothing": "silk",
        "scene_prose": "She stands on the balcony.",
    }
    with (
        _patch_preflight(), _patch_unload(),
        _patch_facet_gen_with_canned(rich_facet),
    ):
        result = engine.run_phase_a(
            ctx,
            models=["gonzalomo_photo_v70", "chroma_v10HD"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    assert result["status"] == "complete"
    # 2 scenes × 2 families (sdxl + chroma) = 4 facet rows.
    assert result["facets_created"] == 4

    conn = sqlite3.connect(str(fresh_db))
    families = conn.execute(
        "SELECT DISTINCT family FROM scene_facets ORDER BY family"
    ).fetchall()
    assert {f[0] for f in families} == {"sdxl", "chroma"}


# ── re-target path ──────────────────────────────────────────────────


def test_run_phase_a_retarget_skips_planning(engine, fresh_db):
    """series_id_existing → no plan/scene-gen calls; facets+prompts only."""
    plan = _series_plan()
    scenes = _scenes(2)
    canned_facet = {"camera_spec": "85mm", "clothing": "silk"}
    mode = _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")

    # First call — fresh series.
    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        first = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )
    series_id = first["series_id"]

    # Re-target with a different model — should skip plan/scene-gen.
    mode.plan.reset_mock()
    mode.generate_scenes.reset_mock()
    new_ctx = _ctx_for(engine, model_id="chroma_v10HD")

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        second = engine.run_phase_a(
            new_ctx,
            models=["chroma_v10HD"],
            series_id_existing=series_id,
        )

    assert second["status"] == "complete"
    assert second["series_id"] == series_id
    # Crucially: scenes_created is 0 (loaded, not generated).
    assert second["scenes_created"] == 0
    assert second["facets_created"] == 2   # chroma facets (new)
    assert second["prompts_created"] == 2

    # Mode methods NOT called on re-target.
    mode.plan.assert_not_called()
    mode.generate_scenes.assert_not_called()


# ── regen flags ─────────────────────────────────────────────────────


def test_run_phase_a_regen_facets_deletes_then_regenerates(engine, fresh_db):
    plan = _series_plan()
    scenes = _scenes(2)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")
    facet_v1 = {"camera_spec": "v1 spec", "clothing": "v1 cloth"}
    facet_v2 = {"camera_spec": "v2 spec", "clothing": "v2 cloth"}

    # First run.
    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(facet_v1):
        first = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )
    series_id = first["series_id"]

    # Verify v1 facet stored.
    conn = sqlite3.connect(str(fresh_db))
    row = conn.execute(
        "SELECT camera_spec FROM scene_facets WHERE family='sdxl' LIMIT 1"
    ).fetchone()
    assert row[0] == "v1 spec"
    conn.close()

    # Re-target with --regen-facets sdxl + --regen-prompts gonzalomo_photo_v70
    # (need both since prompts uniqueness blocks re-insert otherwise).
    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(facet_v2):
        engine.run_phase_a(
            ctx,
            models=["gonzalomo_photo_v70"],
            series_id_existing=series_id,
            regen_facets=["sdxl"],
            regen_prompts=["gonzalomo_photo_v70"],
        )

    # Verify v2 facet now stored.
    conn = sqlite3.connect(str(fresh_db))
    rows = conn.execute(
        "SELECT camera_spec FROM scene_facets WHERE family='sdxl' "
        "ORDER BY scene_id"
    ).fetchall()
    assert all(r[0] == "v2 spec" for r in rows)


def test_run_phase_a_regen_prompts_clears_then_repopulates(engine, fresh_db):
    """Re-running with --regen-prompts succeeds (no UNIQUE collision)
    and the post-state has the same row count (replaced, not duplicated)."""
    plan = _series_plan()
    scenes = _scenes(2)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")
    canned_facet = {"camera_spec": "x", "clothing": "y"}

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        first = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )
    series_id = first["series_id"]

    # 2 prompts after first run.
    conn = sqlite3.connect(str(fresh_db))
    assert conn.execute(
        "SELECT COUNT(*) FROM prompts WHERE series_id = ? AND model_id = ?",
        (series_id, "gonzalomo_photo_v70"),
    ).fetchone()[0] == 2
    conn.close()

    # Re-run with --regen-prompts → DELETE first, then re-insert.
    # Must NOT raise UNIQUE collision.
    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        engine.run_phase_a(
            ctx,
            models=["gonzalomo_photo_v70"],
            series_id_existing=series_id,
            regen_prompts=["gonzalomo_photo_v70"],
        )

    conn = sqlite3.connect(str(fresh_db))
    # Still 2 prompts (replaced, not appended).
    assert conn.execute(
        "SELECT COUNT(*) FROM prompts WHERE series_id = ? AND model_id = ?",
        (series_id, "gonzalomo_photo_v70"),
    ).fetchone()[0] == 2


def test_run_phase_a_double_invoke_without_regen_raises(engine, fresh_db):
    """Re-running same model on same series without --regen-prompts →
    UNIQUE(scene_id, model_id) violation surfaces as IntegrityError."""
    plan = _series_plan()
    scenes = _scenes(2)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")
    canned_facet = {"camera_spec": "x", "clothing": "y"}

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        first = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    # Second run on same series + same model + no --regen-prompts:
    # the UNIQUE(scene_id, model_id) constraint trips at INSERT time.
    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        with pytest.raises(sqlite3.IntegrityError, match=r"(?i)unique"):
            engine.run_phase_a(
                ctx,
                models=["gonzalomo_photo_v70"],
                series_id_existing=first["series_id"],
            )


# ── run_phase_b validation ──────────────────────────────────────────


def test_run_phase_b_missing_series_raises(engine):
    from src.core.engine import EngineError
    with pytest.raises(EngineError, match="not found in DB"):
        engine.run_phase_b(
            series_id="ser_does_not_exist", model_id="gonzalomo_photo_v70",
        )


def test_run_phase_b_missing_prompts_raises_with_helpful_hint(
    engine, fresh_db,
):
    """Series exists but has no prompts for the requested model."""
    from src.core.engine import EngineError
    # Seed a series with no prompts.
    conn = sqlite3.connect(str(fresh_db))
    conn.execute(
        "INSERT INTO series (id, mode, content_level, style_profile_id, "
        "theme, target_count, status, llm_series_plan) "
        "VALUES ('ser_no_prompts', 'character', 'T2_implied', "
        "'golden_hour_natural', 't', 0, 'planned', '{}')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(EngineError, match="No prompts found"):
        engine.run_phase_b(
            series_id="ser_no_prompts", model_id="gonzalomo_photo_v70",
        )


# ── persistence helpers ─────────────────────────────────────────────


def test_load_series_for_retarget_round_trips_llm_series_plan(
    engine, fresh_db,
):
    """Phase A persists series_plan as JSON; re-target reads it back."""
    plan = _series_plan()
    scenes = _scenes(2)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine)
    canned_facet = {"camera_spec": "x", "clothing": "y"}

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        result = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    loaded_plan, loaded_scenes = engine._load_series_for_retarget(
        result["series_id"]
    )
    assert loaded_plan["theme"] == plan["theme"]
    assert loaded_plan["mood"] == plan["mood"]
    assert loaded_plan["variation_axes"] == plan["variation_axes"]
    assert len(loaded_scenes) == 2


def test_delete_prompts_for_target_returns_count(engine, fresh_db):
    plan = _series_plan()
    scenes = _scenes(3)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine)
    canned_facet = {"camera_spec": "x", "clothing": "y"}

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        result = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    n = engine._delete_prompts_for_target(
        result["series_id"], "gonzalomo_photo_v70", engine._default_llm_id,
        target_kind="model",
    )
    assert n == 3

    # Second delete is a no-op.
    n2 = engine._delete_prompts_for_target(
        result["series_id"], "gonzalomo_photo_v70", engine._default_llm_id,
        target_kind="model",
    )
    assert n2 == 0


def test_delete_prompts_for_target_leaves_other_models_intact(
    engine, fresh_db,
):
    plan = _series_plan()
    scenes = _scenes(2)
    _patch_mode_with_canned_data(engine, plan, scenes)
    ctx = _ctx_for(engine, model_id="gonzalomo_photo_v70")
    canned_facet = {"camera_spec": "x", "clothing": "y"}

    with _patch_preflight(), _patch_unload(), _patch_facet_gen_with_canned(canned_facet):
        result = engine.run_phase_a(
            ctx, models=["gonzalomo_photo_v70", "juggernaut_ragnarok"],
            style_profile={"id": "golden_hour_natural", "base_negative_prompt": ""},
            style_profile_id="golden_hour_natural",
        )

    # Delete only gonzalomo_photo_v70's prompts.
    engine._delete_prompts_for_target(
        result["series_id"], "gonzalomo_photo_v70", engine._default_llm_id,
        target_kind="model",
    )

    conn = sqlite3.connect(str(fresh_db))
    rows = conn.execute(
        "SELECT model_id, COUNT(*) FROM prompts WHERE series_id = ? "
        "GROUP BY model_id", (result["series_id"],)
    ).fetchall()
    # Only juggernaut_ragnarok survives.
    assert dict(rows) == {"juggernaut_ragnarok": 2}

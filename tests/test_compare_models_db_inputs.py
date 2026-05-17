"""Tests for ``compare_models.py``'s new ``--series-id`` / ``--scene-id``
input modes (Phase 5).

Covers the prompt-resolver helpers + task builders that turn DB rows
into per-(model, scene) render tasks, plus the missing-prompts hint
shape. The actual ComfyUI render path is mocked / not invoked here —
those paths are exercised in the existing render_set / engine tests.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_PATH = REPO_ROOT / "scripts" / "compare_models.py"


@pytest.fixture(scope="module")
def compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_models", str(COMPARE_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_models"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Fresh DB built from init_db.py SCHEMA_SQL."""
    db_path = tmp_path / "compare_db.db"
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py", "--db-path", str(db_path)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    return db_path


def _seed_series_with_prompts(
    db_path: Path,
    *,
    series_id: str = "ser_seed",
    scene_count: int = 2,
    models: list[str] | None = None,
    aspect_ratio: str = "portrait_23",
):
    """Seed a series + N scenes + 1 prompt per (scene, model)."""
    models = models or ["gonzalomo_photo_v70"]
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO series (id, mode, content_level, style_profile_id, "
        "theme, target_count, status) "
        "VALUES (?, 'character', 'T2_implied', 'golden_hour_natural', "
        "'test', 1, 'planned')",
        (series_id,),
    )
    for i in range(scene_count):
        scene_id = f"{series_id}_sc_{i:03d}"
        conn.execute(
            "INSERT INTO scenes (id, series_id, variation_axis, "
            "aspect_ratio, content_level) "
            "VALUES (?, ?, 'pose', ?, 'T2_implied')",
            (scene_id, series_id, aspect_ratio),
        )
        for j, model_id in enumerate(models):
            conn.execute(
                "INSERT INTO prompts (id, series_id, scene_id, model_id, "
                "llm_id, prompt_text, negative_prompt, prompt_hash, "
                "content_level, status) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, 'T2_implied', 'pending')",
                (f"{series_id}_p_{i}_{j}", series_id, scene_id, model_id,
                 "cydonia_24b_v43",
                 f"prompt for {model_id} scene {i}", "neg",
                 f"hash_{i}_{j}"),
            )
    conn.commit()
    conn.close()


# ── _resolve_db_prompts_per_model ───────────────────────────────────


class TestResolveDbPromptsPerModel:
    def test_series_id_loads_all_scenes_per_model(
        self, compare_module, fresh_db,
    ):
        _seed_series_with_prompts(
            fresh_db, scene_count=3, models=["gonzalomo_photo_v70", "chroma_v10HD"],
        )
        result = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id="ser_seed", scene_id=None,
            model_ids=["gonzalomo_photo_v70", "chroma_v10HD"],
        )
        assert set(result.keys()) == {"gonzalomo_photo_v70", "chroma_v10HD"}
        assert len(result["gonzalomo_photo_v70"]) == 3
        assert len(result["chroma_v10HD"]) == 3
        # Per-row contents.
        first_lustify = result["gonzalomo_photo_v70"][0]
        assert "prompt_text" in first_lustify
        assert "negative_prompt" in first_lustify
        assert "scene_id" in first_lustify
        assert "aspect_ratio" in first_lustify

    def test_scene_id_loads_single_prompt_per_model(
        self, compare_module, fresh_db,
    ):
        _seed_series_with_prompts(
            fresh_db, scene_count=3, models=["gonzalomo_photo_v70", "chroma_v10HD"],
        )
        result = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id=None, scene_id="ser_seed_sc_001",
            model_ids=["gonzalomo_photo_v70", "chroma_v10HD"],
        )
        assert len(result["gonzalomo_photo_v70"]) == 1
        assert len(result["chroma_v10HD"]) == 1
        assert result["gonzalomo_photo_v70"][0]["scene_id"] == "ser_seed_sc_001"
        # Per-model prompt text differs (different model_id in seeding).
        assert (
            result["gonzalomo_photo_v70"][0]["prompt_text"]
            != result["chroma_v10HD"][0]["prompt_text"]
        )

    def test_missing_model_returns_empty_list(
        self, compare_module, fresh_db,
    ):
        _seed_series_with_prompts(fresh_db, models=["gonzalomo_photo_v70"])
        result = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id="ser_seed", scene_id=None,
            model_ids=["gonzalomo_photo_v70", "gonzalomo_flux_v30"],  # second has no prompts
        )
        assert result["gonzalomo_photo_v70"]  # has data
        assert result["gonzalomo_flux_v30"] == []  # empty


# ── _validate_db_prompts ────────────────────────────────────────────


class TestValidateDbPrompts:
    def test_all_have_prompts_passes_silently(self, compare_module):
        # Should not raise.
        compare_module._validate_db_prompts(
            {"gonzalomo_photo_v70": [{"prompt_text": "x"}]},
            series_id="ser_x", scene_id=None,
        )

    def test_any_missing_raises_with_helpful_hint(self, compare_module):
        with pytest.raises(SystemExit) as exc:
            compare_module._validate_db_prompts(
                {
                    "gonzalomo_photo_v70": [{"prompt_text": "x"}],
                    "gonzalomo_flux_v30": [],
                },
                series_id="ser_x", scene_id=None,
            )
        msg = str(exc.value)
        assert "no prompts in DB" in msg
        assert "gonzalomo_flux_v30" in msg
        assert "gonzalomo_photo_v70" not in msg  # only the missing one called out
        assert "ser_x" in msg
        assert "prepare_prompts.py" in msg
        assert "--models gonzalomo_flux_v30" in msg

    def test_scene_mode_error_uses_scene_label(self, compare_module):
        with pytest.raises(SystemExit) as exc:
            compare_module._validate_db_prompts(
                {"gonzalomo_photo_v70": []},
                series_id=None, scene_id="sc_xyz",
            )
        assert "scene 'sc_xyz'" in str(exc.value)


# ── _build_db_render_tasks ──────────────────────────────────────────


def _fake_model(family: str = "sdxl", filename: str = "lustify.safetensors"):
    """A SimpleNamespace stand-in for ModelRegistryEntry — only the
    attributes ``model_resolution_overrides`` and ``get_resolution``
    actually read need to be present."""
    return SimpleNamespace(
        family=family,
        filename=filename,
        resolution_portrait=None,
        resolution_square=None,
        resolution_landscape=None,
    )


class TestBuildDbRenderTasks:
    def test_uses_scene_aspect_ratio_when_no_forced_ratio(self, compare_module):
        db_prompts = [
            {
                "prompt_text": "p0", "negative_prompt": "n0",
                "scene_id": "ser_x_sc_000", "aspect_ratio": "portrait_23",
            },
            {
                "prompt_text": "p1", "negative_prompt": "n1",
                "scene_id": "ser_x_sc_001", "aspect_ratio": "square",
            },
        ]
        tasks = compare_module._build_db_render_tasks(
            db_prompts=db_prompts,
            model=_fake_model(),
            base_seed=100,
            forced_ratio=None,
            fallback_resolution=(1024, 1024),
        )
        assert len(tasks) == 2
        assert tasks[0]["prompt_text"] == "p0"
        assert tasks[0]["seed"] == 100
        assert tasks[1]["seed"] == 101
        # Resolutions differ because scene aspect_ratio differs.
        assert tasks[0]["resolution"] != tasks[1]["resolution"]

    def test_forced_ratio_overrides_per_scene(self, compare_module):
        db_prompts = [
            {
                "prompt_text": "p0", "negative_prompt": "n0",
                "scene_id": "sc_a", "aspect_ratio": "portrait_23",
            },
            {
                "prompt_text": "p1", "negative_prompt": "n1",
                "scene_id": "sc_b", "aspect_ratio": "landscape",
            },
        ]
        tasks = compare_module._build_db_render_tasks(
            db_prompts=db_prompts,
            model=_fake_model(),
            base_seed=42,
            forced_ratio="square",   # override
            fallback_resolution=(1024, 1024),
        )
        # All tasks at the forced ratio's resolution (same).
        assert tasks[0]["resolution"] == tasks[1]["resolution"]

    def test_fallback_resolution_used_when_no_aspect_ratio(self, compare_module):
        db_prompts = [
            {
                "prompt_text": "p", "negative_prompt": "n",
                "scene_id": "sc", "aspect_ratio": None,  # missing
            },
        ]
        tasks = compare_module._build_db_render_tasks(
            db_prompts=db_prompts,
            model=_fake_model(),
            base_seed=0,
            forced_ratio=None,
            fallback_resolution=(1234, 5678),
        )
        assert tasks[0]["resolution"] == (1234, 5678)

    def test_seeds_increment_per_scene(self, compare_module):
        db_prompts = [
            {"prompt_text": f"p{i}", "negative_prompt": "n",
             "scene_id": f"sc_{i}", "aspect_ratio": "square"}
            for i in range(5)
        ]
        tasks = compare_module._build_db_render_tasks(
            db_prompts=db_prompts,
            model=_fake_model(),
            base_seed=1000,
            forced_ratio=None,
            fallback_resolution=(1024, 1024),
        )
        seeds = [t["seed"] for t in tasks]
        assert seeds == [1000, 1001, 1002, 1003, 1004]

    def test_label_uses_scene_id_tail(self, compare_module):
        db_prompts = [
            {
                "prompt_text": "p",
                "negative_prompt": "n",
                "scene_id": "ser_abc_sc_007",
                "aspect_ratio": "square",
            },
        ]
        tasks = compare_module._build_db_render_tasks(
            db_prompts=db_prompts,
            model=_fake_model(),
            base_seed=0,
            forced_ratio=None,
            fallback_resolution=(1024, 1024),
        )
        # Tail of scene_id (after last underscore) is used as label.
        assert tasks[0]["label"] == "007"


# ── _build_text_render_tasks ────────────────────────────────────────


class TestBuildTextRenderTasks:
    def test_count_copies_at_successive_seeds(self, compare_module):
        tasks = compare_module._build_text_render_tasks(
            prompt_text="hello world",
            negative_prompt="bad",
            base_seed=42,
            count=3,
            resolution=(1024, 1024),
        )
        assert len(tasks) == 3
        assert all(t["prompt_text"] == "hello world" for t in tasks)
        assert all(t["negative_prompt"] == "bad" for t in tasks)
        assert [t["seed"] for t in tasks] == [42, 43, 44]
        assert [t["label"] for t in tasks] == ["00", "01", "02"]

    def test_zero_count_returns_empty(self, compare_module):
        tasks = compare_module._build_text_render_tasks(
            prompt_text="x", negative_prompt="y",
            base_seed=0, count=0, resolution=(1024, 1024),
        )
        assert tasks == []


# ── argparse: 4-way XOR enforced ────────────────────────────────────


def test_argparse_requires_one_input_mode():
    """With NO input flag, argparse exits non-zero with a 'one of'
    message listing all 4 options."""
    result = subprocess.run(
        [sys.executable, str(COMPARE_PATH), "--models", "gonzalomo_photo_v70"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    err = result.stderr
    assert "--prompt" in err
    assert "--character" in err
    assert "--series-id" in err
    assert "--scene-id" in err


def test_argparse_rejects_two_input_modes_at_once():
    """e.g. --prompt + --series-id is mutually exclusive."""
    result = subprocess.run(
        [sys.executable, str(COMPARE_PATH),
         "--prompt", "x", "--series-id", "ser_a",
         "--models", "gonzalomo_photo_v70"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


# ── F3: --llm filter coverage ───────────────────────────────────────


def _seed_two_llms(db_path: Path) -> None:
    """Seed one series, one scene, two prompts under different llm_ids."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO series (id, mode, content_level, style_profile_id, "
        "theme, target_count, status) VALUES "
        "('ser_two', 'character', 'T2_implied', 'golden_hour_natural', "
        "'test', 1, 'planned')"
    )
    conn.execute(
        "INSERT INTO scenes (id, series_id, variation_axis, aspect_ratio, "
        "content_level) VALUES "
        "('sc_two_000', 'ser_two', 'pose', 'portrait_23', 'T2_implied')"
    )
    for i, llm_id in enumerate(("cydonia_24b_v43", "magnum_v4_22b")):
        conn.execute(
            "INSERT INTO prompts (id, series_id, scene_id, model_id, llm_id, "
            "prompt_text, negative_prompt, prompt_hash, content_level, "
            "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'T2_implied', 'pending')",
            (f"p_{i}", "ser_two", "sc_two_000", "gonzalomo_photo_v70", llm_id,
             f"prompt from {llm_id}", "neg", f"hash_{i}"),
        )
    conn.commit()
    conn.close()


class TestLlmFilter:
    """Dedicated coverage for the compare_models --llm filter (F3).

    With two LLMs' prompts on the same (series, model), --llm should
    return only the one matching set.
    """

    def test_llm_filter_returns_only_matching_rows(
        self, compare_module, fresh_db,
    ):
        _seed_two_llms(fresh_db)
        out = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id="ser_two",
            scene_id=None,
            model_ids=["gonzalomo_photo_v70"],
            llm_id="cydonia_24b_v43",
        )
        assert len(out["gonzalomo_photo_v70"]) == 1
        assert out["gonzalomo_photo_v70"][0]["prompt_text"] == "prompt from cydonia_24b_v43"

    def test_llm_filter_other_llm(
        self, compare_module, fresh_db,
    ):
        _seed_two_llms(fresh_db)
        out = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id="ser_two",
            scene_id=None,
            model_ids=["gonzalomo_photo_v70"],
            llm_id="magnum_v4_22b",
        )
        assert len(out["gonzalomo_photo_v70"]) == 1
        assert out["gonzalomo_photo_v70"][0]["prompt_text"] == "prompt from magnum_v4_22b"

    def test_no_llm_filter_returns_all(
        self, compare_module, fresh_db,
    ):
        _seed_two_llms(fresh_db)
        out = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id="ser_two",
            scene_id=None,
            model_ids=["gonzalomo_photo_v70"],
            llm_id=None,
        )
        # 2 prompts (one per LLM) when filter omitted.
        assert len(out["gonzalomo_photo_v70"]) == 2

    def test_scene_id_path_with_llm_filter(
        self, compare_module, fresh_db,
    ):
        _seed_two_llms(fresh_db)
        out = compare_module._resolve_db_prompts_per_model(
            db_path=fresh_db,
            series_id=None,
            scene_id="sc_two_000",
            model_ids=["gonzalomo_photo_v70"],
            llm_id="cydonia_24b_v43",
        )
        assert len(out["gonzalomo_photo_v70"]) == 1

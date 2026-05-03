"""Regression: ``scripts/compare_models.py`` character mode must not
read fields that were removed from ``StyleProfile`` in the 2026-04
refactor (``model_id``, ``lora_stack``, ``sampler``, ``scheduler``,
``steps``, ``cfg``, ``clip_skip``).

Before the fix, ``_load_character_context`` raised
``AttributeError: 'StyleProfile' object has no attribute 'model_id'``
on every character-mode invocation because the script was never
updated when the profile dataclass was stripped. This test pins the
corrected shape: only aesthetic-intent fields are read, and LoRA
resolution is deferred to the per-model render loop.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


# ``scripts/`` is not a package; load ``compare_models.py`` by path.
_COMPARE_MODELS_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_models.py"
)


@pytest.fixture(scope="module")
def compare_models_module():
    # Ensure project root is on sys.path so ``from src...`` imports inside
    # compare_models.py resolve when the module is loaded in isolation.
    project_root = str(_COMPARE_MODELS_PATH.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    spec = importlib.util.spec_from_file_location(
        "compare_models", _COMPARE_MODELS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def character_db(tmp_path: Path) -> Path:
    """Tmp SQLite with a minimal characters row pointing at a real profile."""
    db_path = tmp_path / "characters.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE characters (
                id TEXT PRIMARY KEY,
                name TEXT,
                style_profile_id TEXT NOT NULL,
                model_id TEXT,
                base_prompt TEXT NOT NULL,
                negative_prompt TEXT,
                reference_image_path TEXT,
                locked_features TEXT NOT NULL,
                allowed_shift_axes TEXT NOT NULL,
                outfit_pool TEXT,
                version INTEGER DEFAULT 1,
                active BOOLEAN DEFAULT 1,
                total_images_generated INTEGER DEFAULT 0,
                last_used_at TIMESTAMP,
                created_at TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO characters (
                id, name, style_profile_id, base_prompt, locked_features,
                allowed_shift_axes
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "char_test",
                "char_test",
                # Any valid profile id from config/style_profiles.yaml.
                "boudoir_noir",
                "a portrait photo of a woman",
                "{}",
                "[]",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_character_context_does_not_read_removed_profile_fields(
    compare_models_module, character_db
):
    # The regression: before the fix, this line raised AttributeError.
    char_row, style_row = compare_models_module._load_character_context(
        character_db, "char_test"
    )
    # style_row MUST NOT carry the legacy fields that were stripped from
    # StyleProfile — keeping them here is a trap for future authors
    # (they'd read a None and silently mis-render).
    assert "model_id" not in style_row, (
        "style_row must not re-introduce model_id — checkpoint selection "
        "is bound to the character or pipeline default post-refactor."
    )
    assert "lora_stack" not in style_row, (
        "style_row must not re-introduce lora_stack — LoRAs come from "
        "the model YAML and are resolved per-candidate in the render loop."
    )
    # The aesthetic fields the prompt composer actually consumes.
    assert style_row["id"] == "boudoir_noir"
    assert "base_style_keywords" in style_row
    assert "base_negative_prompt" in style_row
    # Character dict round-trips the minimal row.
    assert char_row["id"] == "char_test"
    assert char_row["base_prompt"] == "a portrait photo of a woman"


def test_character_prompt_composer_builds_non_empty_prompt(
    compare_models_module, character_db
):
    char_row, style_row = compare_models_module._load_character_context(
        character_db, "char_test"
    )
    prompt_text, negative_prompt = (
        compare_models_module._compose_prompt_from_character(char_row, style_row)
    )
    assert "a portrait photo of a woman" in prompt_text
    # base_style_keywords from boudoir_noir should also land in the prompt.
    assert len(prompt_text) > len(char_row["base_prompt"])
    # Negative prompt is populated either from style_row or the DEFAULT.
    assert negative_prompt

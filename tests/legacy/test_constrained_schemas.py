"""Tests for the Q9 constrained-decoding schemas.

Covers ``MetadataSchema`` and ``CharacterSchema`` (added in Q9) plus
sanity checks that ``SceneList`` (already wired for SceneGenerator)
still validates as expected.

Each schema is exercised via the ``OllamaClient.generate_json``
``format:`` path indirectly — these tests validate the Pydantic post-
validation layer that catches anything Ollama's JSON-grammar can't
enforce (string length bands, list cardinality, non-empty after strip,
etc.).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.schemas import (
    MetadataSchema,
    Scene,
    SceneList,
)


# ── MetadataSchema ───────────────────────────────────────────────────
class TestMetadataSchema:
    def test_valid_metadata_round_trips(self):
        m = MetadataSchema.model_validate({
            "title": "Golden Hour Reverie",
            "description": "A series exploring soft golden-hour light "
                           "across a quiet bedroom interior.",
            "tags": ["golden hour", "natural light", "bedroom", "soft", "warm"],
        })
        assert m.title == "Golden Hour Reverie"
        assert len(m.tags) == 5

    def test_title_too_short_rejected(self):
        with pytest.raises(ValidationError, match="title"):
            MetadataSchema.model_validate({
                "title": "x",
                "description": "x" * 30,
                "tags": ["a", "b", "c"],
            })

    def test_title_too_long_rejected(self):
        with pytest.raises(ValidationError, match="title"):
            MetadataSchema.model_validate({
                "title": "x" * 200,
                "description": "x" * 30,
                "tags": ["a", "b", "c"],
            })

    def test_description_too_short_rejected(self):
        with pytest.raises(ValidationError, match="description"):
            MetadataSchema.model_validate({
                "title": "Valid Title",
                "description": "short",
                "tags": ["a", "b", "c"],
            })

    def test_description_too_long_rejected(self):
        with pytest.raises(ValidationError, match="description"):
            MetadataSchema.model_validate({
                "title": "Valid Title",
                "description": "x" * 700,
                "tags": ["a", "b", "c"],
            })

    def test_too_few_tags_rejected(self):
        with pytest.raises(ValidationError, match="tags"):
            MetadataSchema.model_validate({
                "title": "Valid Title",
                "description": "x" * 30,
                "tags": ["only", "two"],
            })

    def test_too_many_tags_rejected(self):
        with pytest.raises(ValidationError, match="tags"):
            MetadataSchema.model_validate({
                "title": "Valid Title",
                "description": "x" * 30,
                "tags": [f"tag{i}" for i in range(50)],
            })

    def test_tags_with_blank_strings_filtered(self):
        m = MetadataSchema.model_validate({
            "title": "Valid Title",
            "description": "x" * 30,
            "tags": ["a", "  ", "b", "", "c"],
        })
        assert m.tags == ["a", "b", "c"]

    def test_tags_all_blank_rejected(self):
        with pytest.raises(ValidationError, match="3 non-empty"):
            MetadataSchema.model_validate({
                "title": "Valid Title",
                "description": "x" * 30,
                "tags": ["", "  ", "\t"],
            })

    def test_extra_fields_allowed(self):
        m = MetadataSchema.model_validate({
            "title": "Valid Title",
            "description": "x" * 30,
            "tags": ["a", "b", "c"],
            "extra_field_from_llm": "ignored but accepted",
        })
        assert m.title == "Valid Title"


# CharacterSchema removed 2026-05-20 with character mode



# ── SceneList sanity (already used by SceneGenerator) ────────────────
class TestSceneListSanity:
    def test_empty_list_rejected(self):
        # Post-2026-05-06, SceneList carries MinLen(1): an empty list
        # is rejected at both the grammar level (minItems: 1 lands in
        # the JSON schema Ollama hands to llama.cpp) and Pydantic
        # post-validation. Closes the historical "emit []" path that
        # some 22B-class models hit on edge cases.
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SceneList.model_validate([])

    def test_valid_scene_list(self):
        sl = SceneList.model_validate([
            {
                "variation_axis": "pose",
                "pose": "standing",
                "camera": "medium",
                "camera_angle": "eye level",
                "lighting": "soft natural",
                "environment_detail": "studio",
                "mood_note": "calm",
            }
        ])
        assert len(sl) == 1
        assert sl[0].pose == "standing"

    def test_scene_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            SceneList.model_validate([{"pose": "standing"}])

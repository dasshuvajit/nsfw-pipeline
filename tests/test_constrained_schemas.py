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
    CharacterSchema,
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


# ── CharacterSchema ──────────────────────────────────────────────────
class TestCharacterSchema:
    _VALID = {
        "gender": "female",
        "age_appearance": "late 20s",
        "face": "oval face, sharp jawline, almond eyes",
        "hair": "long dark brown wavy hair",
        "body_type": "slim athletic",
        "distinguishing_features": "light freckles across nose",
        "vibe": "soft elegant",
    }

    def test_valid_character_round_trips(self):
        c = CharacterSchema.model_validate(self._VALID)
        assert c.gender == "female"
        assert c.face.startswith("oval face")

    def test_face_too_short_rejected(self):
        bad = {**self._VALID, "face": "nice"}
        with pytest.raises(ValidationError, match="face"):
            CharacterSchema.model_validate(bad)

    def test_hair_too_short_rejected(self):
        bad = {**self._VALID, "hair": "blonde"}
        with pytest.raises(ValidationError, match="hair"):
            CharacterSchema.model_validate(bad)

    def test_body_type_too_short_rejected(self):
        bad = {**self._VALID, "body_type": "fit"}
        with pytest.raises(ValidationError, match="body_type"):
            CharacterSchema.model_validate(bad)

    def test_vibe_too_short_rejected(self):
        bad = {**self._VALID, "vibe": "yes"}
        with pytest.raises(ValidationError, match="vibe"):
            CharacterSchema.model_validate(bad)

    def test_gender_normalised_lowercase(self):
        bad = {**self._VALID, "gender": "FEMALE  "}
        c = CharacterSchema.model_validate(bad)
        assert c.gender == "female"

    def test_blank_gender_rejected(self):
        bad = {**self._VALID, "gender": "   "}
        with pytest.raises(ValidationError, match="gender"):
            CharacterSchema.model_validate(bad)

    def test_extra_fields_allowed(self):
        bad = {**self._VALID, "personality": "shy and curious"}
        c = CharacterSchema.model_validate(bad)
        assert c.face == self._VALID["face"]


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

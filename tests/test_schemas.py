"""Pydantic schema contracts — SeriesPlan + Scene + SceneList."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.schemas import Scene, SceneList, SeriesPlan


# ----- valid baseline payloads ---------------------------------------------

VALID_SERIES_PLAN = {
    "theme": "rainy autumn evening in a velvet-lit study",
    "mood": "contemplative",
    "environment": "library with leather chairs",
    "variation_axes": ["pose intensity", "camera distance", "lighting warmth"],
}

VALID_SCENE = {
    "variation_axis": "pose intensity",
    "pose": "seated cross-legged on a leather chair",
    "camera": "medium shot",
    "camera_angle": "slightly above",
    "lighting": "warm tungsten from desk lamp",
    "environment_detail": "stack of books behind subject",
    "mood_note": "thoughtful and at ease",
}


# ============================================================================
# SeriesPlan
# ============================================================================

class TestSeriesPlan:
    def test_valid_payload_round_trips(self):
        plan = SeriesPlan.model_validate(VALID_SERIES_PLAN)
        assert plan.theme == VALID_SERIES_PLAN["theme"]
        assert plan.variation_axes == VALID_SERIES_PLAN["variation_axes"]

    def test_missing_theme_raises_with_field_path(self):
        bad = {**VALID_SERIES_PLAN}
        bad.pop("theme")
        with pytest.raises(ValidationError) as exc_info:
            SeriesPlan.model_validate(bad)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("theme",) for e in errors)

    def test_missing_mood_raises(self):
        bad = {**VALID_SERIES_PLAN}
        bad.pop("mood")
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_missing_environment_raises(self):
        bad = {**VALID_SERIES_PLAN}
        bad.pop("environment")
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_missing_variation_axes_raises(self):
        bad = {**VALID_SERIES_PLAN}
        bad.pop("variation_axes")
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_empty_variation_axes_list_raises(self):
        bad = {**VALID_SERIES_PLAN, "variation_axes": []}
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_variation_axes_with_only_blanks_raises(self):
        bad = {**VALID_SERIES_PLAN, "variation_axes": ["", "   ", "\n"]}
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_variation_axes_blanks_filtered_out(self):
        bad = {**VALID_SERIES_PLAN, "variation_axes": ["pose", "", "  ", "lighting"]}
        plan = SeriesPlan.model_validate(bad)
        assert plan.variation_axes == ["pose", "lighting"]

    def test_empty_string_theme_raises(self):
        bad = {**VALID_SERIES_PLAN, "theme": ""}
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_whitespace_only_theme_raises(self):
        bad = {**VALID_SERIES_PLAN, "theme": "   \n  "}
        with pytest.raises(ValidationError):
            SeriesPlan.model_validate(bad)

    def test_extra_fields_preserved(self):
        bonus = {**VALID_SERIES_PLAN, "style_notes": "go heavy on shadow"}
        plan = SeriesPlan.model_validate(bonus)
        dumped = plan.model_dump()
        assert dumped["style_notes"] == "go heavy on shadow"

    def test_model_validate_json_round_trip(self):
        import json
        plan = SeriesPlan.model_validate_json(json.dumps(VALID_SERIES_PLAN))
        assert plan.theme == VALID_SERIES_PLAN["theme"]


# ============================================================================
# Scene
# ============================================================================

class TestScene:
    def test_valid_payload_round_trips(self):
        s = Scene.model_validate(VALID_SCENE)
        assert s.pose == VALID_SCENE["pose"]
        assert s.composition_intent is None

    @pytest.mark.parametrize(
        "missing_field",
        [
            "variation_axis", "pose", "camera", "camera_angle",
            "lighting", "environment_detail", "mood_note",
        ],
    )
    def test_required_field_missing_raises_with_path(self, missing_field):
        bad = {**VALID_SCENE}
        bad.pop(missing_field)
        with pytest.raises(ValidationError) as exc_info:
            Scene.model_validate(bad)
        errors = exc_info.value.errors()
        assert any(e["loc"] == (missing_field,) for e in errors), (
            f"Expected error for {missing_field}, got: {errors}"
        )

    @pytest.mark.parametrize(
        "field",
        ["pose", "camera", "lighting", "environment_detail", "mood_note"],
    )
    def test_required_field_empty_string_raises(self, field):
        bad = {**VALID_SCENE, field: ""}
        with pytest.raises(ValidationError):
            Scene.model_validate(bad)

    def test_phase_a_optional_fields_default_to_none(self):
        s = Scene.model_validate(VALID_SCENE)
        assert s.composition_intent is None
        assert s.framing_hint is None
        assert s.audience_target is None

    def test_phase_a_intent_field_round_trips(self):
        payload = {**VALID_SCENE, "composition_intent": "close-up"}
        s = Scene.model_validate(payload)
        assert s.composition_intent == "close-up"

    def test_audience_target_round_trips(self):
        payload = {**VALID_SCENE, "audience_target": "patreon"}
        s = Scene.model_validate(payload)
        assert s.audience_target == "patreon"

    def test_sdxl_family_fields_round_trip(self):
        payload = {
            **VALID_SCENE,
            "camera_spec": "85mm f/1.8, shallow DoF",
            "clothing": "silk slip dress",
        }
        s = Scene.model_validate(payload)
        assert s.camera_spec == "85mm f/1.8, shallow DoF"
        assert s.clothing == "silk slip dress"

    def test_pony_family_fields_round_trip(self):
        payload = {
            **VALID_SCENE,
            "booru_tags": "1girl, solo, sitting, library",
            "source_tag": "source_photograph",
        }
        s = Scene.model_validate(payload)
        assert s.booru_tags == "1girl, solo, sitting, library"
        assert s.source_tag == "source_photograph"

    def test_flux2_family_fields_round_trip(self):
        payload = {
            **VALID_SCENE,
            "scene_prose": "A young woman seated with a book in soft tungsten light.",
            "subject_focus": "young woman seated reading",
            "lighting_directive": "warm tungsten 3200 K, soft from camera left",
        }
        s = Scene.model_validate(payload)
        assert s.scene_prose.startswith("A young woman")
        assert s.subject_focus
        assert s.lighting_directive

    def test_unknown_extra_fields_preserved(self):
        payload = {**VALID_SCENE, "weird_llm_addition": "go wild"}
        s = Scene.model_validate(payload)
        assert s.model_dump()["weird_llm_addition"] == "go wild"

    def test_strip_whitespace_normalises_required_strings(self):
        payload = {**VALID_SCENE, "pose": "   seated   "}
        s = Scene.model_validate(payload)
        assert s.pose == "seated"

    def test_model_dump_returns_plain_dict(self):
        s = Scene.model_validate(VALID_SCENE)
        d = s.model_dump(mode="python")
        assert isinstance(d, dict)
        for key in VALID_SCENE:
            assert key in d


# ============================================================================
# SceneList
# ============================================================================

class TestSceneList:
    def test_valid_list_validates(self):
        payload = [VALID_SCENE, {**VALID_SCENE, "pose": "standing"}]
        sl = SceneList.model_validate(payload)
        assert len(sl) == 2

    def test_iter_and_index(self):
        sl = SceneList.model_validate([VALID_SCENE, VALID_SCENE])
        assert sl[0].pose == VALID_SCENE["pose"]
        assert sum(1 for _ in sl) == 2

    def test_invalid_inner_scene_raises_with_index(self):
        bad_scene = {**VALID_SCENE}
        bad_scene.pop("pose")
        payload = [VALID_SCENE, bad_scene]
        with pytest.raises(ValidationError) as exc_info:
            SceneList.model_validate(payload)
        errors = exc_info.value.errors()
        # Error path should include the list index
        assert any(1 in e["loc"] for e in errors)

    def test_root_returns_underlying_list_of_scenes(self):
        sl = SceneList.model_validate([VALID_SCENE])
        assert isinstance(sl.root, list)
        assert isinstance(sl.root[0], Scene)

    def test_model_dump_returns_plain_list_of_dicts(self):
        sl = SceneList.model_validate([VALID_SCENE, VALID_SCENE])
        dumped = sl.model_dump(mode="python")
        assert isinstance(dumped, list)
        assert len(dumped) == 2
        assert all(isinstance(d, dict) for d in dumped)

    def test_model_validate_json_round_trip(self):
        import json
        payload = [VALID_SCENE, VALID_SCENE]
        sl = SceneList.model_validate_json(json.dumps(payload))
        assert len(sl) == 2

    def test_empty_list_rejected(self):
        """Post-2026-05-06, ``SceneList`` carries ``MinLen(1)`` so an
        empty list is invalid at both the grammar level (Ollama emits
        ``minItems: 1`` to llama.cpp's grammar) and the Pydantic post-
        validation level. A loose-aligned LLM (Venice, Magnum) used to
        emit ``[]`` as the path of least resistance — that path is
        now closed."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SceneList.model_validate([])


# ============================================================================
# Integration with llm_client.generate_json(schema=...)
# ============================================================================

class TestLLMClientSchemaIntegration:
    """Verify the schema kwarg on OllamaClient.generate_json without
    hitting a real Ollama server. We patch ``generate`` to return a
    canned string, then run it through the schema-validation path."""

    def _patched_client(self, monkeypatch, raw_response: str):
        from src.agents.llm_client import OllamaClient

        client = OllamaClient(base_url="http://test.invalid")
        monkeypatch.setattr(client, "generate", lambda *a, **kw: raw_response)

        # Q6: schema-aware calls go through /api/chat with assistant
        # prefill. The prefill is "Sure, here's the JSON: " (no
        # structural opener), so the chat mock returns the full JSON
        # the same way generate does — _extract_json_payload skips
        # the preamble before parsing.
        monkeypatch.setattr(
            client, "_generate_chat",
            lambda *a, **kw: raw_response,
        )
        return client

    def test_schema_validation_happy_path(self, monkeypatch):
        import json
        client = self._patched_client(monkeypatch, json.dumps(VALID_SERIES_PLAN))
        result = client.generate_json(
            "sys", "user", schema=SeriesPlan, model="test-model",
        )
        assert isinstance(result, dict)
        assert result["theme"] == VALID_SERIES_PLAN["theme"]

    def test_schema_validation_failure_raises_parse_error(self, monkeypatch):
        from src.agents.llm_client import OllamaJSONParseError

        # Missing 'environment' field
        bad = {"theme": "x", "mood": "y", "variation_axes": ["a"]}
        import json
        client = self._patched_client(monkeypatch, json.dumps(bad))
        with pytest.raises(OllamaJSONParseError, match="schema validation"):
            client.generate_json("sys", "user", schema=SeriesPlan, model="test-model")

    def test_invalid_json_with_schema_still_raises(self, monkeypatch):
        from src.agents.llm_client import OllamaJSONParseError

        client = self._patched_client(monkeypatch, "not json at all {")
        with pytest.raises(OllamaJSONParseError):
            client.generate_json("sys", "user", schema=SeriesPlan, model="test-model")

    def test_no_schema_legacy_path_still_works(self, monkeypatch):
        import json
        client = self._patched_client(
            monkeypatch, json.dumps({"any": "shape", "is": "fine"}),
        )
        result = client.generate_json("sys", "user", model="test-model")  # no schema kwarg
        assert result == {"any": "shape", "is": "fine"}

    def test_fence_stripping_happens_before_schema_validation(self, monkeypatch):
        import json
        wrapped = "```json\n" + json.dumps(VALID_SERIES_PLAN) + "\n```"
        client = self._patched_client(monkeypatch, wrapped)
        result = client.generate_json("sys", "user", schema=SeriesPlan, model="test-model")
        assert result["theme"] == VALID_SERIES_PLAN["theme"]

    def test_scene_list_schema_round_trip(self, monkeypatch):
        import json
        payload = [VALID_SCENE, {**VALID_SCENE, "pose": "standing"}]
        client = self._patched_client(monkeypatch, json.dumps(payload))
        result = client.generate_json("sys", "user", schema=SceneList, model="test-model")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["pose"] == VALID_SCENE["pose"]

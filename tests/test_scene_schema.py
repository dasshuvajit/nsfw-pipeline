"""Tests for the simplified ``Scene`` Pydantic schema (model-agnostic
core) and the new ``SCENE_FACET_SCHEMA_BY_STYLE`` dispatcher.

The old per-family schema-string dispatcher (``_build_scene_schema``)
is gone — family-shaped fields now live in dedicated ``SceneFacet*``
schemas tested in ``test_scene_facet_generator.py``. ``Scene`` itself
is the model-agnostic core only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.scene_generator import REQUIRED_SCENE_FIELDS
from src.agents.schemas import (
    SCENE_FACET_SCHEMA_BY_STYLE,
    Scene,
    SceneFacetFlux2,
    SceneFacetFluxNatural,
    SceneFacetIllustrious,
    SceneFacetPony,
    SceneFacetSDXL,
    SceneList,
)
from src.memory.family_loader import FamilyLoader


# ── REQUIRED_SCENE_FIELDS still exported for callers ────────────────


def test_required_scene_fields_export():
    """The set is kept for the supervisor's preview-glance check.
    It mirrors the Pydantic Scene model's required fields."""
    assert REQUIRED_SCENE_FIELDS == {
        "variation_axis", "pose", "camera", "camera_angle",
        "lighting", "environment_detail", "mood_note",
    }


# ── Scene Pydantic model ────────────────────────────────────────────


def _minimal_scene() -> dict:
    return {
        "variation_axis": "pose",
        "pose": "standing",
        "camera": "medium shot",
        "camera_angle": "eye level",
        "lighting": "soft window light",
        "environment_detail": "studio with grey backdrop",
        "mood_note": "calm",
    }


def test_scene_minimal_validates():
    scene = Scene.model_validate(_minimal_scene())
    assert scene.pose == "standing"
    assert scene.expression is None
    assert scene.composition_intent is None


def test_scene_with_phase_a_intent_fields():
    payload = {
        **_minimal_scene(),
        "composition_intent": "close-up",
        "framing_hint": "rule-of-thirds",
        "audience_target": "deviantart",
    }
    scene = Scene.model_validate(payload)
    assert scene.composition_intent == "close-up"
    assert scene.framing_hint == "rule-of-thirds"
    assert scene.audience_target == "deviantart"


def test_scene_missing_required_field_raises():
    payload = _minimal_scene()
    del payload["pose"]
    with pytest.raises(ValidationError, match=r"(?i)pose"):
        Scene.model_validate(payload)


def test_scene_blank_required_field_raises():
    """``min_length=1`` on required fields rejects empty strings."""
    payload = {**_minimal_scene(), "pose": ""}
    with pytest.raises(ValidationError, match=r"(?i)pose"):
        Scene.model_validate(payload)


def test_scene_strips_whitespace_on_string_fields():
    payload = {**_minimal_scene(), "pose": "  standing  "}
    scene = Scene.model_validate(payload)
    assert scene.pose == "standing"


def test_scene_drops_no_longer_required_family_fields():
    """Family-shaped fields (booru_tags, scene_prose, etc.) are no
    longer part of the Scene schema. If the LLM returns them anyway,
    extra='allow' captures them but they're not Scene-typed fields."""
    scene = Scene.model_validate({
        **_minimal_scene(),
        "booru_tags": "should not be here",
        "camera_spec": "85mm",
        "scene_prose": "she stands",
    })
    # No typed attributes for the family-shaped fields.
    assert not hasattr(scene, "booru_tags") or scene.__pydantic_extra__.get("booru_tags") == "should not be here"
    # The Scene's own typed fields remain clean.
    assert scene.pose == "standing"


# ── SceneList wrapping ──────────────────────────────────────────────


def test_scene_list_validates_array_of_scenes():
    payload = [_minimal_scene(), _minimal_scene()]
    sl = SceneList.model_validate(payload)
    assert len(sl) == 2
    assert sl[0].pose == "standing"


def test_scene_list_rejects_when_one_element_invalid():
    bad = _minimal_scene()
    del bad["pose"]
    with pytest.raises(ValidationError, match=r"(?i)pose"):
        SceneList.model_validate([_minimal_scene(), bad])


# ── SCENE_FACET_SCHEMA_BY_STYLE dispatcher ──────────────────────────


def test_dispatcher_covers_every_family():
    """Every prompt_style declared in families.yaml has a facet schema."""
    loader = FamilyLoader()
    expected = {f.prompt_style for f in loader.list_families()}
    actual = set(SCENE_FACET_SCHEMA_BY_STYLE.keys())
    assert expected == actual, (
        f"Missing facet schemas for: {expected - actual}; "
        f"extra: {actual - expected}"
    )


def test_dispatcher_returns_pydantic_models():
    for style, schema in SCENE_FACET_SCHEMA_BY_STYLE.items():
        # Each entry must be a Pydantic BaseModel subclass.
        assert hasattr(schema, "model_validate"), (
            f"{style}: dispatcher entry is not a Pydantic model"
        )


def test_dispatcher_sdxl_to_sdxl_facet():
    assert SCENE_FACET_SCHEMA_BY_STYLE["sdxl_keywords"] is SceneFacetSDXL


def test_dispatcher_pony_to_pony_facet():
    assert SCENE_FACET_SCHEMA_BY_STYLE["pony_danbooru"] is SceneFacetPony


def test_dispatcher_illustrious_to_illustrious_facet():
    assert SCENE_FACET_SCHEMA_BY_STYLE["illustrious_tags"] is SceneFacetIllustrious


def test_dispatcher_flux_natural_used_by_flux_and_chroma():
    """flux + chroma share the flux_natural composer → same facet schema."""
    loader = FamilyLoader()
    flux = loader.get_family("flux")
    chroma = loader.get_family("chroma")
    assert flux.prompt_style == "flux_natural"
    assert chroma.prompt_style == "flux_natural"
    assert SCENE_FACET_SCHEMA_BY_STYLE[flux.prompt_style] is SceneFacetFluxNatural
    assert SCENE_FACET_SCHEMA_BY_STYLE[chroma.prompt_style] is SceneFacetFluxNatural


def test_dispatcher_flux2_to_flux2_facet():
    assert SCENE_FACET_SCHEMA_BY_STYLE["flux2_prose"] is SceneFacetFlux2


# ── Per-facet shape ─────────────────────────────────────────────────


def test_sdxl_facet_requires_camera_spec_and_clothing():
    f = SceneFacetSDXL.model_validate(
        {"camera_spec": "85mm f/1.4", "clothing": "ivory silk slip"}
    )
    assert f.camera_spec == "85mm f/1.4"
    with pytest.raises(ValidationError):
        SceneFacetSDXL.model_validate({"camera_spec": "85mm f/1.4"})  # no clothing


def test_pony_facet_requires_booru_tags_source_tag_optional():
    f = SceneFacetPony.model_validate({"booru_tags": "long_hair, blue_eyes"})
    assert f.booru_tags == "long_hair, blue_eyes"
    assert f.source_tag is None
    # source_tag accepted when present
    f2 = SceneFacetPony.model_validate({
        "booru_tags": "x", "source_tag": "source_photograph",
    })
    assert f2.source_tag == "source_photograph"


def test_illustrious_facet_requires_booru_and_prose():
    f = SceneFacetIllustrious.model_validate({
        "booru_tags": "long_hair, parted_lips",
        "scene_prose": "She stands by the window in soft afternoon light.",
    })
    assert f.scene_prose.startswith("She stands")
    with pytest.raises(ValidationError):
        SceneFacetIllustrious.model_validate({"booru_tags": "x"})  # no prose


def test_flux_natural_facet_requires_scene_prose_only():
    """Round-22 — minimum word count is 20 (was implicit at 1+ chars
    pre-round-22). Real scene_prose is 40-90 words for the target
    band; 25-word fixture below comfortably exceeds the floor."""
    f = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "She leans against the parlour wall in soft golden-hour "
            "rim light. Her gaze drifts to the distant window where "
            "long shadows fall across the antique velvet upholstery."
        ),
    })
    assert f.scene_prose.startswith("She leans")


def test_flux_natural_word_band_rejects_too_short():
    """Round-22 (2026-05-22) — scene_prose word count must be ≥20
    words. Pre-round-22 the spec was 1-3 sentences ~30 words; the
    facet LLM saturated trying to compress everything into too little
    space and nulled structured fields. New band 20-140 (target
    40-90)."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"prose-family hard band is 20"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": "She leans against the wall in soft light.",  # 9 words
        })


def test_flux_natural_word_band_rejects_too_long():
    """Round-22 — hard cap at 140 words; the band protects against
    LLM drift toward giant narrative paragraphs that drown the
    structured-tag picks."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"prose-family hard band is 20"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": " ".join(["word"] * 150),  # 150 words
        })


def test_flux_natural_word_band_warn_outside_target_inside_slack(caplog):
    """Round-22 — soft warn when prose is inside the 20-140 hard band
    but outside the 40-90 target band. Doesn't fail validation; just
    logs at WARNING for operator visibility."""
    import logging
    caplog.set_level(logging.WARNING, logger="src.agents.schemas")
    # 25 words — under 40, over 20.
    SceneFacetFluxNatural.model_validate({
        "scene_prose": " ".join(["word"] * 25),
    })
    assert any(
        "outside the 40–90 target band" in rec.message
        for rec in caplog.records
    ), (
        f"expected warn log about target band drift; got "
        f"{[r.message for r in caplog.records]!r}"
    )


def test_flux2_facet_requires_prose_qa_fields_optional():
    """subject_focus is validated but optional — it's a QA signal,
    not persisted to scene_facets table. Phase 4b promoted
    lighting_directive to a structured enum-tag field."""
    f = SceneFacetFlux2.model_validate({
        "scene_prose": (
            "Mira sits on a low concrete bench in a stark minimalist "
            "loft, raven hair falling softly past her shoulders. "
            "North-facing window light wraps gently around her face "
            "from the left, illuminating natural skin texture. The "
            "atmosphere is quiet, intimate, pensive late-afternoon."
        ),
    })
    assert f.scene_prose.startswith("Mira")
    assert f.subject_focus is None
    assert f.lighting_directive is None

    # QA fields accepted when present — note word-count must be
    # within the 25–95 BFL Klein band (Phase 4b validator).
    f2 = SceneFacetFlux2.model_validate({
        "scene_prose": " ".join(["word"] * 50),
        "subject_focus": "Mira, 28, raven hair",
        "lighting_directive": "north-facing window, soft, 5500 K",
    })
    assert f2.subject_focus is not None


def test_facet_schemas_strip_whitespace():
    f = SceneFacetSDXL.model_validate({
        "camera_spec": "  85mm f/1.4  ",
        "clothing": "  silk dress  ",
    })
    assert f.camera_spec == "85mm f/1.4"
    assert f.clothing == "silk dress"


# ── Phase 4b: model_validator family invariants ────────────────────


def test_pony_validator_rejects_score_tokens():
    """SceneFacetPony rejects facets containing score_* — composer
    prepends the 6-tier score chain."""
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="score_"):
        SceneFacetPony.model_validate({
            "booru_tags": "score_9, 1girl, looking_at_viewer",
        })


def test_pony_validator_rejects_source_pony():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="source_pony"):
        SceneFacetPony.model_validate({
            "booru_tags": "source_pony, 1girl, long_hair",
        })


def test_pony_validator_warns_on_missing_source(caplog):
    """Pony facet without source_tag and no source_* in booru_tags →
    one WARNING (the moved-from-builder logic)."""
    import logging
    with caplog.at_level(logging.WARNING):
        SceneFacetPony.model_validate({
            "booru_tags": "1girl, long_hair, looking_at_viewer",
        })
    assert any(
        "missing source_*" in r.message for r in caplog.records
    )


def test_pony_validator_no_warn_when_source_in_facet(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        SceneFacetPony.model_validate({
            "booru_tags": "1girl, long_hair",
            "source_tag": "source_photograph",
        })
    assert not any(
        "missing source_*" in r.message for r in caplog.records
    )


def test_illustrious_validator_rejects_quality_suffix():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="quality-suffix"):
        SceneFacetIllustrious.model_validate({
            "booru_tags": "1girl, masterpiece, looking_at_viewer",
            "scene_prose": "She stands in soft window light.",
        })


def test_illustrious_validator_rejects_quality_in_prose():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="quality-suffix"):
        SceneFacetIllustrious.model_validate({
            "booru_tags": "1girl, looking_at_viewer",
            "scene_prose": "She stands, very aesthetic, in soft light.",
        })


def test_sdxl_validator_rejects_avoid_words():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="avoid-word"):
        SceneFacetSDXL.model_validate({
            "camera_spec": "85mm masterpiece f/1.4",
            "clothing": "silk dress",
        })


def test_sdxl_validator_rejects_avoid_in_clothing():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="avoid-word"):
        SceneFacetSDXL.model_validate({
            "camera_spec": "85mm f/1.4",
            "clothing": "silk dress, 8k detail",
        })


def test_flux_natural_validator_rejects_weighting_syntax():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="weighting"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": "She stands by the (window:1.3) at sunset.",
        })


def test_flux_natural_validator_rejects_underscored_tags():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="underscored"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": "She has long_hair and stands by the window.",
        })


def test_flux_natural_validator_rejects_tag_soup():
    """Heuristic: too many commas in one sentence = tag-soup style."""
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="tag list"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": (
                "girl, brown hair, blue eyes, soft light, window, "
                "dress, smile, evening."
            ),
        })


def test_flux_natural_validator_accepts_normal_prose():
    """Sanity: ordinary 2-4-sentence prose passes through untouched.
    Round-22 — prose length expanded from "1-3 sentences" (~30 words)
    to 2-4 sentences targeting 40-90 words; fixture below is ~45
    words which lands cleanly in the target band."""
    f = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "She stands by the parlour window in late-afternoon "
            "light, one hand resting lightly on the velvet sill. "
            "The room glows in warm honey tones, soft shadows "
            "tracing the curves of antique upholstery behind her. "
            "Her gaze drifts toward the distant horizon, "
            "contemplative and at ease."
        ),
    })
    assert "window" in f.scene_prose


def test_flux2_validator_rejects_too_short_prose():
    """Below 25-word floor → fail (forces retry)."""
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="25–95"):
        SceneFacetFlux2.model_validate({
            "scene_prose": "She stands by the window. Soft light.",  # 7 words
        })


def test_flux2_validator_rejects_too_long_prose():
    """Above 95-word ceiling → fail."""
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="25–95"):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 100),
        })


def test_flux2_validator_warns_outside_30_80_band(caplog):
    """26-word prose passes (in 25-95 slack) but logs WARNING."""
    import logging
    with caplog.at_level(logging.WARNING):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 26),
        })
    assert any(
        "outside the 30–80 BFL target band" in r.message
        for r in caplog.records
    )


def test_flux2_validator_silent_inside_target_band(caplog):
    """50-word prose (well within 30-80) → no warnings."""
    import logging
    with caplog.at_level(logging.WARNING):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 50),
        })
    assert not any(
        "outside the 30–80" in r.message for r in caplog.records
    )

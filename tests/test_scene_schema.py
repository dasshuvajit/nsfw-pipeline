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
        "scene_prose": "She stands by the window in soft afternoon light. The room glows in warm honey tones, deep umber shadows in the corners, gilt-framed paintings on every wall and a single gas lamp catching her cheek. Her gaze drifts toward the distant horizon, contemplative and at ease in the silent room. She is fully nude, her natural curves rendered in soft chiaroscuro relief against the heavy oxblood leather chair behind her. A vinyl record turns slowly on a nearby phonograph, the only sound in the room. Shot in the Rembrandt lighting tradition with a single warm key light shaping every form.",
    })
    assert f.scene_prose.startswith("She stands")
    with pytest.raises(ValidationError):
        SceneFacetIllustrious.model_validate({"booru_tags": "x"})  # no prose


def test_flux_natural_facet_requires_scene_prose_only():
    """Dual-write pivot — minimum word count is 100 (was 20). The
    LLM's scene_prose IS the prompt body and must cover all axes."""
    f = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "She leans against the parlour wall in soft golden-hour rim "
            "light, one hand resting on the antique velvet upholstery and "
            "the other tracing the spine of a leather-bound book on the "
            "side table. Her gaze drifts toward the tall window where "
            "long shadows fall across the gilt-framed oil paintings that "
            "line every wall of the room. The space glows in warm honey "
            "tones with deep umber blacks gathering in the corners, a "
            "single warm flesh-tone highlight catching her cheek and bare "
            "shoulder. She wears nothing, her body a quiet contemplative "
            "form against the heavy oxblood leather chair behind her. The "
            "room itself feels caught in slow afternoon stillness, "
            "contemplative and unhurried, a private moment held in slow "
            "golden afternoon light. The gas lamp on the far wall casts "
            "a faint warm reflection across the polished floor."
        ),  # ~135 words
    })
    assert f.scene_prose.startswith("She leans")


def test_flux_natural_word_band_rejects_too_short():
    """Dual-write pivot iter1 calibration (2026-05-23) — band adjusted
    to LLM empirical range after series_628bdf54cec6 showed DavidAU
    12B's natural floor is ~65 words. New band: 40-350 hard,
    100-250 target."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"prose-family hard band is 40"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": " ".join(["word"] * 30),  # 30 words — too short
        })


def test_flux_natural_word_band_rejects_too_long():
    """Hard cap at 350 words."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"prose-family hard band is 40"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": " ".join(["word"] * 400),
        })


def test_flux_natural_word_band_warn_outside_target_inside_slack(caplog):
    """Soft warn when prose is inside the 40-350 hard band but outside
    the 100-250 target band."""
    import logging
    caplog.set_level(logging.WARNING, logger="src.agents.schemas")
    # 70 words — under 100, over 60.
    SceneFacetFluxNatural.model_validate({
        "scene_prose": " ".join(["word"] * 70),
    })
    assert any(
        "outside the 100-250 target band" in rec.message
        for rec in caplog.records
    ), (
        f"expected warn log about target band drift; got "
        f"{[r.message for r in caplog.records]!r}"
    )


def test_flux_natural_word_band_silent_inside_target(caplog):
    """Inside 100-250 target band, no warning fires."""
    import logging
    caplog.set_level(logging.WARNING, logger="src.agents.schemas")
    # 150 words — sweet spot.
    SceneFacetFluxNatural.model_validate({
        "scene_prose": " ".join(["word"] * 150),
    })
    band_warns = [
        rec for rec in caplog.records
        if "target band" in rec.message
    ]
    assert not band_warns, (
        f"150-word prose should land inside 100-250 target band — no warn "
        f"expected, got: {[r.message for r in band_warns]!r}"
    )


# ── Hedge-phrase ban (2026-05-23 audit follow-up) ──────────────────────


def test_flux_natural_rejects_visible_through_fabric():
    """External auditor flagged 'breasts visible through fabric' as a
    partial-coverage producer at T4. Schema must reject so retry fires."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"hedge / partial-coverage"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": (
                "A confident woman stands in the studio. Her natural "
                "breasts visible through fabric. Wide hips beneath loose "
                "silk hang low. Mature curves on display in soft side "
                "lighting. The defiant gaze meets the camera directly. "
                "Soft amber wash from the window catches the moment "
                "with intimate photographic composition and natural "
                "skin texture detail throughout the bare frame."
            ),
        })


def test_flux_natural_rejects_barely_conceals():
    """Coverage hedge: 'barely conceals' phrasing implies clothing."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"hedge / partial-coverage"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": (
                "She stands by the window with sheer silk wrapped low. "
                "It barely conceals her body. Natural medium breasts "
                "bare against the cool morning light. Wide hips and "
                "thick thighs catch the warm glow. Confident direct "
                "gaze meets the camera lens. The room glows in honey "
                "tones with deep shadows gathering in the corners of "
                "this intimate photographic composition."
            ),
        })


def test_flux_natural_rejects_visible_against_bedding():
    """The 'breasts visible against bedding' pattern — auditor flagged
    this as a passive phrasing that produces coverage. The subject IS
    the focus, not 'visible against' the prop."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match=r"hedge / partial-coverage"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": (
                "A confident woman reclines on rumpled white sheets. "
                "One knee raised in a languid pose. Her natural medium "
                "breasts visible against the crisp bedding. Wide hips "
                "bare in the late afternoon glow. Lips parted in a "
                "defiant expression at the camera. Warm window light "
                "gilds her body in this intimate photographic "
                "composition with natural skin texture."
            ),
        })


def test_flux_natural_accepts_assertive_nudity():
    """Counter-example: 'bare breasts rest against the sheets' is the
    auditor's recommended phrasing — schema must accept."""
    f = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "A confident woman reclines on rumpled white sheets, one knee "
            "raised. Her bare breasts rest against the crisp bedding, "
            "exposed vulva softly lit by late afternoon window light. Wide "
            "hips and thick thighs naked against the linen, defiant gaze "
            "direct to the camera in this intimate photographic composition "
            "with natural skin texture."
        ),
    })
    assert f.scene_prose.startswith("A confident")


def test_flux_natural_accepts_sheer_curtains():
    """Single-word 'sheer' is NOT banned — sheer curtains / sheer cliff
    face / sheer drop are fine. The ban targets compound coverage
    phrases only."""
    f = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "A confident woman stands fully nude in the bedroom, the "
            "morning light filtering through sheer curtains and falling "
            "across her exposed body. Bare breasts and naked hips lit by "
            "soft window glow, mature curves on display, defiant gaze "
            "direct to the camera lens in this intimate photographic "
            "composition with natural skin texture."
        ),
    })
    assert f.scene_prose.startswith("A confident")


def test_flux2_facet_requires_prose_qa_fields_optional():
    """subject_focus is validated but optional — it's a QA signal,
    not persisted to scene_facets table. Dual-write pivot — word
    band 100-350; this fixture is ~150 words."""
    _long = " ".join([
        "Mira sits on a low concrete bench in a stark minimalist loft,",
        "raven hair falling softly past her shoulders, body fully nude",
        "and rendered in the cool quiet of late-afternoon natural light.",
        "North-facing window light wraps gently around her face from the",
        "left, illuminating natural skin texture and tracing the line of",
        "her shoulder. The atmosphere is quiet, intimate, pensive late-",
        "afternoon. The room glows in pale neutral tones, deep umber",
        "shadows gathering in the corners, polished concrete floor",
        "catching faint reflections. Her gaze drifts toward the floor,",
        "contemplative and at ease. The mood is fine-art figure study —",
        "sculptural, painterly, the body as quiet form against minimal",
        "architecture. Soft natural light only, no artificial source,",
        "the room caught in unhurried golden afternoon stillness.",
    ])
    f = SceneFacetFlux2.model_validate({"scene_prose": _long})
    assert f.scene_prose.startswith("Mira")
    assert f.subject_focus is None
    assert f.lighting_directive is None

    # QA fields accepted when present — note word-count must be
    # within the 100-350 dual-write band.
    f2 = SceneFacetFlux2.model_validate({
        "scene_prose": " ".join(["word"] * 200),
        "subject_focus": "Mira, 28, raven hair",
        "lighting_directive": "LIGHT_WINDOW_SIDE",
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
            "scene_prose": "She stands in soft window light. The room glows in warm honey tones, deep umber shadows in the corners, gilt-framed paintings on every wall and a single gas lamp catching her cheek. Her gaze drifts toward the distant horizon, contemplative and at ease in the silent room. She is fully nude, her natural curves rendered in soft chiaroscuro relief against the heavy oxblood leather chair behind her. A vinyl record turns slowly on a nearby phonograph, the only sound in the room. Shot in the Rembrandt lighting tradition with a single warm key light shaping every form.",
        })


def test_illustrious_validator_rejects_quality_in_prose():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="quality-suffix"):
        SceneFacetIllustrious.model_validate({
            "booru_tags": "1girl, looking_at_viewer",
            "scene_prose": "She stands, very aesthetic, in soft light. The room glows in warm honey tones, deep umber shadows in the corners, gilt-framed paintings on every wall and a single gas lamp catching her cheek. Her gaze drifts toward the distant horizon, contemplative and at ease in the silent room. She is fully nude, her natural curves rendered in soft chiaroscuro relief against the heavy oxblood leather chair behind her. A vinyl record turns slowly on a nearby phonograph, the only sound in the room. Shot in the Rembrandt lighting tradition with a single warm key light shaping every form.",
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
            "scene_prose": "She stands by the (window:1.3) at sunset. The room glows in warm honey tones, deep umber shadows in the corners, gilt-framed paintings on every wall and a single gas lamp catching her cheek. Her gaze drifts toward the distant horizon, contemplative and at ease in the silent room. She is fully nude, her natural curves rendered in soft chiaroscuro relief against the heavy oxblood leather chair behind her. A vinyl record turns slowly on a nearby phonograph, the only sound in the room. Shot in the Rembrandt lighting tradition with a single warm key light shaping every form.",
        })


def test_flux_natural_validator_rejects_underscored_tags():
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="underscored"):
        SceneFacetFluxNatural.model_validate({
            "scene_prose": "She has long_hair and stands by the window. The room glows in warm honey tones, deep umber shadows in the corners, gilt-framed paintings on every wall and a single gas lamp catching her cheek. Her gaze drifts toward the distant horizon, contemplative and at ease in the silent room. She is fully nude, her natural curves rendered in soft chiaroscuro relief against the heavy oxblood leather chair behind her. A vinyl record turns slowly on a nearby phonograph, the only sound in the room. Shot in the Rembrandt lighting tradition with a single warm key light shaping every form.",
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
    """Dual-write pivot — fixture is ~175 words, dead-center in the
    150-300 target band. Single coherent paragraph weaving subject
    + pose + lighting + env + mood + style."""
    f = SceneFacetFluxNatural.model_validate({
        "scene_prose": (
            "A mature woman in her late 30s with dark wild hair stands "
            "by the parlour window in late-afternoon light, one hand "
            "resting lightly on the velvet sill. The room glows in warm "
            "honey tones — soft shadows trace the curves of antique "
            "upholstery behind her, gilt-framed oil paintings lining "
            "every wall, gas-lamp amber glow from the corner. Her bare "
            "shoulder catches the slanting sun in a single warm "
            "highlight, the rest of her body falling into deep umber "
            "shadow against the oxblood leather chair beside her. She "
            "is fully nude, her natural curves rendered in chiaroscuro "
            "relief, a contemplative form held in stillness. Her gaze "
            "drifts toward the distant horizon beyond the window, "
            "contemplative and at ease in the quiet room. Shot in the "
            "Rembrandt lighting tradition — a single warm key light "
            "shaping every form, the falloff into dark velvet behind."
        ),  # ~175 words
    })
    assert "window" in f.scene_prose


def test_flux2_validator_rejects_too_short_prose():
    """Dual-write pivot iter2 calibration — band 40-350 hard."""
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="40-350"):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 30),
        })


def test_flux2_validator_rejects_too_long_prose():
    """Above 350-word ceiling → fail."""
    from pydantic import ValidationError as VE
    with pytest.raises(VE, match="40-350"):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 400),
        })


def test_flux2_validator_warns_outside_30_80_band(caplog):
    """70-word prose passes (in 40-350 slack) but logs WARNING
    when outside 100-250 target band."""
    import logging
    with caplog.at_level(logging.WARNING):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 70),
        })
    assert any(
        "outside the 100-250 target band" in r.message
        for r in caplog.records
    )


def test_flux2_validator_silent_inside_target_band(caplog):
    """200-word prose (well within 150-300) → no warnings."""
    import logging
    with caplog.at_level(logging.WARNING):
        SceneFacetFlux2.model_validate({
            "scene_prose": " ".join(["word"] * 200),
        })
    assert not any(
        "target band" in r.message for r in caplog.records
    )

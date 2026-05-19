"""Round-6 regression: structured-tag schema body lists REQUIRED
fields first, OPTIONAL polish last.

Pre-round-6 the user-prompt schema body led with 6 OPTIONAL polish
fields (realism_camera → realism_framing) and then listed the
tier-REQUIRED fields. Both Cydonia heretic and Magnum v4 calibrated
to "this whole block is optional polish" and nulled the REQUIRED
fields under constrained decoding — 100% of facets in the
2026-05-19 A/B run had lighting_directive / mood_aesthetic /
narrative_moment / environment_setting / environment_atmosphere /
nsfw_anatomy / nsfw_act blank, even after retry-nudge.

The fix:
  - REQUIRED fields lead, OPTIONAL polish trails.
  - Each field gets a bracket-prefixed marker the LLM's first-token
    attention picks up: [REQUIRED — every tier], [REQUIRED — T3+],
    [REQUIRED — T4 only], [OPTIONAL].

These tests would fail if the order were reverted to put OPTIONAL
fields first, OR if a [REQUIRED] marker were stripped.
"""

from __future__ import annotations

import re

from src.agents.scene_facet_generator import (
    _SCHEMA_BODY_BY_STYLE,
    _STRUCTURED_TAG_BODY_NON_PONY,
    _STRUCTURED_TAG_BODY_PONY,
    _TIER_REQUIRED_FIELDS,
    _USER_PROMPT_TEMPLATE,
)


# Fields tier-required at the most-permissive tier (T4 carries the
# full set). All MUST appear before the first [OPTIONAL] marker in
# the structured-tag body.
_T4_REQUIRED_FIELDS = set(_TIER_REQUIRED_FIELDS["T4_explicit"])


def _field_position(body: str, field: str) -> int:
    """Return the character offset where the field name first appears
    in the schema body. -1 if missing."""
    match = re.search(rf'"{re.escape(field)}":', body)
    return match.start() if match else -1


def _first_optional_position(body: str) -> int:
    """Character offset of the first [OPTIONAL] marker. -1 if none."""
    match = re.search(r"\[OPTIONAL\b", body)
    return match.start() if match else -1


def test_non_pony_required_fields_lead_optional():
    """Every T4-required field appears before any [OPTIONAL] marker
    in the non-Pony schema body."""
    first_optional = _first_optional_position(_STRUCTURED_TAG_BODY_NON_PONY)
    assert first_optional > 0, (
        "Schema body must contain at least one [OPTIONAL] marker — "
        "otherwise the round-6 ordering claim is vacuous."
    )
    for field in _T4_REQUIRED_FIELDS:
        pos = _field_position(_STRUCTURED_TAG_BODY_NON_PONY, field)
        assert pos > -1, f"non-Pony body missing required field {field!r}"
        assert pos < first_optional, (
            f"non-Pony body lists tier-required field {field!r} at "
            f"offset {pos} which is AFTER the first [OPTIONAL] marker "
            f"at offset {first_optional}. Round-6 contract: REQUIRED "
            f"first, OPTIONAL last."
        )


def test_pony_required_fields_lead_optional():
    """Same ordering contract for the Pony body. Pony omits the 6
    realism enum fields + composition_principle but participates in
    every tier-required structured tag."""
    first_optional = _first_optional_position(_STRUCTURED_TAG_BODY_PONY)
    assert first_optional > 0
    # Pony skips photographer_ref / art_movement at the series level
    # but still has lighting/mood/narrative/env/nsfw at the scene
    # level — all REQUIRED at T4 per _TIER_REQUIRED_FIELDS.
    for field in _T4_REQUIRED_FIELDS:
        pos = _field_position(_STRUCTURED_TAG_BODY_PONY, field)
        assert pos > -1, f"Pony body missing required field {field!r}"
        assert pos < first_optional, (
            f"Pony body lists tier-required field {field!r} after the "
            f"first [OPTIONAL] marker — round-6 ordering contract "
            f"broken."
        )


def test_non_pony_body_has_required_markers():
    """Every T4-required field carries a `[REQUIRED — ...]` marker."""
    for field in _T4_REQUIRED_FIELDS:
        # Find the line carrying that field and assert it has [REQUIRED
        match = re.search(
            rf'"{re.escape(field)}":\s*"\[REQUIRED',
            _STRUCTURED_TAG_BODY_NON_PONY,
        )
        assert match is not None, (
            f"non-Pony field {field!r} is in _TIER_REQUIRED_FIELDS but "
            f"its schema-body line lacks a `[REQUIRED` marker. The "
            f"LLM relies on the bracket prefix for first-token "
            f"attention; missing it brings back round-6 omission "
            f"behaviour."
        )


def test_pony_body_has_required_markers():
    for field in _T4_REQUIRED_FIELDS:
        match = re.search(
            rf'"{re.escape(field)}":\s*"\[REQUIRED',
            _STRUCTURED_TAG_BODY_PONY,
        )
        assert match is not None, (
            f"Pony field {field!r} lacks `[REQUIRED` marker"
        )


def test_optional_fields_have_optional_marker():
    """Inverse: every [OPTIONAL] field carries the marker explicitly."""
    optional_lines = [
        line for line in _STRUCTURED_TAG_BODY_NON_PONY.split("\n")
        if "[OPTIONAL" in line
    ]
    # Round-6 ships 9 [OPTIONAL] fields in the non-Pony body:
    # nsfw_posture, environment_prop, composition_principle,
    # realism_camera, realism_lens, realism_film_stock,
    # art_style_reference, realism_angle, realism_framing.
    assert len(optional_lines) == 9, (
        f"non-Pony body expected 9 [OPTIONAL] fields, got "
        f"{len(optional_lines)} — schema body has drifted from "
        f"round-6 contract."
    )


def test_all_required_offsets_before_all_optional_offsets():
    """Round-7 NIT-2: tighten the ordering contract — assert that
    EVERY [REQUIRED] marker offset is less than EVERY [OPTIONAL]
    marker offset, not just less than the FIRST [OPTIONAL].

    Pre-fix this test would have missed a REQUIRED field accidentally
    placed at the tail of the body but still before some other
    OPTIONAL — the original `first_optional` check only catches the
    very last violation.
    """
    for label, body in [
        ("non_pony", _STRUCTURED_TAG_BODY_NON_PONY),
        ("pony", _STRUCTURED_TAG_BODY_PONY),
    ]:
        required_offsets = [
            m.start() for m in re.finditer(r"\[REQUIRED\b", body)
        ]
        optional_offsets = [
            m.start() for m in re.finditer(r"\[OPTIONAL\b", body)
        ]
        assert required_offsets, f"{label}: no [REQUIRED] markers found"
        assert optional_offsets, f"{label}: no [OPTIONAL] markers found"
        max_req = max(required_offsets)
        min_opt = min(optional_offsets)
        assert max_req < min_opt, (
            f"{label}: [REQUIRED] block must end before any [OPTIONAL] "
            f"begins. max_required={max_req}, min_optional={min_opt}"
        )


def test_required_lighting_directive_is_first_required():
    """lighting_directive is the single biggest-impact field. Round-6
    puts it first in the structured-tag block so the LLM's first
    attention sees [REQUIRED — every tier] for the most-critical
    field."""
    body = _STRUCTURED_TAG_BODY_NON_PONY
    lighting_pos = _field_position(body, "lighting_directive")
    mood_pos = _field_position(body, "mood_aesthetic")
    narrative_pos = _field_position(body, "narrative_moment")
    assert lighting_pos > -1
    assert mood_pos > lighting_pos
    assert narrative_pos > mood_pos
    # All three before any [OPTIONAL]
    first_optional = _first_optional_position(body)
    assert lighting_pos < first_optional
    assert mood_pos < first_optional
    assert narrative_pos < first_optional


def test_user_prompt_template_has_required_lead_in():
    """The user-prompt template must explain the [REQUIRED] /
    [OPTIONAL] contract BEFORE the JSON schema block, so the LLM
    sees the operating principle when reading the schema."""
    assert "[REQUIRED" in _USER_PROMPT_TEMPLATE, (
        "User-prompt template missing [REQUIRED] lead-in — round-6 "
        "fix expects an explicit explanation of the contract."
    )
    assert "[OPTIONAL]" in _USER_PROMPT_TEMPLATE
    assert "FIRST character" in _USER_PROMPT_TEMPLATE, (
        "User-prompt template missing the JSON-preamble guard — "
        "needed so Cydonia stops prefixing 'Sure, here's the JSON:'."
    )


def test_user_prompt_template_renders_with_schema_body():
    """The template must render without KeyError when given the
    standard substitution dict the engine passes.

    Round-8 note: the schema_body shown to the LLM is now the
    tier-active rewrite (conditionals collapsed). The raw
    _SCHEMA_BODY_BY_STYLE template still carries the conditional
    markers (round-6) — the rewrite happens at compose time."""
    rendered = _USER_PROMPT_TEMPLATE.format(
        content_level="T4_explicit",
        tier_required_list="  - lighting_directive\n  - mood_aesthetic",
        scene_core_json='{"pose":"x"}',
        family_id="chroma",
        prompt_style="flux_natural",
        schema_body=_SCHEMA_BODY_BY_STYLE["flux_natural"],
    )
    # The RAW schema body still carries the conditional markers
    # (round-6 shape); the tier-active rewrite is applied later.
    assert "[REQUIRED — every tier]" in rendered
    assert "[REQUIRED — T3+]" in rendered
    assert "[REQUIRED — T4 only]" in rendered
    # Tier-required list must appear before the schema body.
    list_pos = rendered.find("- lighting_directive")
    schema_pos = rendered.find("Produce the family-shaped fields")
    assert list_pos > 0 and list_pos < schema_pos, (
        "tier_required_list must render BEFORE the schema body so "
        "the LLM sees the per-tier mandate first."
    )
    # No stray template artifacts.
    assert "{n}" not in rendered, (
        "User prompt template has unfilled `{n}` literal — bad "
        "format-string escaping (round-6 caught this)."
    )
    assert "{{" not in rendered  # double-braces should reduce to single


# ── Round-8: tier-active schema body rewrite ───────────────────────


def test_tier_active_body_T4_collapses_all_conditionals():
    """At T4_explicit, every `[REQUIRED — *]` marker collapses to
    bare `[REQUIRED]`. Round-8 fix to make Cydonia stop interpreting
    the conditional as a hedge."""
    from src.agents.scene_facet_generator import _make_tier_active_schema_body
    out = _make_tier_active_schema_body(
        _STRUCTURED_TAG_BODY_NON_PONY, "T4_explicit"
    )
    assert "[REQUIRED — every tier]" not in out
    assert "[REQUIRED — T3+]" not in out
    assert "[REQUIRED — T4 only]" not in out
    # All 7 tier-required fields show bare [REQUIRED] now
    for fld in _T4_REQUIRED_FIELDS:
        m = re.search(rf'"{re.escape(fld)}":\s*"\[REQUIRED\]', out)
        assert m is not None, (
            f"At T4, field {fld!r} should carry bare [REQUIRED] marker"
        )


def test_tier_active_body_T3_demotes_T4_only():
    """At T3_artnude: every-tier + T3+ markers collapse to [REQUIRED];
    T4-only marker DEMOTES to [OPTIONAL (not at this tier)]."""
    from src.agents.scene_facet_generator import _make_tier_active_schema_body
    out = _make_tier_active_schema_body(
        _STRUCTURED_TAG_BODY_NON_PONY, "T3_artnude"
    )
    # T4-only fields demoted
    assert "not required at this tier" in out, (
        "T3 must demote T4-only fields with an explicit demotion note"
    )
    # nsfw_act specifically (only T4-only field) demoted
    m = re.search(r'"nsfw_act":\s*"\[OPTIONAL', out)
    assert m is not None, "nsfw_act should be demoted to OPTIONAL at T3"
    # nsfw_anatomy (T3+ required) stays REQUIRED
    m = re.search(r'"nsfw_anatomy":\s*"\[REQUIRED\]', out)
    assert m is not None


def test_tier_active_body_T2_demotes_all_T3_and_T4():
    """At T2_implied: only every-tier required fields stay REQUIRED;
    T3+ and T4-only both demote."""
    from src.agents.scene_facet_generator import _make_tier_active_schema_body
    out = _make_tier_active_schema_body(
        _STRUCTURED_TAG_BODY_NON_PONY, "T2_implied"
    )
    # Every-tier fields stay REQUIRED
    for fld in ("lighting_directive", "mood_aesthetic", "narrative_moment"):
        m = re.search(rf'"{re.escape(fld)}":\s*"\[REQUIRED\]', out)
        assert m is not None, f"{fld} should remain [REQUIRED] at T2"
    # T3+ fields demoted
    for fld in ("environment_setting", "environment_atmosphere",
                "nsfw_anatomy", "nsfw_act"):
        m = re.search(rf'"{re.escape(fld)}":\s*"\[OPTIONAL', out)
        assert m is not None, (
            f"{fld} should be demoted to [OPTIONAL at T2 (was T3+/T4-only)"
        )


def test_tier_active_body_unknown_tier_passthrough():
    """Unknown content_level — leave body unchanged (back-compat)."""
    from src.agents.scene_facet_generator import _make_tier_active_schema_body
    out = _make_tier_active_schema_body(
        _STRUCTURED_TAG_BODY_NON_PONY, "T9_unknown"
    )
    assert out == _STRUCTURED_TAG_BODY_NON_PONY


def test_tier_active_body_preserves_field_count_and_order():
    """Rewrite must not duplicate / drop any field."""
    from src.agents.scene_facet_generator import _make_tier_active_schema_body
    for tier in ("T1_suggestive", "T2_implied", "T3_artnude",
                 "T4_explicit"):
        for label, body in [
            ("non_pony", _STRUCTURED_TAG_BODY_NON_PONY),
            ("pony", _STRUCTURED_TAG_BODY_PONY),
        ]:
            out = _make_tier_active_schema_body(body, tier)
            orig_fields = re.findall(r'"(\w+)":', body)
            new_fields = re.findall(r'"(\w+)":', out)
            assert orig_fields == new_fields, (
                f"At {tier} on {label}: field order/count changed by "
                f"rewrite. orig={orig_fields} new={new_fields}"
            )


def test_no_defunct_conditional_markers_in_llm_visible_text():
    """Round-8 verifier IMPORTANT-1 + IMPORTANT-2: after round-8
    rewrites every conditional marker in the schema body to
    unconditional [REQUIRED] / [OPTIONAL], the surrounding
    CRITICAL paragraph AND both system prompts must NOT reference
    the defunct conditional markers — otherwise the LLM is told to
    look for needles that no longer exist, re-introducing the hedge
    round-8 was designed to eliminate."""
    from src.agents.scene_facet_generator import (
        SYSTEM_PROMPT, PONY_BOORU_SYSTEM_PROMPT,
        _USER_PROMPT_TEMPLATE,
    )
    defunct = (
        "[REQUIRED — every tier]",
        "[REQUIRED — T3+]",
        "[REQUIRED — T4 only]",
    )
    for marker in defunct:
        assert marker not in _USER_PROMPT_TEMPLATE, (
            f"User-prompt template references defunct conditional "
            f"marker {marker!r} after round-8. LLM is told to look "
            f"for markers that no longer exist in the body."
        )
        assert marker not in SYSTEM_PROMPT, (
            f"SYSTEM_PROMPT references defunct conditional marker "
            f"{marker!r}. Round-8 demands the system prompt describe "
            f"only the post-rewrite shape [REQUIRED]/[OPTIONAL]."
        )
        assert marker not in PONY_BOORU_SYSTEM_PROMPT, (
            f"PONY_BOORU_SYSTEM_PROMPT references defunct conditional "
            f"marker {marker!r}."
        )


# ── Round-9: tier-strict Pydantic schema factory ───────────────────


def test_round9_tier_strict_schema_marks_required_non_nullable():
    """Round-9 BLOCKER fix: SceneFacet schemas declared tier-required
    fields as Optional[str], so Ollama's format=<json_schema> grammar
    decoder ALLOWED null. _make_tier_strict_schema() rewrites those
    fields to non-nullable str (no Optional, min_length=1). The
    grammar engine then literally cannot emit null for those slots."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetFluxNatural

    strict = _make_tier_strict_schema(SceneFacetFluxNatural, "T4_explicit")
    js = strict.model_json_schema()
    required = set(js.get("required", []))
    # All 7 T4 tier-required fields MUST be in required[]
    expected = {
        "lighting_directive", "mood_aesthetic", "narrative_moment",
        "environment_setting", "environment_atmosphere",
        "nsfw_anatomy", "nsfw_act",
    }
    missing = expected - required
    assert not missing, (
        f"T4 strict schema must mark these fields as required "
        f"(JSON schema-level): {missing}"
    )
    # And each must have type=string (no None / null union)
    for fld in expected:
        spec = js["properties"][fld]
        type_ = spec.get("type")
        assert type_ == "string", (
            f"{fld!r} should be type=string (no Optional). Got: "
            f"{spec.get('type')}, anyOf={spec.get('anyOf')}"
        )
        assert spec.get("minLength") == 1, (
            f"{fld!r} should have minLength=1"
        )


def test_round9_tier_strict_at_T2_only_promotes_T2_fields():
    """At T2_implied, only every-tier required fields are non-nullable.
    T3+/T4-only fields stay Optional[str]."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetFluxNatural

    strict = _make_tier_strict_schema(SceneFacetFluxNatural, "T2_implied")
    js = strict.model_json_schema()
    required = set(js.get("required", []))
    # T1/T2 tier-required: lighting/mood/narrative
    assert {"lighting_directive", "mood_aesthetic", "narrative_moment"} <= required
    # T3+ fields NOT promoted at T2
    assert "environment_setting" not in required
    assert "nsfw_anatomy" not in required
    assert "nsfw_act" not in required


def test_round9_tier_strict_at_T3_promotes_T3_not_T4():
    """At T3_artnude, env_setting/env_atmosphere/nsfw_anatomy promote
    to required; nsfw_act stays Optional (T4-only)."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetFluxNatural

    strict = _make_tier_strict_schema(SceneFacetFluxNatural, "T3_artnude")
    js = strict.model_json_schema()
    required = set(js.get("required", []))
    for f in ("lighting_directive", "mood_aesthetic", "narrative_moment",
              "environment_setting", "environment_atmosphere", "nsfw_anatomy"):
        assert f in required, f"{f} should be required at T3"
    # nsfw_act is T4-only, must NOT be required at T3
    assert "nsfw_act" not in required


def test_round9_pony_strict_schema_handles_pony_omission():
    """Pony's facet schema omits some realism fields. The strict
    factory must NOT crash when a tier-required field is absent from
    the base class — it simply skips that field's override."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetPony

    strict = _make_tier_strict_schema(SceneFacetPony, "T4_explicit")
    js = strict.model_json_schema()
    required = set(js.get("required", []))
    # Pony has these fields and they should be promoted to required
    for f in ("lighting_directive", "mood_aesthetic", "narrative_moment",
              "environment_setting", "environment_atmosphere",
              "nsfw_anatomy", "nsfw_act"):
        assert f in required, (
            f"Pony @T4: {f!r} should be required (declared on schema)"
        )


def test_round9_unknown_tier_returns_base_class():
    """Defensive: unknown content_level passes the class through
    unchanged (back-compat for direct callers)."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetFluxNatural

    result = _make_tier_strict_schema(SceneFacetFluxNatural, "T9_unknown")
    assert result is SceneFacetFluxNatural


def test_round9_factory_is_cached():
    """LRU-cached: same (base, tier) returns the SAME class instance."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetFluxNatural

    a = _make_tier_strict_schema(SceneFacetFluxNatural, "T4_explicit")
    b = _make_tier_strict_schema(SceneFacetFluxNatural, "T4_explicit")
    assert a is b, "Strict schema factory must cache its output"


def test_tier_active_body_pony_rewrite_full_matrix():
    """Round-8 NIT-1: explicit coverage of the Pony body across all
    4 tiers. Pony body has 9 fields (vs non-Pony's 16) and includes
    7 of the 7 tier-required fields uniformly. Confirm rewrite has
    the same shape behavior on Pony."""
    from src.agents.scene_facet_generator import _make_tier_active_schema_body
    # At T4_explicit, every conditional marker should collapse to bare
    # [REQUIRED] regardless of body variant.
    out = _make_tier_active_schema_body(_STRUCTURED_TAG_BODY_PONY, "T4_explicit")
    assert "[REQUIRED — every tier]" not in out
    assert "[REQUIRED — T3+]" not in out
    assert "[REQUIRED — T4 only]" not in out
    for fld in _T4_REQUIRED_FIELDS:
        m = re.search(rf'"{re.escape(fld)}":\s*"\[REQUIRED\]', out)
        assert m is not None, (
            f"Pony body @T4: {fld!r} should carry bare [REQUIRED]"
        )
    # At T2_implied, only every-tier required stays REQUIRED; T3+ and
    # T4-only demote.
    out = _make_tier_active_schema_body(_STRUCTURED_TAG_BODY_PONY, "T2_implied")
    for fld in ("lighting_directive", "mood_aesthetic", "narrative_moment"):
        m = re.search(rf'"{re.escape(fld)}":\s*"\[REQUIRED\]', out)
        assert m is not None, (
            f"Pony body @T2: {fld!r} should remain [REQUIRED]"
        )
    for fld in ("environment_setting", "environment_atmosphere",
                "nsfw_anatomy", "nsfw_act"):
        m = re.search(rf'"{re.escape(fld)}":\s*"\[OPTIONAL', out)
        assert m is not None, (
            f"Pony body @T2: {fld!r} should demote to [OPTIONAL"
        )


def test_every_prompt_style_body_includes_required_markers():
    """Every schema-body variant (sdxl_keywords, pony_danbooru,
    illustrious_tags, flux_natural, flux2_prose) inherits the
    round-6 ordering."""
    for style, body in _SCHEMA_BODY_BY_STYLE.items():
        if style == "pony_danbooru":
            # Pony omits 7 fields but still requires the 7 tier-required
            for field in _T4_REQUIRED_FIELDS:
                assert f'"{field}"' in body, (
                    f"{style} body missing {field!r}"
                )
        else:
            for field in _T4_REQUIRED_FIELDS:
                assert f'"{field}"' in body, (
                    f"{style} body missing {field!r}"
                )
        # Required markers present
        assert "[REQUIRED" in body, f"{style} body lacks [REQUIRED markers"
        assert "[OPTIONAL" in body, f"{style} body lacks [OPTIONAL markers"


# ── Anti-leak + JSON-preamble guards ───────────────────────────────


def test_system_prompt_blocks_photographer_names_in_prose():
    """Round-6 anti-leak fix: prose/booru_tags/free-text fields must
    NEVER carry photographer or film-stock brand names — those belong
    to structured tags only. Without this guard, Cydonia consistently
    leaks 'Helmut Newton' / 'Kodak Portra' into scene_prose, which the
    sanitizer then strips (lossy, brittle).

    Round-7 BLOCKER fix: also assert on PONY_BOORU_SYSTEM_PROMPT,
    which Illustrious uses (and Illustrious has BOTH `booru_tags`
    AND `scene_prose` free-text fields — exactly the leak surface
    the rule was designed to plug)."""
    from src.agents.scene_facet_generator import (
        SYSTEM_PROMPT, PONY_BOORU_SYSTEM_PROMPT,
    )
    for label, text in [
        ("SYSTEM_PROMPT", SYSTEM_PROMPT),
        ("PONY_BOORU_SYSTEM_PROMPT", PONY_BOORU_SYSTEM_PROMPT),
    ]:
        assert "photographer names" in text.lower(), (
            f"{label} lacks the anti-leak photographer-name guard"
        )
        # Either scene_prose (non-Pony) or booru_tags (Pony) must be
        # explicitly named in the no-leak rule.
        leak_target_named = (
            "scene_prose" in text or "booru_tags" in text
        )
        assert leak_target_named, (
            f"{label} must explicitly name a free-text field "
            f"(scene_prose / booru_tags) in the no-leak rule"
        )


def test_system_prompt_blocks_json_preamble():
    """The Cydonia 'Sure, here's the JSON:' preamble bug round-6 also
    guards against. System prompt must demand the first character of
    the response is `{`."""
    from src.agents.scene_facet_generator import (
        SYSTEM_PROMPT, PONY_BOORU_SYSTEM_PROMPT,
    )
    for label, text in [
        ("SYSTEM_PROMPT", SYSTEM_PROMPT),
        ("PONY_BOORU_SYSTEM_PROMPT", PONY_BOORU_SYSTEM_PROMPT),
    ]:
        assert "FIRST character" in text or "first character" in text, (
            f"{label} lacks the JSON-preamble guard "
            f"(first-char-must-be-`{{` rule)"
        )


def test_system_prompt_honour_required_marker():
    """Both system prompts (prose + booru) emphasise the [REQUIRED]
    marker contract — the schema-body markers are inert without the
    system prompt reinforcing them."""
    from src.agents.scene_facet_generator import (
        SYSTEM_PROMPT, PONY_BOORU_SYSTEM_PROMPT,
    )
    for label, text in [
        ("SYSTEM_PROMPT", SYSTEM_PROMPT),
        ("PONY_BOORU_SYSTEM_PROMPT", PONY_BOORU_SYSTEM_PROMPT),
    ]:
        assert "REQUIRED" in text, (
            f"{label} doesn't mention REQUIRED markers — schema body's "
            f"bracket prefixes need system-prompt reinforcement."
        )

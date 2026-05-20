"""Tests for ``SceneFacetGenerator`` — per-family LLM expansion of one
scene into family-shaped composer inputs.

These tests mock ``OllamaClient.generate`` to return canned JSON, so
no Ollama instance is needed. They exercise:

  * Per-family schema dispatch (sdxl / pony / illustrious /
    flux_natural / flux2_prose).
  * Pydantic validation via the SCENE_FACET_SCHEMA_BY_STYLE map.
  * Retry-with-nudge on first-attempt failure.
  * SceneFacetGeneratorError raised when 2 attempts fail or when the
    family's prompt_style has no facet schema.
  * System prompt incorporates per-model trigger_words / avoid_words /
    structure_rules from ModelPromptGuide.
  * User prompt only includes the scene's model-agnostic core (no
    family-shaped fields from sibling families leak in).
  * Temperature precedence: explicit > family.llm_temperature > class default.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.agents.scene_facet_generator import (
    SceneFacetGenerator,
    SceneFacetGeneratorError,
)
from src.memory.family_loader import FamilyLoader


# ── helpers ─────────────────────────────────────────────────────────


@pytest.fixture
def loader() -> FamilyLoader:
    return FamilyLoader()


@pytest.fixture
def generator() -> SceneFacetGenerator:
    return SceneFacetGenerator(OllamaClient())


def _scene() -> dict:
    """A model-agnostic scene core."""
    return {
        "variation_axis": "pose",
        "pose": "three-quarter standing",
        "camera": "85mm portrait",
        "camera_angle": "eye-level",
        "lighting": "golden hour rim light",
        "environment_detail": "rooftop balcony at sunset",
        "mood_note": "calm contemplation",
        "expression": "soft smile",
        "composition_intent": "medium",
    }


class _DualPatch:
    """Patch both transports — legacy /api/generate and Q6 /api/chat —
    so schema-aware calls (which trigger the chat-with-prefill path
    in OllamaClient.generate_json) still receive the canned response.

    Strips fences + leading structural opener from the chat-mock so
    ``prefill + chat_continuation`` parses to the same JSON the legacy
    path would return.
    """

    def __init__(self, text: str):
        self._text = text
        self._patches: list = []

    def __enter__(self):
        self._patches.append(
            patch.object(OllamaClient, "generate", return_value=self._text)
        )
        # Q6 prefill is "Sure, here's the JSON: " (no structural opener)
        # so the chat mock returns the full JSON the same way generate
        # does. _extract_json_payload + _strip_fences handle the rest.
        self._patches.append(
            patch.object(
                OllamaClient, "_generate_chat", return_value=self._text,
            )
        )
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


def _patch_generate(text: str):
    """Patch the underlying generate() to return canned LLM output.

    Q6: also patches /api/chat so schema-aware code paths (which use
    assistant-prefill) work transparently in tests.
    """
    return _DualPatch(text)


# ── Per-family dispatch ─────────────────────────────────────────────


# Round-9 tier-strict schema: at each tier, certain structured-tag
# fields are required to be non-null. The canned LLM responses below
# include those fields so the strict schema validator accepts.
_T2_REQUIRED_TAGS_JSON = (
    '"lighting_directive": "LIGHT_WINDOW_SIDE", '
    '"mood_aesthetic": "MOOD_SERENE", '
    '"narrative_moment": "NARR_COFFEE_MORNING_PAPER"'
)

_T3_REQUIRED_TAGS_JSON = (
    _T2_REQUIRED_TAGS_JSON +
    ', "environment_setting": "ENV_MORNING_BEDROOM"'
    ', "environment_atmosphere": "ATM_DUST_MOTES_IN_LIGHT"'
    ', "nsfw_anatomy": "NSFW_BREAST_NATURAL"'
    # Round-12: realism_camera + realism_lens promoted to required at T3+.
    ', "realism_camera": "CAMERA_SONY_A7RV"'
    ', "realism_lens": "LENS_85MM_F14"'
)

_T4_REQUIRED_TAGS_JSON = (
    _T3_REQUIRED_TAGS_JSON +
    ', "nsfw_act": "NSFW_T4_SOLO_TOUCH"'
)


def _required_tags_for(tier: str) -> str:
    """Return the strict-schema required-tag JSON fragment for ``tier``."""
    return {
        "T1_suggestive": _T2_REQUIRED_TAGS_JSON,
        "T2_implied":    _T2_REQUIRED_TAGS_JSON,
        "T3_artnude":    _T3_REQUIRED_TAGS_JSON,
        "T4_explicit":   _T4_REQUIRED_TAGS_JSON,
    }[tier]


def test_sdxl_facet_dispatch(generator, loader):
    family = loader.get_family("sdxl")
    canned = (
        '{"camera_spec": "85mm f/1.4", "clothing": "ivory silk slip", '
        + _T2_REQUIRED_TAGS_JSON + '}'
    )
    with _patch_generate(canned):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert facet["camera_spec"] == "85mm f/1.4"
    assert facet["clothing"] == "ivory silk slip"
    assert facet["lighting_directive"] == "LIGHT_WINDOW_SIDE"


def test_pony_facet_dispatch(generator, loader):
    family = loader.get_family("pony")
    canned = (
        '{"booru_tags": "long_hair, brown_hair, looking_at_viewer", '
        '"source_tag": "source_photograph", '
        + _T2_REQUIRED_TAGS_JSON + '}'
    )
    with _patch_generate(canned):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert facet["booru_tags"].startswith("long_hair")
    assert facet["source_tag"] == "source_photograph"


def test_illustrious_facet_dispatch(generator, loader):
    family = loader.get_family("illustrious")
    canned = (
        '{"booru_tags": "long_hair, soft_focus", '
        '"scene_prose": "She stands on the balcony in golden hour light.", '
        + _T2_REQUIRED_TAGS_JSON + '}'
    )
    with _patch_generate(canned):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert "scene_prose" in facet


def test_flux_facet_dispatch_uses_flux_natural_schema(generator, loader):
    """flux family → flux_natural prompt_style → SceneFacetFluxNatural."""
    family = loader.get_family("flux")
    canned = (
        '{"scene_prose": "She stands on a sunset balcony in an ivory silk '
        'dress, golden-hour rim light catching her hair.", '
        + _T2_REQUIRED_TAGS_JSON + '}'
    )
    with _patch_generate(canned):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert facet["scene_prose"].startswith("She stands")


def test_chroma_facet_uses_same_schema_as_flux(generator, loader):
    """chroma family also uses flux_natural prompt_style."""
    family = loader.get_family("chroma")
    canned = (
        '{"scene_prose": "She leans against the balcony rail.", '
        + _T2_REQUIRED_TAGS_JSON + '}'
    )
    with _patch_generate(canned):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert facet["scene_prose"].startswith("She leans")


def test_flux2_facet_dispatch_with_qa_fields(generator, loader):
    """Fixture prose tuned to 30–80 word BFL target (Phase 4b validator)."""
    family = loader.get_family("flux2")
    canned = (
        '{"scene_prose": "Mira sits on a low concrete bench in a stark '
        'minimalist loft, raven hair falling softly past her shoulders. '
        'North-facing window light wraps gently around her face from the '
        'left, illuminating natural skin texture and casting a soft '
        'shadow on the wall behind. The atmosphere is quiet, intimate, '
        'pensive late-afternoon.", '
        '"subject_focus": "Mira, 28, raven hair", '
        + _T2_REQUIRED_TAGS_JSON + '}'
    )
    with _patch_generate(canned):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert facet["scene_prose"].startswith("Mira")
    assert facet["subject_focus"] == "Mira, 28, raven hair"


# ── Schema validation failures ──────────────────────────────────────


def test_invalid_facet_triggers_retry(generator, loader):
    """First attempt returns junk; second succeeds → returns success.

    Round-9 update: the second-attempt JSON includes the T2 tier-
    required fields so the strict schema accepts the retry."""
    family = loader.get_family("sdxl")
    responses = [
        "garbage not json",
        ('{"camera_spec": "85mm", "clothing": "silk", '
         + _T2_REQUIRED_TAGS_JSON + '}'),
    ]
    call_count = {"n": 0}

    def fake_generate(*args, **kwargs):
        n = call_count["n"]
        call_count["n"] += 1
        return responses[n]

    # Q6 — schema-aware calls use /api/chat (not /api/generate).
    with patch.object(OllamaClient, "_generate_chat", side_effect=fake_generate):
        facet = generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    assert facet["camera_spec"] == "85mm"
    assert call_count["n"] == 2  # one retry


def test_facet_missing_required_field_retries_then_raises(generator, loader):
    """Both attempts return facets missing required fields → raise."""
    family = loader.get_family("sdxl")
    # SDXL requires both camera_spec AND clothing.
    canned = '{"camera_spec": "85mm"}'   # missing clothing
    with _patch_generate(canned):
        with pytest.raises(SceneFacetGeneratorError, match="2 attempts"):
            generator.generate(scene=_scene(), family=family, content_level="T2_implied")


def test_unsupported_prompt_style_raises(generator):
    """A family whose prompt_style isn't in the dispatcher → raise eagerly."""
    from unittest.mock import MagicMock
    bogus_family = MagicMock()
    bogus_family.id = "bogus_family"
    bogus_family.prompt_style = "bogus_style_not_in_dispatcher"
    bogus_family.llm_temperature = None
    bogus_family.guide = None
    with pytest.raises(SceneFacetGeneratorError, match="No facet schema"):
        generator.generate(scene=_scene(), family=bogus_family, content_level="T2_implied")


# ── System prompt content ───────────────────────────────────────────


def test_system_prompt_includes_trigger_words(generator, loader):
    """Per-model trigger_words from ModelPromptGuide are surfaced."""
    family = loader.get_family("sdxl")
    captured = {"system_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return ('{"camera_spec": "x", "clothing": "y", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    from unittest.mock import MagicMock
    guide = MagicMock()
    guide.llm_hint = ""
    guide.structure_rules = ""
    guide.trigger_words = ["shot on Canon EOS 5D", "glamour photography"]
    guide.avoid_words = []
    guide.example_prompt = ""

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(scene=_scene(), family=family, prompt_guide=guide, content_level="T2_implied")
    assert "Canon EOS 5D" in captured["system_prompt"]
    assert "TRIGGER WORDS" in captured["system_prompt"]


def test_system_prompt_includes_avoid_words(generator, loader):
    family = loader.get_family("sdxl")
    captured = {"system_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return ('{"camera_spec": "x", "clothing": "y", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    from unittest.mock import MagicMock
    guide = MagicMock()
    guide.llm_hint = ""
    guide.structure_rules = ""
    guide.trigger_words = []
    guide.avoid_words = ["painting", "illustration"]
    guide.example_prompt = ""

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(scene=_scene(), family=family, prompt_guide=guide, content_level="T2_implied")
    assert "AVOID" in captured["system_prompt"]
    assert "painting" in captured["system_prompt"]


def test_system_prompt_includes_family_guide_for_flux2(generator, loader):
    """FLUX.2 Klein's structure_order + target_words appear in the prompt."""
    family = loader.get_family("flux2")  # has guide block
    captured = {"system_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return (
            '{"scene_prose": "x" * 50}'.replace('"x" * 50', '"' + 'x ' * 30 + '"')
        )

    # Real callable returning valid JSON. Word count tuned to BFL
    # Klein 9B 30-80 target band (Phase 4b validator requires 25-95).
    valid_response = json.dumps({
        "scene_prose": (
            "She stands on the balcony at sunset, raven hair lifting in "
            "the warm breeze. The horizon glows in shades of amber and "
            "rose, with the city below softening into haze. Soft golden "
            "rim light from the right traces her silhouette. The mood "
            "is contemplative, tender, late-summer."
        ),
        # Round-9 strict-schema requires the T2 tier-required fields.
        "lighting_directive": "LIGHT_WINDOW_SIDE",
        "mood_aesthetic": "MOOD_SERENE",
        "narrative_moment": "NARR_COFFEE_MORNING_PAPER",
    })
    # Q6 — schema-aware calls use /api/chat (not /api/generate).
    with patch.object(OllamaClient, "_generate_chat", return_value=valid_response) as mock:
        # Inject capturing side effect inside the patch
        def side(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return valid_response
        mock.side_effect = side
        generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    sp = captured["system_prompt"]
    # FLUX.2 family.guide → PROMPT_STYLE_GUIDE block in the system prompt.
    assert "PROMPT_STYLE_GUIDE" in sp
    assert "Anchor order:" in sp


# ── User prompt content ─────────────────────────────────────────────


def test_user_prompt_includes_scene_core_and_family_id(generator, loader):
    """The user prompt embeds the scene's model-agnostic core +
    family.id + prompt_style + schema body."""
    family = loader.get_family("pony")
    captured = {"user_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return ('{"booru_tags": "1girl, solo, long_hair", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    up = captured["user_prompt"]
    assert "three-quarter standing" in up           # pose from scene
    assert "golden hour rim light" in up            # lighting from scene
    assert "pony" in up                              # family.id
    assert "pony_danbooru" in up                    # prompt_style
    assert "booru_tags" in up                        # schema body field


def test_user_prompt_does_not_leak_other_family_fields(generator, loader):
    """If the scene dict has stray family-shaped fields from a sibling
    (e.g. it was generated with sdxl and now we're asking for pony),
    those should NOT make it into the LLM user prompt."""
    family = loader.get_family("pony")
    scene_with_sdxl_leftovers = {
        **_scene(),
        "camera_spec": "85mm f/1.4",      # sdxl-shaped
        "clothing": "silk dress",          # sdxl-shaped
        "scene_prose": "She stands…",      # flux-shaped
    }
    captured = {"user_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return ('{"booru_tags": "1girl, solo, x", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(scene=scene_with_sdxl_leftovers, family=family, content_level='T2_implied')
    up = captured["user_prompt"]
    # Scene core fields present:
    assert "three-quarter standing" in up
    # Sibling-family fields filtered out:
    assert "85mm f/1.4" not in up
    assert "silk dress" not in up
    assert "She stands" not in up


# ── Temperature precedence ──────────────────────────────────────────


def test_explicit_temperature_overrides_family_default(generator, loader):
    family = loader.get_family("sdxl")
    captured = {"temperature": None}

    def capture(system_prompt, user_prompt, *, temperature, **kwargs):
        captured["temperature"] = temperature
        return ('{"camera_spec": "x", "clothing": "y", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(scene=_scene(), family=family, temperature=0.3, content_level="T2_implied")
    assert captured["temperature"] == 0.3


def test_family_temperature_used_when_no_explicit_override(
    generator, loader,
):
    """A family with `llm_temperature` set takes precedence over the class default."""
    family = loader.get_family("pony")  # pony has llm_temperature: 0.5
    captured = {"temperature": None}

    def capture(system_prompt, user_prompt, *, temperature, **kwargs):
        captured["temperature"] = temperature
        return ('{"booru_tags": "1girl, solo, x", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(scene=_scene(), family=family, content_level="T2_implied")
    # If pony.llm_temperature is set, it wins; else falls back to 0.7.
    if family.llm_temperature is not None:
        assert captured["temperature"] == family.llm_temperature
    else:
        assert captured["temperature"] == SceneFacetGenerator.TEMPERATURE


# ── Phase A: content_level + tier directive surfacing ─────────────


@pytest.mark.parametrize("tier", [
    "T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit",
])
def test_user_prompt_contains_content_level_line(
    generator, loader, tier,
):
    """User prompt must surface ``Content level: <tier>`` to the LLM
    (Phase A — pre-fix, the facet generator ran tier-blind)."""
    family = loader.get_family("sdxl")
    captured = {"user_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
        return ('{"camera_spec": "x", "clothing": "y", '
                + _required_tags_for(tier) + '}')

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(
            scene=_scene(), family=family, content_level=tier,
        )
    assert f"Content level: {tier}" in captured["user_prompt"]


def test_system_prompt_carries_llm_directive(generator, loader):
    """When ``llm_directive`` is supplied, the SceneFacetGenerator
    injects it verbatim into the system prompt right after the
    standard preamble."""
    family = loader.get_family("sdxl")
    captured = {"system_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        # Round-9: at T4, the strict schema requires all 7 tier fields.
        return ('{"camera_spec": "x", "clothing": "y", '
                + _T4_REQUIRED_TAGS_JSON + '}')

    directive = "CONTENT TIER: T4_explicit. Depict the subject NUDE."

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(
            scene=_scene(), family=family,
            content_level="T4_explicit",
            llm_directive=directive,
        )
    assert "CONTENT TIER: T4_explicit" in captured["system_prompt"]
    assert "Depict the subject NUDE" in captured["system_prompt"]


def test_system_prompt_skips_directive_when_empty(generator, loader):
    """No directive (empty string) → system prompt stays clean."""
    family = loader.get_family("sdxl")
    captured = {"system_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        return ('{"camera_spec": "x", "clothing": "y", '
                + _T2_REQUIRED_TAGS_JSON + '}')

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(
            scene=_scene(), family=family,
            content_level="T1_suggestive",
            llm_directive="",
        )
    assert "CONTENT TIER:" not in captured["system_prompt"]


def test_t4_directive_pushes_for_nsfw_act(generator, loader):
    """When the T4 directive lands in the system prompt, it explicitly
    asks the LLM to pick a nsfw_act tag from the menu — the load-bearing
    bit that re-enables T4 NSFW prompts."""
    family = loader.get_family("flux")
    captured = {"system_prompt": ""}

    def capture(system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        # Return valid Flux prose (≥ 25 words for the validator)
        # plus round-9 strict-schema T4 tier-required tags.
        return (
            '{"scene_prose": "She stands in a sunlit room with morning '
            'light pouring across the bare floor and a single window '
            'illuminating the side of her face in a quiet pensive '
            'moment of solitude.", '
            + _T4_REQUIRED_TAGS_JSON + '}'
        )

    from src.memory.categories_loader import CategoriesLoader
    rules = CategoriesLoader().content_level_rules("T4_explicit")
    assert rules.llm_directive  # YAML must declare it
    assert "nsfw_act" in rules.llm_directive

    # Q6 — schema-aware calls use /api/chat (not /api/generate). Patch
    # the chat endpoint so the capture sees the system prompt.
    with patch.object(OllamaClient, "_generate_chat", side_effect=capture):
        generator.generate(
            scene=_scene(), family=family,
            content_level="T4_explicit",
            llm_directive=rules.llm_directive,
        )
    sp = captured["system_prompt"]
    assert "T4_explicit" in sp
    assert "nsfw_act" in sp


def test_categories_yaml_declares_llm_directive_for_every_tier():
    """Every T1-T4 row in categories.yaml MUST declare a non-empty
    llm_directive so SceneFacetGenerator never runs tier-blind."""
    from src.memory.categories_loader import CategoriesLoader
    loader = CategoriesLoader()
    for tier in ("T1_suggestive", "T2_implied", "T3_artnude", "T4_explicit"):
        rules = loader.content_level_rules(tier)
        assert rules.llm_directive, f"{tier} missing llm_directive"
        assert tier in rules.llm_directive, (
            f"{tier} llm_directive should mention the tier name for clarity"
        )


# ── Phase D: T3/T4 boost_keywords reduced to [] ────────────────────


def test_t3_t4_boost_keywords_are_empty():
    """Phase D — T3 and T4 boost_keywords are deliberately empty
    because the LLM directive (Phase A) drives nudity/anatomy at
    those tiers. Appending soft adjectives at sanitizer time was
    decorative and didn't actually push toward NSFW content."""
    from src.memory.categories_loader import CategoriesLoader
    loader = CategoriesLoader()
    t3 = loader.content_level_rules("T3_artnude")
    t4 = loader.content_level_rules("T4_explicit")
    assert t3.prompt_boost_keywords == [], (
        f"T3 boost_keywords should be empty; got {t3.prompt_boost_keywords}"
    )
    assert t4.prompt_boost_keywords == [], (
        f"T4 boost_keywords should be empty; got {t4.prompt_boost_keywords}"
    )


def test_t1_t2_boost_keywords_still_present():
    """T1 and T2 keep their gentle nudges — those tiers don't have
    NSFW content to push toward, so the soft adjectives still help."""
    from src.memory.categories_loader import CategoriesLoader
    loader = CategoriesLoader()
    t1 = loader.content_level_rules("T1_suggestive")
    t2 = loader.content_level_rules("T2_implied")
    assert len(t1.prompt_boost_keywords) > 0, "T1 should keep boost keywords"
    assert len(t2.prompt_boost_keywords) > 0, "T2 should keep boost keywords"


# ── Round-12: realism_camera + realism_lens required at T3+ ─────────


def test_t3_requires_realism_camera_and_lens():
    """Round-12 (2026-05-21) fix — the 2026-05-20 A/B run showed both
    Cydonia and Qwen3 nulling realism_camera/lens on 23+/24 facets
    despite the family few-shot exemplars. Promoting these to
    tier-required at T3+ lets the retry-nudge fire on misses."""
    from src.agents.scene_facet_generator import _TIER_REQUIRED_FIELDS
    assert "realism_camera" in _TIER_REQUIRED_FIELDS["T3_artnude"]
    assert "realism_lens" in _TIER_REQUIRED_FIELDS["T3_artnude"]
    assert "realism_camera" in _TIER_REQUIRED_FIELDS["T4_explicit"]
    assert "realism_lens" in _TIER_REQUIRED_FIELDS["T4_explicit"]


def test_t1_t2_do_not_require_realism_camera_lens():
    """T1/T2 stay realism-flexible — those tiers don't necessarily
    need a specific camera body / lens spec (soft pose work doesn't
    always benefit from a baked-in 85mm portrait look)."""
    from src.agents.scene_facet_generator import _TIER_REQUIRED_FIELDS
    assert "realism_camera" not in _TIER_REQUIRED_FIELDS["T1_suggestive"]
    assert "realism_lens" not in _TIER_REQUIRED_FIELDS["T1_suggestive"]
    assert "realism_camera" not in _TIER_REQUIRED_FIELDS["T2_implied"]
    assert "realism_lens" not in _TIER_REQUIRED_FIELDS["T2_implied"]


def test_pony_schema_skips_realism_camera_lens_promotion():
    """Pony schema omits realism_camera / realism_lens entirely —
    booru tagging carries those implicitly via source_photograph +
    tag patterns. The strict-schema factory must auto-skip fields
    not declared on the base class."""
    from src.agents.scene_facet_generator import _make_tier_strict_schema
    from src.agents.schemas import SceneFacetPony
    strict = _make_tier_strict_schema(
        SceneFacetPony, "T3_artnude", is_booru_native=True,
    )
    # realism_camera / realism_lens absent from SceneFacetPony →
    # not promoted; the strict schema doesn't grow these fields.
    assert "realism_camera" not in strict.model_fields
    assert "realism_lens" not in strict.model_fields


def test_field_example_tags_cover_realism_camera_lens():
    """Retry-nudge inlines example tags per missing required field.
    Round-12 — realism_camera / realism_lens need entries so the
    retry attempt carries concrete CAMERA_/LENS_ examples."""
    from src.agents.scene_facet_generator import _FIELD_EXAMPLE_TAGS
    assert "realism_camera" in _FIELD_EXAMPLE_TAGS
    assert "realism_lens" in _FIELD_EXAMPLE_TAGS
    assert any(
        ex.startswith("CAMERA_")
        for ex in _FIELD_EXAMPLE_TAGS["realism_camera"]
    )
    assert any(
        ex.startswith("LENS_")
        for ex in _FIELD_EXAMPLE_TAGS["realism_lens"]
    )


# ── Round-12: tag-frequency dominance tracker ──────────────────────


def test_diversity_tracker_silent_before_min_facets():
    """Less than 4 facets recorded → tracker stays silent regardless
    of how dominant a tag is. Avoids false-positive nudges on tiny
    series."""
    from src.agents.scene_facet_generator import _DiversityTracker
    t = _DiversityTracker()
    for _ in range(3):
        t.record({"lighting_directive": "LIGHT_WINDOW_SIDE"})
    assert t.overused_summary() == ""


def test_diversity_tracker_fires_nudge_at_dominance_threshold():
    """50%+ of facets-so-far use the same tag for a tracked axis →
    nudge text mentions the axis + tag + count."""
    from src.agents.scene_facet_generator import _DiversityTracker
    t = _DiversityTracker()
    # 4 facets all locked to one lighting tag — 4/4 = 100% dominance,
    # well above the 50% threshold.
    for _ in range(4):
        t.record({"lighting_directive": "LIGHT_WINDOW_SIDE"})
    nudge = t.overused_summary()
    assert "lighting_directive" in nudge
    assert "LIGHT_WINDOW_SIDE" in nudge
    assert "4/4" in nudge
    assert "Diversity nudge" in nudge


def test_diversity_tracker_silent_under_dominance_threshold():
    """A balanced spread keeps every tag under 50% → no nudge fires."""
    from src.agents.scene_facet_generator import _DiversityTracker
    t = _DiversityTracker()
    # 4 facets, 4 different lighting tags (each 1/4 = 25%, under 50%).
    for tag in (
        "LIGHT_WINDOW_SIDE", "LIGHT_GOLDEN_HOUR",
        "LIGHT_RIM_BACK", "LIGHT_REMBRANDT",
    ):
        t.record({"lighting_directive": tag})
    assert t.overused_summary() == ""


def test_diversity_tracker_handles_multiple_overused_axes():
    """When two axes both hit dominance, both appear in the nudge."""
    from src.agents.scene_facet_generator import _DiversityTracker
    t = _DiversityTracker()
    for _ in range(5):
        t.record({
            "lighting_directive": "LIGHT_WINDOW_SIDE",
            "mood_aesthetic": "MOOD_PENSIVE",
            # nsfw_anatomy varies → won't trigger.
            "nsfw_anatomy": "NSFW_FULL_NUDE",
        })
    for _ in range(2):
        t.record({"nsfw_anatomy": "NSFW_BREAST_NATURAL"})
    nudge = t.overused_summary()
    assert "lighting_directive" in nudge
    assert "mood_aesthetic" in nudge
    # 5/7 NSFW_FULL_NUDE = 71% > 50% → also in nudge.
    assert "nsfw_anatomy" in nudge


def test_diversity_tracker_ignores_null_and_empty_tags():
    """``None`` / empty string values shouldn't count toward dominance
    — otherwise an LLM that nulls a field 4 times gets a nudge about
    the null."""
    from src.agents.scene_facet_generator import _DiversityTracker
    t = _DiversityTracker()
    for _ in range(5):
        t.record({"lighting_directive": None})
        t.record({"lighting_directive": ""})
    # Both null and empty ignored → tracker is empty → no nudge.
    assert t.overused_summary() == ""


def test_diversity_tracker_records_each_axis_independently():
    """Each tracked axis has its own counter; cross-axis pollution
    should not happen."""
    from src.agents.scene_facet_generator import _DiversityTracker
    t = _DiversityTracker()
    # Lighting locks; mood varies.
    for tag in (
        "MOOD_SERENE", "MOOD_PENSIVE", "MOOD_CONFIDENT",
        "MOOD_INTIMATE", "MOOD_PLAYFUL",
    ):
        t.record({
            "lighting_directive": "LIGHT_WINDOW_SIDE",
            "mood_aesthetic": tag,
        })
    nudge = t.overused_summary()
    # Lighting hit dominance.
    assert "LIGHT_WINDOW_SIDE" in nudge
    # Mood didn't (5 tags × 1 = each 20%, well under 50%).
    for mood in ("MOOD_SERENE", "MOOD_PENSIVE", "MOOD_PLAYFUL"):
        assert mood not in nudge

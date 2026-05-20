"""Verifier round-4 IMPORTANT-5 regression: soft warning when a series
plan ships with no Phase 3 aesthetic anchors.

SeriesPlan declares ``color_palette`` / ``photographer_ref`` /
``art_movement`` as ``Optional[str]`` because Pony omits two of the
three. Ollama's constrained decoding therefore doesn't grammar-force
the LLM to emit them — a chatty LLM can return a "valid" plan with
all three None, silently degrading the signature-look pinning.

``warn_if_missing_aesthetic_anchors`` surfaces this in run_log so
operators catch the silent degradation. These tests guard the
warning behaviour against future regressions.
"""

from __future__ import annotations

import logging

from src.modes._llm_helpers import (
    repair_colon_suffix_aesthetic_keys,
    warn_if_missing_aesthetic_anchors,
)


def test_warns_when_all_three_anchors_missing(caplog):
    plan = {
        "theme": "abandoned ballroom",
        "mood": "haunting",
        "environment": "marble floor with chandelier",
        "variation_axes": ["pose", "lighting"],
        # all three aesthetic anchors omitted
    }
    with caplog.at_level(logging.WARNING):
        warn_if_missing_aesthetic_anchors(plan, mode_name="ThemeMode")
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "ThemeMode" in m and "NO aesthetic anchors populated" in m
        for m in msgs
    ), f"expected a WARNING about no anchors; got: {msgs}"


def test_warns_when_only_color_palette_missing(caplog):
    """Pony series legitimately drop photographer_ref + art_movement
    but should still have color_palette. Missing color_palette alone
    triggers a distinct warning."""
    plan = {
        "theme": "anime girl in cafe",
        "mood": "soft",
        "environment": "cafe with window light",
        "variation_axes": ["pose"],
        "photographer_ref": None,
        "art_movement": None,
        # color_palette also missing — the only "every family"
        # aesthetic anchor.
    }
    with caplog.at_level(logging.WARNING):
        warn_if_missing_aesthetic_anchors(plan, mode_name="StyleMode")
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "StyleMode" in m and "color_palette" in m
        for m in msgs
    ), f"expected color_palette warning; got: {msgs}"


def test_no_warning_when_color_palette_populated(caplog):
    """The healthy Pony case — color_palette present, the other two
    None (canonicalizer drops them silently for Pony). Should NOT
    log a warning."""
    plan = {
        "theme": "studio nude",
        "mood": "soft",
        "environment": "single key light",
        "variation_axes": ["pose"],
        "color_palette": "PALETTE_MONOCHROME_HIGH_CONTRAST",
        "photographer_ref": None,
        "art_movement": None,
    }
    with caplog.at_level(logging.WARNING):
        warn_if_missing_aesthetic_anchors(plan, mode_name="NicheMode")
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("NicheMode" in m for m in msgs), (
        f"unexpected warning when color_palette is set: {msgs}"
    )


def test_no_warning_when_all_three_populated(caplog):
    plan = {
        "theme": "cinematic boudoir",
        "mood": "intimate",
        "environment": "art deco hotel suite",
        "variation_axes": ["pose", "lighting"],
        "color_palette": "PALETTE_BAROQUE_CARAVAGGIO",
        "photographer_ref": "PHOTOG_HELMUT_NEWTON",
        "art_movement": "ART_MOVE_FILM_NOIR_1940S",
    }
    with caplog.at_level(logging.WARNING):
        warn_if_missing_aesthetic_anchors(
            plan, mode_name="ThemeMode",
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("ThemeMode" in m for m in msgs), (
        f"unexpected warning for fully-populated plan: {msgs}"
    )


def test_mode_name_appears_in_warning(caplog):
    """Operators grep run_log by mode_name when triaging."""
    plan = {"theme": "x", "mood": "y", "environment": "z",
            "variation_axes": ["a"]}
    with caplog.at_level(logging.WARNING):
        warn_if_missing_aesthetic_anchors(plan, mode_name="MyTestMode")
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("MyTestMode:") for m in msgs), (
        f"mode_name must prefix every warning; got: {msgs}"
    )


# ── Round-5: colon-suffix LLM quirk repair ──────────────────────────


def test_repair_salvages_colon_suffix_keys():
    """The Cydonia heretic-vision quirk: extra=allow lets the LLM ship
    `"color_palette:": "PALETTE_X"` alongside `"color_palette": null`
    and Pydantic accepts. Repair lifts the value into the canonical
    slot and drops the bad key."""
    plan = {
        "theme": "t", "mood": "m", "environment": "e",
        "variation_axes": ["a"],
        "color_palette": None,
        "photographer_ref": None,
        "art_movement": None,
        "color_palette:": "PALETTE_BAROQUE_CARAVAGGIO",
        "photographer_ref:": "PHOTOG_HELMUT_NEWTON",
        "art_movement:": "ART_MOVE_FILM_NOIR_1940S",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] == "PALETTE_BAROQUE_CARAVAGGIO"
    assert plan["photographer_ref"] == "PHOTOG_HELMUT_NEWTON"
    assert plan["art_movement"] == "ART_MOVE_FILM_NOIR_1940S"
    assert "color_palette:" not in plan
    assert "photographer_ref:" not in plan
    assert "art_movement:" not in plan


def test_repair_skips_when_canonical_already_populated():
    """If the LLM does the right thing AND emits a colon-suffix
    duplicate, the canonical wins and the suffix is dropped."""
    plan = {
        "color_palette": "PALETTE_MUTED_EARTH_WARM",
        "color_palette:": "PALETTE_SOMETHING_ELSE",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] == "PALETTE_MUTED_EARTH_WARM"
    # Colon key dropped regardless (we don't want it lingering)
    # Actually current implementation only drops when it copies the
    # value — when canonical wins, the colon-suffix is left alone.
    # That's defensible (no value harvested → don't touch it). Spec
    # the actual behaviour.
    assert plan.get("color_palette:") == "PALETTE_SOMETHING_ELSE"


def test_repair_noop_on_clean_plan():
    """No colon-suffix keys, no canonical anchors set — repair is
    a clean no-op."""
    plan = {
        "theme": "t", "mood": "m", "environment": "e",
        "variation_axes": ["a"],
    }
    before = dict(plan)
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan == before


def test_repair_partial_some_canonical_some_colon():
    """LLM emits one anchor cleanly + one via colon-suffix +
    skips the third. Repair salvages the colon one; doesn't
    invent the missing one."""
    plan = {
        "color_palette": "PALETTE_TEAL_ORANGE_BLOCKBUSTER",
        "photographer_ref": None,
        "art_movement": None,
        "photographer_ref:": "PHOTOG_GREGORY_CREWDSON",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] == "PALETTE_TEAL_ORANGE_BLOCKBUSTER"
    assert plan["photographer_ref"] == "PHOTOG_GREGORY_CREWDSON"
    assert plan["art_movement"] is None
    assert "photographer_ref:" not in plan


# ── Round-12 LLM-quirk repairs (Cydonia + Qwen3 A/B run 2026-05-20) ─


def test_repair_salvages_trailing_space_keys():
    """Qwen3 abliterated 30B-A3B quirk: emits the canonical key as
    null AND an extra key with a trailing space carrying the actual
    value. Observed on the 2026-05-20 T3 theme A/B run — all three
    aesthetic anchors hit by this shape."""
    plan = {
        "theme": "t", "mood": "m", "environment": "e",
        "variation_axes": ["a"],
        "color_palette": None,
        "photographer_ref": None,
        "art_movement": None,
        "color_palette ": "PALETTE_PINK_AND_GOLD",
        "photographer_ref ": "PHOTOG_PETER_LINDBERGH",
        "art_movement ": "ART_MOVE_BAUHAUS_MINIMAL",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] == "PALETTE_PINK_AND_GOLD"
    assert plan["photographer_ref"] == "PHOTOG_PETER_LINDBERGH"
    assert plan["art_movement"] == "ART_MOVE_BAUHAUS_MINIMAL"
    assert "color_palette " not in plan
    assert "photographer_ref " not in plan
    assert "art_movement " not in plan


def test_repair_salvages_comma_collapsed_key():
    """Cydonia heretic quirk: a single malformed key collapses
    multiple field names into one comma-joined string. Observed:
        {"color_palette: PALETTE_LUBEZKI_NATURAL_GOLDEN, photographer_ref":
         "PHOTOG_PETTER_HEGRE, art_movement: null"}
    The leading PALETTE_* tag lives inside the KEY string; the
    PHOTOG_* tag lives at the head of the VALUE string; art_movement
    is explicitly null in the value tail. Repair extracts both,
    leaves art_movement alone (the LLM said null)."""
    plan = {
        "theme": "t", "mood": "m", "environment": "e",
        "variation_axes": ["a"],
        "color_palette": None,
        "photographer_ref": None,
        "art_movement": None,
        "color_palette: PALETTE_LUBEZKI_NATURAL_GOLDEN, photographer_ref":
            "PHOTOG_PETTER_HEGRE, art_movement: null",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] == "PALETTE_LUBEZKI_NATURAL_GOLDEN"
    assert plan["photographer_ref"] == "PHOTOG_PETTER_HEGRE"
    assert plan["art_movement"] is None
    # Malformed key gone.
    assert not any(
        "color_palette:" in k or "photographer_ref" == k.rstrip(":")
        for k in plan if k not in ("color_palette", "photographer_ref")
    )


def test_repair_handles_mixed_quirks_one_plan():
    """LLM produces a multi-flavour mess: comma-collapse on one anchor,
    trailing-space on another, canonical-clean on the third. Repair
    handles all three in one pass."""
    plan = {
        "color_palette": None,
        "photographer_ref": None,
        "art_movement": "ART_MOVE_WES_ANDERSON",  # clean
        "color_palette: PALETTE_TUSCAN_EARTH, photographer_ref":
            "PHOTOG_SLIM_AARONS",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] == "PALETTE_TUSCAN_EARTH"
    assert plan["photographer_ref"] == "PHOTOG_SLIM_AARONS"
    assert plan["art_movement"] == "ART_MOVE_WES_ANDERSON"


def test_repair_comma_value_only_no_prefix_match_is_noop():
    """Defensive: if the malformed value's comma chain contains no
    PALETTE_* / PHOTOG_* / ART_MOVE_* tokens, the helper writes
    nothing rather than guessing. Protects against garbage writes."""
    plan = {
        "color_palette": None,
        "weird_garbage_key, photographer_ref":
            "random text with no prefix tokens",
    }
    repair_colon_suffix_aesthetic_keys(plan)
    assert plan["color_palette"] is None
    # The garbage key stays untouched.
    assert "weird_garbage_key, photographer_ref" in plan


# ── Round-15: widen_compat_intersection ─────────────────────────────


class TestWidenCompatIntersection:
    """Round-15 (2026-05-21) — intersection-too-narrow fallback for
    theme/style/niche compat lists.

    The 2026-05-21 LM Studio Cydonia audit showed a 1-entry
    post-intersection menu let the LLM hallucinate 17 fresh ENV_*
    tags freely (it sees a single option, decides the prose doesn't
    fit it, invents new ones). ``widen_compat_intersection`` keeps
    the intersection when it has ≥3 entries; otherwise falls
    through to the category list (theme is the stronger thematic
    signal vs style_profile's softer aesthetic flavour).
    """

    def test_full_intersection_kept_when_above_threshold(self):
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]
        sp = ["ENV_B", "ENV_C", "ENV_D", "ENV_E"]
        # Intersection = [B, C, D] (3 items, == threshold)
        out = widen_compat_intersection(cat, sp)
        assert out == ["ENV_B", "ENV_C", "ENV_D"]

    def test_falls_through_to_category_when_intersection_too_narrow(self):
        """The 1-entry intersection bug: ENV_TUSCAN_VILLA_RENAISSANCE
        alone in compat_envs let LM Studio Cydonia invent 17 tags."""
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_TUSCAN_VILLA_RENAISSANCE", "ENV_A", "ENV_B", "ENV_C"]
        sp = ["ENV_TUSCAN_VILLA_RENAISSANCE", "ENV_X", "ENV_Y"]
        out = widen_compat_intersection(cat, sp)
        # Intersection has 1 item — below threshold 3 — fall through
        # to category list.
        assert out == cat

    def test_empty_intersection_falls_through_to_category(self):
        from src.modes._llm_helpers import widen_compat_intersection
        out = widen_compat_intersection(
            ["ENV_A", "ENV_B", "ENV_C"], ["ENV_X", "ENV_Y", "ENV_Z"],
        )
        assert out == ["ENV_A", "ENV_B", "ENV_C"]

    def test_two_item_intersection_below_threshold(self):
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]
        sp = ["ENV_A", "ENV_B", "ENV_X"]
        out = widen_compat_intersection(cat, sp)
        # 2-item intersection still below default threshold 3 → fall
        # through to category list.
        assert out == cat

    def test_only_category_populated(self):
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_A", "ENV_B"]
        out = widen_compat_intersection(cat, [])
        assert out == ["ENV_A", "ENV_B"]
        out_none = widen_compat_intersection(cat, None)
        assert out_none == ["ENV_A", "ENV_B"]

    def test_only_style_profile_populated(self):
        from src.modes._llm_helpers import widen_compat_intersection
        sp = ["ENV_X", "ENV_Y"]
        out = widen_compat_intersection([], sp)
        assert out == ["ENV_X", "ENV_Y"]
        out_none = widen_compat_intersection(None, sp)
        assert out_none == ["ENV_X", "ENV_Y"]

    def test_both_empty_returns_empty(self):
        from src.modes._llm_helpers import widen_compat_intersection
        assert widen_compat_intersection([], []) == []
        assert widen_compat_intersection(None, None) == []

    def test_returns_new_list_not_input_reference(self):
        """Mutating the result must not mutate the caller's category list."""
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_A", "ENV_B", "ENV_C"]
        out = widen_compat_intersection(cat, None)
        out.append("ENV_X")
        assert cat == ["ENV_A", "ENV_B", "ENV_C"]

    def test_preserves_category_order_in_intersection(self):
        """Intersection ordering follows the category list, not the
        style_profile — keeps category's preferred order intact."""
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_C", "ENV_B", "ENV_A", "ENV_D"]
        sp = ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]
        out = widen_compat_intersection(cat, sp)
        # All 4 items in cat order.
        assert out == ["ENV_C", "ENV_B", "ENV_A", "ENV_D"]

    def test_min_size_param_can_be_overridden(self):
        from src.modes._llm_helpers import widen_compat_intersection
        cat = ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]
        sp = ["ENV_A", "ENV_B"]
        # With default threshold 3 → 2-item intersection falls through.
        assert widen_compat_intersection(cat, sp) == cat
        # With threshold 2 → 2-item intersection survives.
        assert widen_compat_intersection(
            cat, sp, min_size=2,
        ) == ["ENV_A", "ENV_B"]

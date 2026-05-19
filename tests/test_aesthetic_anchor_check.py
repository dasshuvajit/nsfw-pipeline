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

from src.modes._llm_helpers import warn_if_missing_aesthetic_anchors


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
            plan, mode_name="CharacterMode",
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("CharacterMode" in m for m in msgs), (
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

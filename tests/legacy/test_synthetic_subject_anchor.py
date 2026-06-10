"""Tests for ``src.core.engine._synthetic_subject_anchor`` (round-22 F8).

Round-2 audit identified style_mode + variation_mode as the highest-risk
gap from F5: their series_plan dicts emit neither ``subject_description``
(theme_mode) nor ``subject_bias`` (niche_mode). Pre-F8 the facet
generator's user prompt rendered "(not provided)" for those two modes.

F8 adds a tier-aware synthetic anchor as a third-level fallback in the
engine's facet call site. This test pins the per-tier strings and the
back-compat default for unknown tiers.
"""

from __future__ import annotations

from src.core.engine import _synthetic_subject_anchor


def test_synthetic_subject_t1_clothed():
    out = _synthetic_subject_anchor("T1_suggestive")
    assert "adult woman" in out
    assert "mature features" in out
    assert "dressed elegantly" in out


def test_synthetic_subject_t2_implied():
    out = _synthetic_subject_anchor("T2_implied")
    assert "adult woman" in out
    assert "suggestive dress" in out or "implied undress" in out


def test_synthetic_subject_t3_artnude():
    out = _synthetic_subject_anchor("T3_artnude")
    assert "adult woman" in out
    assert "artistic nudity" in out


def test_synthetic_subject_t4_explicit():
    out = _synthetic_subject_anchor("T4_explicit")
    assert "adult woman" in out
    assert "fully nude" in out


def test_synthetic_subject_back_compat_unknown_tier():
    """Unknown tier → minimal back-compat anchor (just the age + solo
    invariants the composer relies on downstream)."""
    out = _synthetic_subject_anchor("T99_unknown")
    assert "adult woman" in out
    assert "mature features" in out


def test_synthetic_subject_never_empty():
    """Every tier-level value must produce a non-empty string so the
    facet generator's user prompt never gets "" or "(not provided)"
    when the synthetic anchor is the fallback."""
    for tier in (
        "T1_suggestive", "T2_implied",
        "T3_artnude", "T4_explicit",
        "",  # explicit empty
        None,  # noqa: typecheck — runtime should not blow up
    ):
        out = _synthetic_subject_anchor(tier or "")
        assert out, f"synthetic anchor returned empty for tier={tier!r}"


# ── Round-22 F8 — fallback-chain integration tests ───────────────


def test_resolve_subject_anchor_prefers_subject_description():
    """Theme mode emits ``subject_description``. The resolver must
    return that field's value verbatim before falling back."""
    from src.core.engine import resolve_subject_anchor
    series_plan = {
        "subject_description": "A confident adult woman, fully nude",
        "subject_bias": "should be ignored",
    }
    assert (
        resolve_subject_anchor(series_plan, "T3_artnude")
        == "A confident adult woman, fully nude"
    )


def test_resolve_subject_anchor_falls_to_subject_bias_when_description_empty():
    """Niche mode emits ``subject_bias``, not ``subject_description``.
    The resolver must fall through to bias when description is empty."""
    from src.core.engine import resolve_subject_anchor
    series_plan = {
        "subject_description": "",
        "subject_bias": "a model in fine-art nude posing",
    }
    assert (
        resolve_subject_anchor(series_plan, "T3_artnude")
        == "a model in fine-art nude posing"
    )


def test_resolve_subject_anchor_falls_to_synthetic_when_both_absent():
    """Style mode + variation mode emit NEITHER subject_description
    NOR subject_bias. The resolver must fall through to the tier-aware
    synthetic anchor — never returns empty."""
    from src.core.engine import resolve_subject_anchor
    series_plan = {
        "theme": "Some theme",
        "mood": "Some mood",
        # NO subject_description, NO subject_bias
    }
    out = resolve_subject_anchor(series_plan, "T4_explicit")
    assert "adult woman" in out, (
        f"resolver should fall to synthetic anchor at T4, got: {out!r}"
    )
    assert "fully nude" in out, (
        "T4 synthetic anchor missing the tier-aware nudity clause"
    )


def test_resolve_subject_anchor_none_series_plan_falls_to_synthetic():
    """Back-compat — when series_plan itself is None (defensive caller
    path), resolver still returns a non-empty tier-aware anchor."""
    from src.core.engine import resolve_subject_anchor
    out = resolve_subject_anchor(None, "T2_implied")
    assert "adult woman" in out
    assert "suggestive dress" in out or "implied undress" in out


def test_resolve_subject_anchor_treats_whitespace_as_empty():
    """Whitespace-only / None values in the series_plan dict must be
    treated as ABSENT so the chain advances. ``subject_description: " "``
    should NOT block the fallback to subject_bias."""
    from src.core.engine import resolve_subject_anchor
    # falsy check accepts both None and "" — confirm via empty string
    series_plan = {
        "subject_description": "",
        "subject_bias": "a model anchor",
    }
    out = resolve_subject_anchor(series_plan, "T3_artnude")
    assert out == "a model anchor"

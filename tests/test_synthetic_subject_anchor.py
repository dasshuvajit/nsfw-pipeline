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

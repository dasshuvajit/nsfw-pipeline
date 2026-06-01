"""Tests for the niche selector (src/niche/selector.py).

Selection is a pure function of (library, cursor, options) — deterministic,
no randomness — so these assert exact, reproducible behavior.
"""
from __future__ import annotations

import pytest

from src.niche.selector import (
    AestheticLock,
    NicheLibrary,
    NicheLibraryError,
    build_brief,
    build_selection,
    select_aesthetic_lock,
    select_niche,
    select_persona,
)


@pytest.fixture
def lib() -> NicheLibrary:
    return NicheLibrary.from_yaml()  # the real config/niche_library.yaml


# ── library loading ────────────────────────────────────────────────

def test_library_loads_with_core_and_trend(lib):
    assert lib.version >= 1
    assert lib.core(), "no core niches"
    assert lib.trend(), "no trend niches"
    # every niche declares a tier_band and tags
    for n in lib.niches:
        assert n.tier_band, f"{n.id} missing tier_band"
        assert n.tags, f"{n.id} missing tags"
        assert n.sub_looks, f"{n.id} missing sub_looks"


def test_by_id_and_unknown_raises(lib):
    n = lib.by_id("fine_art_figure_study")
    assert n.niche_class == "core"
    with pytest.raises(NicheLibraryError):
        lib.by_id("does_not_exist")


# ── niche selection ────────────────────────────────────────────────

def test_select_is_deterministic(lib):
    a = select_niche(lib, 5, tier="T3_artnude")
    b = select_niche(lib, 5, tier="T3_artnude")
    assert a.id == b.id


def test_force_id_overrides(lib):
    n = select_niche(lib, 0, tier="T3_artnude", force_id="film_noir_boudoir")
    assert n.id == "film_noir_boudoir"


def test_force_id_rejects_unsupported_tier(lib):
    # pinup_1950s supports only T1/T2 — forcing it at T4 must raise
    with pytest.raises(NicheLibraryError):
        select_niche(lib, 0, tier="T4_explicit", force_id="pinup_1950s")


def test_trend_turn_injects_trend(lib):
    # cursor % trend_period == trend_period-1 → trend niche
    n = select_niche(lib, 2, tier="T3_artnude", trend_period=3)
    assert n.niche_class == "trend"
    # a non-trend-turn cursor → core
    n2 = select_niche(lib, 0, tier="T3_artnude", trend_period=3)
    assert n2.niche_class == "core"


def test_selected_niche_supports_requested_tier(lib):
    for cur in range(12):
        n = select_niche(lib, cur, tier="T3_artnude")
        assert "T3_artnude" in n.tier_band


def test_weighting_reflected_over_a_sweep(lib):
    # over many core-turn cursors the higher-weight niche should appear
    # at least as often as a lower-weight one (round-robin over expansion)
    from collections import Counter
    picks = Counter()
    for cur in range(0, 60):
        if cur % 3 == 2:
            continue  # skip trend turns
        picks[select_niche(lib, cur, tier="T3_artnude").id] += 1
    # fine_art_figure_study (weight 1.2) >= art_deco_boudoir (weight 0.9)
    assert picks["fine_art_figure_study"] >= picks.get("art_deco_boudoir", 0)


# ── aesthetic lock ─────────────────────────────────────────────────

def test_aesthetic_lock_picks_from_niche(lib):
    n = lib.by_id("old_hollywood_glamour")
    lock = select_aesthetic_lock(n, 0)
    assert lock.palette in n.aesthetics["palettes"]
    assert lock.lighting in n.aesthetics["lighting"]
    assert lock.photographer in n.aesthetics["photographers"]


def test_aesthetic_lock_tolerates_missing_lists():
    from src.niche.selector import Niche
    bare = Niche(id="x", niche_class="core", weight=1.0, tier_band=["T3_artnude"],
                 da_folder="X", source="", tags=["a"], sub_looks=["l"],
                 aesthetics={}, brief_seed="seed")
    lock = select_aesthetic_lock(bare, 3)
    assert lock == AestheticLock(palette="", lighting="", photographer="")


# ── persona ────────────────────────────────────────────────────────

def test_persona_disabled_returns_none(lib):
    assert select_persona(lib, 0, enabled=False) is None


def test_persona_enabled_rotates(lib):
    p = select_persona(lib, 0, enabled=True)
    assert p is not None and p.name


def test_persona_by_name(lib):
    p = select_persona(lib, 0, name="Clara")
    assert p is not None and p.name == "Clara"
    with pytest.raises(NicheLibraryError):
        select_persona(lib, 0, name="Nobody")


# ── full selection + brief ─────────────────────────────────────────

def test_build_brief_contains_seed_and_lock(lib):
    sel = build_selection(lib, 0, tier="T3_artnude",
                          force_niche="fine_art_figure_study")
    brief = build_brief(sel)
    assert sel.niche.brief_seed.split(".")[0][:20] in brief
    assert "SERIES AESTHETIC LOCK" in brief
    assert sel.sub_looks  # carried for the prompt engine


def test_build_brief_includes_persona_when_bound(lib):
    sel = build_selection(lib, 0, tier="T3_artnude",
                          force_niche="old_hollywood_glamour",
                          persona=True, persona_name="Margot")
    brief = build_brief(sel)
    assert "RECURRING SUBJECT" in brief
    assert "Margot" in brief


def test_build_brief_no_persona_clause_when_unbound(lib):
    sel = build_selection(lib, 0, tier="T3_artnude",
                          force_niche="modern_boudoir")
    assert "RECURRING SUBJECT" not in build_brief(sel)

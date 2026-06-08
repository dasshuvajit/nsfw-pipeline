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
    select_niche_cycle,
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


def test_every_niche_has_enough_sub_looks(lib):
    """Cross-series variety needs scene room: ≥3 sub-looks per niche so the
    per-run rotation offset lands on a different scene each run. The 6 niches
    that were thin (2 sub-looks → near-identical re-runs) are expanded to ≥5."""
    EXPANDED = {"goth_romantic", "bohemian_naturallight", "poolside_goldenhour",
                "burlesque_cabaret", "cyberpunk_pinup", "cottagecore_pastoral"}
    for n in lib.niches:
        assert len(n.sub_looks) >= 3, f"{n.id} has only {len(n.sub_looks)} sub_looks"
    for n in lib.niches:
        if n.id in EXPANDED:
            assert len(n.sub_looks) >= 5, \
                f"{n.id} expected ≥5 sub_looks, has {len(n.sub_looks)}"
    # the expanded sub-looks must not reintroduce a mirror motif (the validator
    # bans literal mirrors — only a negated "no mirror" instruction is allowed).
    for n in lib.niches:
        for s in n.sub_looks:
            assert "mirror" not in s.lower() or "no mirror" in s.lower(), \
                f"{n.id} sub_look reintroduces a mirror: {s!r}"


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


# ── per-niche avoid_motifs (mirror-prone niches) ───────────────────────

def test_mirror_prone_niches_declare_avoid_motifs(lib):
    for nid in ("old_hollywood_glamour", "modern_boudoir", "film_noir_boudoir",
                "art_deco_boudoir", "burlesque_cabaret"):
        n = lib.by_id(nid)
        assert n.avoid_motifs, f"{nid} should declare avoid_motifs"
        assert any("mirror" in m.lower() for m in n.avoid_motifs)


def test_build_brief_injects_avoid_motifs(lib):
    sel = build_selection(lib, 0, tier="T3_artnude",
                          force_niche="old_hollywood_glamour")
    brief = build_brief(sel)
    assert "DO NOT DEPICT" in brief
    assert "mirror" in brief.lower()


def test_build_brief_omits_avoid_clause_when_none(lib):
    # fine_art_figure_study has no avoid_motifs
    sel = build_selection(lib, 0, tier="T3_artnude",
                          force_niche="fine_art_figure_study")
    assert not lib.by_id("fine_art_figure_study").avoid_motifs
    assert "DO NOT DEPICT" not in build_brief(sel)


# ── --auto niche cycle: exhaust all before repeating ───────────────────

def test_cycle_exhausts_all_supporting_niches_before_repeat(lib):
    supporting = [n.id for n in lib.niches if "T3_artnude" in n.tier_band]
    used: list[str] = []
    picks: list[str] = []
    for _ in range(len(supporting)):
        n, reset = select_niche_cycle(lib, used, tier="T3_artnude")
        assert reset is False
        picks.append(n.id)
        used.append(n.id)
    assert sorted(picks) == sorted(supporting)          # every niche, exactly once
    assert len(set(picks)) == len(supporting)           # no repeats within the cycle


def test_cycle_resets_after_wrap(lib):
    supporting = [n.id for n in lib.niches if "T3_artnude" in n.tier_band]
    used = list(supporting)  # everything already used
    n, reset = select_niche_cycle(lib, used, tier="T3_artnude")
    assert reset is True
    assert n.id in supporting                            # picks a valid fresh start


def test_cycle_only_returns_tier_supporting(lib):
    used: list[str] = []
    for _ in range(len(lib.niches) + 3):
        n, reset = select_niche_cycle(lib, used, tier="T4_explicit")
        assert "T4_explicit" in n.tier_band
        if reset:
            used = []
        used.append(n.id)


def test_cycle_is_deterministic(lib):
    a, _ = select_niche_cycle(lib, ["fine_art_figure_study"], tier="T3_artnude")
    b, _ = select_niche_cycle(lib, ["fine_art_figure_study"], tier="T3_artnude")
    assert a.id == b.id


# ── fantasy/historical/monochrome niches (2026-06-04 competitor-proven) ──

def test_new_fantasy_historical_niches_present(lib):
    for nid in ("renaissance_baroque", "mythology_goddess", "medieval_lady",
                "angelic_divine", "arabian_nights", "dark_fantasy_vampire",
                "monochrome_fine_art"):
        n = lib.by_id(nid)
        assert n.sub_looks and n.aesthetics.get("lighting")
        assert "T3_artnude" in n.tier_band or "T2_implied" in n.tier_band


def test_risky_theme_niches_block_problematic_tropes(lib):
    # arabian_nights/vampire/medieval must avoid harem-slave/victim/damsel
    motifs = " ".join(lib.by_id("arabian_nights").avoid_motifs).lower()
    assert "harem" in motifs and "slave" in motifs
    assert "victim" in " ".join(lib.by_id("dark_fantasy_vampire").avoid_motifs).lower()
    assert "damsel" in " ".join(lib.by_id("medieval_lady").avoid_motifs).lower()


def test_monochrome_niche_specifies_black_and_white(lib):
    n = lib.by_id("monochrome_fine_art")
    assert "black-and-white" in n.brief_seed.lower() or "black and white" in n.brief_seed.lower()

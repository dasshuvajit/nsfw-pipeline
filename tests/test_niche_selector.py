"""Tests for the niche selector (src/niche/selector.py).

Selection is a pure function of (library, cursor, options) — deterministic,
no randomness — so these assert exact, reproducible behavior.
"""
from __future__ import annotations

import pytest

from src.niche.selector import (
    AestheticLock,
    Niche,
    NicheLibrary,
    NicheLibraryError,
    Persona,
    build_brief,
    build_selection,
    persona_locked_look,
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
    per-run rotation offset lands on a different scene each run. The niches that
    were thin (near-identical re-runs) are expanded to ≥6 (2026-06-17 content
    pass)."""
    EXPANDED = {"goth_romantic", "bohemian_naturallight", "poolside_goldenhour",
                "burlesque_cabaret", "cyberpunk_pinup", "cottagecore_pastoral"}
    for n in lib.niches:
        assert len(n.sub_looks) >= 3, f"{n.id} has only {len(n.sub_looks)} sub_looks"
    for n in lib.niches:
        if n.id in EXPANDED:
            assert len(n.sub_looks) >= 6, \
                f"{n.id} expected ≥6 sub_looks, has {len(n.sub_looks)}"
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
    # pinup_1950s supports only T1 (T2 dropped after the drift audit) — T4 must raise
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


def test_persona_carries_face_and_complexion(lib):
    # The identity axes that let a persona fully replace the look-pool rotation.
    for name in ("Clara", "Sable", "Margot", "Imani", "Mei"):
        p = select_persona(lib, 0, name=name)
        assert p.face, f"{name} missing face"
        assert p.complexion, f"{name} missing complexion"


def test_persona_pool_has_complexion_diversity(lib):
    # G6: the roster should span skin tones, not skew light/European.
    comps = " ".join(p.complexion.lower() for p in lib.personas)
    assert "deep brown" in comps          # Imani
    assert "east-asian" in comps          # Mei
    assert "olive" in comps               # Margot


def test_persona_locked_look_orders_axes():
    p = Persona(name="X", hair="raven bob", build="petite",
                age_anchor="in her late twenties", wardrobe_vibe="silk",
                arc="muse", face="green eyes", complexion="olive skin")
    # hair, build, face, complexion, age — the _creative_look axis order.
    assert persona_locked_look(p) == (
        "raven bob, petite, green eyes, olive skin, in her late twenties")
    # wardrobe is set-dressing, never part of the locked identity.
    assert "silk" not in persona_locked_look(p)


def test_persona_locked_look_drops_empty_axes():
    # A legacy/partial persona (no face/complexion) still locks gracefully.
    p = Persona(name="Old", hair="blonde waves", build="slender",
                age_anchor="in her thirties", wardrobe_vibe="", arc="")
    assert persona_locked_look(p) == "blonde waves, slender, in her thirties"


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
    # Asserts the same-woman CONTRACT and defers appearance to the per-image
    # SUBJECT LOOK (the contradiction fix) rather than re-listing hair/build.
    assert "SAME" in brief and "SUBJECT LOOK" in brief


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


# ── aspirational-luxe lifestyle lane (2026-06-17) ──────────────────

def test_aspirational_luxe_lifestyle_niche_present(lib):
    """The modern aspirational-luxe lifestyle lane. PUBLIC lane is T1 only —
    a T2 validation render drifted to bare breasts on the undress-coded scenes
    (NudeNet false-negatived the topless), so T2 is dropped like poolside; T1
    forces covered wardrobe for DA, T3/T4 are gated/Fanvue."""
    n = lib.by_id("aspirational_luxe")
    assert n.family == "golden_hour"
    assert n.tier_band == ["T1_suggestive", "T3_artnude", "T4_explicit"]
    assert "T2_implied" not in n.tier_band      # the drift-prone public tier is excluded
    assert len(n.sub_looks) >= 6
    assert n.aesthetics.get("palettes") and n.aesthetics.get("lighting") \
        and n.aesthetics.get("photographers")


def test_aspirational_luxe_is_modern_lifestyle_not_fantasy(lib):
    """It must read as contemporary lifestyle editorial — NOT a period/fantasy
    costume piece — so it stays a distinct lane and respects the DA hyperrealism
    posture (modern-lifestyle framing, not 'real person' claims)."""
    n = lib.by_id("aspirational_luxe")
    bs = n.brief_seed.lower()
    assert "lifestyle" in bs and "not" in bs and ("period" in bs or "fantasy" in bs)
    for s in n.sub_looks:
        low = s.lower()
        assert not any(w in low for w in ("velvet", "marble", "chiton", "laurel")), \
            f"lifestyle sub_look drifts to a fantasy/period material: {s!r}"


def test_aspirational_luxe_has_santorini_sublook(lib):
    """The 2026-06-18 Santorini caldera sub_look adds the influencer-glamour
    reference iconography (blue domes / Aegean / Cyclades) the generic Mediterranean
    terrace lacked — present, modern-lifestyle (plaster not marble), gate-clean."""
    from scripts.audit_prompts import score_prompt
    n = lib.by_id("aspirational_luxe")
    santorini = [s for s in n.sub_looks if "santorini" in s.lower()]
    assert santorini, "Santorini sub_look missing from aspirational_luxe"
    s = santorini[0].lower()
    assert "blue-domed" in s and "aegean" in s and "cyclades" in s
    assert "marble" not in s          # period-material ban — whitewashed plaster instead
    score, issues = score_prompt(santorini[0], "T1_suggestive")
    assert score >= 9.0, f"Santorini sub_look should pass the gate, got {score} {issues}"


def test_drift_prone_funnel_niches_are_t1_public(lib):
    """2026-06-17 drift audit: niches that strip clothing at the implied tier are
    pinned to a T1-only public lane (T2 dropped). Pinup/poolside become T1-only;
    art_deco keeps T1 public + T3 gated but drops the strip-prone T2, and its
    sub_looks no longer pre-undress (the self-inflicted 44% T1 strip)."""
    assert lib.by_id("pinup_1950s").tier_band == ["T1_suggestive"]
    assert lib.by_id("poolside_goldenhour").tier_band == ["T1_suggestive"]
    assert lib.by_id("art_deco_boudoir").tier_band == ["T1_suggestive", "T3_artnude"]
    looks = " ".join(lib.by_id("art_deco_boudoir").sub_looks).lower()
    assert "pooling off" not in looks and "half-undone" not in looks


def test_no_sublook_pre_undresses_the_subject(lib):
    """Catalog-wide invariant: a sub_look describes scene + INTACT wardrobe; the
    TIER governs undress. Sub_looks that pre-undress ('robe slipping', 'gown
    loosened at the lacing', 'velvet pooled', 'knit loose over bare skin') fight
    the T1 covered directive and drove render-drift (art_deco 44%, old_hollywood
    25% T1 strip). These verbs are banned from sub_looks."""
    BANNED = ("slipping", "pooled", "loosened at", "loose over bare",
              "half-undone", "discarded", "falling open", "pooling off")
    offenders = []
    for n in lib.niches:
        for s in n.sub_looks:
            low = s.lower()
            hits = [b for b in BANNED if b in low]
            if hits:
                offenders.append((n.id, hits, s[:60]))
    assert not offenders, f"sub_looks pre-undress (let the tier govern): {offenders}"


def test_era_and_culture_locked_niches_lock_wardrobe(lib):
    """The contemporary-Western GARMENT_TYPES variety axis must NOT override the
    wardrobe of niches whose dress is era/genre/culture-DEFINING (it once put a
    black tuxedo jacket and a swimsuit on a South-Asian heritage editorial). Those
    niches carry lock_wardrobe; contemporary/period-glamour niches do not."""
    LOCKED = {"south_asian_editorial", "iberian_flamenco", "slavic_folk",
              "renaissance_baroque", "medieval_lady", "mythology_goddess",
              "angelic_divine", "arabian_nights", "dark_fantasy_vampire",
              "fantasy_glamour"}
    for nid in LOCKED:
        assert lib.by_id(nid).lock_wardrobe is True, f"{nid} must lock_wardrobe"
    # contemporary niches keep the garment axis for wardrobe variety
    for nid in ("modern_boudoir", "aspirational_luxe", "athletic_studio",
                "wild_nature", "poolside_goldenhour"):
        assert lib.by_id(nid).lock_wardrobe is False, f"{nid} should keep the garment axis"


def test_thermal_bathhouse_niche(lib):
    """2026-06-22 wellness lane: a bathing/sauna/hot-spring niche (the home for the
    previously-orphaned hot-spring + sauna scenes), spanning world bathing culture.
    Full T1-T3: T2 (steam-veiled towel/swimsuit tease) is the niche's signature
    register and is KEPT — the NudeNet gate quarantines any render-drift, and
    lock_wardrobe (below) removes the falling-open garment-axis wear that drove most
    of it. Wardrobe is LOCKED so the setting's bathing wear (towel/robe/swimsuit,
    named in the sub_look) governs, not the portable garment axis."""
    n = lib.by_id("thermal_bathhouse")
    assert len(n.sub_looks) == 12
    assert n.tier_band == ["T1_suggestive", "T2_implied", "T3_artnude"]
    assert n.family == "golden_hour"
    # bathing wear (towel/robe/swimsuit, from the sub_look) governs — not the portable
    # contemporary GARMENT_TYPES axis (a jersey jumpsuit in a lagoon read wrong).
    assert n.lock_wardrobe is True
    blob = " ".join(n.sub_looks).lower()
    assert "sauna" in blob and ("hot spring" in blob or "hot-spring" in blob or "onsen" in blob)
    # no sub_look pre-undresses or reintroduces a mirror (catalog invariants)
    for s in n.sub_looks:
        assert "mirror" not in s.lower()
        assert not any(b in s.lower() for b in ("slipping", "pooled", "half-undone", "falling open"))


def test_summer_gap_filled_in_outdoor_niches(lib):
    """2026-06-22 seasonal enrichment: the outdoor niches that were thin on SUMMER
    now carry at least one high-summer scene, rounding out the 4-season spread."""
    summer = r"summer|midsummer|sun-baked|haymaking|high-summer|cicada"
    import re
    for nid in ("wild_nature", "cottagecore_pastoral", "bohemian_naturallight",
                "aspirational_luxe", "athletic_studio"):
        looks = " ".join(lib.by_id(nid).sub_looks).lower()
        assert re.search(summer, looks), f"{nid} still missing a summer scene"


def test_aspirational_luxe_sub_looks_clear_audit_specificity(lib):
    """Each sub_look front-loads the optical craft the audit gate rewards — a
    named light DIRECTION, ≥2 whitelisted MATERIAL nouns, ≥1 MICRO-TEXTURE token,
    and ZERO cliché phrases — so the seed clears the (2026-06-17 softened)
    specificity ladder. The materials must also VARY across scenes (the sameness
    fix): no single material appears in every sub_look."""
    from scripts.audit_prompts import (
        _CLICHE_PHRASES,
        _LIGHT_DIRECTION_TOKENS,
        _MATERIAL_NOUNS,
        _MICRO_TEXTURE_TOKENS,
    )
    n = lib.by_id("aspirational_luxe")
    per_look_materials = []
    for s in n.sub_looks:
        low = s.lower()
        mats = {t for t in _MATERIAL_NOUNS if t in low}
        per_look_materials.append(mats)
        assert any(t in low for t in _LIGHT_DIRECTION_TOKENS), f"no light direction: {s!r}"
        assert len(mats) >= 2, f"thin materials: {s!r}"
        assert len({t for t in _MICRO_TEXTURE_TOKENS if t in low}) >= 1, f"thin micro-texture: {s!r}"
        assert not [c for c in _CLICHE_PHRASES if c in low], f"cliché in sub_look: {s!r}"
    # diversity: no material is shared by ALL sub_looks (the brass/satin/stone
    # cramming that made every render look alike).
    shared = set.intersection(*per_look_materials)
    assert not shared, f"material(s) in EVERY sub_look (sameness): {shared}"


def test_auto_tier_in_band_and_set_on_couture_niches(lib):
    """2026-06-26 niche×tier fix: couture/cultural niches whose VALUE is the garment
    declare an auto_tier (the tasteful tier --auto runs them at, instead of forcing
    T3 where the nudity contract fought the costume → dropped scenes). Every auto_tier
    must be IN the niche's tier_band; nude-coded niches leave it blank."""
    expected = {
        # garment-LOCKED niches → T1 (keep the heritage/period dress WORN; T2 drapes
        # it off → generic glamour + lost identity, verified by render):
        "medieval_lady": "T1_suggestive",
        "south_asian_editorial": "T1_suggestive",
        "iberian_flamenco": "T1_suggestive",
        "slavic_folk": "T1_suggestive",
        "art_deco_boudoir": "T1_suggestive",    # band has no T2 (dropped for drift)
        # non-locked glamour niches → T2 (sensual-implied is on-brand; no garment to keep):
        "film_noir_boudoir": "T2_implied",
        "goth_romantic": "T2_implied",
    }
    by_id = {n.id: n for n in lib.niches}
    for nid, tier in expected.items():
        assert by_id[nid].auto_tier == tier, f"{nid} auto_tier"
    for n in lib.niches:
        if n.auto_tier:
            # any auto_tier is in-band and a tasteful (non-explicit) public tier
            assert n.auto_tier in n.tier_band, f"{n.id}: auto_tier not in band"
            assert n.auto_tier in ("T1_suggestive", "T2_implied"), \
                f"{n.id}: auto_tier should be a tasteful public tier"
            # garment-locked niches must use T1 — T2's implied-undress directive
            # strips the locked heritage/period garment that IS the niche's value.
            if n.lock_wardrobe:
                assert n.auto_tier == "T1_suggestive", \
                    f"{n.id}: lock_wardrobe niche must auto_tier to T1 (T2 strips the garment)"
    # nude-coded niches keep it blank → --auto still runs them at the T3 default
    assert by_id["fine_art_figure_study"].auto_tier == ""
    assert by_id["aspirational_luxe"].auto_tier == ""


def test_niche_rejects_auto_tier_outside_band():
    """The model guards against a typo'd auto_tier that isn't in the band (which would
    later blow up at build_selection's tier check)."""
    with pytest.raises(NicheLibraryError):
        Niche.from_dict({
            "id": "x", "tier_band": ["T1_suggestive", "T3_artnude"],
            "auto_tier": "T2_implied",  # not in band
            "sub_looks": ["a"], "brief_seed": "s",
        })


def test_monochrome_niche_flags_grayscale(lib):
    """2026-06-26: monochrome_fine_art is flagged grayscale so its renders are
    deterministically desaturated to true B&W at package time (the model renders
    muted colour from a 'monochrome' prompt). Other niches stay colour."""
    by_id = {n.id: n for n in lib.niches}
    assert by_id["monochrome_fine_art"].grayscale is True
    assert by_id["mythology_goddess"].grayscale is False
    # grayscale is a narrow opt-in, not accidentally broad
    grays = [n.id for n in lib.niches if n.grayscale]
    assert grays == ["monochrome_fine_art"], grays

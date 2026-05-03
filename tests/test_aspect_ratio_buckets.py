"""Per-family megapixel-tier resolution buckets."""

from __future__ import annotations

import pytest

from src.core.aspect_ratio_buckets import (
    FAMILY_TIER,
    RATIO_BUCKETS_1MP,
    RATIO_BUCKETS_1_5MP,
    RATIO_BUCKETS_2MP,
    TIER_BUCKETS,
    get_family_bucket,
    get_family_resolution,
)

ALL_RATIOS = ("portrait_23", "portrait_916", "square", "landscape")


# ---- Tier-bucket integrity --------------------------------------------------

@pytest.mark.parametrize("bucket", [RATIO_BUCKETS_1MP, RATIO_BUCKETS_1_5MP, RATIO_BUCKETS_2MP])
def test_every_bucket_covers_all_4_ratios(bucket):
    assert set(bucket.keys()) == set(ALL_RATIOS)


@pytest.mark.parametrize("bucket", [RATIO_BUCKETS_1MP, RATIO_BUCKETS_1_5MP, RATIO_BUCKETS_2MP])
def test_every_resolution_is_multiple_of_64(bucket):
    """SDXL UNet + Flux MMDiT both reject non-multiples-of-64."""
    for ratio, (w, h) in bucket.items():
        assert w % 64 == 0, f"{ratio} width {w} is not a multiple of 64"
        assert h % 64 == 0, f"{ratio} height {h} is not a multiple of 64"


def test_1mp_bucket_lands_near_one_megapixel():
    for ratio, (w, h) in RATIO_BUCKETS_1MP.items():
        mp = (w * h) / 1_000_000
        assert 0.9 <= mp <= 1.1, f"{ratio} at {w}×{h} is {mp:.2f}MP, not ~1MP"


def test_15mp_bucket_lands_near_one_and_a_half_megapixel():
    for ratio, (w, h) in RATIO_BUCKETS_1_5MP.items():
        mp = (w * h) / 1_000_000
        # Slight slack on portrait_916 — the 9:16 ratio at 1.5MP rounds
        # awkwardly against multiples of 64.
        assert 1.4 <= mp <= 1.7, f"{ratio} at {w}×{h} is {mp:.2f}MP, not ~1.5MP"


def test_2mp_bucket_lands_near_two_megapixel():
    for ratio, (w, h) in RATIO_BUCKETS_2MP.items():
        mp = (w * h) / 1_000_000
        assert 1.8 <= mp <= 2.05, f"{ratio} at {w}×{h} is {mp:.2f}MP, not ~2MP"


@pytest.mark.parametrize(
    "ratio,expected_orientation",
    [
        ("portrait_23", "portrait"),
        ("portrait_916", "portrait"),
        ("square", "square"),
        ("landscape", "landscape"),
    ],
)
def test_orientations_match_ratio_names(ratio, expected_orientation):
    """portrait_* must be taller than wide; landscape must be wider; square is square."""
    for bucket in (RATIO_BUCKETS_1MP, RATIO_BUCKETS_1_5MP, RATIO_BUCKETS_2MP):
        w, h = bucket[ratio]
        if expected_orientation == "portrait":
            assert h > w, f"{ratio} should be portrait, got {w}×{h}"
        elif expected_orientation == "landscape":
            assert w > h, f"{ratio} should be landscape, got {w}×{h}"
        else:
            assert w == h, f"{ratio} should be square, got {w}×{h}"


# ---- Family→tier mapping ----------------------------------------------------

def test_all_six_families_have_a_tier_assignment():
    for family in ("sdxl", "pony", "illustrious", "flux", "chroma", "flux2"):
        assert family in FAMILY_TIER


def test_family_tier_keys_resolve_to_known_buckets():
    for family, tier in FAMILY_TIER.items():
        assert tier in TIER_BUCKETS, f"family {family} maps to unknown tier {tier!r}"


# ---- get_family_bucket / get_family_resolution -----------------------------

def test_sdxl_resolves_to_1mp_bucket():
    bucket = get_family_bucket("sdxl")
    assert bucket is RATIO_BUCKETS_1MP


def test_flux_resolves_to_1_5mp_bucket():
    bucket = get_family_bucket("flux")
    assert bucket is RATIO_BUCKETS_1_5MP


def test_flux2_resolves_to_2mp_bucket():
    bucket = get_family_bucket("flux2")
    assert bucket is RATIO_BUCKETS_2MP


def test_unknown_family_falls_back_to_1mp():
    bucket = get_family_bucket("nonexistent_family")
    assert bucket is RATIO_BUCKETS_1MP


def test_none_family_falls_back_to_1mp():
    bucket = get_family_bucket(None)
    assert bucket is RATIO_BUCKETS_1MP


def test_get_family_resolution_returns_specific_size():
    res = get_family_resolution("flux", "portrait_23")
    assert res == RATIO_BUCKETS_1_5MP["portrait_23"]


def test_get_family_resolution_unknown_ratio_returns_none():
    assert get_family_resolution("sdxl", "bogus_ratio") is None


def test_get_family_resolution_none_family_uses_1mp():
    res = get_family_resolution(None, "square")
    assert res == RATIO_BUCKETS_1MP["square"]

"""Per-family ratio→resolution lookup at 1MP / 1.5MP / 2MP tiers.

The legacy ``RATIO_TO_RESOLUTION`` in ``ratio_selector.py`` is sized for
SDXL (1MP, 1024×1024 baseline). Newer model families render natively at
larger resolutions:

- **SDXL / Pony / Illustrious** — 1.0 MP. Trained at 1024², degrades on
  the obvious off-distribution sizes (extreme 9:16 / 16:9).
- **Flux / Chroma** — 1.5 MP. Trained on a wider distribution, handles
  1216×832 cleanly.
- **Flux.2 Klein 9B** — 2.0 MP. BFL guide explicitly recommends
  1408×1408 / 1536×1024 as native targets.

All tiers use multiples of 64 — both SDXL's UNet and Flux's MMDiT
reject non-multiples (auto-pad or crash, depending on the version).

Resolution precedence in ``ratio_selector.get_resolution`` becomes:
  1. ``model_resolutions`` (per-model YAML override) — highest priority
  2. ``family_bucket(family_id)`` (this module) — middle priority
  3. ``RATIO_TO_RESOLUTION`` (legacy 1MP defaults) — fallback
"""

from __future__ import annotations


# ----- per-tier ratio → (width, height) tables -----------------------------

RATIO_BUCKETS_1MP: dict[str, tuple[int, int]] = {
    "portrait_23":  (832, 1216),    # 2:3 @ 1.01 MP
    "portrait_916": (768, 1344),    # 9:16 @ 1.03 MP (closest mult-of-64)
    "square":       (1024, 1024),   # 1:1 @ 1.05 MP
    "landscape":    (1216, 832),    # 3:2 @ 1.01 MP
}

RATIO_BUCKETS_1_5MP: dict[str, tuple[int, int]] = {
    "portrait_23":  (1024, 1536),   # 2:3 @ 1.57 MP
    "portrait_916": (896, 1600),    # 9:16 @ 1.43 MP
    "square":       (1280, 1280),   # 1:1 @ 1.64 MP
    "landscape":    (1536, 1024),   # 3:2 @ 1.57 MP
}

RATIO_BUCKETS_2MP: dict[str, tuple[int, int]] = {
    "portrait_23":  (1152, 1728),   # 2:3 @ 1.99 MP
    "portrait_916": (1024, 1792),   # 9:16 @ 1.83 MP
    "square":       (1408, 1408),   # 1:1 @ 1.98 MP
    "landscape":    (1728, 1152),   # 3:2 @ 1.99 MP
}


# ----- family → tier --------------------------------------------------------

FAMILY_TIER: dict[str, str] = {
    "sdxl":        "1MP",
    "pony":        "1MP",
    "illustrious": "1MP",
    "flux":        "1.5MP",
    "chroma":      "1.5MP",
    "flux2":       "2MP",
}

TIER_BUCKETS: dict[str, dict[str, tuple[int, int]]] = {
    "1MP":   RATIO_BUCKETS_1MP,
    "1.5MP": RATIO_BUCKETS_1_5MP,
    "2MP":   RATIO_BUCKETS_2MP,
}


# ----- public API -----------------------------------------------------------


def get_family_bucket(family_id: str | None) -> dict[str, tuple[int, int]]:
    """Return the ratio → (w, h) map a family renders at natively.

    Falls back to 1MP when the family is unknown or ``None``. Callers
    should prefer the model-level ``resolution_*`` override (handled by
    ``ratio_selector.model_resolution_overrides``) before hitting this.
    """
    if not family_id:
        return RATIO_BUCKETS_1MP
    tier = FAMILY_TIER.get(family_id, "1MP")
    return TIER_BUCKETS[tier]


def get_family_resolution(
    family_id: str | None, ratio: str
) -> tuple[int, int] | None:
    """Look up one ratio's native resolution for a family.

    Returns ``None`` when the family is unknown or the ratio key is not
    in the bucket — callers fall back to the legacy 1MP defaults.
    """
    bucket = get_family_bucket(family_id)
    return bucket.get(ratio)

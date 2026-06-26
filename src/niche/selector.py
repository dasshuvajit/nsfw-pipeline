"""Niche selector — deterministic "what to make" selection.

Loads ``config/niche_library.yaml`` and, given a persisted integer
``cursor`` (the run index — managed by the caller, mirroring
``art_series.py``'s ``.last_seed`` pattern), produces a :class:`Selection`:

  * ONE niche per run — mostly evergreen *core* (weighted round-robin),
    with a *trend* niche injected every ``trend_period``-th run for freshness
    (the user's "evergreen core + rotating trend" decision).
  * an optional recurring *persona* (held constant across the series), and
  * a per-series *aesthetic-lock* (one palette + one lighting recipe + one
    photographer reference, chosen once) for a coherent signature look.

Selection is a PURE function of ``(library, cursor, options)`` — no
randomness — so runs are reproducible and the logic is unit-testable.
``Math.random``-style nondeterminism is deliberately avoided; variety comes
from advancing the cursor across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LIBRARY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "niche_library.yaml"
)

# Every Nth run injects a trend niche instead of a core one.
DEFAULT_TREND_PERIOD = 3


class NicheLibraryError(Exception):
    """Raised when the niche library is missing, malformed, or a request
    cannot be satisfied (e.g. no niche supports the requested tier)."""


@dataclass(frozen=True)
class Niche:
    id: str
    niche_class: str                 # "core" | "trend"
    weight: float
    tier_band: list[str]
    da_folder: str
    source: str
    tags: list[str]
    sub_looks: list[str]
    aesthetics: dict[str, list[str]]  # palettes / lighting / photographers
    brief_seed: str
    avoid_motifs: list[str] = field(default_factory=list)  # per-niche "do NOT depict" hints
    family: str = ""                 # 5-family DA architecture (DA_GO_TO_MARKET.md §3)
    lock_wardrobe: bool = False      # wardrobe is era/genre/culture/setting-defining (heritage
    #                                  dress, period costume, or a bathhouse's robe/towel) → skip
    #                                  the portable GARMENT_TYPES axis (the sub_look governs)
    auto_tier: str = ""              # preferred tier for the --auto rotation (must be in
    #                                  tier_band). Garment/cultural niches whose VALUE is the
    #                                  couture (the dress IS the content) declare a tasteful
    #                                  T1/T2 here so --auto runs them where they shine + feeds
    #                                  the public DA funnel; nude-coded niches leave it blank
    #                                  and --auto uses its default (T3). Empty → no override.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Niche":
        try:
            niche = cls(
                id=str(d["id"]),
                niche_class=str(d.get("class", "core")),
                weight=float(d.get("weight", 1.0)),
                tier_band=list(d.get("tier_band") or []),
                da_folder=str(d.get("da_folder", d["id"])),
                source=str(d.get("source", "")),
                tags=list(d.get("tags") or []),
                sub_looks=list(d.get("sub_looks") or []),
                aesthetics=dict(d.get("aesthetics") or {}),
                brief_seed=str(d.get("brief_seed", "")),
                avoid_motifs=list(d.get("avoid_motifs") or []),
                family=str(d.get("family", "")),
                lock_wardrobe=bool(d.get("lock_wardrobe", False)),
                auto_tier=str(d.get("auto_tier", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NicheLibraryError(f"invalid niche entry {d!r}: {exc}") from exc
        if niche.auto_tier and niche.auto_tier not in niche.tier_band:
            raise NicheLibraryError(
                f"niche {niche.id!r}: auto_tier {niche.auto_tier!r} not in "
                f"tier_band {niche.tier_band}")
        return niche


@dataclass(frozen=True)
class Persona:
    name: str
    hair: str
    build: str
    age_anchor: str
    wardrobe_vibe: str
    arc: str
    # Identity axes mirroring config/creative_direction.yaml look_pools — so a
    # bound persona can fully REPLACE the per-image look rotation (face + skin
    # were the missing axes that let the rotation contradict the persona).
    # Defaulted (must follow the non-default fields) → legacy personas still load.
    face: str = ""
    complexion: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Persona":
        return cls(
            name=str(d.get("name", "")),
            hair=str(d.get("hair", "")),
            build=str(d.get("build", "")),
            age_anchor=str(d.get("age_anchor", "an adult woman")),
            wardrobe_vibe=str(d.get("wardrobe_vibe", "")),
            arc=str(d.get("arc", "")),
            face=str(d.get("face", "")),
            complexion=str(d.get("complexion", "")),
        )


def persona_locked_look(p: Persona) -> str:
    """A persona's fixed appearance as ONE look string, in the same axis order
    ``art_director._creative_look`` emits (hair, build/figure, face, complexion,
    age) so it drops straight in as a locked ``look_target`` for every image of
    the series. Empties are skipped so partial/legacy personas degrade
    gracefully. Wardrobe is intentionally EXCLUDED — it is set-dressing the tier
    overrides per image, not identity."""
    parts = [p.hair, p.build, p.face, p.complexion, p.age_anchor]
    return ", ".join(s for s in (x.strip() for x in parts) if s)


@dataclass(frozen=True)
class AestheticLock:
    palette: str
    lighting: str
    photographer: str


@dataclass(frozen=True)
class Selection:
    niche: Niche
    tier: str
    aesthetic_lock: AestheticLock
    sub_looks: list[str]
    persona: Persona | None = None


@dataclass
class NicheLibrary:
    version: int
    niches: list[Niche]
    personas: list[Persona] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "NicheLibrary":
        p = Path(path) if path else DEFAULT_LIBRARY_PATH
        if not p.exists():
            raise NicheLibraryError(f"niche library not found: {p}")
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            raise NicheLibraryError(f"niche library {p} is not valid YAML: {exc}") from exc
        niches = [Niche.from_dict(n) for n in (data.get("niches") or [])]
        if not niches:
            raise NicheLibraryError(f"niche library {p} declares no niches")
        personas = [Persona.from_dict(x) for x in (data.get("persona_pool") or [])]
        return cls(version=int(data.get("version", 1)), niches=niches,
                   personas=personas)

    def core(self) -> list[Niche]:
        return [n for n in self.niches if n.niche_class == "core"]

    def trend(self) -> list[Niche]:
        return [n for n in self.niches if n.niche_class == "trend"]

    def by_id(self, niche_id: str) -> Niche:
        for n in self.niches:
            if n.id == niche_id:
                return n
        raise NicheLibraryError(
            f"unknown niche id {niche_id!r}; known: {[n.id for n in self.niches]}"
        )


def _weighted_expand(niches: list[Niche]) -> list[Niche]:
    """Repeat each niche ~proportional to its weight (×10, min 1) so a
    cursor-modulo index respects weighting via deterministic round-robin."""
    out: list[Niche] = []
    for n in niches:
        out.extend([n] * max(1, round(n.weight * 10)))
    return out


def select_niche(
    library: NicheLibrary,
    cursor: int,
    *,
    tier: str,
    trend_period: int = DEFAULT_TREND_PERIOD,
    force_id: str | None = None,
) -> Niche:
    """Pick ONE niche for this run. ``force_id`` overrides selection.
    Otherwise: a trend niche every ``trend_period``-th run (cursor-based),
    else a weighted core niche; both pools filtered to those supporting
    ``tier``. Falls back across class/tier so a run never fails to pick."""
    if force_id:
        n = library.by_id(force_id)
        if tier not in n.tier_band:
            raise NicheLibraryError(
                f"niche {force_id!r} does not support tier {tier!r} "
                f"(supports {n.tier_band})"
            )
        return n

    def _supporting(pool: list[Niche]) -> list[Niche]:
        return [n for n in pool if tier in n.tier_band]

    core = _supporting(library.core())
    trend = _supporting(library.trend())
    trend_turn = trend_period > 0 and (cursor % trend_period == trend_period - 1)

    primary = trend if (trend_turn and trend) else core
    if not primary:
        # tier-compatible fallback: whichever pool has members, else any niche
        primary = core or trend or [n for n in library.niches if tier in n.tier_band]
    if not primary:
        raise NicheLibraryError(f"no niche supports tier {tier!r}")

    expanded = _weighted_expand(primary)
    return expanded[cursor % len(expanded)]


def select_niche_cycle(
    library: NicheLibrary,
    used_ids: list[str],
    *,
    tier: str,
) -> tuple[Niche, bool]:
    """Pick the next niche NOT yet used this cycle, so ``--auto`` exhausts
    EVERY tier-supporting niche before repeating any (unlike the cursor-modulo
    ``select_niche`` which over-samples high-weight niches and repeats early).

    Deterministic order: core first then trend, each by descending weight then
    id. Returns ``(niche, cycle_reset)`` — ``cycle_reset=True`` means every
    tier-supporting niche was already used, so this pick starts a fresh cycle
    (the caller should clear its used-set, keeping only this niche).
    """
    def _order(pool: list[Niche]) -> list[Niche]:
        return sorted((n for n in pool if tier in n.tier_band),
                      key=lambda n: (-n.weight, n.id))

    candidates = _order(library.core()) + _order(library.trend())
    if not candidates:  # no class membership — fall back to any tier match
        candidates = _order(library.niches)
    if not candidates:
        raise NicheLibraryError(f"no niche supports tier {tier!r}")

    used = set(used_ids)
    for n in candidates:
        if n.id not in used:
            return n, False
    return candidates[0], True  # all used → new cycle


def select_aesthetic_lock(niche: Niche, cursor: int) -> AestheticLock:
    """Lock one palette + one lighting recipe + one photographer for the
    series. MIXED-RADIX indexing (2026-06): the old constant-offset scheme
    moved all three equal-length axes in perfect lockstep — only 3 of 27
    combinations were ever reachable and two niches received the IDENTICAL
    full lock on consecutive runs. Dividing the cursor by each axis's radix
    walks every combination before repeating. Tolerates missing/empty lists."""
    div = 1
    picks: dict[str, str] = {}
    for key in ("palettes", "lighting", "photographers"):
        opts = niche.aesthetics.get(key) or []
        if opts:
            picks[key] = opts[(cursor // div) % len(opts)]
            div *= len(opts)
        else:
            picks[key] = ""

    return AestheticLock(
        palette=picks["palettes"],
        lighting=picks["lighting"],
        photographer=picks["photographers"],
    )


def select_persona(
    library: NicheLibrary,
    cursor: int,
    *,
    enabled: bool = False,
    name: str | None = None,
) -> Persona | None:
    """Return a persona when ``enabled`` (or ``name`` given). ``name``
    selects by exact name; otherwise a cursor-rotated pick from the pool."""
    if name:
        for p in library.personas:
            if p.name.lower() == name.lower():
                return p
        raise NicheLibraryError(
            f"unknown persona {name!r}; known: {[p.name for p in library.personas]}"
        )
    if not enabled or not library.personas:
        return None
    return library.personas[cursor % len(library.personas)]


def build_selection(
    library: NicheLibrary,
    cursor: int,
    *,
    tier: str,
    force_niche: str | None = None,
    persona: bool = False,
    persona_name: str | None = None,
    trend_period: int = DEFAULT_TREND_PERIOD,
) -> Selection:
    """Assemble the full per-run creative selection."""
    niche = select_niche(library, cursor, tier=tier,
                          trend_period=trend_period, force_id=force_niche)
    lock = select_aesthetic_lock(niche, cursor)
    who = select_persona(library, cursor, enabled=persona, name=persona_name)
    return Selection(
        niche=niche, tier=tier, aesthetic_lock=lock,
        sub_looks=list(niche.sub_looks), persona=who,
    )


def build_brief(selection: Selection) -> str:
    """Compose the single ``brief`` string handed to ``art_director``.

    Folds the per-series aesthetic-lock + optional persona into the brief so
    they thread into EVERY prompt of the series (the coherence anchor the
    LLM-direct path otherwise lacks). Sub-looks are passed separately."""
    parts: list[str] = [selection.niche.brief_seed.strip()]

    lock = selection.aesthetic_lock
    lock_bits = []
    if lock.palette:
        lock_bits.append(f"a {lock.palette} palette")
    if lock.lighting:
        lock_bits.append(lock.lighting)
    if lock.photographer:
        lock_bits.append(f"in the tradition of {lock.photographer}")
    if lock_bits:
        parts.append(
            "SERIES AESTHETIC LOCK (keep CONSISTENT across every image in the "
            "series): " + "; ".join(lock_bits) + "."
        )

    p = selection.persona
    if p:
        # The persona's APPEARANCE is delivered per image as the locked SUBJECT
        # LOOK (art_director uses persona_locked_look as look_target); the brief
        # only asserts the same-woman CONTRACT — re-listing hair/build here would
        # re-introduce the contradiction the lock exists to remove.
        parts.append(
            f"RECURRING SUBJECT — every image in this series depicts the SAME "
            f"named woman, {p.name} ({p.arc}); her appearance is held IDENTICAL "
            f"across the set (given per image as the SUBJECT LOOK). Vary only "
            f"scene, light, wardrobe and mood. Her wardrobe leans "
            f"{p.wardrobe_vibe} where it suits the scene."
        )

    # Per-niche avoidances (e.g. mirror-prone niches: vanities WITHOUT a
    # mirror). The model warps mirror reflections into double faces and the
    # validator hard-rejects them — naming the avoidance up front stops the
    # LLM reaching for the motif and churning through retries / dropped scenes.
    if selection.niche.avoid_motifs:
        parts.append(
            "DO NOT DEPICT (renders poorly / rejected — design around it): "
            + "; ".join(selection.niche.avoid_motifs) + "."
        )

    return " ".join(parts)

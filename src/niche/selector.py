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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Niche":
        try:
            return cls(
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NicheLibraryError(f"invalid niche entry {d!r}: {exc}") from exc


@dataclass(frozen=True)
class Persona:
    name: str
    hair: str
    build: str
    age_anchor: str
    wardrobe_vibe: str
    arc: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Persona":
        return cls(
            name=str(d.get("name", "")),
            hair=str(d.get("hair", "")),
            build=str(d.get("build", "")),
            age_anchor=str(d.get("age_anchor", "an adult woman")),
            wardrobe_vibe=str(d.get("wardrobe_vibe", "")),
            arc=str(d.get("arc", "")),
        )


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


def select_aesthetic_lock(niche: Niche, cursor: int) -> AestheticLock:
    """Lock one palette + one lighting recipe + one photographer for the
    series. Offsets stagger so the three axes don't move in lockstep across
    runs. Tolerates missing/empty aesthetic lists."""
    def _pick(key: str, offset: int) -> str:
        opts = niche.aesthetics.get(key) or []
        return opts[(cursor + offset) % len(opts)] if opts else ""

    return AestheticLock(
        palette=_pick("palettes", 0),
        lighting=_pick("lighting", 1),
        photographer=_pick("photographers", 2),
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
        parts.append(
            f"RECURRING SUBJECT — keep her identity consistent across the "
            f"series: {p.name}, {p.age_anchor}, {p.hair}, {p.build}, styled in "
            f"{p.wardrobe_vibe} ({p.arc}). Vary scene, light, wardrobe and mood "
            f"per image, but she is the same woman throughout."
        )

    return " ".join(parts)

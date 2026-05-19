"""Category loader — reads ``config/categories.yaml``.

Replaces 4 static SQLite tables (``theme_categories``,
``style_categories``, ``niche_clusters``, ``content_level_rules``).
These rows are immutable once seeded; runtime never writes them.

Methods return plain dict lists matching the column shapes the rest
of the codebase expects — minimal churn for callers.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "categories.yaml"
)


class CategoriesLoaderError(Exception):
    """Base for categories loader errors."""


class ContentLevelNotFound(CategoriesLoaderError):
    """Unknown content_level key."""


@dataclass(frozen=True)
class ThemeCategory:
    id: str
    name: str
    weight: float
    description: str = ""
    variations: list[str] = field(default_factory=list)
    aesthetic_affinity: list[str] = field(default_factory=list)
    suited_tiers: list[str] = field(default_factory=list)
    # Phase 3 (vocab v6) — series-level aesthetic-anchor compatibility
    # lists. Intersected with style_profile.compatible_* in ThemeMode
    # to narrow the SeriesPlanner LLM menu. Optional — empty list
    # means "no per-theme narrowing; fall through to style_profile".
    compatible_palettes: list[str] = field(default_factory=list)
    compatible_photographers: list[str] = field(default_factory=list)
    compatible_art_movements: list[str] = field(default_factory=list)
    compatible_environments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StyleCategory:
    id: str
    name: str
    weight: float
    description: str = ""
    lighting_bias: str = ""
    color_bias: str = ""
    aesthetic_affinity: list[str] = field(default_factory=list)
    # Phase 3 (vocab v6) — see ThemeCategory.
    compatible_palettes: list[str] = field(default_factory=list)
    compatible_photographers: list[str] = field(default_factory=list)
    compatible_art_movements: list[str] = field(default_factory=list)
    compatible_environments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NicheCluster:
    id: str
    name: str
    weight: float
    keywords: list[str] = field(default_factory=list)
    aesthetic_affinity: list[str] = field(default_factory=list)
    # Phase 3 (vocab v6) — see ThemeCategory.
    compatible_palettes: list[str] = field(default_factory=list)
    compatible_photographers: list[str] = field(default_factory=list)
    compatible_art_movements: list[str] = field(default_factory=list)
    compatible_environments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContentLevelRules:
    """Structured view of one row under ``content_levels:``.

    ``level`` is one of the 4 tier strings: ``T1_suggestive``,
    ``T2_implied``, ``T3_artnude``, ``T4_explicit``.
    """

    level: str
    allowed_pose_types: list[str]
    scene_constraints: dict[str, Any]
    prompt_boost_keywords: list[str]
    prompt_suppress_keywords: list[str]
    platform_allowlist: list[str] = field(default_factory=list)
    description: str = ""
    # llm_directive (Phase A) — tier-specific creative-direction text
    # injected into the SceneFacetGenerator system prompt. Empty when
    # the YAML row doesn't declare it; the facet generator skips
    # injection in that case (back-compat).
    llm_directive: str = ""


class CategoriesLoader:
    """Read-only accessor for ``config/categories.yaml``."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self._data = _load_categories(self.config_path)

    def theme_categories(self) -> list[ThemeCategory]:
        return [
            ThemeCategory(
                id=str(row["id"]),
                name=str(row["name"]),
                weight=float(row.get("weight", 1.0)),
                description=str(row.get("description") or ""),
                variations=list(row.get("variations") or []),
                aesthetic_affinity=list(row.get("aesthetic_affinity") or []),
                suited_tiers=list(row.get("suited_tiers") or []),
                # Phase 3 (vocab v6) — propagate compat lists into the
                # dataclass so ThemeMode's compat-intersection logic
                # actually sees them. Without these the YAML lists were
                # dead data (verifier B1).
                compatible_palettes=list(row.get("compatible_palettes") or []),
                compatible_photographers=list(row.get("compatible_photographers") or []),
                compatible_art_movements=list(row.get("compatible_art_movements") or []),
                compatible_environments=list(row.get("compatible_environments") or []),
            )
            for row in self._data.get("themes") or []
        ]

    def style_categories(self) -> list[StyleCategory]:
        return [
            StyleCategory(
                id=str(row["id"]),
                name=str(row["name"]),
                weight=float(row.get("weight", 1.0)),
                description=str(row.get("description") or ""),
                lighting_bias=str(row.get("lighting_bias") or ""),
                color_bias=str(row.get("color_bias") or ""),
                aesthetic_affinity=list(row.get("aesthetic_affinity") or []),
                compatible_palettes=list(row.get("compatible_palettes") or []),
                compatible_photographers=list(row.get("compatible_photographers") or []),
                compatible_art_movements=list(row.get("compatible_art_movements") or []),
                compatible_environments=list(row.get("compatible_environments") or []),
            )
            for row in self._data.get("styles") or []
        ]

    def niche_clusters(self) -> list[NicheCluster]:
        return [
            NicheCluster(
                id=str(row["id"]),
                name=str(row["name"]),
                weight=float(row.get("weight", 1.0)),
                keywords=list(row.get("keywords") or []),
                aesthetic_affinity=list(row.get("aesthetic_affinity") or []),
                compatible_palettes=list(row.get("compatible_palettes") or []),
                compatible_photographers=list(row.get("compatible_photographers") or []),
                compatible_art_movements=list(row.get("compatible_art_movements") or []),
                compatible_environments=list(row.get("compatible_environments") or []),
            )
            for row in self._data.get("niches") or []
        ]

    def content_level_rules(self, level: str) -> ContentLevelRules:
        levels = self._data.get("content_levels") or {}
        if level not in levels:
            raise ContentLevelNotFound(
                f"content_level {level!r} not in {self.config_path}. "
                f"Known: {sorted(levels)}"
            )
        row = levels[level]
        return ContentLevelRules(
            level=level,
            allowed_pose_types=list(row.get("allowed_pose_types") or []),
            scene_constraints=dict(row.get("scene_constraints") or {}),
            prompt_boost_keywords=list(row.get("prompt_boost_keywords") or []),
            prompt_suppress_keywords=list(row.get("prompt_suppress_keywords") or []),
            platform_allowlist=list(row.get("platform_allowlist") or []),
            description=str(row.get("description") or "").strip(),
            llm_directive=str(row.get("llm_directive") or "").strip(),
        )


@functools.lru_cache(maxsize=4)
def _load_categories(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise CategoriesLoaderError(f"categories.yaml not found at {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    for section in ("themes", "styles", "niches", "content_levels"):
        if section not in data:
            raise CategoriesLoaderError(
                f"{config_path}: missing top-level section `{section}:`"
            )
    return data

"""Style profile loader — reads ``config/style_profiles.yaml``.

A style profile is pure aesthetic intent — palette, lighting, lens
character, mood. It does NOT encode render tuning; sampler / scheduler
/ steps / cfg / clip_skip / lora_stack all live with the model in
``config/models/*.yaml`` and are resolved by
``StyleProfileForWorkflow`` from ``model.default_*``.

Characters reference a profile via ``characters.style_profile_id``;
the render path dereferences it here.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "config"
    / "style_profiles.yaml"
)


class StyleProfileLoaderError(Exception):
    """Base for style profile loader errors."""


class StyleProfileNotFound(StyleProfileLoaderError):
    """Lookup by id returned no profile."""


@dataclass(frozen=True)
class StyleProfile:
    """One aesthetic archetype from ``config/style_profiles.yaml``.

    Render tuning (sampler / steps / cfg / lora_stack / model_id) is
    intentionally absent — those come from the model YAML.

    Verifier round-3 BLOCKER fix — the 4 ``compatible_*`` lists are
    the style-profile side of the aesthetic-menu narrowing
    (Phase 3 vocab v6). Pre-fix this dataclass declared none of them
    and ``from_dict`` silently dropped the YAML rows; every mode read
    ``ctx.style_profile.get("compatible_palettes", [])`` as ``[]``
    and the SeriesPlanner's aesthetic-menu narrowing rendered inert.
    Now they round-trip from YAML through the loader, intersect with
    the category/cluster compat in each mode, and reach the LLM via
    ``_resolve_aesthetic_menu`` + ``llm_vocabulary_block``.
    """

    id: str
    name: str
    description: str
    base_style_keywords: str
    base_negative_prompt: str
    palette_hint: str
    lighting_hint: str
    suited_tiers: list[str] = field(default_factory=list)
    suited_families: list[str] = field(default_factory=list)
    compatible_palettes: list[str] = field(default_factory=list)
    compatible_photographers: list[str] = field(default_factory=list)
    compatible_art_movements: list[str] = field(default_factory=list)
    compatible_environments: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, profile_id: str, d: dict) -> "StyleProfile":
        required = {"base_style_keywords"}
        missing = required - d.keys()
        if missing:
            raise StyleProfileLoaderError(
                f"style profile {profile_id!r}: missing keys {sorted(missing)}"
            )
        return cls(
            id=profile_id,
            name=str(d.get("name") or profile_id),
            description=(d.get("description") or "").strip(),
            base_style_keywords=(d.get("base_style_keywords") or "").strip(),
            base_negative_prompt=(d.get("base_negative_prompt") or "").strip(),
            palette_hint=(d.get("palette_hint") or "").strip(),
            lighting_hint=(d.get("lighting_hint") or "").strip(),
            suited_tiers=list(d.get("suited_tiers") or []),
            suited_families=list(d.get("suited_families") or []),
            compatible_palettes=list(d.get("compatible_palettes") or []),
            compatible_photographers=list(
                d.get("compatible_photographers") or []
            ),
            compatible_art_movements=list(
                d.get("compatible_art_movements") or []
            ),
            compatible_environments=list(
                d.get("compatible_environments") or []
            ),
        )


class StyleProfileLoader:
    """Read-only accessor for ``config/style_profiles.yaml``."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self._profiles = _load_profiles(self.config_path)

    def get_profile(self, profile_id: str) -> StyleProfile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise StyleProfileNotFound(
                f"Style profile {profile_id!r} not found in "
                f"{self.config_path}. Known: {sorted(self._profiles)}"
            ) from None

    def list_profiles(self) -> list[StyleProfile]:
        return [self._profiles[k] for k in sorted(self._profiles)]

    def has_profile(self, profile_id: str) -> bool:
        return profile_id in self._profiles


@functools.lru_cache(maxsize=4)
def _load_profiles(config_path: Path) -> dict[str, StyleProfile]:
    if not config_path.exists():
        raise StyleProfileLoaderError(
            f"style_profiles.yaml not found at {config_path}"
        )
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("profiles")
    if not isinstance(raw, dict) or not raw:
        raise StyleProfileLoaderError(
            f"{config_path}: expected top-level `profiles:` mapping"
        )
    return {pid: StyleProfile.from_dict(pid, pd) for pid, pd in raw.items()}

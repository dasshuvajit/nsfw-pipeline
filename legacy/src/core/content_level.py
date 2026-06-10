"""Content-level rules — YAML-backed lookup over ``config/categories.yaml``.

ARCHITECTURE.md §5 defines four content tiers —
``T1_suggestive | T2_implied | T3_artnude | T4_explicit`` — each with
its own pose whitelist, scene constraints, platform allow-list, and
prompt boost/suppress keyword lists. These are immutable static
config, so they live in YAML (``config/categories.yaml`` →
``content_levels:``), not SQLite.

This module keeps the same public shape (``ContentLevelRules``,
``ContentLevelLoader``, ``UnknownContentLevel``) so callers don't have
to change — only the backing store did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.memory.categories_loader import (
    CategoriesLoader,
    CategoriesLoaderError,
    ContentLevelNotFound,
)


class ContentLevelError(Exception):
    """Base for content-level loader errors."""


class UnknownContentLevel(ContentLevelError):
    """Lookup by content_level returned no row."""


@dataclass(frozen=True)
class ContentLevelRules:
    """Parsed content-level row (YAML-sourced).

    The ``raw_*`` fields remain JSON-encoded strings so callers that
    used the legacy ``json.loads(rules.allowed_pose_types)`` snippet
    from ARCHITECTURE.md §5 keep working verbatim.
    """

    content_level: str
    allowed_pose_types: list[str]
    scene_constraints: dict[str, Any]
    prompt_boost_keywords: list[str]
    prompt_suppress_keywords: list[str]
    platform_allowlist: list[str] = field(default_factory=list)
    description: str | None = None
    # Phase A — tier-specific creative-direction text for the
    # SceneFacetGenerator's system prompt. Empty string when the YAML
    # row doesn't declare it; the facet generator skips injection.
    llm_directive: str = ""

    raw_allowed_pose_types: str = field(default="[]", repr=False)
    raw_scene_constraints: str = field(default="{}", repr=False)
    raw_prompt_boost_keywords: str = field(default="[]", repr=False)
    raw_prompt_suppress_keywords: str = field(default="[]", repr=False)


class ContentLevelLoader:
    """Read-only accessor for content-level rules.

    The ``db_path`` parameter is accepted (and ignored) so existing
    call sites (``ContentLevelLoader(db_path).load('T2_implied')``)
    keep working without churn — the data comes from YAML now.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        self._loader = CategoriesLoader(config_path)

    def load(self, content_level: str) -> ContentLevelRules:
        try:
            rules = self._loader.content_level_rules(content_level)
        except ContentLevelNotFound as exc:
            raise UnknownContentLevel(str(exc)) from exc
        except CategoriesLoaderError as exc:
            raise ContentLevelError(str(exc)) from exc

        return ContentLevelRules(
            content_level=rules.level,
            allowed_pose_types=rules.allowed_pose_types,
            scene_constraints=rules.scene_constraints,
            prompt_boost_keywords=rules.prompt_boost_keywords,
            prompt_suppress_keywords=rules.prompt_suppress_keywords,
            platform_allowlist=rules.platform_allowlist,
            description=rules.description or None,
            llm_directive=rules.llm_directive,
            raw_allowed_pose_types=json.dumps(rules.allowed_pose_types),
            raw_scene_constraints=json.dumps(rules.scene_constraints),
            raw_prompt_boost_keywords=json.dumps(rules.prompt_boost_keywords),
            raw_prompt_suppress_keywords=json.dumps(rules.prompt_suppress_keywords),
        )

    def list_levels(self) -> list[str]:
        # categories.yaml keys the dict by level name.
        data = self._loader._data.get("content_levels") or {}
        return sorted(data)

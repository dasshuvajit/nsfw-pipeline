"""Realism + NSFW vocabulary library — abstract concept → family phrasing.

The user-facing data lives in ``config/prompt_vocabulary.yaml``. This
module is the read-only accessor + canonicalizer:

* :class:`VocabularyLoader` parses + caches the YAML.
* :func:`canonicalize_facet` translates the abstract enum tags emitted
  by :class:`SceneFacetGenerator` into family-shaped phrases (the
  composer threads them into the final prompt).
* :func:`llm_vocabulary_block` builds the formatted system-prompt block
  that lists the abstract tags the LLM may emit for a given family.

Design constraints (per the prompt-quality plan):

* The LLM picks abstract tags from a small enumerated menu — it never
  invents realism vocabulary. The composer (not the LLM) is the
  single source of truth for family-specific phrasing.
* NSFW concepts are gated by ``tier_min:`` against the active
  ``content_level`` — the canonicalizer silently drops below-tier
  concepts so a T2_implied scene cannot leak T3_artnude vocabulary.
* Missing concepts are not an error — they're logged INFO and skipped.
  This keeps the pipeline forgiving of LLM drift (e.g. ``LIGHT_FOO``
  from a hallucinated tag) without crashing.

Three additive enum-tag fields land on every applicable SceneFacet
schema in Phase 4a (``realism_camera``, ``realism_lens``,
``realism_film_stock``, ``lighting_directive``, ``mood_aesthetic``,
``art_style_reference``). All are ``Optional[str]`` so existing
facets continue to validate.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "config" / "prompt_vocabulary.yaml"
)

# Tier ordering — must match content_levels.py. Concepts with
# ``tier_min: T3_artnude`` are dropped when active content_level is
# T2_implied or lower.
_TIER_ORDER: tuple[str, ...] = (
    "T1_suggestive",
    "T2_implied",
    "T3_artnude",
    "T4_explicit",
)

# Single-female-mode banned concept tags (2026-05-17). The user-facing
# constraint is "always exactly one adult female subject"; the
# partnered ``nsfw.act`` tags are by definition multi-subject and so
# cannot be picked under solo mode. The filter applies in two places:
#
#   * :meth:`VocabularyLoader.all_concepts_for_family` — drops banned
#     tags so they don't appear in the system-prompt menu the LLM sees
#     (Venice / Cydonia / Magnum can't pick what they're not shown).
#
#   * :meth:`VocabularyLoader.canonicalize` — defence-in-depth. If the
#     LLM picks a banned tag anyway (training-set bias), the
#     canonicalizer drops it with an ERROR log so drift surfaces at
#     facet-write time, not in the rendered image. The facet still
#     ships (the structured tag stays null) — the operator can re-prep.
#
# Future multi-subject mode lifts this by setting the constant to
# ``frozenset()``; no YAML schema change required.
_SOLO_MODE_BANNED_TAGS: frozenset[str] = frozenset({
    "NSFW_T4_EMBRACE_NUDE",
    "NSFW_T4_KISS_PASSIONATE",
    "NSFW_T4_PARTNERED_INTIMATE",
    "NSFW_T4_AFTERGLOW",
})


# Map SceneFacet enum-tag field names to their vocabulary namespace.
# Each pair is (schema_field, "<top>.<sub>"). When the LLM populates
# ``realism_camera="CAMERA_85MM_F14"``, the canonicalizer looks up
# ``realism.camera.CAMERA_85MM_F14`` for the family-shaped phrase.
_FIELD_TO_NAMESPACE: dict[str, tuple[str, str]] = {
    "realism_camera":      ("realism", "camera"),
    "realism_lens":        ("realism", "lens"),
    "realism_film_stock":  ("realism", "film_stock"),
    "lighting_directive":  ("realism", "lighting"),
    "mood_aesthetic":      ("realism", "mood"),
    "art_style_reference": ("realism", "art_style"),
    # Q10 (vocab v4) — composition vocab gap-fill (Pony omits, see
    # SceneFacetPony schema).
    "realism_angle":       ("realism", "angle"),
    "realism_framing":     ("realism", "framing"),
    # NSFW namespaces — canonicalizer keys reuse the same field names
    # but the concept tags are NSFW_*-prefixed and gated by tier_min.
    "nsfw_anatomy":  ("nsfw", "anatomy"),
    "nsfw_posture":  ("nsfw", "posture"),
    # Phase 4-bis (Phase B audit fix) — T4 explicit-act namespace.
    # Tier_min is T4_explicit; canonicalizer drops at T1/T2/T3.
    "nsfw_act":      ("nsfw", "act"),
}


class VocabularyError(Exception):
    """Bad shape or content in prompt_vocabulary.yaml."""


class VocabularyLoader:
    """Read-only accessor for the vocabulary library.

    Loaded once at import (cached), with O(1) lookup by
    ``(namespace, sub, concept, family)``. Mutating the YAML and
    re-instantiating gets a fresh load (the lru_cache is keyed by path).
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = (
            Path(config_path) if config_path else CONFIG_PATH
        )
        self._data = _load_vocabulary(self.config_path)

    @property
    def version(self) -> int:
        """Top-level version stamp; tracked in ``prompts.vocab_version``."""
        return int(self._data.get("version", 1))

    def concepts_by_namespace(
        self, top: str, sub: str,
    ) -> dict[str, dict[str, Any]]:
        """Return all concepts under e.g. ``realism.lighting``."""
        return self._data.get(top, {}).get(sub, {}) or {}

    def canonicalize(
        self,
        concept: str,
        family_id: str,
        *,
        content_level: str | None = None,
    ) -> str | None:
        """Translate one abstract concept tag to a family phrase.

        Returns ``None`` when:
        * The concept doesn't exist in any namespace (LLM drift).
        * The concept is NSFW-gated and the active ``content_level`` is
          below the concept's ``tier_min``.
        * The concept is in :data:`_SOLO_MODE_BANNED_TAGS` (single-
          female-mode partnered-act suppression). Logs at ERROR.
        * The concept exists but has no phrasing for ``family_id`` (e.g.
          Pony deliberately omits camera / lens / film_stock).

        Logs at INFO when dropping for tier-gate / family-omission;
        logs at ERROR when dropping for solo-mode suppression — any
        LLM drift onto partnered tags surfaces at facet-write time,
        not at render.
        """
        if not concept:
            return None
        # Solo-mode banned tags (partnered nsfw_act). Reject early
        # before walking namespaces so the ERROR log fires with the
        # exact tag the LLM tried to use.
        if concept in _SOLO_MODE_BANNED_TAGS:
            logger.error(
                "vocabulary: dropping solo-mode-banned partnered tag "
                "%r at content_level=%r — single-female enforcement; "
                "if multi-subject is required, lift "
                "_SOLO_MODE_BANNED_TAGS.",
                concept, content_level,
            )
            return None
        # Walk every namespace looking for the concept.
        for top, top_dict in self._data.items():
            if top in {"version"}:
                continue
            if not isinstance(top_dict, dict):
                continue
            for sub, sub_dict in top_dict.items():
                if not isinstance(sub_dict, dict):
                    continue
                row = sub_dict.get(concept)
                if not isinstance(row, dict):
                    continue
                # Match found — apply tier gating + family lookup.
                if top == "nsfw":
                    tier_min = row.get("tier_min")
                    if not _tier_meets(content_level, tier_min):
                        logger.info(
                            "vocabulary: dropping NSFW concept %r "
                            "(tier_min=%r) at content_level=%r",
                            concept, tier_min, content_level,
                        )
                        return None
                    phrasing = row.get("phrasing") or {}
                else:
                    phrasing = row
                phrase = phrasing.get(family_id)
                if not phrase:
                    logger.info(
                        "vocabulary: concept %r has no phrasing for "
                        "family %r — skipping",
                        concept, family_id,
                    )
                    return None
                return str(phrase).strip()
        # No match anywhere.
        logger.info(
            "vocabulary: unknown concept %r — skipping (LLM drift?)",
            concept,
        )
        return None

    def is_nsfw_concept(self, concept: str) -> bool:
        for sub_dict in (self._data.get("nsfw") or {}).values():
            if isinstance(sub_dict, dict) and concept in sub_dict:
                return True
        return False

    def tier_min_for(self, concept: str) -> str | None:
        for sub_dict in (self._data.get("nsfw") or {}).values():
            if isinstance(sub_dict, dict):
                row = sub_dict.get(concept)
                if isinstance(row, dict):
                    tier_min = row.get("tier_min")
                    if tier_min:
                        return str(tier_min)
        return None

    def all_concepts_for_family(self, family_id: str) -> dict[str, list[str]]:
        """List every concept that has phrasing for ``family_id``,
        keyed by ``"<top>.<sub>"`` namespace.

        Used by :func:`llm_vocabulary_block` to build the
        system-prompt menu the LLM may pick from. Tags in
        :data:`_SOLO_MODE_BANNED_TAGS` are silently omitted so Venice /
        Cydonia never see them as selectable options.

        Walks every top-level namespace in the YAML (realism / nsfw /
        environment / aesthetic / narrative / composition / ...).
        Concept-row shape determines phrasing lookup:

        * NSFW concepts wrap phrasing in a ``phrasing:`` sub-dict
          (alongside ``tier_min:``).
        * Every other namespace puts family keys directly on the
          concept row.

        Pre-Phase-0 this iteration was hardcoded to
        ``for top in ("realism", "nsfw"):`` — a residue from when those
        were the only two top-level namespaces. The hardcode made every
        new top-level namespace invisible to the LLM menu (verifier B1).
        """
        out: dict[str, list[str]] = {}
        for top, top_dict in self._data.items():
            if top == "version":
                continue  # version stamp, not a namespace
            if not isinstance(top_dict, dict):
                continue
            for sub, sub_dict in top_dict.items():
                if not isinstance(sub_dict, dict):
                    continue
                concepts = []
                for concept, row in sub_dict.items():
                    if not isinstance(row, dict):
                        continue
                    if concept in _SOLO_MODE_BANNED_TAGS:
                        continue  # solo-mode banned — hide from LLM menu
                    # Shape introspection: NSFW concepts have a
                    # ``phrasing:`` sub-dict; all other namespaces put
                    # family keys directly on the row. Phase 0 added
                    # this generic detection so new top-level
                    # namespaces ``just work`` without per-namespace
                    # special-casing here.
                    phrasing = row.get("phrasing") if "phrasing" in row else row
                    if isinstance(phrasing, dict) and family_id in phrasing:
                        concepts.append(concept)
                if concepts:
                    out[f"{top}.{sub}"] = sorted(concepts)
        return out


def canonicalize_facet(
    scene_facet: Mapping[str, Any],
    family_id: str,
    *,
    content_level: str | None = None,
    loader: VocabularyLoader | None = None,
) -> list[str]:
    """Translate every enum-tag field in ``scene_facet`` to a list of
    family-shaped phrases.

    Field order in the output preserves declaration order in
    :data:`_FIELD_TO_NAMESPACE` so the composer's downstream segment
    ordering stays stable. Empty / unknown / below-tier concepts are
    silently dropped (logged at INFO inside :meth:`VocabularyLoader.canonicalize`).
    """
    loader = loader or _default_loader()
    out: list[str] = []
    for field_name in _FIELD_TO_NAMESPACE:
        concept = scene_facet.get(field_name)
        if not concept:
            continue
        phrase = loader.canonicalize(
            str(concept), family_id, content_level=content_level,
        )
        if phrase:
            out.append(phrase)
    return out


def llm_vocabulary_block(
    family_id: str,
    *,
    content_level: str | None = None,
    loader: VocabularyLoader | None = None,
) -> str:
    """Build the system-prompt block listing the abstract tags the LLM
    may pick for ``family_id``.

    Placed verbatim into :class:`SceneFacetGenerator`'s system prompt
    so the LLM sees a small enumerated menu (~30 tags per family) and
    knows it doesn't have to invent realism vocabulary.

    Phase C — when ``content_level`` is supplied, the trailing
    instructional line becomes tier-aware:

    * T1 / T2 → "Pick at most one tag per namespace. Do NOT pick
      `nsfw_*` tags (T3+ only)."
    * T3 → "REQUIRED: pick exactly one `nsfw_anatomy` tag from the
      menu. Optionally pick a `nsfw_posture` tag."
    * T4 → "REQUIRED: pick exactly one `nsfw_anatomy` tag AND one
      `nsfw_act` tag. Optionally pick a `nsfw_posture` tag."

    ``content_level=None`` keeps the legacy generic line for callers
    that haven't migrated.
    """
    loader = loader or _default_loader()
    by_ns = loader.all_concepts_for_family(family_id)
    if not by_ns:
        return ""
    lines = [
        "REALISM VOCABULARY (abstract tags — composer translates to "
        "family-specific phrasing):",
    ]
    for ns, concepts in sorted(by_ns.items()):
        lines.append(f"  {ns}: {', '.join(concepts)}")
    lines.append(_tier_directive_line(content_level))
    return "\n".join(lines)


def _tier_directive_line(content_level: str | None) -> str:
    """Return the trailing instructional line of the vocabulary block,
    tier-aware. Phase C."""
    if content_level in ("T1_suggestive", "T2_implied"):
        return (
            "Pick exactly one tag per realism.* namespace that fits the "
            "scene. Do NOT pick any nsfw_* tag — those are gated to T3+ "
            "and the canonicalizer will drop them."
        )
    if content_level == "T3_artnude":
        return (
            "Pick exactly one tag per realism.* namespace that fits the "
            "scene. REQUIRED: pick exactly one `nsfw_anatomy` tag from "
            "the menu. Optionally pick a `nsfw_posture` tag when the "
            "pose calls for it. Do NOT pick `nsfw_act` tags (T4-only)."
        )
    if content_level == "T4_explicit":
        return (
            "Pick exactly one tag per realism.* namespace that fits the "
            "scene. REQUIRED: pick exactly one `nsfw_anatomy` tag AND "
            "exactly one `nsfw_act` tag from the menu. Optionally pick "
            "a `nsfw_posture` tag when the pose calls for it. The "
            "composer translates each tag into family-shaped phrasing."
        )
    # No content_level supplied (back-compat for direct callers).
    return (
        "Pick at most one tag per namespace. Setting any to null is "
        "fine; the composer will skip it. NSFW tags are tier-gated and "
        "auto-dropped below their min content_level."
    )


# ── helpers ─────────────────────────────────────────────────────────


def _tier_meets(active: str | None, min_tier: str | None) -> bool:
    """Is ``active`` content_level at or above ``min_tier``?

    A missing ``min_tier`` means "no gate". A missing ``active`` is
    treated as the lowest tier — concepts with any tier_min are
    rejected (defence-in-depth).
    """
    if not min_tier:
        return True
    if active not in _TIER_ORDER or min_tier not in _TIER_ORDER:
        return False
    return _TIER_ORDER.index(active) >= _TIER_ORDER.index(min_tier)


@functools.lru_cache(maxsize=4)
def _load_vocabulary(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise VocabularyError(
            f"prompt_vocabulary.yaml not found at {config_path}"
        )
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise VocabularyError(
            f"{config_path}: top-level must be a mapping"
        )
    if "version" not in data:
        raise VocabularyError(
            f"{config_path}: missing top-level `version:` field"
        )
    return data


@functools.lru_cache(maxsize=1)
def _default_loader() -> VocabularyLoader:
    return VocabularyLoader()

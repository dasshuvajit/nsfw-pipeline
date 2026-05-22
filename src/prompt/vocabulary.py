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
    # Phase 1 (vocab v6) — environment vocab gap-fill. All families
    # participate; categories whitelist compatible_environments per
    # theme to keep the LLM menu coherent.
    "environment_setting":    ("environment", "setting"),
    "environment_atmosphere": ("environment", "atmosphere"),
    # Phase 2 (vocab v6) — narrative_moment (the single highest-leverage
    # axis per market research). Tier-AGNOSTIC: one NARR_* per scene
    # at every tier (T1+). Anchors the editorial-moment that the
    # facet's pose/setting/lighting hang off — solves the "she just
    # poses" failure mode.
    "narrative_moment": ("narrative", "moment"),
    # Phase 4 (vocab v6) — env.prop + composition.principle polish.
    # Optional, not tier-required — LLM picks when they add value.
    "environment_prop":      ("environment", "prop"),
    "composition_principle": ("composition", "principle"),
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
            _record_drop("solo")
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
                        _record_drop("tier")
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
                    _record_drop("family")
                    return None
                return str(phrase).strip()
        # No match anywhere.
        logger.info(
            "vocabulary: unknown concept %r — skipping (LLM drift?)",
            concept,
        )
        _record_drop("unknown")
        return None

    def is_nsfw_concept(self, concept: str) -> bool:
        for sub_dict in (self._data.get("nsfw") or {}).values():
            if isinstance(sub_dict, dict) and concept in sub_dict:
                return True
        return False

    def get_place_constraint(
        self, top: str, sub: str, tag: str,
    ) -> dict | None:
        """Round-22 F15 (2026-05-22) — return the ``place_constraint``
        dict declared on a vocabulary tag, or ``None`` if the tag has
        none.

        ``place_constraint`` shape (only set on tags with hard
        environment dependencies, e.g. ATM_FABRIC_FLOATING_UNDERWATER
        or NARR_LETTER_BURNING_FIRE)::

            place_constraint:
              requires_env_keyword: ["pool", "underwater", "bath", ...]
              reason: "underwater fabric requires a water environment"

        The :func:`facet_env_coherent` helper consumes this to validate
        cross-field facet coherence — when the planner-side LLM picks
        an ``environment_setting`` whose family-shape prose contains
        NONE of the required env keywords, the facet is incoherent and
        the Pydantic validator on SceneFacetFluxNatural rejects → the
        retry-nudge fires.
        """
        if top == "version":
            return None
        top_dict = self._data.get(top)
        if not isinstance(top_dict, dict):
            return None
        sub_dict = top_dict.get(sub)
        if not isinstance(sub_dict, dict):
            return None
        row = sub_dict.get(tag)
        if not isinstance(row, dict):
            return None
        constraint = row.get("place_constraint")
        return constraint if isinstance(constraint, dict) else None

    def env_setting_prose(
        self, env_tag: str, family_id: str = "chroma",
    ) -> str:
        """Round-22 F15 — fetch the family-shape prose for an
        ``environment_setting`` tag. Used by the coherence validator to
        check whether an env's prose contains the keywords required by
        a paired atm / narr tag.

        Falls through to any family flavor if ``family_id`` doesn't
        have one (since the constraint check just looks for keywords
        in any prose flavor — they're synonyms across families).
        """
        env_block = (self._data.get("environment") or {}).get("setting") or {}
        row = env_block.get(env_tag)
        if not isinstance(row, dict):
            return ""
        # Prefer the requested family; fall through to chroma → flux →
        # sdxl → any.
        for fam in (family_id, "chroma", "flux", "sdxl", "flux2", "pony", "illustrious"):
            v = row.get(fam)
            if isinstance(v, str) and v:
                return v
        return ""

    def tag_exists(self, top: str, sub: str, tag: str) -> bool:
        """Round-22 F14 (2026-05-22) — check whether a concept tag
        exists in the specified vocabulary namespace.

        Used by the planner-side validation pass to catch
        hallucinated aesthetic anchors before they reach the
        canonicalizer (e.g. SeriesPlanner emitting
        ``PALETTE_DUTCH_GOLDEN_VERMEER`` when the namespace only has
        ``ART_MOVE_DUTCH_GOLDEN_VERMEER``). The pre-F14 schema
        validator checked only the ``PALETTE_*`` regex prefix —
        invalid full names slipped through.

        Returns True iff ``self._data[top][sub][tag]`` is a dict
        (the standard concept-row shape). Returns False for missing
        top / sub / tag, OR for the special ``version`` key.
        """
        if top == "version":
            return False
        top_dict = self._data.get(top)
        if not isinstance(top_dict, dict):
            return False
        sub_dict = top_dict.get(sub)
        if not isinstance(sub_dict, dict):
            return False
        return isinstance(sub_dict.get(tag), dict)

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


# Phase 3 (vocab v6) — series-level aesthetic-anchor fields. The
# SeriesPlanner picks these ONCE per series; the composer threads
# them into every scene's prompt so the set carries a coherent
# "signature look" (the commercial differentiator every top Patreon
# boudoir creator has). Distinct from _FIELD_TO_NAMESPACE which maps
# per-scene fields: this maps per-series fields.
_SERIES_FIELD_TO_NAMESPACE: dict[str, tuple[str, str]] = {
    "color_palette":    ("aesthetic", "color_palette"),
    "photographer_ref": ("aesthetic", "photographer_ref"),
    "art_movement":     ("aesthetic", "art_movement"),
}


def canonicalize_series_aesthetic(
    series_plan: Mapping[str, Any] | None,
    family_id: str,
    *,
    content_level: str | None = None,
    loader: VocabularyLoader | None = None,
) -> list[str]:
    """Translate the series-plan's aesthetic-anchor concept tags to
    family-shaped phrases that the composer threads into every scene.

    The aesthetic anchors are series-level inherited fields: chosen
    ONCE by SeriesPlanner and held constant across every scene in
    the set. Editorial-photography convention (Hegre, Newton, Crewdson)
    is to hold (palette + photographer + art_movement) constant
    across a series while varying location / props / narrative
    per-scene — that's where "signature look" comes from.

    Returns empty list when:
      * series_plan is None (back-compat for callers without a plan).
      * series_plan has no aesthetic fields populated (back-compat
        for older series predating Phase 3 — they continue to render
        without aesthetic enrichment, no error).
      * All concept lookups drop (Pony lookups for photographer_ref
        / art_movement are silently omitted — those namespaces have
        no Pony phrasing).

    Field order in the output preserves declaration order in
    :data:`_SERIES_FIELD_TO_NAMESPACE` so the composer's downstream
    segment ordering stays stable.
    """
    if series_plan is None:
        return []
    loader = loader or _default_loader()
    out: list[str] = []
    for field_name in _SERIES_FIELD_TO_NAMESPACE:
        concept = series_plan.get(field_name)
        if not concept:
            continue
        phrase = loader.canonicalize(
            str(concept), family_id, content_level=content_level,
        )
        if phrase:
            out.append(phrase)
    return out


# Round-22 F13 (2026-05-22) — per-series drop counter for observability.
# The canonicalizer silently drops unknown / tier-gated / family-omitted
# concept tags with an INFO log per drop. For long series the INFO
# stream is noisy; round-22 audit round-6 recommended an aggregate
# count surfaced at end of Phase A so the operator sees the planner-
# side drift rate at a glance ("3% of picks dropped — normal" vs
# "60% dropped — planner is hallucinating, investigate").
#
# Engine usage:
#   reset_drop_counter()   # at start of phase A
#   ... facet generation ...
#   counts = get_drop_counts()  # at end of phase A
#   logger.info("Vocab drops: unknown=%d tier=%d family=%d solo=%d",
#               counts["unknown"], counts["tier"], counts["family"],
#               counts["solo"])
_DROP_COUNTS: dict[str, int] = {
    "unknown": 0,  # concept not in any namespace (LLM drift)
    "tier": 0,     # NSFW tier-min above content_level
    "family": 0,   # concept exists but no phrasing for this family
    "solo": 0,     # solo-mode banned tag (partnered nsfw_act)
}


def reset_drop_counter() -> None:
    """Zero the per-series drop counter. Call once at start of Phase A."""
    for k in _DROP_COUNTS:
        _DROP_COUNTS[k] = 0


def get_drop_counts() -> dict[str, int]:
    """Return a copy of the per-series drop counter."""
    return dict(_DROP_COUNTS)


def _record_drop(kind: str) -> None:
    """Increment the per-kind drop counter. Internal — used by
    :meth:`VocabularyLoader.canonicalize`."""
    if kind in _DROP_COUNTS:
        _DROP_COUNTS[kind] += 1


# ── Pose ↔ nsfw_act / nsfw_posture geometric coherence ───────────────
#
# Defense-in-depth for COHERENCE INVARIANT clause 3. System-prompt
# instruction alone proved unreliable: a T4 verification series
# (series_c572709666a2) showed 4/5 reclining poses correctly paired
# with SOLO_RECLINING / SOLO_GAZE but scene_021 ignored the clause
# and picked SOLO_DISPLAY on a reclining body. This module-level
# check rejects the incoherent facet so the existing retry-with-
# nudge mechanism fires, same architecture as F15.
#
# Pose-class detection: keyword scan on the scene's pose string. The
# scene's pose is free-text from the SceneGenerator, not a structured
# tag, so we classify by substring matching. Add new keywords here as
# the SceneGenerator's pose vocabulary evolves.
_POSE_ORIENTATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "RECLINING": ("reclin", "lying", "lounging", "sprawled", "supine"),
    "STANDING": ("standing", "stand", "leaning",),
    "KNEELING": ("kneeling", "kneel", "on her knees"),
    "SEATED": ("sitting", "seated", "perched"),
    "ARCHED": ("arching", "arched back"),
}

# Per-orientation acts that the camera physically cannot see when the
# body is in that orientation. ARCHED back is left unconstrained
# because the spine arch is compatible with reclining + display +
# standing; the orientation is the supporting pose, not the arch.
_POSE_INCOMPATIBLE_ACTS: dict[str, frozenset[str]] = {
    "RECLINING": frozenset({
        "NSFW_T4_SOLO_DISPLAY",
        "NSFW_T4_SOLO_PERFORMER",
    }),
    "STANDING": frozenset({
        "NSFW_T4_SOLO_RECLINING",
        "NSFW_T4_AFTERGLOW",
    }),
    "KNEELING": frozenset({
        "NSFW_T4_SOLO_RECLINING",
        "NSFW_T4_AFTERGLOW",
    }),
    "SEATED": frozenset({
        "NSFW_T4_SOLO_RECLINING",
        "NSFW_T4_AFTERGLOW",
    }),
}

# Each pose orientation has exactly ONE matching nsfw_posture tag.
# Any other posture tag is by definition contradictory.
_POSE_MATCHING_POSTURE: dict[str, str] = {
    "RECLINING": "NSFW_RECLINED_NUDE",
    "STANDING": "NSFW_STANDING_NUDE",
    "KNEELING": "NSFW_KNEELING_NUDE",
    "SEATED": "NSFW_SEATED_NUDE",
    "ARCHED": "NSFW_ARCHED_BACK_NUDE",
}


def _classify_pose_orientation(pose: str | None) -> str | None:
    """Return the orientation class for ``pose`` (RECLINING / STANDING /
    KNEELING / SEATED / ARCHED) by keyword scan, or ``None`` when no
    keyword matches (ambiguous pose → skip the check)."""
    if not pose:
        return None
    pose_lower = pose.lower()
    # Order matters — RECLINING / KNEELING checked first so "leaning"
    # in STANDING doesn't match "kneeling with arms leaning over chair".
    # ARCHED checked before STANDING so "arched back standing" is
    # classified as ARCHED.
    for orientation in ("ARCHED", "RECLINING", "KNEELING", "SEATED", "STANDING"):
        keywords = _POSE_ORIENTATION_KEYWORDS[orientation]
        if any(kw in pose_lower for kw in keywords):
            return orientation
    return None


# Anatomy ↔ act direction conflicts. Back-facing anatomy (glutes,
# back view) cannot pair with front-presenting acts (SOLO_DISPLAY,
# full-frontal). Same physical-impossibility class as the pose-act
# checks above.
_BACK_FACING_ANATOMY: frozenset[str] = frozenset({
    "NSFW_GLUTES",
    "NSFW_BACK_VIEW_NUDE",
})

# Acts that imply a front-facing body presentation. When the
# nsfw_anatomy is back-facing, these acts are geometrically
# impossible (the camera can't see glutes AND a frontal display
# in one shot).
_FRONT_FACING_ACTS: frozenset[str] = frozenset({
    "NSFW_T4_SOLO_DISPLAY",
    "NSFW_T4_SOLO_PERFORMER",
})

# Bath-class nsfw_act tags — require an env with water-context
# keywords. ENV_FIRE_ESCAPE_NEON + SOLO_BATH = no bath. Same
# fail-mode as F15's env coherence, but specifically for the act
# field which isn't covered by place_constraint metadata.
_BATH_ACTS: frozenset[str] = frozenset({
    "NSFW_T4_SOLO_BATH",
})

_BATH_ENV_KEYWORDS: tuple[str, ...] = (
    "bath", "shower", "tub", "water", "hammam", "spa", "pool",
    "sauna", "steam",
)


def check_pose_act_coherence(
    *,
    pose: str | None,
    nsfw_act: str | None,
    nsfw_posture: str | None,
    nsfw_anatomy: str | None = None,
    environment_setting: str | None = None,
    loader: VocabularyLoader | None = None,
) -> list[tuple[str, str, str]]:
    """Validate that the LLM's ``nsfw_act`` + ``nsfw_posture`` +
    ``nsfw_anatomy`` picks are geometrically compatible with the
    scene's ``pose`` field AND the environment's prose.

    Returns a list of ``(field, tag, reason)`` tuples for each
    incoherent pair. Empty list = the facet's pose/act/posture/anatomy/
    env form a physically valid composition.

    Used by SceneFacetGenerator's retry loop to reject facets that
    violate COHERENCE INVARIANT clause 3 — same retry-with-nudge
    mechanism as the dominance check + F15.

    Five sub-checks:
      1. pose orientation ↔ nsfw_act compatibility (RECLINING +
         SOLO_DISPLAY = invalid).
      2. pose orientation ↔ nsfw_posture exact match (RECLINING
         pose requires NSFW_RECLINED_NUDE posture).
      3. nsfw_anatomy direction ↔ nsfw_act direction (GLUTES /
         BACK_VIEW + SOLO_DISPLAY = front+back contradiction).
      4. nsfw_act is a bath act ↔ env contains bath/water keyword
         (SOLO_BATH + courtyard env = no bath).
      5. partial-reveal anatomy ↔ full-display act (mutually
         exclusive coverage states).

    When ``pose`` doesn't match any known orientation keyword the
    pose-related checks are skipped (we don't know the body's
    orientation, so we can't judge the act). Same fail-open
    principle as F15. Anatomy/bath checks still fire regardless.
    """
    orientation = _classify_pose_orientation(pose)
    violations: list[tuple[str, str, str]] = []
    # 1 + 2 — pose orientation checks.
    if orientation is not None:
        # nsfw_act check — orientation-incompatible acts.
        if nsfw_act:
            incompatible = _POSE_INCOMPATIBLE_ACTS.get(orientation, frozenset())
            if str(nsfw_act).strip() in incompatible:
                violations.append((
                    "nsfw_act",
                    str(nsfw_act),
                    f"scene pose={pose!r} is {orientation}-oriented but "
                    f"nsfw_act={nsfw_act} implies a different body "
                    f"orientation — the camera physically cannot see this "
                    f"combination as one coherent shot. Pick an act compatible "
                    f"with {orientation} (e.g. NSFW_T4_SOLO_GAZE / "
                    f"NSFW_T4_SOLO_TOUCH for any orientation; "
                    f"NSFW_T4_SOLO_RECLINING for RECLINING; "
                    f"NSFW_T4_SOLO_DISPLAY / NSFW_T4_SOLO_PERFORMER for "
                    f"STANDING / KNEELING / SEATED)."
                ))
        # nsfw_posture check — exact-match enforcement.
        if nsfw_posture:
            expected = _POSE_MATCHING_POSTURE.get(orientation)
            if expected and str(nsfw_posture).strip() != expected:
                violations.append((
                    "nsfw_posture",
                    str(nsfw_posture),
                    f"scene pose={pose!r} is {orientation}-oriented; "
                    f"nsfw_posture must be {expected!r} to match. Got "
                    f"{nsfw_posture!r} which describes a different "
                    f"orientation entirely."
                ))
    # 3 — anatomy direction vs act direction.
    if nsfw_anatomy and nsfw_act:
        anatomy_tag = str(nsfw_anatomy).strip()
        act_tag = str(nsfw_act).strip()
        if anatomy_tag in _BACK_FACING_ANATOMY and act_tag in _FRONT_FACING_ACTS:
            violations.append((
                "nsfw_anatomy",
                anatomy_tag,
                f"nsfw_anatomy={anatomy_tag} is BACK-facing (glutes / "
                f"dorsal view) but nsfw_act={act_tag} is FRONT-presenting "
                f"(full-frontal display). The camera cannot see both back "
                f"and front in one shot. Pick anatomy compatible with the "
                f"act (e.g. NSFW_NIPPLES_VISIBLE / NSFW_FULL_FRONTAL / "
                f"NSFW_VULVA_VISIBLE for front acts; keep glutes/back for "
                f"SOLO_GAZE / SOLO_TOUCH / SOLO_RECLINING from behind)."
            ))
    # 4 — bath act ↔ env water-context check.
    if nsfw_act and environment_setting:
        act_tag = str(nsfw_act).strip()
        if act_tag in _BATH_ACTS:
            resolved_loader = loader or _default_loader()
            env_prose = resolved_loader.env_setting_prose(environment_setting).lower()
            if env_prose and not any(kw in env_prose for kw in _BATH_ENV_KEYWORDS):
                violations.append((
                    "nsfw_act",
                    act_tag,
                    f"nsfw_act={act_tag} is a bath-class act (requires "
                    f"water/tub/shower) but environment_setting={environment_setting!r} "
                    f"prose contains none of {list(_BATH_ENV_KEYWORDS)}. Re-pick "
                    f"a non-bath act (SOLO_GAZE / SOLO_TOUCH / SOLO_DISPLAY) "
                    f"or change environment_setting to ENV_CLAWFOOT_BATHROOM "
                    f"/ ENV_HAMMAM_STEAM / ENV_INDOOR_POOL_NIGHT."
                ))
    return violations


def check_facet_env_coherence(
    *,
    environment_setting: str | None,
    environment_atmosphere: str | None,
    narrative_moment: str | None,
    loader: VocabularyLoader | None = None,
) -> list[tuple[str, str, str]]:
    """Round-22 F15 (2026-05-22) — validate cross-field environmental
    coherence between ``environment_setting`` and the
    ``environment_atmosphere`` + ``narrative_moment`` picks.

    Some ATM / NARR tags declare a ``place_constraint`` in the
    vocabulary: a list of keywords at least one of which MUST appear in
    the chosen environment_setting's family-shape prose. If the
    environment_setting's prose contains NONE of those keywords, the
    paired tag is physically incoherent with the environment (e.g.
    ATM_FABRIC_FLOATING_UNDERWATER paired with ENV_FIRE_ESCAPE_NEON —
    underwater fabric in a Manhattan fire escape is not a real scene).

    Returns a list of ``(field, tag, reason)`` tuples for each
    incoherent pair. Empty list means the facet's three env-related
    tags pass coherence. The list is used by the Pydantic validator on
    SceneFacetFluxNatural to reject incoherent facets, triggering the
    existing retry-with-nudge mechanism in :mod:`src.agents.scene_facet_generator`.

    When ``environment_setting`` is None (LLM didn't pick one), the
    constraint check is skipped — atmosphere and narrative get the
    benefit of the doubt (they may still render coherently against the
    scene-core fields).
    """
    if not environment_setting:
        return []
    loader = loader or _default_loader()
    env_prose = loader.env_setting_prose(environment_setting).lower()
    if not env_prose:
        # Unknown environment_setting — the canonicalizer will catch
        # it elsewhere; nothing to validate against.
        return []
    violations: list[tuple[str, str, str]] = []
    for field, top, sub, tag in (
        ("environment_atmosphere", "environment", "atmosphere", environment_atmosphere),
        ("narrative_moment", "narrative", "moment", narrative_moment),
    ):
        if not tag:
            continue
        constraint = loader.get_place_constraint(top, sub, str(tag))
        if not constraint:
            continue
        required = constraint.get("requires_env_keyword") or []
        if not required:
            continue
        if any(kw.lower() in env_prose for kw in required):
            continue
        # No required keyword present in env's prose → incoherent pair.
        reason = (
            constraint.get("reason")
            or f"requires env containing one of {required}"
        )
        violations.append((
            field,
            str(tag),
            f"{reason} — but environment_setting={environment_setting!r} "
            f"has none of {required}",
        ))
    return violations


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

    Round-22 F13 — drops also increment :data:`_DROP_COUNTS` so the
    engine can surface the aggregate at end of Phase A.
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
    compatible_environments: Iterable[str] | None = None,
    compatible_narratives: Iterable[str] | None = None,
    compatible_art_styles: Iterable[str] | None = None,
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

    Verifier round-2 I4 — when ``compatible_environments`` is supplied
    (categories.yaml's ``compatible_environments`` per theme/style/
    niche, intersected upstream by the mode), the ``environment.setting``
    line of the menu is narrowed to that whitelist. Tags absent from
    the family's full menu are dropped; an empty intersection falls
    back to the full menu (defence-in-depth: never serve the LLM a
    blank ``environment.setting`` line — that yields garbage retries).

    Round-12 (2026-05-21) — ``compatible_narratives`` applies the same
    pattern to the ``narrative.moment`` namespace. The 2026-05-20 A/B
    audit showed both Cydonia and Qwen3 picking NARR_AFTER_THE_PARTY
    (silk dress + champagne) for a Gothic Chapel series and
    NARR_STEPPING_FROM_BATH (clawfoot bath) for 14/24 industrial-loft
    scenes — the global narrative menu has no theme-coherence filter,
    so the LLM picks pop-culture tags that visually contradict the
    setting. Narrowing the menu by category gives the LLM only
    theme-coherent narrative choices.
    """
    loader = loader or _default_loader()
    by_ns = loader.all_concepts_for_family(family_id)
    if not by_ns:
        return ""
    # Narrow environment.setting to the compat whitelist if supplied
    # and non-empty. Intersection happens in-place on the by_ns dict
    # so the sort+render loop below picks up the narrowed list.
    if compatible_environments:
        whitelist = {str(t).strip() for t in compatible_environments if t}
        env_key = "environment.setting"
        if whitelist and env_key in by_ns:
            narrowed = [t for t in by_ns[env_key] if t in whitelist]
            # Empty intersection means the LLM is offered nothing —
            # worse than showing the full menu. Fall back silently.
            if narrowed:
                by_ns[env_key] = narrowed
    # Same pattern for narrative.moment (round-12). The category's
    # compatible_narratives list comes from categories.yaml; intersect
    # with the family's narrative menu and fall back to full if empty.
    if compatible_narratives:
        narr_whitelist = {str(t).strip() for t in compatible_narratives if t}
        narr_key = "narrative.moment"
        if narr_whitelist and narr_key in by_ns:
            narrowed = [t for t in by_ns[narr_key] if t in narr_whitelist]
            if narrowed:
                by_ns[narr_key] = narrowed
    # 2026-05-23 — same pattern for realism.art_style. Verifier audit
    # of series_79ae3b962c8d found Lindbergh-anchored series picking
    # ART_HELMUT_NEWTON (opposite school) on 5/28 scenes. Style-
    # profile-driven compat list constrains per-scene art_style_
    # reference to schools coherent with the series anchors.
    if compatible_art_styles:
        art_whitelist = {str(t).strip() for t in compatible_art_styles if t}
        art_key = "realism.art_style"
        if art_whitelist and art_key in by_ns:
            narrowed = [t for t in by_ns[art_key] if t in art_whitelist]
            if narrowed:
                by_ns[art_key] = narrowed
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

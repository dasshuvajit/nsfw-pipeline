"""Series-level aesthetic-anchor menu helper.

Relocated from `src/agents/series_planner.py` (deleted 2026-05-20 with
the character-mode cleanup). The helper is consumed by theme / style /
niche mode planners to build the per-namespace tag menu offered to the
LLM for color_palette / photographer_ref / art_movement selection.

Phase 3 vocab v6 narrows the menu via the active style_profile's
`compatible_*` lists when present; otherwise the full vocab namespace
is offered. Pony omits photographer_ref / art_movement phrasings — but
the planner still picks a tag (which canonicalizes for sibling
SDXL/Flux/Chroma renders); Pony composer drops via the existing
family-omission canonicalizer path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.model_registry import ModelPromptGuide


def _resolve_aesthetic_menu(
    prompt_guide: "ModelPromptGuide | None" = None,
    style_profile_compat: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Return the per-namespace tag list the planner offers to the LLM
    for aesthetic-anchor selection.

    Returns a dict with keys ``color_palette`` / ``photographer_ref``
    / ``art_movement``, each valued by the tag-id list the LLM may
    pick from. When ``style_profile_compat`` provides a non-empty
    ``compatible_palettes`` / ``compatible_photographers`` /
    ``compatible_art_movements`` list, the menu narrows to the
    intersection of (full vocab) ∩ (compat). Otherwise the full
    menu is returned for that namespace.

    Pony omission: photographer_ref + art_movement namespaces have
    no Pony phrasing, so when the planner is invoked for a Pony-only
    target the LLM still picks a tag (which canonicalizes to phrasing
    for SDXL/Flux/Chroma siblings). The Pony composer silently drops
    those at compose time via the existing canonicalizer family-
    omission path. Behaviour is identical to how Pony has always
    handled realism.camera / realism.lens (vocab v4).
    """
    from src.prompt.vocabulary import _default_loader
    loader = _default_loader()
    # Use SDXL as the canonical menu source — every non-Pony-omitted
    # namespace has SDXL phrasing, so SDXL is the union of all tags.
    full_menu = loader.all_concepts_for_family("sdxl")
    palettes = list(full_menu.get("aesthetic.color_palette", []))
    photographers = list(full_menu.get("aesthetic.photographer_ref", []))
    art_movements = list(full_menu.get("aesthetic.art_movement", []))

    # Apply compat-list narrowing when provided + non-empty.
    if style_profile_compat:
        cp = style_profile_compat.get("compatible_palettes") or []
        cph = style_profile_compat.get("compatible_photographers") or []
        cam = style_profile_compat.get("compatible_art_movements") or []
        if cp:
            palettes = [t for t in palettes if t in cp]
        if cph:
            photographers = [t for t in photographers if t in cph]
        if cam:
            art_movements = [t for t in art_movements if t in cam]
        # Defensive: if a filter zeroed a namespace (compat list had
        # stale tag names), fall back to the full menu for that namespace
        # rather than offering an empty menu to the LLM.
        if not palettes:
            palettes = list(full_menu.get("aesthetic.color_palette", []))
        if not photographers:
            photographers = list(full_menu.get("aesthetic.photographer_ref", []))
        if not art_movements:
            art_movements = list(full_menu.get("aesthetic.art_movement", []))

    return {
        "color_palette": palettes,
        "photographer_ref": photographers,
        "art_movement": art_movements,
    }

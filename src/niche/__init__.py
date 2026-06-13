"""Niche intelligence — the "what to make" layer for the LLM-direct pipeline.

Loads ``config/niche_library.yaml`` and turns a deterministic per-run
cursor into a concrete creative selection (one niche + optional persona +
a per-series aesthetic-lock) for ``scripts/art_director.py``.
"""

from src.niche.selector import (
    AestheticLock,
    Niche,
    NicheLibrary,
    NicheLibraryError,
    Persona,
    Selection,
    build_brief,
    build_selection,
    persona_locked_look,
)

__all__ = [
    "AestheticLock",
    "Niche",
    "NicheLibrary",
    "NicheLibraryError",
    "Persona",
    "Selection",
    "build_brief",
    "build_selection",
    "persona_locked_look",
]

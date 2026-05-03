"""Negative-axis taxonomy — split family negatives into 7 buckets.

Pre-Phase-D, every family's negatives lived as one flat comma string
under ``negative_prompt:`` in ``config/families.yaml``. That collapses
all signal into a single bag of words: a per-model YAML can't say
"keep ``bad anatomy`` but drop ``cartoon``" without rewriting the
whole string.

Phase D restructures negatives into 7 axes::

    negative_axes:
      anatomy:    [bad anatomy, extra limbs, deformed hands, ...]
      medium:     [cartoon, painting, ...]
      skin:       [plastic skin, oversharpened, ...]
      quality:    [low quality, jpeg artifacts, blurry, ...]
      watermark:  [watermark, text, logo, signature, ...]
      safety:     []                  # HARD_BLOCK_NEGATIVE owns this axis
      censor:     [censored, mosaic, bar censor]

The composer flattens axes back into the comma-string the encoder
expects — but each axis can now be ``extend``ed or ``override``d
independently from a per-model YAML.

This module is pure (no I/O, no logging side-effects) so the YAML
loader, merge_overrides, and the prompt builder can all share its
small primitives.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping


# Canonical axis order — flatten always emits in this order so the
# composed string is deterministic across runs / processes.
AXIS_KEYS: tuple[str, ...] = (
    "anatomy",
    "medium",
    "skin",
    "quality",
    "watermark",
    "safety",
    "censor",
)


def empty_axes() -> dict[str, list[str]]:
    """A fresh dict with every canonical axis present and empty."""
    return {axis: [] for axis in AXIS_KEYS}


def normalize_axes(raw: Mapping[str, Iterable[str]] | None) -> dict[str, list[str]]:
    """Coerce a YAML-shaped axes mapping into the canonical 7-axis dict.

    Unknown axis keys raise — typo protection. Missing axes default to
    an empty list. Token strings are stripped; blank tokens dropped.
    """
    out = empty_axes()
    if not raw:
        return out
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"negative_axes must be a mapping, got {type(raw).__name__}"
        )
    unknown = set(raw) - set(AXIS_KEYS)
    if unknown:
        raise ValueError(
            f"unknown negative_axes keys: {sorted(unknown)}. "
            f"Allowed: {list(AXIS_KEYS)}"
        )
    for axis in AXIS_KEYS:
        tokens = raw.get(axis) or []
        if isinstance(tokens, str):
            # Allow a single string per axis as a YAML convenience —
            # split on commas and strip.
            tokens = [t.strip() for t in tokens.split(",")]
        if not isinstance(tokens, (list, tuple)):
            raise ValueError(
                f"negative_axes.{axis} must be a list, got {type(tokens).__name__}"
            )
        out[axis] = [str(t).strip() for t in tokens if t and str(t).strip()]
    return out


def flatten_axes(axes: Mapping[str, Iterable[str]] | None) -> str:
    """Comma-join every axis's tokens in canonical axis order.

    Within each axis, token order is preserved. Across axes, the order
    is fixed by ``AXIS_KEYS``. Duplicate tokens (case-insensitive,
    across axes) are collapsed — the first occurrence wins. Empty axes
    contribute nothing.
    """
    if not axes:
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for axis in AXIS_KEYS:
        for tok in axes.get(axis, ()) or ():
            t = str(tok).strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return ", ".join(out)


def merge_axes(
    base: Mapping[str, Iterable[str]] | None,
    extend: Mapping[str, Iterable[str]] | None = None,
    override: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, list[str]]:
    """Apply ``extend:`` (per-axis append) and ``override:`` (per-axis
    replace) to a base axes dict.

    Both ``extend`` and ``override`` are partial — only the axes they
    name are touched. Per-axis ``override`` to ``[]`` empties an axis
    deliberately; an axis missing from both blocks keeps its base
    value.

    Conflict policy: an axis named in *both* extend and override on the
    same call is rejected by ``merge_overrides`` upstream; this helper
    blindly applies override last so per-axis override wins if both
    arrive (defensive — should never happen in practice).
    """
    out = normalize_axes(base)
    if extend:
        if not isinstance(extend, Mapping):
            raise ValueError("negative_axes extend must be a mapping")
        unknown = set(extend) - set(AXIS_KEYS)
        if unknown:
            raise ValueError(
                f"unknown negative_axes extend keys: {sorted(unknown)}"
            )
        for axis, tokens in extend.items():
            if isinstance(tokens, str):
                tokens = [t.strip() for t in tokens.split(",")]
            if not isinstance(tokens, (list, tuple)):
                raise ValueError(
                    f"extend.negative_axes.{axis} must be a list"
                )
            out[axis] = list(out[axis]) + [
                str(t).strip() for t in tokens if t and str(t).strip()
            ]
    if override:
        if not isinstance(override, Mapping):
            raise ValueError("negative_axes override must be a mapping")
        unknown = set(override) - set(AXIS_KEYS)
        if unknown:
            raise ValueError(
                f"unknown negative_axes override keys: {sorted(unknown)}"
            )
        for axis, tokens in override.items():
            if isinstance(tokens, str):
                tokens = [t.strip() for t in tokens.split(",")]
            if not isinstance(tokens, (list, tuple)):
                raise ValueError(
                    f"override.negative_axes.{axis} must be a list"
                )
            out[axis] = [
                str(t).strip() for t in tokens if t and str(t).strip()
            ]
    return out


# Token-comparison normalizer for conflict filtering. We strip
# weighting parentheses / colon-weight, lowercase, and collapse
# whitespace so ``(blurry:1.2)`` matches ``blurry`` and ``Bad Anatomy``
# matches ``bad anatomy``. Multi-word tokens are kept as-is for whole-
# phrase substring matching.
_WEIGHT_PARENS = re.compile(r"[()]")
_WEIGHT_TAIL = re.compile(r":\d+(?:\.\d+)?$")


def _normalize_token(text: str) -> str:
    cleaned = _WEIGHT_PARENS.sub("", text or "").strip()
    cleaned = _WEIGHT_TAIL.sub("", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).lower()
    return cleaned


def filter_conflicts(
    axes: Mapping[str, Iterable[str]] | None,
    conflict_terms: Iterable[str] | None,
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Drop tokens that collide with positive-side / hard-block vocab.

    Returns ``(filtered_axes, dropped)`` where ``dropped`` is a list of
    ``(axis, token)`` pairs the caller can log at WARNING. A negative
    token *collides* when its normalized form appears as a whole-word
    substring of any normalized conflict term — the goal is to catch
    classic foot-guns like ``model_negative: "naked"`` colliding with
    ``positive: "nude pose"`` (both share the ``naked`` concept once
    we lower-case + strip weights).

    ``conflict_terms`` is typically: ``HARD_BLOCK_NEGATIVE`` (pre-split
    by comma), the assembled positive prompt, and the character's
    ``base_prompt``. We don't try to be clever about substring vs.
    whole-word matching beyond a token-boundary regex — the cost of a
    false positive (one rare token dropped) is way lower than the cost
    of a false negative (the encoder generating an age-ambiguous body).
    """
    out = empty_axes()
    dropped: list[tuple[str, str]] = []
    base = normalize_axes(axes)
    if not conflict_terms:
        return base, dropped

    # Build the conflict set once. We tokenize each conflict term on
    # commas and word boundaries so phrases like "blurry, low-res"
    # contribute both "blurry" and "low-res".
    conflict_set: set[str] = set()
    for term in conflict_terms:
        if not term:
            continue
        for chunk in re.split(r"[,\n]", str(term)):
            norm = _normalize_token(chunk)
            if norm:
                conflict_set.add(norm)
                # Also seed individual words so a multi-word negative
                # like "extra limbs" gets caught when the positive
                # prompt mentions just "limbs". This is intentionally
                # aggressive — see the docstring.
                for word in re.findall(r"[a-z0-9_]{3,}", norm):
                    conflict_set.add(word)

    if not conflict_set:
        return base, dropped

    for axis in AXIS_KEYS:
        kept: list[str] = []
        for tok in base[axis]:
            norm = _normalize_token(tok)
            if not norm:
                continue
            if norm in conflict_set:
                dropped.append((axis, tok))
                continue
            # Whole-word check — conflict term contains the negative
            # token as a standalone word.
            words = set(re.findall(r"[a-z0-9_]{3,}", norm))
            if words and words.issubset(conflict_set):
                dropped.append((axis, tok))
                continue
            kept.append(tok)
        out[axis] = kept
    return out, dropped

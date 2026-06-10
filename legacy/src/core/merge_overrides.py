"""Merge family defaults with per-model extend/override blocks.

Per-model YAMLs may declare::

    prompt:
      extend:
        trigger_words: [...]           # appended to family's list/str
        negative_prompt: "..."         # ", "-joined onto family negative
        negative_embeddings: [...]     # textual-inversion embedding tokens
        avoid_words: [...]             # appended to family list
      override:
        max_tokens: 60                 # replaces family scalar outright
        example_prompt: "..."          # replaces family example
        supports_negative_prompt: ...  # replaces family capability flag

``extend:`` is list/str concatenation; ``override:`` is wholesale
replacement. Setting the same key in both blocks is an explicit error —
we refuse to guess which intent the author meant.

``negative_embeddings:`` accepts three input shapes::

    - epiCNegative                  # bare name → "embedding:epiCNegative"
    - CyberRealistic_Negative:0.8   # name:weight → "embedding:Foo:0.8"
    - "embedding:BadDream"          # canonical pass-through

All three are normalised to ``embedding:Name[:weight]`` at merge time
so downstream assembler code receives one canonical shape. ``embedding:``
tokens are NOT permitted inside ``negative_prompt:`` — they belong in
``negative_embeddings:`` so the assembler can hoist them to the front
of the negative prompt; mixing them into the keyword string defeats
the position-weighting and breaks identity dedup.

This helper is pure — no YAML I/O. Callers pass already-parsed
``FamilyConfig`` + raw per-model ``prompt:`` dict.
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from typing import Any

from src.memory.family_loader import FamilyConfig
from src.prompt.negative_axes import flatten_axes, merge_axes, normalize_axes


# Matches ``embedding:Name`` or ``embedding:Name:0.8`` anywhere in a
# string. Used to detect TI tokens that have been (mis)placed into a
# ``negative_prompt:`` field; mirrors the regex in ``builder.py``.
_TI_TOKEN_RE = re.compile(
    r"\bembedding:[A-Za-z0-9_\-]+(?::\d+(?:\.\d+)?)?",
    re.IGNORECASE,
)

# Match a ``name`` or ``name:weight`` form for the bare-shape sugar
# accepted by ``negative_embeddings:``. Names may contain letters,
# digits, hyphens, underscores. Weight is a positive decimal.
_BARE_TI_RE = re.compile(
    r"^([A-Za-z0-9_\-]+)(?::(\d+(?:\.\d+)?))?$"
)


class MergeOverrideError(Exception):
    """Raised on conflicting extend/override keys or bad shape."""


# Keys that extend:/override: blocks may mention. Anything else is a typo.
_KNOWN_KEYS = {
    "trigger_words",
    "negative_prompt",
    "negative_axes",
    "negative_embeddings",
    "avoid_words",
    "quality_prefix",
    "quality_suffix",
    "structure_intro",
    "max_tokens",
    "clip_skip",
    "structure_rules",
    "llm_hint",
    "example_prompt",
    "supports_negative_prompt",
    "supports_weighting",
    "prompt_style",
    "adult_anchor",
}

# Keys safe to extend (list-appendable, string-concatenable, or
# axis-mergeable). ``negative_axes`` is a dict[str, list[str]]; extend
# means per-axis append, override means per-axis replace — both handled
# by ``merge_axes`` below.
_EXTENDABLE = {
    "trigger_words",
    "avoid_words",
    "quality_prefix",
    "quality_suffix",
    "structure_intro",
    "negative_prompt",
    "negative_axes",
    "negative_embeddings",
}


def merge_prompt_config(
    family: FamilyConfig, prompt_block: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a merged dict usable by PromptBuilder.

    Result keys match ``FamilyConfig`` fields plus ``trigger_words``
    (which lives on the per-model side only — the family has no
    trigger_words field). The output is a plain dict (not a dataclass)
    so callers can tweak it before handing to the builder if needed.
    """
    base = asdict(family)
    base.setdefault("trigger_words", [])

    if not prompt_block:
        return base

    extend = prompt_block.get("extend") or {}
    override = prompt_block.get("override") or {}

    if not isinstance(extend, dict) or not isinstance(override, dict):
        raise MergeOverrideError(
            "prompt.extend and prompt.override must be mappings"
        )

    _validate_keys(extend, "extend")
    _validate_keys(override, "override")

    conflicts = extend.keys() & override.keys()
    if conflicts:
        raise MergeOverrideError(
            f"keys appear in both extend: and override:: {sorted(conflicts)}"
        )

    # The legacy flat ``negative_prompt`` and the Phase-D ``negative_axes``
    # both feed the same downstream string. Letting both coexist in the
    # same prompt block silently loses the legacy extend (axes re-flatten
    # overwrites it). Refuse — pick one shape per model YAML.
    legacy_keys = {"negative_prompt"}
    axes_keys = {"negative_axes"}
    legacy_seen = (extend.keys() | override.keys()) & legacy_keys
    axes_seen = (extend.keys() | override.keys()) & axes_keys
    if legacy_seen and axes_seen:
        raise MergeOverrideError(
            "prompt block uses both `negative_prompt` (legacy flat string) "
            "and `negative_axes` (taxonomy mapping). Pick one shape — the "
            "axes form re-flattens and would silently lose the legacy "
            "extend. Migrate to negative_axes."
        )

    # ``embedding:`` tokens belong in ``negative_embeddings:`` only — never
    # in the keyword string. Caught at merge time so the user gets a clear
    # error pointing them to the right field instead of silently breaking
    # the position-weighting at assembler time.
    for source, block in (("extend", extend), ("override", override)):
        neg_str = block.get("negative_prompt")
        if isinstance(neg_str, str) and _TI_TOKEN_RE.search(neg_str):
            raise MergeOverrideError(
                f"prompt.{source}.negative_prompt contains `embedding:` "
                f"token(s); textual-inversion embeddings must be declared "
                f"in `prompt.extend.negative_embeddings:` so the assembler "
                f"hoists them to the front of the negative prompt. Move the "
                f"token(s) to a list under negative_embeddings."
            )

    for key, val in extend.items():
        if key not in _EXTENDABLE:
            raise MergeOverrideError(
                f"extend: key {key!r} is not list/str-extendable; "
                f"use override: instead"
            )
        if key == "negative_axes":
            base[key] = merge_axes(base.get(key) or {}, extend=val)
            continue
        if key == "negative_embeddings":
            base[key] = (base.get(key) or []) + _normalize_embedding_list(val)
            continue
        base[key] = _extend_value(base.get(key), val)

    for key, val in override.items():
        if key == "negative_axes":
            base[key] = merge_axes(base.get(key) or {}, override=val)
            continue
        if key == "negative_embeddings":
            base[key] = _normalize_embedding_list(val)
            continue
        base[key] = val

    # Re-flatten ``negative_prompt`` whenever axes changed so downstream
    # callers that read the flat string get the merged view, not the
    # stale family-level baseline.
    if "negative_axes" in extend or "negative_axes" in override:
        base["negative_prompt"] = flatten_axes(base["negative_axes"])
    elif "negative_axes" in base and base.get("negative_axes"):
        # Defensive: if the family already has axes and the per-model
        # block didn't touch them, the flat string still reflects them
        # (FamilyConfig.from_dict already did this) — no action needed.
        pass

    return base


def _validate_keys(block: dict, name: str) -> None:
    unknown = set(block) - _KNOWN_KEYS
    if unknown:
        raise MergeOverrideError(
            f"prompt.{name} has unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_KNOWN_KEYS)}"
        )


def _normalize_embedding_list(raw: Any) -> list[str]:
    """Normalise the three accepted ``negative_embeddings:`` shapes
    into canonical ``embedding:Name[:weight]`` form.

    Inputs accepted (in any combination within the same list):

    * ``epiCNegative``                — bare name → ``embedding:epiCNegative``
    * ``CyberRealistic_Neg_SDXL:0.8`` — name:weight → ``embedding:Foo:0.8``
    * ``embedding:BadDream``          — full form pass-through
    * ``embedding:BadDream:0.8``      — full form with weight pass-through

    Empty / blank entries are silently dropped (callers don't have to
    pre-filter). Anything that does not match either sugar shape OR the
    canonical prefix raises :class:`MergeOverrideError`.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MergeOverrideError(
            f"negative_embeddings must be a list of strings, got "
            f"{type(raw).__name__}"
        )
    out: list[str] = []
    for tok in raw:
        if tok is None:
            continue
        if not isinstance(tok, str):
            raise MergeOverrideError(
                f"negative_embeddings entries must be strings, got "
                f"{type(tok).__name__}: {tok!r}"
            )
        clean = tok.strip().rstrip(",").strip()
        if not clean:
            continue
        if clean.lower().startswith("embedding:"):
            # Canonical form pass-through; preserve the user's casing on
            # the name itself but normalise the prefix to lowercase since
            # ComfyUI accepts either case.
            body = clean[len("embedding:"):]
            if not body:
                raise MergeOverrideError(
                    f"negative_embeddings token {tok!r} has empty body "
                    f"after `embedding:` prefix"
                )
            out.append(f"embedding:{body}")
            continue
        # Bare or name:weight sugar.
        m = _BARE_TI_RE.match(clean)
        if not m:
            raise MergeOverrideError(
                f"negative_embeddings token {tok!r} is not a valid "
                f"embedding name. Expected: bare 'epiCNegative', "
                f"'name:0.8' (with weight), or 'embedding:Name[:weight]' "
                f"(canonical form)."
            )
        name, weight = m.group(1), m.group(2)
        if weight is not None:
            out.append(f"embedding:{name}:{weight}")
        else:
            out.append(f"embedding:{name}")
    return out


def _extend_value(base: Any, extra: Any) -> Any:
    if base is None or base == "":
        base_is_empty = True
    else:
        base_is_empty = False

    if isinstance(extra, list):
        return list(base or []) + list(extra)
    if isinstance(extra, str):
        if not extra.strip():
            return base or ""
        if base_is_empty:
            return extra.strip()
        return f"{base}, {extra.strip()}"
    raise MergeOverrideError(
        f"cannot extend value of type {type(extra).__name__}"
    )


def family_with_overrides(
    family: FamilyConfig, prompt_block: dict[str, Any] | None
) -> FamilyConfig:
    """Return a new ``FamilyConfig`` with per-model overrides applied.

    Trigger words are NOT part of FamilyConfig, so the caller still
    needs ``merge_prompt_config`` if it wants those. This variant is
    useful for code paths that only care about family-level fields
    (composer tuning, capability flags) and can fetch triggers
    separately.
    """
    merged = merge_prompt_config(family, prompt_block)
    family_fields = {f for f in asdict(family)}
    kwargs = {k: v for k, v in merged.items() if k in family_fields}
    return replace(family, **kwargs)

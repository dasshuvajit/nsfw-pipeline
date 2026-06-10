"""Token-budget enforcement for family-aware prompt composition.

Pre-Phase-C the builder's ``_warn_if_over_budget`` estimated tokens as
``len(words) × 1.3`` and only logged. CLIP silently truncates at 77,
T5 at 256/512 — and the truncation always lops off the *tail*, where
camera/lens/lighting tokens live. Over-budget prompts thus quietly
lose the very tokens that carry the family's quality signal.

Phase C swaps the heuristic for the real tokenizer per family:

  * CLIP families (sdxl / pony / illustrious) → ``open_clip`` SimpleTokenizer
    (ships with ``open-clip-torch``; no extra download)
  * T5 families (flux / chroma) → ``transformers`` T5Tokenizer for
    ``google/t5-v1_1-base`` — same SentencePiece vocab as T5-XXL/Schnell
    so token counts match the encoder
  * FLUX.2 family (flux2) → word-count heuristic. Klein 9B uses Qwen3-8B
    as text encoder; the Qwen3 tokenizer is large and the family runs
    distilled with no FluxGuidance, so a tight token cap is less
    critical. We approximate at ``words × 1.4``.

When over budget, ``fit_to_budget`` trims from the *middle* — the
front (subject identity) and tail (quality / camera tokens) carry the
most signal. Pony's ``BREAK`` token splits the prompt into two
encoder windows; each window fits independently.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Iterable, Protocol

logger = logging.getLogger(__name__)


# Reserve budget for BOS + EOS that the encoder adds. Two for CLIP
# (bos+eos), one for T5 (eos only), but two-everywhere keeps the math
# simple and the worst case is one wasted content slot.
_RESERVED_TOKENS = 2


class _Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


# ── Backends ────────────────────────────────────────────────────────


class _ClipTokenizer:
    """Wraps ``open_clip.tokenizer`` (SimpleTokenizer)."""

    def __init__(self) -> None:
        from open_clip.tokenizer import SimpleTokenizer

        self._inner = SimpleTokenizer()

    def count(self, text: str) -> int:
        if not text:
            return 0
        # ``encode`` returns the BPE token IDs without BOS/EOS padding.
        return len(self._inner.encode(text))


class _T5Tokenizer:
    """Wraps Hugging Face T5 (v1.1) tokenizer."""

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        self._inner = AutoTokenizer.from_pretrained("google/t5-v1_1-base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        # ``encode`` adds a trailing EOS — strip it so the count matches
        # bare content tokens. ``add_special_tokens=False`` is the clean
        # way to do this on HF tokenizers.
        ids = self._inner.encode(text, add_special_tokens=False)
        return len(ids)


class _HeuristicTokenizer:
    """Word-count heuristic fallback for tokenizers we don't ship.

    Calibrated to ``words × 1.4`` — a touch above the legacy 1.3 since
    English NSFW prompts skew long-form. Used for FLUX.2 (Qwen3 BPE
    too heavy to download for token-counting alone).
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = len(text.split())
        return int(words * 1.4 + 0.5)


@functools.lru_cache(maxsize=4)
def _get_tokenizer(tokenizer_id: str) -> _Tokenizer:
    """Lazy-load and cache one tokenizer per family-class."""
    if tokenizer_id == "clip":
        return _ClipTokenizer()
    if tokenizer_id == "t5":
        return _T5Tokenizer()
    if tokenizer_id == "heuristic":
        return _HeuristicTokenizer()
    raise ValueError(
        f"unknown tokenizer_id {tokenizer_id!r}; expected one of "
        f"'clip', 't5', 'heuristic'"
    )


# ── Public API ──────────────────────────────────────────────────────


def count_tokens(text: str, tokenizer_id: str) -> int:
    """Return the number of content tokens (no BOS/EOS) for ``text``."""
    return _get_tokenizer(tokenizer_id).count(text)


def fits_budget(text: str, *, max_tokens: int, tokenizer_id: str) -> bool:
    """Cheap "does this fit?" without doing a trim."""
    return count_tokens(text, tokenizer_id) <= max(0, max_tokens - _RESERVED_TOKENS)


def fit_to_budget(
    text: str,
    *,
    max_tokens: int,
    tokenizer_id: str,
    preserve_prefix_tokens: int = 30,
    preserve_suffix_tokens: int = 20,
    break_marker: str | None = None,
) -> str:
    """Trim ``text`` so its token count fits within ``max_tokens``.

    The first ``preserve_prefix_tokens`` and last ``preserve_suffix_tokens``
    are kept whole; trimming happens in the middle, at the nearest
    natural boundary (commas first, then periods, then whitespace as a
    last resort). Returns ``text`` unchanged when it already fits.

    ``break_marker`` (e.g. ``"BREAK"`` for Pony) splits the prompt into
    independent encoder windows; each window is fit separately and the
    marker re-inserted.
    """
    if not text:
        return text

    if break_marker and break_marker in text:
        # Pony BREAK semantics — each side hits its own 75-token CLIP
        # window. Trim each side independently, then re-join.
        parts = text.split(break_marker)
        trimmed = [
            fit_to_budget(
                p.strip().strip(","),
                max_tokens=max_tokens,
                tokenizer_id=tokenizer_id,
                preserve_prefix_tokens=preserve_prefix_tokens,
                preserve_suffix_tokens=preserve_suffix_tokens,
                break_marker=None,
            )
            for p in parts
        ]
        # Re-insert with surrounding spaces matching the canonical
        # Pony composer output ("..., BREAK, ...").
        return f", {break_marker}, ".join(t for t in trimmed if t)

    budget = max(0, max_tokens - _RESERVED_TOKENS)
    if budget <= 0:
        return ""

    full = count_tokens(text, tokenizer_id)
    if full <= budget:
        return text

    # Pick the first separator that yields >1 piece. Comma boundaries
    # are best for keyword families; periods catch prose families;
    # whitespace is the last-resort fallback for one-blob prompts.
    for sep in (",", ".", " "):
        pieces = _split_keep_tokens(text, sep)
        if len(pieces) > 1:
            joined = _pack_pieces(
                pieces,
                separator=sep,
                budget=budget,
                tokenizer_id=tokenizer_id,
                preserve_prefix_tokens=preserve_prefix_tokens,
                preserve_suffix_tokens=preserve_suffix_tokens,
            )
            if joined and count_tokens(joined, tokenizer_id) <= budget:
                return joined
    # Final fallback: hard character truncation. Only fires when the
    # input has no natural boundaries at all.
    return _hard_truncate(text, budget=budget, tokenizer_id=tokenizer_id)


# ── Helpers ──────────────────────────────────────────────────────────


def _split_keep_tokens(text: str, separator: str) -> list[str]:
    """Split on ``separator`` and discard empty pieces."""
    if separator == " ":
        return [p for p in text.split() if p]
    return [p.strip() for p in text.split(separator) if p.strip()]


def _pack_pieces(
    pieces: list[str],
    *,
    separator: str,
    budget: int,
    tokenizer_id: str,
    preserve_prefix_tokens: int,
    preserve_suffix_tokens: int,
) -> str:
    """Greedy packer: take prefix pieces, suffix pieces, fill middle.

    Always emits at least the first piece (subject) when the budget is
    non-zero; under extreme pressure the suffix may be empty rather
    than the prefix — front-loaded composition matches CLIP's
    position-weighted behaviour.
    """
    counts = [count_tokens(p, tokenizer_id) for p in pieces]

    # Build prefix until it spans preserve_prefix_tokens.
    prefix: list[str] = []
    prefix_total = 0
    i = 0
    while i < len(pieces) and prefix_total < preserve_prefix_tokens:
        # Always take at least one piece so the subject leads.
        if prefix and prefix_total + counts[i] > preserve_prefix_tokens:
            break
        prefix.append(pieces[i])
        prefix_total += counts[i]
        i += 1

    # Build suffix from the tail.
    suffix: list[str] = []
    suffix_total = 0
    j = len(pieces) - 1
    while j >= i and suffix_total < preserve_suffix_tokens:
        if suffix and suffix_total + counts[j] > preserve_suffix_tokens:
            break
        suffix.insert(0, pieces[j])
        suffix_total += counts[j]
        j -= 1

    # Pack middle into whatever's left.
    middle: list[str] = []
    middle_total = 0
    middle_budget = budget - prefix_total - suffix_total
    for k in range(i, j + 1):
        if middle_total + counts[k] > middle_budget:
            break
        middle.append(pieces[k])
        middle_total += counts[k]

    # Choose joining separator. Comma joins are universally legible;
    # period-split prose stays period-joined to keep sentence shape.
    join = ", " if separator == "," else (
        ". " if separator == "." else " "
    )
    parts = prefix + middle + suffix
    out = join.join(parts)
    if separator == "." and not out.endswith("."):
        out += "."
    return out


def _hard_truncate(text: str, *, budget: int, tokenizer_id: str) -> str:
    """Character-level binary search — last-resort trim."""
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip(", .")
        c = count_tokens(candidate, tokenizer_id)
        if c <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best

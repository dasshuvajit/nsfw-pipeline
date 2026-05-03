"""Content-level prompt sanitizer — ARCHITECTURE.md Section 5.

Applies the per-level keyword policies stored in ``content_level_rules``:

- ``prompt_suppress_keywords`` — words that must NOT appear in a prompt at
  this content level. Removed verbatim. T1/T2 prompts must not say
  "explicit, raw, aggressive, harsh"; T3/T4 prompts must not say
  "childlike, innocent, cute".

- ``prompt_boost_keywords`` — words that should appear in every prompt at
  this content level. Appended (case-insensitively de-duped) so the
  generator can lean on them as cues.

The Section 5 snippet does this with naked ``str.replace`` and a single
regex pass; we keep that behaviour but harden the edges:

1. **Word-boundary suppression.** ``str.replace("aggressive", "")`` would
   eat the substring inside ``"aggressively"``. We use a word-boundary
   regex so we only delete whole words.

2. **Idempotent.** Running the sanitizer twice on the same prompt yields
   the same result — important because the engine reorders sanitize/dedup
   passes between modes and we don't want compounding effects.

3. **Contract preserved.** ``apply(prompts, ctx)`` mutates each dict
   in-place exactly as the spec snippet does, sets ``content_level``, and
   returns the same list for call-site chaining.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

logger = logging.getLogger(__name__)


# Collapse runs of commas/whitespace introduced by suppression. Used after
# every prompt rewrite so we never emit ", , ," sequences.
_COMMA_RUN_RE = re.compile(r"\s*,\s*(,\s*)+")
_LEADING_TRAILING_COMMAS_RE = re.compile(r"^[\s,]+|[\s,]+$")
# Collapse runs of internal whitespace introduced by suppression
# (",  pose" → ", pose"). We do NOT touch newlines because prompts are
# always single-line.
_WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")

# Celebrity-likeness heuristic — two consecutive capitalized tokens
# (each ≥3 chars) trigger a warn + strip unless the 2-gram is on the
# allowlist. The allowlist mixes:
#   * archetype aesthetic 2-grams (loaded from style_profiles.yaml)
#   * common geography / photography / fashion 2-grams
# Sentence-starts rarely false-positive because prose composers
# capitalize only the *first* word of a sentence — the pattern requires
# two adjacent capitalized tokens. "Emma Stone" matches; "She reclines"
# does not.
_CELEB_2GRAM_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")

_CELEBRITY_ALLOWLIST_STATIC: frozenset[str] = frozenset(
    {
        # aesthetic / archetype (redundant with dynamic loader but kept
        # as a safety net when style_profiles.yaml is missing)
        "old hollywood", "hollywood glamour", "golden hour",
        "neo noir", "film noir", "wet set", "fine art", "art nude",
        "pin up", "boudoir noir", "fashion nude", "editorial fashion",
        "fantasy castlecore", "moody monochrome", "vintage pinup",
        "kodachrome tone",
        # lens + photography vocabulary
        "shallow depth", "soft light", "natural light", "low key",
        "high key", "rim light", "key light", "fill light",
        "back light", "hard light", "diffused light", "window light",
        "beauty dish", "soft box", "ring flash", "portrait lens",
        "prime lens", "zoom lens", "wide angle", "close up",
        "medium shot", "full body", "upper body", "side profile",
        "eye level", "low angle", "high angle", "over shoulder",
        "film grain", "film stock", "film tone", "cinematic grade",
        "color grading", "color palette",
        # geography (places that would otherwise match as 2-gram
        # proper nouns)
        "new york", "los angeles", "san francisco", "hong kong",
        "las vegas", "new orleans", "south beach",
        # tech / stylistic phrases that look like proper-name pairs
        "old master", "fine print", "art deco", "high fashion",
    }
)

_STYLE_PROFILES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "style_profiles.yaml"
)


@functools.lru_cache(maxsize=1)
def _load_archetype_2grams(path: Path = _STYLE_PROFILES_PATH) -> frozenset[str]:
    """Derive 2-gram allowlist entries from style_profiles display names.

    Hyphens and punctuation are stripped so ``"Old-Hollywood Glamour"``
    contributes ``"old hollywood"`` *and* ``"hollywood glamour"``. Missing
    or malformed YAML returns an empty set — the static allowlist is
    sufficient fallback.
    """
    if not path.exists():
        return frozenset()
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover — YAML parse failure
        logger.warning("celebrity allowlist: failed to parse %s: %s", path, exc)
        return frozenset()
    profiles = data.get("profiles") or {}
    grams: set[str] = set()
    for pdata in profiles.values():
        name = str(pdata.get("name", "")).strip()
        if not name:
            continue
        cleaned = re.sub(r"[^A-Za-z\s]", " ", name).lower()
        tokens = [t for t in cleaned.split() if len(t) >= 3]
        for i in range(len(tokens) - 1):
            grams.add(f"{tokens[i]} {tokens[i + 1]}")
    return frozenset(grams)


def _celebrity_allowlist() -> frozenset[str]:
    return _CELEBRITY_ALLOWLIST_STATIC | _load_archetype_2grams()


def _detect_celebrity_likeness(body: str) -> list[str]:
    """Return capitalized 2-gram matches not on the allowlist.

    Used for platform-compliance safety: real-celebrity-likeness is an
    instant ban vector on DA / Patreon, so we strip matches and log at
    WARN. False-positive on an aesthetic 2-gram is painful (the strip
    breaks the prompt), so the allowlist is deliberately generous.
    """
    if not body:
        return []
    allow = _celebrity_allowlist()
    candidates = _CELEB_2GRAM_PATTERN.findall(body)
    return [m for m in candidates if m.lower() not in allow]


def _strip_celebrity_likeness(text: str) -> str:
    """Strip celebrity-likeness 2-grams from a prompt body. Logs at WARN."""
    matches = _detect_celebrity_likeness(text)
    if not matches:
        return text
    logger.warning(
        "Celebrity-likeness 2-grams detected — stripping. matches=%s",
        sorted(set(matches)),
    )
    out = text
    for m in matches:
        out = re.sub(rf"\b{re.escape(m)}\b", "", out)
    out = _COMMA_RUN_RE.sub(", ", out)
    out = _WHITESPACE_RUN_RE.sub(" ", out)
    out = _LEADING_TRAILING_COMMAS_RE.sub("", out).strip()
    return out


class PromptSanitizer:
    """Per-content-level keyword filter for built prompts."""

    def __init__(
        self,
        boost_keywords: Iterable[str] | None = None,
        suppress_keywords: Iterable[str] | None = None,
    ) -> None:
        """Construct from explicit lists.

        Most callers should use :meth:`from_rules` to wire the sanitizer
        directly to a :class:`ContentLevelRules` instance loaded from
        ``config/categories.yaml`` via ``ContentLevelLoader``. The
        explicit-list constructor is here for tests and for ad-hoc
        one-off sanitization.
        """
        self.boost_keywords: list[str] = list(boost_keywords or [])
        self.suppress_keywords: list[str] = list(suppress_keywords or [])

    @classmethod
    def from_rules(cls, rules: Any) -> "PromptSanitizer":
        """Build from a :class:`ContentLevelRules` (or any object with the
        ``prompt_boost_keywords`` / ``prompt_suppress_keywords`` attrs).
        """
        return cls(
            boost_keywords=getattr(rules, "prompt_boost_keywords", []) or [],
            suppress_keywords=getattr(rules, "prompt_suppress_keywords", []) or [],
        )

    # ----- single-prompt path -----------------------------------------------

    def sanitize_text(self, prompt_text: str) -> str:
        """Apply suppress + boost to a single prompt string. Idempotent."""
        text = prompt_text or ""

        # 1. Suppress: word-boundary regex per keyword. Three branches so
        # we eat one adjacent comma+space along with the keyword whenever
        # possible — that prevents ", explicit pose" from collapsing to
        # ",  pose" with a stranded double space. Branch order matters:
        # the leading-comma form has to come first or the lone-word form
        # would gobble the keyword and leave the trailing comma behind.
        for word in self.suppress_keywords:
            if not word:
                continue
            pattern = (
                rf"\s*,\s*\b{re.escape(word)}\b"  # ", explicit"
                rf"|\b{re.escape(word)}\b\s*,\s*"  # "explicit, "
                rf"|\b{re.escape(word)}\b"        # bare "explicit"
            )
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # 2. Collapse the comma + whitespace noise that suppression
        # leaves behind.
        text = _WHITESPACE_RUN_RE.sub(" ", text)
        text = _COMMA_RUN_RE.sub(", ", text)
        text = _LEADING_TRAILING_COMMAS_RE.sub("", text).strip()

        # 3. Boost: append any missing boost keyword (case-insensitive
        # whole-word check, so "soft lighting" already in the prompt
        # short-circuits the addition).
        for word in self.boost_keywords:
            if not word:
                continue
            if not re.search(
                rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE
            ):
                text = (text.rstrip(", ") + (", " if text else "") + word).strip()

        # 4. Celebrity-likeness safety. Always-on, tier-agnostic — the
        # heuristic is the same platform-compliance gate regardless of
        # content level.
        text = _strip_celebrity_likeness(text)

        # 5. Final tidy: collapse one more time in case boost-append
        # introduced anything unusual, then trim.
        text = _COMMA_RUN_RE.sub(", ", text)
        text = _LEADING_TRAILING_COMMAS_RE.sub("", text).strip()
        return text

    # ----- batch path (the spec contract) -----------------------------------

    def apply(
        self,
        prompts: list[dict[str, Any]],
        content_level: str | None = None,
    ) -> list[dict[str, Any]]:
        """Sanitize every prompt dict in-place; return the same list.

        Mirrors the Section 5 ``PromptSanitizer.apply(prompts, ctx)`` shape.
        We accept ``content_level`` as a plain string (rather than a full
        ``ctx``) because the first caller —``scripts/render_set.py`` —
        doesn't yet build a real :class:`GenerationContext`. When that
        lands, swap to ``apply(prompts, ctx)`` and read
        ``ctx.content_level``.

        Each dict must have a ``prompt_text`` key; we write back the
        sanitized text and stamp ``content_level`` if provided.
        """
        for prompt in prompts:
            if "prompt_text" not in prompt:
                logger.warning(
                    "PromptSanitizer.apply: prompt dict missing 'prompt_text': %r",
                    list(prompt.keys()),
                )
                continue
            prompt["prompt_text"] = self.sanitize_text(prompt["prompt_text"])
            if content_level is not None:
                prompt["content_level"] = content_level
        return prompts

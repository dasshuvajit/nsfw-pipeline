"""Per-tier sanitizer suppress/boost behaviour — word-boundary safe + idempotent."""

from __future__ import annotations

from src.prompt.sanitizer import PromptSanitizer


def test_suppress_respects_word_boundary():
    s = PromptSanitizer(
        suppress_keywords=["explicit"],
        boost_keywords=[],
    )
    # "explicitly" must NOT be eaten — "explicit" as a bare word must.
    out = s.sanitize_text("elara, explicit pose, explicitly lit")
    assert "explicit pose" not in out
    assert "explicitly lit" in out


def test_boost_appended_when_missing():
    s = PromptSanitizer(
        suppress_keywords=[],
        boost_keywords=["tasteful", "artistic"],
    )
    out = s.sanitize_text("elara, seated")
    assert "tasteful" in out
    assert "artistic" in out


def test_boost_strips_trailing_period_before_appending_comma():
    """Round-22 followup (2026-05-22) — when the input ends with "."
    (prose-family composer outputs end with a period), the boost-keyword
    append must strip that period before joining with ", ". Pre-fix the
    boost produced ", lingerie" after a "." giving "natural skin texture.,
    lingerie" — period-comma jammed together looks like a cosmetic bug
    in the final prompt."""
    s = PromptSanitizer(
        suppress_keywords=[],
        boost_keywords=["lingerie", "swimwear"],
    )
    out = s.sanitize_text(
        "She stands in golden hour light. f/1.8, 35mm, photographic, "
        "natural skin texture."
    )
    # Boost keywords landed.
    assert "lingerie" in out
    assert "swimwear" in out
    # NO period-then-comma sequence.
    assert ".," not in out, (
        f"period-then-comma found in: {out!r}"
    )
    # ALSO no "., " (period followed by comma+space) — same defect class.
    assert "., " not in out, (
        f"period-then-comma+space found in: {out!r}"
    )


def test_boost_idempotent():
    s = PromptSanitizer(
        suppress_keywords=[],
        boost_keywords=["tasteful"],
    )
    once = s.sanitize_text("elara, seated")
    twice = s.sanitize_text(once)
    assert once == twice


def test_sanitize_is_idempotent_across_full_pipeline():
    s = PromptSanitizer(
        suppress_keywords=["explicit", "harsh"],
        boost_keywords=["tasteful"],
    )
    first = s.sanitize_text("elara, explicit, harsh lighting")
    second = s.sanitize_text(first)
    third = s.sanitize_text(second)
    assert first == second == third


def test_no_double_commas_after_suppression():
    s = PromptSanitizer(
        suppress_keywords=["raw"],
        boost_keywords=[],
    )
    out = s.sanitize_text("elara, raw, seated, raw pose, ready")
    assert ",," not in out
    assert ", ," not in out


def test_apply_batch_stamps_content_level():
    s = PromptSanitizer(suppress_keywords=[], boost_keywords=["tasteful"])
    prompts = [
        {"prompt_text": "elara, seated", "prompt_hash": "abc"},
        {"prompt_text": "mara, lying", "prompt_hash": "def"},
    ]
    out = s.apply(prompts, content_level="T2_implied")
    assert all(p["content_level"] == "T2_implied" for p in out)
    assert all("tasteful" in p["prompt_text"] for p in out)


def test_empty_prompt_survives():
    s = PromptSanitizer(suppress_keywords=["x"], boost_keywords=["y"])
    out = s.sanitize_text("")
    # Empty input → boost still appended (matches current contract)
    assert out == "y" or out == ""  # either behaviour is acceptable

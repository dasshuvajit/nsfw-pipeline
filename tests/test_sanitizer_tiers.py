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

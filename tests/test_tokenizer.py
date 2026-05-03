"""Phase C — token-budget enforcement tests.

Covers the three tokenizer backends (``clip`` / ``t5`` / ``heuristic``)
and the trim path:

  * ``count_tokens`` — empty, normal, long, weighting-syntax pass-through
  * ``fits_budget`` — at, just-under, just-over budget
  * ``fit_to_budget`` — already-fits no-op, mid-trim, prefix/suffix
    preservation, BREAK splitting (Pony), hard-truncate fallback.

The tests use real tokenizer counts where the inputs are short and
deterministic; no mocks. Avoids redownloading by relying on
``functools.lru_cache`` inside ``_get_tokenizer``.
"""

from __future__ import annotations

import pytest

from src.prompt.tokenizer import (
    count_tokens,
    fit_to_budget,
    fits_budget,
)


# ── count_tokens ────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_returns_zero_clip(self):
        assert count_tokens("", "clip") == 0

    def test_empty_returns_zero_t5(self):
        assert count_tokens("", "t5") == 0

    def test_empty_returns_zero_heuristic(self):
        assert count_tokens("", "heuristic") == 0

    def test_clip_short_text(self):
        n = count_tokens("a woman in a red dress", "clip")
        assert 6 <= n <= 8  # SimpleTokenizer BPE on this phrase

    def test_t5_short_text(self):
        # T5 SentencePiece is denser per word than CLIP BPE for short
        # English — this lets the test stay deterministic across HF
        # cache versions.
        n = count_tokens("a woman in a red dress", "t5")
        assert 6 <= n <= 9

    def test_heuristic_words_times_one_point_four(self):
        # 10 words * 1.4 = 14
        text = " ".join(["word"] * 10)
        assert count_tokens(text, "heuristic") == 14

    def test_unknown_tokenizer_id_raises(self):
        with pytest.raises(ValueError, match="unknown tokenizer_id"):
            count_tokens("hello", "qwen3")  # not yet supported


# ── fits_budget ─────────────────────────────────────────────────────


class TestFitsBudget:
    def test_short_text_fits(self):
        assert fits_budget("a woman", max_tokens=77, tokenizer_id="clip")

    def test_empty_text_fits(self):
        # Zero tokens vs. budget of 1 (minus 2 reserved → 0). Boundary case.
        assert fits_budget("", max_tokens=10, tokenizer_id="clip")

    def test_long_text_does_not_fit_clip(self):
        long_text = ", ".join(f"keyword_{i}" for i in range(80))
        assert not fits_budget(long_text, max_tokens=77, tokenizer_id="clip")

    def test_long_text_fits_at_high_budget(self):
        long_text = ", ".join(f"keyword_{i}" for i in range(40))
        assert fits_budget(long_text, max_tokens=512, tokenizer_id="t5")


# ── fit_to_budget ───────────────────────────────────────────────────


class TestFitToBudgetNoOp:
    def test_empty_returns_unchanged(self):
        assert fit_to_budget("", max_tokens=77, tokenizer_id="clip") == ""

    def test_short_text_returns_unchanged(self):
        text = "a woman in a red dress, 85mm lens, soft light"
        out = fit_to_budget(text, max_tokens=77, tokenizer_id="clip")
        assert out == text

    def test_zero_budget_returns_empty(self):
        text = "a woman"
        # max_tokens=2 → budget 0 after reservation → empty.
        out = fit_to_budget(text, max_tokens=2, tokenizer_id="clip")
        assert out == ""


class TestFitToBudgetTrim:
    def test_clip_trim_keeps_under_budget(self):
        long_text = ", ".join(f"keyword_{i}" for i in range(80))
        out = fit_to_budget(long_text, max_tokens=77, tokenizer_id="clip")
        assert count_tokens(out, "clip") <= 75  # 77 - 2 reserved

    def test_clip_trim_preserves_prefix(self):
        # First several tokens should survive — that's the subject identity.
        long_text = (
            "1girl, long_hair, looking_at_viewer, "
            + ", ".join(f"middle_filler_{i}" for i in range(60))
            + ", masterpiece, best quality"
        )
        out = fit_to_budget(long_text, max_tokens=77, tokenizer_id="clip")
        assert out.startswith("1girl, long_hair, looking_at_viewer")

    def test_clip_trim_preserves_suffix_when_possible(self):
        # The quality suffix at the tail should survive a middle trim.
        long_text = (
            "1girl, "
            + ", ".join(f"middle_filler_{i}" for i in range(60))
            + ", masterpiece, best quality, very aesthetic"
        )
        out = fit_to_budget(long_text, max_tokens=77, tokenizer_id="clip")
        assert "very aesthetic" in out

    def test_t5_trim_long_prose(self):
        long_prose = (
            "A confident woman in a modern loft. "
            + ". ".join(
                f"Sentence number {i} with several descriptive words"
                for i in range(100)
            )
            + ". She has realistic skin texture."
        )
        out = fit_to_budget(long_prose, max_tokens=256, tokenizer_id="t5")
        assert count_tokens(out, "t5") <= 254

    def test_heuristic_trim(self):
        long_text = " ".join(["word"] * 1000)
        out = fit_to_budget(long_text, max_tokens=100, tokenizer_id="heuristic")
        assert count_tokens(out, "heuristic") <= 98


class TestFitToBudgetBreakMarker:
    def test_pony_break_each_window_independently_fit(self):
        # Pony's BREAK splits prompt into two CLIP-77 windows. Each side
        # must fit independently; the marker re-inserted between.
        prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
        body = ", ".join(f"keyword_{i}" for i in range(80))
        text = f"{prefix}, BREAK, {body}"
        out = fit_to_budget(
            text, max_tokens=77, tokenizer_id="clip", break_marker="BREAK",
        )
        assert "BREAK" in out
        # Each side independently fits 75 tokens.
        left, right = out.split("BREAK", 1)
        assert count_tokens(left.strip(", "), "clip") <= 75
        assert count_tokens(right.strip(", "), "clip") <= 75

    def test_no_break_marker_when_not_present(self):
        text = ", ".join(f"keyword_{i}" for i in range(40))
        out = fit_to_budget(
            text, max_tokens=77, tokenizer_id="clip", break_marker="BREAK",
        )
        assert "BREAK" not in out

    def test_break_marker_pony_score_prefix_survives(self):
        # The 6-tier score prefix is the identity of Pony — must always
        # survive trimming on the left window.
        prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
        body = ", ".join(f"long_keyword_{i}" for i in range(80))
        text = f"{prefix}, BREAK, {body}"
        out = fit_to_budget(
            text, max_tokens=77, tokenizer_id="clip", break_marker="BREAK",
        )
        for tier in ("score_9", "score_8_up", "score_7_up"):
            assert tier in out


class TestFitToBudgetExactBoundaries:
    @pytest.mark.parametrize("offset", [-1, 0, 1])
    def test_around_budget_clip(self, offset):
        # Build a text whose token count is exactly budget+offset.
        # Budget = 77 - 2 = 75. Build prefix until at boundary.
        target = 75 + offset
        tokens: list[str] = []
        while True:
            tokens.append(f"k{len(tokens)}")
            if count_tokens(", ".join(tokens), "clip") >= target:
                break
        # Trim to exactly target.
        while count_tokens(", ".join(tokens), "clip") > target:
            tokens.pop()
        text = ", ".join(tokens)
        actual = count_tokens(text, "clip")
        out = fit_to_budget(text, max_tokens=77, tokenizer_id="clip")
        if actual <= 75:
            assert out == text  # already fit, no-op
        else:
            assert count_tokens(out, "clip") <= 75


class TestFitToBudgetHardTruncate:
    def test_one_blob_no_separators(self):
        # No commas, periods, or spaces — should fall back to char-trim.
        # Use a long single token-soup; CLIP will tokenize it densely.
        text = "a" * 10000
        out = fit_to_budget(text, max_tokens=77, tokenizer_id="clip")
        # Hard truncate keeps it under budget.
        assert count_tokens(out, "clip") <= 75
        assert len(out) < len(text)

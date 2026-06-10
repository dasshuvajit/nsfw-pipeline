"""Phase H — fixture-based regression harness for the prompt pipeline.

Discovers every ``tests/fixtures/regression/{family}/case_*.input.yaml``,
runs it through ``PromptBuilder`` + ``RatioSelector`` + the tokenizer,
and asserts the result matches the paired ``case_*.expected.yaml`` file
exactly.

This catches drift in:

  * Per-family composers (sdxl_keywords, pony_danbooru, illustrious_tags,
    flux_natural, flux2_prose, chroma's period-realism tail).
  * Negative-axis flatten + HARD_BLOCK ordering (Phase D / E).
  * Token-budget trimming (Phase C — affects which tokens survive).
  * Multi-axis ratio scoring (Phase A — content_level + audience +
    composition + family bonuses + pose hard-overrides).
  * Per-family resolution buckets (sdxl 1MP vs flux/flux2 1.5MP/2MP).

Failure mode: a clean ``assert expected == actual`` diff per field, with
the operator-facing instruction to re-run
``scripts/regenerate_regression_fixtures.py`` after intentional drift.

When you change a family's quality_suffix, a composer's join logic, the
tokenizer trim contract, or any ratio_signals.yaml entry, expect ~3
fixtures (one tier × the affected family) to fail. Re-run the
regenerator, eyeball ``git diff tests/fixtures/regression/``, commit if
the change is intentional. If the diff is unexpected, you've found a
regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._regression_harness import (
    compute_expected,
    discover_input_fixtures,
    expected_path_for,
    load_yaml,
    relpath,
)


_INPUT_FIXTURES = discover_input_fixtures()

# Spell out the expected count so a missing/misnamed fixture trips a
# loud failure rather than silently shrinking the regression surface.
# Round-22 (2026-05-22) — chroma gained a 4th case (case_4) covering
# the F1 lens-populated branch (realism_lens canonicalizes to "85mm
# f/1.4 lens..." and the chroma realism tail drops its hardcoded
# "f/1.8, 35mm" tokens to avoid two-focal-length contradiction).
_EXPECTED_FAMILIES = {"sdxl", "pony", "illustrious", "flux", "chroma", "flux2"}
_EXPECTED_TOTAL = (len(_EXPECTED_FAMILIES) * 3) + 1  # +1 for chroma/case_4


def test_fixture_inventory_complete() -> None:
    """6 families × 3 cases each = 18 input fixtures, plus chroma/case_4
    (round-22 F1 regression coverage) = 19 total."""
    assert len(_INPUT_FIXTURES) == _EXPECTED_TOTAL, (
        f"Expected {_EXPECTED_TOTAL} input fixtures, found "
        f"{len(_INPUT_FIXTURES)}: {[relpath(p) for p in _INPUT_FIXTURES]}"
    )
    families_seen = {p.parent.name for p in _INPUT_FIXTURES}
    assert families_seen == _EXPECTED_FAMILIES, (
        f"Missing families: {_EXPECTED_FAMILIES - families_seen}; "
        f"unexpected: {families_seen - _EXPECTED_FAMILIES}"
    )


@pytest.mark.parametrize(
    "input_path",
    _INPUT_FIXTURES,
    ids=[f"{p.parent.name}-{p.stem.replace('.input', '')}" for p in _INPUT_FIXTURES],
)
def test_fixture_matches_expected(input_path: Path) -> None:
    """Compose + select + count, assert against frozen expected output.

    On failure, the assert message points at the regenerator script —
    that's the canonical way to update the baseline.
    """
    expected_path = expected_path_for(input_path)
    if not expected_path.exists():
        pytest.fail(
            f"Missing expected fixture {relpath(expected_path)}.\n"
            f"Run:  python scripts/regenerate_regression_fixtures.py\n"
            f"to bake the current pipeline output as the new baseline."
        )

    input_data = load_yaml(input_path)
    expected = load_yaml(expected_path).get("expected") or {}
    if not expected:
        pytest.fail(
            f"{relpath(expected_path)} has no top-level `expected:` block. "
            f"Re-run scripts/regenerate_regression_fixtures.py."
        )

    actual = compute_expected(input_data)

    # Field-by-field assertion so the failure pinpoints which signal
    # drifted (rather than dumping a giant dict-vs-dict diff).
    for key in (
        "prompt_text",
        "prompt_token_count",
        "negative_prompt",
        "ratio",
        "resolution",
        "reason",
    ):
        assert key in expected, (
            f"{relpath(expected_path)}: expected block missing key {key!r}. "
            f"Re-run the regenerator."
        )
        # Lists round-trip from YAML as lists; tuples don't survive YAML
        # so we already coerce in the harness.
        assert actual[key] == expected[key], (
            f"\n[{relpath(input_path)}] {key} drift:\n"
            f"  expected: {expected[key]!r}\n"
            f"  actual  : {actual[key]!r}\n"
            f"If intentional, re-run:\n"
            f"  python scripts/regenerate_regression_fixtures.py"
        )

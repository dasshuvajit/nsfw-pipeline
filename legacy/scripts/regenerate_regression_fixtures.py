#!/usr/bin/env python3
"""Phase H — bake the regression-fixture baseline.

Reads every ``tests/fixtures/regression/{family}/case_*.input.yaml``,
runs it through the live ``PromptBuilder`` + ``RatioSelector`` +
tokenizer, and writes the paired ``case_*.expected.yaml`` file. Pairs
with ``tests/test_regression_render_smoke.py`` — the test asserts
exact equality with what this script wrote.

When to re-run:

  * After an intentional change to family YAML (quality prefix/suffix,
    avoid_words, negative axes), per-model YAML, ``ratio_signals.yaml``,
    or any composer in ``src/prompt/builder.py``.
  * After a tokenizer-backend change (rare; affects token counts).
  * After adding / removing input fixtures.

Workflow:

  1. ``python scripts/regenerate_regression_fixtures.py``
  2. ``git diff tests/fixtures/regression/`` — review every drift.
  3. If the drift is intentional, commit. If not, you've caught a
     regression — fix the code, re-run, re-diff.

Usage:

  python scripts/regenerate_regression_fixtures.py            # all fixtures
  python scripts/regenerate_regression_fixtures.py --family sdxl
  python scripts/regenerate_regression_fixtures.py --check     # dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``tests/_regression_harness.py`` importable without installing
# the project as a package — same trick the other scripts/ files use.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._regression_harness import (  # noqa: E402
    compute_expected,
    discover_input_fixtures,
    dump_yaml,
    expected_path_for,
    load_yaml,
    relpath,
)


def regenerate_one(input_path: Path) -> tuple[Path, dict, dict | None]:
    """Compute the expected block for one input fixture.

    Returns ``(expected_path, new_expected, prior_expected | None)`` so the
    caller can diff before writing.
    """
    input_data = load_yaml(input_path)
    expected_block = compute_expected(input_data)

    expected_path = expected_path_for(input_path)
    prior = None
    if expected_path.exists():
        prior_doc = load_yaml(expected_path)
        prior = prior_doc.get("expected") if isinstance(prior_doc, dict) else None

    return expected_path, expected_block, prior


def _has_drift(prior: dict | None, fresh: dict) -> bool:
    return prior != fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        help="Only regenerate fixtures for this family id (sdxl/pony/...).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: report drift, write nothing. Exits 1 if any "
             "fixture would change.",
    )
    args = parser.parse_args()

    fixtures = discover_input_fixtures()
    if args.family:
        fixtures = [p for p in fixtures if p.parent.name == args.family]
        if not fixtures:
            print(f"No fixtures matched --family={args.family!r}.", file=sys.stderr)
            return 2

    if not fixtures:
        print("No regression fixtures discovered. Nothing to do.")
        return 0

    drifted: list[Path] = []
    written: list[Path] = []

    for input_path in fixtures:
        expected_path, fresh, prior = regenerate_one(input_path)
        if _has_drift(prior, fresh):
            drifted.append(expected_path)
            if args.check:
                print(f"DRIFT: {relpath(expected_path)}")
            else:
                dump_yaml(expected_path, {"expected": fresh})
                written.append(expected_path)

    print(
        f"\n{len(fixtures)} fixture(s) processed; "
        f"{len(drifted)} drifted; "
        f"{len(written)} expected file(s) written."
    )
    if args.check and drifted:
        print(
            "\n--check failed: re-run without --check to update fixtures, "
            "then `git diff tests/fixtures/regression/` to review.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

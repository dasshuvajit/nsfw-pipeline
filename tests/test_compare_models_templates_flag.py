"""CLI contract for ``scripts/compare_models.py`` ``--template`` / ``--templates``.

Two pure helpers are tested here without going through the full render
loop:

- ``_build_candidates(model_ids, template_tokens)`` — returns a flat
  list of ``(model_id, template_token)`` tuples, and raises ``SystemExit``
  when a non-``system`` template is combined with multiple models
  (the user's explicit N×M-is-nonsense rule).

- ``_assign_slugs(candidates)`` — attaches the output-subdir slug to
  each candidate. ``'system'`` / ``'default'`` → literal ``"system"``,
  other paths → ``Path.stem.lower()``, with ``__2`` / ``__3`` suffixes
  when the same ``model_id + slug`` repeats.

Also asserts the argparse mutually-exclusive group between
``--template`` (singular) and ``--templates`` (CSV), and the
``required=True`` constraint on ``--models`` (existing behavior kept).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_COMPARE_MODELS_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_models.py"
)


@pytest.fixture(scope="module")
def compare_models_module():
    project_root = str(_COMPARE_MODELS_PATH.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    spec = importlib.util.spec_from_file_location(
        "compare_models", _COMPARE_MODELS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- _slug_for_template ------------------------------------------


def test_slug_for_system_token(compare_models_module):
    assert compare_models_module._slug_for_template("system") == "system"


def test_slug_for_default_token_aliases_system(compare_models_module):
    assert compare_models_module._slug_for_template("default") == "system"


def test_slug_for_template_path_uses_stem(compare_models_module):
    assert (
        compare_models_module._slug_for_template(
            "templates/chroma/chroma_done_properly.json"
        )
        == "chroma_done_properly"
    )


def test_slug_for_template_lowercases(compare_models_module):
    assert (
        compare_models_module._slug_for_template(
            "templates/chroma/UPPERCASE_Template.json"
        )
        == "uppercase_template"
    )


def test_slug_for_template_path_strips_whitespace(compare_models_module):
    assert compare_models_module._slug_for_template("   a.json  ") == "a"


# ---------- _build_candidates -------------------------------------------


def test_build_candidates_single_model_multiple_templates(compare_models_module):
    out = compare_models_module._build_candidates(
        model_ids=["chroma_1hd"],
        template_tokens=["system", "templates/chroma/chroma_done_properly.json"],
    )
    assert out == [
        ("chroma_1hd", "system"),
        ("chroma_1hd", "templates/chroma/chroma_done_properly.json"),
    ]


def test_build_candidates_single_model_single_external(compare_models_module):
    out = compare_models_module._build_candidates(
        model_ids=["chroma_1hd"],
        template_tokens=["templates/chroma/chroma_done_properly.json"],
    )
    assert out == [("chroma_1hd", "templates/chroma/chroma_done_properly.json")]


def test_build_candidates_multi_model_all_system_ok(compare_models_module):
    """Multiple models × ``system`` is the pre-existing behavior —
    must keep working."""
    out = compare_models_module._build_candidates(
        model_ids=["m1", "m2", "m3"],
        template_tokens=["system"],
    )
    assert out == [("m1", "system"), ("m2", "system"), ("m3", "system")]


def test_build_candidates_multi_model_default_alias_ok(compare_models_module):
    out = compare_models_module._build_candidates(
        model_ids=["m1", "m2"],
        template_tokens=["default"],
    )
    assert out == [("m1", "default"), ("m2", "default")]


def test_build_candidates_single_external_template_broadcasts_to_all_models(
    compare_models_module,
):
    """**Phase 5 behavior change**: a single external template now
    broadcasts to every model (used to reject N×M).

    Rationale: with positional pairing in place, "1 template applied to
    all" is now a deliberate, common case (e.g. 3 SDXL siblings tested
    against a Chroma template — user knows what they're doing). The
    family-mismatch concern is left to the user's discretion.
    """
    out = compare_models_module._build_candidates(
        model_ids=["m1", "m2"],
        template_tokens=["templates/chroma/chroma_done_properly.json"],
    )
    assert out == [
        ("m1", "templates/chroma/chroma_done_properly.json"),
        ("m2", "templates/chroma/chroma_done_properly.json"),
    ]


def test_build_candidates_n_models_n_templates_paired_positionally(
    compare_models_module,
):
    """**Phase 5 behavior change**: N==N is now positional pairing
    (model[i] uses template[i]), NOT Cartesian (which would have been
    N×N renders before Phase 5). Documented in PROJECT_GUIDE.md and
    the CLI docstring."""
    out = compare_models_module._build_candidates(
        model_ids=["lustify_v7", "chroma_v10HD"],
        template_tokens=["system", "templates/chroma/x.json"],
    )
    assert out == [
        ("lustify_v7", "system"),
        ("chroma_v10HD", "templates/chroma/x.json"),
    ]


def test_build_candidates_n_models_n_templates_three_pairs(
    compare_models_module,
):
    """Three models + three templates → three paired renders."""
    out = compare_models_module._build_candidates(
        model_ids=["a", "b", "c"],
        template_tokens=["system", "templates/x.json", "templates/y.json"],
    )
    assert out == [
        ("a", "system"),
        ("b", "templates/x.json"),
        ("c", "templates/y.json"),
    ]


def test_build_candidates_mismatched_lengths_both_above_one_rejected(
    compare_models_module,
):
    """N models ≠ M templates (both > 1) → explicit error explaining
    positional-pairing requirement."""
    with pytest.raises(SystemExit) as exc:
        compare_models_module._build_candidates(
            model_ids=["m1", "m2"],
            template_tokens=["a", "b", "c"],
        )
    msg = str(exc.value)
    assert "must equal" in msg
    assert "positional pairing" in msg.lower()
    # Educational note about the breaking change.
    assert "Cartesian" in msg or "cartesian" in msg.lower()


def test_build_candidates_empty_templates_defaults_to_system(
    compare_models_module,
):
    """When --templates/--template is omitted, fall back to 'system'
    for every model. Preserves the pre-plan default behavior."""
    out = compare_models_module._build_candidates(
        model_ids=["m1", "m2"],
        template_tokens=[],
    )
    assert out == [("m1", "system"), ("m2", "system")]


# ---------- _assign_slugs -----------------------------------------------


def test_assign_slugs_system_and_external(compare_models_module):
    out = compare_models_module._assign_slugs([
        ("chroma_1hd", "system"),
        ("chroma_1hd", "templates/chroma/chroma_done_properly.json"),
    ])
    assert out == [
        ("chroma_1hd", "system", "system"),
        ("chroma_1hd", "templates/chroma/chroma_done_properly.json",
         "chroma_done_properly"),
    ]


def test_assign_slugs_collision_gets_numeric_suffix(compare_models_module):
    """Two templates with the same stem on the same model → second
    gets ``__2`` to keep output dirs unique."""
    out = compare_models_module._assign_slugs([
        ("chroma_1hd", "templates/a/chroma_done_properly.json"),
        ("chroma_1hd", "templates/b/chroma_done_properly.json"),
    ])
    slugs = [s for _, _, s in out]
    assert slugs == ["chroma_done_properly", "chroma_done_properly__2"]


def test_assign_slugs_triple_collision(compare_models_module):
    out = compare_models_module._assign_slugs([
        ("m1", "templates/a/x.json"),
        ("m1", "templates/b/x.json"),
        ("m1", "templates/c/x.json"),
    ])
    slugs = [s for _, _, s in out]
    assert slugs == ["x", "x__2", "x__3"]


def test_assign_slugs_no_collision_across_models(compare_models_module):
    """Same slug on different models is fine — output dirs include the
    model id so they can't actually collide."""
    out = compare_models_module._assign_slugs([
        ("m1", "templates/a/x.json"),
        ("m2", "templates/a/x.json"),
    ])
    slugs = [s for _, _, s in out]
    assert slugs == ["x", "x"]


# ---------- argparse: mutually-exclusive group --------------------------


def test_template_and_templates_mutually_exclusive(
    compare_models_module, capsys
):
    """``--template`` and ``--templates`` cannot be combined."""
    parser = compare_models_module._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--prompt", "x",
            "--models", "m1",
            "--template", "a.json",
            "--templates", "system,b.json",
        ])


def test_template_singular_sets_templates_list(compare_models_module):
    """When only --template is passed, the CLI should behave as if
    --templates had that one value."""
    parser = compare_models_module._build_arg_parser()
    args = parser.parse_args([
        "--prompt", "x",
        "--models", "m1",
        "--template", "templates/chroma/x.json",
    ])
    tokens = compare_models_module._parse_template_tokens(args)
    assert tokens == ["templates/chroma/x.json"]


def test_templates_csv_is_split_and_stripped(compare_models_module):
    parser = compare_models_module._build_arg_parser()
    args = parser.parse_args([
        "--prompt", "x",
        "--models", "m1",
        "--templates", " system , templates/x.json  ,templates/y.json ",
    ])
    tokens = compare_models_module._parse_template_tokens(args)
    assert tokens == ["system", "templates/x.json", "templates/y.json"]


def test_template_flags_absent_gives_empty_tokens(compare_models_module):
    """Neither --template nor --templates → empty list (defaults to
    'system' downstream via _build_candidates)."""
    parser = compare_models_module._build_arg_parser()
    args = parser.parse_args([
        "--prompt", "x",
        "--models", "m1,m2",
    ])
    tokens = compare_models_module._parse_template_tokens(args)
    assert tokens == []

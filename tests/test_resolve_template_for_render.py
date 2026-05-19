"""Unit tests for the engine's template-resolution helpers.

These two helpers landed in the 2026-05-20 external-templates-only
cleanup and are NOT directly exercised by the older render_prompts
test suite (which mocks at a higher level). The post-impl verifier
flagged this as an I-2 gap — direct coverage protects the
family-extraction regex + the family-default fallback path.

Covers:
  - `_extract_family_from_template_path` regex (the family-match
    validator's input).
  - `PipelineEngine._resolve_template_for_render`:
    * explicit override returns the resolved path
    * no override + family-default present returns the default
    * no override + family-default null raises EngineError with the
      doc-quoted message
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.engine import (
    EngineError,
    PipelineEngine,
    _extract_family_from_template_path,
)


# ── _extract_family_from_template_path ─────────────────────────────


@pytest.mark.parametrize(
    "path,expected",
    [
        # Relative path under workflow_dir (typical YAML-default form).
        ("templates/chroma/gonzaLomo_Chroma_Refiner_v11.json", "chroma"),
        # Absolute path on macOS / Linux.
        ("/Users/x/repo/config/comfyui_workflows/templates/chroma/x.json", "chroma"),
        # Sub-directory under family is OK — first segment wins.
        ("templates/sdxl/sub/X.json", "sdxl"),
        # Underscored family id.
        ("templates/flux2/base.json", "flux2"),
        # Numeric / alphanumeric ids still match.
        ("templates/abc123/x.json", "abc123"),
        # Mixed-case path (macOS case-insensitive HFS+) is lowercased
        # so the validator never silently skips on a typo.
        ("templates/CHROMA/x.json", "chroma"),
        ("Templates/Chroma/x.json", "chroma"),
    ],
)
def test_extract_family_match(path, expected):
    assert _extract_family_from_template_path(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        # No `templates/<family>/` segment — outside the convention.
        "/tmp/scratch.json",
        "config/other/X.json",
        # `templates/` with nothing after — no family captured.
        "templates/",
        # Hyphenated family name — regex requires `[A-Za-z0-9_]+`, hyphen rejected.
        "templates/my-fam/X.json",
    ],
)
def test_extract_family_no_match(path):
    assert _extract_family_from_template_path(path) is None


# ── PipelineEngine._resolve_template_for_render ────────────────────


@pytest.fixture
def engine(tmp_path):
    """A bare PipelineEngine instance suitable for calling the
    resolver helper without booting LLM / DB."""
    # We need just the workflow_dir attribute + the resolver method
    # — bypass __init__'s heavy setup.
    e = object.__new__(PipelineEngine)
    e.workflow_dir = Path(__file__).resolve().parent.parent / "config" / "comfyui_workflows"
    return e


def test_resolve_explicit_override(engine):
    """Override path passes through `_resolve_template_path` unchanged
    when the file exists under workflow_dir."""
    out = engine._resolve_template_for_render(
        family_id="chroma",
        override="templates/chroma/chroma_done_properly.json",
    )
    assert out == "templates/chroma/chroma_done_properly.json"


def test_resolve_family_default_chroma(engine):
    """No override + chroma → the YAML-declared default path."""
    out = engine._resolve_template_for_render(
        family_id="chroma",
        override=None,
    )
    assert out == "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"


@pytest.mark.parametrize("family_id", ["sdxl", "pony", "illustrious", "flux", "flux2"])
def test_resolve_family_default_null_raises(engine, family_id):
    """Non-chroma families have `default_template: null` in
    families.yaml. Missing override → clean EngineError with the
    doc-quoted message naming the family + the families.yaml key
    + the --templates fallback."""
    with pytest.raises(EngineError) as excinfo:
        engine._resolve_template_for_render(
            family_id=family_id,
            override=None,
        )
    msg = str(excinfo.value)
    assert f"family {family_id!r}" in msg
    assert f"families.yaml::{family_id}::default_template" in msg
    assert "--templates" in msg

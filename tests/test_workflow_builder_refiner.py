"""Tests for the optional refiner-stage contract in
``WorkflowBuilder`` (added 2026-05-15).

The contract extends the existing 4-ID external-template contract
(``positive_prompt`` / ``negative_prompt`` / ``ksampler`` /
``empty_latent``) with 4 OPTIONAL refiner IDs:

- ``refiner_positive_prompt`` — patched (same text as base)
- ``refiner_negative_prompt`` — template-owned (NOT patched)
- ``refiner_ksampler`` — patched (same seed as base)
- ``refiner_checkpoint_loader`` — metadata-only (NOT patched)

Plus the **all-or-none pair rule**: if either
``refiner_positive_prompt`` or ``refiner_ksampler`` is present in the
template, both must be. Catches half-renamed templates.

Fixtures copy the user's gonzaLomo template + the canonical no-refiner
template into a tmp workflow_dir so tests stay pinned to real shapes.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.render.workflow_builder import (
    WorkflowBuilder,
    WorkflowTemplateError,
    _assert_external_template_inputs,
    _OPTIONAL_NODES_EXTERNAL,
    _REQUIRED_NODES_EXTERNAL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REFINER_TEMPLATE = (
    PROJECT_ROOT
    / "config"
    / "comfyui_workflows"
    / "templates"
    / "chroma"
    / "gonzaLomo_Chroma_Refiner_v11.json"
)
_NO_REFINER_TEMPLATE = (
    PROJECT_ROOT
    / "config"
    / "comfyui_workflows"
    / "templates"
    / "chroma"
    / "chroma_done_properly.json"
)


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    """A tmp workflow_dir seeded with both real templates."""
    wf_dir = tmp_path / "workflows"
    (wf_dir / "templates" / "chroma").mkdir(parents=True)
    shutil.copy(
        _REFINER_TEMPLATE,
        wf_dir / "templates" / "chroma" / "gonzaLomo_Chroma_Refiner_v11.json",
    )
    shutil.copy(
        _NO_REFINER_TEMPLATE,
        wf_dir / "templates" / "chroma" / "chroma_done_properly.json",
    )
    return wf_dir


@pytest.fixture
def refiner_workflow() -> dict[str, Any]:
    """Parsed gonzaLomo template — for tests that just inspect the shape."""
    with open(_REFINER_TEMPLATE) as f:
        return json.load(f)


@pytest.fixture
def no_refiner_workflow() -> dict[str, Any]:
    """Parsed chroma_done_properly template — backward-compat baseline."""
    with open(_NO_REFINER_TEMPLATE) as f:
        return json.load(f)


def _write_mutant(
    workflow_dir: Path, source: Path, name: str, mutator
) -> str:
    """Load `source`, apply mutator, write under templates/chroma/{name}
    and return the relative path (as `_load` accepts)."""
    with open(source) as f:
        data = json.load(f)
    mutator(data)
    out = workflow_dir / "templates" / "chroma" / name
    out.write_text(json.dumps(data))
    return f"templates/chroma/{name}"


# ── 1. Refiner-containing template passes preflight ────────────────


def test_refiner_template_passes_preflight(refiner_workflow):
    """The gonzaLomo template (post-rename) carries all 4 required IDs
    plus all 4 optional refiner IDs. Preflight must accept it cleanly."""
    WorkflowBuilder._assert_required_nodes(
        refiner_workflow,
        "external",
        "gonzaLomo_Chroma_Refiner_v11.json",
        _REQUIRED_NODES_EXTERNAL,
    )
    _assert_external_template_inputs(
        refiner_workflow, "gonzaLomo_Chroma_Refiner_v11.json"
    )
    # No exception → pass.


# ── 2. Missing refiner_ksampler.seed → PreflightError ──────────────


def test_refiner_template_missing_refiner_seed_rejected(refiner_workflow):
    """A refiner template that defines `refiner_ksampler` but strips
    its `inputs.seed` must fail preflight with a precise error pointing
    at the missing field."""
    bad = copy.deepcopy(refiner_workflow)
    del bad["refiner_ksampler"]["inputs"]["seed"]

    with pytest.raises(WorkflowTemplateError) as excinfo:
        _assert_external_template_inputs(bad, "broken_refiner.json")
    msg = str(excinfo.value)
    assert "refiner_ksampler.inputs.seed" in msg


# ── 3. build_external patches refiner_positive_prompt.text ─────────


def test_build_external_patches_refiner_positive_with_base_prompt(workflow_dir):
    """The refiner stage gets the SAME prompt_text as the base.
    Verifies the new conditional block in build_external."""
    wb = WorkflowBuilder(workflow_dir)
    workflow = wb.build_external(
        external_template=(
            "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"
        ),
        prompt_text="a portrait of an elegant woman, golden hour",
        negative_prompt="ugly, bad",
        resolution=(1024, 1536),
    )
    base_text = workflow["positive_prompt"]["inputs"]["text"]
    refiner_text = workflow["refiner_positive_prompt"]["inputs"]["text"]
    assert base_text == "a portrait of an elegant woman, golden hour"
    assert refiner_text == base_text  # same text, both encoders


# ── 4. build_external propagates the seed to refiner_ksampler ──────


def test_build_external_propagates_seed_to_refiner_ksampler(workflow_dir):
    """The refiner sampler gets the SAME seed as the base sampler
    (deterministic / reproducible end-to-end)."""
    wb = WorkflowBuilder(workflow_dir)
    workflow = wb.build_external(
        external_template=(
            "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"
        ),
        prompt_text="x",
        negative_prompt="",
        resolution=(1024, 1536),
        seed=12345,
    )
    assert workflow["ksampler"]["inputs"]["seed"] == 12345
    assert workflow["refiner_ksampler"]["inputs"]["seed"] == 12345


# ── 5. Non-refiner template regression guard ───────────────────────


def test_non_refiner_template_still_validates_and_patches(workflow_dir):
    """The existing 4-ID `chroma_done_properly.json` template (no
    refiner stage) must continue to work unchanged — the optional
    refiner loop is a silent no-op."""
    wb = WorkflowBuilder(workflow_dir)
    workflow = wb.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="anything",
        negative_prompt="nothing",
        resolution=(864, 1536),
        seed=42,
    )
    # All four required fields patched.
    assert workflow["positive_prompt"]["inputs"]["text"] == "anything"
    assert workflow["negative_prompt"]["inputs"]["text"] == "nothing"
    assert workflow["ksampler"]["inputs"]["seed"] == 42
    assert workflow["empty_latent"]["inputs"]["width"] == 864
    assert workflow["empty_latent"]["inputs"]["height"] == 1536
    # No refiner nodes introduced.
    for opt in _OPTIONAL_NODES_EXTERNAL:
        assert opt not in workflow, opt


# ── 6. Pair-rule rejects half-refiner templates ────────────────────


def test_pair_rule_rejects_refiner_positive_without_refiner_ksampler(
    no_refiner_workflow,
):
    """Half-refiner template: has `refiner_positive_prompt` but not
    `refiner_ksampler`. Catches a user who forgot to rename one half
    of the pair."""
    bad = copy.deepcopy(no_refiner_workflow)
    bad["refiner_positive_prompt"] = {
        "inputs": {"text": ""},
        "class_type": "CLIPTextEncode",
    }
    # No refiner_ksampler added.

    with pytest.raises(WorkflowTemplateError) as excinfo:
        _assert_external_template_inputs(bad, "half_refiner.json")
    msg = str(excinfo.value)
    assert "refiner_positive_prompt" in msg
    assert "refiner_ksampler" in msg
    assert "pair" in msg.lower()


def test_pair_rule_rejects_refiner_ksampler_without_refiner_positive(
    no_refiner_workflow,
):
    """Symmetric case: has `refiner_ksampler` but not
    `refiner_positive_prompt`. Should also fail."""
    bad = copy.deepcopy(no_refiner_workflow)
    bad["refiner_ksampler"] = {
        "inputs": {"seed": 0},
        "class_type": "KSampler",
    }
    # No refiner_positive_prompt added.

    with pytest.raises(WorkflowTemplateError) as excinfo:
        _assert_external_template_inputs(bad, "half_refiner.json")
    msg = str(excinfo.value)
    assert "refiner_ksampler" in msg
    assert "refiner_positive_prompt" in msg


# ── 7. Misnamed optional node silently ignored ─────────────────────


def test_misnamed_optional_node_ignored(no_refiner_workflow):
    """A node with a typo'd semantic ID (`refiner_pos_prompt` instead
    of `refiner_positive_prompt`) is treated as a regular numeric/named
    node — the optional contract is presence-based by exact key, so the
    misnamed node has no effect on validation. Render proceeds as
    a no-refiner template would (the typo'd node is just unused
    payload)."""
    bad = copy.deepcopy(no_refiner_workflow)
    bad["refiner_pos_prompt"] = {  # note the typo
        "inputs": {"text": ""},
        "class_type": "CLIPTextEncode",
    }
    # Must NOT raise — the typo'd ID isn't in the optional contract,
    # so the pair-rule sees no refiner_positive_prompt + no
    # refiner_ksampler and is satisfied.
    _assert_external_template_inputs(bad, "typo_template.json")


# ── 8. refiner_checkpoint_loader carries the refiner ckpt_name ─────


def test_refiner_checkpoint_loader_preserved_through_build(workflow_dir):
    """`refiner_checkpoint_loader` is metadata-only — the pipeline
    never writes to it. After `build_external`, its `inputs.ckpt_name`
    must match the template-author's original value."""
    wb = WorkflowBuilder(workflow_dir)
    workflow = wb.build_external(
        external_template=(
            "templates/chroma/gonzaLomo_Chroma_Refiner_v11.json"
        ),
        prompt_text="x",
        negative_prompt="",
        resolution=(1024, 1536),
        seed=42,
    )
    assert workflow["refiner_checkpoint_loader"]["inputs"]["ckpt_name"] == (
        "gonzalomoXLFluxPony_v60PhotoXLDMD.safetensors"
    )


# ── 9. refiner_negative_prompt left untouched even when present ────


def test_refiner_negative_not_patched(workflow_dir):
    """`refiner_negative_prompt` is fully template-owned — pipeline
    must not patch its `inputs.text` even when the template wires
    the node. Verifies by setting the template's value to a sentinel
    and confirming build_external leaves it alone."""
    sentinel = "TEMPLATE_OWNED_REFINER_NEGATIVE_SENTINEL"
    rel = _write_mutant(
        workflow_dir,
        _REFINER_TEMPLATE,
        "refiner_with_negative_text.json",
        lambda wf: wf["refiner_negative_prompt"]["inputs"].__setitem__(
            "text", sentinel
        ),
    )
    wb = WorkflowBuilder(workflow_dir)
    workflow = wb.build_external(
        external_template=rel,
        prompt_text="x",
        negative_prompt="this should NOT leak into refiner negative",
        resolution=(1024, 1536),
        seed=42,
    )
    assert workflow["refiner_negative_prompt"]["inputs"]["text"] == sentinel

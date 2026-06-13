"""External-template contract for ``WorkflowBuilder.build_external``.

User flow: paste a ComfyUI-UI-saved workflow (API format) under
``config/comfyui_workflows/templates/{family}/``, rename four nodes to
the semantic IDs ``positive_prompt`` / ``negative_prompt`` / ``ksampler``
/ ``empty_latent``, and run the pipeline against it. The template
ships with its own checkpoint, LoRAs, sampler, VAE, CLIP, post-
processing — the pipeline must NOT touch any of that. It injects
exactly four fields: prompt text, negative prompt text, seed, and
(width, height).

Fixtures copy the user's verified
``tests/fixtures/chroma_done_properly.json`` (a dedicated fixture)
into the tmp workflow_dir so tests stay pinned to the real canonical
shape. Negative-path variants are derived by mutating the loaded
dict before writing — no hand-rolled synthetic JSON.
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
    _REQUIRED_NODES_EXTERNAL,
    _resolve_template_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# A stable, dedicated TEST FIXTURE (a real ComfyUI API-format chroma graph)
# — lives under tests/fixtures/, NOT in the production templates/ dir, so the
# external-contract test isn't coupled to a live-tuned production template.
# (Relocated 2026-06-13 when the production workflows folder was pruned.)
_CANONICAL_TEMPLATE = PROJECT_ROOT / "tests" / "fixtures" / "chroma_done_properly.json"


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    """A tmp workflow_dir seeded with the real canonical template.

    We copy rather than point at the repo file directly so mutation
    tests can rewrite the template without polluting the repo copy.
    """
    wf_dir = tmp_path / "workflows"
    (wf_dir / "templates" / "chroma").mkdir(parents=True)
    shutil.copy(
        _CANONICAL_TEMPLATE,
        wf_dir / "templates" / "chroma" / "chroma_done_properly.json",
    )
    return wf_dir


@pytest.fixture
def canonical_workflow() -> dict[str, Any]:
    """Parsed canonical template — for tests that just inspect the shape."""
    with open(_CANONICAL_TEMPLATE) as f:
        return json.load(f)


def _write_mutant(
    workflow_dir: Path, name: str, mutator
) -> str:
    """Load the canonical template, apply ``mutator`` to the dict,
    write it under ``templates/chroma/{name}`` and return the relative
    path (as _load accepts)."""
    with open(_CANONICAL_TEMPLATE) as f:
        data = json.load(f)
    mutator(data)
    rel = f"templates/chroma/{name}"
    out = workflow_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    return rel


# ---------- canonical contract ------------------------------------------


def test_canonical_template_has_four_semantic_ids(canonical_workflow):
    """Guard against repo drift: the canonical template the docs cite
    must carry the 4 semantic IDs. If someone re-exports it and forgets
    to rename the nodes, catch it here before users hit it."""
    for node_id in _REQUIRED_NODES_EXTERNAL:
        assert node_id in canonical_workflow, (
            f"Canonical template is missing {node_id!r}. "
            f"Re-run the rename step described in "
            f"docs/COMFYUI_WORKFLOWS.md § External templates."
        )


def test_canonical_template_has_required_input_fields(canonical_workflow):
    assert "text" in canonical_workflow["positive_prompt"]["inputs"]
    assert "text" in canonical_workflow["negative_prompt"]["inputs"]
    assert "seed" in canonical_workflow["ksampler"]["inputs"]
    assert "width" in canonical_workflow["empty_latent"]["inputs"]
    assert "height" in canonical_workflow["empty_latent"]["inputs"]


# ---------- build_external: happy path ----------------------------------


def test_build_external_injects_four_fields(workflow_dir: Path):
    builder = WorkflowBuilder(workflow_dir)
    wf = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="a portrait of a woman",
        negative_prompt="low quality",
        resolution=(768, 1152),
        seed=4242,
    )
    assert wf["positive_prompt"]["inputs"]["text"] == "a portrait of a woman"
    assert wf["negative_prompt"]["inputs"]["text"] == "low quality"
    assert wf["ksampler"]["inputs"]["seed"] == 4242
    assert wf["empty_latent"]["inputs"]["width"] == 768
    assert wf["empty_latent"]["inputs"]["height"] == 1152


def test_build_external_generates_random_seed_when_none(workflow_dir: Path):
    builder = WorkflowBuilder(workflow_dir)
    wf = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="x", negative_prompt="y", resolution=(512, 512),
        seed=None,
    )
    seed = wf["ksampler"]["inputs"]["seed"]
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_build_external_leaves_checkpoint_untouched(
    workflow_dir: Path, canonical_workflow
):
    """The canonical template bakes in its own checkpoint via the
    numerically-keyed '731' UNETLoader. build_external must NOT rewrite
    it — the user's explicit directive is that the template's own
    checkpoint drives the render. This test pins identity to whatever
    the template currently declares; the user owns the checkpoint pick
    in `chroma_done_properly.json`."""
    builder = WorkflowBuilder(workflow_dir)
    wf = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="x", negative_prompt="y", resolution=(512, 512),
        seed=0,
    )
    original_ckpt = canonical_workflow["731"]["inputs"]["unet_name"]
    # Identity check — the build path didn't rewrite the field.
    assert wf["731"]["inputs"]["unet_name"] == original_ckpt
    # Sanity — the value is a non-empty .safetensors file name.
    assert original_ckpt.endswith(".safetensors")


def test_build_external_leaves_sampler_config_untouched(
    workflow_dir: Path, canonical_workflow
):
    """ksampler.inputs.seed is the ONLY field we inject. sampler_name,
    scheduler, cfg, steps, eta, bongmath, etc. are the template
    author's choice."""
    builder = WorkflowBuilder(workflow_dir)
    wf = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="x", negative_prompt="y", resolution=(512, 512),
        seed=7,
    )
    ks_in = wf["ksampler"]["inputs"]
    orig = canonical_workflow["ksampler"]["inputs"]
    for k in ("sampler_name", "scheduler", "cfg", "steps", "eta",
             "bongmath", "sampler_mode", "denoise"):
        if k in orig:
            assert ks_in[k] == orig[k], f"build_external mutated {k}"


def test_build_external_leaves_numeric_nodes_untouched(
    workflow_dir: Path, canonical_workflow
):
    """VAELoader ('730'), UNETLoader ('731'), CLIPLoader ('732'),
    VAEDecode ('397'), SaveImage ('398') — all baked in. Untouched."""
    builder = WorkflowBuilder(workflow_dir)
    wf = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="x", negative_prompt="y", resolution=(512, 512),
        seed=0,
    )
    for key in ("397", "398", "730", "731", "732"):
        assert wf[key] == canonical_workflow[key]


# ---------- build_external: cache / isolation ---------------------------


def test_build_external_cache_isolation(workflow_dir: Path):
    """Two calls with different seeds must produce two independent
    workflows — mutation of one cannot leak into the other. This is
    guaranteed by _load returning a deepcopy on every call, but pin
    the behavior so future refactors can't lose it."""
    builder = WorkflowBuilder(workflow_dir)
    wf1 = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="A", negative_prompt="x", resolution=(512, 512),
        seed=1,
    )
    wf2 = builder.build_external(
        external_template="templates/chroma/chroma_done_properly.json",
        prompt_text="B", negative_prompt="y", resolution=(640, 640),
        seed=2,
    )
    assert wf1["ksampler"]["inputs"]["seed"] == 1
    assert wf2["ksampler"]["inputs"]["seed"] == 2
    assert wf1["positive_prompt"]["inputs"]["text"] == "A"
    assert wf2["positive_prompt"]["inputs"]["text"] == "B"
    assert wf1["empty_latent"]["inputs"]["width"] == 512
    assert wf2["empty_latent"]["inputs"]["width"] == 640


# ---------- build_external: preflight errors ----------------------------


def test_build_external_missing_file(workflow_dir: Path):
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template="templates/chroma/does_not_exist.json",
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "not found" in str(exc.value).lower()


def test_build_external_malformed_json(workflow_dir: Path):
    bad = workflow_dir / "templates" / "chroma" / "broken.json"
    bad.write_text("{ this is not json")
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template="templates/chroma/broken.json",
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "not valid json" in str(exc.value).lower()


def test_build_external_missing_positive_prompt(workflow_dir: Path):
    rel = _write_mutant(
        workflow_dir, "no_positive.json",
        lambda d: d.pop("positive_prompt"),
    )
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "positive_prompt" in str(exc.value)
    assert "missing required semantic node IDs" in str(exc.value)


def test_build_external_missing_ksampler(workflow_dir: Path):
    rel = _write_mutant(
        workflow_dir, "no_ksampler.json",
        lambda d: d.pop("ksampler"),
    )
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "ksampler" in str(exc.value)


def test_build_external_missing_empty_latent(workflow_dir: Path):
    rel = _write_mutant(
        workflow_dir, "no_empty_latent.json",
        lambda d: d.pop("empty_latent"),
    )
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "empty_latent" in str(exc.value)


def test_build_external_missing_negative_prompt(workflow_dir: Path):
    rel = _write_mutant(
        workflow_dir, "no_negative.json",
        lambda d: d.pop("negative_prompt"),
    )
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "negative_prompt" in str(exc.value)


def test_build_external_empty_latent_missing_width(workflow_dir: Path):
    """Top-level key present but required input field missing."""
    def mutator(d):
        del d["empty_latent"]["inputs"]["width"]

    rel = _write_mutant(workflow_dir, "no_width.json", mutator)
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "empty_latent.inputs.width" in str(exc.value)
    assert "missing input fields" in str(exc.value)


def test_build_external_empty_latent_missing_height(workflow_dir: Path):
    def mutator(d):
        del d["empty_latent"]["inputs"]["height"]

    rel = _write_mutant(workflow_dir, "no_height.json", mutator)
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "empty_latent.inputs.height" in str(exc.value)


def test_build_external_ksampler_missing_seed_input(workflow_dir: Path):
    def mutator(d):
        del d["ksampler"]["inputs"]["seed"]

    rel = _write_mutant(workflow_dir, "no_seed.json", mutator)
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "ksampler.inputs.seed" in str(exc.value)


def test_build_external_positive_prompt_missing_text_input(workflow_dir: Path):
    def mutator(d):
        del d["positive_prompt"]["inputs"]["text"]

    rel = _write_mutant(workflow_dir, "no_text.json", mutator)
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "positive_prompt.inputs.text" in str(exc.value)


def test_build_external_inputs_not_a_dict(workflow_dir: Path):
    """A template where ``ksampler.inputs`` isn't an object is a
    structural violation — the API-format exporter would never produce
    this, but a hand-edited template could. We must not crash with a
    raw TypeError; we must emit WorkflowTemplateError."""
    def mutator(d):
        d["ksampler"]["inputs"] = ["not", "an", "object"]

    rel = _write_mutant(workflow_dir, "bad_inputs.json", mutator)
    builder = WorkflowBuilder(workflow_dir)
    with pytest.raises(WorkflowTemplateError) as exc:
        builder.build_external(
            external_template=rel,
            prompt_text="x", negative_prompt="y", resolution=(512, 512),
        )
    assert "ksampler.inputs" in str(exc.value)


# ---------- _resolve_template_path --------------------------------------


def test_resolve_template_path_absolute(workflow_dir: Path):
    target = workflow_dir / "templates" / "chroma" / "chroma_done_properly.json"
    out = _resolve_template_path(
        str(target), workflow_dir, PROJECT_ROOT,
    )
    assert out == str(target)


def test_resolve_template_path_relative_under_workflow_dir(workflow_dir: Path):
    out = _resolve_template_path(
        "templates/chroma/chroma_done_properly.json",
        workflow_dir, PROJECT_ROOT,
    )
    assert out == "templates/chroma/chroma_done_properly.json"


def test_resolve_template_path_nonexistent_passes_through(workflow_dir: Path):
    """When neither workflow_dir nor project_root holds the file, return
    the raw value so _load raises its normal not-found error — don't
    swallow the error here."""
    out = _resolve_template_path(
        "templates/chroma/does_not_exist.json",
        workflow_dir, PROJECT_ROOT,
    )
    assert out == "templates/chroma/does_not_exist.json"


# ---------- _assert_external_template_inputs ----------------------------
# Directly exercise the helper so preflight callers (engine) can reuse
# it without going through the full build_external path.


def test_assert_external_template_inputs_accepts_canonical(canonical_workflow):
    _assert_external_template_inputs(canonical_workflow, "chroma_done_properly.json")


def test_assert_external_template_inputs_rejects_missing_width():
    wf = {
        "positive_prompt": {"inputs": {"text": "x"}},
        "negative_prompt": {"inputs": {"text": "y"}},
        "ksampler": {"inputs": {"seed": 0}},
        "empty_latent": {"inputs": {"height": 512}},  # no width
    }
    with pytest.raises(WorkflowTemplateError) as exc:
        _assert_external_template_inputs(wf, "t.json")
    assert "empty_latent.inputs.width" in str(exc.value)

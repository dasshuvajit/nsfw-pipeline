"""Phase F — upscale workflow template + Upscaler family-aware tests.

Each templated family (sdxl/pony/illustrious) ships a pure-ESRGAN
upscale workflow at ``config/comfyui_workflows/{family}/upscale.json``.
The chain is deterministic (no prompt encoder, no sampler) so the
tests can be entirely shape-and-topology checks — no ComfyUI required.

Tests cover:

  * Each templated family file loads as JSON and has all 5 expected
    nodes (LoadImage, UpscaleModelLoader, ImageUpscaleWithModel,
    ImageScaleBy, SaveImage) wired correctly.
  * The downscale factor lands the output at ~1.4× the source (4× model
    output × 0.35 scale_by).
  * Upscaler raises eagerly for unsupported families (flux / chroma /
    flux2) so misconfig is caught before any render begins.
  * Upscaler raises eagerly when the template file is missing.
  * Upscaler accepts the three templated families without errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.postprocess.upscaler import (
    Upscaler,
    UpscaleError,
    _REQUIRED_NODES,
    _SUPPORTED_FAMILIES,
)


WORKFLOW_DIR = (
    Path(__file__).resolve().parent.parent / "config" / "comfyui_workflows"
)


# ── Template shape ──────────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(_SUPPORTED_FAMILIES))
class TestTemplateShape:
    def test_template_file_exists(self, family: str) -> None:
        assert (WORKFLOW_DIR / family / "upscale.json").exists()

    def test_template_loads_as_json(self, family: str) -> None:
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data, f"{family}/upscale.json is empty"

    def test_template_has_required_semantic_nodes(self, family: str) -> None:
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        for node in _REQUIRED_NODES:
            assert node in data, (
                f"{family}/upscale.json missing required semantic node {node!r}"
            )

    def test_load_image_class_is_LoadImage(self, family: str) -> None:
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        assert data["load_image"]["class_type"] == "LoadImage"

    def test_upscale_model_loader_is_UpscaleModelLoader(self, family: str) -> None:
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        assert data["upscale_model_loader"]["class_type"] == "UpscaleModelLoader"
        assert "model_name" in data["upscale_model_loader"]["inputs"]

    def test_save_image_class_is_SaveImage(self, family: str) -> None:
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        assert data["save_image"]["class_type"] == "SaveImage"
        assert "filename_prefix" in data["save_image"]["inputs"]

    def test_topology_references_resolve(self, family: str) -> None:
        # Every input that's a [node_id, output_idx] reference must point
        # to a real node in the same template — catches a copy-paste
        # rename that would crash mid-render.
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        ids = set(data.keys())
        for node_id, node in data.items():
            for input_name, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in ids, (
                        f"{family}/upscale.json: node {node_id!r} input "
                        f"{input_name!r} references missing node {value[0]!r}"
                    )

    def test_downscale_lands_at_1_4x(self, family: str) -> None:
        # 4× ESRGAN output × 0.35 scale_by ≈ 1.4× source.
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        scale_node = data.get("downscale_to_target")
        assert scale_node is not None
        assert scale_node["class_type"] == "ImageScaleBy"
        assert abs(scale_node["inputs"]["scale_by"] - 0.35) < 0.01

    def test_filename_prefix_is_family_specific(self, family: str) -> None:
        # The default prefix encodes the family so output filenames stay
        # browseable by family without round-tripping through the DB.
        with open(WORKFLOW_DIR / family / "upscale.json") as f:
            data = json.load(f)
        prefix = data["save_image"]["inputs"]["filename_prefix"]
        assert family in prefix.lower()


# ── Untemplated families ─────────────────────────────────────────────


@pytest.mark.parametrize("family", ["flux", "chroma", "flux2"])
def test_untemplated_families_have_no_upscale_json(family: str) -> None:
    # Documenting decision: these families render natively at 1024+ and
    # don't ship an upscale template. If someone adds one later, the
    # Upscaler's _SUPPORTED_FAMILIES allowlist must be updated too.
    assert not (WORKFLOW_DIR / family / "upscale.json").exists()


# ── Upscaler eager validation ───────────────────────────────────────


class _StubComfyClient:
    """Minimal ComfyUIClient stub — Upscaler() init never calls it."""


@pytest.mark.parametrize("family", sorted(_SUPPORTED_FAMILIES))
def test_upscaler_constructs_for_supported_family(family: str, tmp_path: Path) -> None:
    """All three templated families construct cleanly with the real templates."""
    upscaler = Upscaler(
        WORKFLOW_DIR,
        _StubComfyClient(),
        tmp_path,  # unused at construction
        family_id=family,
    )
    assert upscaler.family_id == family
    assert upscaler.template_path == WORKFLOW_DIR / family / "upscale.json"


@pytest.mark.parametrize("family", ["flux", "chroma", "flux2", "unknown"])
def test_upscaler_rejects_unsupported_family(family: str, tmp_path: Path) -> None:
    """Untemplated / unknown families fail at construction, not mid-batch."""
    with pytest.raises(UpscaleError, match=f"family {family!r}"):
        Upscaler(
            WORKFLOW_DIR,
            _StubComfyClient(),
            tmp_path,
            family_id=family,
        )


def test_upscaler_rejects_missing_template(tmp_path: Path) -> None:
    """Wrong workflow_dir → eager error on missing template file."""
    with pytest.raises(UpscaleError, match="not found"):
        Upscaler(
            tmp_path,  # empty dir — no templates inside
            _StubComfyClient(),
            tmp_path,
            family_id="sdxl",
        )


def test_upscaler_load_template_validates_required_nodes(tmp_path: Path) -> None:
    """A template missing one of the required semantic IDs should fail."""
    fam_dir = tmp_path / "sdxl"
    fam_dir.mkdir()
    # Write a syntactically valid JSON missing the upscale_model_loader node.
    bad = {
        "load_image": {"inputs": {}, "class_type": "LoadImage"},
        "save_image": {"inputs": {"filename_prefix": "x"}, "class_type": "SaveImage"},
    }
    (fam_dir / "upscale.json").write_text(json.dumps(bad))
    upscaler = Upscaler(
        tmp_path,
        _StubComfyClient(),
        tmp_path,
        family_id="sdxl",
    )
    # Construction passes (file exists). Loading raises.
    with pytest.raises(UpscaleError, match="missing required semantic"):
        upscaler._load_template()

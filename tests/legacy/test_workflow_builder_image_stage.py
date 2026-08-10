"""Image-stage contract for ``WorkflowBuilder.build_image_stage``.

Staged rendering: a base render is decoded to a PNG, then a later stage
(SDXL refine, or 4K upscale + detailers) loads that PNG via a
``load_image`` (VHS_LoadImagePath) node. These templates have no
``empty_latent`` / prompt nodes, so they use ``build_image_stage`` (not
``build_external``). This pins: the load_image path + optional
seed/upscale_by/largest_size patching, the required-node preflight, the
optional-node tolerance, and the absolute-existing-path guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.render.workflow_builder import WorkflowBuilder, WorkflowTemplateError


def _img_stage_template() -> dict:
    """A minimal valid image-stage template: load_image + optional
    stage_ksampler + upscale + prelift + save."""
    return {
        "load_image": {"inputs": {"image": ""}, "class_type": "VHS_LoadImagePath"},
        "stage_ksampler": {
            "inputs": {"seed": 0, "steps": 6, "denoise": 0.2,
                       "sampler_name": "lcm", "scheduler": "karras",
                       "latent_image": ["load_image", 0]},
            "class_type": "KSampler",
        },
        "prelift": {"inputs": {"largest_size": 2048, "image": ["load_image", 0]},
                    "class_type": "ImageScaleToMaxDimension"},
        "upscale": {"inputs": {"upscale_by": 2.0, "seed": 0, "image": ["prelift", 0]},
                    "class_type": "UltimateSDUpscale"},
        "save": {"inputs": {"images": ["upscale", 0]}, "class_type": "SaveImage"},
    }


@pytest.fixture
def stage_dir(tmp_path: Path) -> Path:
    d = tmp_path / "workflows" / "templates" / "sdxl"
    d.mkdir(parents=True)
    with open(d / "stage.json", "w") as f:
        json.dump(_img_stage_template(), f)
    return tmp_path / "workflows"


@pytest.fixture
def real_image(tmp_path: Path) -> str:
    p = tmp_path / "base.png"
    Image.new("RGB", (64, 96), (120, 90, 70)).save(p)
    return str(p)


def _write(stage_dir: Path, name: str, wf: dict) -> None:
    with open(stage_dir / "templates" / "sdxl" / name, "w") as f:
        json.dump(wf, f)


def test_patches_path_seed_upscale_largest(stage_dir, real_image):
    b = WorkflowBuilder(stage_dir)
    wf = b.build_image_stage(
        external_template="templates/sdxl/stage.json", image_path=real_image,
        seed=4242, upscale_by=2.67, largest_size=3840,
    )
    assert wf["load_image"]["inputs"]["image"] == str(Path(real_image))
    assert wf["stage_ksampler"]["inputs"]["seed"] == 4242
    assert wf["upscale"]["inputs"]["seed"] == 4242          # USDU seed patched too
    assert wf["upscale"]["inputs"]["upscale_by"] == 2.67
    assert wf["prelift"]["inputs"]["largest_size"] == 3840
    # other inputs untouched
    assert wf["stage_ksampler"]["inputs"]["steps"] == 6


def test_random_seed_when_none(stage_dir, real_image):
    b = WorkflowBuilder(stage_dir)
    a = b.build_image_stage(external_template="templates/sdxl/stage.json", image_path=real_image)
    c = b.build_image_stage(external_template="templates/sdxl/stage.json", image_path=real_image)
    # both valid uint32 seeds; cache isolation means patching one didn't mutate the other
    assert 0 <= a["stage_ksampler"]["inputs"]["seed"] <= 2**32 - 1
    assert 0 <= c["stage_ksampler"]["inputs"]["seed"] <= 2**32 - 1


def test_relative_path_rejected(stage_dir):
    b = WorkflowBuilder(stage_dir)
    with pytest.raises(WorkflowTemplateError, match="absolute"):
        b.build_image_stage(external_template="templates/sdxl/stage.json",
                            image_path="relative/base.png")


def test_nonexistent_path_rejected(stage_dir):
    b = WorkflowBuilder(stage_dir)
    with pytest.raises(WorkflowTemplateError, match="does not exist"):
        b.build_image_stage(external_template="templates/sdxl/stage.json",
                            image_path="/tmp/definitely_not_here_8810.png")


def test_missing_load_image_rejected(stage_dir, real_image):
    bad = _img_stage_template()
    del bad["load_image"]
    _write(stage_dir, "bad.json", bad)
    b = WorkflowBuilder(stage_dir)
    with pytest.raises(WorkflowTemplateError, match="load_image"):
        b.build_image_stage(external_template="templates/sdxl/bad.json", image_path=real_image)


def test_missing_save_rejected(stage_dir, real_image):
    bad = _img_stage_template()
    del bad["save"]
    _write(stage_dir, "bad.json", bad)
    b = WorkflowBuilder(stage_dir)
    with pytest.raises(WorkflowTemplateError, match="save"):
        b.build_image_stage(external_template="templates/sdxl/bad.json", image_path=real_image)


def test_load_image_missing_field_rejected(stage_dir, real_image):
    bad = _img_stage_template()
    del bad["load_image"]["inputs"]["image"]
    _write(stage_dir, "bad.json", bad)
    b = WorkflowBuilder(stage_dir)
    with pytest.raises(WorkflowTemplateError, match="load_image.inputs.image"):
        b.build_image_stage(external_template="templates/sdxl/bad.json", image_path=real_image)


def test_inputs_not_a_dict_is_clean_error(stage_dir, real_image):
    bad = _img_stage_template()
    bad["load_image"]["inputs"] = "oops"
    _write(stage_dir, "bad.json", bad)
    b = WorkflowBuilder(stage_dir)
    with pytest.raises(WorkflowTemplateError):  # not a raw TypeError
        b.build_image_stage(external_template="templates/sdxl/bad.json", image_path=real_image)


def test_optional_nodes_absent_is_fine(stage_dir, real_image):
    """A stage with only load_image + save (no ksampler/upscale/prelift)
    validates and patches just the path."""
    minimal = {
        "load_image": {"inputs": {"image": ""}, "class_type": "VHS_LoadImagePath"},
        "save": {"inputs": {"images": ["load_image", 0]}, "class_type": "SaveImage"},
    }
    _write(stage_dir, "min.json", minimal)
    b = WorkflowBuilder(stage_dir)
    wf = b.build_image_stage(external_template="templates/sdxl/min.json",
                            image_path=real_image, upscale_by=2.0)
    assert wf["load_image"]["inputs"]["image"] == str(Path(real_image))
    assert "upscale" not in wf  # upscale_by silently ignored when no upscale node

"""Integrity of the staged production templates (base / refine / 4K).

The monolith `gonzaLomo_Chroma_4K_v12.json` is split by model domain into
three stage templates (+ a T4 variant). These tests pin: MPS-safe
samplers/schedulers (no RES4LYF res_* which crash on Apple MPS), acyclic
graphs with a single `save` sink, model-domain purity (base has no SDXL
nodes), the staged contracts (base = build_external 4-field; refine/4K =
build_image_stage load_image+save), and the research-backed value fixes
(refine lcm/karras/6/denoise 0.20; hands detailer 0.35).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TPL = PROJECT_ROOT / "config" / "comfyui_workflows" / "templates"
BASE = TPL / "chroma" / "base.json"
REFINE = TPL / "chroma" / "refine.json"
UPSCALE = TPL / "sdxl" / "upscale_4k.json"
UPSCALE_T4 = TPL / "sdxl" / "upscale_4k_T4.json"

_MPS_BANNED_SAMPLERS = {"res_2m", "res_2s", "res_3m", "res_multistep", "res_2m_sde"}
_MPS_BANNED_SCHEDULERS = {"bong_tangent"}


def _load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _acyclic_single_sink(wf: dict) -> tuple[bool, list[str]]:
    ids = set(wf)
    for n, nd in wf.items():
        for k, v in nd.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f"{n}.{k} -> missing {v[0]}"
    deps = {n: {v[0] for v in nd.get("inputs", {}).values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in ids}
            for n, nd in wf.items()}
    indeg = {n: len(deps[n]) for n in ids}
    queue = [n for n in ids if indeg[n] == 0]
    seen = 0
    while queue:
        x = queue.pop()
        seen += 1
        for m in ids:
            if x in deps[m]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
    sinks = [n for n, nd in wf.items() if nd["class_type"] == "SaveImage"]
    return seen == len(ids), sinks


@pytest.mark.parametrize("path", [BASE, REFINE, UPSCALE, UPSCALE_T4])
def test_exists(path: Path):
    assert path.exists(), f"missing stage template {path}"


@pytest.mark.parametrize("path", [BASE, REFINE, UPSCALE, UPSCALE_T4])
def test_mps_safe_and_acyclic_single_sink(path: Path):
    wf = _load(path)
    for nid, node in wf.items():
        ins = node.get("inputs", {})
        if "sampler_name" in ins:
            assert ins["sampler_name"] not in _MPS_BANNED_SAMPLERS, f"{path.name}:{nid}"
        if "scheduler" in ins:
            assert ins["scheduler"] not in _MPS_BANNED_SCHEDULERS, f"{path.name}:{nid}"
    acyclic, sinks = _acyclic_single_sink(wf)
    assert acyclic, f"{path.name} has a cycle"
    assert sinks == ["save"], f"{path.name} sink must be the single id 'save', got {sinks}"


def test_base_is_chroma_domain_only():
    wf = _load(BASE)
    # build_external 4-field contract present
    for nid in ("positive_prompt", "negative_prompt", "ksampler", "empty_latent"):
        assert nid in wf, f"base.json missing contract node {nid}"
    # no SDXL-domain nodes leaked in
    classes = {nd["class_type"] for nd in wf.values()}
    assert "CheckpointLoaderSimple" not in classes
    assert "UltimateSDUpscale" not in classes
    assert "FaceDetailer" not in classes


@pytest.mark.parametrize("path", [REFINE, UPSCALE, UPSCALE_T4])
def test_load_image_has_required_widget_inputs(path: Path):
    """VHS_LoadImagePath requires image + custom_width + custom_height in the
    API JSON (ComfyUI rejects the prompt otherwise — regression guard)."""
    li = _load(path)["load_image"]["inputs"]
    assert "image" in li
    assert li.get("custom_width") == 0 and li.get("custom_height") == 0


def test_refine_contract_and_values():
    wf = _load(REFINE)
    assert wf["load_image"]["class_type"] == "VHS_LoadImagePath"
    assert "empty_latent" not in wf            # image-input stage, not a generator
    assert "UltimateSDUpscale" not in {nd["class_type"] for nd in wf.values()}
    sk = wf["stage_ksampler"]["inputs"]
    assert sk["sampler_name"] == "lcm" and sk["scheduler"] == "karras"
    assert sk["steps"] == 6                    # value fix (was 4)
    assert sk["denoise"] == 0.20               # value fix (was 0.15 ≈ noop)
    # light face+eyes detailer makes review images anatomy-sharp for selection
    assert wf["detailer_face"]["inputs"]["image"] == ["133", 0]
    assert wf["detailer_eyes"]["inputs"]["image"] == ["detailer_face", 0]
    assert wf["detailer_face"]["inputs"]["denoise"] == 0.15   # light (4K does the heavy pass)
    assert wf["save"]["inputs"]["images"] == ["detailer_eyes", 0]


def test_upscale_contract_and_values():
    wf = _load(UPSCALE)
    assert wf["load_image"]["class_type"] == "VHS_LoadImagePath"
    u = wf["upscale"]
    assert u["class_type"] == "UltimateSDUpscale"
    assert u["inputs"]["sampler_name"] == "lcm" and u["inputs"]["scheduler"] == "karras"
    assert u["inputs"]["steps"] == 6
    assert u["inputs"]["denoise"] == 0.18
    # tile 1536 (6 tiles for a ~3000x3848 4K): the staged 4K stage is SDXL-only
    # with ~40GB free, so larger tiles than the monolith's 1280 are memory-safe
    # and ~halve the tile count.
    assert u["inputs"]["tile_width"] == 1536
    assert u["inputs"]["seam_fix_mode"] == "Half Tile"
    assert wf["skin_upscale_model"]["inputs"]["model_name"] == "4x_foolhardy_Remacri.pth"
    # detail-after-upscale: face detailer reads the upscale output
    assert wf["detailer_face"]["inputs"]["image"] == ["upscale", 0]
    assert wf["detailer_hands"]["inputs"]["denoise"] == 0.35   # value fix (was 0.40)


def test_t4_appends_nsfw_detailers():
    wf = _load(UPSCALE_T4)
    assert wf["det_nipple_detector"]["inputs"]["model_name"] == "bbox/nipples_yolov8s.pt"
    assert wf["det_vagina_detector"]["inputs"]["model_name"] == "bbox/vagina-v3.2.pt"
    # chained after the SFW detailers, sink reads the last NSFW detailer
    assert wf["detailer_nipples"]["inputs"]["image"] == ["detailer_hands", 0]
    assert wf["detailer_vagina"]["inputs"]["image"] == ["detailer_nipples", 0]
    assert wf["save"]["inputs"]["images"] == ["detailer_vagina", 0]

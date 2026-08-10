"""Integrity of the production render templates (zimage base / base_hires).

The pipeline is BASE-ONLY: one Z-Image Turbo render per (prompt x seed).
(The chroma / SDXL-refine / 4K-upscale templates and their integrity tests
were archived 2026-08 — see legacy/config_templates_chroma,
legacy/config_templates_pruned and tests/legacy.)

These tests pin: MPS-safe samplers/schedulers (no RES4LYF res_* which crash
on Apple MPS), acyclic graphs with a single `save` sink, and the
build_external 4-field contract the render stage relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TPL = PROJECT_ROOT / "config" / "comfyui_workflows" / "templates"
BASE = TPL / "zimage" / "base.json"
BASE_HIRES = TPL / "zimage" / "base_hires.json"

# RES4LYF res_* family — float64 path crashes on MPS. The zimage production
# sampler is dpmpp_sde/beta (a core ComfyUI sampler, MPS-safe).
_MPS_BANNED_SAMPLERS = {"res_2m", "res_2s", "res_3m", "res_2m_sde"}
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


@pytest.mark.parametrize("path", [BASE, BASE_HIRES])
def test_exists(path: Path):
    assert path.exists(), f"missing stage template {path}"


@pytest.mark.parametrize("path", [BASE, BASE_HIRES])
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


@pytest.mark.parametrize("path", [BASE, BASE_HIRES])
def test_base_contract_and_zimage_domain(path: Path):
    """build_external 4-field contract present; the graph stays in the
    Z-Image domain (no SDXL checkpoint / upscaler / detailer nodes)."""
    wf = _load(path)
    for nid in ("positive_prompt", "negative_prompt", "ksampler", "empty_latent"):
        assert nid in wf, f"{path.name} missing contract node {nid}"
    classes = {nd["class_type"] for nd in wf.values()}
    assert "CheckpointLoaderSimple" not in classes
    assert "UltimateSDUpscale" not in classes
    assert "FaceDetailer" not in classes
    # cfg-1 base: the negative branch routes through ConditioningZeroOut (inert)
    assert wf["ksampler"]["inputs"]["cfg"] == 1
    assert "neg_zero" in wf


def test_hires_variant_is_base_plus_deepshrink():
    """base_hires == base + PatchModelAddDownscale (deep-shrink) at a higher
    starting resolution; the LoRA stack / TE / VAE / sampler stay identical."""
    base, hires = _load(BASE), _load(BASE_HIRES)
    assert hires["deepshrink"]["class_type"] == "PatchModelAddDownscale"
    for nid in ("unet", "clip", "vae", "lora_dpo", "lora_nsfw", "lora_style"):
        assert hires[nid]["inputs"] == base[nid]["inputs"], \
            f"base_hires drifted from base at node {nid}"
    for key in ("steps", "cfg", "sampler_name", "scheduler"):
        assert hires["ksampler"]["inputs"][key] == base["ksampler"]["inputs"][key]

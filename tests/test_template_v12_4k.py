"""Production-template integrity for ``gonzaLomo_Chroma_4K_v12.json``.

v12 is the default ``art_series`` template: gonzaLomo Chroma base ->
1.25x lanczos -> SDXL DMD low-denoise refine -> UltimateSDUpscale true-4K
-> face/eyes/hands detailers -> SaveImage. Every SDXL pass uses one
template-owned generic detail prompt, and the base Chroma prose is never
re-encoded through SDXL's 77-token CLIP. (The embedded-refiner contract
that an earlier two-stage template used — ``refiner_positive_prompt`` /
``refiner_ksampler`` — was retired 2026-06-13; ``build_external`` no
longer recognises those IDs at all.)

These tests pin: (1) build_external patches only the four contracted
fields and injects no refiner nodes; (2) every sampler/scheduler is
MPS-safe (no RES4LYF res_* samplers, which crash on Apple MPS); (3) the
graph is acyclic with SaveImage as the sink; (4) the resolution math
yields a true-4K (>=3840px) long edge for the portrait base preset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.render.workflow_builder import WorkflowBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_V12 = (
    PROJECT_ROOT
    / "config" / "comfyui_workflows" / "templates" / "chroma"
    / "gonzaLomo_Chroma_4K_v12.json"
)

# RES4LYF / experimental samplers that force float64 and crash on Apple MPS.
_MPS_BANNED_SAMPLERS = {"res_2m", "res_2s", "res_3m", "res_multistep", "res_2m_sde"}
_MPS_BANNED_SCHEDULERS = {"bong_tangent"}


@pytest.fixture
def v12() -> dict:
    with open(_V12) as f:
        return json.load(f)


@pytest.fixture
def v12_dir(tmp_path: Path) -> Path:
    """tmp workflow_dir seeded with the real v12 file (copy, not pointer)."""
    d = tmp_path / "workflows" / "templates" / "chroma"
    d.mkdir(parents=True)
    shutil.copy(_V12, d / "gonzaLomo_Chroma_4K_v12.json")
    return tmp_path / "workflows"


def test_v12_build_external_patches_four_fields_and_omits_pair(v12_dir: Path) -> None:
    b = WorkflowBuilder(v12_dir)
    prose = "A long Chroma prose prompt. " + "word " * 140
    wf = b.build_external(
        external_template="templates/chroma/gonzaLomo_Chroma_4K_v12.json",
        prompt_text=prose, negative_prompt="NEG", resolution=(1024, 1536), seed=4242,
    )
    assert wf["positive_prompt"]["inputs"]["text"] == prose
    assert wf["negative_prompt"]["inputs"]["text"] == "NEG"
    assert wf["ksampler"]["inputs"]["seed"] == 4242
    assert wf["empty_latent"]["inputs"]["width"] == 1024
    assert wf["empty_latent"]["inputs"]["height"] == 1536
    # build_external introduces no embedded-refiner nodes (contract retired).
    assert "refiner_positive_prompt" not in wf
    assert "refiner_ksampler" not in wf
    # The SDXL passes keep their own template-owned seeds (NOT patched).
    assert wf["refiner_ksampler_local"]["inputs"]["seed"] == 12345
    assert wf["ultimate_upscale"]["inputs"]["seed"] == 12346


def test_v12_all_samplers_mps_safe(v12: dict) -> None:
    for nid, node in v12.items():
        ins = node.get("inputs", {})
        if "sampler_name" in ins:
            assert ins["sampler_name"] not in _MPS_BANNED_SAMPLERS, f"{nid} uses MPS-dead sampler"
        if "scheduler" in ins:
            assert ins["scheduler"] not in _MPS_BANNED_SCHEDULERS, f"{nid} uses MPS-dead scheduler"


def test_v12_graph_acyclic_with_saveimage_sink(v12: dict) -> None:
    ids = set(v12)
    # Dedup edges: a node depending on an upstream via several inputs
    # (model+clip+vae from one loader) is ONE edge, not three.
    deps = {
        n: {v[0] for v in node.get("inputs", {}).values()
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in ids}
        for n, node in v12.items()
    }
    # every reference resolves
    for n, node in v12.items():
        for k, v in node.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f"{n}.{k} -> missing {v[0]}"
    # acyclic (Kahn)
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
    assert seen == len(ids), "graph has a cycle"
    saves = [n for n, nd in v12.items() if nd["class_type"] == "SaveImage"]
    assert len(saves) == 1
    assert v12[saves[0]]["inputs"]["images"][0] == "detailer_hands"


def test_v12_portrait_base_yields_true_4k() -> None:
    # base portrait * ImageScaleBy(1.25) * UltimateSDUpscale(upscale_by)
    with open(_V12) as f:
        wf = json.load(f)
    scale_by = wf["143"]["inputs"]["scale_by"]
    upscale_by = wf["ultimate_upscale"]["inputs"]["upscale_by"]
    base_long_edge = 1536  # art_series ORIENTATIONS["portrait"] = (1024, 1536)
    final_long_edge = base_long_edge * scale_by * upscale_by
    assert final_long_edge >= 3840, f"portrait final long edge {final_long_edge} < 4K"

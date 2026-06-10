"""Integrity of the staged production templates (base / refine / 4K).

The monolith `gonzaLomo_Chroma_4K_v12.json` is split by model domain into
three stage templates (+ a T4 variant). These tests pin: MPS-safe
samplers/schedulers (no RES4LYF res_* which crash on Apple MPS), acyclic
graphs with a single `save` sink, model-domain purity (base has no SDXL
nodes), the staged contracts (base = build_external 4-field; refine/4K =
build_image_stage load_image+save), and the Chroma-face-preservation contract
(the refine stage runs NO SDXL on the face — no global img2img refine and no
face detailer; only targeted hand / foot / nipple detailer crops on the
upscaled Chroma base).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TPL = PROJECT_ROOT / "config" / "comfyui_workflows" / "templates"
BASE = TPL / "chroma" / "base.json"
REFINE = TPL / "chroma" / "refine.json"
REFINE_T4 = TPL / "chroma" / "refine_T4.json"
UPSCALE = TPL / "sdxl" / "upscale_4k.json"
# (upscale_4k_T4.json was deleted 2026-06-10: it had become byte-identical to
#  upscale_4k.json after the MPS post-upscale-detailer removal — 4K is tier-
#  neutral; explicit detail comes from the REVIEW stage.)

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


@pytest.mark.parametrize("path", [BASE, REFINE, REFINE_T4, UPSCALE])
def test_exists(path: Path):
    assert path.exists(), f"missing stage template {path}"


@pytest.mark.parametrize("path", [BASE, REFINE, REFINE_T4, UPSCALE])
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


@pytest.mark.parametrize("path", [REFINE, REFINE_T4, UPSCALE])
def test_load_image_has_required_widget_inputs(path: Path):
    """VHS_LoadImagePath requires image + custom_width + custom_height in the
    API JSON (ComfyUI rejects the prompt otherwise — regression guard)."""
    li = _load(path)["load_image"]["inputs"]
    assert "image" in li
    assert li.get("custom_width") == 0 and li.get("custom_height") == 0


def test_refine_contract_and_values():
    wf = _load(REFINE)
    classes = {nd["class_type"] for nd in wf.values()}
    assert wf["load_image"]["class_type"] == "VHS_LoadImagePath"
    assert "empty_latent" not in wf            # image-input stage, not a generator
    assert "UltimateSDUpscale" not in classes
    # refiner is KEPT (it supplies model/CLIP/VAE to the detailer crops) — but it
    # never runs a global pass over the image any more.
    assert wf["refiner_checkpoint_loader"]["inputs"]["ckpt_name"] \
        == "gonzalomoXLFluxPony_v70PhotoXLDMD.safetensors"
    # CHROMA-FACE PRESERVED (user: "I prefer the chroma base face only"): the refine
    # stage runs NO SDXL on the face. BOTH the global img2img refine (stage_ksampler
    # + its VAE encode/decode 132/133) AND the dedicated face detailer are REMOVED —
    # the only SDXL touches are targeted crops on hands / feet / nipples, so the
    # keeper face IS the (upscaled) Chroma base face.
    for absent in ("stage_ksampler", "132", "133", "detailer_face", "det_face_detector",
                   "detailer_eyes", "det_eye_detector"):
        assert absent not in wf, f"{absent} must be removed (no SDXL on the face)"
    assert "KSampler" not in classes           # no global img2img repaint at all
    # The FOOT detailer was DROPPED: neither bare-foot YOLO (foot-yolov8l /
    # FootYolov8x_v20) detects nude feet (0 detections even at conf 0.05), so it
    # never fired — pure overhead. Feet now rely on prompt guidance + framing.
    assert "detailer_feet" not in wf and "det_foot_detector" not in wf
    # 1.25x lanczos upscale feeds the detailer chain DIRECTLY (no VAE round-trip)
    assert wf["143"]["class_type"] == "ImageScaleBy" and wf["143"]["inputs"]["scale_by"] == 1.25
    # detailer chain: hands → nipples → save, all on the upscaled Chroma image
    assert wf["detailer_hands"]["inputs"]["image"] == ["143", 0]
    assert wf["detailer_nipples"]["inputs"]["image"] == ["detailer_hands", 0]
    assert wf["save"]["inputs"]["images"] == ["detailer_nipples", 0]
    assert wf["det_hand_detector"]["inputs"]["model_name"] == "bbox/hand_yolov9c.pt"
    assert wf["det_nipple_detector"]["inputs"]["model_name"] == "bbox/nipples_yolov8s.pt"
    # hand detailer is STRONGER (denoise 0.45 ×2 cycles) to actually RESTRUCTURE
    # bad fingers, not just polish them; per-region anatomy wildcards steer the crops
    assert wf["detailer_hands"]["inputs"]["denoise"] == 0.45
    assert wf["detailer_hands"]["inputs"]["cycle"] == 2
    assert "five fingers" in wf["detailer_hands"]["inputs"]["wildcard"]
    assert "areola" in wf["detailer_nipples"]["inputs"]["wildcard"]
    # every detailer is MPS-safe lcm/karras on the DifferentialDiffusion-wrapped model
    assert wf["diffdiff_model"]["class_type"] == "DifferentialDiffusion"
    assert wf["diffdiff_model"]["inputs"]["model"] == ["refiner_checkpoint_loader", 0]
    for d in ("detailer_hands", "detailer_nipples"):
        assert wf[d]["class_type"] == "FaceDetailer"
        assert wf[d]["inputs"]["model"] == ["diffdiff_model", 0]
        assert wf[d]["inputs"]["sampler_name"] == "lcm" and wf[d]["inputs"]["scheduler"] == "karras"
    # nipple detailer is LIGHT (cosmetic; a no-op when no nipples are detected)
    assert wf["detailer_nipples"]["inputs"]["denoise"] <= 0.25
    # TIER PURITY: the base (non-T4) refine carries NO vagina detailer, so a
    # tasteful T3 art-nude never has its genitals detailed.
    assert "detailer_vagina" not in wf and "det_vagina_detector" not in wf


def test_refine_t4_is_refine_plus_vagina_detailer():
    """The T4 refine variant == the base refine + a light vagina detailer
    (vagina-v3.2, which DOES reliably detect — 0.86–0.91 conf in testing, unlike
    the bare-foot models). T4-only routing keeps T3 tasteful (tier purity).

    This pins the TEMPLATE STRUCTURE only; the tier ROUTING that keeps this
    template off sub-T4 main images + all covers is tested in
    tests/test_sellable_pipeline.py (test_refine_templates_for_keeps_covers_sfw +
    test_genital_detailer_detection_drives_tier_purity_guard)."""
    base = _load(REFINE)
    t4 = _load(REFINE_T4)
    # drift guard: every base-refine node is present UNCHANGED in the T4 variant
    # (the sink is the only rewire — it points at the new vagina detailer).
    for nid, node in base.items():
        if nid == "save":
            continue
        assert t4.get(nid) == node, f"refine_T4 drifted from refine at node {nid}"
    # the T4 variant adds exactly the vagina detector + a light detailer, chained
    # last (hands → nipples → vagina → save)
    assert t4["det_vagina_detector"]["inputs"]["model_name"] == "bbox/vagina-v3.2.pt"
    assert t4["detailer_vagina"]["class_type"] == "FaceDetailer"
    assert t4["detailer_vagina"]["inputs"]["image"] == ["detailer_nipples", 0]
    assert t4["detailer_vagina"]["inputs"]["model"] == ["diffdiff_model", 0]
    assert t4["detailer_vagina"]["inputs"]["denoise"] <= 0.25          # light polish
    wc = t4["detailer_vagina"]["inputs"]["wildcard"]
    assert "vulva" in wc or "labia" in wc
    assert t4["save"]["inputs"]["images"] == ["detailer_vagina", 0]
    # Chroma face is still preserved in the T4 variant (no SDXL on the face)
    for absent in ("stage_ksampler", "detailer_face", "det_face_detector"):
        assert absent not in t4


def test_upscale_contract_and_values():
    wf = _load(UPSCALE)
    assert wf["load_image"]["class_type"] == "VHS_LoadImagePath"
    assert wf["refiner_checkpoint_loader"]["inputs"]["ckpt_name"] \
        == "gonzalomoXLFluxPony_v70PhotoXLDMD.safetensors"   # v7.0 refiner
    u = wf["upscale"]
    assert u["class_type"] == "UltimateSDUpscale"
    assert u["inputs"]["sampler_name"] == "lcm" and u["inputs"]["scheduler"] == "karras"
    assert u["inputs"]["steps"] == 8           # v7.0: steps≈10×CFG (was 6)
    # FACE-TRUE 4K (2026-06-10 A/B): denoise 0.18 visibly restructured the
    # Chroma face on the PAID product (sharpened eyes/brows, changed makeup);
    # 0.05 ≈ Remacri upscale + seam blend, face nearly identical to source.
    assert u["inputs"]["denoise"] == 0.05
    assert u["inputs"]["seam_fix_denoise"] == 0.10
    # tile 1536 (6 tiles for a ~3000x3848 4K): the staged 4K stage is SDXL-only
    # with ~40GB free, so larger tiles than the monolith's 1280 are memory-safe
    # and ~halve the tile count.
    assert u["inputs"]["tile_width"] == 1536
    assert u["inputs"]["seam_fix_mode"] == "Half Tile + Intersections"  # v35: corner seam fix
    assert wf["skin_upscale_model"]["inputs"]["model_name"] == "4x_foolhardy_Remacri.pth"
    # USDU-ONLY at 4K: the post-upscale FaceDetailers cannot run at 4K on Apple
    # MPS — the FaceDetailer's VAE-encode self-attention on a large face crop hits
    # "tensor dims larger than INT_MAX" / a 55 GiB buffer (it does not tile). The
    # USDU pass itself (4x Remacri + tiled SDXL DMD refine) already refines the 4K
    # image, and the review keeper already carries the Chroma face + repaired hands.
    no_detailers = {nd["class_type"] for nd in wf.values()}
    assert "FaceDetailer" not in no_detailers
    assert "DifferentialDiffusion" not in wf and "detailer_face" not in wf
    assert wf["save"]["inputs"]["images"] == ["upscale", 0]   # save the USDU 4K directly

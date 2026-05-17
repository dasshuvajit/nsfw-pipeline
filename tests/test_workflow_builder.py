"""WorkflowBuilder — dispatch, required-node guards, flux2 clamps.

flux2 (FLUX.2 Klein 9B) is distilled: cfg=1.0, steps=4, sampler=euler,
scheduler=simple are contract, not preference. This test file asserts
those clamps fire and that the graph doesn't accidentally grow any of
the flux.1 nodes (FluxGuidance, ModelSamplingFlux, DualCLIPLoader) that
would melt the image if added.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.render.workflow_builder import (
    WorkflowBuilder,
    WorkflowTemplateError,
)


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "config" / "comfyui_workflows"


@pytest.fixture
def builder():
    return WorkflowBuilder(workflow_dir=WORKFLOW_DIR)


def _flux2_profile(**overrides):
    base = dict(
        workflow_family="flux2",
        model_checkpoint="flux-2-klein-9b.safetensors",
        sampler="euler",
        scheduler="simple",
        steps=4,
        cfg=1.0,
        lora_stack=None,
        supports_ipadapter=False,
        supports_lora=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---- dispatch ----------------------------------------------------------


def test_flux2_dispatch_loads_klein_template(builder):
    graph = builder.build(
        prompt_text="A woman in golden light.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(1024, 1024),
        seed=42,
    )
    nodes = {n["class_type"] for n in graph.values()}
    assert "UNETLoader" in nodes
    assert "CLIPLoader" in nodes
    assert "VAELoader" in nodes
    assert "EmptySD3LatentImage" in nodes
    assert "KSampler" in nodes


def test_flux2_omits_flux1_only_nodes(builder):
    graph = builder.build(
        prompt_text="A woman reclines on silk sheets in warm window light.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(1024, 1024),
        seed=1,
    )
    nodes = {n["class_type"] for n in graph.values()}
    assert "FluxGuidance" not in nodes, (
        "FluxGuidance melts Klein output — distillation removes guidance."
    )
    assert "ModelSamplingFlux" not in nodes
    assert "DualCLIPLoader" not in nodes, (
        "Klein uses single Qwen3 CLIPLoader, not DualCLIPLoader."
    )


def test_flux2_clip_loader_type_is_flux2(builder):
    graph = builder.build(
        prompt_text="A woman in a studio.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(1024, 1024),
        seed=1,
    )
    clip = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
    assert clip["inputs"]["type"] == "flux2"
    # Qwen3-8B is the only valid encoder for Klein — template bakes
    # the filename, but the type enum must stay flux2.


def test_flux2_ksampler_contract_honored(builder):
    graph = builder.build(
        prompt_text="A woman in a studio.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(1024, 1024),
        seed=99,
    )
    ks = graph["ksampler"]["inputs"]
    assert ks["cfg"] == 1.0
    assert ks["steps"] == 4
    assert ks["sampler_name"] == "euler"
    assert ks["scheduler"] == "simple"
    assert ks["seed"] == 99


# ---- distilled-contract clamps -----------------------------------------


def test_flux2_clamps_cfg_to_1(builder, caplog):
    caplog.set_level(logging.WARNING, logger="src.render.workflow_builder")
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(cfg=3.5),
        resolution=(1024, 1024),
        seed=1,
    )
    assert graph["ksampler"]["inputs"]["cfg"] == 1.0
    assert any("cfg" in r.message for r in caplog.records)


def test_flux2_clamps_steps_when_over_6(builder, caplog):
    caplog.set_level(logging.WARNING, logger="src.render.workflow_builder")
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(steps=28),
        resolution=(1024, 1024),
        seed=1,
    )
    assert graph["ksampler"]["inputs"]["steps"] == 4
    assert any("steps" in r.message for r in caplog.records)


def test_flux2_allows_steps_within_budget(builder):
    # Low step counts 1..6 should pass through unchanged — only >6 clamps.
    for n in (1, 4, 6):
        graph = builder.build(
            prompt_text="A woman.",
            negative_prompt="",
            style_profile=_flux2_profile(steps=n),
            resolution=(1024, 1024),
            seed=1,
        )
        assert graph["ksampler"]["inputs"]["steps"] == n


def test_flux2_forces_sampler_euler(builder, caplog):
    caplog.set_level(logging.WARNING, logger="src.render.workflow_builder")
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(sampler="dpmpp_2m"),
        resolution=(1024, 1024),
        seed=1,
    )
    assert graph["ksampler"]["inputs"]["sampler_name"] == "euler"
    assert any("sampler" in r.message for r in caplog.records)


def test_flux2_forces_scheduler_simple(builder, caplog):
    caplog.set_level(logging.WARNING, logger="src.render.workflow_builder")
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(scheduler="beta"),
        resolution=(1024, 1024),
        seed=1,
    )
    assert graph["ksampler"]["inputs"]["scheduler"] == "simple"
    assert any("scheduler" in r.message for r in caplog.records)


# ---- resolution, seed, prompt --------------------------------------------


def test_flux2_resolution_wired_into_latent(builder):
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(832, 1216),
        seed=1,
    )
    lat = graph["empty_latent"]["inputs"]
    assert lat["width"] == 832
    assert lat["height"] == 1216
    assert lat["batch_size"] == 1


def test_flux2_prompt_text_injected(builder):
    graph = builder.build(
        prompt_text="She reclines on cream silk in late-afternoon light.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(1024, 1024),
        seed=1,
    )
    assert "reclines on cream silk" in graph["positive_prompt"]["inputs"]["text"]


def test_flux2_negative_prompt_wired_even_when_empty(builder):
    # KSampler's `negative` input has to reference a valid node; the
    # empty string is fine but the node must survive into the graph.
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(),
        resolution=(1024, 1024),
        seed=1,
    )
    assert "negative_prompt" in graph
    assert graph["negative_prompt"]["inputs"]["text"] == ""


# ---- LoRA stack cap ------------------------------------------------------


def test_flux2_rejects_more_than_two_loras(builder):
    stack = (
        '[{"name":"a","strength":0.7},'
        '{"name":"b","strength":0.6},'
        '{"name":"c","strength":0.5}]'
    )
    with pytest.raises(WorkflowTemplateError, match="caps LoRAs at 2"):
        builder.build(
            prompt_text="A woman.",
            negative_prompt="",
            style_profile=_flux2_profile(lora_stack=stack),
            resolution=(1024, 1024),
            seed=1,
        )


def test_flux2_no_loras_strips_loader_nodes(builder):
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_flux2_profile(lora_stack=None),
        resolution=(1024, 1024),
        seed=1,
    )
    # Both lora_loader_0 and lora_loader_1 should be stripped and the
    # KSampler rewired directly to load_unet.
    assert "lora_loader_0" not in graph
    assert "lora_loader_1" not in graph
    assert graph["ksampler"]["inputs"]["model"] == ["load_unet", 0]


# ---- flux.1 regression (guard against cross-family leakage) --------------


# ---- model-YAML lora_stack wiring (via StyleProfileForWorkflow) ----------


# ---- flux.1 regression (guard against cross-family leakage) --------------


def test_flux1_still_dispatches_to_its_own_builder(builder):
    # flux.1 still uses the old GGUF + DualCLIPLoader + FluxGuidance
    # graph; adding flux2 must not break it.
    profile = SimpleNamespace(
        workflow_family="flux",
        model_checkpoint="gonzalomoXLFluxPony_v30FluxDAIO.safetensors",
        sampler="dpmpp_2m",
        scheduler="beta",
        steps=28,
        cfg=3.5,
        lora_stack=None,
        supports_ipadapter=False,
        supports_lora=True,
    )
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=profile,
        resolution=(1024, 1024),
        seed=1,
    )
    nodes = {n["class_type"] for n in graph.values()}
    assert "FluxGuidance" in nodes
    assert "ModelSamplingFlux" in nodes
    assert "DualCLIPLoader" in nodes


# ---- Phase 2: lora_stack wires for SDXL/Pony/Illustrious/Flux ------------
# (flux2 covered above; chroma intentionally has no LoRA wiring.)


def _profile_with_loras(family: str, sampler: str, scheduler: str,
                        steps: int, cfg: float, lora_stack_json: str | None):
    """Build a SimpleNamespace profile that satisfies the WorkflowBuilder
    duck-type for non-flux2 family LoRA tests."""
    return SimpleNamespace(
        workflow_family=family,
        model_checkpoint=f"{family}_smoke.safetensors",
        sampler=sampler,
        scheduler=scheduler,
        steps=steps,
        cfg=cfg,
        lora_stack=lora_stack_json,
        supports_ipadapter=False,
        supports_lora=True,
    )


@pytest.mark.parametrize("family,sampler,scheduler,steps,cfg", [
    ("sdxl",        "dpmpp_2m",   "karras", 30, 6.0),
    ("pony",        "dpmpp_sde",  "karras", 30, 5.0),
    ("illustrious", "euler_a",    "normal", 28, 6.0),
])
def test_two_lora_stack_wires_both_loaders_for_clip_family(
    builder, family, sampler, scheduler, steps, cfg,
):
    """SDXL / Pony / Illustrious — base.json templates lora_loader_0
    and lora_loader_1; a profile-supplied 2-LoRA stack must wire both."""
    stack = (
        '[{"name":"realism_skin","strength":0.65},'
        '{"name":"detail_boost","strength":0.85}]'
    )
    graph = builder.build(
        prompt_text="A woman in soft light.",
        negative_prompt="blurry",
        style_profile=_profile_with_loras(
            family, sampler, scheduler, steps, cfg, stack,
        ),
        resolution=(896, 1152),
        seed=42,
    )
    loaded = {
        n["inputs"]["lora_name"]: n["inputs"]["strength_model"]
        for n in graph.values()
        if n.get("class_type") == "LoraLoader"
    }
    assert loaded == {
        "realism_skin.safetensors": pytest.approx(0.65),
        "detail_boost.safetensors": pytest.approx(0.85),
    }


def test_flux1_two_lora_stack_wires_both_loaders(builder):
    """Flux.1 has its own builder branch; the schema reads the same
    JSON shape and wires lora_loader_0/1 just like SDXL."""
    stack = (
        '[{"name":"flux_realism","strength":0.80},'
        '{"name":"flux_detail","strength":0.55}]'
    )
    profile = SimpleNamespace(
        workflow_family="flux",
        model_checkpoint="fluxNSFW.gguf",
        sampler="dpmpp_2m",
        scheduler="beta",
        steps=28,
        cfg=3.5,
        lora_stack=stack,
        supports_ipadapter=False,
        supports_lora=True,
    )
    graph = builder.build(
        prompt_text="A woman in soft light.",
        negative_prompt="",
        style_profile=profile,
        resolution=(1024, 1024),
        seed=42,
    )
    loaded = {
        n["inputs"]["lora_name"]: n["inputs"]["strength_model"]
        for n in graph.values()
        if n.get("class_type") == "LoraLoader"
    }
    assert loaded == {
        "flux_realism.safetensors": pytest.approx(0.80),
        "flux_detail.safetensors":  pytest.approx(0.55),
    }


@pytest.mark.parametrize("family,sampler,scheduler,steps,cfg", [
    ("sdxl",        "dpmpp_2m",   "karras", 30, 6.0),
    ("pony",        "dpmpp_sde",  "karras", 30, 5.0),
    ("illustrious", "euler_a",    "normal", 28, 6.0),
])
def test_no_lora_stack_strips_loader_nodes_for_clip_family(
    builder, family, sampler, scheduler, steps, cfg,
):
    """When the profile has no LoRAs, the lora_loader_* nodes must be
    stripped from the workflow graph and the KSampler / sampler chain
    rerouted to drain through the checkpoint loader directly."""
    graph = builder.build(
        prompt_text="A woman.",
        negative_prompt="",
        style_profile=_profile_with_loras(
            family, sampler, scheduler, steps, cfg, None,
        ),
        resolution=(1024, 1024),
        seed=1,
    )
    assert "lora_loader_0" not in graph
    assert "lora_loader_1" not in graph

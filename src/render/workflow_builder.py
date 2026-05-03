"""Build ComfyUI workflows from a template + dynamic parameters.

Loads a workflow JSON template (built in the ComfyUI UI, exported as API
JSON, with node IDs renamed to semantic names — see
``docs/COMFYUI_WORKFLOWS.md``) and injects per-render parameters into
it. Templates are cached and a deep copy is returned for each build so
callers can mutate freely.

The family of template files to load is taken from
``style_profile.workflow_family`` (``sdxl``, ``flux``, ``pony``, ``chroma``,
or ``flux2``), which is populated by ``render_set.py`` from the
``model_registry`` row's ``workflow_family`` column. Phase 1 ships
``sdxl/base.json``, ``sdxl/ipadapter.json``, ``pony/base.json``,
``pony/ipadapter.json``, ``chroma/base.json``, ``flux/base.json``, and
``flux2/base.json``. The pony templates add a ``CLIPSetLastLayer`` node
for clip_skip 2. The chroma template uses ``UnetLoaderGGUF`` +
``ModelSamplingAuraFlow`` + CFGGuider + ``SamplerCustomAdvanced``. The
flux template uses ``UnetLoaderGGUF`` + ``DualCLIPLoader`` (type=flux)
+ ``ModelSamplingFlux`` + ``FluxGuidance`` + standard ``KSampler`` (cfg
hardcoded to 1.0 — the user-tunable "guidance" is
``FluxGuidance.guidance`` and is mapped from ``style_profile.cfg``).
The flux2 template loads a safetensors UNET from ``diffusion_models/``
via ``UNETLoader`` with the Klein-9B distilled contract enforced by
``_build_flux2`` (cfg=1.0, steps≤6, sampler=euler, scheduler=simple).

The style_profile argument is duck-typed: any object with these attributes
works (DB row adapter, dataclass, ``types.SimpleNamespace``):

    workflow_family:   str           # 'sdxl'|'flux'|'pony'|'chroma'|'flux2' (from registry)
    model_checkpoint:  str           # checkpoint or GGUF filename in ComfyUI
                                     # (GGUF for chroma/flux; safetensors UNET for flux2)
    sampler:           str
    scheduler:         str
    steps:             int
    cfg:               float
    lora_stack:        Optional[str] # JSON list[{"name","strength"}], or None
    supports_ipadapter: bool         # from model_registry row
    supports_lora:     bool          # from model_registry row

The two ``supports_*`` flags drive capability guards: a render that
asks for IPAdapter against a model whose ``supports_ipadapter`` is
False **raises** (rather than silently falling back) because a silent
downgrade produces random faces when the user explicitly asked for
character lock. Same for LoRAs: a LoRA-bearing style profile with
``supports_lora=False`` raises, because LoRAs change identity and a
silent drop is a footgun.

Required semantic node IDs in every template (raise WorkflowTemplateError
if missing): load_checkpoint, positive_prompt, negative_prompt,
empty_latent, ksampler. If the style profile has N LoRAs, the template
must additionally contain lora_loader_0 ... lora_loader_(N-1). The
ipadapter template adds three more required nodes:
ipadapter_unified_loader, ipadapter_apply, load_reference_image.

Reference-image path handling: WorkflowBuilder is intentionally pure — it
takes whatever string the caller passes and writes it into
``load_reference_image.inputs.image``. ComfyUI's LoadImage node reads
filenames relative to its own ``input/`` directory, so the caller is
responsible for (a) copying the file there and (b) passing only the
basename. ``scripts/render_set.py`` and ``scripts/test_ipadapter.py`` do
this via ``comfyui.input_dir`` from ``pipeline.yaml``.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Required semantic IDs for the SDXL base workflow.
_REQUIRED_NODES_BASE = (
    "load_checkpoint",
    "positive_prompt",
    "negative_prompt",
    "empty_latent",
    "ksampler",
)

# IPAdapter template = base nodes + three extras for the reference-image path.
_REQUIRED_NODES_IPADAPTER = _REQUIRED_NODES_BASE + (
    "ipadapter_unified_loader",
    "ipadapter_apply",
    "load_reference_image",
)

# Chroma uses a completely different node graph: UnetLoaderGGUF +
# SamplerCustomAdvanced instead of CheckpointLoaderSimple + KSampler.
_REQUIRED_NODES_CHROMA = (
    "load_unet",
    "positive_prompt",
    "negative_prompt",
    "empty_latent",
    "random_noise",
    "sampler_select",
    "beta_scheduler",
    "cfg_guider",
    "sampler_advanced",
)

# Flux (flux-dev GGUF): UnetLoaderGGUF + DualCLIPLoader + standard
# KSampler, plus ModelSamplingFlux and FluxGuidance (flux-specific).
# LoRA loader nodes (lora_loader_0/1) are optional — scaffolded in the
# template and stripped at build time when the style profile has fewer
# than 2 LoRAs. KSampler.cfg is hardcoded to 1.0; style_profile.cfg
# maps onto FluxGuidance.guidance.
_REQUIRED_NODES_FLUX = (
    "load_unet",
    "model_sampling",
    "load_clip",
    "load_vae",
    "positive_prompt",
    "negative_prompt",
    "flux_guidance",
    "empty_latent",
    "ksampler",
)

# FLUX.2 Klein 9B (distilled): UNETLoader (bf16 safetensors, NOT GGUF) +
# single CLIPLoader(type=flux2) carrying Qwen3-8B + VAELoader (32-ch VAE)
# + standard KSampler. Distillation removes guidance: no FluxGuidance,
# no ModelSamplingFlux, no DualCLIPLoader — adding any of them is a
# silent footgun (FluxGuidance with cfg=1 no-ops until someone flips it
# on). cfg/steps/sampler/scheduler are CONTRACT (see _build_flux2's
# clamps), not preference.
_REQUIRED_NODES_FLUX2 = (
    "load_unet",
    "load_clip",
    "load_vae",
    "positive_prompt",
    "negative_prompt",
    "empty_latent",
    "ksampler",
)

# External user-authored templates (``--template`` path). Uniform across
# all families — the user's convention is ``ksampler`` for the seed-
# carrying node regardless of the ComfyUI ``class_type`` (may be
# ``KSampler``, ``SamplerCustomAdvanced``, ``ClownsharKSampler_Beta``,
# etc.). The pipeline validates these four top-level keys exist and
# then injects prompt text, negative text, seed, and resolution — and
# nothing else. Checkpoint, LoRAs, sampler config, VAE, CLIP, and any
# post-processing are whatever the template author baked in.
_REQUIRED_NODES_EXTERNAL: tuple[str, ...] = (
    "positive_prompt",
    "negative_prompt",
    "ksampler",
    "empty_latent",
)

# Required input fields inside those four nodes — a template with the
# right top-level keys but a malformed node shape still fails preflight
# with a precise error that points at the missing field.
_REQUIRED_EXTERNAL_INPUTS: dict[str, tuple[str, ...]] = {
    "positive_prompt": ("text",),
    "negative_prompt": ("text",),
    "ksampler":        ("seed",),
    "empty_latent":    ("width", "height"),
}

# Default IPAdapter weight when the caller doesn't override. 0.7 is the
# Section 7 sweet spot — strong enough to lock identity, soft enough to
# let pose/lighting/expression vary across a set. Bumped from the cubiq
# default of 1.0 because at 1.0 the reference dominates and every render
# in a set looks like the same shot.
_DEFAULT_IPADAPTER_WEIGHT = 0.7

# Max uint32 — ComfyUI/SD seeds are 32-bit unsigned ints.
_MAX_SEED = 2**32 - 1


class WorkflowTemplateError(Exception):
    """A workflow template is missing or malformed."""


def _assert_external_template_inputs(
    workflow: dict[str, Any], template_name: str,
) -> None:
    """Validate that each required external-template node carries the
    input fields ``build_external`` injects into.

    Called AFTER ``_assert_required_nodes`` confirms top-level keys:
    this layer catches the case where a template has e.g.
    ``empty_latent`` at the top level but its ``inputs`` dict is
    missing ``width`` or ``height`` (which would raise a KeyError
    mid-inject).
    """
    missing: list[str] = []
    for node_id, field_keys in _REQUIRED_EXTERNAL_INPUTS.items():
        node = workflow.get(node_id, {})
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            missing.append(f"{node_id}.inputs (missing or not an object)")
            continue
        for field in field_keys:
            if field not in inputs:
                missing.append(f"{node_id}.inputs.{field}")
    if missing:
        raise WorkflowTemplateError(
            f"External template {template_name} has the required nodes "
            f"but is missing input fields: {missing}. Each required "
            f"node must expose these keys (see docs/COMFYUI_WORKFLOWS.md "
            f"§ External templates)."
        )


def _resolve_template_path(
    raw: str, workflow_dir: Path, project_root: Path,
) -> str:
    """Canonicalize a ``--template`` CLI value into a string that
    ``WorkflowBuilder._load`` accepts.

    ``_load`` joins its argument with ``workflow_dir`` via pathlib, and
    Python's ``Path("rel") / "/abs"`` already returns the absolute path
    — so absolute values pass through unchanged. For relative values we
    first try ``workflow_dir`` (the natural home under
    ``config/comfyui_workflows/templates/...``), then the project root
    (for paths the user typed from the repo root), and finally fall
    through so ``_load`` raises its normal not-found error with the
    user's original string.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    if (workflow_dir / raw).exists():
        return raw
    if (project_root / raw).exists():
        abs_from_root = (project_root / raw).resolve()
        try:
            return str(abs_from_root.relative_to(workflow_dir.resolve()))
        except ValueError:
            return str(abs_from_root)
    return raw


def _strip_unused_lora_loaders(
    workflow: dict[str, Any], used_lora_count: int
) -> None:
    """Remove unused ``lora_loader_N`` nodes and rewire downstream refs.

    Templates bake in ``lora_loader_0`` / ``lora_loader_1`` for the
    2-LoRA Appendix A cap. When fewer LoRAs are used, unused loader
    nodes still hold placeholder filenames that ComfyUI would reject.
    Remove them high-index first and redirect downstream consumers
    (any node input that points to ``[loader_key, 0|1]``) back to the
    loader's own model/clip source — flattening the chain.
    """
    max_idx = max(
        (int(k.rsplit("_", 1)[-1]) for k in workflow if k.startswith("lora_loader_")),
        default=-1,
    )
    for i in reversed(range(used_lora_count, max_idx + 1)):
        node_key = f"lora_loader_{i}"
        if node_key not in workflow:
            continue
        model_src = workflow[node_key]["inputs"]["model"]
        clip_src = workflow[node_key]["inputs"]["clip"]
        for other_node in workflow.values():
            for inp_key, inp_val in other_node.get("inputs", {}).items():
                if (
                    isinstance(inp_val, list)
                    and len(inp_val) == 2
                    and inp_val[0] == node_key
                ):
                    other_node["inputs"][inp_key] = (
                        model_src if inp_val[1] == 0 else clip_src
                    )
        del workflow[node_key]


class WorkflowBuilder:
    def __init__(self, workflow_dir: Path | str) -> None:
        self.workflow_dir = Path(workflow_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def build(
        self,
        prompt_text: str,
        negative_prompt: str,
        style_profile: Any,
        resolution: tuple[int, int],
        seed: int | None = None,
        *,
        ipadapter_image: str | None = None,
        ipadapter_weight: float = _DEFAULT_IPADAPTER_WEIGHT,
    ) -> dict[str, Any]:
        """Return a workflow ready to POST to ComfyUI's /prompt endpoint.

        See module docstring for the style_profile contract and required
        node IDs.

        IPAdapter path: when ``ipadapter_image`` is a non-empty string, the
        builder loads ``{family}/ipadapter.json`` instead of the base
        template, validates the three additional required nodes, and
        injects the reference filename into ``load_reference_image`` and
        the weight into ``ipadapter_apply``. The caller is responsible
        for copying the actual image bytes into ComfyUI's ``input/`` dir
        before submitting the workflow — see the module docstring.

        Capability guards (from the model_registry row carried on
        style_profile): if ``ipadapter_image`` is truthy but the model's
        ``supports_ipadapter`` is False, raise rather than silently
        falling back to the base template. If ``lora_stack`` is non-empty
        but the model's ``supports_lora`` is False, raise rather than
        silently dropping the LoRAs. Both of these silent-downgrade paths
        produce wrong-looking output (random faces / missing styling) and
        cost real render time, so we refuse up-front.
        """
        family = style_profile.workflow_family

        # Chroma and Flux both have fundamentally different node graphs from
        # the SDXL/Pony base (no single CheckpointLoader, no shared KSampler
        # contract) — dispatch to dedicated builders before the SDXL/Pony
        # capability guards which reference nodes that don't exist in these
        # templates.
        if family == "chroma":
            return self._build_chroma(
                prompt_text=prompt_text,
                negative_prompt=negative_prompt,
                style_profile=style_profile,
                resolution=resolution,
                seed=seed,
            )
        if family == "flux":
            return self._build_flux(
                prompt_text=prompt_text,
                negative_prompt=negative_prompt,
                style_profile=style_profile,
                resolution=resolution,
                seed=seed,
            )
        if family == "flux2":
            return self._build_flux2(
                prompt_text=prompt_text,
                negative_prompt=negative_prompt,
                style_profile=style_profile,
                resolution=resolution,
                seed=seed,
            )

        use_ipadapter = bool(ipadapter_image)

        # ----- Capability guards (run before anything expensive) ----------
        # These fire BEFORE the 2-LoRA Appendix A cap so the error message
        # points at the model-compat issue first — that's the root cause
        # and the cap check would be confusing noise after it.
        if use_ipadapter and not getattr(
            style_profile, "supports_ipadapter", True
        ):
            raise WorkflowTemplateError(
                f"Model {getattr(style_profile, 'model_checkpoint', '?')!r} "
                f"does not support IPAdapter (supports_ipadapter=False in "
                f"model_registry). Remove --reference-image, or pick a "
                f"model with IPAdapter support (see "
                f"`python scripts/list_models.py`)."
            )

        lora_stack_json = getattr(style_profile, "lora_stack", None)
        if lora_stack_json and not getattr(
            style_profile, "supports_lora", True
        ):
            raise WorkflowTemplateError(
                f"Model {getattr(style_profile, 'model_checkpoint', '?')!r} "
                f"does not support LoRAs (supports_lora=False in "
                f"model_registry) but the style profile declares a "
                f"lora_stack. Create a no-LoRA style profile for this "
                f"model, or pick a LoRA-compatible model (see "
                f"`python scripts/list_models.py`)."
            )

        template_name = "ipadapter.json" if use_ipadapter else "base.json"
        template_path = f"{family}/{template_name}"
        required = (
            _REQUIRED_NODES_IPADAPTER if use_ipadapter else _REQUIRED_NODES_BASE
        )

        workflow = self._load(template_path)
        self._assert_required_nodes(workflow, family, template_name, required)

        # Resolution (per-scene, from RatioSelector once it exists)
        workflow["empty_latent"]["inputs"]["width"] = int(resolution[0])
        workflow["empty_latent"]["inputs"]["height"] = int(resolution[1])
        workflow["empty_latent"]["inputs"]["batch_size"] = 1

        # Prompts
        workflow["positive_prompt"]["inputs"]["text"] = prompt_text
        if "negative_prompt" in workflow:
            workflow["negative_prompt"]["inputs"]["text"] = negative_prompt

        # Checkpoint
        workflow["load_checkpoint"]["inputs"]["ckpt_name"] = (
            style_profile.model_checkpoint
        )

        # Sampler / scheduler / seed.
        # NOTE: deliberate deviation from the Section 14 snippet, which uses
        # `seed or random.randint(...)` — that treats seed=0 as "no seed".
        # We use `if seed is None` so 0 is a valid, honored seed.
        chosen_seed = (
            int(seed) if seed is not None else random.randint(0, _MAX_SEED)
        )
        workflow["ksampler"]["inputs"].update({
            "sampler_name": style_profile.sampler,
            "scheduler": style_profile.scheduler,
            "steps": int(style_profile.steps),
            "cfg": float(style_profile.cfg),
            "seed": chosen_seed,
        })

        # LoRAs (skipped entirely if the profile has none, which is the
        # common case for the test smoke test). The lora_stack_json was
        # already read above for the supports_lora guard — reuse it here.
        loras = json.loads(lora_stack_json) if lora_stack_json else []
        if len(loras) > 2:
            # Appendix A rule 1: max 2 LoRAs per render.
            raise WorkflowTemplateError(
                f"Style profile has {len(loras)} LoRAs but the pipeline "
                f"caps LoRAs at 2 per render (Appendix A, rule 1)."
            )
        for i, lora in enumerate(loras):
            node_key = f"lora_loader_{i}"
            if node_key not in workflow:
                raise WorkflowTemplateError(
                    f"Style profile has a LoRA at index {i} ('{lora.get('name')}') "
                    f"but template {family}/base.json has no node '{node_key}'. "
                    f"Add a LoraLoader node with that semantic ID and chain it "
                    f"between load_checkpoint and ksampler."
                )
            workflow[node_key]["inputs"].update({
                "lora_name": lora["name"] + ".safetensors",
                "strength_model": float(lora["strength"]),
                "strength_clip": float(lora["strength"]),
            })

        # Remove unused LoRA loader nodes and rewire the graph.
        _strip_unused_lora_loaders(workflow, used_lora_count=len(loras))

        # IPAdapter — only when the caller asked for it. We've already
        # validated that the IPAdapter nodes exist, so direct injection
        # is safe. We only touch `image` and `weight`; everything else
        # (preset, weight_type, start_at, end_at, ...) stays at whatever
        # the template author baked in.
        if use_ipadapter:
            workflow["load_reference_image"]["inputs"]["image"] = ipadapter_image
            workflow["ipadapter_apply"]["inputs"]["weight"] = float(ipadapter_weight)

        return workflow

    # ----- chroma builder -----------------------------------------------------

    def _build_chroma(
        self,
        prompt_text: str,
        negative_prompt: str,
        style_profile: Any,
        resolution: tuple[int, int],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a Chroma workflow (UnetLoaderGGUF + SamplerCustomAdvanced).

        Chroma uses a fundamentally different ComfyUI graph from SDXL/Pony:
        three separate loaders (UNET, CLIP, VAE), CFGGuider instead of
        KSampler, BetaSamplingScheduler, and RandomNoise. No LoRA or
        IPAdapter support.
        """
        template_path = "chroma/base.json"

        workflow = self._load(template_path)
        self._assert_required_nodes(
            workflow, "chroma", template_path, _REQUIRED_NODES_CHROMA
        )

        chosen_seed = (
            int(seed) if seed is not None else random.randint(0, _MAX_SEED)
        )

        # UNET model
        workflow["load_unet"]["inputs"]["unet_name"] = (
            style_profile.model_checkpoint
        )

        # Resolution
        workflow["empty_latent"]["inputs"]["width"] = int(resolution[0])
        workflow["empty_latent"]["inputs"]["height"] = int(resolution[1])
        workflow["empty_latent"]["inputs"]["batch_size"] = 1

        # Prompts
        workflow["positive_prompt"]["inputs"]["text"] = prompt_text
        workflow["negative_prompt"]["inputs"]["text"] = negative_prompt

        # Seed
        workflow["random_noise"]["inputs"]["noise_seed"] = chosen_seed

        # Sampler
        workflow["sampler_select"]["inputs"]["sampler_name"] = (
            style_profile.sampler
        )

        # Steps (via BetaSamplingScheduler)
        workflow["beta_scheduler"]["inputs"]["steps"] = int(
            style_profile.steps
        )

        # CFG (via CFGGuider)
        workflow["cfg_guider"]["inputs"]["cfg"] = float(style_profile.cfg)

        return workflow

    # ----- flux builder -----------------------------------------------------

    def _build_flux(
        self,
        prompt_text: str,
        negative_prompt: str,
        style_profile: Any,
        resolution: tuple[int, int],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a Flux workflow (UnetLoaderGGUF + standard KSampler + LoRA scaffold).

        Flux-dev is flow-matched, not diffusion: KSampler.cfg must stay
        at 1.0, and the real "guidance" knob lives on a FluxGuidance
        node. We map ``style_profile.cfg`` onto ``flux_guidance.guidance``.
        The negative prompt is wired but effectively ignored by KSampler
        at cfg=1.0 (kept in the graph so KSampler's `negative` input has
        something valid to point at).

        LoRAs are applied with the same contract as the SDXL path (max
        2 per render, Appendix A rule 1). Unused ``lora_loader_N`` nodes
        are stripped from the graph and downstream references are
        rewired to the upstream source.
        """
        template_path = "flux/base.json"
        workflow = self._load(template_path)
        self._assert_required_nodes(
            workflow, "flux", template_path, _REQUIRED_NODES_FLUX
        )

        chosen_seed = (
            int(seed) if seed is not None else random.randint(0, _MAX_SEED)
        )

        workflow["load_unet"]["inputs"]["unet_name"] = (
            style_profile.model_checkpoint
        )

        workflow["empty_latent"]["inputs"]["width"] = int(resolution[0])
        workflow["empty_latent"]["inputs"]["height"] = int(resolution[1])
        workflow["empty_latent"]["inputs"]["batch_size"] = 1

        workflow["model_sampling"]["inputs"]["width"] = int(resolution[0])
        workflow["model_sampling"]["inputs"]["height"] = int(resolution[1])

        workflow["positive_prompt"]["inputs"]["text"] = prompt_text
        workflow["negative_prompt"]["inputs"]["text"] = negative_prompt

        # The user-tunable "cfg" scalar on style_profile drives
        # FluxGuidance, not KSampler. KSampler.cfg stays at 1.0.
        workflow["flux_guidance"]["inputs"]["guidance"] = float(
            style_profile.cfg
        )
        workflow["ksampler"]["inputs"].update({
            "seed": chosen_seed,
            "steps": int(style_profile.steps),
            "sampler_name": style_profile.sampler,
            "scheduler": style_profile.scheduler,
            "cfg": 1.0,
        })

        # LoRAs — identical contract to the SDXL path. Appendix A rule 1: max 2.
        lora_stack_json = getattr(style_profile, "lora_stack", None)
        loras = json.loads(lora_stack_json) if lora_stack_json else []
        if len(loras) > 2:
            raise WorkflowTemplateError(
                f"Style profile has {len(loras)} LoRAs but the pipeline "
                f"caps LoRAs at 2 per render (Appendix A, rule 1)."
            )
        for i, lora in enumerate(loras):
            node_key = f"lora_loader_{i}"
            if node_key not in workflow:
                raise WorkflowTemplateError(
                    f"Style profile has a LoRA at index {i} "
                    f"('{lora.get('name')}') but template flux/base.json "
                    f"has no node '{node_key}'."
                )
            workflow[node_key]["inputs"].update({
                "lora_name": lora["name"] + ".safetensors",
                "strength_model": float(lora["strength"]),
                "strength_clip": float(lora["strength"]),
            })

        # Strip unused lora_loader_N nodes and rewire downstream refs.
        _strip_unused_lora_loaders(workflow, used_lora_count=len(loras))

        return workflow

    # ----- flux2 builder ----------------------------------------------------

    def _build_flux2(
        self,
        prompt_text: str,
        negative_prompt: str,
        style_profile: Any,
        resolution: tuple[int, int],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a FLUX.2 Klein 9B workflow (distilled, cfg=1 / steps=4).

        Klein 9B is step- and guidance-distilled from FLUX.2-[dev] 32B.
        Four numbers are CONTRACT, not preference:
          - cfg = 1.0           (cfg > 1 burns out highlights)
          - steps = 4           (distilled; >6 adds noise, not detail)
          - sampler = euler
          - scheduler = simple

        The builder clamps these four back even if the model YAML
        overrides them, and logs a warning. This exists so a future edit
        to shared defaults (pipeline-level steps bump, say) can't
        silently wreck Klein output.

        Negative prompt is wired into the graph (KSampler's `negative`
        input needs something valid to point at) but is effectively
        ignored at cfg=1.0.
        """
        template_path = "flux2/base.json"
        workflow = self._load(template_path)
        self._assert_required_nodes(
            workflow, "flux2", template_path, _REQUIRED_NODES_FLUX2
        )

        chosen_seed = (
            int(seed) if seed is not None else random.randint(0, _MAX_SEED)
        )

        workflow["load_unet"]["inputs"]["unet_name"] = (
            style_profile.model_checkpoint
        )

        workflow["empty_latent"]["inputs"]["width"] = int(resolution[0])
        workflow["empty_latent"]["inputs"]["height"] = int(resolution[1])
        workflow["empty_latent"]["inputs"]["batch_size"] = 1

        workflow["positive_prompt"]["inputs"]["text"] = prompt_text
        workflow["negative_prompt"]["inputs"]["text"] = negative_prompt

        # Distilled contract clamps. Log if the incoming profile deviates
        # so an accidental override surfaces in run_log rather than as
        # silently-bad output.
        requested_cfg = float(style_profile.cfg)
        requested_steps = int(style_profile.steps)
        requested_sampler = style_profile.sampler
        requested_scheduler = style_profile.scheduler

        if requested_cfg != 1.0:
            logger.warning(
                "flux2: cfg %s → 1.0 (Klein 9B distilled contract)",
                requested_cfg,
            )
        if requested_steps > 6:
            logger.warning(
                "flux2: steps %s → 4 (Klein 9B distilled contract)",
                requested_steps,
            )
        if requested_sampler != "euler":
            logger.warning(
                "flux2: sampler %r → 'euler' (Klein 9B distilled contract)",
                requested_sampler,
            )
        if requested_scheduler != "simple":
            logger.warning(
                "flux2: scheduler %r → 'simple' (Klein 9B distilled contract)",
                requested_scheduler,
            )

        final_steps = requested_steps if requested_steps <= 6 else 4

        workflow["ksampler"]["inputs"].update({
            "seed": chosen_seed,
            "steps": final_steps,
            "sampler_name": "euler",
            "scheduler": "simple",
            "cfg": 1.0,
        })

        # LoRAs — max 2 (Appendix A rule 1). FLUX.1 LoRAs are
        # architecturally incompatible with FLUX.2 (different attention
        # dims / encoder) — we can't detect that here, but the model
        # YAML's `lora_stack` block is hand-maintained to only list
        # Klein-compatible LoRAs. Turbo LoRAs must never stack on Klein
        # (it's already distilled — see config/models/flux2_klein_9b.yaml).
        lora_stack_json = getattr(style_profile, "lora_stack", None)
        loras = json.loads(lora_stack_json) if lora_stack_json else []
        if len(loras) > 2:
            raise WorkflowTemplateError(
                f"Style profile has {len(loras)} LoRAs but the pipeline "
                f"caps LoRAs at 2 per render (Appendix A, rule 1)."
            )
        for i, lora in enumerate(loras):
            node_key = f"lora_loader_{i}"
            if node_key not in workflow:
                raise WorkflowTemplateError(
                    f"Style profile has a LoRA at index {i} "
                    f"('{lora.get('name')}') but template flux2/base.json "
                    f"has no node '{node_key}'."
                )
            workflow[node_key]["inputs"].update({
                "lora_name": lora["name"] + ".safetensors",
                "strength_model": float(lora["strength"]),
                "strength_clip": float(lora["strength"]),
            })

        _strip_unused_lora_loaders(workflow, used_lora_count=len(loras))

        return workflow

    # ----- external templates (--template) ----------------------------------

    def build_external(
        self,
        *,
        external_template: str,
        prompt_text: str,
        negative_prompt: str,
        resolution: tuple[int, int],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Load a user-provided external template and inject only the four
        contracted fields.

        ``positive_prompt.inputs.text``, ``negative_prompt.inputs.text``,
        ``ksampler.inputs.seed``, and ``empty_latent.inputs.width/height``
        are overwritten. Everything else — the template's checkpoint /
        UNET / GGUF loader, its LoRAs, its sampler config (name,
        scheduler, cfg, steps, eta, …), its VAE / CLIP, any IPAdapter
        chain the author wired in, any post-processing (face detailer,
        upscaler) — runs exactly as baked in. The user pastes a
        ComfyUI-UI-saved graph (API format) under
        ``config/comfyui_workflows/templates/{family}/``, renames four
        nodes to the semantic IDs, and the pipeline honors that choice.

        Preflight (before injection):
          - File exists + is valid JSON (via ``_load``).
          - Top-level keys present: ``positive_prompt``, ``negative_prompt``,
            ``ksampler``, ``empty_latent``.
          - Each of those nodes carries the required input fields
            (``inputs.text``, ``inputs.seed``, ``inputs.width``,
            ``inputs.height``).

        Any preflight failure raises ``WorkflowTemplateError`` with a
        specific message. There is no fallback to the built-in template.
        """
        workflow = self._load(external_template)
        template_name = Path(external_template).name
        self._assert_required_nodes(
            workflow, "external", template_name, _REQUIRED_NODES_EXTERNAL,
        )
        _assert_external_template_inputs(workflow, template_name)

        chosen_seed = (
            int(seed) if seed is not None else random.randint(0, _MAX_SEED)
        )
        workflow["positive_prompt"]["inputs"]["text"] = prompt_text
        workflow["negative_prompt"]["inputs"]["text"] = negative_prompt
        workflow["ksampler"]["inputs"]["seed"] = chosen_seed
        workflow["empty_latent"]["inputs"]["width"] = int(resolution[0])
        workflow["empty_latent"]["inputs"]["height"] = int(resolution[1])
        return workflow

    # ----- internals --------------------------------------------------------

    def _load(self, rel_path: str) -> dict[str, Any]:
        if rel_path not in self._cache:
            full_path = self.workflow_dir / rel_path
            if not full_path.exists():
                raise WorkflowTemplateError(
                    f"Workflow template not found: {full_path}\n"
                    f"See docs/COMFYUI_WORKFLOWS.md for step-by-step "
                    f"instructions on building each template."
                )
            try:
                with open(full_path) as f:
                    self._cache[rel_path] = json.load(f)
            except json.JSONDecodeError as exc:
                raise WorkflowTemplateError(
                    f"Workflow template {full_path} is not valid JSON: {exc}"
                ) from exc
        return copy.deepcopy(self._cache[rel_path])

    @staticmethod
    def _assert_required_nodes(
        workflow: dict[str, Any],
        family: str,
        template_name: str,
        required: tuple[str, ...],
    ) -> None:
        missing = [n for n in required if n not in workflow]
        if not missing:
            return
        raise WorkflowTemplateError(
            f"Workflow template {family}/{template_name} is missing required "
            f"semantic node IDs: {missing}. The template must contain these "
            f"top-level keys (see docs/COMFYUI_WORKFLOWS.md for the "
            f"{family}/{template_name} walkthrough): {list(required)}"
        )

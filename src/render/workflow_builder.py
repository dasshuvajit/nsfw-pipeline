"""External-template ComfyUI workflow loader.

The pipeline loads JSON workflows the user authors in ComfyUI, exports
as "Save (API Format)", renames 4 nodes to the contract IDs below, and
drops into ``config/comfyui_workflows/templates/<family>/``.

Per-render injection is minimal: prompt text, negative prompt text,
random seed, and resolution. Everything else (checkpoint, sampler,
scheduler, steps, cfg, VAE, CLIP, LoRAs, FaceDetailer, upscaling,
post-processing) is baked into the template by the author.

Required semantic node IDs (raise WorkflowTemplateError if missing):

    positive_prompt   inputs.text     CLIPTextEncode (or compatible)
    negative_prompt   inputs.text     CLIPTextEncode (or compatible)
    ksampler          inputs.seed     KSampler / SamplerCustomAdvanced /
                                      ClownsharKSampler_Beta / …
    empty_latent      inputs.width    EmptyLatentImage / EmptySD3LatentImage /
                      inputs.height   EmptyHunyuanLatentVideo / …

(The single-pass embedded-refiner contract was retired 2026-06-13; the
image-stage contract — ``build_image_stage`` for the SDXL refine / 4K
upscale stages — was archived 2026-08 with those stages, see legacy/.
``build_external`` is the only entrypoint.)
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# External templates — the only contract surviving the 2026-05-20 cleanup.
# These four top-level node IDs are required in every external template.
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

    Required-field pass — every node in `_REQUIRED_EXTERNAL_INPUTS` must
    expose every listed field.
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
    """Canonicalize a template CLI value (e.g. ``--base-template``,
    or a `render_pipeline` YAML value) into a
    string that ``WorkflowBuilder._load`` accepts.

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


class WorkflowBuilder:
    """External-template loader + injector.

    Only entrypoint is :meth:`build_external`. The user authors the
    workflow in ComfyUI, exports as API JSON, renames 4 nodes to the
    contract IDs, drops it under ``config/comfyui_workflows/templates/
    <family>/``. The builder loads, validates, injects the four
    contracted fields, and returns the rendered workflow.
    """

    def __init__(self, workflow_dir: Path | str) -> None:
        self.workflow_dir = Path(workflow_dir)
        self._cache: dict[str, dict[str, Any]] = {}

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
        specific message. There is no fallback.
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
                    f"instructions on authoring an external template."
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
            f"4-node contract): {list(required)}"
        )

"""External-template ComfyUI workflow loader.

After the 2026-05-20 cleanup, this module is external-templates-only —
the prior built-in `<family>/base.json` / `ipadapter.json` / `upscale.json`
templates + the per-family `_build_chroma` / `_build_flux` / `_build_flux2`
constructors + the IPAdapter + LoRA staging are all gone. The pipeline
now only loads JSON workflows the user authors in ComfyUI, exports as
"Save (API Format)", renames 4 nodes to the contract IDs below, and
drops into ``config/comfyui_workflows/templates/<family>/``.

Per-render injection is minimal: prompt text, negative prompt text,
random seed, and resolution. Everything else (checkpoint, sampler,
scheduler, steps, cfg, VAE, CLIP, LoRAs, IPAdapter, FaceDetailer,
upscaling, post-processing) is baked into the template by the author.

Required semantic node IDs (raise WorkflowTemplateError if missing):

    positive_prompt   inputs.text     CLIPTextEncode (or compatible)
    negative_prompt   inputs.text     CLIPTextEncode (or compatible)
    ksampler          inputs.seed     KSampler / SamplerCustomAdvanced /
                                      ClownsharKSampler_Beta / …
    empty_latent      inputs.width    EmptyLatentImage / EmptySD3LatentImage /
                      inputs.height   EmptyHunyuanLatentVideo / …

Optional refiner-stage IDs (2026-05-15) — validated only when present
in the workflow JSON. Templates without these IDs (single-stage
renders) pass the optional-loop as a no-op.

    refiner_positive_prompt    inputs.text   SDXL-CLIP-encoded copy of base text
    refiner_negative_prompt    (not patched) Template-owned; usually empty
    refiner_ksampler           inputs.seed   Same seed as base ksampler
    refiner_checkpoint_loader  (metadata-only) Pipeline reads inputs.ckpt_name
                                              OR inputs.unet_name for PNG meta
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

# Optional refiner-stage IDs (2026-05-15 — added for two-pass workflows
# like Chroma-base + SDXL-refiner). Validated only when present in the
# workflow JSON. Backward-compatible — templates without these IDs (the
# user's single-stage chroma_done_properly.json) pass the optional-loop
# as a no-op.
_OPTIONAL_NODES_EXTERNAL: tuple[str, ...] = (
    "refiner_positive_prompt",
    "refiner_negative_prompt",
    "refiner_ksampler",
    "refiner_checkpoint_loader",
)

_OPTIONAL_EXTERNAL_INPUTS: dict[str, tuple[str, ...]] = {
    "refiner_positive_prompt":     ("text",),
    # refiner_negative_prompt — no required fields (not patched).
    "refiner_ksampler":            ("seed",),
    # refiner_checkpoint_loader — neither field is REQUIRED (different
    # loader classes use different field names); presence-of-node is
    # the only check. Field lookup at metadata-extraction time is
    # tolerant of either ckpt_name or unet_name.
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

    Three passes:

    1. **Required-field pass** — every node in `_REQUIRED_EXTERNAL_INPUTS`
       must expose every listed field.
    2. **Optional-field pass** — for each node in
       `_OPTIONAL_NODES_EXTERNAL` that is present in the workflow,
       validate any required fields. Absent nodes are skipped silently.
    3. **Refiner-pair-consistency pass** — if either
       ``refiner_positive_prompt`` or ``refiner_ksampler`` is present,
       both must be. (``refiner_negative_prompt`` and
       ``refiner_checkpoint_loader`` are fully optional and don't
       participate in the pair check.) This rule catches a half-renamed
       template — the kind of bug where the user changes one ID and
       forgets the other, getting silent broken renders.
    """
    missing: list[str] = []

    # Pass 1 — required fields on required nodes.
    for node_id, field_keys in _REQUIRED_EXTERNAL_INPUTS.items():
        node = workflow.get(node_id, {})
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            missing.append(f"{node_id}.inputs (missing or not an object)")
            continue
        for field in field_keys:
            if field not in inputs:
                missing.append(f"{node_id}.inputs.{field}")

    # Pass 2 — required fields on present optional nodes.
    for node_id, field_keys in _OPTIONAL_EXTERNAL_INPUTS.items():
        if node_id not in workflow:
            continue
        node = workflow[node_id]
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

    # Pass 3 — refiner-pair consistency.
    has_refiner_pos = "refiner_positive_prompt" in workflow
    has_refiner_ks = "refiner_ksampler" in workflow
    if has_refiner_pos != has_refiner_ks:
        present, missing_id = (
            ("refiner_positive_prompt", "refiner_ksampler")
            if has_refiner_pos
            else ("refiner_ksampler", "refiner_positive_prompt")
        )
        raise WorkflowTemplateError(
            f"External template {template_name} has '{present}' but is "
            f"missing '{missing_id}'. The refiner pair must be present "
            f"together (refiner stage wired) or both absent (no-refiner "
            f"template). See docs/COMFYUI_WORKFLOWS.md § Refiner pipelines."
        )


def _resolve_template_path(
    raw: str, workflow_dir: Path, project_root: Path,
) -> str:
    """Canonicalize a ``--template`` CLI value (or a `default_template`
    YAML value) into a string that ``WorkflowBuilder._load`` accepts.

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

        Optional refiner stage: when the template wires
        ``refiner_positive_prompt`` + ``refiner_ksampler`` (the pair
        rule), they're patched with the SAME prompt text + SAME seed as
        the base nodes. ``refiner_negative_prompt`` is template-owned
        (NOT patched). ``refiner_checkpoint_loader`` is metadata-only
        (read for PNG `refiner_checkpoint` field, never written).

        Preflight (before injection):
          - File exists + is valid JSON (via ``_load``).
          - Top-level keys present: ``positive_prompt``, ``negative_prompt``,
            ``ksampler``, ``empty_latent``.
          - Each of those nodes carries the required input fields
            (``inputs.text``, ``inputs.seed``, ``inputs.width``,
            ``inputs.height``).
          - Refiner-pair consistency (if any refiner node present, both
            ``refiner_positive_prompt`` and ``refiner_ksampler`` must be).

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

        # Optional refiner stage — same prompt + same seed as base.
        # Patched only when the template wired in the refiner pair.
        # `refiner_negative_prompt` is fully template-owned (not
        # patched); `refiner_checkpoint_loader` is metadata-only.
        # The pair-rule check in _assert_external_template_inputs
        # already guarantees that if either side is present, both are.
        if "refiner_positive_prompt" in workflow:
            workflow["refiner_positive_prompt"]["inputs"]["text"] = prompt_text
        if "refiner_ksampler" in workflow:
            workflow["refiner_ksampler"]["inputs"]["seed"] = chosen_seed

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

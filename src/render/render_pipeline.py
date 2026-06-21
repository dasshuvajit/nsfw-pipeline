"""Resolve the staged render-pipeline config.

The staged pipeline (base -> refine -> manual 4K) is configured by a
``render_pipeline:`` block in ``config/pipeline.yaml`` with an optional
per-model override (``config/models/<id>.yaml::render_pipeline``) and
optional CLI overrides. Precedence, highest first:

    CLI flags  >  per-model render_pipeline  >  global render_pipeline  >  DEFAULTS

``base_resolution`` is merged per-orientation (a model that overrides only
``portrait`` still inherits ``square``/``landscape`` from the global block).
Everything else is a flat last-writer-wins merge.
"""

from __future__ import annotations

from typing import Any

# Hard defaults — used when a key is absent from every layer. Keep in sync
# with the documented block in config/pipeline.yaml.
DEFAULTS: dict[str, Any] = {
    "base_template": "templates/chroma/base.json",
    "refine_template": "templates/chroma/refine.json",
    "refine_template_t4": "templates/chroma/refine_T4.json",
    "upscale_template": "templates/sdxl/upscale_4k.json",
    # (upscale_template_t4 removed 2026-06-10 — the 4K stage is tier-neutral;
    #  the T4 variant had become byte-identical after the MPS detailer removal)
    # Base-only by default (2026-06-21): the SDXL detailer refine — like 4K — is
    # OPT-IN (art_series --refine). Keeps a series on a single resident model, no
    # zimage↔SDXL swap thrash. enable_refine stays a real switch for when it's asked.
    "enable_refine": False,
    "target_4k_long_edge": 3840,
    "base_resolution": {
        "portrait": [896, 1152],     # ~3:4 (7:9) native ~1MP bucket
        "square": [1024, 1024],      # 1:1
        "landscape": [1152, 896],    # ~4:3 (9:7) native ~1MP bucket
        "widescreen": [1360, 768],   # 16:9 cinematic (off-native, ~1MP, 4080 latent tokens)
        "story": [768, 1360],        # 9:16 vertical/social (off-native, ~1MP)
    },
}

_SCALAR_KEYS = (
    "base_template", "refine_template", "refine_template_t4", "upscale_template",
    "enable_refine", "target_4k_long_edge",
)


def resolve_render_pipeline(
    global_rp: dict[str, Any] | None = None,
    model_rp: dict[str, Any] | None = None,
    cli: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep-merge the render-pipeline layers and return the effective config.

    ``base_resolution`` is merged per-orientation across layers; all other
    keys are last-writer-wins. ``None`` values in a layer are ignored (so a
    CLI flag the user didn't pass doesn't clobber a lower layer).
    """
    out: dict[str, Any] = {k: (v.copy() if isinstance(v, dict) else v)
                           for k, v in DEFAULTS.items()}
    out["base_resolution"] = {k: list(v) for k, v in DEFAULTS["base_resolution"].items()}

    for layer in (global_rp, model_rp, cli):
        if not layer:
            continue
        for key in _SCALAR_KEYS:
            if layer.get(key) is not None:
                out[key] = layer[key]
        br = layer.get("base_resolution")
        if isinstance(br, dict):
            for orient, dims in br.items():
                if dims is not None:
                    out["base_resolution"][orient] = list(dims)
    return out


def base_resolution_for(rp: dict[str, Any], orientation: str) -> tuple[int, int]:
    """Return the (width, height) base resolution for an orientation,
    falling back to portrait then a hard 896x1152 if unknown."""
    table = rp.get("base_resolution") or DEFAULTS["base_resolution"]
    dims = table.get(orientation) or table.get("portrait") or [896, 1152]
    return int(dims[0]), int(dims[1])

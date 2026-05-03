#!/usr/bin/env python3
"""Render the same prompt against N models for a visual A/B/C comparison.

When you download a new checkpoint and want to see whether it beats
``juggernaut_ragnarok`` for realistic portraits, this is the tool. Give it
a prompt (or a character id), a comma-separated list of model ids, and
a seed — it renders ``--count`` images per model using the same seed
sequence so differences on screen are differences in the model, not
differences in the noise.

Two input modes:

  1. ``--prompt "..."`` — pass a raw prompt string. No character or
     style profile is consulted. No LoRAs. The cleanest possible
     apples-to-apples test of "what does this checkpoint look like
     on its own".

  2. ``--character char_001`` — look up the character row, dereference
     its style profile, and build the prompt from
     ``base_prompt + base_style_keywords``. Each candidate model
     renders with ITS OWN LoRA stack from its YAML
     (``config/models/{id}.yaml::lora_stack``) so the comparison
     reflects how each model performs at its best-tuned configuration
     — the point is "which model renders *my character* best" and the
     model-YAML LoRA stack is part of that answer. (Post-2026-04
     refactor: style profiles no longer carry a lora_stack; the
     checkpoint-plus-LoRAs tuning is bound to the model YAML.) If a
     candidate model has ``supports_lora=False`` in the registry, we
     warn and render it without LoRAs so the comparison still
     produces an image.

Output: ``output/comparisons/{timestamp}/{model_id}/{scene:02d}.png``.
Nothing is scored, nothing is persisted to the DB, and the character's
``total_images_generated`` is NOT incremented — this is a throwaway
visual check, not a real render.

Example:
    python scripts/compare_models.py \\
        --prompt "a beautiful portrait photo, studio lighting" \\
        --models juggernaut_ragnarok,lustify_v7,realvis_v4 \\
        --count 3 --seed 42

    python scripts/compare_models.py \\
        --character char_001 \\
        --models juggernaut_ragnarok,biglust_v16,lustify_v7 \\
        --count 2 --seed 123

    python scripts/compare_models.py \\
        --prompt "masterpiece, best quality, 1girl, portrait" \\
        --models perfection_realistic_ilxl,lustify_v7 \\
        --ratio portrait_23 --count 2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ratio_selector import (  # noqa: E402
    get_resolution,
    model_resolution_overrides,
)
from src.memory.model_registry import (  # noqa: E402
    ModelNotFound,
    ModelRegistryEntry,
    ModelRegistryLoader,
)
from src.render.comfyui_client import (  # noqa: E402
    ComfyUIClient,
    ComfyUIError,
)
from src.render.workflow_builder import (  # noqa: E402
    WorkflowBuilder,
    WorkflowTemplateError,
    _resolve_template_path,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
WORKFLOW_DIR = PROJECT_ROOT / "config" / "comfyui_workflows"

# Default comparison resolution: 1024x1024 is a safe midpoint for every
# SDXL checkpoint (they were all trained at this size). Picking a
# non-square ratio here would bias toward models that were LoRA-tuned on
# that ratio, defeating the point of the comparison.
_DEFAULT_RESOLUTION: tuple[int, int] = (1024, 1024)

# Fallback negative prompt used in --prompt mode (where no style profile
# is consulted). Matches the vibe of the seeded style_profiles without
# being tied to any specific one.
_DEFAULT_NEGATIVE = (
    "blurry, low quality, bad anatomy, distorted face, extra limbs, "
    "deformed hands, watermark, text, logo"
)


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"ERROR: config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _resolve_db_path(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    raw = (cfg.get("pipeline") or {}).get("db_path") or "nsfw_pipeline.db"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _load_character_context(
    db_path: Path, character_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch (character_row, style_profile_dict) as plain dicts.

    The character lives in the DB; its style profile lives in
    ``config/style_profiles.yaml`` and is loaded via
    ``StyleProfileLoader``. We return dicts in both cases so the
    downstream prompt composer doesn't care about the backing store.
    """
    from src.memory.style_profile_loader import (
        StyleProfileLoader, StyleProfileNotFound,
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        char_row = conn.execute(
            "SELECT * FROM characters WHERE id = ?", (character_id,)
        ).fetchone()
        if char_row is None:
            raise SystemExit(
                f"ERROR: character {character_id!r} not found. "
                f"Run `python scripts/bootstrap_character.py`."
            )
        char_dict = dict(char_row)
    finally:
        conn.close()

    try:
        profile = StyleProfileLoader().get_profile(char_dict["style_profile_id"])
    except StyleProfileNotFound:
        raise SystemExit(
            f"ERROR: character {character_id!r} points at "
            f"style_profile_id={char_dict['style_profile_id']!r} "
            f"which is missing from config/style_profiles.yaml."
        )
    # Post-2026-04 refactor: StyleProfile carries only aesthetic intent
    # (keywords + negatives + palette/lighting hints). No model_id, no
    # lora_stack, no sampler tuning — those live on the model YAML and
    # are resolved per-candidate inside _render_one_model.
    style_dict = {
        "id": profile.id,
        "name": profile.name,
        "base_style_keywords": profile.base_style_keywords,
        "base_negative_prompt": profile.base_negative_prompt,
    }
    return char_dict, style_dict


def _compose_prompt_from_character(
    char_row: dict[str, Any], style_row: dict[str, Any]
) -> tuple[str, str]:
    """Build (prompt_text, negative_prompt) from a character + style row.

    Mirrors the minimal part of :class:`PromptBuilder` that we need
    here — no scene fields, no extra keywords, just
    ``base_prompt, base_style_keywords`` comma-joined. We don't reuse
    PromptBuilder because it requires a scene dict and we don't have
    scenes in the comparison path.
    """
    segments = [char_row.get("base_prompt") or ""]
    keywords = style_row.get("base_style_keywords") or ""
    if keywords:
        segments.append(keywords)
    prompt_text = ", ".join(s.strip() for s in segments if s and s.strip())

    negative_prompt = (
        (char_row.get("negative_prompt") or "").strip()
        or (style_row.get("base_negative_prompt") or "").strip()
        or _DEFAULT_NEGATIVE
    )
    return prompt_text, negative_prompt


def _resolve_db_prompts_per_model(
    *,
    db_path: Path,
    series_id: str | None,
    scene_id: str | None,
    model_ids: list[str],
    llm_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load DB prompts for each model_id matching ``(series_id, scene_id)``.

    When ``llm_id`` is set, only prompts from that generating LLM are
    returned (matches the engine's run_phase_b filter). ``None`` skips
    the LLM filter (legacy / fresh-bootstrap path).

    Returns ``{model_id: [{prompt_text, negative_prompt, scene_id,
    aspect_ratio}, ...]}``. Empty list for a model means no prompts
    exist for it on the requested target.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for mid in model_ids:
            if scene_id:
                base = (
                    "SELECT p.prompt_text, p.negative_prompt, p.scene_id, "
                    "       s.aspect_ratio "
                    "FROM prompts p JOIN scenes s ON p.scene_id = s.id "
                    "WHERE p.scene_id = ? AND p.model_id = ?"
                )
                params: list[Any] = [scene_id, mid]
            else:
                base = (
                    "SELECT p.prompt_text, p.negative_prompt, p.scene_id, "
                    "       s.aspect_ratio "
                    "FROM prompts p JOIN scenes s ON p.scene_id = s.id "
                    "WHERE p.series_id = ? AND p.model_id = ?"
                )
                params = [series_id, mid]
            if llm_id is not None:
                base += " AND p.llm_id = ?"
                params.append(llm_id)
            if not scene_id:
                base += " ORDER BY p.scene_id"
            rows = conn.execute(base, params).fetchall()
            out[mid] = [dict(r) for r in rows]
    finally:
        conn.close()
    return out


def _validate_db_prompts(
    db_prompts: dict[str, list[dict[str, Any]]],
    *,
    series_id: str | None,
    scene_id: str | None,
) -> None:
    """Abort with a helpful hint if any model has zero prompts."""
    missing = [mid for mid, lst in db_prompts.items() if not lst]
    if not missing:
        return
    target_desc = (
        f"scene '{scene_id}'" if scene_id else f"series '{series_id}'"
    )
    raise SystemExit(
        f"ERROR: no prompts in DB for "
        f"{', '.join(repr(m) for m in missing)} on {target_desc}.\n"
        f"Run prepare_prompts first:\n"
        f"  python scripts/prepare_prompts.py --series-id "
        f"{series_id or '<series>'} --models {','.join(missing)}"
    )


def _build_db_render_tasks(
    *,
    db_prompts: list[dict[str, Any]],
    model: ModelRegistryEntry,
    base_seed: int,
    forced_ratio: str | None,
    fallback_resolution: tuple[int, int],
) -> list[dict[str, Any]]:
    """Convert DB prompt rows into render-task dicts for one model.

    For each row: pick the resolution from
      1. ``forced_ratio`` (CLI ``--ratio``) → per-target-model lookup
      2. ``row.aspect_ratio`` (the scene's stored ratio) → per-target-
         model lookup
      3. ``fallback_resolution`` (compare-mode default 1024×1024)

    Seed: ``base_seed + i`` so the i-th scene uses the same seed
    across every (model, template) pairing in this comparison run —
    same noise, different model = clean diff.
    """
    model_res_map = model_resolution_overrides(model)
    family_id = model.family

    tasks: list[dict[str, Any]] = []
    for i, row in enumerate(db_prompts):
        ratio = forced_ratio or row.get("aspect_ratio")
        if ratio:
            resolution = get_resolution(
                ratio,
                model.filename,
                model_resolutions=model_res_map,
                family_id=family_id,
            )
        else:
            resolution = fallback_resolution

        # Output filename label = scene_id tail (last 6 chars of the
        # uuid hex) so each rendered file is traceable back to its scene.
        scene_id = row.get("scene_id") or ""
        label = scene_id.split("_")[-1] if scene_id else f"{i:02d}"

        tasks.append({
            "prompt_text": row["prompt_text"],
            "negative_prompt": row["negative_prompt"],
            "seed": base_seed + i,
            "resolution": resolution,
            "label": label,
        })
    return tasks


def _build_text_render_tasks(
    *,
    prompt_text: str,
    negative_prompt: str,
    base_seed: int,
    count: int,
    resolution: tuple[int, int],
) -> list[dict[str, Any]]:
    """Convert a single (prompt, negative) into ``count`` render tasks
    at successive seeds — the prompt/character mode path."""
    return [
        {
            "prompt_text": prompt_text,
            "negative_prompt": negative_prompt,
            "seed": base_seed + i,
            "resolution": resolution,
            "label": f"{i:02d}",
        }
        for i in range(count)
    ]


def _resolve_models(
    loader: ModelRegistryLoader, ids: list[str]
) -> list[ModelRegistryEntry]:
    """Look up each id. Warn + skip missing/inactive rather than abort.

    A comparison across 4 models shouldn't be derailed by one stale id —
    we warn and continue with the survivors. If NONE survive we do
    abort, because there's nothing left to render.
    """
    resolved: list[ModelRegistryEntry] = []
    for mid in ids:
        try:
            resolved.append(loader.get_model(mid))
        except ModelNotFound as exc:
            print(
                f"WARN: skipping model {mid!r} — {exc}",
                file=sys.stderr,
            )
    if not resolved:
        raise SystemExit(
            "ERROR: none of the requested --models are valid. "
            "Run `python scripts/list_models.py` to see registered ids."
        )
    return resolved


def _build_style_profile(
    model: ModelRegistryEntry,
    *,
    lora_stack_json: str | None,
) -> SimpleNamespace:
    """Synthesize the flat-bag object ``WorkflowBuilder.build`` expects.

    Always pulls sampler/scheduler/steps/cfg from the registry — the
    whole point of comparison mode is "what does THIS checkpoint do at
    ITS default sampler settings", same logic as tier 1 in
    ``render_set.py``. We do NOT consult any style profile's per-model
    sampler overrides here.
    """
    return SimpleNamespace(
        workflow_family=model.family,
        model_checkpoint=model.filename,
        sampler=model.default_sampler,
        scheduler=model.default_scheduler,
        steps=model.default_steps,
        cfg=model.default_cfg,
        supports_ipadapter=model.supports_ipadapter,
        supports_lora=model.supports_lora,
        lora_stack=lora_stack_json,  # None in --prompt mode
    )


def _render_one_candidate(
    *,
    model: ModelRegistryEntry,
    template_token: str,
    slug: str,
    tasks: list[dict[str, Any]],
    lora_stack_json: str | None,
    builder: WorkflowBuilder,
    client: ComfyUIClient,
    out_dir: Path,
    timeout: int,
) -> tuple[int, int, float]:
    """Render every task in ``tasks`` for one (model, template) candidate.

    Each ``task`` dict carries ``prompt_text``, ``negative_prompt``,
    ``seed``, ``resolution``, and an optional ``label`` (used as the
    output filename stem, default = ``f"{i:02d}"``). The unified
    tasks list lets one function serve both:

      * Prompt/character mode — N copies of the same prompt at
        seed = base + i.
      * Series/scene mode — one task per (model, stored prompt)
        with that prompt's text + the scene's resolution.

    ``template_token`` is either ``'system'`` / ``'default'`` (built-in
    template, full injection via ``WorkflowBuilder.build``) or a path
    to an external template (strict 4-field injection via
    ``build_external`` — the template's baked-in checkpoint / LoRAs
    / sampler run as authored). Returns (ok, failed, seconds).
    """
    is_external = template_token.strip() not in ("system", "default")
    n_tasks = len(tasks)

    # Capability downgrade: comparison mode should be best-effort. If a
    # model can't carry LoRAs but the character's style profile has
    # them, drop the LoRAs for that model so we still get an image —
    # and print a clear warning so the user knows the comparison isn't
    # strictly apples-to-apples for this row. This applies to the
    # system-template path only; external templates carry their own
    # LoRAs (or don't) and the pipeline doesn't touch them.
    effective_lora: str | None = lora_stack_json
    if not is_external and lora_stack_json and not model.supports_lora:
        print(
            f"  WARN: model {model.id!r} has supports_lora=False — "
            f"rendering this row without LoRAs for the comparison."
        )
        effective_lora = None

    style_profile = _build_style_profile(
        model, lora_stack_json=effective_lora
    )
    candidate_out_dir = out_dir / f"{model.id}__{slug}"
    candidate_out_dir.mkdir(parents=True, exist_ok=True)

    # Banner. Resolution may differ per-task (series mode), so just show
    # the first task's value here.
    first_resolution = tasks[0]["resolution"] if tasks else (0, 0)
    if is_external:
        resolved_template = _resolve_template_path(
            template_token, builder.workflow_dir, PROJECT_ROOT,
        )
        print(
            f"\n[{model.id} @ {slug}] via {template_token} "
            f"@ {first_resolution[0]}x{first_resolution[1]} "
            f"(external — pipeline injects prompt/negative/seed/resolution only)"
        )
    else:
        resolved_template = None
        print(
            f"\n[{model.id} @ system] {model.filename} "
            f"({model.default_sampler}/{model.default_scheduler} "
            f"steps={model.default_steps} cfg={model.default_cfg}) "
            f"@ {first_resolution[0]}x{first_resolution[1]}"
        )

    ok = 0
    failed = 0
    t0 = time.time()
    for i, task in enumerate(tasks):
        prompt_text = task["prompt_text"]
        negative_prompt = task["negative_prompt"]
        seed_i = task["seed"]
        resolution = task["resolution"]
        label = task.get("label") or f"{i:02d}"

        print(
            f"  [{i + 1}/{n_tasks}] seed={seed_i} {label} "
            f"@ {resolution[0]}x{resolution[1]} ... ",
            end="", flush=True,
        )
        try:
            if resolved_template:
                workflow = builder.build_external(
                    external_template=resolved_template,
                    prompt_text=prompt_text,
                    negative_prompt=negative_prompt,
                    resolution=resolution,
                    seed=seed_i,
                )
            else:
                workflow = builder.build(
                    prompt_text=prompt_text,
                    negative_prompt=negative_prompt,
                    style_profile=style_profile,
                    resolution=resolution,
                    seed=seed_i,
                    ipadapter_image=None,  # comparison mode: no IPAdapter
                )
        except WorkflowTemplateError as exc:
            # Template errors are fatal for this candidate (they'd fire
            # on every task) — bail on this candidate, continue to the
            # next.
            print(f"FAIL (template): {exc}")
            failed += n_tasks - i
            break

        try:
            rendered = client.render_single_with_retry(
                workflow, max_attempts=1, timeout=timeout
            )
        except ComfyUIError as exc:
            print(f"FAIL: {exc}")
            failed += 1
            continue

        if not rendered:
            print("FAIL: no images returned")
            failed += 1
            continue

        first = rendered[0]
        dest = candidate_out_dir / f"{label}{first.file_path.suffix}"
        shutil.copy2(first.file_path, dest)
        print(f"OK {dest.relative_to(PROJECT_ROOT)}")
        ok += 1

    elapsed = time.time() - t0
    return ok, failed, elapsed


def _slug_for_template(token: str) -> str:
    """Derive the output-subdir slug for a template token.

    ``'system'`` and ``'default'`` collapse to the literal ``'system'``
    (they mean "use the built-in family template"). Any other value is
    treated as a path — its stem, stripped + lowercased, is the slug.
    """
    stripped = token.strip()
    if stripped in ("system", "default"):
        return "system"
    return Path(stripped).stem.strip().lower()


def _build_candidates(
    model_ids: list[str], template_tokens: list[str],
) -> list[tuple[str, str]]:
    """Flatten ``--models`` and ``--templates`` into a render list.

    Rules (as of Phase 5 — note the **breaking change** on N==N):

    | ``--models`` len | ``--templates`` len | Behavior                              |
    |:-:|:-:|:--|
    | N | (none)         | each model uses its ``'system'`` template (Cartesian × 1) |
    | N | 1              | every model uses that one template                        |
    | 1 | M              | the one model is rendered M times, once per template      |
    | **N | N**          | **NEW**: positional pairing — ``model[i]`` ↔ ``template[i]`` (was: Cartesian = N×N) |
    | N | M (mismatched, both >1) | error                                              |

    The N==N case used to produce N² renders (Cartesian); it now
    produces N (paired) renders. Users who want Cartesian should
    invoke ``compare_models`` once per template.
    """
    if not template_tokens:
        return [(m, "system") for m in model_ids]

    n_models = len(model_ids)
    n_templates = len(template_tokens)

    # Single template applied to all models (existing rule).
    if n_templates == 1:
        return [(m, template_tokens[0]) for m in model_ids]

    # Single model × multiple templates (existing rule).
    if n_models == 1:
        return [(model_ids[0], t) for t in template_tokens]

    # NEW: N == N → positional pairing.
    if n_models == n_templates:
        return list(zip(model_ids, template_tokens))

    # Mismatched lengths (both > 1) — explicit error.
    raise SystemExit(
        f"ERROR: --templates count ({n_templates}) must equal "
        f"--models count ({n_models}) for positional pairing, or be 1 "
        f"(applied to all). Got: models={model_ids}, "
        f"templates={template_tokens}.\n"
        f"Note: as of Phase 5, N×N is positional pairing (model[i] uses "
        f"template[i]), not Cartesian. To get Cartesian behavior, run "
        f"compare_models once per template."
    )


def _assign_slugs(
    candidates: list[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Append an output-subdir slug to each (model_id, token) candidate.

    Collision resolution: if the same ``model_id + slug`` has already
    been seen in this list, append ``__2``, ``__3``, ... until
    unique. Collisions across different ``model_id`` don't count —
    each model has its own output subdir.
    """
    used: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for model_id, token in candidates:
        base_slug = _slug_for_template(token)
        slug = base_slug
        counter = 2
        while f"{model_id}::{slug}" in used:
            slug = f"{base_slug}__{counter}"
            counter += 1
        used.add(f"{model_id}::{slug}")
        out.append((model_id, token, slug))
    return out


def _parse_template_tokens(args: argparse.Namespace) -> list[str]:
    """Collapse --template (singular) and --templates (CSV) into a list.

    Returns ``[]`` when neither flag is set — ``_build_candidates``
    converts that to the default ``'system'`` behavior.
    """
    if getattr(args, "template", None):
        return [args.template.strip()]
    if getattr(args, "templates", None):
        return [
            t.strip() for t in args.templates.split(",") if t.strip()
        ]
    return []


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the compare_models argparse parser.

    Extracted from main() so tests can exercise the argparse surface
    without the config / DB / ComfyUI plumbing around main().
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt", default=None,
        help="Raw prompt text. Renders with no LoRAs, no character. "
             "Same prompt for every model.",
    )
    input_group.add_argument(
        "--character", default=None,
        help="Character id — pulls base_prompt + style_profile LoRAs. "
             "Same prompt for every model.",
    )
    input_group.add_argument(
        "--series-id", default=None,
        help="Render every scene in this series. Each model uses ITS "
             "OWN per-(scene, model) prompt from the prompts table — "
             "no shared text. Run prepare_prompts first to populate "
             "prompts for each --models entry.",
    )
    input_group.add_argument(
        "--scene-id", default=None,
        help="Render the prompt for a single scene across every model. "
             "Each model uses ITS OWN per-(scene, model) prompt from "
             "the DB.",
    )
    parser.add_argument(
        "--models", required=True,
        help="Comma-separated list of model ids from config/models/ "
             "(e.g. 'juggernaut_ragnarok,lustify_v7').",
    )
    template_group = parser.add_mutually_exclusive_group(required=False)
    template_group.add_argument(
        "--template", default=None,
        help="Path to ONE external ComfyUI workflow template. "
             "Equivalent to `--templates <value>`. Requires exactly "
             "one --models entry (external templates are bound to the "
             "checkpoint they were authored against; N×M is rejected).",
    )
    template_group.add_argument(
        "--templates", default=None,
        help="Comma-separated list of templates to render the "
             "--models entry through. Each value is either 'system' / "
             "'default' (built-in template for the family) or a path "
             "to an external template. Only valid with a single "
             "--models entry when any value is non-'system'.",
    )
    parser.add_argument(
        "--count", type=int, default=3,
        help="Images per model (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base seed — per-image seed is (base + i). The same "
             "sequence is used for every model so differences are "
             "purely from the checkpoint. Default: a fresh random seed "
             "per invocation (logged in the banner) to avoid ComfyUI's "
             "execution cache when the same prompt+seed combination was "
             "already rendered in the current ComfyUI session. Pass an "
             "explicit value (e.g. --seed 42) for deterministic A/B runs.",
    )
    parser.add_argument(
        "--ratio",
        choices=("portrait_23", "portrait_916", "square", "landscape"),
        default=None,
        help="Force a single aspect ratio for all rendered images. "
             "When set, each model uses ITS OWN resolution for that ratio "
             "(from each model YAML's resolution_* fields) — Chroma's "
             "square is 1152x1152, SDXL's is 1024x1024. This is the same "
             "behavior as render_set.py --ratio. When unset (default), "
             "all models render at 1024x1024 for strict apples-to-apples "
             "comparison.",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Override DB path. Default: read from config/pipeline.yaml.",
    )
    parser.add_argument(
        "--comfyui-url", default=None,
        help="ComfyUI HTTP endpoint (default: pipeline.yaml).",
    )
    parser.add_argument(
        "--comfyui-output-dir", default=None,
        help="ComfyUI output dir (default: pipeline.yaml).",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Per-render timeout in seconds (default: pipeline.yaml).",
    )
    parser.add_argument(
        "--llm", default=None,
        help=(
            "Filter DB-loaded prompts to one generating LLM (id from "
            "config/llm_models.yaml). REQUIRED with --series-id / "
            "--scene-id / --character when the target has prompts from "
            "any LLM — silent fallback to one LLM is a footgun (plan "
            "§3.5b). Irrelevant for --prompt mode (no DB read)."
        ),
    )
    return parser


def main() -> int:
    cfg = _load_config()
    comfy_cfg = cfg.get("comfyui") or {}
    default_url = comfy_cfg.get("base_url", "http://127.0.0.1:8188")
    default_output_dir = comfy_cfg.get(
        "output_dir", str(Path.home() / "AI" / "apps" / "ComfyUI" / "output")
    )
    default_timeout = int(comfy_cfg.get("render_timeout_seconds", 300))

    parser = _build_arg_parser()
    args = parser.parse_args()
    # Fill in config-backed defaults that _build_arg_parser couldn't
    # reach (it's a pure function so tests can construct it without
    # loading pipeline.yaml).
    if args.comfyui_url is None:
        args.comfyui_url = default_url
    if args.comfyui_output_dir is None:
        args.comfyui_output_dir = default_output_dir
    if args.timeout is None:
        args.timeout = default_timeout

    if args.count <= 0:
        print(f"ERROR: --count must be > 0 (got {args.count})", file=sys.stderr)
        return 1

    # Seed resolution: explicit --seed wins; otherwise pick a fresh
    # random base so repeated runs don't collide on ComfyUI's per-session
    # execution cache (which returns cached outputs without producing
    # new files when every input is byte-identical to a prior prompt).
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        seed_source = "random"
    else:
        seed_source = "explicit"

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        print("ERROR: --models must contain at least one id", file=sys.stderr)
        return 1

    db_path = _resolve_db_path(cfg, args.db_path)
    if not db_path.exists():
        print(
            f"ERROR: database not found at {db_path}.\n"
            f"Run `python scripts/init_db.py` first.",
            file=sys.stderr,
        )
        return 1

    try:
        loader = ModelRegistryLoader(db_path)
    except Exception as exc:  # ModelRegistryError subclasses Exception
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        models = _resolve_models(loader, model_ids)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Templates — collapse --template / --templates into a token list
    # and build the (model, token) candidate list with N×M rejection.
    template_tokens = _parse_template_tokens(args)
    surviving_model_ids = [m.id for m in models]
    try:
        candidates = _build_candidates(surviving_model_ids, template_tokens)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    candidates_with_slugs = _assign_slugs(candidates)

    # Preflight external templates BEFORE the render loop. An unknown
    # path or missing semantic node IDs fail fast instead of after the
    # first render attempt.
    probe_builder = WorkflowBuilder(WORKFLOW_DIR)
    from src.render.workflow_builder import (  # local import to avoid cycle
        _REQUIRED_NODES_EXTERNAL,
        _assert_external_template_inputs,
    )
    for _mid, token, _slug in candidates_with_slugs:
        if token.strip() in ("system", "default"):
            continue
        try:
            resolved = _resolve_template_path(
                token, probe_builder.workflow_dir, PROJECT_ROOT,
            )
            probe_workflow = probe_builder._load(resolved)
            WorkflowBuilder._assert_required_nodes(
                probe_workflow, "external", Path(resolved).name,
                _REQUIRED_NODES_EXTERNAL,
            )
            _assert_external_template_inputs(
                probe_workflow, Path(resolved).name,
            )
        except WorkflowTemplateError as exc:
            print(
                f"ERROR: --template validation failed: {exc}",
                file=sys.stderr,
            )
            return 9

    # ── Dispatch on input mode ─────────────────────────────────────
    # Each mode resolves to a per-model dict of "render tasks" — small
    # plain dicts that ``_render_one_candidate`` consumes uniformly.
    apply_model_loras = bool(args.character)
    db_mode = bool(args.series_id or args.scene_id)

    surviving_model_ids = [m.id for m in models]

    if db_mode:
        # series-id / scene-id mode: each model has its OWN per-prompt
        # text from the prompts table. The same scene index across all
        # models uses the same base_seed offset (so noise is held
        # constant; the only variable is the model + its prompt text).
        # Validate --llm at the CLI boundary first.
        if args.llm is not None:
            from src.memory.llm_registry import (
                LLMRegistryLoader,
                LLMNotFound,
            )
            try:
                LLMRegistryLoader().get_llm(args.llm, require_active=True)
            except LLMNotFound as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(2)
        try:
            db_prompts_per_model = _resolve_db_prompts_per_model(
                db_path=db_path,
                series_id=args.series_id,
                scene_id=args.scene_id,
                model_ids=surviving_model_ids,
                llm_id=args.llm,
            )
            _validate_db_prompts(
                db_prompts_per_model,
                series_id=args.series_id,
                scene_id=args.scene_id,
            )
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2

        prompt_banner = (
            f"<DB-loaded; {sum(len(v) for v in db_prompts_per_model.values())} "
            f"prompts across {len(surviving_model_ids)} model(s)>"
        )
        mode_label = (
            f"series-id={args.series_id}" if args.series_id
            else f"scene-id={args.scene_id}"
        )
    elif args.character:
        try:
            char_row, style_row = _load_character_context(db_path, args.character)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2
        prompt_text, negative_prompt = _compose_prompt_from_character(
            char_row, style_row
        )
        mode_label = f"character={args.character}"
        prompt_banner = prompt_text[:100] + ("..." if len(prompt_text) > 100 else "")
    else:
        prompt_text = args.prompt.strip()
        if not prompt_text:
            print("ERROR: --prompt cannot be empty", file=sys.stderr)
            return 1
        negative_prompt = _DEFAULT_NEGATIVE
        mode_label = "prompt-only"
        prompt_banner = prompt_text[:100] + ("..." if len(prompt_text) > 100 else "")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / "output" / "comparisons" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Comparison run: {out_root.relative_to(PROJECT_ROOT)}")
    print(f"  mode:         {mode_label}")
    print(f"  models:       {', '.join(m.id for m in models)}")
    templates_banner = ", ".join(t for _, t, _ in candidates_with_slugs)
    print(f"  templates:    {templates_banner}")
    if db_mode:
        # per-model count varies in series mode; show the dict.
        per_model_counts = {
            mid: len(lst) for mid, lst in db_prompts_per_model.items()
        }
        print(f"  prompts/model: {per_model_counts}")
    else:
        print(f"  count/cand.:  {args.count}")
    print(
        f"  base seed:    {args.seed}  ({seed_source}; "
        f"per-image seed = base + i)"
    )
    if db_mode:
        if args.ratio is not None:
            print(f"  ratio:        {args.ratio} (override; per-model resolution)")
        else:
            print(f"  ratio:        per-scene (from scenes.aspect_ratio)")
    elif args.ratio is not None:
        print(f"  ratio:        {args.ratio} (per-model resolution)")
    else:
        print(f"  resolution:   {_DEFAULT_RESOLUTION[0]}x{_DEFAULT_RESOLUTION[1]}")
    print(
        f"  loras:        "
        + ("per-model (from model YAML)" if apply_model_loras else "no")
    )
    print(f"  prompt:       {prompt_banner}")
    print(f"  comfy:        {args.comfyui_url}")

    builder = WorkflowBuilder(WORKFLOW_DIR)
    client = ComfyUIClient(
        base_url=args.comfyui_url, output_dir=args.comfyui_output_dir
    )

    # Index models by id so the candidate loop can look each one up
    # without re-resolving against the registry.
    models_by_id = {m.id: m for m in models}

    overall_start = time.time()
    totals: list[tuple[str, str, int, int, float]] = []
    for model_id, token, slug in candidates_with_slugs:
        model = models_by_id[model_id]

        # Build render tasks — DB-mode loads per-prompt scene data;
        # text mode generates count×same-prompt at successive seeds.
        if db_mode:
            tasks = _build_db_render_tasks(
                db_prompts=db_prompts_per_model[model_id],
                model=model,
                base_seed=args.seed,
                forced_ratio=args.ratio,
                fallback_resolution=_DEFAULT_RESOLUTION,
            )
        else:
            if args.ratio is not None:
                resolution = get_resolution(
                    args.ratio,
                    model.filename,
                    model_resolutions=model_resolution_overrides(model),
                )
            else:
                resolution = _DEFAULT_RESOLUTION
            tasks = _build_text_render_tasks(
                prompt_text=prompt_text,
                negative_prompt=negative_prompt,
                base_seed=args.seed,
                count=args.count,
                resolution=resolution,
            )

        # Per-model LoRA resolution — character mode applies each
        # candidate's own enabled lora_stack (parsed by the registry into
        # {"name","strength"} dicts); prompt-only mode never applies LoRAs.
        # DB mode (series/scene) also doesn't apply LoRAs separately —
        # the prompt text already has any per-model trigger words baked in.
        if apply_model_loras and model.lora_stack:
            per_model_lora_json: str | None = json.dumps(list(model.lora_stack))
        else:
            per_model_lora_json = None

        ok, failed, elapsed = _render_one_candidate(
            model=model,
            template_token=token,
            slug=slug,
            tasks=tasks,
            lora_stack_json=per_model_lora_json,
            builder=builder,
            client=client,
            out_dir=out_root,
            timeout=args.timeout,
        )
        totals.append((model.id, slug, ok, failed, elapsed))

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 60)
    print(f"Comparison complete in {overall_elapsed:.1f}s")
    print(f"Output: {out_root}")
    print()
    width_label = max(len(f"{mid}__{slug}") for mid, slug, _, _, _ in totals)
    for model_id, slug, ok, failed, elapsed in totals:
        marker = "OK " if failed == 0 else "!! "
        label = f"{model_id}__{slug}"
        print(
            f"  {marker}{label.ljust(width_label)}  "
            f"{ok}/{ok + failed} ok  ({elapsed:.1f}s)"
        )

    total_failed = sum(f for _, _, _, f, _ in totals)
    return 0 if total_failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())

"""Pipeline engine — two-phase orchestrator.

Three phases per ARCHITECTURE.md Section 4:

  **Phase A — LLM (ComfyUI NOT in memory):**
    Build context → preflight → plan → scenes → enforce → filter →
    ratios → prompts → sanitize → dedup → save dry_run → supervised
    pause 1 → unload LLM.

  **Phase B — Render (LLM NOT in memory):**
    Memory preflight → render batch → score → select → postprocess →
    supervised pause 2.

  **Phase C — Package:**
    Set builder → metadata (brief LLM reload) → export → persist →
    memory record → mark complete.

The engine wires all components together. Individual components are
tested via their own modules; the engine's job is sequencing and
state management.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import psutil
import yaml

from src.agents.llm_client import OllamaClient
from src.agents.metadata_generator import MetadataGenerator
from src.agents.scene_facet_generator import (
    SceneFacetGenerator,
    SceneFacetGeneratorError,
)
from src.core.content_level import ContentLevelLoader
from src.core.generation_context import GenerationContext, build_context
from src.core.mode_selector import ModeSelector
from src.core.ratio_selector import (
    RatioSelector,
    get_resolution,
    model_resolution_overrides,
)
from src.core.style_profile_adapter import StyleProfileForWorkflow
from src.export.exporter import Exporter
from src.filter.set_builder import SetBuilder, SetTooSmall
from src.memory.character_manager import CharacterManager, CharacterNotFound
from src.memory.memory_manager import MemoryManager
from src.memory.scene_facets_repo import (
    delete_facets_for_family,
    get_facet,
    has_facet,
    insert_facet,
)
from src.modes.base_mode import BaseMode
from src.modes.character_mode import CharacterMode
from src.modes.niche_mode import NicheMode
from src.modes.style_mode import StyleMode
from src.modes.theme_mode import ThemeMode
from src.modes.variation_mode import VariationMode
from src.prompt.builder import PromptBuilder, compute_prompt_hash
from src.prompt.deduplicator import PromptDeduplicator
from src.prompt.sanitizer import PromptSanitizer
from src.prompt.vocabulary import VocabularyLoader
from src.render.comfyui_client import ComfyUIClient
from src.render.metadata import (
    build_a1111_parameters,
    build_pipeline_metadata,
    write_png_metadata,
)
from src.render.workflow_builder import (
    WorkflowBuilder,
    WorkflowTemplateError,
    _REQUIRED_NODES_EXTERNAL,
    _assert_external_template_inputs,
    _resolve_template_path,
)
from src.postprocess.face_refiner import FaceRefiner, FaceRefineError
from src.postprocess.upscaler import Upscaler, UpscaleError
from src.postprocess.watermarker import Watermarker
from src.review.supervisor import Supervisor, SupervisorAbort
from src.scoring.image_scorer import ImageScorer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE_YAML = _PROJECT_ROOT / "config" / "pipeline.yaml"


class EngineError(Exception):
    """Fatal engine error."""


class InsufficientMemoryError(EngineError):
    """Not enough free memory for the render phase."""


class PreflightError(EngineError):
    """A preflight check failed."""


def _load_config() -> dict:
    if _PIPELINE_YAML.exists():
        with open(_PIPELINE_YAML) as f:
            return yaml.safe_load(f) or {}
    return {}


def _model_subfolder(family: str) -> str:
    """Return the ComfyUI ``models/<subfolder>/`` that holds this family's weights.

    - ``chroma`` / ``flux``: GGUF loaded via ``UnetLoaderGGUF`` → ``unet/``.
    - ``flux2``: safetensors loaded via ``UNETLoader`` (BFL convention) →
      ``diffusion_models/``.
    - Everything else (SDXL, Pony, Illustrious): ``CheckpointLoaderSimple`` →
      ``checkpoints/``.
    """
    if family in ("chroma", "flux"):
        return "unet"
    if family == "flux2":
        return "diffusion_models"
    return "checkpoints"


def _resolve_db_path(cfg: dict) -> Path:
    raw = cfg.get("pipeline", {}).get("db_path", "nsfw_pipeline.db")
    return _PROJECT_ROOT / raw


def _load_style_profile(db_path: Path, style_profile_id: str) -> dict[str, Any]:
    """Return a style profile as a dict (YAML-sourced since 2026-04).

    ``db_path`` is retained for signature compatibility but unused —
    style profiles live in ``config/style_profiles.yaml`` now. LoRA
    stacks are re-emitted as a JSON string to match the shape the rest
    of the engine expects (WorkflowBuilder, save_dry_run, etc.).
    """
    from src.memory.style_profile_loader import (
        StyleProfileLoader,
        StyleProfileNotFound,
    )

    try:
        p = StyleProfileLoader().get_profile(style_profile_id)
    except StyleProfileNotFound as exc:
        raise EngineError(str(exc)) from exc

    # Post-2026-04 refactor: style profiles carry only aesthetic intent.
    # Render tuning (sampler/scheduler/steps/cfg/clip_skip/lora_stack)
    # now resolves from the model YAML via ModelRegistryEntry.default_*.
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "base_style_keywords": p.base_style_keywords,
        "base_negative_prompt": p.base_negative_prompt,
        "palette_hint": p.palette_hint,
        "lighting_hint": p.lighting_hint,
        "suited_tiers": list(p.suited_tiers),
        "suited_families": list(p.suited_families),
        # Kept for signature parity with WorkflowBuilder's LoRA stage;
        # archetypes declare an empty stack — per-model LoRA defaults
        # go on the model YAML's prompt.extend hook.
        "lora_stack": None,
    }


class PipelineEngine:
    """Main orchestrator — runs a full plan → render → package cycle.

    Parameters
    ----------
    config : dict
        Full pipeline.yaml config.
    db_path : Path
        Path to the SQLite database.
    dry_run : bool
        If True, run Phase A only (no rendering).
    model_override : str | None
        Override model_id from CLI.
    force_mode : str | None
        Force a specific mode (default: character).
    force_character : str | None
        Force a specific character id.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        db_path: Path | None = None,
        *,
        dry_run: bool = False,
        model_override: str | None = None,
        force_mode: str | None = None,
        force_character: str | None = None,
        template_override: str | None = None,
    ) -> None:
        self.config = config or _load_config()
        self.db_path = db_path or _resolve_db_path(self.config)
        self.dry_run = dry_run
        self.model_override = model_override
        self.force_mode = force_mode
        self.force_character = force_character
        self.template_override = template_override

        # Fail fast on a typo'd --model rather than surfacing it mid-
        # Phase-A (by then the LLM has already been pinged and the
        # pipeline has done non-trivial setup work).
        self._commercial_mode = bool(
            self.config.get("compliance", {}).get("commercial_mode", False)
        )
        if model_override:
            from src.memory.model_registry import (
                ModelNotFound,
                ModelRegistryLoader,
            )
            try:
                ModelRegistryLoader(
                    self.db_path, commercial_mode=self._commercial_mode,
                ).get_model(model_override)
            except ModelNotFound as exc:
                raise EngineError(
                    f"--model {model_override!r} is not a valid model id: "
                    f"{exc}"
                ) from exc

        self.execution_mode = self.config.get("execution", {}).get("mode", "supervised")

        # Components (lazy-initialized where expensive)
        self.llm_client = OllamaClient()
        # LLM registry + router. The router resolves per-role / per-family
        # LLM choices from `pipeline.yaml::llm.routing` (empty by default
        # → every role gets default_llm). At Phase A call sites, the
        # engine passes `cli_llm_override` from `--llm <id>` so the
        # override → routing → default chain runs per call. Validation
        # (every routing target points to an active registry entry) fires
        # at construction time — typo in routing config → fail fast here.
        from src.memory.llm_registry import LLMRegistryLoader
        from src.agents.llm_router import LLMRouter
        self._llm_registry = LLMRegistryLoader()
        self._llm_router = LLMRouter(
            self._llm_registry,
            routing_config=self.config.get("llm", {}).get("routing") or {},
        )
        self._default_llm_id = self._llm_registry.default_llm_id
        self.mode_selector = ModeSelector(self.config)
        self.prompt_builder = PromptBuilder()
        self.ratio_selector = RatioSelector.from_config(
            self.config,
            ratio_signals_path=_PROJECT_ROOT / "config" / "ratio_signals.yaml",
        )
        self.deduplicator = PromptDeduplicator(self.db_path)
        self.memory = MemoryManager(self.db_path)
        self.supervisor = Supervisor(
            auto_approve=(self.execution_mode != "supervised" or self.dry_run)
        )

        # Mode registry — all 5 modes per ARCHITECTURE.md Sections 7–11.
        # Each mode receives the LLMRouter so its plan() / generate_scenes()
        # can resolve role → ollama_id with the override→routing→default
        # chain. Without this, per-role routing only fires for the
        # SceneFacetGenerator (which the engine resolves directly).
        self._mode_registry: dict[str, BaseMode] = {
            "character": CharacterMode(self.llm_client, self._llm_router),
            "theme": ThemeMode(self.llm_client, self._llm_router),
            "style": StyleMode(self.llm_client, self._llm_router),
            "niche": NicheMode(self.llm_client, self._llm_router),
            "variation": VariationMode(self.llm_client, self._llm_router),
        }

        # Render-side components (not needed for dry run)
        comfy_cfg = self.config.get("comfyui", {})
        self.comfy_base_url = comfy_cfg.get("base_url", "http://127.0.0.1:8188")
        self.comfy_output_dir = Path(
            comfy_cfg.get("output_dir", "~/AI/apps/ComfyUI/output")
        ).expanduser()
        self.comfy_input_dir = Path(
            comfy_cfg.get("input_dir", "~/AI/apps/ComfyUI/input")
        ).expanduser()
        self.workflow_dir = _PROJECT_ROOT / comfy_cfg.get(
            "workflow_dir", "config/comfyui_workflows"
        )
        self.render_timeout = comfy_cfg.get("render_timeout_seconds", 300)
        self.max_retry = comfy_cfg.get("max_retry_per_image", 3)

        set_cfg = self.config.get("set_builder", {})
        self.quality_cutoff = set_cfg.get("quality_cutoff", 0.55)

        # Postprocess config
        pp_cfg = self.config.get("postprocess", {})
        self.upscale_enabled = pp_cfg.get("upscale_enabled", False)
        self.face_refine_enabled = pp_cfg.get("face_refine_enabled", False)
        self.upscale_model = pp_cfg.get("upscale_model", "4x-UltraSharp.pth")
        self.face_denoise = pp_cfg.get("face_denoise", 0.35)

        # Scoring config — Phase G adds opt-in HPS v2 + ImageReward.
        score_cfg = self.config.get("scoring", {})
        self.use_hps_v2 = bool(score_cfg.get("use_hps_v2", False))
        self.use_image_reward = bool(score_cfg.get("use_image_reward", False))

        # Watermark config
        self.watermarker = Watermarker(self.config.get("watermark", {}))

        self.output_dir = _PROJECT_ROOT / self.config.get("pipeline", {}).get(
            "output_dir", "output"
        )

    def run_cycle(
        self,
        *,
        content_level: str = "T2_implied",
        style_profile_id: str | None = None,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Execute a full pipeline cycle (single model, end-to-end).

        Composition wrapper:
          1. Build the baseline ``GenerationContext``.
          2. ``run_phase_a(ctx, models=[ctx.model_id])`` — LLM phase
             (plan + scenes + per-(scene,family) facet + per-model
             prompts). LLM unloaded at end.
          3. ``run_phase_b(series_id, model_id)`` — render + score +
             postprocess.
          4. ``run_phase_c(...)`` — set build + watermark + export +
             persist + memory + run_log.

        For multi-model production runs, use the ``prepare_prompts``
        and ``render_prompts`` CLIs (which call ``run_phase_a`` with
        ``--models a,b,c`` and then loop ``run_phase_b`` per model).

        ``cli_llm_override`` (Phase 4) — registry id from ``--llm <id>``.
        When set, every agent role uses this LLM (full pipeline single-LLM
        path). When None, the LLMRouter falls back through the resolution
        chain (routing → default_llm).

        Returns a summary dict with series_id, status, counts, paths.
        """
        run_start = time.time()

        # Resolution-display logging (plan §3.2c). Print the table to
        # stderr so the user sees what each role resolves to before any
        # agent fires. Particularly important when --llm overrides
        # explicit routing config — the table shows what was overridden.
        try:
            table = self._llm_router.format_resolution_table(cli_llm_override)
            if table:
                logger.info("\n%s", table)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Failed to render LLM resolution table: %s", exc)

        # cli_llm_override flows through every agent call site below
        # via the mode constructors (which received the router) and
        # through the engine's direct router lookups (SceneFacetGenerator
        # in run_phase_a; MetadataGenerator in run_phase_c). Every agent
        # call gets its model resolved per-call from the override →
        # routing → default chain — OllamaClient is now a pure transport
        # with no fallback model state of its own.

        pipe_cfg = self.config.get("pipeline", {})
        if style_profile_id is None:
            style_profile_id = pipe_cfg.get(
                "default_style_profile_id", "golden_hour_natural"
            )

        # ── Build context ──────────────────────────────────────────
        mode_name = self.mode_selector.select(force_mode=self.force_mode)
        style_profile = _load_style_profile(self.db_path, style_profile_id)
        content_rules = ContentLevelLoader(self.db_path).load(content_level)

        # Resolve baseline model_id: character row (if character mode +
        # forced) > pipeline.default_model_id. --model CLI override is
        # applied inside build_context.
        baseline_character: dict[str, Any] | None = None
        if self.force_character and mode_name == "character":
            baseline_character = CharacterManager(self.db_path).get_character(
                self.force_character
            )

        default_model_id = pipe_cfg.get("default_model_id")
        if not default_model_id:
            raise EngineError(
                "pipeline.default_model_id is not set in config/pipeline.yaml"
            )
        resolved_model_id = (
            (baseline_character or {}).get("model_id") or default_model_id
        )

        ctx = build_context(
            mode=mode_name,
            content_level=content_level,
            execution_mode=self.execution_mode,
            style_profile=style_profile,
            content_rules=content_rules,
            db_path=self.db_path,
            model_id=resolved_model_id,
            model_override=self.model_override,
            commercial_mode=self._commercial_mode,
        )
        if baseline_character is not None:
            ctx.character = baseline_character
            ctx.character_id = self.force_character

        logger.info(
            "Pipeline cycle: mode=%s, level=%s, model=%s (%s), exec=%s",
            ctx.mode, ctx.content_level, ctx.model_id,
            ctx.model_config.family, ctx.execution_mode,
        )

        # ── Phase A: LLM ───────────────────────────────────────────
        # cli_llm_override threads to mode.plan / mode.generate_scenes
        # (each mode resolves series_planner / scene_generator via
        # router internally) AND to facet generation (via the engine's
        # router.resolve_facet_family in run_phase_a's per-model loop).
        phase_a_result = self.run_phase_a(
            ctx,
            models=[ctx.model_id],
            style_profile=style_profile,
            style_profile_id=style_profile_id,
            cli_llm_override=cli_llm_override,
        )
        if phase_a_result["status"] == "aborted":
            return {"series_id": phase_a_result["series_id"], "status": "aborted"}
        series_id = phase_a_result["series_id"]

        # Resolve the registry id used by Phase A (for downstream DB
        # filters in Phase B / dry-run summary). Single-LLM-per-cycle:
        # series_planner role is representative of the whole cycle when
        # --llm override is set; with routing it may differ per agent
        # but the prompts row's llm_id is whatever was used at
        # _save_prompts_for_model time.
        cycle_llm_id = self._llm_router.resolve_role(
            "series_planner", override=cli_llm_override,
        ).id

        if self.dry_run:
            # Reload from DB so the summary is built from persisted data
            # (consistent with what Phase B would see if it ran).
            series_plan, scenes = self._load_series_for_retarget(series_id)
            prompts = self._load_prompts_for_summary(
                series_id, ctx.model_id, cycle_llm_id,
            )
            return self._dry_run_summary(
                series_id, ctx, series_plan, scenes, prompts
            )

        # ── Phase B: Render ────────────────────────────────────────
        rendered_images, ctx_state = self.run_phase_b(
            series_id=series_id,
            model_id=ctx.model_id,
            template_override=self.template_override,
            cli_llm_override=cli_llm_override,
        )

        if ctx_state.get("aborted"):
            return {"series_id": series_id, "status": "aborted"}
        if not rendered_images:
            return {
                "series_id": series_id,
                "status": "failed",
                "reason": "zero rendered",
            }

        # ── Phase C: Package ───────────────────────────────────────
        elapsed = time.time() - run_start
        return self.run_phase_c(
            series_id=series_id,
            model_id=ctx_state["ctx"].model_id,
            content_level=content_level,
            rendered_images=rendered_images,
            series_plan=ctx_state["series_plan"],
            scenes=ctx_state["scenes"],
            prompts=ctx_state["prompts"],
            ctx=ctx_state["ctx"],
            style_profile=ctx_state["style_profile"],
            style_profile_id=ctx_state["style_profile_id"],
            elapsed_seconds=elapsed,
            llm_id=ctx_state.get("llm_id", self._default_llm_id),
            cli_llm_override=cli_llm_override,
        )

    def _load_prompts_for_summary(
        self, series_id: str, model_id: str, llm_id: str,
    ) -> list[dict[str, Any]]:
        """Cheap helper used by ``run_cycle`` dry-run path to read
        prompts back from the DB after Phase A persists them.

        Filters by ``llm_id`` so multi-LLM data on the same series
        doesn't silently merge into the dry-run summary.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM prompts "
                "WHERE series_id = ? AND model_id = ? AND llm_id = ?",
                (series_id, model_id, llm_id),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════
    # PHASE A — series plan + scene gen + per-(scene,family) facet
    #          + per-model prompts (multi-model fan-out)
    # ═══════════════════════════════════════════════════════════════

    def run_phase_a(
        self,
        ctx: GenerationContext,
        *,
        models: list[str],
        series_id_existing: str | None = None,
        regen_facets: list[str] | None = None,
        regen_prompts: list[str] | None = None,
        style_profile: dict[str, Any] | None = None,
        style_profile_id: str | None = None,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Phase A — LLM planning + per-model prompt persistence.

        Two callers today:
          1. ``run_cycle`` → ``models=[ctx.model_id]`` (single-model
             behavior identical to pre-refactor).
          2. ``prepare_prompts.py`` (new in Phase 4) → multi-model
             fan-out, optional re-target via ``series_id_existing``.

        Internal flow:
          - **Preflight** (always, baseline ctx).
          - **If new series**: SeriesPlanner.plan + SceneGenerator.generate
            + assign IDs + assign initial aspect ratios. Persist series
            and scenes rows once.
          - **If re-target**: load series + scenes from DB; reuse
            ``series.llm_series_plan`` JSON for niche `keyword_cluster`
            and variation `source_mode/character_id` lookups.
          - **For each model in models**:
              - Build per-model ctx (look up model → entry → family →
                prompt_guide).
              - If ``regen_facets`` includes this family: bulk-DELETE
                its facet rows across the series's scenes.
              - For each scene: call ``SceneFacetGenerator`` if the
                facet row is missing, else reuse the persisted facet.
              - If ``regen_prompts`` includes this model: DELETE its
                existing prompts for this series before composing.
              - For each scene: merge facet into scene → PromptBuilder
                → sanitize → assemble negative → prompts table INSERT.
              - In supervised mode, call ``Supervisor.approve_plan``
                once per model (one human review per model is the
                natural unit).
          - **Single LLM unload at end** (regardless of how many models
            ran).

        Skips ``run_log`` writes — Phase A is not a render run. The
        prompts table itself records what was generated.

        Returns
        -------
        dict
            ``{series_id, status, scenes_created, facets_created,
              prompts_created, models}``. ``status`` is "complete" or
            "aborted" (supervisor rejected one model).
        """
        # ── 1. Resolve series_id + load-or-generate series + scenes ─
        is_new_series = series_id_existing is None
        if is_new_series:
            series_id = f"series_{uuid.uuid4().hex[:12]}"
        else:
            series_id = series_id_existing
            assert series_id is not None  # for type-checkers

        # Resolve style_profile (may be supplied by caller or fall back
        # to ctx). Re-targeting reads from the existing series row.
        if style_profile is None or style_profile_id is None:
            if not is_new_series:
                conn = sqlite3.connect(str(self.db_path))
                conn.row_factory = sqlite3.Row
                try:
                    row = conn.execute(
                        "SELECT style_profile_id FROM series WHERE id = ?",
                        (series_id,),
                    ).fetchone()
                finally:
                    conn.close()
                if row is None:
                    raise EngineError(
                        f"Series {series_id!r} not found in DB "
                        f"(can't re-target)."
                    )
                if style_profile_id is None:
                    style_profile_id = row["style_profile_id"]
                if style_profile is None:
                    style_profile = _load_style_profile(
                        self.db_path, style_profile_id
                    )
            else:
                # Fresh series with no caller-supplied profile.
                if style_profile is None:
                    style_profile = ctx.style_profile or {}
                if style_profile_id is None:
                    raise EngineError(
                        "run_phase_a needs style_profile_id for new series. "
                        "Pass it explicitly."
                    )

        logger.info("=" * 60)
        logger.info(
            "Phase A: series=%s (%s), models=%s",
            series_id,
            "new" if is_new_series else "re-target",
            ",".join(models),
        )
        logger.info("=" * 60)

        # ── 2. Preflight (baseline ctx — series-level concerns) ────
        self._preflight(ctx)

        # ── 3. New-series path: plan + generate scenes + persist ────
        if is_new_series:
            mode = self._mode_registry.get(ctx.mode)
            if mode is None:
                raise EngineError(
                    f"Mode {ctx.mode!r} not implemented. "
                    f"Available: {list(self._mode_registry)}"
                )

            series_plan = mode.plan(ctx, cli_llm_override=cli_llm_override)
            logger.info("Plan: theme=%r", series_plan.get("theme"))

            scenes = mode.generate_scenes(
                series_plan, ctx, cli_llm_override=cli_llm_override,
            )
            logger.info("Scenes: %d generated", len(scenes))

            # Assign scene IDs + content_level
            for i, scene in enumerate(scenes):
                scene["id"] = f"{series_id}_scene_{i:03d}"
                scene["content_level"] = ctx.content_level

            # Assign aspect ratios + initial resolutions (recomputed
            # per-target-model at render time in run_phase_b).
            baseline_model_res = model_resolution_overrides(ctx.model_config)
            baseline_family_id = getattr(ctx.model_config, "family", None)
            for scene in scenes:
                pick = self.ratio_selector.select(
                    scene, ctx.content_level,
                    model_checkpoint=ctx.model_config.filename,
                    model_resolutions=baseline_model_res,
                    family_id=baseline_family_id,
                )
                scene["aspect_ratio"] = pick.ratio
                scene["resolution_w"] = pick.resolution[0]
                scene["resolution_h"] = pick.resolution[1]

            # Persist series + scenes (no prompts yet — those come in
            # the per-model loop below). FK enforcement: scene_facets
            # and prompts both reference scenes(id).
            self._save_series_and_scenes(
                series_id=series_id,
                ctx=ctx,
                series_plan=series_plan,
                scenes=scenes,
                style_profile_id=style_profile_id,
            )
        else:
            # Re-target path: load series + scenes from DB.
            series_plan, scenes = self._load_series_for_retarget(series_id)
            logger.info(
                "Re-target: loaded %d scenes from existing series %s",
                len(scenes), series_id,
            )

        # ── 4. Per-model fan-out ────────────────────────────────────
        sanitizer = PromptSanitizer.from_rules(ctx.content_rules)
        facet_gen = SceneFacetGenerator(self.llm_client)
        facets_created_total = 0
        prompts_created_total = 0
        models_completed: list[str] = []

        for model_id in models:
            # Per-model ctx (model_config + family + prompt_guide vary).
            try:
                model_ctx = build_context(
                    mode=ctx.mode,
                    content_level=ctx.content_level,
                    execution_mode=ctx.execution_mode,
                    style_profile=style_profile,
                    content_rules=ctx.content_rules,
                    db_path=self.db_path,
                    model_id=model_id,
                    commercial_mode=self._commercial_mode,
                )
            except Exception as exc:
                raise EngineError(
                    f"Could not build context for model {model_id!r}: {exc}"
                ) from exc

            # Carry over character from baseline ctx.
            if ctx.character is not None:
                model_ctx.character = ctx.character
                model_ctx.character_id = ctx.character_id

            family = model_ctx.family
            guide = model_ctx.model_prompt_guide

            # Resolve the LLM for this family's facet generation. The
            # registry id (`facet_llm_id`) is what gets stamped on every
            # facet + prompt row; the ollama tag (`facet_ollama_id`) is
            # what OllamaClient.generate consumes. With --llm override,
            # both flow from a single registry lookup.
            facet_llm_entry = self._llm_router.resolve_facet_family(
                family.prompt_style, override=cli_llm_override,
            )
            facet_llm_id = facet_llm_entry.id
            facet_ollama_id = facet_llm_entry.ollama_id

            logger.info(
                "Model %s (family=%s, llm=%s): facets + prompts …",
                model_id, family.id, facet_llm_id,
            )

            # Optional regen-facets for this family.
            if regen_facets and family.id in regen_facets:
                scene_ids = [s["id"] for s in scenes]
                n = delete_facets_for_family(
                    self.db_path,
                    scene_ids,
                    family.id,
                    facet_llm_id,
                )
                logger.info(
                    "regen_facets: deleted %d %s facets (llm=%s) across "
                    "%d scenes",
                    n, family.id, facet_llm_id, len(scene_ids),
                )

            # Optional regen-prompts for this model.
            if regen_prompts and model_id in regen_prompts:
                n = self._delete_prompts_for_model(
                    series_id, model_id, facet_llm_id,
                )
                logger.info(
                    "regen_prompts: deleted %d existing prompts for "
                    "(%s, %s, %s)",
                    n, series_id, model_id, facet_llm_id,
                )

            # Per-scene: ensure facet + compose prompt.
            mode_obj = self._mode_registry.get(model_ctx.mode)
            extra_keywords: list[str] | None = None
            if isinstance(mode_obj, NicheMode):
                extra_keywords = mode_obj.get_prompt_keywords(series_plan)

            prompts_for_model: list[dict[str, Any]] = []
            for scene in scenes:
                scene_id = scene["id"]

                # Step 1: facet (lookup or generate).
                facet: dict[str, Any]
                if has_facet(
                    self.db_path, scene_id, family.id, facet_llm_id,
                ):
                    facet = get_facet(
                        self.db_path,
                        scene_id,
                        family.id,
                        facet_llm_id,
                    )
                else:
                    try:
                        # Phase A — surface content_level + tier-directive
                        # to the SceneFacetGenerator so it generates
                        # tier-appropriate prose (T4 explicit, T1 SFW, etc.)
                        # Per-family Ollama tag flows in via the router.
                        facet = facet_gen.generate(
                            scene=scene,
                            family=family,
                            content_level=ctx.content_level,
                            prompt_guide=guide,
                            llm_directive=ctx.content_rules.llm_directive,
                            model=facet_ollama_id,
                        )
                        # Persist (Flux2 QA fields auto-dropped by repo).
                        insert_facet(
                            self.db_path,
                            scene_id,
                            family.id,
                            facet,
                            facet_llm_id,
                        )
                        facets_created_total += 1
                    except SceneFacetGeneratorError as exc:
                        logger.warning(
                            "Facet gen failed for scene %s family %s: %s — "
                            "composer falls back to universal segment "
                            "assembly", scene_id, family.id, exc,
                        )
                        facet = {}

                # Step 2: merge facet into scene dict for the composer.
                scene_with_facet = {**scene, **facet}
                # Drop facet metadata keys.
                for k in ("scene_id", "family", "created_at"):
                    scene_with_facet.pop(k, None)

                # Step 3: resolve character (synthetic for non-character modes).
                scene_character = self._resolve_character_for_scene(
                    mode_name=model_ctx.mode,
                    base_character=model_ctx.character,
                    series_plan=series_plan,
                    style_profile=style_profile,
                    scene=scene,
                )

                # Step 4: compose prompt.
                try:
                    prompt_dict = self.prompt_builder.build_one(
                        scene_character, scene_with_facet, style_profile,
                        extra_keywords=extra_keywords,
                        family=family,
                        trigger_words=guide.trigger_words if guide else None,
                        avoid_words=guide.avoid_words if guide else None,
                        content_level=ctx.content_level,
                    )
                    prompt_dict["prompt_text"] = sanitizer.sanitize_text(
                        prompt_dict["prompt_text"]
                    )

                    # Phase D negative assembly with conflict-axis filter.
                    guide_axes = (
                        guide.negative_axes
                        if guide and guide.negative_axes
                        else None
                    )
                    prompt_dict["negative_prompt"] = (
                        self.prompt_builder.assemble_negative_prompt(
                            model_negative=(
                                guide.base_negative_prompt
                                if guide and not guide_axes
                                else None
                            ),
                            model_negative_axes=guide_axes,
                            style_negative=style_profile.get(
                                "base_negative_prompt"
                            ),
                            character_negative=scene_character.get(
                                "negative_prompt"
                            ),
                            supports_negative=model_ctx.supports_negative_prompt,
                            negative_embeddings=(
                                guide.negative_embeddings if guide else None
                            ),
                            conflict_terms=[
                                scene_character.get("base_prompt", ""),
                                prompt_dict["prompt_text"],
                            ],
                            family=family,
                        )
                    )
                    prompt_dict["prompt_hash"] = compute_prompt_hash(
                        prompt_dict["prompt_text"]
                    )
                    prompt_dict["scene_id"] = scene_id
                    prompt_dict["id"] = (
                        f"{series_id}_{model_id}_prompt_"
                        f"{len(prompts_for_model):03d}"
                    )
                    prompt_dict["content_level"] = model_ctx.content_level
                    prompt_dict["model_id"] = model_id
                    prompts_for_model.append(prompt_dict)
                except Exception as exc:
                    logger.warning(
                        "Failed to build prompt for scene %s model %s: %s",
                        scene_id, model_id, exc,
                    )

            # Within-model dedup (different models compose to different
            # text by design, so cross-model dedup wouldn't fire even if
            # the deduplicator ran across all model prompts).
            prompts_for_model = self.deduplicator.deduplicate(
                prompts_for_model, scenes, model_ctx,
            )
            logger.info(
                "Model %s: %d prompts after dedup", model_id,
                len(prompts_for_model),
            )

            # Persist prompts for this (model, llm) pair. The llm_id
            # matches the facet that fed the composer — same resolution
            # path so the prompt and its facet always co-locate.
            self._save_prompts_for_model(
                series_id=series_id,
                ctx=model_ctx,
                prompts=prompts_for_model,
                llm_id=facet_llm_id,
            )
            prompts_created_total += len(prompts_for_model)
            models_completed.append(model_id)

            # Supervised pause once per model.
            if model_ctx.execution_mode == "supervised" and not self.dry_run:
                approved = self.supervisor.approve_plan(
                    mode=model_ctx.mode,
                    content_level=model_ctx.content_level,
                    character_id=model_ctx.character_id,
                    model_id=model_id,
                    series_plan=series_plan,
                    scenes=scenes,
                    prompts=prompts_for_model,
                )
                if not approved:
                    self._update_series_status(series_id, "aborted")
                    logger.info(
                        "Plan rejected by supervisor for model %s. "
                        "Aborting Phase A.", model_id,
                    )
                    # Free every LLM the cycle has loaded — multiple
                    # tags may be live when per-role routing is active.
                    self.llm_client.unload_all()
                    return {
                        "series_id": series_id,
                        "status": "aborted",
                        "scenes_created": (
                            len(scenes) if is_new_series else 0
                        ),
                        "facets_created": facets_created_total,
                        "prompts_created": prompts_created_total,
                        "models_completed": models_completed,
                    }

        # ── 5. Unload every LLM the cycle loaded ────────────────────
        # With per-role routing a single Phase A may have loaded
        # series_planner, scene_generator, and per-family
        # scene_facet_generator tags simultaneously. unload_all walks
        # client.loaded_models and frees each.
        self.llm_client.unload_all()
        logger.info("LLM unloaded. Phase A complete.")

        return {
            "series_id": series_id,
            "status": "complete",
            "scenes_created": len(scenes) if is_new_series else 0,
            "facets_created": facets_created_total,
            "prompts_created": prompts_created_total,
            "models_completed": models_completed,
        }

    # ═══════════════════════════════════════════════════════════════
    # PHASE B — render + score + postprocess
    # ═══════════════════════════════════════════════════════════════

    def run_phase_b(
        self,
        *,
        series_id: str,
        model_id: str,
        scene_ids: list[str] | None = None,
        template_override: str | None = None,
        cli_llm_override: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Phase B — render every (scene × prompt) for a (series, model).

        Reloads everything it needs from the DB (series row, scenes,
        prompts), rebuilds the GenerationContext for the **target**
        model_id (which may differ from the model used to create the
        series), recomputes per-target-model resolution at render
        time, stages IPAdapter where applicable, runs the render loop
        with retries, scores, postprocesses, and (in supervised mode)
        prompts the operator for image-level review.

        Returns ``(rendered_images, ctx_state)``. ``ctx_state`` carries
        everything :meth:`run_phase_c` needs (``series_plan, scenes,
        prompts, ctx, style_profile, style_profile_id``) so the caller
        doesn't have to re-load.

        Used by ``render_prompts.py`` (standalone Phase B+C invocation
        on an existing series) and indirectly by :meth:`run_cycle` (which
        wraps phase_a → phase_b → phase_c).

        Raises
        ------
        EngineError
            Series not in DB; no prompts for the (series, model) pair;
            target model not in registry.
        """
        logger.info(
            "Phase B: Rendering series=%s model=%s …", series_id, model_id,
        )

        # ── 1. Reload series + scenes + prompts from DB ─────────────
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            series_row = conn.execute(
                "SELECT * FROM series WHERE id = ?", (series_id,),
            ).fetchone()
            if not series_row:
                raise EngineError(f"Series {series_id!r} not found in DB.")
            series_plan = (
                json.loads(series_row["llm_series_plan"])
                if series_row["llm_series_plan"] else {}
            )
            content_level = series_row["content_level"]
            style_profile_id = series_row["style_profile_id"]
            mode_name = series_row["mode"]
            character_id = series_row["character_id"]

            # Reload scenes (model-agnostic).
            scenes_query = "SELECT * FROM scenes WHERE series_id = ?"
            scenes_params: list[Any] = [series_id]
            if scene_ids:
                placeholders = ", ".join(["?"] * len(scene_ids))
                scenes_query += f" AND id IN ({placeholders})"
                scenes_params.extend(scene_ids)
            scene_rows = conn.execute(scenes_query, scenes_params).fetchall()
            scenes = [dict(r) for r in scene_rows]

            # Reload prompts filtered by (series, model, llm_id,
            # optional scenes). The llm_id comes from --llm <id>
            # (cli_llm_override) when set, else falls back to default_llm.
            # Phase 5 will tighten this with strict-ambiguity checking
            # at the CLI boundary so the user always specifies which
            # LLM's prompts to render against a non-empty series.
            phase_b_llm_id = (
                cli_llm_override
                if cli_llm_override is not None
                else self._default_llm_id
            )
            prompts_query = (
                "SELECT * FROM prompts "
                "WHERE series_id = ? AND model_id = ? AND llm_id = ?"
            )
            prompts_params: list[Any] = [
                series_id, model_id, phase_b_llm_id,
            ]
            if scene_ids:
                placeholders = ", ".join(["?"] * len(scene_ids))
                prompts_query += f" AND scene_id IN ({placeholders})"
                prompts_params.extend(scene_ids)
            prompt_rows = conn.execute(prompts_query, prompts_params).fetchall()
            prompts = [dict(r) for r in prompt_rows]
        finally:
            conn.close()

        if not prompts:
            raise EngineError(
                f"No prompts found in DB for series_id={series_id!r}, "
                f"model_id={model_id!r}, llm_id={phase_b_llm_id!r}. "
                f"Run prepare_prompts first."
            )

        # ── 2. Reload style_profile + content_rules ─────────────────
        style_profile = _load_style_profile(self.db_path, style_profile_id)
        content_rules = ContentLevelLoader(self.db_path).load(content_level)

        # ── 3. Rebuild ctx for the TARGET model ─────────────────────
        # The target model may differ from the model used to create
        # the series — that's the whole point of multi-model rendering.
        character: dict[str, Any] | None = None
        if character_id:
            try:
                character = CharacterManager(self.db_path).get_character(
                    character_id
                )
            except CharacterNotFound:
                logger.warning(
                    "character_id %r referenced by series %r not found",
                    character_id, series_id,
                )

        ctx = build_context(
            mode=mode_name,
            content_level=content_level,
            execution_mode=self.execution_mode,
            style_profile=style_profile,
            content_rules=content_rules,
            db_path=self.db_path,
            model_id=model_id,
            commercial_mode=self._commercial_mode,
        )
        if character is not None:
            ctx.character = character
            ctx.character_id = character_id

        # ── 4. Memory preflight + status update ─────────────────────
        self._memory_preflight(min_free_gb=14.0)
        self._update_series_status(series_id, "rendering")

        # ── 5. Resolve mode (dispatch for IPAdapter staging) ────────
        mode = self._mode_registry.get(mode_name)
        if mode is None:
            raise EngineError(
                f"Mode {mode_name!r} not implemented. "
                f"Available: {list(self._mode_registry)}"
            )

        # ── 6. Recompute resolution per-target-model ────────────────
        # Scene rows store resolution at scene-creation time, computed
        # against whatever model existed then. When re-targeting (e.g.
        # an SDXL series rendered through Flux2), we want the TARGET
        # model's preferred resolution, not the stored one. Stored
        # value becomes informational; render uses the freshly-computed
        # value.
        model_res = model_resolution_overrides(ctx.model_config)
        target_family_id = getattr(ctx.model_config, "family", None)
        for scene in scenes:
            if scene.get("aspect_ratio"):
                res = get_resolution(
                    scene["aspect_ratio"],
                    ctx.model_config.filename,
                    model_resolutions=model_res,
                    family_id=target_family_id,
                )
                scene["resolution_w"] = res[0]
                scene["resolution_h"] = res[1]

        # ── 7. Build StyleProfileForWorkflow ────────────────────────
        style_for_wf = StyleProfileForWorkflow(style_profile, ctx.model_config)

        # ── 8. IPAdapter staging ────────────────────────────────────
        # Same logic as before — character mode (model + reference) or
        # variation mode inheriting from a character series. Disabled
        # entirely under --template (external workflow is authoritative).
        use_ipadapter = False
        ipadapter_basename: str | None = None

        if template_override:
            if isinstance(mode, CharacterMode) and ctx.character and (
                ctx.character.get("reference_image_path")
            ):
                logger.info(
                    "external_template: IPAdapter disabled (template "
                    "owns the workflow). Reference image for character "
                    "%r ignored.", ctx.character_id,
                )
        elif isinstance(mode, CharacterMode) and mode.use_ipadapter(ctx):
            ref_path = mode.get_reference_image_path(ctx)
            if ref_path:
                full_ref = _PROJECT_ROOT / ref_path
                if full_ref.exists():
                    dst = self.comfy_input_dir / full_ref.name
                    shutil.copy2(full_ref, dst)
                    ipadapter_basename = full_ref.name
                    use_ipadapter = True
                    logger.info("IPAdapter: staged %s → %s", ref_path, dst)
                else:
                    logger.warning(
                        "IPAdapter: reference image not found: %s", full_ref,
                    )

        # Variation mode inherits character ref from the source series'
        # llm_series_plan JSON (loaded above).
        if (
            not template_override
            and isinstance(mode, VariationMode)
            and ctx.supports_ipadapter
        ):
            source_mode = series_plan.get("source_mode", "")
            var_char_id = series_plan.get("character_id")
            if source_mode == "character" and var_char_id:
                try:
                    cm = CharacterManager(self.db_path)
                    char = cm.get_character(var_char_id)
                    ref_path = char.get("reference_image_path")
                    if ref_path:
                        full_ref = _PROJECT_ROOT / ref_path
                        if full_ref.exists():
                            dst = self.comfy_input_dir / full_ref.name
                            shutil.copy2(full_ref, dst)
                            ipadapter_basename = full_ref.name
                            use_ipadapter = True
                            logger.info(
                                "IPAdapter (variation→character): staged "
                                "%s for %s", ref_path, var_char_id,
                            )
                except Exception as exc:
                    logger.warning(
                        "IPAdapter (variation): failed for %s: %s",
                        var_char_id, exc,
                    )

        # ── 9. Render loop ──────────────────────────────────────────
        wf_builder = WorkflowBuilder(self.workflow_dir)
        comfy_client = ComfyUIClient(
            base_url=self.comfy_base_url,
            output_dir=self.comfy_output_dir,
        )
        rendered_images: list[dict[str, Any]] = []
        # Per-LLM output dir so Cydonia/Magnum/Venice renders of the
        # same series don't overwrite each other on disk. Plan §3.5.
        output_series_dir = (
            self.output_dir
            / content_level
            / series_id
            / phase_b_llm_id
            / "images"
        )
        output_series_dir.mkdir(parents=True, exist_ok=True)

        for prompt in prompts:
            scene = self._find_scene(scenes, prompt["scene_id"])
            if not scene:
                logger.warning("No scene for prompt %s", prompt["id"])
                continue

            resolution = (scene["resolution_w"], scene["resolution_h"])

            try:
                if template_override:
                    resolved_template = _resolve_template_path(
                        template_override, self.workflow_dir, _PROJECT_ROOT,
                    )
                    workflow = wf_builder.build_external(
                        external_template=resolved_template,
                        prompt_text=prompt["prompt_text"],
                        negative_prompt=prompt["negative_prompt"],
                        resolution=resolution,
                    )
                else:
                    workflow = wf_builder.build(
                        prompt_text=prompt["prompt_text"],
                        negative_prompt=prompt["negative_prompt"],
                        style_profile=style_for_wf,
                        resolution=resolution,
                        ipadapter_image=(
                            ipadapter_basename if use_ipadapter else None
                        ),
                    )
            except WorkflowTemplateError as exc:
                logger.error("Workflow build failed: %s", exc)
                continue

            comfy_images = None
            for attempt in range(self.max_retry):
                try:
                    cf_prompt_id = comfy_client.queue_prompt(workflow)
                    comfy_images = comfy_client.wait_for_completion(
                        cf_prompt_id, timeout=self.render_timeout,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Render attempt %d/%d failed for %s: %s",
                        attempt + 1, self.max_retry, prompt["id"], exc,
                    )
                    if attempt < self.max_retry - 1:
                        time.sleep(5)

            if not comfy_images:
                logger.warning("All render attempts failed for %s", prompt["id"])
                continue

            for ci in comfy_images:
                src_path = (
                    Path(ci.file_path) if hasattr(ci, "file_path")
                    else Path(str(ci))
                )
                if not src_path.exists():
                    logger.warning("Rendered file not found: %s", src_path)
                    continue
                dst_name = f"{prompt['id']}_{src_path.name}"
                dst_path = output_series_dir / dst_name
                shutil.copy2(src_path, dst_path)
                seed_val = (
                    getattr(ci, "seed", 0) if hasattr(ci, "seed") else 0
                )
                # Phase 4b — embed AUTOMATIC1111 + nsfw_pipeline PNG
                # metadata chunks so the file carries its own
                # reproduction parameters even if disconnected from
                # the SQLite DB.
                # GenerationContext exposes the resolved model row as
                # ``ctx.model_config`` (a ModelRegistryEntry), and the
                # family is reachable as ``ctx.family``. Read the
                # render-tuning defaults straight off model_config so
                # PNG metadata captures the actual sampler/cfg/steps.
                self._embed_png_metadata(
                    dst_path,
                    prompt=prompt,
                    model_id=model_id,
                    series_id=series_id,
                    seed=seed_val,
                    resolution=resolution,
                    scene=scene,
                    content_level=content_level,
                    family_id=family.id,
                    sampler=ctx.model_config.default_sampler,
                    scheduler=ctx.model_config.default_scheduler,
                    steps=ctx.model_config.default_steps,
                    cfg=ctx.model_config.default_cfg,
                    clip_skip=ctx.model_config.default_clip_skip,
                )
                rendered_images.append({
                    "id": uuid.uuid4().hex,
                    "prompt_id": prompt["id"],
                    "series_id": series_id,
                    "model_id": model_id,
                    "file_path": str(dst_path),
                    "width": resolution[0],
                    "height": resolution[1],
                    "seed": seed_val,
                    "content_level": content_level,
                    "prompt_text": prompt["prompt_text"],
                    "aspect_ratio": scene.get("aspect_ratio"),
                })

        logger.info("Rendered: %d images", len(rendered_images))

        ctx_state: dict[str, Any] = {
            "ctx": ctx,
            "series_plan": series_plan,
            "scenes": scenes,
            "prompts": prompts,
            "style_profile": style_profile,
            "style_profile_id": style_profile_id,
            # Phase 4 — surface llm_id so run_phase_c can pass it to
            # the Exporter (which reconstructs its own export_dir and
            # would otherwise overwrite Cydonia/Magnum exports).
            "llm_id": phase_b_llm_id,
        }

        if not rendered_images:
            self._update_series_status(series_id, "failed")
            return [], ctx_state

        # ── 10. Score ───────────────────────────────────────────────
        self._update_series_status(series_id, "filtering")
        scorer = ImageScorer(
            use_hps_v2=self.use_hps_v2,
            use_image_reward=self.use_image_reward,
        )
        for img in rendered_images:
            try:
                result = scorer.score(
                    img["file_path"], prompt=img.get("prompt_text"),
                )
                img["quality_score"] = result.get("composite")
                img["aesthetic_score"] = result.get("aesthetic")
                img["blur_score"] = result.get("blur")
                img["face_confidence"] = result.get("face_confidence")
                img["hps_v2_score"] = result.get("hps_v2")
                img["image_reward_score"] = result.get("image_reward")
                img["quality_flags"] = result.get("flags", [])
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", img["file_path"], exc)
                img["quality_score"] = 0.0
                img["hps_v2_score"] = None
                img["image_reward_score"] = None
                img["quality_flags"] = ["scorer_error"]

        # ── 11. Postprocess ─────────────────────────────────────────
        if self.upscale_enabled and not self.dry_run:
            logger.info(
                "Postprocess: upscaling %d images …", len(rendered_images),
            )
            try:
                upscaler = Upscaler(
                    self.workflow_dir, comfy_client, self.comfy_input_dir,
                    family_id=target_family_id or "sdxl",
                    upscale_model=self.upscale_model,
                )
                upscaler.upscale_batch(rendered_images)
            except UpscaleError as exc:
                logger.warning("Upscale pipeline error: %s", exc)

        if self.face_refine_enabled and not self.dry_run:
            logger.info(
                "Postprocess: face refinement on %d images …",
                len(rendered_images),
            )
            try:
                refiner = FaceRefiner(
                    self.workflow_dir, comfy_client, self.comfy_input_dir,
                    denoise_strength=self.face_denoise,
                )
                refiner.refine_batch(rendered_images)
            except FaceRefineError as exc:
                logger.warning("Face refine pipeline error: %s", exc)

        # ── 12. Supervised pause 2 (image review) ───────────────────
        if ctx.execution_mode == "supervised":
            # Per-LLM preview dir so dual-LLM A/B series get separate
            # review folders (matches output_series_dir shape above).
            preview_dir = (
                self.output_dir
                / content_level
                / series_id
                / phase_b_llm_id
                / "preview"
            )
            preview_dir.mkdir(parents=True, exist_ok=True)
            try:
                rendered_images = self.supervisor.review_images(
                    rendered_images,
                    series_id=series_id,
                    preview_dir=preview_dir,
                )
            except SupervisorAbort:
                self._update_series_status(series_id, "aborted")
                logger.info("Images rejected by supervisor.")
                ctx_state["aborted"] = True
                return [], ctx_state

        return rendered_images, ctx_state

    # ═══════════════════════════════════════════════════════════════
    # PHASE C — set build + watermark + export + persist + memory + log
    # ═══════════════════════════════════════════════════════════════

    def run_phase_c(
        self,
        *,
        series_id: str,
        model_id: str,
        content_level: str,
        rendered_images: list[dict[str, Any]],
        series_plan: dict[str, Any],
        scenes: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        ctx: GenerationContext,
        style_profile: dict[str, Any],
        style_profile_id: str,
        elapsed_seconds: float | None = None,
        llm_id: str | None = None,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Phase C — package + persist + log.

        Independent of the LLM (briefly reloads it for metadata, then
        unloads again). Designed to be called once per (series_id,
        model_id) — the new ``render_prompts.py`` CLI will call it
        per-model after each ``run_phase_b`` completes.

        Parameters
        ----------
        series_id, model_id, content_level
            Identity for the row written to ``run_log`` and metadata.
        rendered_images : list[dict]
            Output of :meth:`run_phase_b` — each img dict already has
            quality scores, file_path, dimensions, prompt_text.
        series_plan, scenes, prompts
            Used by ``MemoryManager.record_series`` to anti-repeat
            future series. Reload from DB when called from
            ``render_prompts.py``; in ``run_cycle`` they're carried
            through from Phase A in memory.
        ctx
            The baseline GenerationContext for this run. Used for
            character_id (LRU update), supports_negative_prompt, and
            family.llm_temperature (metadata generation tuning).
        style_profile, style_profile_id
            Aesthetic + identity for the export manifest.
        elapsed_seconds : float | None
            Total cycle time. Recorded in ``run_log``. None → 0.0.

        Returns
        -------
        dict
            Summary: series_id, status, image counts, export_dir,
            elapsed_seconds, model_id, theme.
        """
        logger.info("Phase C: Packaging …")
        self._update_series_status(series_id, "packaging")

        # Set builder
        set_builder = SetBuilder(
            min_images=self.config.get("set_builder", {}).get("min_images", 10),
            max_images=self.config.get("set_builder", {}).get("max_images", 25),
            quality_cutoff=self.quality_cutoff,
        )
        try:
            selected = set_builder.build(rendered_images, content_level)
        except SetTooSmall as exc:
            logger.warning("Set too small: %s", exc)
            self._update_series_status(series_id, "partial")
            selected = [
                img for img in rendered_images
                if (img.get("quality_score") or 0) >= self.quality_cutoff
            ]
            if not selected:
                selected = sorted(
                    rendered_images,
                    key=lambda x: x.get("quality_score", 0),
                    reverse=True,
                )

        # Brief LLM reload for metadata. The metadata role gets its
        # own router resolution (override→routing→default) so a user
        # can route metadata to e.g. venice_24b for its lower refusal
        # floor while keeping cydonia for the rest of the pipeline.
        metadata = None
        meta_llm_entry = self._llm_router.resolve_role(
            "metadata_generator", override=cli_llm_override,
        )
        meta_model = meta_llm_entry.ollama_id
        try:
            meta_gen = MetadataGenerator(self.llm_client)
            metadata = meta_gen.generate(
                theme=series_plan.get("theme", ""),
                mood=series_plan.get("mood", ""),
                environment=series_plan.get("environment", ""),
                character_name=ctx.character_id or "",
                content_level=content_level,
                image_count=len(selected),
                style_keywords=style_profile.get("base_style_keywords", ""),
                temperature=ctx.family.llm_temperature,
                model=meta_model,
            )
            self.llm_client.unload_model(meta_model)
        except Exception as exc:
            logger.warning("Metadata generation failed: %s", exc)
            self.llm_client.unload_model(meta_model)

        # Watermark T1/T2 exports (T3/T4 ship to paid platforms unwatermarked)
        self.watermarker.apply_batch(selected, content_level=content_level)

        # Export. llm_id flows from run_phase_b's ctx_state via run_cycle
        # so exports land under output/<level>/<series>/<llm_id>/.
        # Without it, two LLMs A/B-rendering the same series would
        # silently overwrite each other's exports.
        exporter = Exporter(self.output_dir)
        export_dir = exporter.export(
            series_id=series_id,
            content_level=content_level,
            images=selected,
            metadata=metadata,
            series_plan=series_plan,
            model_id=model_id,
            style_profile_id=style_profile_id,
            llm_id=llm_id or self._default_llm_id,
        )

        # Persist images to DB
        self._persist_images(series_id, selected, ctx)

        # Record to memory (anti-repetition)
        self.memory.record_series(
            series_plan, scenes, prompts,
            content_level=content_level,
            character_id=ctx.character_id,
        )

        # Update character usage
        if ctx.character_id:
            try:
                cm = CharacterManager(self.db_path)
                cm.update_usage(ctx.character_id, len(selected))
            except Exception as exc:
                logger.warning("Failed to update character usage: %s", exc)

        self._update_series_status(series_id, "complete")

        elapsed = elapsed_seconds if elapsed_seconds is not None else 0.0
        logger.info(
            "Pipeline complete: %s/%s — %d images exported to %s (%.1fs)",
            series_id, model_id, len(selected), export_dir, elapsed,
        )

        # One run_log row per (series, model). prepare_prompts skips this
        # entirely; render_prompts + run_cycle write here.
        self._log_run(
            mode=ctx.mode,
            content_level=content_level,
            series_id=series_id,
            execution_mode=ctx.execution_mode,
            status="success",
            images_generated=len(rendered_images),
            images_selected=len(selected),
            duration_seconds=elapsed,
        )

        return {
            "series_id": series_id,
            "status": "complete",
            "images_rendered": len(rendered_images),
            "images_selected": len(selected),
            "export_dir": str(export_dir),
            "elapsed_seconds": elapsed,
            "model_id": model_id,
            "theme": series_plan.get("theme"),
        }

    # ═══════════════════════════════════════════════════════════════
    # PREFLIGHT CHECKS
    # ═══════════════════════════════════════════════════════════════

    def _preflight(self, ctx: GenerationContext) -> None:
        """Run all preflight checks before the LLM phase.

        Two branches. The Ollama check always runs (Phase A LLM
        planning is unaffected by which workflow renders the result).
        The system-template path checks the checkpoint file,
        ``{family}/base.json``, and IPAdapter-capability compatibility.
        The external-template path (``--template``) skips all three
        because the external graph ships its own checkpoint, its own
        graph file, and IPAdapter is forced off — and instead validates
        that the external template is loadable and carries the four
        contracted semantic IDs with the right input fields.
        """
        errors: list[str] = []

        # 1. Ollama reachable — ALWAYS required.
        if not self.llm_client.is_available():
            errors.append(
                f"Ollama not reachable at {self.llm_client.base_url}. "
                f"Run `ollama serve`."
            )

        if self.template_override:
            errors.extend(self._preflight_external_template(ctx))
        else:
            errors.extend(self._preflight_system_template(ctx))

        if errors:
            msg = "Preflight check(s) failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise PreflightError(msg)

        logger.info("Preflight: all checks passed")

    def _preflight_system_template(
        self, ctx: GenerationContext,
    ) -> list[str]:
        """Checks that apply when the pipeline renders through its own
        built-in ``{family}/base.json`` or ``{family}/ipadapter.json``."""
        errors: list[str] = []

        # 2. Model checkpoint file exists.
        model_dir = (
            self.comfy_output_dir.parent
            / "models"
            / _model_subfolder(ctx.model_config.family)
        )
        ckpt_path = model_dir / ctx.model_config.filename
        if not ckpt_path.exists():
            errors.append(
                f"Checkpoint file not found: {ckpt_path}\n"
                f"Download {ctx.model_config.filename} to {model_dir}/"
            )

        # 3. Workflow JSON exists.
        family = ctx.model_config.family
        base_workflow = self.workflow_dir / family / "base.json"
        if not base_workflow.exists():
            errors.append(
                f"Workflow template not found: {base_workflow}\n"
                f"See docs/COMFYUI_WORKFLOWS.md → {family}/base.json"
            )

        # 4. IPAdapter check for character mode.
        if ctx.mode == "character" and ctx.character:
            ref = ctx.character.get("reference_image_path")
            if ref and not ctx.supports_ipadapter:
                errors.append(
                    f"Character {ctx.character_id} has a reference image but "
                    f"model {ctx.model_id} does not support IPAdapter. "
                    f"Pick a different model or remove the reference image."
                )
            # 4b. When IPAdapter WOULD be used, require the ipadapter
            # template file. Pre-existing gap: without this, the
            # character-with-ref-image run burned 20+ min of Phase A
            # before failing at the first render when the file was
            # missing. Gated on supports_ipadapter to avoid noise.
            if ref and ctx.supports_ipadapter:
                ipadapter_workflow = (
                    self.workflow_dir / family / "ipadapter.json"
                )
                if not ipadapter_workflow.exists():
                    errors.append(
                        f"IPAdapter workflow template not found: "
                        f"{ipadapter_workflow}\n"
                        f"Character {ctx.character_id} has a reference "
                        f"image and model {ctx.model_id} supports "
                        f"IPAdapter, so the pipeline will try to render "
                        f"through {family}/ipadapter.json. Create that "
                        f"template or remove the reference image."
                    )

        return errors

    def _preflight_external_template(
        self, ctx: GenerationContext,
    ) -> list[str]:
        """Validate the ``--template`` path. Skips system-path checks
        (checkpoint file, ``{family}/base.json``, IPAdapter capability)
        because the external template ships its own graph and IPAdapter
        is forced off for this run."""
        errors: list[str] = []
        try:
            resolved = _resolve_template_path(
                self.template_override, self.workflow_dir, _PROJECT_ROOT,
            )
            wf_builder = WorkflowBuilder(self.workflow_dir)
            workflow = wf_builder._load(resolved)
            template_name = Path(resolved).name
            WorkflowBuilder._assert_required_nodes(
                workflow, "external", template_name,
                _REQUIRED_NODES_EXTERNAL,
            )
            _assert_external_template_inputs(workflow, template_name)
        except WorkflowTemplateError as exc:
            errors.append(f"--template validation failed: {exc}")
        except Exception as exc:
            errors.append(f"--template validation failed: {exc!r}")
        return errors

    def _memory_preflight(self, min_free_gb: float = 14.0) -> None:
        """Check that enough memory is free for the render phase."""
        available = psutil.virtual_memory().available / (1024**3)
        if available < min_free_gb:
            raise InsufficientMemoryError(
                f"{available:.1f} GB free, need {min_free_gb} GB. "
                f"LLM may not have unloaded — check Ollama."
            )
        logger.info("Memory preflight: %.1f GB free (need %.1f)", available, min_free_gb)

    # ═══════════════════════════════════════════════════════════════
    # PNG metadata embedding (Phase 4b)
    # ═══════════════════════════════════════════════════════════════

    def _embed_png_metadata(
        self,
        path: Path,
        *,
        prompt: dict[str, Any],
        model_id: str,
        series_id: str,
        seed: int,
        resolution: tuple[int, int],
        scene: dict[str, Any],
        content_level: str,
        family_id: str | None,
        sampler: str | None,
        scheduler: str | None,
        steps: int | None,
        cfg: float | None,
        clip_skip: int | None,
    ) -> None:
        """Attach AUTOMATIC1111 ``parameters`` + pipeline ``nsfw_pipeline``
        PNG tEXt chunks to ``path``. Best-effort: any failure logs at
        WARNING and proceeds; embedding never blocks render output."""
        try:
            vocab_version = 1
            try:
                vocab_version = VocabularyLoader().version
            except Exception:  # pragma: no cover — defensive
                pass

            a1111 = build_a1111_parameters(
                prompt_text=prompt.get("prompt_text", ""),
                negative_prompt=prompt.get("negative_prompt"),
                steps=steps,
                sampler=sampler,
                scheduler=scheduler,
                cfg_scale=cfg,
                seed=seed,
                model=model_id,
                size=resolution,
                clip_skip=clip_skip,
            )
            # Distil structured-facet enum tags from the scene dict —
            # the canonicalizer fields the LLM populated, useful for
            # round-tripping a generation back into our pipeline.
            structured_keys = (
                "realism_camera", "realism_lens", "realism_film_stock",
                "art_style_reference", "lighting_directive",
                "mood_aesthetic", "nsfw_anatomy", "nsfw_posture",
            )
            structured_facet = {
                k: scene.get(k) for k in structured_keys
                if scene.get(k) is not None
            }
            pipeline_json = build_pipeline_metadata(
                vocab_version=vocab_version,
                family=family_id or "",
                model_id=model_id,
                scene_id=prompt.get("scene_id"),
                series_id=series_id,
                prompt_hash=prompt.get("prompt_hash"),
                seed=seed,
                sampler=sampler,
                scheduler=scheduler,
                steps=steps,
                cfg=cfg,
                content_level=content_level,
                structured_facet=structured_facet or None,
                # Stamp the generating LLM on the PNG itself so a
                # forensic reader can identify it without DB access.
                llm_id=prompt.get("llm_id"),
            )
            write_png_metadata(
                path,
                a1111_parameters=a1111,
                pipeline_metadata=pipeline_json,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "PNG metadata embedding failed for %s: %s",
                path, exc,
            )

    # ═══════════════════════════════════════════════════════════════
    # DB PERSISTENCE
    # ═══════════════════════════════════════════════════════════════

    def _save_dry_run(
        self,
        *,
        series_id: str,
        ctx: GenerationContext,
        series_plan: dict[str, Any],
        scenes: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
        style_profile_id: str,
    ) -> None:
        """Save series + scenes + prompts with status='dry_run'.

        Composition wrapper kept for back-compat with callers (and
        ``test_engine_save_dry_run.py``). Calls
        :meth:`_save_series_and_scenes` then
        :meth:`_save_prompts_for_model`.
        """
        self._save_series_and_scenes(
            series_id=series_id,
            ctx=ctx,
            series_plan=series_plan,
            scenes=scenes,
            style_profile_id=style_profile_id,
            target_count=len(prompts),
        )
        self._save_prompts_for_model(
            series_id=series_id,
            ctx=ctx,
            prompts=prompts,
            llm_id=self._default_llm_id,
        )
        logger.info(
            "Saved to DB: series=%s, %d scenes, %d prompts (status=dry_run)",
            series_id, len(scenes), len(prompts),
        )

    def _save_series_and_scenes(
        self,
        *,
        series_id: str,
        ctx: GenerationContext,
        series_plan: dict[str, Any],
        scenes: list[dict[str, Any]],
        style_profile_id: str,
        target_count: int | None = None,
    ) -> None:
        """Persist a fresh series row + its scenes (model-agnostic).

        Used by :meth:`run_phase_a` for new-series creation. Prompts
        are persisted separately via :meth:`_save_prompts_for_model`
        (one call per model in the multi-model fan-out).
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO series (
                    id, mode, content_level, character_id, style_profile_id,
                    theme, mood, environment, variation_axes, target_count,
                    actual_count, status, llm_series_plan
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    series_id,
                    ctx.mode,
                    ctx.content_level,
                    ctx.character_id,
                    style_profile_id,
                    series_plan.get("theme", ""),
                    series_plan.get("mood"),
                    series_plan.get("environment"),
                    json.dumps(series_plan.get("variation_axes")),
                    # target_count defaults to scene count when not given.
                    target_count if target_count is not None else len(scenes),
                    0,
                    "dry_run",
                    json.dumps(series_plan),
                ),
            )

            for scene in scenes:
                conn.execute(
                    """
                    INSERT INTO scenes (
                        id, series_id, variation_axis,
                        pose, camera, camera_angle, lighting,
                        environment_detail, mood_note, expression,
                        aspect_ratio, resolution_w, resolution_h,
                        content_level, status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        scene["id"],
                        series_id,
                        scene.get("variation_axis", "llm"),
                        scene.get("pose"),
                        scene.get("camera"),
                        scene.get("camera_angle"),
                        scene.get("lighting"),
                        scene.get("environment_detail"),
                        scene.get("mood_note"),
                        scene.get("expression"),
                        scene.get("aspect_ratio"),
                        scene.get("resolution_w"),
                        scene.get("resolution_h"),
                        ctx.content_level,
                        "active",
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _save_prompts_for_model(
        self,
        *,
        series_id: str,
        ctx: GenerationContext,
        prompts: list[dict[str, Any]],
        llm_id: str,
    ) -> None:
        """Insert one (model, llm) pair's prompts for a series.

        Called once per (model, llm) tuple in :meth:`run_phase_a`'s
        fan-out. The UNIQUE(scene_id, model_id, llm_id) constraint
        protects against accidental double-insert; re-roll requires
        explicit :meth:`_delete_prompts_for_model` first.

        Each prompt dict's ``model_id`` key wins over ``ctx.model_id``
        when present (multi-model loop sets it explicitly). ``llm_id``
        is method-level (not per-prompt) because every prompt in this
        call comes from the same LLM resolution.
        """
        # Phase 4a — capture the active vocabulary version at insert time
        # so readers can answer "which vocab produced this prompt?" after
        # the YAML is bumped.
        try:
            vocab_version = VocabularyLoader().version
        except Exception:  # pragma: no cover — defensive
            vocab_version = 1

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            for prompt in prompts:
                conn.execute(
                    """
                    INSERT INTO prompts (
                        id, series_id, scene_id, model_id, llm_id,
                        prompt_text, negative_prompt, prompt_hash,
                        content_level, vocab_version, status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        prompt["id"],
                        series_id,
                        prompt.get("scene_id"),
                        prompt.get("model_id") or ctx.model_id,
                        llm_id,
                        prompt["prompt_text"],
                        prompt["negative_prompt"],
                        prompt["prompt_hash"],
                        ctx.content_level,
                        vocab_version,
                        "pending",
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _delete_prompts_for_model(
        self, series_id: str, model_id: str, llm_id: str,
    ) -> int:
        """DELETE all prompts for one (series, model, llm) triple.

        Used by :meth:`run_phase_a` when ``regen_prompts`` includes
        the model_id — clears the way for fresh INSERTs without
        tripping the UNIQUE(scene_id, model_id, llm_id) constraint.
        Other LLMs' prompts on the same (series, model) are untouched
        so a Cydonia regen doesn't blow away Magnum's parallel data.

        Returns the number of rows deleted.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.execute(
                "DELETE FROM prompts "
                "WHERE series_id = ? AND model_id = ? AND llm_id = ?",
                (series_id, model_id, llm_id),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def _load_series_for_retarget(
        self, series_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Load series_plan + scenes from DB for a re-target run_phase_a.

        Returns ``(series_plan_dict, scenes_list)``. The series_plan
        is parsed from ``series.llm_series_plan`` JSON — carries niche
        ``keyword_cluster`` and variation ``source_mode/character_id``
        that downstream PromptBuilder + IPAdapter staging may need.

        Raises
        ------
        EngineError
            Series not found in DB or has no llm_series_plan stored.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            series_row = conn.execute(
                "SELECT * FROM series WHERE id = ?", (series_id,),
            ).fetchone()
            if not series_row:
                raise EngineError(
                    f"Series {series_id!r} not found in DB."
                )
            scene_rows = conn.execute(
                "SELECT * FROM scenes WHERE series_id = ?", (series_id,),
            ).fetchall()
        finally:
            conn.close()

        series_plan_json = series_row["llm_series_plan"]
        series_plan = json.loads(series_plan_json) if series_plan_json else {}
        scenes = [dict(r) for r in scene_rows]
        return series_plan, scenes

    @staticmethod
    def _resolve_character_for_scene(
        *,
        mode_name: str,
        base_character: dict[str, Any] | None,
        series_plan: dict[str, Any],
        style_profile: dict[str, Any],
        scene: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the character dict the PromptBuilder needs for this scene.

        Character mode → use the loaded character row.
        Theme/style/niche → synthesize from subject_description /
            subject_detail / subject_bias (per-scene override possible
            via ``scene["subject_detail"]``).
        Variation → use base_character (the source series's character)
            if present.

        Returns a dict with at minimum ``base_prompt`` and
        ``negative_prompt`` keys.
        """
        if mode_name == "character" and base_character:
            return base_character

        # Synthetic character for non-character modes.
        subject = (
            series_plan.get("subject_description")
            or series_plan.get("subject_bias")
            or ""
        )
        if mode_name == "style" and series_plan.get("style_keywords"):
            sk = series_plan["style_keywords"]
            subject = f"{sk}, {subject}" if subject else sk
        synthetic = {
            "base_prompt": subject,
            "negative_prompt": style_profile.get("base_negative_prompt", ""),
        }

        # Per-scene override for theme/style/niche.
        if mode_name in ("theme", "style", "niche"):
            sd = scene.get("subject_detail", "")
            if sd:
                return {
                    "base_prompt": sd,
                    "negative_prompt": synthetic["negative_prompt"],
                }
        return synthetic

    def _save_scene_facets(
        self,
        scene_id: str,
        family: str,
        facet: dict[str, Any],
        llm_id: str,
    ) -> None:
        """Persist one (scene, family, llm_id) facet row.

        Thin wrapper over ``scene_facets_repo.insert_facet`` so engine
        callers don't have to import the repo directly. Raises
        ``SceneFacetExists`` on PRIMARY KEY conflict — caller must
        ``delete_facet`` first to overwrite (e.g.
        ``prepare_prompts --regen-facets --llm <id>``).
        """
        from src.memory.scene_facets_repo import insert_facet
        insert_facet(self.db_path, scene_id, family, facet, llm_id)

    def _persist_images(
        self,
        series_id: str,
        images: list[dict[str, Any]],
        ctx: GenerationContext,
    ) -> None:
        """Persist rendered image rows to the DB."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            for img in images:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO images (
                        id, prompt_id, series_id, model_id,
                        file_path, width, height, seed,
                        content_level, quality_score, aesthetic_score,
                        blur_score, face_confidence, hps_v2_score,
                        image_reward_score, quality_flags, selected
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        img["id"],
                        img.get("prompt_id"),
                        series_id,
                        img.get("model_id"),
                        img["file_path"],
                        img.get("width", 0),
                        img.get("height", 0),
                        img.get("seed", 0),
                        ctx.content_level,
                        img.get("quality_score"),
                        img.get("aesthetic_score"),
                        img.get("blur_score"),
                        img.get("face_confidence"),
                        img.get("hps_v2_score"),
                        img.get("image_reward_score"),
                        json.dumps(img.get("quality_flags", [])),
                        1,  # selected
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _update_series_status(self, series_id: str, status: str) -> None:
        """Update the series status in the DB."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            completed_at = "CURRENT_TIMESTAMP" if status == "complete" else "NULL"
            conn.execute(
                f"""
                UPDATE series SET status = ?, actual_count = (
                    SELECT COUNT(*) FROM images WHERE series_id = ? AND selected = 1
                ), completed_at = {completed_at}
                WHERE id = ?
                """,
                (status, series_id, series_id),
            )
            conn.commit()
        finally:
            conn.close()
        logger.debug("Series %s → status=%s", series_id, status)

    # ═══════════════════════════════════════════════════════════════
    # RUN LOG
    # ═══════════════════════════════════════════════════════════════

    def _log_run(
        self,
        *,
        mode: str,
        content_level: str,
        series_id: str | None = None,
        execution_mode: str = "manual",
        status: str = "success",
        images_generated: int = 0,
        images_selected: int = 0,
        duration_seconds: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Write a row to the run_log table."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT INTO run_log (
                    mode, content_level, series_id, execution_mode,
                    status, images_generated, images_selected,
                    duration_seconds, error_message
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    mode, content_level, series_id, execution_mode,
                    status, images_generated, images_selected,
                    duration_seconds, error_message,
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("Failed to write run_log: %s", exc)
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════
    # DRY RUN SUMMARY
    # ═══════════════════════════════════════════════════════════════

    def _dry_run_summary(
        self,
        series_id: str,
        ctx: GenerationContext,
        series_plan: dict[str, Any],
        scenes: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Print and return a summary for dry-run mode."""
        print("\n" + "=" * 60)
        print("  DRY RUN COMPLETE — Phase A only, no rendering")
        print("=" * 60)
        print(f"  Series ID:      {series_id}")
        print(f"  Mode:           {ctx.mode}")
        print(f"  Content level:  {ctx.content_level}")
        print(f"  Character:      {ctx.character_id or '(none)'}")
        print(f"  Model:          {ctx.model_id} ({ctx.model_config.family})")
        print(f"  Execution:      {ctx.execution_mode}")
        print(f"\n  Series Plan:")
        print(f"    Theme:        {series_plan.get('theme')}")
        print(f"    Mood:         {series_plan.get('mood')}")
        print(f"    Environment:  {series_plan.get('environment')}")
        print(f"    Axes:         {series_plan.get('variation_axes')}")
        # Mode-specific plan details
        if series_plan.get("category_name"):
            print(f"    Category:     {series_plan['category_name']}")
        if series_plan.get("cluster_name"):
            print(f"    Cluster:      {series_plan['cluster_name']}")
        if series_plan.get("style_keywords"):
            print(f"    Style kw:     {series_plan['style_keywords']}")
        if series_plan.get("visual_elements"):
            print(f"    Vis. elems:   {series_plan['visual_elements']}")
        if series_plan.get("keyword_cluster"):
            print(f"    Niche kw:     {series_plan['keyword_cluster'][:5]}")
        if series_plan.get("source"):
            print(f"    Base source:  {series_plan['source']}")
        print(f"\n  Scenes: {len(scenes)}")

        # Ratio distribution
        ratios: dict[str, int] = {}
        for s in scenes:
            r = s.get("aspect_ratio", "unknown")
            ratios[r] = ratios.get(r, 0) + 1
        print(f"  Ratio distribution: {dict(sorted(ratios.items()))}")

        print(f"\n  Prompts: {len(prompts)}")
        for i, p in enumerate(prompts[:5]):
            text = p["prompt_text"]
            display = (text[:90] + "…") if len(text) > 90 else text
            print(f"    [{i+1}] {display}")
        if len(prompts) > 5:
            print(f"    … and {len(prompts) - 5} more")

        print(f"\n  DB: series + scenes + prompts saved with status='dry_run'")
        print(f"  To render: switch execution.mode to 'supervised' and re-run")
        print("=" * 60)

        return {
            "series_id": series_id,
            "status": "dry_run",
            "mode": ctx.mode,
            "content_level": ctx.content_level,
            "character_id": ctx.character_id,
            "model_id": ctx.model_id,
            "theme": series_plan.get("theme"),
            "scenes": len(scenes),
            "prompts": len(prompts),
        }

    # ═══════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _find_scene(
        scenes: list[dict[str, Any]], scene_id: str
    ) -> dict[str, Any] | None:
        for s in scenes:
            if s.get("id") == scene_id:
                return s
        return None

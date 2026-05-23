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
import re
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import psutil
import yaml

from src.agents.llm_client import LLMClientPool, OllamaClient
from src.agents.metadata_generator import MetadataGenerator
from src.agents.scene_facet_generator import (
    SceneFacetGenerator,
    SceneFacetGeneratorError,
    _DiversityTracker,
)
from src.core.content_level import ContentLevelLoader
from src.core.generation_context import (
    GenerationContext,
    build_context,
    build_family_context,
)
from src.core.mode_selector import ModeSelector
from src.core.ratio_selector import (
    RatioSelector,
    get_resolution,
    model_resolution_overrides,
)
from src.export.exporter import Exporter
from src.filter.set_builder import SetBuilder, SetTooSmall
from src.memory.memory_manager import MemoryManager
from src.memory.scene_facets_repo import (
    delete_facets_for_family,
    get_facet,
    has_facet,
    insert_facet,
)
from src.modes.base_mode import BaseMode
from src.modes.niche_mode import NicheMode
from src.modes.style_mode import StyleMode
from src.modes.theme_mode import ThemeMode
from src.modes.variation_mode import VariationMode
from src.prompt.builder import (
    PromptBuilder,
    archetype_overridden_by_planner,
    compute_prompt_hash,
)
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


def _extract_seed_from_workflow(workflow: dict[str, Any]) -> int:
    """Read the chosen render seed straight out of a built workflow dict.

    Authoritative because ``WorkflowBuilder.build_external`` patched the
    seed field at build time; ComfyUI's history response does NOT carry
    the seed back, so reading from the response gives 0 for renamed
    semantic-ID ksampler nodes (the bug this helper fixes).

    Every external template carries ``ksampler.inputs.seed`` per
    ``_REQUIRED_NODES_EXTERNAL``. Returns 0 when the field isn't an
    integer — defensive fallback only; the preflight rejects templates
    without ``ksampler.inputs.seed`` so this branch should be unreachable.
    """
    if "ksampler" in workflow:
        inputs = workflow["ksampler"].get("inputs", {})
        seed = inputs.get("seed")
        if isinstance(seed, int):
            return seed
    return 0


# Family-match validator — extracts the family from a resolved template
# path. Expects paths like ``templates/<family>/X.json`` (relative) or
# ``/abs/.../templates/<family>/X.json`` (absolute). Returns None when
# the path falls outside the convention — caller treats None as
# "family unknown, skip the check" rather than raising.
_TEMPLATE_FAMILY_RE = re.compile(
    r"templates[/\\]([A-Za-z0-9_]+)[/\\]", re.IGNORECASE
)


def _extract_family_from_template_path(path: str) -> str | None:
    """Extract family id from a path under ``templates/<family>/``.

    Returns lowercase family id so case-insensitive macOS HFS+ paths
    (`Templates/CHROMA/...`) still match `family: chroma`. Returns
    None when the path has no `templates/<family>/` segment.
    """
    m = _TEMPLATE_FAMILY_RE.search(path)
    return m.group(1).lower() if m else None


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


def _synthetic_subject_anchor(content_level: str) -> str:
    """Round-22 (2026-05-22) — synthesize a generic subject anchor for
    modes that don't emit one in their series_plan.

    Theme mode emits ``subject_description``. Niche mode emits
    ``subject_bias``. Style and variation modes emit NEITHER — pre-fix
    the facet generator's user prompt rendered "(not provided)" for
    those two modes, which gave the LLM zero subject grounding when
    picking nsfw_anatomy / nsfw_act.

    The synthetic anchor is tier-aware: at T3/T4 it names the nudity
    state directly so the facet LLM picks coherent NSFW tags; at
    T1/T2 it stays clothed-implied. Keeps the single-female + adult-
    age invariants the composer enforces downstream anyway.

    Round-2 audit (2026-05-22) flagged style_mode + variation_mode as
    the highest-risk gap from F5 — this helper closes it.
    """
    base = "An adult woman with mature features"
    if content_level == "T1_suggestive":
        return f"{base}, dressed elegantly with intention"
    if content_level == "T2_implied":
        return f"{base}, in suggestive dress or implied undress"
    if content_level == "T3_artnude":
        return f"{base}, artistic nudity per the scene's tier directive"
    if content_level == "T4_explicit":
        return f"{base}, fully nude per the scene's tier directive"
    # Unknown / back-compat — minimal anchor.
    return base


def resolve_subject_anchor(
    series_plan: dict[str, Any] | None,
    content_level: str,
) -> str:
    """Round-22 (2026-05-22) — resolve the subject anchor string from a
    series_plan, applying the three-level fallback chain:

    1. ``series_plan.subject_description`` (theme_mode emits this)
    2. ``series_plan.subject_bias`` (niche_mode emits this — semantic
       equivalent)
    3. :func:`_synthetic_subject_anchor` (style_mode / variation_mode
       emit neither — tier-aware synthetic fallback)

    Public-ish helper (not underscored) so the engine's facet call
    site can use it AND tests can exercise the fallback chain directly
    without spinning up a full engine run. Round-3 audit (2026-05-22)
    identified the absence of a mode-level integration test as a HIGH
    risk; factoring this into a testable function closes that gap.

    Always returns a non-empty string.
    """
    if not series_plan:
        return _synthetic_subject_anchor(content_level)
    desc = series_plan.get("subject_description")
    if desc:
        return str(desc)
    bias = series_plan.get("subject_bias")
    if bias:
        return str(bias)
    return _synthetic_subject_anchor(content_level)


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

    # Style profiles carry only aesthetic intent. Render tuning
    # (sampler/scheduler/steps/cfg/clip_skip/LoRA stack/VAE/CLIP) lives
    # entirely in the user's external ComfyUI template JSON
    # (config/comfyui_workflows/templates/<family>/*.json) after the
    # 2026-05-20 cleanup.
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
        # Phase 3 aesthetic-menu compat lists round-trip through this
        # dict so every mode can read
        # ``ctx.style_profile.get("compatible_palettes", [])`` etc.
        # The planner's menu narrowing reads these at compose time.
        "compatible_palettes": list(p.compatible_palettes),
        "compatible_photographers": list(p.compatible_photographers),
        "compatible_art_movements": list(p.compatible_art_movements),
        "compatible_environments": list(p.compatible_environments),
        "compatible_art_styles": list(p.compatible_art_styles),
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
        Force a specific mode (weighted random across {theme,style,niche,
        variation} by default).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        db_path: Path | None = None,
        *,
        dry_run: bool = False,
        model_override: str | None = None,
        force_mode: str | None = None,
        template_override: str | None = None,
    ) -> None:
        self.config = config or _load_config()
        self.db_path = db_path or _resolve_db_path(self.config)
        self.dry_run = dry_run
        self.model_override = model_override
        self.force_mode = force_mode
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
        # Round-14 (2026-05-21) — LLMClientPool is the backend-aware
        # facade. Agents call ``llm_client.generate_json(model=tag)``;
        # the pool dispatches to OllamaClient or LMStudioClient based
        # on the tag's owning entry's ``backend`` field in
        # config/llm_models.yaml. Drop-in replacement for the legacy
        # OllamaClient — same public interface.
        self.llm_client = LLMClientPool()
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

        # Mode registry — 4 modes. Each mode receives the LLMRouter so
        # its plan() / generate_scenes() can resolve role → ollama_id
        # with the override → routing → default chain.
        self._mode_registry: dict[str, BaseMode] = {
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
        # input_dir is reserved for future LoadImage-style nodes inside
        # external templates. No current code path writes here, but the
        # YAML key is preserved.
        self.workflow_dir = _PROJECT_ROOT / comfy_cfg.get(
            "workflow_dir", "config/comfyui_workflows"
        )
        self.render_timeout = comfy_cfg.get("render_timeout_seconds", 300)
        self.max_retry = comfy_cfg.get("max_retry_per_image", 3)

        set_cfg = self.config.get("set_builder", {})
        self.quality_cutoff = set_cfg.get("quality_cutoff", 0.55)

        # Postprocess subsystem deleted 2026-05-20. Post-processing
        # (upscale, face-detailer) goes inside external templates now.

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

        # Resolve baseline model_id from pipeline.default_model_id.
        # --model CLI override is applied inside build_context.
        default_model_id = pipe_cfg.get("default_model_id")
        if not default_model_id:
            raise EngineError(
                "pipeline.default_model_id is not set in config/pipeline.yaml"
            )

        ctx = build_context(
            mode=mode_name,
            content_level=content_level,
            execution_mode=self.execution_mode,
            style_profile=style_profile,
            content_rules=content_rules,
            db_path=self.db_path,
            model_id=default_model_id,
            model_override=self.model_override,
            commercial_mode=self._commercial_mode,
        )

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
        # _save_prompts_for_target time.
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
            target_kind=ctx_state.get("target_kind", "model"),
            target_id=ctx_state.get("target_id"),
        )

    def _load_prompts_for_summary(
        self,
        series_id: str,
        model_id: str,
        llm_id: str,
        target_kind: str = "model",
    ) -> list[dict[str, Any]]:
        """Cheap helper used by ``run_cycle`` dry-run path to read
        prompts back from the DB after Phase A persists them.

        Filters by ``target_kind`` + ``llm_id`` so multi-target /
        multi-LLM data on the same series doesn't silently merge into
        the dry-run summary. ``run_cycle`` always passes
        ``target_kind='model'`` (the scheduler entry path is model-only
        — see Plan §D10/B2).
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM prompts "
                "WHERE series_id = ? AND target_kind = ? "
                "AND model_id = ? AND llm_id = ?",
                (series_id, target_kind, model_id, llm_id),
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
        models: list[str] | None = None,
        families: list[str] | None = None,
        series_id_existing: str | None = None,
        regen_facets: list[str] | None = None,
        regen_prompts: list[str] | None = None,
        regen_family_prompts: list[str] | None = None,
        style_profile: dict[str, Any] | None = None,
        style_profile_id: str | None = None,
        cli_llm_override: str | None = None,
    ) -> dict[str, Any]:
        """Phase A — LLM planning + per-target prompt persistence.

        Three caller patterns today:
          1. ``run_cycle`` → ``models=[ctx.model_id]`` (single-model
             behavior identical to pre-refactor).
          2. ``prepare_prompts.py --models X,Y`` → multi-model
             fan-out, optional re-target via ``series_id_existing``.
          3. ``prepare_prompts.py --families F,G`` (2026-05) →
             family-level prompt prep with no per-model overlay
             (no trigger words / avoid words / negative_embeddings).
             ``--models`` and ``--families`` are mutually exclusive.

        Internal flow:
          - **Preflight** (always, baseline ctx — must be model-kind).
          - **If new series**: SeriesPlanner.plan + SceneGenerator.generate
            + assign IDs + assign initial aspect ratios. Persist series
            and scenes rows once.
          - **If re-target**: load series + scenes from DB; reuse
            ``series.llm_series_plan`` JSON for niche `keyword_cluster`
            and variation `source_mode` lookups.
          - **For each target in targets** (normalized list of
            (target_kind, target_id) tuples; families first then
            models when both supplied):
              - Build per-target ctx: ``build_context(model_id=...)``
                for model-kind, ``build_family_context(family_id=...)``
                for family-kind. Family-kind ctx has no per-model
                overlay (no trigger_words, etc.).
              - If ``regen_facets`` includes this family: bulk-DELETE
                its facet rows across the series's scenes.
              - For each scene: call ``SceneFacetGenerator`` if the
                facet row is missing, else reuse the persisted facet.
                Facet rows are family-keyed and shared across
                model-kind and family-kind invocations on the same
                family — efficient.
              - If ``regen_prompts`` (model-kind) /
                ``regen_family_prompts`` (family-kind) includes this
                target_id: DELETE its existing prompts for this
                series before composing.
              - For each scene: merge facet into scene → PromptBuilder
                → sanitize → assemble negative → prompts table INSERT
                with target_kind discriminator.
              - In supervised mode, call ``Supervisor.approve_plan``
                once per target.
          - **Single LLM unload at end** (regardless of how many
            targets ran).

        Skips ``run_log`` writes — Phase A is not a render run. The
        prompts table itself records what was generated.

        Returns
        -------
        dict
            ``{series_id, status, scenes_created, facets_created,
              prompts_created, models_completed, families_completed}``.
            ``status`` is "complete" or "aborted" (supervisor rejected
            one target).
        """
        # Build targets list — families first, then models (preserves
        # CLI order semantics). At least one must be non-empty.
        models = models or []
        families = families or []
        if not models and not families:
            raise EngineError(
                "run_phase_a requires at least one of models / families"
            )
        targets: list[tuple[str, str]] = (
            [("family", f) for f in families]
            + [("model", m) for m in models]
        )
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

        # ── 2. Preflight (Phase A — Ollama reachability only) ──────
        # Render-time checks (checkpoint file + external template
        # validation) run from run_phase_b instead — Phase A is purely
        # LLM and doesn't touch ComfyUI files.
        self._preflight_phase_a(ctx, cli_llm_override=cli_llm_override)

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
        families_completed: list[str] = []
        # Round-22 F13 — reset the canonicalizer's drop counter so this
        # series's facet+prompt loop reports clean drift metrics at the
        # end of Phase A.
        from src.prompt.vocabulary import (
            reset_drop_counter as _vocab_reset_drops,
        )
        _vocab_reset_drops()

        for target_kind, target_id in targets:
            # Per-target ctx — model-kind uses build_context (existing
            # path with full per-model overlay); family-kind uses
            # build_family_context (no per-model overlay; family-only
            # ModelPromptGuide). The baseline ctx is always model-kind
            # (D3 invariant); per-target ctx may differ.
            try:
                if target_kind == "model":
                    model_ctx = build_context(
                        mode=ctx.mode,
                        content_level=ctx.content_level,
                        execution_mode=ctx.execution_mode,
                        style_profile=style_profile,
                        content_rules=ctx.content_rules,
                        db_path=self.db_path,
                        model_id=target_id,
                        commercial_mode=self._commercial_mode,
                    )
                else:  # target_kind == "family"
                    model_ctx = build_family_context(
                        family_id=target_id,
                        mode=ctx.mode,
                        content_level=ctx.content_level,
                        execution_mode=ctx.execution_mode,
                        style_profile=style_profile,
                        content_rules=ctx.content_rules,
                        db_path=self.db_path,
                        commercial_mode=self._commercial_mode,
                    )
            except Exception as exc:
                raise EngineError(
                    f"Could not build context for {target_kind} "
                    f"{target_id!r}: {exc}"
                ) from exc

            family = model_ctx.family
            guide = model_ctx.model_prompt_guide

            # Resolve the LLM for this family's facet generation. The
            # registry id (`facet_llm_id`) is what gets stamped on every
            # facet + prompt row; the ollama tag (`facet_ollama_id`) is
            # what OllamaClient.generate consumes. With --llm override,
            # both flow from a single registry lookup. Note: family-kind
            # and model-kind invocations on the same family share facet
            # rows (PK is (scene_id, family.id, llm_id) — independent
            # of target_kind).
            facet_llm_entry = self._llm_router.resolve_facet_family(
                family.prompt_style, override=cli_llm_override,
            )
            facet_llm_id = facet_llm_entry.id
            # Round-14 — model_tag picks the right backend identifier:
            # ollama_id for ollama-backed entries, lm_studio_id for
            # lm_studio entries. The pool dispatches by backend lookup.
            facet_ollama_id = facet_llm_entry.model_tag

            logger.info(
                "%s %s (family=%s, llm=%s): facets + prompts …",
                target_kind.capitalize(), target_id, family.id,
                facet_llm_id,
            )

            # Optional regen-facets for this family. Facets are
            # family-keyed (independent of target_kind), so regen-facets
            # applies to both kinds equally — one DELETE clears the
            # shared facet row.
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

            # Optional regen-prompts. Model-kind uses --regen-prompts;
            # family-kind uses --regen-family-prompts. The two flags
            # are kept distinct so a typo can't accidentally
            # cross-delete (model-prep deleting family rows, etc.).
            regen_list = (
                regen_prompts if target_kind == "model"
                else regen_family_prompts
            )
            if regen_list and target_id in regen_list:
                n = self._delete_prompts_for_target(
                    series_id, target_id, facet_llm_id,
                    target_kind=target_kind,
                )
                logger.info(
                    "regen_%s_prompts: deleted %d existing prompts for "
                    "(%s, %s, %s, %s)",
                    target_kind, n, series_id, target_kind, target_id,
                    facet_llm_id,
                )

            # Per-scene: ensure facet + compose prompt.
            mode_obj = self._mode_registry.get(model_ctx.mode)
            extra_keywords: list[str] | None = None
            if isinstance(mode_obj, NicheMode):
                extra_keywords = mode_obj.get_prompt_keywords(series_plan)

            prompts_for_target: list[dict[str, Any]] = []
            # Round-12 (2026-05-21) — per-(target, family, llm) diversity
            # tracker. Counts each axis's tag picks as facets are emitted;
            # SceneFacetGenerator queries it before each scene to inject
            # an "avoid over-used tags" nudge once any axis crosses the
            # 50% dominance threshold (after 4 facets). Re-target paths
            # (existing-series facets loaded from DB) feed those facets
            # in too so the nudge accounts for prior LLM picks.
            diversity_tracker = _DiversityTracker()
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
                        # Verifier round-2 I4 — pass series_plan's
                        # compatible_environments (already intersected
                        # upstream by the mode across theme + style_profile
                        # + niche) so the LLM's environment.setting menu
                        # narrows to theme-coherent locations.
                        facet = facet_gen.generate(
                            scene=scene,
                            family=family,
                            content_level=ctx.content_level,
                            prompt_guide=guide,
                            llm_directive=ctx.content_rules.llm_directive,
                            model=facet_ollama_id,
                            compatible_environments=(
                                series_plan.get("compatible_environments")
                                or None
                            ),
                            # Round-12 (2026-05-21) — narrow the narrative
                            # menu by category. Mirrors compatible_environments;
                            # the planner mode intersects category's
                            # compatible_narratives with style_profile's
                            # equivalent (when present) and surfaces the
                            # result on series_plan.
                            compatible_narratives=(
                                series_plan.get("compatible_narratives")
                                or None
                            ),
                            # 2026-05-23 — narrow the per-scene
                            # `art_style_reference` menu so a Lindbergh-
                            # anchored series can't pick Helmut Newton
                            # (Verifier audit C3). Style-profile-driven.
                            compatible_art_styles=(
                                series_plan.get("compatible_art_styles")
                                or None
                            ),
                            diversity_tracker=diversity_tracker,
                            # Round-22 (2026-05-22) — thread the series'
                            # subject_description so the facet LLM can
                            # pick nsfw_anatomy / nsfw_act coherent with
                            # the locked subject identity. Three-level
                            # fallback:
                            #   1. theme_mode emits subject_description
                            #   2. niche_mode emits subject_bias (alias)
                            #   3. style_mode / variation_mode emit
                            #      neither → fall back to a tier-aware
                            #      synthetic hint ("An adult woman with
                            #      mature features, ...") so the facet
                            #      LLM still sees a non-empty subject
                            #      anchor instead of "(not provided)".
                            #      Round-2 audit identified this as the
                            #      single highest-risk gap from F5.
                            subject_description=resolve_subject_anchor(
                                series_plan, ctx.content_level,
                            ),
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

                # Round-12 diversity tracker — record this scene's
                # picked tags (either freshly generated or loaded from
                # DB on a regen / re-target). Empty facet from the
                # SceneFacetGeneratorError fallback contributes nothing.
                if facet:
                    diversity_tracker.record(facet)

                # Step 2: merge facet into scene dict for the composer.
                scene_with_facet = {**scene, **facet}
                # Drop facet metadata keys.
                for k in ("scene_id", "family", "created_at"):
                    scene_with_facet.pop(k, None)

                # Step 3: resolve synthetic character context for the
                # composer. Character mode was deleted in the 2026-05-20
                # cleanup; every surviving mode (theme/style/niche/
                # variation) produces synthetic character data from the
                # series plan + scene fields.
                scene_character = self._synthetic_character_for_scene(
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
                        # Phase 3 (vocab v6) — series-level aesthetic
                        # anchors (color_palette / photographer_ref /
                        # art_movement) thread through to the composer
                        # so every scene in the series carries the
                        # same signature visual world.
                        series_plan=series_plan,
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
                    # Round-21 (2026-05-21) — when the planner provided
                    # its own aesthetic anchors (color_palette /
                    # photographer_ref / art_movement) the operator-
                    # archetype style_profile is overridden. Drop the
                    # archetype's flat ``base_negative_prompt`` from the
                    # stack — it carries quality-axis tokens calibrated
                    # for the archetype (e.g. ``golden_hour_natural``
                    # blocks "neon, low-key, indoor window-less") which
                    # directly contradict planner-chosen themes.
                    # ``base_style_keywords`` is already suppressed on
                    # the positive side in ``build_one``; the negative
                    # equivalent gets the same treatment here.
                    style_neg_value = (
                        None
                        if archetype_overridden_by_planner(series_plan)
                        else style_profile.get("base_negative_prompt")
                    )
                    prompt_dict["negative_prompt"] = (
                        self.prompt_builder.assemble_negative_prompt(
                            model_negative=(
                                guide.base_negative_prompt
                                if guide and not guide_axes
                                else None
                            ),
                            model_negative_axes=guide_axes,
                            style_negative=style_neg_value,
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
                    # PK includes facet_llm_id so the same series can
                    # carry parallel prompts from multiple LLMs (the
                    # canonical A/B workflow). Pre-2026-05-18 the ID
                    # formula was just (series, target, index), which
                    # collided with the existing cydonia prompts when
                    # retargeting with a second LLM — the schema's
                    # UNIQUE(scene, kind, target, llm) was supposed to
                    # protect this but the row's PK ``id`` collided
                    # first. Fix: include llm_id in the prompt PK.
                    prompt_dict["id"] = (
                        f"{series_id}_{target_id}_{facet_llm_id}_prompt_"
                        f"{len(prompts_for_target):03d}"
                    )
                    prompt_dict["content_level"] = model_ctx.content_level
                    # `model_id` column carries the target_id —
                    # model-id for model-kind, family-id for family-kind.
                    prompt_dict["model_id"] = target_id
                    prompts_for_target.append(prompt_dict)
                except Exception as exc:
                    logger.warning(
                        "Failed to build prompt for scene %s %s %s: %s",
                        scene_id, target_kind, target_id, exc,
                    )

            # Within-target dedup (different targets compose to different
            # text by design, so cross-target dedup wouldn't fire even
            # if the deduplicator ran across all prompts).
            prompts_for_target = self.deduplicator.deduplicate(
                prompts_for_target, scenes, model_ctx,
            )
            logger.info(
                "%s %s: %d prompts after dedup",
                target_kind.capitalize(), target_id,
                len(prompts_for_target),
            )

            # Persist prompts for this (target_kind, target_id, llm)
            # tuple. The llm_id matches the facet that fed the composer
            # — same resolution path so prompt and facet co-locate.
            self._save_prompts_for_target(
                series_id=series_id,
                ctx=model_ctx,
                prompts=prompts_for_target,
                llm_id=facet_llm_id,
                target_kind=target_kind,
            )
            prompts_created_total += len(prompts_for_target)
            if target_kind == "model":
                models_completed.append(target_id)
            else:
                families_completed.append(target_id)

            # Supervised pause once per target.
            if model_ctx.execution_mode == "supervised" and not self.dry_run:
                approved = self.supervisor.approve_plan(
                    mode=model_ctx.mode,
                    content_level=model_ctx.content_level,
                    model_id=target_id,
                    series_plan=series_plan,
                    scenes=scenes,
                    prompts=prompts_for_target,
                )
                if not approved:
                    self._update_series_status(series_id, "aborted")
                    logger.info(
                        "Plan rejected by supervisor for %s %s. "
                        "Aborting Phase A.", target_kind, target_id,
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
                        "families_completed": families_completed,
                    }

        # ── 5. Unload every LLM the cycle loaded ────────────────────
        # With per-role routing a single Phase A may have loaded
        # series_planner, scene_generator, and per-family
        # scene_facet_generator tags simultaneously. unload_all walks
        # client.loaded_models and frees each.
        self.llm_client.unload_all()
        logger.info("LLM unloaded. Phase A complete.")

        # Round-22 F13 — surface aggregate vocabulary-drop metrics so the
        # operator can spot planner-side hallucinations or LLM drift at
        # a glance. Sum across all canonicalize calls during this Phase
        # A (per-scene + per-series). 0% = clean; >20% suggests planner
        # is emitting invalid tag names that don't exist in the vocab.
        from src.prompt.vocabulary import (
            get_drop_counts as _vocab_get_drops,
        )
        drops = _vocab_get_drops()
        if any(drops.values()):
            logger.info(
                "Vocab drops this series: unknown=%d (LLM drift) "
                "tier_gated=%d (tier-min above content_level) "
                "family_omitted=%d (concept exists but no phrasing for "
                "this family) solo_banned=%d (partnered nsfw_act blocked "
                "by single-female enforcement)",
                drops["unknown"], drops["tier"],
                drops["family"], drops["solo"],
            )

        return {
            "series_id": series_id,
            "status": "complete",
            "scenes_created": len(scenes) if is_new_series else 0,
            "facets_created": facets_created_total,
            "prompts_created": prompts_created_total,
            "models_completed": models_completed,
            "families_completed": families_completed,
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
        target_kind: str = "model",
        render_model_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Phase B — render every (scene × prompt) for a (series, target).

        Two render modes since the family-level prompt-prep feature
        (2026-05):

        * ``target_kind='model'`` (default) — renders the model-kind
          prompts for ``model_id``. The render uses ``model_id``'s
          checkpoint. ``render_model_id`` is ignored.
        * ``target_kind='family'`` — renders the family-kind prompts
          stored under ``model_id`` (which holds a family id when
          target_kind='family'; e.g. 'flux'). ``render_model_id`` is
          REQUIRED and must be a model whose family matches
          ``model_id``; the render uses ``render_model_id``'s
          checkpoint. CLI validates the family-membership match.

        Reloads everything it needs from the DB (series row, scenes,
        prompts), rebuilds the GenerationContext for the **render
        checkpoint** (which is ``render_model_id`` in family-kind, or
        ``model_id`` in model-kind), recomputes per-target-model
        resolution at render time, runs the render loop with retries,
        scores, and (in supervised mode) prompts the operator for
        image-level review.

        Returns ``(rendered_images, ctx_state)``. ``ctx_state`` carries
        everything :meth:`run_phase_c` needs (``series_plan, scenes,
        prompts, ctx, style_profile, style_profile_id``, plus
        ``target_kind`` and ``target_id`` for output paths) so the
        caller doesn't have to re-load.

        Used by ``render_prompts.py`` (standalone Phase B+C invocation
        on an existing series) and indirectly by :meth:`run_cycle` (which
        wraps phase_a → phase_b → phase_c, always model-kind).

        Raises
        ------
        EngineError
            Series not in DB; no prompts for the (series, target_kind,
            target_id, llm) tuple; target_kind='family' but
            render_model_id is None or the model's family doesn't
            match.
        """
        # Resolve the render checkpoint id. For model-kind, this is
        # just `model_id`. For family-kind, it's the explicit
        # `render_model_id` (validated upstream).
        if target_kind == "family":
            if not render_model_id:
                raise EngineError(
                    "run_phase_b: target_kind='family' requires "
                    "render_model_id (which model checkpoint to use). "
                    "Pass --render-with-model on the CLI."
                )
            checkpoint_model_id = render_model_id
        else:
            checkpoint_model_id = model_id

        logger.info(
            "Phase B: Rendering series=%s target_kind=%s target_id=%s "
            "render_with=%s …",
            series_id, target_kind, model_id, checkpoint_model_id,
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

            # Reload scenes (model-agnostic).
            scenes_query = "SELECT * FROM scenes WHERE series_id = ?"
            scenes_params: list[Any] = [series_id]
            if scene_ids:
                placeholders = ", ".join(["?"] * len(scene_ids))
                scenes_query += f" AND id IN ({placeholders})"
                scenes_params.extend(scene_ids)
            scene_rows = conn.execute(scenes_query, scenes_params).fetchall()
            scenes = [dict(r) for r in scene_rows]

            # Reload prompts filtered by (series, target_kind, model_id,
            # llm_id, optional scenes). model_id holds the target_id
            # (image-model id for target_kind='model'; family id for
            # target_kind='family').
            phase_b_llm_id = (
                cli_llm_override
                if cli_llm_override is not None
                else self._default_llm_id
            )
            prompts_query = (
                "SELECT * FROM prompts "
                "WHERE series_id = ? AND target_kind = ? "
                "AND model_id = ? AND llm_id = ?"
            )
            prompts_params: list[Any] = [
                series_id, target_kind, model_id, phase_b_llm_id,
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
                f"target_kind={target_kind!r}, target_id={model_id!r}, "
                f"llm_id={phase_b_llm_id!r}. Run prepare_prompts "
                f"--{'families' if target_kind == 'family' else 'models'} "
                f"{model_id} first."
            )

        # ── 2. Reload style_profile + content_rules ─────────────────
        style_profile = _load_style_profile(self.db_path, style_profile_id)
        content_rules = ContentLevelLoader(self.db_path).load(content_level)

        # ── 3. Rebuild ctx for the TARGET model ─────────────────────
        # The target model may differ from the model used to create
        # the series — that's the whole point of multi-model rendering.
        # Render-time ctx uses the CHECKPOINT model (which equals
        # model_id for target_kind='model', or render_model_id for
        # target_kind='family'). The render workflow needs a concrete
        # model_config for the family/checkpoint lookup.
        ctx = build_context(
            mode=mode_name,
            content_level=content_level,
            execution_mode=self.execution_mode,
            style_profile=style_profile,
            content_rules=content_rules,
            db_path=self.db_path,
            model_id=checkpoint_model_id,
            commercial_mode=self._commercial_mode,
        )

        # ── 4. Render-time preflight (checkpoint / external-template) ─
        # Fail fast BEFORE memory/status work — a missing checkpoint
        # is cheaper to surface here than to crash inside ComfyUI.
        self._preflight_phase_b(ctx)

        # ── 5. Memory preflight + status update ─────────────────────
        self._memory_preflight(min_free_gb=14.0)
        self._update_series_status(series_id, "rendering")

        # ── 5. Resolve mode object (used for run_log labeling) ──────
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

        # External templates carry their own sampler / scheduler / steps
        # / cfg / VAE / CLIP / LoRA wiring; the pipeline only injects
        # positive/negative prompt text + seed + resolution.

        # ── 9. Render loop ──────────────────────────────────────────
        wf_builder = WorkflowBuilder(self.workflow_dir)
        comfy_client = ComfyUIClient(
            base_url=self.comfy_base_url,
            output_dir=self.comfy_output_dir,
        )
        rendered_images: list[dict[str, Any]] = []
        # Per-LLM + per-target output dir so multi-LLM and
        # model-vs-family renders on the same series don't overwrite
        # each other. target_id is the model_id for target_kind='model'
        # or the family_id for target_kind='family' — collapses to a
        # single segment. Plan §3.5 + family-level prep D7.
        output_series_dir = (
            self.output_dir
            / content_level
            / series_id
            / phase_b_llm_id
            / model_id  # = target_id (image model id or family id)
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
                # Resolve the template: explicit --templates override
                # wins, otherwise fall back to the per-family default
                # from families.yaml. Mismatched family raises a clean
                # error so an sdxl prompt can't be rendered through a
                # flux template.
                resolved_template = self._resolve_template_for_render(
                    family_id=target_family_id,
                    override=template_override,
                )
                # Family-match validator — convention-only (raises only
                # when the template path follows `templates/<family>/`
                # AND the extracted family disagrees with the prompt's
                # family). Out-of-convention paths skip the check.
                tf = _extract_family_from_template_path(resolved_template)
                if tf is not None and tf != target_family_id:
                    raise EngineError(
                        f"Template {resolved_template!r} belongs to "
                        f"family {tf!r} but prompt is for family "
                        f"{target_family_id!r}. Render mismatched "
                        f"families is not supported."
                    )
                workflow = wf_builder.build_external(
                    external_template=resolved_template,
                    prompt_text=prompt["prompt_text"],
                    negative_prompt=prompt["negative_prompt"],
                    resolution=resolution,
                )
            except WorkflowTemplateError as exc:
                logger.error("Workflow build failed: %s", exc)
                continue

            # Refiner-stage detection — populates the two PNG metadata
            # fields (`refiner_used`, `refiner_checkpoint`). Inspecting
            # the resolved workflow rather than the template-on-disk
            # means a built-in template that wires a refiner stage (when
            # we add one later) gets the same forensic record. False/None
            # for any template without the refiner contract IDs.
            refiner_used = "refiner_positive_prompt" in workflow
            refiner_checkpoint = None
            if "refiner_checkpoint_loader" in workflow:
                rcl_inputs = workflow["refiner_checkpoint_loader"].get(
                    "inputs", {}
                )
                refiner_checkpoint = (
                    rcl_inputs.get("ckpt_name")
                    or rcl_inputs.get("unet_name")
                )

            # Read seed from the workflow we just built — ComfyUI's
            # history response (RenderedImage) doesn't carry the seed
            # back, so getattr(ci, "seed", 0) always returned 0 and
            # broke PNG-metadata reproducibility. The workflow was
            # patched with the chosen seed at build time, so this is
            # the authoritative source. Handles both ksampler.seed
            # (every family + every external template) and
            # random_noise.noise_seed (Chroma's built-in graph).
            seed_val = _extract_seed_from_workflow(workflow)

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
                # Phase 4b — embed AUTOMATIC1111 + nsfw_pipeline PNG
                # metadata chunks so the file carries its own
                # reproduction parameters even if disconnected from
                # the SQLite DB.
                # Sampler/scheduler/steps/cfg live in the external-template
                # JSON; defensive `.get()` chain so missing keys produce a
                # degraded A1111 string but do not block the render.
                ks_inputs = workflow.get("ksampler", {}).get("inputs", {}) or {}
                self._embed_png_metadata(
                    dst_path,
                    prompt=prompt,
                    model_id=model_id,  # target_id (family id when family-kind)
                    series_id=series_id,
                    seed=seed_val,
                    resolution=resolution,
                    scene=scene,
                    content_level=content_level,
                    family_id=target_family_id,
                    sampler=ks_inputs.get("sampler_name"),
                    scheduler=ks_inputs.get("scheduler"),
                    steps=ks_inputs.get("steps"),
                    cfg=ks_inputs.get("cfg"),
                    clip_skip=None,  # no fixed node home in external templates
                    target_kind=target_kind,
                    render_model_id=(
                        checkpoint_model_id
                        if target_kind == "family"
                        else None
                    ),
                    refiner_used=refiner_used,
                    refiner_checkpoint=refiner_checkpoint,
                    series_plan=series_plan,
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
            # Family-level prompt-prep — surface target_kind and
            # target_id so run_phase_c's Exporter can place the
            # exported set under <target_id>/ (matches the per-target
            # output_series_dir shape).
            "target_kind": target_kind,
            "target_id": model_id,
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
                    img["file_path"],
                    prompt=img.get("prompt_text"),
                    content_level=content_level,
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

        # Post-processing (upscale, face-detailer) lives inside the
        # external-template JSON — see gonzaLomo_Chroma_Refiner_v11.json
        # for an example that wires an SDXL refiner stage + FaceDetailer
        # after the Chroma base.

        # ── 12. Supervised pause 2 (image review) ───────────────────
        if ctx.execution_mode == "supervised":
            # Per-LLM + per-target preview dir matches output_series_dir.
            preview_dir = (
                self.output_dir
                / content_level
                / series_id
                / phase_b_llm_id
                / model_id  # = target_id
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
        target_kind: str = "model",
        target_id: str | None = None,
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
            supports_negative_prompt + family.llm_temperature (metadata
            generation tuning).
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
        # can route metadata to a specialised LLM (e.g. one tuned for
        # social-media-platform tags) while keeping the default for
        # the rest of the pipeline. With routing disabled, every role
        # collapses to default_llm.
        metadata = None
        meta_llm_entry = self._llm_router.resolve_role(
            "metadata_generator", override=cli_llm_override,
        )
        meta_model = meta_llm_entry.model_tag
        try:
            meta_gen = MetadataGenerator(self.llm_client)
            metadata = meta_gen.generate(
                theme=series_plan.get("theme", ""),
                mood=series_plan.get("mood", ""),
                environment=series_plan.get("environment", ""),
                character_name="",
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

        # Export. llm_id + target_id flow from run_phase_b's ctx_state
        # so exports land under
        # output/<level>/<series>/<llm_id>/<target_id>/. Without
        # target_id, two A/B targets (model + family, or two families)
        # rendering the same series under the same LLM would silently
        # overwrite each other's exports.
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
            target_id=target_id or model_id,
            target_kind=target_kind,
        )

        # Persist images to DB
        self._persist_images(series_id, selected, ctx)

        # Record to memory (anti-repetition)
        self.memory.record_series(
            series_plan, scenes, prompts,
            content_level=content_level,
        )

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

    def _preflight_phase_a(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> None:
        """Phase A (LLM planning) preflight — Ollama reachability only.

        Phase A produces prompts in the DB; nothing renders here. The
        checkpoint file + external template validation are all
        render-time concerns and live in :meth:`_preflight_phase_b`.
        Pre-2026-05-06 these were bundled into a single ``_preflight``
        called from Phase A — which broke ``prepare_prompts --families
        <f>`` (the family-mode baseline ctx picks an arbitrary family
        member; its ``.gguf`` file may not be installed yet, and Phase
        A doesn't need it).
        """
        if not self.llm_client.is_available():
            # Round-14/20 — pool.is_available is true if ANY backend is
            # reachable. False here means every registered backend is
            # down. Message each endpoint so operator knows which to
            # start.
            ollama_url = self.llm_client.ollama.base_url
            lm_studio_url = self.llm_client.lm_studio.base_url
            mlx_url = self.llm_client.mlx.base_url
            raise PreflightError(
                "Preflight check(s) failed:\n"
                f"  - No LLM backend reachable.\n"
                f"    Ollama at {ollama_url}: not reachable. "
                f"Run `ollama serve` if you use Ollama.\n"
                f"    LM Studio at {lm_studio_url}: not reachable. "
                f"Start the LM Studio local server if you use LM Studio.\n"
                f"    MLX at {mlx_url}: not reachable. "
                f"Start mlx_lm.server if you use MLX (see "
                f"pipeline.yaml::mlx for the startup command)."
            )
        logger.info("Preflight (Phase A): LLM backend reachable")
        # Round-17 (2026-05-21) — eager-load ONLY the LLM that will
        # actually be used this run. Round-14 pre-loaded every active
        # LM Studio entry, which on a 2-LLM install meant a 14 GB
        # Cydonia load even when the run targeted a 10 GB Qwen3.5.
        # Now we resolve the override/default first, check its
        # backend, and only ensure-load if it's LM-Studio-backed.
        self._ensure_lm_studio_models_loaded(
            ctx, cli_llm_override=cli_llm_override,
        )

    def _ensure_lm_studio_models_loaded(
        self,
        ctx: GenerationContext,
        *,
        cli_llm_override: str | None = None,
    ) -> None:
        """Targeted LM-Studio ensure-load: only the LLM that will
        actually be used this run.

        Resolution: ``cli_llm_override`` wins when set (every agent
        role collapses onto it for the run); otherwise the registry's
        ``default_llm`` is used (per-role routing isn't easily
        forecast here, but ``default_llm`` is by definition the
        fallback every role lands on when no routing rule matches).
        Skips the load entirely for Ollama-backed LLMs — Ollama JIT-
        loads with the model's metadata-defined context length.
        """
        from src.memory.llm_registry import (
            BACKEND_LM_STUDIO, LLMRegistryLoader,
        )

        registry = ctx.llm_registry if hasattr(ctx, "llm_registry") else None
        if registry is None:
            registry = LLMRegistryLoader()
        try:
            entry = (
                registry.get_llm(cli_llm_override, require_active=True)
                if cli_llm_override
                else registry.get_default_llm()
            )
        except Exception as exc:
            logger.warning(
                "ensure_loaded: could not resolve target LLM "
                "(override=%r): %s — skipping LM Studio pre-load.",
                cli_llm_override, exc,
            )
            return
        if entry.backend != BACKEND_LM_STUDIO:
            return
        ctx_tokens = entry.context_tokens or 32768
        try:
            self.llm_client.lm_studio.ensure_loaded(
                entry.lm_studio_id, context_length=ctx_tokens,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "ensure_loaded failed for LM Studio model %s: %s "
                "— will retry inline on first call.",
                entry.lm_studio_id, exc,
            )

    def _preflight_phase_b(self, ctx: GenerationContext) -> None:
        """Phase B (render) preflight — checkpoint + external template.

        After the 2026-05-20 cleanup, rendering ONLY goes through
        external templates (built-in `<family>/base.json` deleted).
        Two checks:

        1. **Checkpoint file** — the `.safetensors` / `.gguf` file
           declared on the model YAML exists on disk in
           `~/AI/apps/ComfyUI/models/<family>/<filename>`. This is
           independent of the template (the template references the
           checkpoint by name).
        2. **External template** — load + validate the 4-node contract
           + refiner-pair consistency. Template path comes from
           `--templates` override OR the family's `default_template`
           in `config/families.yaml`.

        Aggregates errors so the user sees every missing-file issue
        at once instead of one-at-a-time.
        """
        errors: list[str] = []

        # Checkpoint file existence (independent of template).
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

        # External-template validation (resolved via override or family default).
        try:
            resolved = self._resolve_template_for_render(
                family_id=ctx.model_config.family,
                override=self.template_override,
            )
            wf_builder = WorkflowBuilder(self.workflow_dir)
            workflow = wf_builder._load(resolved)
            template_name = Path(resolved).name
            WorkflowBuilder._assert_required_nodes(
                workflow, "external", template_name,
                _REQUIRED_NODES_EXTERNAL,
            )
            _assert_external_template_inputs(workflow, template_name)
            # Family-match check — only when the path follows the
            # `templates/<family>/X.json` convention. Out-of-convention
            # paths skip the check (user gets the runtime error from
            # the build_external call instead).
            tf = _extract_family_from_template_path(resolved)
            if tf is not None and tf != ctx.model_config.family:
                errors.append(
                    f"Template {resolved!r} belongs to family {tf!r} but "
                    f"prompt is for family {ctx.model_config.family!r}. "
                    f"Render mismatched families is not supported."
                )
        except WorkflowTemplateError as exc:
            errors.append(f"Template validation failed: {exc}")
        except EngineError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Template validation failed: {exc!r}")

        if errors:
            msg = "Preflight check(s) failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise PreflightError(msg)

        logger.info("Preflight (Phase B): all render-time checks passed")

    def _resolve_template_for_render(
        self, *, family_id: str, override: str | None,
    ) -> str:
        """Resolve the template path for a render: explicit ``--templates``
        override wins; otherwise fall back to the family's
        ``default_template`` from ``config/families.yaml``.

        Raises ``EngineError`` when neither is set — no silent fallback
        to a "system default" since system templates were deleted in the
        2026-05-20 cleanup.
        """
        if override:
            return _resolve_template_path(
                override, self.workflow_dir, _PROJECT_ROOT,
            )
        # Look up family default
        from src.memory.family_loader import FamilyLoader
        fam = FamilyLoader().get_family(family_id)
        default = getattr(fam, "default_template", None)
        if default:
            return _resolve_template_path(
                default, self.workflow_dir, _PROJECT_ROOT,
            )
        raise EngineError(
            f"No default template set for family {family_id!r}. "
            f"Set families.yaml::{family_id}::default_template or pass "
            f"--templates <path>."
        )

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
        target_kind: str = "model",
        render_model_id: str | None = None,
        refiner_used: bool = False,
        refiner_checkpoint: str | None = None,
        series_plan: Mapping[str, Any] | None = None,
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
            # Verifier round-3 IMPORTANT-4 — series-level aesthetic
            # anchors persisted in the PNG so a forensic reader can
            # reproduce the signature look without DB access. The 3
            # fields are SeriesPlanner's Phase 3 output (color_palette,
            # photographer_ref, art_movement); when missing on older
            # series, the metadata payload simply omits the
            # series_aesthetic sub-dict.
            series_aesthetic: dict[str, Any] | None = None
            if series_plan:
                aesthetic = {
                    k: series_plan.get(k) for k in (
                        "color_palette",
                        "photographer_ref",
                        "art_movement",
                    )
                }
                if any(v for v in aesthetic.values()):
                    series_aesthetic = aesthetic

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
                # Family-mode forensics: when target_kind='family',
                # model_id holds the family id (e.g. 'flux') and
                # render_model_id holds the actual checkpoint
                # ('gonzalomo_flux_v30'). For target_kind='model' both
                # collapse — render_model_id stays None.
                target_kind=target_kind,
                render_model_id=render_model_id,
                refiner_used=refiner_used,
                refiner_checkpoint=refiner_checkpoint,
                series_aesthetic=series_aesthetic,
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
        :meth:`_save_prompts_for_target` (always with
        target_kind='model' — dry-run is a model-level path).
        """
        self._save_series_and_scenes(
            series_id=series_id,
            ctx=ctx,
            series_plan=series_plan,
            scenes=scenes,
            style_profile_id=style_profile_id,
            target_count=len(prompts),
        )
        self._save_prompts_for_target(
            series_id=series_id,
            ctx=ctx,
            prompts=prompts,
            llm_id=self._default_llm_id,
            target_kind="model",
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
        are persisted separately via :meth:`_save_prompts_for_target`
        (one call per (target_kind, target_id) in the per-target
        fan-out).
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO series (
                    id, mode, content_level, style_profile_id,
                    theme, mood, environment, variation_axes, target_count,
                    actual_count, status, llm_series_plan
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    series_id,
                    ctx.mode,
                    ctx.content_level,
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

    def _save_prompts_for_target(
        self,
        *,
        series_id: str,
        ctx: GenerationContext,
        prompts: list[dict[str, Any]],
        llm_id: str,
        target_kind: str = "model",
    ) -> None:
        """Insert one (target_kind, target_id, llm) batch of prompts.

        Called once per target tuple in :meth:`run_phase_a`'s fan-out.
        The UNIQUE(scene_id, target_kind, model_id, llm_id) constraint
        protects against accidental double-insert; re-roll requires
        explicit :meth:`_delete_prompts_for_target` first.

        ``target_kind`` discriminates the row:
          * ``'model'`` — `model_id` column carries an image-model id
            (validated against ``config/models/*.yaml``). Per-prompt
            ``model_id`` key wins over ``ctx.model_id`` when present
            (multi-model loop sets it explicitly).
          * ``'family'`` — `model_id` column carries a family id
            (validated against ``config/families.yaml``). Falls back
            to ``ctx.family.id`` when the per-prompt key is absent.

        ``llm_id`` is method-level (not per-prompt) because every
        prompt in this call comes from the same LLM resolution.
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
                # target_id resolution per D2: per-prompt key wins,
                # else ctx.model_id (model-kind) or ctx.family.id
                # (family-kind). Defensive assert before INSERT —
                # NULL would trip the schema NOT NULL constraint.
                target_id = prompt.get("model_id")
                if target_id is None:
                    target_id = (
                        ctx.model_id
                        if ctx.target_kind == "model"
                        else ctx.family.id
                    )
                if target_id is None:
                    raise ValueError(
                        f"_save_prompts_for_target: cannot resolve "
                        f"target_id for prompt {prompt.get('id')!r} "
                        f"(target_kind={target_kind!r}, "
                        f"ctx.target_kind={ctx.target_kind!r})"
                    )
                conn.execute(
                    """
                    INSERT INTO prompts (
                        id, series_id, scene_id, target_kind, model_id,
                        llm_id, prompt_text, negative_prompt,
                        prompt_hash, content_level, vocab_version,
                        status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        prompt["id"],
                        series_id,
                        prompt.get("scene_id"),
                        target_kind,
                        target_id,
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

    def _delete_prompts_for_target(
        self,
        series_id: str,
        target_id: str,
        llm_id: str,
        target_kind: str = "model",
    ) -> int:
        """DELETE all prompts for one (series, target_kind, target_id, llm).

        Used by :meth:`run_phase_a` when ``regen_prompts`` /
        ``regen_family_prompts`` includes the target_id — clears the
        way for fresh INSERTs without tripping
        UNIQUE(scene_id, target_kind, model_id, llm_id). Other LLMs'
        prompts on the same (series, target) are untouched so a
        Cydonia regen doesn't blow away Magnum's parallel data, and
        family-kind regen never touches model-kind rows (and vice
        versa).

        Returns the number of rows deleted.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.execute(
                "DELETE FROM prompts "
                "WHERE series_id = ? AND target_kind = ? "
                "AND model_id = ? AND llm_id = ?",
                (series_id, target_kind, target_id, llm_id),
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
        ``keyword_cluster`` and variation ``source_mode`` that downstream
        PromptBuilder may need.

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
    def _synthetic_character_for_scene(
        *,
        series_plan: dict[str, Any],
        style_profile: dict[str, Any],
        scene: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the synthetic character dict the PromptBuilder needs.

        Every surviving mode (theme/style/niche/variation) produces a
        synthetic character context from the series plan + scene fields.
        Per-scene `subject_detail` overrides the series-level
        `subject_description` / `subject_bias` when present.

        Round-16 (2026-05-21) — added a defensive fallback chain when
        every primary source is empty. The 2026-05-21 Qwen3.5-9b MLX
        prep showed both `subject_detail` (per-scene, scene schema
        `extra="allow"`) and `subject_description` (series-level,
        SeriesPlan schema `extra="allow"`) coming through empty —
        neither field is strictly required by their Pydantic schemas
        so any LLM that doesn't emit them silently sets base_prompt
        to "". PromptBuilder then raises and every scene's prompt
        fails. The fallback assembles a one-sentence subject anchor
        from the scene's pose + camera + expression so the prompt
        body still has SOMETHING to lead with.

        Returns a dict with `base_prompt` + `negative_prompt` keys.
        ``base_prompt`` is guaranteed non-empty.

        Round-21 verification (2026-05-21) — when the planner provided
        its own aesthetic anchors (color_palette / photographer_ref /
        art_movement), the operator-archetype style_profile is
        overridden. Drop the archetype's flat ``base_negative_prompt``
        here too; otherwise it lands on ``scene_character.negative_prompt``
        and ``assemble_negative_prompt`` pulls it back in via the
        ``character_negative=`` parameter — bypassing the engine-level
        suppression added in the first round-21 patch. Discovered when
        the round-21 verification run still emitted golden_hour_natural's
        "studio flash, fluorescent, low-key, neon, indoor window-less,
        harsh midday sun" tokens on every prompt despite the engine-
        level fix being in place.
        """
        if archetype_overridden_by_planner(series_plan):
            negative = ""
        else:
            negative = style_profile.get("base_negative_prompt", "")
        # Per-scene override wins when present.
        sd = scene.get("subject_detail", "") if scene else ""
        if sd:
            return {"base_prompt": sd, "negative_prompt": negative}
        subject = (
            series_plan.get("subject_description")
            or series_plan.get("subject_bias")
            or ""
        )
        # Style mode prepends style_keywords for an extra style anchor.
        if series_plan.get("style_keywords"):
            sk = series_plan["style_keywords"]
            subject = f"{sk}, {subject}" if subject else sk
        if subject:
            return {"base_prompt": subject, "negative_prompt": negative}

        # Round-16 fallback — neither subject_detail nor subject_
        # description were emitted by the LLM. Synthesize a minimum-
        # viable subject anchor from the scene's other fields so
        # PromptBuilder doesn't reject the whole scene. Order: pose →
        # camera (shot type) → expression. "adult woman" is always
        # prepended (single-female invariant from CLAUDE.md).
        scene_fields: list[str] = []
        if scene:
            for key in ("pose", "camera", "expression"):
                val = scene.get(key)
                if val:
                    scene_fields.append(str(val).strip())
        if scene_fields:
            fallback = "adult woman, " + ", ".join(scene_fields)
        else:
            # Hard floor — even with no scene either, ship a generic
            # adult-woman anchor rather than letting PromptBuilder
            # raise. Should be unreachable in production (scenes
            # always have at least pose).
            fallback = "adult woman"
        logger.warning(
            "Synthetic character: both subject_detail (per-scene) "
            "and subject_description (series-level) are empty; "
            "falling back to scene-derived anchor %r. Re-prep with a "
            "stricter LLM if this hurts prompt quality.",
            fallback,
        )
        return {"base_prompt": fallback, "negative_prompt": negative}

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

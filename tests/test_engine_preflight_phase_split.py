"""Tests for the Phase A / Phase B preflight split.

Pre-2026-05-06 the engine had a single ``_preflight(ctx)`` called from
``run_phase_a`` that bundled Ollama reachability with render-time disk
checks (checkpoint file, workflow JSON, IPAdapter capability,
external-template validation). The disk checks fired during prompt
prep — which is LLM-only — and broke ``prepare_prompts --families
<f>`` because the family-mode baseline ctx picks an arbitrary family
member whose ``.gguf`` may not be installed.

The fix splits ``_preflight`` into:

* ``_preflight_phase_a(ctx)`` — Ollama reachability only. Called from
  ``run_phase_a``.
* ``_preflight_phase_b(ctx)`` — checkpoint file, workflow JSON,
  IPAdapter, external-template. Called from ``run_phase_b``.

These tests prove the split holds: Phase A is checkpoint-blind,
Phase B is the home of the disk checks, and neither method
duplicates the other's responsibilities.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_TEMPLATE = (
    PROJECT_ROOT
    / "config"
    / "comfyui_workflows"
    / "templates"
    / "chroma"
    / "chroma_done_properly.json"
)


@pytest.fixture
def engine_module():
    from src.core import engine
    return engine


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    """Workflow dir WITHOUT any system-path templates — base.json,
    ipadapter.json, etc. are absent. Use this to verify Phase A is
    happy with no render-time files at all (the bug fix)."""
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True)
    return wf_dir


@pytest.fixture
def workflow_dir_with_external_template(tmp_path: Path) -> Path:
    """Workflow dir with the canonical chroma external template
    available — used to test that Phase B's external-template branch
    skips the system-path checks."""
    wf_dir = tmp_path / "workflows"
    (wf_dir / "templates" / "chroma").mkdir(parents=True)
    shutil.copy(
        _CANONICAL_TEMPLATE,
        wf_dir / "templates" / "chroma" / "chroma_done_properly.json",
    )
    return wf_dir


def _fake_ctx(**overrides) -> SimpleNamespace:
    base = dict(
        mode="character",
        model_id="chroma_1hd",
        model_config=SimpleNamespace(
            family="chroma",
            filename="Chroma1-HD-Q8_0.gguf",
            supports_ipadapter=False,
            supports_lora=True,
        ),
        supports_ipadapter=False,
        character={"id": "char_test", "reference_image_path": None},
        character_id="char_test",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _engine(
    engine_module,
    workflow_dir: Path,
    monkeypatch,
    *,
    ollama_available: bool = True,
    template_override: str | None = None,
):
    """PipelineEngine pointed at a workflow_dir with no system templates,
    a checkpoint folder that doesn't exist, and Ollama defaulting to up.
    Individual tests can flip the Ollama flag."""
    monkeypatch.setattr(
        "src.agents.llm_client.OllamaClient.is_available",
        lambda self: ollama_available,
    )
    engine = engine_module.PipelineEngine(
        config={
            "pipeline": {"default_model_id": "chroma_1hd", "output_dir": "output"},
            "comfyui": {"base_url": "http://127.0.0.1:8188"},
            "execution": {"mode": "manual"},
            "aspect_ratio_weights": {
                "T1_suggestive": {"portrait_23": 1.0},
                "T2_implied": {"portrait_23": 1.0},
                "T3_artnude": {"portrait_23": 1.0},
                "T4_explicit": {"portrait_23": 1.0},
            },
        },
        db_path=Path("/tmp/not_a_real_db"),
        dry_run=True,
        template_override=template_override,
    )
    engine.workflow_dir = workflow_dir
    # comfy_output_dir's parent / "models" / <subfolder> is where the
    # checkpoint check looks. Point at a path that does not exist —
    # any system-path checkpoint check will surface "Checkpoint file
    # not found".
    engine.comfy_output_dir = workflow_dir.parent / "nonexistent_comfy"
    return engine


# ── 1. The user's bug: Phase A succeeds when checkpoint is absent ──


def test_preflight_phase_a_succeeds_when_checkpoint_absent(
    engine_module, workflow_dir, monkeypatch,
):
    """Repro of the reported bug. ``prepare_prompts --families flux``
    runs Phase A only; the family-mode baseline ctx picks a model
    whose ``.gguf`` file isn't on disk. Phase A must not care —
    rendering happens in Phase B, and that's where the file matters.

    Pre-fix this raised ``PreflightError("Checkpoint file not found:
    ...")`` from ``run_phase_a:_preflight``.
    """
    engine = _engine(engine_module, workflow_dir, monkeypatch)
    # No file exists under engine.comfy_output_dir.parent — but
    # Phase A should not look there at all.
    engine._preflight_phase_a(_fake_ctx())  # must not raise


def test_preflight_phase_a_succeeds_when_workflow_json_absent(
    engine_module, workflow_dir, monkeypatch,
):
    """Sister case: no chroma/base.json under workflow_dir. Phase A
    is LLM-only; the workflow JSON is read by ComfyUI in Phase B."""
    engine = _engine(engine_module, workflow_dir, monkeypatch)
    assert not (workflow_dir / "chroma" / "base.json").exists()
    engine._preflight_phase_a(_fake_ctx())  # must not raise


# ── 2. Secondary fix: Phase B fast-fails on missing checkpoint ─────


def test_preflight_phase_b_raises_when_checkpoint_absent(
    engine_module, workflow_dir, monkeypatch,
):
    """Pre-fix, ``run_phase_b`` only ran ``_memory_preflight`` — so a
    missing checkpoint surfaced as a deeper, opaque ComfyUI error.
    Post-fix, Phase B fast-fails with the same ``PreflightError`` the
    full-cycle path used to produce."""
    engine = _engine(engine_module, workflow_dir, monkeypatch)
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight_phase_b(_fake_ctx())
    msg = str(exc.value)
    assert "Checkpoint file not found" in msg
    assert "Chroma1-HD-Q8_0.gguf" in msg


# ── 3. Regression guard: Phase A still requires Ollama ─────────────


def test_preflight_phase_a_raises_when_ollama_unreachable(
    engine_module, workflow_dir, monkeypatch,
):
    engine = _engine(
        engine_module, workflow_dir, monkeypatch,
        ollama_available=False,
    )
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight_phase_a(_fake_ctx())
    msg = str(exc.value)
    assert "Ollama" in msg or "ollama" in msg


# ── 4. Phase B branch on --template skips system-path checks ───────


def test_preflight_phase_b_skips_system_checks_under_template(
    engine_module, workflow_dir_with_external_template, monkeypatch,
):
    """When ``template_override`` is set, the external-template branch
    fires and the system-path checkpoint / base.json / IPAdapter
    checks are skipped. Existing behavior preserved by the split."""
    engine = _engine(
        engine_module,
        workflow_dir_with_external_template,
        monkeypatch,
        template_override="templates/chroma/chroma_done_properly.json",
    )
    # No checkpoint file anywhere; no chroma/base.json. Both would
    # fire under the system-path branch.
    engine._preflight_phase_b(_fake_ctx())  # must not raise


# ── 5. Split is real (no duplicated responsibilities) ──────────────


def test_preflight_phase_b_does_not_check_ollama(
    engine_module, workflow_dir_with_external_template, monkeypatch,
):
    """Phase B should NOT re-check Ollama — that's Phase A's job, and
    duplicating it would mean a render-only invocation that happens
    after the LLM is intentionally torn down would falsely fail."""
    engine = _engine(
        engine_module,
        workflow_dir_with_external_template,
        monkeypatch,
        ollama_available=False,                 # Ollama explicitly down
        template_override="templates/chroma/chroma_done_properly.json",
    )
    # Phase B must succeed despite Ollama being down — render doesn't
    # need it. (Phase A intentionally unloads the LLM before Phase B
    # starts.)
    engine._preflight_phase_b(_fake_ctx())  # must not raise


def test_preflight_phase_a_does_not_check_template(
    engine_module, workflow_dir, monkeypatch,
):
    """Phase A should NOT validate ``template_override`` — that's a
    render-time concern. A user could prep prompts on a machine that
    doesn't yet have the external template downloaded."""
    engine = _engine(
        engine_module, workflow_dir, monkeypatch,
        template_override="templates/chroma/does_not_exist.json",
    )
    engine._preflight_phase_a(_fake_ctx())  # must not raise

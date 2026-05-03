"""Engine preflight for ``--template`` / ``template_override``.

When ``PipelineEngine(template_override=...)`` is set, preflight
must:

- Accept the new kwarg on the constructor.
- Validate the external template BEFORE pinging Ollama (the caller
  doesn't want to wait for a 2s Ollama probe just to be told their
  template is malformed).
- Skip the system-path checks that are irrelevant: ``{family}/base.json``
  existence, the model checkpoint file existence, and the IPAdapter-
  capability rule (character-with-reference-image + model without
  ``supports_ipadapter``). None of those matter because the external
  template ships its own graph, its own checkpoint, and IPAdapter is
  forced off for this run.
- Still require Ollama, because Phase A LLM planning runs regardless
  of which workflow renders the result.

Fixture strategy mirrors ``test_workflow_builder_external_template.py``:
the canonical ``chroma_done_properly.json`` is copied into a tmp
workflow_dir and mutated for negative-path tests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


# Import lazily inside fixtures so failures during engine construction
# don't mask the test-collection step.
@pytest.fixture
def engine_module():
    from src.core import engine  # noqa: WPS433 — deliberate late import
    return engine


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    wf_dir = tmp_path / "workflows"
    (wf_dir / "templates" / "chroma").mkdir(parents=True)
    shutil.copy(
        _CANONICAL_TEMPLATE,
        wf_dir / "templates" / "chroma" / "chroma_done_properly.json",
    )
    return wf_dir


def _fake_ctx(**overrides) -> SimpleNamespace:
    """Minimal ``GenerationContext`` substitute — preflight reads only a
    handful of fields, so a namespace is enough to exercise it without
    building a real context end-to-end."""
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


def _engine_with_template(
    engine_module, workflow_dir: Path, template_override: str | None,
    monkeypatch,
):
    """Build a PipelineEngine bypassing the DB / registry checks we
    don't care about here, and stubbed so OllamaClient.is_available
    returns True by default (individual tests can override)."""
    # Neutralize model_override validation — the ctor won't hit the
    # registry because we pass None. Ollama stays available by default.
    monkeypatch.setattr(
        "src.agents.llm_client.OllamaClient.is_available",
        lambda self: True,
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
    # Point the checkpoint-folder resolution somewhere that doesn't
    # exist — if the external path short-circuits correctly, we never
    # look here; if it doesn't, the test surfaces a clear "checkpoint
    # not found" error from the system-path checks.
    engine.comfy_output_dir = workflow_dir.parent / "nonexistent_comfy"
    return engine


# ---------- ctor acceptance ---------------------------------------------


def test_pipeline_engine_accepts_template_override(
    engine_module, workflow_dir: Path, monkeypatch
):
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/chroma_done_properly.json",
        monkeypatch,
    )
    assert engine.template_override == (
        "templates/chroma/chroma_done_properly.json"
    )


def test_pipeline_engine_default_template_override_is_none(
    engine_module, workflow_dir: Path, monkeypatch
):
    engine = _engine_with_template(
        engine_module, workflow_dir, None, monkeypatch,
    )
    assert engine.template_override is None


# ---------- preflight: external-template path ---------------------------


def test_preflight_passes_on_valid_external_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/chroma_done_properly.json",
        monkeypatch,
    )
    engine._preflight(_fake_ctx())  # must not raise


def test_preflight_fails_on_missing_external_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/does_not_exist.json",
        monkeypatch,
    )
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight(_fake_ctx())
    assert "not found" in str(exc.value).lower()
    assert "--template" in str(exc.value)


def test_preflight_fails_on_malformed_external_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    (workflow_dir / "templates" / "chroma" / "broken.json").write_text(
        "{ this is not json"
    )
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/broken.json",
        monkeypatch,
    )
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight(_fake_ctx())
    assert "not valid json" in str(exc.value).lower()


def test_preflight_fails_on_missing_positive_prompt(
    engine_module, workflow_dir: Path, monkeypatch
):
    # Mutate the canonical template.
    with open(_CANONICAL_TEMPLATE) as f:
        data = json.load(f)
    data.pop("positive_prompt")
    dst = workflow_dir / "templates" / "chroma" / "mutant.json"
    dst.write_text(json.dumps(data))

    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/mutant.json",
        monkeypatch,
    )
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight(_fake_ctx())
    assert "positive_prompt" in str(exc.value)


def test_preflight_fails_on_missing_input_field(
    engine_module, workflow_dir: Path, monkeypatch
):
    with open(_CANONICAL_TEMPLATE) as f:
        data = json.load(f)
    del data["ksampler"]["inputs"]["seed"]
    dst = workflow_dir / "templates" / "chroma" / "no_seed.json"
    dst.write_text(json.dumps(data))

    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/no_seed.json",
        monkeypatch,
    )
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight(_fake_ctx())
    assert "ksampler.inputs.seed" in str(exc.value)


# ---------- preflight: skipped system-path checks -----------------------


def test_preflight_skips_checkpoint_file_check_under_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    """Checkpoint-file existence is system-path only. Under --template,
    the template's own UNETLoader / CheckpointLoader carries the model
    file, which the user has manually verified in ComfyUI UI — the
    pipeline can't and shouldn't re-validate it."""
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/chroma_done_properly.json",
        monkeypatch,
    )
    # engine.comfy_output_dir points at a nonexistent dir; if the
    # system-path checkpoint check runs, it'll fail here.
    engine._preflight(_fake_ctx())  # must not raise


def test_preflight_skips_base_json_check_under_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    """{family}/base.json is irrelevant under --template — no file
    under config/comfyui_workflows/chroma/ is read for the render."""
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/chroma_done_properly.json",
        monkeypatch,
    )
    # workflow_dir has NO chroma/base.json under it — only the
    # external template under templates/chroma/. Preflight must not
    # reach for the system path.
    assert not (workflow_dir / "chroma" / "base.json").exists()
    engine._preflight(_fake_ctx())  # must not raise


def test_preflight_skips_ipadapter_capability_check_under_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    """A character with a reference_image_path + a model without
    supports_ipadapter normally fails preflight. Under --template,
    IPAdapter is forced off — so the capability mismatch doesn't
    matter."""
    ctx = _fake_ctx(
        character={
            "id": "char_with_ref",
            "reference_image_path": "/tmp/some_ref.png",
        },
    )
    # model_config.supports_ipadapter is False in the fake ctx.
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/chroma_done_properly.json",
        monkeypatch,
    )
    engine._preflight(ctx)  # must not raise


# ---------- preflight: Ollama still required ----------------------------


def test_preflight_still_requires_ollama_under_template(
    engine_module, workflow_dir: Path, monkeypatch
):
    """Phase A LLM planning runs regardless of which workflow renders
    the result, so Ollama must still be up."""
    engine = _engine_with_template(
        engine_module, workflow_dir,
        "templates/chroma/chroma_done_properly.json",
        monkeypatch,
    )
    # Now flip Ollama off.
    monkeypatch.setattr(
        "src.agents.llm_client.OllamaClient.is_available",
        lambda self: False,
    )
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight(_fake_ctx())
    assert "Ollama" in str(exc.value) or "ollama" in str(exc.value)


# ---------- preflight: no --template preserves existing behavior --------


def test_preflight_without_template_still_checks_base_json(
    engine_module, workflow_dir: Path, monkeypatch
):
    """Regression: when --template is NOT set, the system-path checks
    must still fire exactly as before. This test exists to catch a
    future refactor that accidentally inverts the branch."""
    engine = _engine_with_template(
        engine_module, workflow_dir, None, monkeypatch,
    )
    # workflow_dir has no chroma/base.json — system path should fail.
    with pytest.raises(engine_module.PreflightError) as exc:
        engine._preflight(_fake_ctx())
    msg = str(exc.value)
    assert "base.json" in msg or "Workflow template" in msg

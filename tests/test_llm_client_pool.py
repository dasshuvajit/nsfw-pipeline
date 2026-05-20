"""Round-14 (2026-05-21) tests for :class:`LLMClientPool`.

The pool is the backend-aware facade the engine constructs in place of
the legacy ``OllamaClient`` instance. Public-API parity (same shape as
``OllamaClient``) is the contract — agents call
``pool.generate_json(model=tag)`` and the pool dispatches to the right
sub-client based on the registry's per-entry ``backend`` field.

These tests use mock sub-clients so they're hermetic (no live Ollama
or LM Studio process needed). The end-to-end "did the LM Studio
generate path actually wire" check lives in the smoke run, not here.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.llm_client import LLMClientPool


def _write_registry(tmp_path: Path) -> Path:
    yaml = tmp_path / "registry.yaml"
    yaml.write_text(textwrap.dedent("""
        llms:
          ollama_one:
            ollama_id: "tag_ollama"
            display_name: "Ollama"
          lm_studio_one:
            backend: lm_studio
            lm_studio_id: "tag_lms"
            display_name: "LM Studio"
        default_llm: ollama_one
    """))
    return yaml


@pytest.fixture
def pool(tmp_path):
    """Pool wired with mock sub-clients + a tiny test registry."""
    from src.memory.llm_registry import LLMRegistryLoader
    registry = LLMRegistryLoader(_write_registry(tmp_path))
    return LLMClientPool(
        ollama_client=MagicMock(name="OllamaClient"),
        lm_studio_client=MagicMock(name="LMStudioClient"),
        registry=registry,
    )


# ── Dispatch logic ─────────────────────────────────────────────────


def test_generate_json_dispatches_ollama_tag_to_ollama_client(pool):
    pool.ollama.generate_json.return_value = {"ok": True}
    out = pool.generate_json(
        "system", "user", model="tag_ollama", temperature=0.5,
    )
    assert out == {"ok": True}
    pool.ollama.generate_json.assert_called_once()
    pool.lm_studio.generate_json.assert_not_called()


def test_generate_json_dispatches_lm_studio_tag_to_lm_studio_client(pool):
    pool.lm_studio.generate_json.return_value = {"ok": True}
    out = pool.generate_json(
        "system", "user", model="tag_lms", schema=None,
    )
    assert out == {"ok": True}
    pool.lm_studio.generate_json.assert_called_once()
    pool.ollama.generate_json.assert_not_called()


def test_generate_forwards_kwargs(pool):
    pool.ollama.generate.return_value = "text"
    pool.generate(
        "sys", "usr", model="tag_ollama",
        temperature=0.3, num_predict=512,
    )
    call = pool.ollama.generate.call_args
    assert call.kwargs["temperature"] == 0.3
    assert call.kwargs["num_predict"] == 512
    assert call.kwargs["model"] == "tag_ollama"


def test_unload_model_routes_per_backend(pool):
    pool.unload_model("tag_ollama")
    pool.ollama.unload_model.assert_called_once_with("tag_ollama")
    pool.lm_studio.unload_model.assert_not_called()

    pool.unload_model("tag_lms")
    pool.lm_studio.unload_model.assert_called_once_with("tag_lms")


def test_unload_all_cascades_to_every_backend(pool):
    pool.unload_all()
    pool.ollama.unload_all.assert_called_once()
    pool.lm_studio.unload_all.assert_called_once()


def test_unknown_tag_defaults_to_ollama_backend(pool):
    """Legacy / un-registered tags fall back to Ollama (pool's
    defensive default)."""
    pool.ollama.generate_json.return_value = {}
    pool.generate_json("sys", "usr", model="some-unregistered-tag")
    pool.ollama.generate_json.assert_called_once()
    pool.lm_studio.generate_json.assert_not_called()


def test_is_available_true_when_any_backend_reachable(pool):
    pool.ollama.is_available.return_value = False
    pool.lm_studio.is_available.return_value = True
    assert pool.is_available() is True


def test_is_available_false_when_both_offline(pool):
    pool.ollama.is_available.return_value = False
    pool.lm_studio.is_available.return_value = False
    assert pool.is_available() is False


def test_sub_client_accessors_expose_underlying_clients(pool):
    """``pool.ollama`` / ``pool.lm_studio`` expose the underlying clients
    for diagnostics. The engine's preflight error reads
    ``pool.ollama.base_url`` to message the operator."""
    assert pool.ollama is pool._ollama
    assert pool.lm_studio is pool._lm_studio

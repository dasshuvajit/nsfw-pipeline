"""Shared pytest fixtures.

The LLM registry fixtures (Phase 1) supply tests with a known-shape
``llm_models.yaml`` under tmpdir. Tests of the *real* on-disk registry
(``config/llm_models.yaml``) instantiate ``LLMRegistryLoader()`` with
no args; tests of registry mechanics use ``llm_registry_yaml``.

The autouse cache reset prevents the LRU-cached ``_load_registry`` from
serving stale entries across tests that build different fixture files
on the same path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


_MINIMAL_REGISTRY = textwrap.dedent("""\
    llms:
      test_llm_a:
        ollama_id: "test/llm-a:latest"
        display_name: "Test LLM A"
        description: "Primary test LLM."
        quant: "Q4_K_M"
        size_gb: 4
        context_tokens: 8192
        refusal_rate: 5.0
        strengths: ["test"]
        families_recommended: ["all"]
        active: true

      test_llm_b:
        ollama_id: "test/llm-b:latest"
        display_name: "Test LLM B"
        description: "Secondary test LLM."
        quant: "Q5_K_M"
        size_gb: 5
        context_tokens: 16384
        refusal_rate: 2.0
        strengths: ["test"]
        families_recommended: ["flux", "flux2"]
        active: true

      test_llm_inactive:
        ollama_id: "test/llm-inactive:latest"
        display_name: "Test LLM Inactive"
        description: "Inactive test LLM."
        quant: "Q4_K_M"
        size_gb: 4
        context_tokens: 8192
        refusal_rate: 5.0
        active: false

    default_llm: test_llm_a
""")


@pytest.fixture
def llm_registry_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid ``llm_models.yaml`` to tmp and return its path.

    Two active LLMs (``test_llm_a`` is default; ``test_llm_b`` is reserved
    for routing tests in Phase 3) plus one inactive LLM (``test_llm_inactive``)
    so registry tests can exercise the active-filter and require-active code
    paths without fabricating their own YAML.
    """
    path = tmp_path / "llm_models.yaml"
    path.write_text(_MINIMAL_REGISTRY)
    return path


@pytest.fixture(autouse=True)
def _reset_llm_registry_cache():
    """Clear the LRU cache between tests so each path loads fresh.

    ``LLMRegistryLoader._load_registry`` is module-level LRU-cached for
    production efficiency; without this autouse reset, two tests using
    different fixture YAMLs at the same tmp path could see each other's
    state.
    """
    try:
        from src.memory.llm_registry import _load_registry
        _load_registry.cache_clear()
    except ImportError:  # pragma: no cover — module present in the repo
        pass
    yield
    try:
        from src.memory.llm_registry import _load_registry
        _load_registry.cache_clear()
    except ImportError:  # pragma: no cover
        pass

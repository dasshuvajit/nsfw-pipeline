"""Tests for ``src.memory.llm_registry``.

Covers:
  * Basic load + lookup via ``LLMRegistryLoader``.
  * Construction-time validation (default_llm exists + active).
  * Entry parsing (required-field enforcement; optional-field defaults).
  * ``get_llm`` happy + sad paths (inactive, missing id, helpful error).
  * ``list_llms`` filters (active-only by default; include_inactive).
  * Real on-disk ``config/llm_models.yaml`` smoke (verifies the shipped
    file is well-formed and points to an active default).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.memory.llm_registry import (
    LLMNotFound,
    LLMRegistryEntry,
    LLMRegistryError,
    LLMRegistryLoader,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


# ── basic loading ────────────────────────────────────────────────────
class TestLoad:
    def test_load_minimal_valid(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        assert loader.default_llm_id == "test_llm_a"
        assert len(loader.list_llms()) == 2  # active only

    def test_get_default_llm(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        default = loader.get_default_llm()
        assert default.id == "test_llm_a"
        assert default.ollama_id == "test/llm-a:latest"
        assert default.active is True

    def test_yaml_missing_raises(self, tmp_path):
        with pytest.raises(LLMRegistryError, match="not found"):
            LLMRegistryLoader(tmp_path / "missing.yaml")

    def test_empty_llms_raises(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms: {}
            default_llm: foo
        """)
        with pytest.raises(LLMRegistryError, match="at least one entry"):
            LLMRegistryLoader(path)

    def test_missing_default_llm_key_raises(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              foo:
                ollama_id: "foo:latest"
                display_name: "Foo"
                active: true
        """)
        with pytest.raises(LLMRegistryError, match="default_llm"):
            LLMRegistryLoader(path)

    def test_blank_default_llm_rejected(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              foo:
                ollama_id: "foo:latest"
                display_name: "Foo"
                active: true
            default_llm: ""
        """)
        with pytest.raises(LLMRegistryError, match="default_llm"):
            LLMRegistryLoader(path)


# ── default_llm validation ────────────────────────────────────────────
class TestDefaultValidation:
    def test_default_pointing_to_missing_id_rejected(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              real_llm:
                ollama_id: "real:latest"
                display_name: "Real"
                active: true
            default_llm: not_a_real_id
        """)
        with pytest.raises(
            LLMRegistryError, match="default_llm 'not_a_real_id'"
        ):
            LLMRegistryLoader(path)

    def test_default_pointing_to_inactive_entry_rejected(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              dead_llm:
                ollama_id: "dead:latest"
                display_name: "Dead"
                active: false
            default_llm: dead_llm
        """)
        with pytest.raises(LLMRegistryError, match="active=false"):
            LLMRegistryLoader(path)

    def test_validation_error_lists_known_ids(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              foo:
                ollama_id: "foo:latest"
                display_name: "Foo"
                active: true
              bar:
                ollama_id: "bar:latest"
                display_name: "Bar"
                active: true
            default_llm: missing_typo
        """)
        with pytest.raises(LLMRegistryError) as exc_info:
            LLMRegistryLoader(path)
        msg = str(exc_info.value)
        assert "foo" in msg
        assert "bar" in msg


# ── get_llm ──────────────────────────────────────────────────────────
class TestGetLlm:
    def test_get_existing_active(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        entry = loader.get_llm("test_llm_b")
        assert entry.id == "test_llm_b"
        assert entry.ollama_id == "test/llm-b:latest"
        assert entry.families_recommended == ["flux", "flux2"]

    def test_get_inactive_default_raises(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        with pytest.raises(LLMNotFound, match="inactive"):
            loader.get_llm("test_llm_inactive")

    def test_get_inactive_with_require_active_false(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        entry = loader.get_llm("test_llm_inactive", require_active=False)
        assert entry.id == "test_llm_inactive"
        assert entry.active is False

    def test_get_unknown_raises_with_help(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        with pytest.raises(LLMNotFound) as exc_info:
            loader.get_llm("not_in_registry")
        msg = str(exc_info.value)
        assert "not_in_registry" in msg
        assert "Available LLMs" in msg
        assert "test_llm_a" in msg
        assert "default" in msg.lower()
        assert "ollama pull" in msg

    def test_helpful_error_includes_inactive_entries(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        with pytest.raises(LLMNotFound) as exc_info:
            loader.get_llm("typo")
        msg = str(exc_info.value)
        assert "test_llm_inactive" in msg
        assert "(inactive)" in msg


# ── has_llm ──────────────────────────────────────────────────────────
class TestHasLlm:
    def test_has_active_returns_true(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        assert loader.has_llm("test_llm_a") is True

    def test_has_unknown_returns_false(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        assert loader.has_llm("not_a_thing") is False

    def test_has_inactive_default_includes(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        # Default include_inactive=True — counts inactive as present.
        assert loader.has_llm("test_llm_inactive") is True

    def test_has_inactive_with_active_only(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        assert (
            loader.has_llm("test_llm_inactive", include_inactive=False)
            is False
        )


# ── list_llms ────────────────────────────────────────────────────────
class TestListLlms:
    def test_list_default_active_only(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        ids = [e.id for e in loader.list_llms()]
        assert ids == ["test_llm_a", "test_llm_b"]

    def test_list_include_inactive(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        ids = [e.id for e in loader.list_llms(include_inactive=True)]
        assert ids == ["test_llm_a", "test_llm_b", "test_llm_inactive"]

    def test_list_sorted_alphabetically(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        entries = loader.list_llms(include_inactive=True)
        assert [e.id for e in entries] == sorted(e.id for e in entries)


# ── entry parsing ─────────────────────────────────────────────────────
class TestEntryParsing:
    def test_required_fields_missing(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              missing_ollama:
                display_name: "Missing"
                active: true
            default_llm: missing_ollama
        """)
        with pytest.raises(LLMRegistryError, match="missing required keys"):
            LLMRegistryLoader(path)

    def test_optional_fields_default_to_none(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              minimal:
                ollama_id: "minimal:latest"
                display_name: "Minimal"
                active: true
            default_llm: minimal
        """)
        loader = LLMRegistryLoader(path)
        entry = loader.get_llm("minimal")
        assert entry.quant is None
        assert entry.size_gb is None
        assert entry.context_tokens is None
        assert entry.refusal_rate is None
        assert entry.description == ""
        assert entry.strengths == []
        assert entry.families_recommended == []

    def test_active_defaults_to_true(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              implicit_active:
                ollama_id: "implicit:latest"
                display_name: "Implicit"
            default_llm: implicit_active
        """)
        loader = LLMRegistryLoader(path)
        entry = loader.get_llm("implicit_active")
        assert entry.active is True

    def test_full_entry_typed_correctly(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        entry = loader.get_llm("test_llm_a")
        assert isinstance(entry, LLMRegistryEntry)
        assert entry.size_gb == 4.0
        assert entry.refusal_rate == 5.0
        assert entry.context_tokens == 8192
        assert "test" in entry.strengths

    def test_frozen_dataclass(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        entry = loader.get_llm("test_llm_a")
        with pytest.raises(Exception):
            # Frozen dataclass — assignment raises FrozenInstanceError.
            entry.id = "mutated"  # type: ignore[misc]


# ── real registry sanity ──────────────────────────────────────────────
class TestRealRegistry:
    """Verify the on-disk ``config/llm_models.yaml`` is well-formed.

    These tests fail fast if the registry file gets corrupted in source
    control — equivalent to a CI lint for the YAML.
    """

    def test_real_registry_loads(self):
        loader = LLMRegistryLoader()
        assert loader.default_llm_id
        assert len(loader.list_llms()) >= 1

    def test_real_default_is_active(self):
        loader = LLMRegistryLoader()
        default = loader.get_default_llm()
        assert default.active is True

    def test_real_default_is_cydonia_heretic(self):
        """Cydonia 24B v4.3 Heretic Vision is the project's chosen default
        (2026-05-18 LLM swap)."""
        loader = LLMRegistryLoader()
        assert loader.default_llm_id == "cydonia_heretic_24b"

    def test_real_installed_llms_active(self):
        """The installed LLMs (Cydonia Heretic Vision 24B + Qwen3
        30B-A3B Abliterated + Magnum v4 27B as of 2026-05-19) all
        register as active. Cydonia is the default + chosen quality
        target; Qwen3 30B-A3B is the fallback (MoE, different lineage
        — Qwen vs Mistral); Magnum v4 27B is a dense Gemma 2-based
        creative-prose variant added for an A/B comparison
        (registered 2026-05-19, replacing qwen36_abliterated_27b)."""
        loader = LLMRegistryLoader()
        ids = {e.id for e in loader.list_llms()}
        assert "cydonia_heretic_24b" in ids
        assert "qwen3_abliterated_30b" in ids
        assert "magnum_v4_27b" in ids


# ── Round-14 (2026-05-21): multi-backend entries ───────────────────


class TestBackendDispatch:
    def test_backend_defaults_to_ollama_when_unspecified(
        self, llm_registry_yaml,
    ):
        """Legacy entries without a `backend:` key default to ollama —
        keeps every pre-existing entry working under the new schema."""
        loader = LLMRegistryLoader(llm_registry_yaml)
        entry = loader.get_default_llm()
        assert entry.backend == "ollama"

    def test_lm_studio_entry_parses(self, tmp_path):
        yaml = _write(tmp_path / "lms.yaml", """
            llms:
              cydonia_lms:
                backend: lm_studio
                lm_studio_id: "thedrummer_cydonia-24b-v4.3"
                display_name: "Cydonia LM Studio"
            default_llm: cydonia_lms
        """)
        loader = LLMRegistryLoader(yaml)
        entry = loader.get_default_llm()
        assert entry.backend == "lm_studio"
        assert entry.lm_studio_id == "thedrummer_cydonia-24b-v4.3"
        # ollama_id is empty for LM Studio entries.
        assert entry.ollama_id == ""

    def test_model_tag_property_returns_per_backend_id(self, tmp_path):
        yaml = _write(tmp_path / "mix.yaml", """
            llms:
              ollama_one:
                ollama_id: "tag_ollama"
                display_name: "Ollama"
              lm_studio_one:
                backend: lm_studio
                lm_studio_id: "tag_lms"
                display_name: "LM Studio"
            default_llm: ollama_one
        """)
        loader = LLMRegistryLoader(yaml)
        assert loader.get_llm("ollama_one").model_tag == "tag_ollama"
        assert loader.get_llm("lm_studio_one").model_tag == "tag_lms"

    def test_lm_studio_entry_missing_lm_studio_id_raises(self, tmp_path):
        yaml = _write(tmp_path / "bad.yaml", """
            llms:
              broken:
                backend: lm_studio
                display_name: "no id"
            default_llm: broken
        """)
        with pytest.raises(LLMRegistryError, match="lm_studio_id"):
            LLMRegistryLoader(yaml)

    def test_unknown_backend_raises(self, tmp_path):
        yaml = _write(tmp_path / "bad.yaml", """
            llms:
              broken:
                backend: vllm
                ollama_id: "x"
                display_name: "vLLM stub"
            default_llm: broken
        """)
        with pytest.raises(LLMRegistryError, match="unknown backend"):
            LLMRegistryLoader(yaml)

    def test_backend_for_tag_round_trips(self, tmp_path):
        yaml = _write(tmp_path / "mix.yaml", """
            llms:
              ollama_one:
                ollama_id: "tag_ollama"
                display_name: "Ollama"
              lm_studio_one:
                backend: lm_studio
                lm_studio_id: "tag_lms"
                display_name: "LM Studio"
            default_llm: ollama_one
        """)
        loader = LLMRegistryLoader(yaml)
        assert loader.backend_for_tag("tag_ollama") == "ollama"
        assert loader.backend_for_tag("tag_lms") == "lm_studio"

    def test_backend_for_unknown_tag_defaults_to_ollama(self, tmp_path):
        """Pool's defensive default — unknown tag routes to Ollama so
        legacy callers passing arbitrary Ollama tags that aren't yet
        registered keep working."""
        yaml = _write(tmp_path / "min.yaml", """
            llms:
              one:
                ollama_id: "tag1"
                display_name: "A"
            default_llm: one
        """)
        loader = LLMRegistryLoader(yaml)
        assert loader.backend_for_tag("totally_unknown") == "ollama"

    def test_real_registry_includes_lm_studio_cydonia(self):
        """Round-14 — the shipped llm_models.yaml ships the
        cydonia_v43_lm_studio entry for the A/B against the heretic
        Ollama variant."""
        loader = LLMRegistryLoader()
        entry = loader.get_llm("cydonia_v43_lm_studio")
        assert entry.backend == "lm_studio"
        assert entry.lm_studio_id == "thedrummer_cydonia-24b-v4.3"

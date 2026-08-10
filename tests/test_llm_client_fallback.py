"""Tests for Q11 — registry-aware fallback model + second-chance retry.

Two layers:
  1. Registry — ``llm_models.yaml::fallback_llm`` (optional; mirrors
     ``default_llm`` when absent), validated like ``default_llm`` at
     startup.
  2. Client — ``OllamaClient.generate_json(fallback_model=...)`` retries
     ONCE against the fallback when the primary model raises
     ``OllamaJSONParseError``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from src.agents.llm_client import OllamaClient, OllamaJSONParseError
from src.memory.llm_registry import (
    LLMRegistryError,
    LLMRegistryLoader,
)


class _Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


# ── Registry-side: fallback_llm parsing + validation ────────────────


class TestRegistryFallbackLlm:
    def test_omitted_falls_back_to_default(self, llm_registry_yaml):
        loader = LLMRegistryLoader(llm_registry_yaml)
        # The conftest fixture has no fallback_llm key; should mirror
        # default_llm.
        assert loader.fallback_llm_id == loader.default_llm_id

    def test_explicit_fallback_llm(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              alpha:
                ollama_id: "alpha:latest"
                display_name: "Alpha"
                active: true
              beta:
                ollama_id: "beta:latest"
                display_name: "Beta"
                active: true
            default_llm: alpha
            fallback_llm: beta
        """)
        loader = LLMRegistryLoader(path)
        assert loader.default_llm_id == "alpha"
        assert loader.fallback_llm_id == "beta"
        # get_fallback_llm returns the entry
        entry = loader.get_fallback_llm()
        assert entry.id == "beta"
        assert entry.ollama_id == "beta:latest"

    def test_fallback_pointing_to_missing_id_rejected(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              alpha:
                ollama_id: "alpha:latest"
                display_name: "Alpha"
                active: true
            default_llm: alpha
            fallback_llm: not_a_real_id
        """)
        with pytest.raises(LLMRegistryError, match="fallback_llm"):
            LLMRegistryLoader(path)

    def test_fallback_pointing_to_inactive_rejected(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              alpha:
                ollama_id: "alpha:latest"
                display_name: "Alpha"
                active: true
              dead:
                ollama_id: "dead:latest"
                display_name: "Dead"
                active: false
            default_llm: alpha
            fallback_llm: dead
        """)
        with pytest.raises(LLMRegistryError, match="active=false"):
            LLMRegistryLoader(path)

    def test_blank_fallback_rejected(self, tmp_path):
        path = _write(tmp_path / "x.yaml", """
            llms:
              alpha:
                ollama_id: "alpha:latest"
                display_name: "Alpha"
                active: true
            default_llm: alpha
            fallback_llm: ""
        """)
        with pytest.raises(LLMRegistryError, match="non-empty string"):
            LLMRegistryLoader(path)


# ── Client-side: second-chance retry path ───────────────────────────


class TestGenerateJsonFallback:
    def test_fallback_fires_on_primary_failure(self):
        """Primary model raises OllamaJSONParseError once; fallback
        succeeds. Result returned."""
        call_count = {"n": 0}

        def fake_chat(
            system_prompt, user_prompt, assistant_prefill,
            model, temperature, num_predict, format_schema,
        ):
            call_count["n"] += 1
            call_count[f"call_{call_count['n']}_model"] = model
            if model == "primary:latest":
                # Return un-parseable garbage (fails Pydantic validation).
                return "this is not JSON at all"
            # Fallback model returns valid JSON.
            return '{"name": "x", "age": 1}'

        with patch.object(OllamaClient, "_generate_chat", side_effect=fake_chat):
            result = OllamaClient().generate_json(
                "sys", "user",
                schema=_Person,
                model="primary:latest",
                fallback_model="fallback:latest",
            )
        assert result == {"name": "x", "age": 1}
        assert call_count["n"] == 2
        assert call_count["call_1_model"] == "primary:latest"
        assert call_count["call_2_model"] == "fallback:latest"

    def test_no_fallback_when_primary_succeeds(self):
        """Primary succeeds → fallback never invoked."""
        call_count = {"n": 0}

        def fake_chat(*args, **kwargs):
            call_count["n"] += 1
            return '{"name": "x", "age": 1}'

        with patch.object(OllamaClient, "_generate_chat", side_effect=fake_chat):
            OllamaClient().generate_json(
                "sys", "user",
                schema=_Person,
                model="primary:latest",
                fallback_model="fallback:latest",
            )
        assert call_count["n"] == 1

    def test_no_fallback_param_defaults_to_no_retry(self):
        """Without fallback_model, primary failure raises immediately."""
        with patch.object(
            OllamaClient, "_generate_chat",
            return_value="not json at all",
        ):
            with pytest.raises(OllamaJSONParseError):
                OllamaClient().generate_json(
                    "sys", "user",
                    schema=_Person,
                    model="primary:latest",
                )

    def test_same_tag_fallback_skips_retry(self):
        """If fallback_model == model, the retry is a no-op compared
        to the in-agent retry-with-nudge already wrapping this. We
        skip it to avoid wasting an extra LLM call."""
        call_count = {"n": 0}

        def fake_chat(*args, **kwargs):
            call_count["n"] += 1
            return "still not json"

        with patch.object(OllamaClient, "_generate_chat", side_effect=fake_chat):
            with pytest.raises(OllamaJSONParseError):
                OllamaClient().generate_json(
                    "sys", "user",
                    schema=_Person,
                    model="same:latest",
                    fallback_model="same:latest",
                )
        # Single call — no retry against the same tag.
        assert call_count["n"] == 1

    def test_fallback_failure_raises_primary_error(self):
        """When BOTH primary AND fallback fail, the primary error is
        what the caller sees (so they can debug the original failure)."""

        def always_fail(*args, **kwargs):
            return "garbage"

        with patch.object(OllamaClient, "_generate_chat", side_effect=always_fail):
            with pytest.raises(OllamaJSONParseError) as exc_info:
                OllamaClient().generate_json(
                    "sys", "user",
                    schema=_Person,
                    model="primary:latest",
                    fallback_model="fallback:latest",
                )
        # The error message refers to the primary model's failure
        # context (the test verifies we're not silently swallowing).
        assert "schema validation" in str(exc_info.value).lower() or \
            "failed to parse" in str(exc_info.value).lower()

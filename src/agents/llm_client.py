# WARNING: Always call llm_client.unload_model() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48GB unified memory.
"""Ollama HTTP client — generate text and structured JSON from a local LLM.

Wraps the Ollama ``/api/generate`` endpoint with:

  * ``generate()``     — free-form text completion
  * ``generate_json()`` — text completion + markdown-fence stripping + JSON parse
  * ``unload_model()``  — sends ``keep_alive: 0`` to evict the model from VRAM,
    then sleeps 2 s so macOS unified memory actually releases before ComfyUI
    loads its own models

Configuration is read from ``config/pipeline.yaml`` under the ``llm:`` key.
The caller can override ``model`` and ``base_url`` at construction time.

See ARCHITECTURE.md Section 13 for the full spec.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import yaml

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE_YAML = _PROJECT_ROOT / "config" / "pipeline.yaml"


class OllamaError(Exception):
    """Any error talking to the Ollama HTTP API."""


class OllamaConnectionError(OllamaError):
    """Ollama server is not reachable."""


class OllamaGenerateError(OllamaError):
    """Non-200 response from /api/generate."""


class OllamaJSONParseError(OllamaError):
    """LLM response could not be parsed as valid JSON."""


def _load_llm_config() -> dict:
    """Read the ``llm:`` block from pipeline.yaml."""
    if _PIPELINE_YAML.exists():
        with open(_PIPELINE_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("llm", {})
    return {}


def resolve_default_ollama_id() -> str:
    """Resolve the registry's ``default_llm`` to its backend-specific
    model identifier.

    Round-14 (2026-05-21) — historically returned ``ollama_id``;
    now returns ``model_tag`` (the same string for Ollama-backed
    defaults, the LM Studio identifier when the default is LM-Studio
    backed). Kept under the legacy name for back-compat with the
    handful of agent fallback call sites.
    """
    from src.memory.llm_registry import LLMRegistryLoader

    return LLMRegistryLoader().get_default_llm().model_tag


class OllamaClient:
    """Stateless HTTP wrapper around the Ollama ``/api/generate`` endpoint.

    The client is model-agnostic — every call to :meth:`generate` /
    :meth:`generate_json` / :meth:`unload_model` takes the Ollama tag
    explicitly. The :class:`src.agents.llm_router.LLMRouter` resolves
    role/family → tag at the engine boundary; this class is a pure
    transport.

    Parameters
    ----------
    base_url : str | None
        Override the Ollama server URL.  Falls back to
        ``pipeline.yaml → llm.base_url`` then ``http://localhost:11434``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        cfg = _load_llm_config()
        self.base_url = (base_url or cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        # Per-request HTTP timeout for /api/generate + /api/chat calls.
        # Default 900s (15 min) accommodates 70B-class models on M4 Pro
        # under unified-memory pressure where individual generations
        # can legitimately take 5-10 min. Smaller models (24B and
        # below) finish well under this. Override via
        # ``pipeline.yaml::llm.request_timeout_seconds``.
        self.request_timeout = int(cfg.get("request_timeout_seconds", 900))
        # Track every Ollama tag we've sent a request for so
        # ``unload_all()`` can free each one at end-of-cycle. Populated
        # by ``generate()`` whenever a call fires (the model is loaded
        # into Ollama before generation regardless of whether we got a
        # 200 back).
        self.loaded_models: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        format_schema: dict | None = None,
    ) -> str:
        """Send a prompt to Ollama and return the raw text response.

        ``model`` — Ollama tag for this call (REQUIRED). The router
        resolves role/family → tag at the engine boundary; this client
        is a pure transport.

        ``format_schema`` (Phase 4b) — when supplied, Ollama 0.5+ uses
        constrained-decoding via llama.cpp grammars to ensure the
        response matches the JSON schema. Eliminates most "malformed
        JSON" retries at the cost of a tiny per-call overhead.

        Raises
        ------
        OllamaConnectionError
            If the server is unreachable.
        OllamaGenerateError
            If the server returns a non-200 status.
        """
        # Track every model the client has loaded so unload_all() can
        # free each one at end-of-cycle.
        self.loaded_models.add(model)
        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if format_schema is not None:
            payload["format"] = format_schema
        logger.debug("Ollama request: model=%s temp=%.1f num_predict=%d format=%s",
                     model, temperature, num_predict,
                     "schema" if format_schema is not None else "free")

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.request_timeout,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is `ollama serve` running?\n{exc}"
            ) from exc

        if resp.status_code != 200:
            raise OllamaGenerateError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        text = data.get("response", "")
        logger.debug("Ollama response: %d chars, eval_count=%s", len(text), data.get("eval_count"))
        return text

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float = 0.6,
        num_predict: int = 4096,
        schema: "type[BaseModel] | None" = None,
        prefill: str | None = None,
        fallback_model: str | None = None,
    ) -> dict | list:
        """Generate text, strip markdown fences, parse as JSON.

        Uses a slightly lower temperature than ``generate()`` for more
        deterministic structured output.

        When ``schema`` is provided, three layers of safety apply:

        1. **Assistant-prefill** (Q6): when ``schema`` is set the call
           uses Ollama's ``/api/chat`` endpoint with a partial assistant
           message ("Sure, here's the JSON: "). The model continues
           from that opening — defeats refusals while format:schema
           commits to JSON shape. Override with ``prefill="..."`` (any
           string) or ``prefill=""`` to skip and use the legacy
           ``/api/generate`` path.
        2. **Constrained decoding** (Phase 4b): the schema's JSON-schema
           dict is passed to Ollama 0.5+ via the ``format:`` field —
           llama.cpp's grammar enforces structural validity at decode
           time, eliminating most "malformed JSON" retries.
        3. **Pydantic post-validation** (defence in depth): the parsed
           text is still validated through the Pydantic model so
           type-narrowing constraints (``min_length``, ``Literal``,
           ``model_validator`` decorators) catch anything the JSON
           grammar can't enforce.

        ``fallback_model`` (Q11) — when set, an
        :class:`OllamaJSONParseError` from the primary ``model`` triggers
        ONE second-chance retry against the fallback Ollama tag. The
        most common cause for the primary failing is a refusal-shaped
        output that even constrained-decoding + prefill couldn't
        prevent; switching to the configured ``fallback_llm`` (a
        different-lineage, lower-refusal-floor LLM) recovers most of
        those cases. ``None`` skips the fallback path entirely (legacy
        single-model behaviour). Caller resolves the tag via
        ``router.fallback()`` / ``router.resolve_fallback().ollama_id``.

        The returned value is always a plain ``dict`` / ``list`` —
        schema validation does not change the call signature.

        Raises
        ------
        OllamaJSONParseError
            If the response cannot be parsed as JSON after fence
            stripping, or if Pydantic schema validation fails.
        """
        try:
            return self._generate_json_once(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                schema=schema,
                prefill=prefill,
            )
        except OllamaJSONParseError as primary_exc:
            # Q11 — second-chance retry against the fallback model
            # when one is supplied AND it's a different tag. Same-tag
            # fallback would just retry against the same model, which
            # is a no-op compared to the in-agent retry-with-nudge
            # loop already wrapping this call.
            if fallback_model and fallback_model != model:
                logger.warning(
                    "Primary model %r failed JSON parse — retrying once "
                    "with fallback %r",
                    model, fallback_model,
                )
                return self._generate_json_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=fallback_model,
                    temperature=temperature,
                    num_predict=num_predict,
                    schema=schema,
                    prefill=prefill,
                )
            raise primary_exc

    def _generate_json_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        num_predict: int,
        schema: "type[BaseModel] | None",
        prefill: str | None,
    ) -> dict | list:
        """One attempt of generate_json — primary OR fallback model.

        Same logic as the public :meth:`generate_json` but without
        the fallback retry loop (caller orchestrates that).
        """
        # Phase 4b — pass the JSON schema to Ollama for constrained
        # decoding when a schema is supplied. Falls back to free-form
        # generation when no schema (test fixtures, ad-hoc calls).
        format_schema: dict | None = None
        if schema is not None:
            try:
                format_schema = schema.model_json_schema()
            except Exception:  # pragma: no cover — defensive
                format_schema = None

        # Q6 — auto-pick assistant prefill when a schema is provided
        # and the caller didn't override. The prefill is a "Sure, here's
        # the JSON: " preamble (NOT including the structural opener) —
        # the preamble defeats refusal training, and Ollama's
        # format:schema grammar already enforces the structural shape.
        effective_prefill: str | None
        if prefill is None and schema is not None:
            effective_prefill = "Sure, here's the JSON: "
        elif prefill == "":
            # Caller explicitly opted out of chat-path prefill.
            effective_prefill = None
        else:
            effective_prefill = prefill

        if effective_prefill is not None:
            raw = self._generate_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                assistant_prefill=effective_prefill,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                format_schema=format_schema,
            )
            full_text = effective_prefill + raw
            cleaned_chat = self._strip_fences(raw)
            cleaned = self._extract_json_payload(
                effective_prefill + cleaned_chat
            )
        else:
            raw = self.generate(
                system_prompt,
                user_prompt,
                temperature=temperature,
                num_predict=num_predict,
                format_schema=format_schema,
                model=model,
            )
            full_text = raw
            cleaned = self._strip_fences(raw)

        # Error messages include the FULL response text (not just the
        # extracted JSON) so an operator debugging a parse failure sees
        # exactly what the LLM emitted, including any preamble or
        # markdown the prefill flow stripped off.
        if schema is not None:
            from pydantic import ValidationError as _ValidationError

            try:
                validated = schema.model_validate_json(cleaned)
            except _ValidationError as exc:
                raise OllamaJSONParseError(
                    f"LLM response failed schema validation against "
                    f"{schema.__name__}.\nErrors:\n{exc}\n"
                    f"Cleaned text ({len(cleaned)} chars):\n{cleaned[:500]}\n"
                    f"Full response ({len(full_text)} chars):\n"
                    f"{full_text[:500]}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise OllamaJSONParseError(
                    f"Failed to parse LLM response as JSON.\n"
                    f"Cleaned text ({len(cleaned)} chars):\n"
                    f"{cleaned[:500]}\n"
                    f"Full response ({len(full_text)} chars):\n"
                    f"{full_text[:500]}\n"
                    f"Parse error: {exc}"
                ) from exc
            return validated.model_dump(mode="python")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OllamaJSONParseError(
                f"Failed to parse LLM response as JSON.\n"
                f"Cleaned text ({len(cleaned)} chars):\n"
                f"{cleaned[:500]}\n"
                f"Parse error: {exc}"
            ) from exc

    def unload_model(self, model: str) -> None:
        """Evict one model from VRAM/unified memory.

        ``model`` — Ollama tag to unload (REQUIRED).

        Sends ``keep_alive: 0`` to Ollama, then sleeps 2 seconds so
        macOS unified memory actually frees before ComfyUI loads its
        checkpoint + LoRAs + VAE.

        Safe to call even if no model is loaded — Ollama treats it as
        a no-op.
        """
        logger.info("Unloading LLM model %s from memory …", model)
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": 0,
                },
                timeout=30,
            )
        except requests.ConnectionError:
            logger.warning("Ollama not reachable during unload — may already be stopped.")

        time.sleep(2)  # Allow memory to free
        # Forget this model so unload_all doesn't double-unload it.
        self.loaded_models.discard(model)
        logger.info("LLM model unloaded. 2 s grace period complete.")

    def unload_all(self) -> None:
        """Evict every model that this client has ever generated against.

        In multi-LLM cycles the engine may load multiple Ollama tags
        (one per agent role) before the LLM phase ends. This iterates
        ``self.loaded_models`` and calls ``unload_model`` for each, with
        a 2-second grace per call (handled by ``unload_model``).

        After this method returns, ``self.loaded_models`` is empty.
        Safe to call repeatedly; the second call is a no-op.
        """
        if not self.loaded_models:
            logger.info("unload_all: no models tracked as loaded.")
            return
        # Iterate a snapshot — unload_model mutates self.loaded_models.
        for model in list(self.loaded_models):
            self.unload_model(model)
        logger.info("unload_all: all tracked models unloaded.")

    def is_available(self) -> bool:
        """Quick health check — can we reach the Ollama server?"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """Return model names currently available on the Ollama server."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.ConnectionError, KeyError):
            return []

    def _generate_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        assistant_prefill: str,
        model: str,
        temperature: float,
        num_predict: int,
        format_schema: dict | None,
    ) -> str:
        """Q6 — assistant-prefill via Ollama's ``/api/chat`` endpoint.

        Sends a 3-message conversation
        ``[system, user, assistant(prefill)]`` and returns the model's
        continuation. Ollama returns ONLY the model's added text — the
        prefill is NOT echoed — so the caller is responsible for
        concatenating ``assistant_prefill + return_value`` to form the
        complete output.

        ``format_schema`` flows through to Ollama's ``format:`` field
        for constrained decoding (same shape as ``/api/generate``).
        """
        # Track the model the client has loaded (parity with generate()).
        self.loaded_models.add(model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prefill},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if format_schema is not None:
            payload["format"] = format_schema
        logger.debug(
            "Ollama chat request: model=%s temp=%.1f num_predict=%d format=%s",
            model, temperature, num_predict,
            "schema" if format_schema is not None else "free",
        )

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.request_timeout,
            )
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is `ollama serve` running?\n{exc}"
            ) from exc

        if resp.status_code != 200:
            raise OllamaGenerateError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        # /api/chat returns the assistant continuation under message.content
        # (vs /api/generate's top-level "response" field).
        message = data.get("message") or {}
        text = message.get("content", "")
        logger.debug("Ollama chat response: %d chars", len(text))
        return text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        """Find the JSON payload inside a string that may have a preamble.

        After the prefill concat (``"Sure, here's the JSON: {..."``) we
        need to strip the preamble before parsing. Locates the first
        ``{`` or ``[`` and returns the substring from there.

        If neither character is found, returns the input unchanged so
        the JSON parser produces a sensible error message.
        """
        first_brace = text.find("{")
        first_bracket = text.find("[")
        # Pick whichever comes first (and is non-negative).
        candidates = [c for c in (first_brace, first_bracket) if c >= 0]
        if not candidates:
            return text
        start = min(candidates)
        return text[start:]

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences that LLMs love to wrap JSON in."""
        stripped = text.strip()
        # Handle ```json ... ``` or ``` ... ```
        if stripped.startswith("```"):
            # Remove opening fence (possibly with language tag)
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            # Remove closing fence
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()
                stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()


# ── Round-14 (2026-05-21): multi-backend dispatch ─────────────────────


class LLMClientPool:
    """Backend-aware facade over :class:`OllamaClient` and
    :class:`src.agents.lm_studio_client.LMStudioClient`.

    Same call shape as :class:`OllamaClient` so the engine and agents
    treat the pool as a drop-in replacement. Per-call dispatch:
    :meth:`generate_json` / :meth:`generate` / :meth:`unload_model` take
    a ``model`` argument (the backend-specific identifier — Ollama tag
    or LM Studio identifier). The pool looks up the owning backend via
    :meth:`LLMRegistryLoader.backend_for_tag` and forwards to the right
    sub-client.

    :meth:`unload_all` cascades to both sub-clients — guaranteeing every
    loaded model on every backend is freed before ComfyUI starts.

    Why a thin facade rather than refactoring every agent: agents call
    ``client.generate_json(..., model=resolved_tag)`` after the router
    has already resolved ``role → registry_entry → model_tag``. The
    pool just keeps that call shape working when ``model_tag`` happens
    to belong to a non-Ollama backend.
    """

    def __init__(
        self,
        *,
        ollama_client: "OllamaClient | None" = None,
        lm_studio_client: "object | None" = None,
        mlx_client: "object | None" = None,
        openai_compatible_client: "object | None" = None,
        registry: "object | None" = None,
    ) -> None:
        # Lazy-import to avoid a circular dependency: LLMRegistryLoader
        # is imported by callers of OllamaClient via the default-llm
        # fallback path, and we don't want to wire that loop tighter.
        if registry is None:
            from src.memory.llm_registry import LLMRegistryLoader
            registry = LLMRegistryLoader()
        if lm_studio_client is None:
            from src.agents.lm_studio_client import LMStudioClient
            lm_studio_client = LMStudioClient()
        if mlx_client is None:
            from src.agents.mlx_client import MlxClient
            mlx_client = MlxClient()
        if openai_compatible_client is None:
            from src.agents.openai_client import OpenAICompatibleClient
            openai_compatible_client = OpenAICompatibleClient()
        self._ollama = ollama_client or OllamaClient()
        self._lm_studio = lm_studio_client
        self._mlx = mlx_client
        self._openai_compatible = openai_compatible_client
        self._registry = registry

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _client_for(self, model: str):
        """Resolve a backend client for ``model`` (registry lookup)."""
        from src.memory.llm_registry import (
            BACKEND_LM_STUDIO, BACKEND_MLX, BACKEND_OPENAI_COMPATIBLE,
        )
        backend = self._registry.backend_for_tag(model)
        if backend == BACKEND_LM_STUDIO:
            return self._lm_studio
        if backend == BACKEND_MLX:
            return self._mlx
        if backend == BACKEND_OPENAI_COMPATIBLE:
            return self._openai_compatible
        return self._ollama

    # ------------------------------------------------------------------
    # Public API — forwards to the right backend client
    # ------------------------------------------------------------------

    def generate(self, *args, model: str, **kwargs):
        return self._client_for(model).generate(*args, model=model, **kwargs)

    def generate_json(self, *args, model: str, **kwargs):
        # Round-18 — for LM-Studio reasoning models, tell the client to
        # skip the response_format:json_schema grammar (which fights
        # the model's <think>...</think> chain-of-thought trace).
        # Pydantic post-validation still runs. Ollama path is
        # unaffected (no kwarg recognised there).
        client = self._client_for(model)
        if (
            hasattr(client, "_chat")
            and "skip_grammar_constraint" not in kwargs
        ):
            # LM Studio (json_schema grammar) AND the OpenAI-compatible backend
            # (json_object nudge) both fight a model's <think> trace — skip the
            # constraint for reasoning models on either; Pydantic still validates.
            from src.agents.lm_studio_client import LMStudioClient
            from src.agents.openai_client import OpenAICompatibleClient
            if isinstance(client, (LMStudioClient, OpenAICompatibleClient)):
                kwargs["skip_grammar_constraint"] = (
                    self._registry.is_reasoning_model(model)
                )
        return client.generate_json(*args, model=model, **kwargs)

    def unload_model(self, model: str) -> None:
        self._client_for(model).unload_model(model)

    def unload_all(self) -> None:
        """Cascade unload to every sub-client.

        Each sub-client tracks its own ``loaded_models`` set. Calling
        each one's ``unload_all`` is cheap when nothing's tracked
        (logged + returned). MLX's ``unload_all`` logs a reminder
        instead of actually unloading (mlx_lm.server has no HTTP
        unload protocol). After this method returns every backend's
        ``loaded_models`` is empty.
        """
        self._ollama.unload_all()
        self._lm_studio.unload_all()
        self._mlx.unload_all()
        self._openai_compatible.unload_all()   # no-op (remote API)

    def is_available(self) -> bool:
        """True iff at least one sub-backend is reachable.

        The pool can still serve calls that target the reachable
        backend; failures for the offline one surface at call time.
        """
        return (
            self._ollama.is_available()
            or self._lm_studio.is_available()
            or self._mlx.is_available()
            or self._openai_compatible.is_available()
        )

    # ------------------------------------------------------------------
    # Sub-client accessors (mainly for tests + diagnostics)
    # ------------------------------------------------------------------

    @property
    def ollama(self) -> "OllamaClient":
        return self._ollama

    @property
    def lm_studio(self):
        return self._lm_studio

    @property
    def mlx(self):
        return self._mlx

    @property
    def openai_compatible(self):
        return self._openai_compatible

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
    """Resolve the registry's ``default_llm`` to its Ollama tag.

    Used by agents as a tolerant fallback when a caller invokes the
    agent's public method without an explicit ``model=``. Production
    paths (engine → mode → agent) always pass a router-resolved tag;
    this fallback exists so direct agent tests don't have to fabricate
    a model string.
    """
    from src.memory.llm_registry import LLMRegistryLoader

    return LLMRegistryLoader().get_default_llm().ollama_id


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
                timeout=300,
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
        prevent; switching to a lower-refusal-floor LLM (Venice in the
        recommended config) recovers most of those cases. ``None``
        skips the fallback path entirely (legacy single-model
        behaviour). Caller resolves the tag via
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
                timeout=300,
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

# WARNING: This is a REMOTE API backend (no local memory to free) — unload is a
# no-op. The LLM/ComfyUI co-residence rule does NOT apply to API calls.
"""OpenAI-compatible HTTP client — generate text and structured JSON against any
``/v1/chat/completions`` gateway. Built for **Agent Router** (agentrouter.org)
and equally usable for OpenAI or any OpenAI-compatible endpoint (same shape,
different ``base_url`` + key).

Mirrors :class:`src.agents.lm_studio_client.LMStudioClient`'s public interface so
:class:`src.agents.llm_client.LLMClientPool` dispatches a registry-id-keyed call
to this backend identically to the local ones.

Backend specifics
-----------------
* **Endpoint** — ``{base_url}/chat/completions`` where ``base_url`` already
  includes ``/v1`` (e.g. ``https://agentrouter.org/v1``). Auth via an
  ``Authorization: Bearer <key>`` header; the key is read from the env var named
  in ``pipeline.yaml::openai_compatible.api_key_env`` (default
  ``AGENTROUTER_API_KEY``). The key is NEVER stored in config or logged.
* **JSON output** — uses the ROBUST path (prompt-instructed JSON + markdown-fence
  stripping + brace-finding + Pydantic post-validation), NOT OpenAI ``json_schema``
  (not all routed providers support it). When a schema is set and the model is not
  a reasoning model, ``response_format: {"type": "json_object"}`` is sent as a
  light nudge (widely supported); reasoning models (``<think>`` traces) get pure
  free-form + post-validation.
* **Unload** — no-op (remote; nothing local to free).
* **NSFW** — the routed frontier models are censored for explicit content, so this
  backend is for non-explicit / experimentation; the local uncensored default
  stays the explicit-prompt engine.

Configuration is read from ``config/pipeline.yaml`` under ``openai_compatible:``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import yaml

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE_YAML = _PROJECT_ROOT / "config" / "pipeline.yaml"

_DEFAULT_BASE_URL = "https://agentrouter.org/v1"
_DEFAULT_API_KEY_ENV = "AGENTROUTER_API_KEY"


class OpenAICompatibleError(Exception):
    """Any error talking to the OpenAI-compatible HTTP API."""


class OpenAICompatibleConnectionError(OpenAICompatibleError):
    """The API endpoint is not reachable (or no API key is configured)."""


class OpenAICompatibleGenerateError(OpenAICompatibleError):
    """Non-200 response from /chat/completions (incl. auth / quota errors)."""


# Inherit from OllamaJSONParseError so existing ``except OllamaJSONParseError``
# blocks catch this backend's parse failures too (parity with LMStudioClient).
from src.agents.llm_client import OllamaJSONParseError as _OllamaJSONParseError


class OpenAICompatibleJSONParseError(OpenAICompatibleError, _OllamaJSONParseError):
    """LLM response could not be parsed/validated as JSON."""


def _load_openai_compatible_config() -> dict:
    """Read the ``openai_compatible:`` block from pipeline.yaml."""
    if _PIPELINE_YAML.exists():
        with open(_PIPELINE_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("openai_compatible", {})
    return {}


class OpenAICompatibleClient:
    """Stateless HTTP wrapper around an OpenAI-compatible chat API.

    Model-agnostic: every call takes the remote ``model`` string explicitly;
    :class:`LLMClientPool` resolves registry-id → backend + identifier at the
    engine boundary. Constructs gracefully even when the API key is unset (so the
    pool still builds for local-only runs) — only actual API calls raise.

    Parameters
    ----------
    base_url : str | None
        Full OpenAI base incl. ``/v1`` (default ``https://agentrouter.org/v1`` or
        ``pipeline.yaml::openai_compatible.base_url``).
    api_key : str | None
        Override the key; otherwise read from the env var named in
        ``openai_compatible.api_key_env`` (default ``AGENTROUTER_API_KEY``).
    use_json_object : bool | None
        Send ``response_format: json_object`` for schema'd non-reasoning calls
        (default True; ``openai_compatible.use_json_object`` overrides).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        use_json_object: bool | None = None,
    ) -> None:
        cfg = _load_openai_compatible_config()
        self.base_url = (base_url or cfg.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = cfg.get("api_key_env", _DEFAULT_API_KEY_ENV)
        self.api_key = api_key or os.environ.get(self.api_key_env, "")
        self.request_timeout = int(cfg.get("request_timeout_seconds", 600))
        self.use_json_object = (
            use_json_object if use_json_object is not None
            else bool(cfg.get("use_json_object", True))
        )
        # Interface parity with LMStudioClient (the pool / callers may touch it).
        self.loaded_models: set[str] = set()

    # ------------------------------------------------------------------
    # Public API (mirrors LMStudioClient)
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        format_schema: dict | None = None,   # accepted for parity; unused
    ) -> str:
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            response_format=None,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float = 0.6,
        num_predict: int = 4096,
        schema: "type[BaseModel] | None" = None,
        prefill: str | None = None,            # parity; no effect on chat API
        fallback_model: str | None = None,
        skip_grammar_constraint: bool = False,
    ) -> dict | list:
        """Generate text, strip fences, parse + Pydantic-validate as JSON.

        Same call shape as LMStudioClient.generate_json so the pool forwards
        arguments verbatim. ``skip_grammar_constraint=True`` (reasoning models)
        sends NO ``response_format`` (pure free-form); otherwise a light
        ``json_object`` nudge is used when ``use_json_object`` and a schema is set.
        """
        try:
            return self._generate_json_once(
                system_prompt=system_prompt, user_prompt=user_prompt, model=model,
                temperature=temperature, num_predict=num_predict, schema=schema,
                skip_grammar_constraint=skip_grammar_constraint,
            )
        except OpenAICompatibleJSONParseError as primary_exc:
            if fallback_model and fallback_model != model:
                logger.warning("Primary API model %r failed JSON parse — retrying "
                               "once with fallback %r", model, fallback_model)
                return self._generate_json_once(
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    model=fallback_model, temperature=temperature,
                    num_predict=num_predict, schema=schema,
                    skip_grammar_constraint=skip_grammar_constraint,
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
        skip_grammar_constraint: bool = False,
    ) -> dict | list:
        response_format: dict | None = None
        if schema is not None and not skip_grammar_constraint and self.use_json_object:
            response_format = {"type": "json_object"}

        raw = self._chat(
            system_prompt=system_prompt, user_prompt=user_prompt, model=model,
            temperature=temperature, num_predict=num_predict,
            response_format=response_format,
        )
        cleaned = self._strip_fences(raw)
        # Frontier/reasoning models sometimes prefix prose or whitespace — start
        # at the first JSON opener so post-validation sees clean text.
        if cleaned:
            for i, ch in enumerate(cleaned):
                if ch in "{[":
                    cleaned = cleaned[i:]
                    break

        if schema is not None:
            from pydantic import ValidationError as _ValidationError
            try:
                validated = schema.model_validate_json(cleaned)
            except _ValidationError as exc:
                raise OpenAICompatibleJSONParseError(
                    f"API response failed schema validation against "
                    f"{schema.__name__}.\nErrors:\n{exc}\n"
                    f"Cleaned text ({len(cleaned)} chars):\n{cleaned[:500]}\n"
                    f"Full response ({len(raw)} chars):\n{raw[:500]}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise OpenAICompatibleJSONParseError(
                    f"Failed to parse API response as JSON.\nCleaned "
                    f"({len(cleaned)} chars):\n{cleaned[:500]}\nParse error: {exc}"
                ) from exc
            return validated.model_dump(mode="python")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleJSONParseError(
                f"Failed to parse API response as JSON.\nCleaned "
                f"({len(cleaned)} chars):\n{cleaned[:500]}\nParse error: {exc}"
            ) from exc

    def unload_model(self, model: str) -> None:
        """No-op — remote API has no local memory to free."""
        self.loaded_models.discard(model)

    def unload_all(self) -> None:
        """No-op — remote API has no local memory to free."""
        self.loaded_models.clear()

    def is_available(self) -> bool:
        """True iff an API key is configured (cheap — no network call). Actual
        reachability surfaces at generate time."""
        return bool(self.api_key)

    def list_models(self) -> list[str]:
        """Return the model ids the gateway exposes (``GET {base_url}/models``)."""
        if not self.api_key:
            return []
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            return [m["id"] for m in (resp.json().get("data") or [])]
        except (requests.ConnectionError, KeyError, ValueError):
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        num_predict: int,
        response_format: dict | None,
    ) -> str:
        """Send a chat-completion request and return the assistant text."""
        if not self.api_key:
            raise OpenAICompatibleConnectionError(
                f"No API key — set the {self.api_key_env} environment variable "
                f"(get a token at the Agent Router console)."
            )
        self.loaded_models.add(model)
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.request_timeout,
            )
        except requests.ConnectionError as exc:
            raise OpenAICompatibleConnectionError(
                f"Cannot reach the API at {self.base_url}.\n{exc}"
            ) from exc

        if resp.status_code != 200:
            raise OpenAICompatibleGenerateError(
                f"API returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise OpenAICompatibleGenerateError(
                f"API returned no choices: {resp.text[:500]}"
            )
        message = choices[0].get("message") or {}
        text = message.get("content", "") or ""
        usage = data.get("usage") or {}
        logger.debug("API response: %d chars, completion_tokens=%s",
                     len(text), usage.get("completion_tokens"))
        return text

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences wrapping JSON (parity with LMStudioClient)."""
        stripped = (text or "").strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()
                stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()

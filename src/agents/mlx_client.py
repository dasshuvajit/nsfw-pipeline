# WARNING: Always call llm_client.unload_all() after all LLM calls
# BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run simultaneously
# on 48 GB unified memory.
"""MLX HTTP client — generate text and structured JSON via the
``mlx_lm.server`` OpenAI-compatible REST API running locally.

Mirrors :class:`src.agents.llm_client.OllamaClient`'s public interface
so :class:`src.agents.llm_client.LLMClientPool` can dispatch a
registry-id-keyed call to either backend (Ollama / LM Studio / MLX)
without the caller knowing the difference.

Backend specifics
-----------------

* **Endpoint** — ``http://127.0.0.1:8080/v1/chat/completions``
  (OpenAI-compatible, ``mlx_lm.server`` default). Override the host
  + port via ``pipeline.yaml::mlx.base_url``.

* **One model per server** — ``mlx_lm.server`` loads exactly ONE model
  at startup via its ``--model`` argument. Switching models means
  restarting the server. The pipeline doesn't try to manage that —
  it assumes the operator started the server with the model the
  ``--llm`` flag will route to.

* **Soft-hint schema decoding** — unlike Ollama's ``format:`` or LM
  Studio's ``response_format: json_schema``, the MLX server accepts
  the ``response_format`` parameter but does NOT enforce the schema
  at decode time. The model is free to wander beyond enum / pattern
  / additionalProperties constraints. We still send the schema in
  every request because empirically it nudges the model toward the
  right top-level field names (without it, MLX Cydonia v3.1 wraps
  output in a self-invented ``series_plan`` shell and adds steps /
  materials / duration fields). Pydantic post-validation catches
  any leakage past the schema's structural shape and the standard
  retry-with-nudge loop in :mod:`src.modes._llm_helpers` re-rolls
  on drift. The client also strips ```` ```json ... ``` ```` markdown
  fences via the shared ``_strip_fences`` helper.

* **No unload protocol** — ``mlx_lm.server`` is a single-model
  per-process server with no HTTP endpoint for unload. The pipeline's
  phase-A-then-render contract requires the LLM to release unified
  memory before ComfyUI starts. For MLX this means the OPERATOR
  stops the ``mlx_lm.server`` process between Phase A and Phase B
  (or starts a fresh one). ``unload_model`` / ``unload_all`` log a
  reminder; they don't kill the server (a kill-by-port heuristic
  is too brittle on shared developer machines).

* **Startup command** (documented for the operator)::

      HF_HOME=~/AI/models ~/AI/envs/mlx-env/bin/mlx_lm.server \\
          --model mlx-community/Cydonia-24B-v3.1-4bit \\
          --host 127.0.0.1 --port 8080 --log-level INFO

  Verify with ``curl http://127.0.0.1:8080/v1/models``.

Configuration is read from ``config/pipeline.yaml`` under the ``mlx:``
key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import yaml

from src.agents.llm_client import OllamaJSONParseError as _OllamaJSONParseError

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPELINE_YAML = _PROJECT_ROOT / "config" / "pipeline.yaml"


class MlxError(Exception):
    """Any error talking to the mlx_lm.server HTTP API."""


class MlxConnectionError(MlxError):
    """mlx_lm.server is not reachable."""


class MlxGenerateError(MlxError):
    """Non-200 response from /v1/chat/completions."""


class MlxJSONParseError(MlxError, _OllamaJSONParseError):
    """LLM response could not be parsed as valid JSON.

    Inherits from :class:`OllamaJSONParseError` (round-20) so the
    existing retry catch blocks in ``src.modes._llm_helpers`` and
    ``src.agents.scene_facet_generator`` catch MLX failures too —
    same fix as :class:`LMStudioJSONParseError`.
    """


def _load_mlx_config() -> dict:
    """Read the ``mlx:`` block from pipeline.yaml."""
    if _PIPELINE_YAML.exists():
        with open(_PIPELINE_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("mlx", {})
    return {}


class MlxClient:
    """Stateless HTTP wrapper around ``mlx_lm.server``'s OpenAI-
    compatible REST API.

    Parameters
    ----------
    base_url : str | None
        Override the MLX server URL. Falls back to
        ``pipeline.yaml → mlx.base_url`` then
        ``http://127.0.0.1:8080``.
    """

    def __init__(self, base_url: str | None = None) -> None:
        cfg = _load_mlx_config()
        self.base_url = (
            base_url or cfg.get("base_url", "http://127.0.0.1:8080")
        ).rstrip("/")
        # MLX inference on Apple Silicon is typically faster per token
        # than llama.cpp via Ollama; 900s mirrors the Ollama default.
        # Override via ``pipeline.yaml::mlx.request_timeout_seconds``.
        self.request_timeout = int(cfg.get("request_timeout_seconds", 900))
        # Track every model identifier we've sent a request for so
        # ``unload_all()`` can log a reminder per model.
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
        format_schema: dict | None = None,  # accepted for interface parity, ignored
    ) -> str:
        """Send a prompt to mlx_lm.server and return the raw assistant text.

        ``format_schema`` is accepted for parity with
        :meth:`OllamaClient.generate` / :meth:`LMStudioClient.generate`
        but ignored — mlx_lm.server doesn't enforce JSON schema at
        decode time. Schema discipline is enforced via Pydantic
        post-validation in :meth:`generate_json`.

        Raises
        ------
        MlxConnectionError
            If the server is unreachable.
        MlxGenerateError
            If the server returns a non-200 status.
        """
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
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
        prefill: str | None = None,
        fallback_model: str | None = None,
        skip_grammar_constraint: bool = True,  # NB: ignored, always free-form
    ) -> dict | list:
        """Generate text, strip markdown fences, parse as JSON.

        Same call shape as :meth:`OllamaClient.generate_json` so
        :class:`LLMClientPool` forwards arguments verbatim.

        ``schema`` — when supplied, applied as Pydantic post-validation
        ONLY. mlx_lm.server doesn't enforce JSON schema at decode time
        the way Ollama's ``format:`` or LM Studio's
        ``response_format: json_schema`` do, so there's no point sending
        the schema in the request. The model is free to wander outside
        the schema's pattern / enum / additionalProperties constraints
        until Pydantic catches the drift and the retry loop fires.

        ``prefill`` / ``skip_grammar_constraint`` are accepted for
        interface parity with the Ollama + LM Studio clients but have
        no effect — mlx_lm.server's chat endpoint doesn't accept
        assistant-prefill and the grammar mode is already always off.

        Raises
        ------
        MlxJSONParseError
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
            )
        except MlxJSONParseError as primary_exc:
            if fallback_model and fallback_model != model:
                logger.warning(
                    "Primary MLX model %r failed JSON parse — retrying "
                    "once with fallback %r", model, fallback_model,
                )
                return self._generate_json_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=fallback_model,
                    temperature=temperature,
                    num_predict=num_predict,
                    schema=schema,
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
    ) -> dict | list:
        """One attempt of generate_json — primary OR fallback model."""
        # Round-20 — send the schema as a soft hint via ``response_
        # format``. mlx_lm.server doesn't enforce but the model uses
        # it to choose top-level field names. Without it MLX Cydonia
        # wraps output in a self-invented ``series_plan`` shell.
        format_schema: dict | None = None
        if schema is not None:
            try:
                pydantic_schema = schema.model_json_schema()
            except Exception:  # pragma: no cover — defensive
                pydantic_schema = None
            if pydantic_schema is not None:
                format_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": pydantic_schema,
                    },
                }
        raw = self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            format_schema=format_schema,
        )
        cleaned = self._strip_fences(raw)
        # MLX models commonly prefix their JSON with prose / newlines.
        # Find the first { or [ and start there.
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
                raise MlxJSONParseError(
                    f"MLX response failed schema validation against "
                    f"{schema.__name__}.\nErrors:\n{exc}\n"
                    f"Cleaned text ({len(cleaned)} chars):\n"
                    f"{cleaned[:500]}\n"
                    f"Full response ({len(raw)} chars):\n{raw[:500]}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise MlxJSONParseError(
                    f"Failed to parse MLX response as JSON.\n"
                    f"Cleaned text ({len(cleaned)} chars):\n"
                    f"{cleaned[:500]}\n"
                    f"Parse error: {exc}"
                ) from exc
            return validated.model_dump(mode="python")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise MlxJSONParseError(
                f"Failed to parse MLX response as JSON.\n"
                f"Cleaned text ({len(cleaned)} chars):\n"
                f"{cleaned[:500]}\n"
                f"Parse error: {exc}"
            ) from exc

    def unload_model(self, model: str) -> None:
        """Log a reminder; mlx_lm.server has no unload protocol.

        mlx_lm.server loads exactly one model at startup and stays
        in memory until the process exits. To free unified memory
        before ComfyUI rendering, stop the mlx_lm.server process
        (Ctrl-C in its terminal, or ``pkill -f mlx_lm.server``).
        """
        logger.warning(
            "MLX unload requested for %s — mlx_lm.server has no HTTP "
            "unload endpoint. Stop the mlx_lm.server process manually "
            "before running ComfyUI (e.g. `pkill -f mlx_lm.server`).",
            model,
        )
        self.loaded_models.discard(model)

    def unload_all(self) -> None:
        """Cascading unload — logs a reminder for each tracked model."""
        if not self.loaded_models:
            logger.info("MLX unload_all: no models tracked as loaded.")
            return
        logger.warning(
            "MLX unload_all requested for %d model(s) — mlx_lm.server "
            "has no HTTP unload endpoint. Stop the server process "
            "manually (`pkill -f mlx_lm.server`) before ComfyUI runs, "
            "otherwise the model stays in unified memory and ComfyUI's "
            "checkpoint load will OOM.",
            len(self.loaded_models),
        )
        self.loaded_models.clear()

    def is_available(self) -> bool:
        """Quick health check — can we reach the mlx_lm.server?"""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

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
        format_schema: dict | None = None,
    ) -> str:
        """Send a chat-completion request and return the assistant text."""
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
        if format_schema is not None:
            # mlx_lm.server doesn't enforce, but accepts the param and
            # uses it as a strong hint on field names. See module
            # docstring for round-20 context.
            payload["response_format"] = format_schema
        logger.debug(
            "MLX request: model=%s temp=%.1f max_tokens=%d format=%s",
            model, temperature, num_predict,
            "schema" if format_schema is not None else "free",
        )

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.request_timeout,
            )
        except requests.ConnectionError as exc:
            raise MlxConnectionError(
                f"Cannot reach mlx_lm.server at {self.base_url}. "
                f"Start it with: HF_HOME=~/AI/models "
                f"~/AI/envs/mlx-env/bin/mlx_lm.server --model {model} "
                f"--host 127.0.0.1 --port 8080\n{exc}"
            ) from exc

        if resp.status_code != 200:
            raise MlxGenerateError(
                f"mlx_lm.server returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise MlxGenerateError(
                f"mlx_lm.server returned no choices: {resp.text[:500]}"
            )
        message = choices[0].get("message") or {}
        text = message.get("content", "")
        usage = data.get("usage") or {}
        logger.debug(
            "MLX response: %d chars, completion_tokens=%s",
            len(text), usage.get("completion_tokens"),
        )
        return text

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences (``` ... ```). MLX models often
        wrap JSON in ``` ```json ... ``` ``` even when told not to.
        Identical semantics to :meth:`OllamaClient._strip_fences`."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()
                stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()

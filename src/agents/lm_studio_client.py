# WARNING: Always call llm_client.unload_model() / unload_all() after all
# LLM calls BEFORE any ComfyUI rendering. LLM and ComfyUI cannot run
# simultaneously on 48GB unified memory.
"""LM Studio HTTP client — generate text and structured JSON via the
OpenAI-compatible REST API LM Studio exposes locally.

Mirrors :class:`src.agents.llm_client.OllamaClient`'s public interface so
:class:`src.agents.llm_client.LLMClientPool` can dispatch a registry-id-
keyed call to either backend without the caller knowing the difference.

Backend specifics
-----------------

* **Endpoint** — ``http://localhost:1234/v1/chat/completions``
  (OpenAI-compatible). LM Studio also exposes ``/v1/models``,
  ``/v1/completions``, ``/v1/embeddings``; we only use chat.

* **JIT model loading** — LM Studio auto-loads the requested model on
  first request when it's not in memory (analogous to Ollama's
  ``keep_alive``). We don't issue an explicit load call.

* **Constrained-decoding JSON output** — the request's
  ``response_format`` field accepts an OpenAI ``json_schema`` shape::

      {
        "type": "json_schema",
        "json_schema": {
          "name": "MySchema",
          "strict": false,
          "schema": {...}
        }
      }

  Pydantic's ``extra="allow"`` produces a schema with
  ``additionalProperties: true``, which strict mode rejects — we ship
  ``strict: false`` to keep parity with Ollama's permissive ``format:``
  semantics. Pydantic post-validation in the caller catches any drift.

* **Unload** — LM Studio doesn't have an HTTP endpoint for model
  unload via the OpenAI-compatible API. We shell out to the ``lms`` CLI
  (bundled with LM Studio at ``~/.lmstudio/bin/lms``) and run
  ``lms unload <identifier>`` or ``lms unload --all``. After the
  subprocess returns we sleep 2 s so macOS unified memory actually
  frees before ComfyUI loads its checkpoint.

* **Assistant prefill** — OpenAI-compatible chat doesn't support a
  free-form assistant-prefill the way Ollama's ``/api/chat`` does. We
  emulate the same anti-refusal behaviour by appending a "Begin the
  JSON with `{`. Do NOT preface with 'Sure'" reminder to the user
  message when ``prefill`` is non-empty + schema is set. Schema-
  constrained decoding handles the structural guarantee.

Configuration is read from ``config/pipeline.yaml`` under the
``lm_studio:`` key.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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

# Default location of the lms binary on macOS after LM Studio install.
# Configurable via ``pipeline.yaml::lm_studio.lms_binary``.
_DEFAULT_LMS_BINARY = str(Path.home() / ".lmstudio" / "bin" / "lms")


class LMStudioError(Exception):
    """Any error talking to the LM Studio HTTP API."""


class LMStudioConnectionError(LMStudioError):
    """LM Studio server is not reachable."""


class LMStudioGenerateError(LMStudioError):
    """Non-200 response from /v1/chat/completions."""


# Round-20 (2026-05-22) — LMStudioJSONParseError inherits from
# OllamaJSONParseError so the existing ``except OllamaJSONParseError``
# blocks in ``src.modes._llm_helpers`` and
# ``src.agents.scene_facet_generator`` catch LM Studio failures too.
# Pre-round-20 the inheritance was off (LMStudioError) and LM Studio
# parse failures crashed through the retry loop on the rare paths
# where the grammar mode didn't catch the issue first.
from src.agents.llm_client import OllamaJSONParseError as _OllamaJSONParseError


class LMStudioJSONParseError(LMStudioError, _OllamaJSONParseError):
    """LLM response could not be parsed as valid JSON."""


def _load_lm_studio_config() -> dict:
    """Read the ``lm_studio:`` block from pipeline.yaml."""
    if _PIPELINE_YAML.exists():
        with open(_PIPELINE_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("lm_studio", {})
    return {}


class LMStudioClient:
    """Stateless HTTP wrapper around the LM Studio OpenAI-compatible API.

    The client is model-agnostic — every call to :meth:`generate` /
    :meth:`generate_json` / :meth:`unload_model` takes the LM Studio
    model identifier explicitly. :class:`LLMClientPool` resolves
    registry-id → backend + identifier at the engine boundary.

    Parameters
    ----------
    base_url : str | None
        Override the LM Studio server URL. Falls back to
        ``pipeline.yaml → lm_studio.base_url`` then
        ``http://localhost:1234``.
    lms_binary : str | None
        Override the lms-CLI path used for unload. Falls back to
        ``pipeline.yaml → lm_studio.lms_binary`` then
        ``~/.lmstudio/bin/lms``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        lms_binary: str | None = None,
    ) -> None:
        cfg = _load_lm_studio_config()
        self.base_url = (
            base_url
            or cfg.get("base_url", "http://localhost:1234")
        ).rstrip("/")
        # LM Studio JIT loads can be slower than Ollama on first call —
        # 900s mirrors the Ollama default so a 24B-class model has
        # headroom under macOS unified-memory pressure.
        self.request_timeout = int(cfg.get("request_timeout_seconds", 900))
        self.lms_binary = os.path.expanduser(
            lms_binary or cfg.get("lms_binary", _DEFAULT_LMS_BINARY)
        )
        # Track every LM Studio identifier we've sent a request for so
        # ``unload_all()`` can free each one at end-of-cycle.
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
        """Send a prompt to LM Studio and return the raw assistant text.

        ``model`` — LM Studio identifier for this call (REQUIRED). The
        pool resolves registry-id → identifier at the engine boundary.

        ``format_schema`` — when supplied, the request uses LM Studio's
        ``response_format: json_schema`` for constrained decoding (the
        OpenAI-compatible analogue of Ollama's ``format:`` parameter).

        Raises
        ------
        LMStudioConnectionError
            If the server is unreachable.
        LMStudioGenerateError
            If the server returns a non-200 status.
        """
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            format_schema=format_schema,
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
        skip_grammar_constraint: bool = False,
    ) -> dict | list:
        """Generate text, strip markdown fences, parse as JSON.

        Uses a slightly lower temperature than ``generate()`` for more
        deterministic structured output. Same call shape as
        :meth:`OllamaClient.generate_json` so :class:`LLMClientPool`
        forwards arguments verbatim.

        When ``schema`` is provided, two layers of safety apply:

        1. **Constrained decoding** — the schema's JSON-schema dict is
           passed to LM Studio via ``response_format: json_schema``
           (``strict=false`` because Pydantic ``extra="allow"`` produces
           ``additionalProperties: true``, which strict mode rejects).
        2. **Pydantic post-validation** — the parsed text is validated
           through the Pydantic model so ``min_length`` / ``Literal`` /
           ``model_validator`` constraints catch anything the JSON
           grammar can't enforce.

        ``prefill`` is accepted for OllamaClient interface parity but
        has no effect under LM Studio's OpenAI-compatible chat — that
        endpoint doesn't accept a free-form assistant turn. Schema-
        constrained decoding handles the structural guarantee.

        ``fallback_model`` (Q11) — same semantics as OllamaClient.
        On JSON parse failure, retries once against the fallback id
        (which may itself be on a different backend; the pool dispatches
        again).

        Raises
        ------
        LMStudioJSONParseError
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
                skip_grammar_constraint=skip_grammar_constraint,
            )
        except LMStudioJSONParseError as primary_exc:
            if fallback_model and fallback_model != model:
                logger.warning(
                    "Primary LM Studio model %r failed JSON parse — "
                    "retrying once with fallback %r",
                    model, fallback_model,
                )
                return self._generate_json_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=fallback_model,
                    temperature=temperature,
                    num_predict=num_predict,
                    schema=schema,
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
        """One attempt of generate_json — primary OR fallback model.

        ``skip_grammar_constraint`` — Round-18 (2026-05-22). When True,
        the ``response_format: json_schema`` field is omitted from the
        request. Used for Qwen 3.5+ reasoning models whose
        ``<think>...</think>`` chain-of-thought block is incompatible
        with the grammar-constrained-decoding mode (the grammar
        rejects the first ``<`` token, model stalls, content comes
        back empty). Schema is still used for Pydantic post-validation.
        """
        format_schema: dict | None = None
        if schema is not None and not skip_grammar_constraint:
            try:
                pydantic_schema = schema.model_json_schema()
            except Exception:  # pragma: no cover — defensive
                pydantic_schema = None
            if pydantic_schema is not None:
                format_schema = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        # strict=false — Pydantic `extra="allow"` produces
                        # `additionalProperties: true`, which the strict
                        # variant rejects. Permissive parity with Ollama;
                        # Pydantic post-validation is the strict guard.
                        "strict": False,
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
        # Round-18 — reasoning models sometimes prefix their JSON with
        # whitespace / newlines. Find the first { or [ and start there.
        if skip_grammar_constraint and cleaned:
            for i, ch in enumerate(cleaned):
                if ch in "{[":
                    cleaned = cleaned[i:]
                    break

        if schema is not None:
            from pydantic import ValidationError as _ValidationError

            try:
                validated = schema.model_validate_json(cleaned)
            except _ValidationError as exc:
                raise LMStudioJSONParseError(
                    f"LM Studio response failed schema validation "
                    f"against {schema.__name__}.\nErrors:\n{exc}\n"
                    f"Cleaned text ({len(cleaned)} chars):\n"
                    f"{cleaned[:500]}\n"
                    f"Full response ({len(raw)} chars):\n{raw[:500]}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise LMStudioJSONParseError(
                    f"Failed to parse LM Studio response as JSON.\n"
                    f"Cleaned text ({len(cleaned)} chars):\n"
                    f"{cleaned[:500]}\n"
                    f"Parse error: {exc}"
                ) from exc
            return validated.model_dump(mode="python")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LMStudioJSONParseError(
                f"Failed to parse LM Studio response as JSON.\n"
                f"Cleaned text ({len(cleaned)} chars):\n"
                f"{cleaned[:500]}\n"
                f"Parse error: {exc}"
            ) from exc

    def unload_model(self, model: str) -> None:
        """Evict one model from memory via the ``lms`` CLI.

        ``model`` — LM Studio identifier to unload (REQUIRED).

        Sleeps 2 s after the subprocess returns so macOS unified memory
        actually frees before ComfyUI loads its checkpoint + LoRAs + VAE.
        Safe to call even if no model is loaded — the lms CLI treats
        ``unload <not-loaded-model>`` as a graceful no-op.
        """
        logger.info("Unloading LM Studio model %s from memory …", model)
        if not self._lms_available():
            logger.warning(
                "lms binary not found at %s — skipping unload. Models "
                "will stay loaded; set lm_studio.lms_binary in "
                "pipeline.yaml to point at your install.",
                self.lms_binary,
            )
            self.loaded_models.discard(model)
            return
        try:
            subprocess.run(
                [self.lms_binary, "unload", model],
                check=False,
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "lms unload %s timed out after 30 s — model may still "
                "be loaded.", model,
            )

        time.sleep(2)  # Allow memory to free
        self.loaded_models.discard(model)
        logger.info("LM Studio model unloaded. 2 s grace period complete.")

    def unload_all(self) -> None:
        """Evict every model this client has loaded.

        Uses ``lms unload --all`` (single subprocess) rather than
        iterating per-model — LM Studio's unload-all is atomic and
        faster than N separate calls. After this method returns,
        ``self.loaded_models`` is empty.
        """
        if not self.loaded_models:
            logger.info("LM Studio unload_all: no models tracked as loaded.")
            return
        if not self._lms_available():
            logger.warning(
                "lms binary not found at %s — skipping unload_all. "
                "Models will stay loaded.", self.lms_binary,
            )
            self.loaded_models.clear()
            return
        logger.info(
            "Unloading all LM Studio models (%d tracked) …",
            len(self.loaded_models),
        )
        try:
            subprocess.run(
                [self.lms_binary, "unload", "--all"],
                check=False,
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "lms unload --all timed out after 30 s — models may "
                "still be loaded.",
            )
        time.sleep(2)
        self.loaded_models.clear()
        logger.info(
            "LM Studio unload_all complete. 2 s grace period elapsed.",
        )

    def is_available(self) -> bool:
        """Quick health check — can we reach the LM Studio server?"""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def ensure_loaded(
        self,
        model: str,
        *,
        context_length: int,
        gpu_offload: str = "max",
    ) -> None:
        """Ensure ``model`` is loaded in LM Studio with at least
        ``context_length`` tokens of context.

        Round-14 (2026-05-21) — LM Studio's REST API doesn't accept a
        context_length parameter per request, so a model loaded with
        the GUI default (4096) will reject prompts longer than that
        with HTTP 400. We parse ``lms ps`` to see what's loaded and
        re-load via ``lms load`` if the current context is too small.

        Call this once per model BEFORE the first ``generate_json`` /
        ``generate`` call. Safe to invoke multiple times — when the
        model is already loaded with sufficient context it's a no-op
        beyond the subprocess overhead.

        Skips silently when the lms binary isn't available — the
        request will fail later with a clearer "context too small"
        error from LM Studio itself, which the operator can resolve
        by running ``lms load --context-length <N>`` manually.
        """
        if not self._lms_available():
            logger.warning(
                "ensure_loaded: lms binary not found at %s — cannot "
                "auto-load model with the right context length. Load "
                "manually if you hit context-overflow errors.",
                self.lms_binary,
            )
            return
        # Inspect current load state. `lms ps` prints a table; the
        # JSON flag (--json) gives machine-parseable output on newer
        # LM Studio builds, but to stay compatible with older installs
        # we parse the human table.
        loaded_ok = self._loaded_with_context(model, context_length)
        if loaded_ok:
            logger.debug(
                "ensure_loaded: %s already loaded with sufficient "
                "context (>= %d).", model, context_length,
            )
            self.loaded_models.add(model)
            return
        logger.info(
            "ensure_loaded: loading %s with context_length=%d gpu=%s …",
            model, context_length, gpu_offload,
        )
        # 2026-05-23 — when model id has a ":N" suffix (multi-instance
        # selector e.g. ":3" for 32K context), split it: pass the base
        # model id to `lms load` and the full id to --identifier so
        # the loaded instance gets the expected identifier. Without
        # this split, `lms load mistral-...:3` looks for a file with
        # ":3" in the name and fails.
        base_model = model
        load_args = [
            self.lms_binary, "load", model,
            "--context-length", str(context_length),
            "--gpu", gpu_offload,
        ]
        if ":" in model.split("/")[-1]:
            base_model = model.rsplit(":", 1)[0]
            load_args = [
                self.lms_binary, "load", base_model,
                "--identifier", model,
                "--context-length", str(context_length),
                "--gpu", gpu_offload,
            ]
        try:
            result = subprocess.run(
                load_args,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,  # bumped 180 → 300; 12B Q4 can take ~60-90s cold
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "ensure_loaded: lms load timed out after 180s for %s",
                model,
            )
            return
        if result.returncode != 0:
            logger.warning(
                "ensure_loaded: lms load returned %d; stderr=%s",
                result.returncode, (result.stderr or "")[:300],
            )
            return
        self.loaded_models.add(model)
        logger.info(
            "ensure_loaded: %s loaded with context_length=%d.",
            model, context_length,
        )

    def _loaded_with_context(self, model: str, min_context: int) -> bool:
        """Best-effort check of ``lms ps`` output to see whether
        ``model`` is loaded with at least ``min_context`` tokens of
        context. Returns False on any parse failure so the caller
        re-loads safely."""
        try:
            result = subprocess.run(
                [self.lms_binary, "ps"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False
        if result.returncode != 0:
            return False
        for line in (result.stdout or "").splitlines():
            stripped = line.strip()
            if not stripped or not stripped.startswith(model.split("/")[-1].split(":")[0]):
                continue
            # Line format (whitespace-separated):
            # IDENTIFIER  MODEL  STATUS  SIZE  CONTEXT  PARALLEL  DEVICE  TTL
            parts = stripped.split()
            if len(parts) < 5:
                return False
            # Find a column that looks like the context length.
            for tok in parts[2:]:
                if tok.isdigit():
                    return int(tok) >= min_context
            return False
        return False

    def list_models(self) -> list[str]:
        """Return model identifiers currently available on the LM Studio
        server (loaded OR downloaded but unloaded — LM Studio's
        ``/v1/models`` returns the union)."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except (requests.ConnectionError, KeyError):
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
        format_schema: dict | None,
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
            payload["response_format"] = format_schema
        logger.debug(
            "LM Studio request: model=%s temp=%.1f max_tokens=%d format=%s",
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
            raise LMStudioConnectionError(
                f"Cannot reach LM Studio at {self.base_url}. "
                f"Is LM Studio running with the server enabled?\n{exc}"
            ) from exc

        if resp.status_code != 200:
            raise LMStudioGenerateError(
                f"LM Studio returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LMStudioGenerateError(
                f"LM Studio returned no choices: {resp.text[:500]}"
            )
        message = choices[0].get("message") or {}
        text = message.get("content", "")
        usage = data.get("usage") or {}
        logger.debug(
            "LM Studio response: %d chars, completion_tokens=%s",
            len(text), usage.get("completion_tokens"),
        )
        return text

    def _lms_available(self) -> bool:
        """True iff the lms binary is on disk + executable."""
        return os.path.isfile(self.lms_binary) and os.access(
            self.lms_binary, os.X_OK,
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences (``` ... ```) that LLMs sometimes
        wrap JSON in despite a structural format directive. Identical
        semantics to :meth:`OllamaClient._strip_fences`."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()
                stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()

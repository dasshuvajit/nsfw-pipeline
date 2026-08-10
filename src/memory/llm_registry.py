"""LLM Registry — YAML-backed lookup over ``config/llm_models.yaml``.

Mirrors :mod:`src.memory.model_registry` and :mod:`src.memory.family_loader`
patterns: a single YAML file declares which locally-installed LLMs the
pipeline can use, plus a ``default_llm`` pointer for "use my pick when
no --llm flag and no per-role routing applies".

Construction validates the registry:

  * ``default_llm`` is present in the registry AND ``active: true``,
    otherwise :class:`LLMRegistryError` is raised at load time.
  * Each entry has the required identity fields (``display_name`` +
    the backend-specific identifier — ``ollama_id`` for Ollama
    entries, ``lm_studio_id`` for LM Studio entries).

Multi-backend support (round-14, 2026-05-21):
  Each entry declares ``backend: ollama`` (default) or
  ``backend: lm_studio``. The model identifier lives under
  ``ollama_id`` or ``lm_studio_id`` respectively; the registry's
  :attr:`LLMRegistryEntry.model_tag` property returns the right one
  per backend. :class:`src.agents.llm_client.LLMClientPool` uses this
  to dispatch generate-calls to the right backend client.

Read sites:

  * :class:`src.agents.llm_client.LLMClientPool` — looks up backend
    per registry id at call time.
  * CLIs (``--llm <id>`` flag, Phase 5) — validate user input and
    produce the "Available LLMs:" error per plan §3.6.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "llm_models.yaml"
)


class LLMRegistryError(Exception):
    """Base for LLM registry errors."""


class LLMNotFound(LLMRegistryError):
    """Lookup by id returned no row, or the row is marked ``active=False``."""


BACKEND_OLLAMA = "ollama"
BACKEND_LM_STUDIO = "lm_studio"
BACKEND_MLX = "mlx"
# Remote OpenAI-compatible API backend — any /v1/chat/completions gateway
# (OpenRouter, OpenAI, DeepSeek, GLM/Zhipu, Together, Groq, …); name describes
# the wire protocol, not a vendor. Dormant until a provider base_url + key are
# set. Identifier lives in ``openai_compatible_id``; auth + base_url come from
# pipeline.yaml::openai_compatible. NOTE: the API frontier models are CENSORED
# for explicit NSFW, so this backend is for non-explicit / experimentation only —
# the local uncensored default stays the explicit-prompt engine.
BACKEND_OPENAI_COMPATIBLE = "openai_compatible"
_KNOWN_BACKENDS: frozenset[str] = frozenset(
    {BACKEND_OLLAMA, BACKEND_LM_STUDIO, BACKEND_MLX, BACKEND_OPENAI_COMPATIBLE}
)


@dataclass(frozen=True)
class LLMRegistryEntry:
    """One LLM entry from ``config/llm_models.yaml``.

    The registry ``id`` is the stable handle used by ``--llm <id>``,
    ``pipeline.yaml::llm.routing.*``, and ``prompts.llm_id`` /
    ``scene_facets.llm_id`` DB rows.

    Backend identity (round-14, 2026-05-21):
      ``backend`` selects which local server reaches the model:

        * ``ollama`` (default) → Ollama's ``/api/generate``;
          identifier lives in ``ollama_id``.
        * ``lm_studio`` → LM Studio's OpenAI-compatible
          ``/v1/chat/completions``; identifier in ``lm_studio_id``.

      :attr:`model_tag` returns the backend-appropriate identifier.
      Both id fields are stored on the dataclass so callers can
      cross-reference (e.g. an Ollama-served LLM may carry an
      ``lm_studio_id`` annotation for documentation; the backend
      field decides which one the pool sends).
    """

    id: str
    ollama_id: str
    display_name: str
    description: str
    quant: str | None
    size_gb: float | None
    context_tokens: int | None
    refusal_rate: float | None
    strengths: list[str] = field(default_factory=list)
    families_recommended: list[str] = field(default_factory=list)
    active: bool = True
    # Round-14: backend dispatch. Default ``ollama`` keeps every
    # pre-existing entry working without YAML changes.
    backend: str = BACKEND_OLLAMA
    # Round-14: LM Studio identifier. Required when backend=lm_studio,
    # ignored otherwise. Empty string is the YAML-side "not set".
    lm_studio_id: str = ""
    # Round-20 (2026-05-22): MLX identifier (Hugging Face repo id the
    # mlx_lm.server was started with). Required when backend=mlx,
    # ignored otherwise.
    mlx_model_id: str = ""
    # Remote OpenAI-compatible model id (the `model` string sent to the gateway,
    # e.g. "anthropic/claude-opus-4.1" on OpenRouter). Required when
    # backend=openai_compatible.
    openai_compatible_id: str = ""
    # Round-18 (2026-05-22) — reasoning-model flag. Qwen 3.5+ thinking
    # models emit a ``<think>...</think>`` chain-of-thought block
    # before the answer, which is incompatible with LM Studio's
    # ``response_format: json_schema`` grammar (the grammar forces the
    # first token to be ``{`` but the model wants to start the
    # reasoning trace, hits the grammar wall, and emits empty content).
    # When ``is_reasoning_model: true``, LMStudioClient skips passing
    # response_format and falls back to free-form generation +
    # Pydantic post-validation (the reasoning trace goes into the
    # separate ``reasoning_content`` field, and the actual answer
    # lands in ``content`` cleanly).
    is_reasoning_model: bool = False

    @property
    def model_tag(self) -> str:
        """Backend-specific model identifier used on the wire.

        Returns the appropriate id for the entry's backend:
          - ``ollama`` → ``ollama_id``
          - ``lm_studio`` → ``lm_studio_id``
          - ``mlx`` → ``mlx_model_id``

        :class:`LLMClientPool` dispatches generate-calls using this
        value.
        """
        if self.backend == BACKEND_LM_STUDIO:
            return self.lm_studio_id
        if self.backend == BACKEND_MLX:
            return self.mlx_model_id
        if self.backend == BACKEND_OPENAI_COMPATIBLE:
            return self.openai_compatible_id
        return self.ollama_id

    @classmethod
    def from_dict(cls, llm_id: str, d: dict[str, Any]) -> "LLMRegistryEntry":
        backend = str(d.get("backend") or BACKEND_OLLAMA).strip()
        if backend not in _KNOWN_BACKENDS:
            raise LLMRegistryError(
                f"LLM {llm_id!r}: unknown backend {backend!r}. "
                f"Supported: {sorted(_KNOWN_BACKENDS)}."
            )
        # Required identity per backend.
        if backend == BACKEND_OLLAMA:
            required = {"ollama_id", "display_name"}
        elif backend == BACKEND_LM_STUDIO:
            required = {"lm_studio_id", "display_name"}
        elif backend == BACKEND_OPENAI_COMPATIBLE:
            required = {"openai_compatible_id", "display_name"}
        else:  # BACKEND_MLX
            required = {"mlx_model_id", "display_name"}
        missing = required - d.keys()
        if missing:
            raise LLMRegistryError(
                f"LLM {llm_id!r} (backend={backend}): missing required "
                f"keys {sorted(missing)}"
            )
        return cls(
            id=llm_id,
            ollama_id=str(d.get("ollama_id") or ""),
            lm_studio_id=str(d.get("lm_studio_id") or ""),
            mlx_model_id=str(d.get("mlx_model_id") or ""),
            openai_compatible_id=str(d.get("openai_compatible_id") or ""),
            backend=backend,
            display_name=str(d["display_name"]),
            description=str(d.get("description") or ""),
            quant=(str(d["quant"]) if d.get("quant") is not None else None),
            size_gb=(
                float(d["size_gb"]) if d.get("size_gb") is not None else None
            ),
            context_tokens=(
                int(d["context_tokens"])
                if d.get("context_tokens") is not None
                else None
            ),
            refusal_rate=(
                float(d["refusal_rate"])
                if d.get("refusal_rate") is not None
                else None
            ),
            strengths=list(d.get("strengths") or []),
            families_recommended=list(d.get("families_recommended") or []),
            active=bool(d.get("active", True)),
            is_reasoning_model=bool(d.get("is_reasoning_model", False)),
        )


class LLMRegistryLoader:
    """Read-only accessor for ``config/llm_models.yaml``.

    Construction validates the registry: ``default_llm`` must point to
    an active entry. Re-instantiation is cheap (parse is LRU-cached per
    path).
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        llms, default_id, fallback_id = _load_registry(self.config_path)
        self._llms = dict(llms)
        self._default_llm_id = default_id
        # Q11 — fallback_llm is optional in YAML. When absent it
        # mirrors default_llm so legacy single-LLM behaviour is the
        # default; ``LLMRouter.fallback()`` reads this.
        self._fallback_llm_id = fallback_id or default_id
        self._validate_default()
        self._validate_fallback()

    # ── public API ────────────────────────────────────────────────────
    def get_llm(
        self, llm_id: str, *, require_active: bool = True
    ) -> LLMRegistryEntry:
        entry = self._llms.get(llm_id)
        if entry is None:
            available = self._format_available()
            raise LLMNotFound(
                f"LLM {llm_id!r} is not in your registry "
                f"({self.config_path.name}).\n{available}"
            )
        if require_active and not entry.active:
            raise LLMNotFound(
                f"LLM {llm_id!r} is marked inactive in {self.config_path.name}. "
                f"Set 'active: true' or pick a different LLM "
                f"(see `python scripts/list_models.py`)."
            )
        return entry

    def list_llms(
        self, *, include_inactive: bool = False
    ) -> list[LLMRegistryEntry]:
        out = []
        for lid in sorted(self._llms):
            entry = self._llms[lid]
            if not include_inactive and not entry.active:
                continue
            out.append(entry)
        return out

    @property
    def default_llm_id(self) -> str:
        return self._default_llm_id

    def get_default_llm(self) -> LLMRegistryEntry:
        return self.get_llm(self._default_llm_id, require_active=True)

    @property
    def fallback_llm_id(self) -> str:
        return self._fallback_llm_id

    def get_fallback_llm(self) -> LLMRegistryEntry:
        return self.get_llm(self._fallback_llm_id, require_active=True)

    def has_llm(self, llm_id: str, *, include_inactive: bool = True) -> bool:
        entry = self._llms.get(llm_id)
        if entry is None:
            return False
        if not include_inactive and not entry.active:
            return False
        return True

    def backend_for_tag(self, model_tag: str) -> str:
        """Round-14 — return the backend that owns ``model_tag``.

        Used by :class:`src.agents.llm_client.LLMClientPool` to dispatch
        generate-calls to the right client. Looks up the first entry
        (active or inactive) whose ``model_tag`` matches; returns the
        Ollama backend as a defensive default when no match is found
        (legacy callers passing arbitrary Ollama tags that aren't yet
        registered should keep working).
        """
        for entry in self._llms.values():
            if entry.model_tag == model_tag:
                return entry.backend
        return BACKEND_OLLAMA

    def resolve_model_tag(self, value: str) -> str:
        """Resolve a ``--model-tag`` / ``--llm`` CLI value to a canonical backend
        model_tag. Accepts EITHER a backend model_tag (lm_studio_id / ollama_id —
        what :attr:`LLMRegistryEntry.model_tag` and ``DEFAULT_LLM_TAG`` provide) OR
        a registry KEY (the ``llms:`` entry id, e.g. ``gemma_4_26b_a4b_heretic``),
        returning the canonical model_tag.

        Raises :class:`LLMNotFound` (with the available list) when ``value`` matches
        neither active form — so a typo or a key/tag mix-up fails LOUDLY at the CLI
        instead of silently routing to Ollama (``backend_for_tag`` defaults there)
        and degrading every prompt to the fallback model. 2026-06-15: passing the
        registry KEY to ``--model-tag`` quietly produced an all-Cydonia canary."""
        for entry in self.list_llms(include_inactive=False):
            if entry.model_tag == value:
                return value                       # already a valid active model_tag
        if self.has_llm(value, include_inactive=False):
            return self.get_llm(value).model_tag   # a registry key → its model_tag
        raise LLMNotFound(
            f"--model-tag {value!r} is neither a known model id nor an active "
            f"registry key. If it is a new model, add it to "
            f"{self.config_path.name} first.\n{self._format_available()}"
        )

    def is_reasoning_model(self, model_tag: str) -> bool:
        """Round-18 — return True iff ``model_tag`` matches a registry
        entry flagged ``is_reasoning_model: true``. Used by the pool
        to know when to skip LM Studio's ``response_format: json_schema``
        constraint (incompatible with the reasoning <think> trace)."""
        for entry in self._llms.values():
            if entry.model_tag == model_tag:
                return entry.is_reasoning_model
        return False

    # ── internals ─────────────────────────────────────────────────────
    def _validate_default(self) -> None:
        if self._default_llm_id not in self._llms:
            raise LLMRegistryError(
                f"{self.config_path.name}: default_llm "
                f"{self._default_llm_id!r} is not declared under llms:. "
                f"Add an entry or change default_llm to one of: "
                f"{sorted(self._llms)}."
            )
        entry = self._llms[self._default_llm_id]
        if not entry.active:
            raise LLMRegistryError(
                f"{self.config_path.name}: default_llm "
                f"{self._default_llm_id!r} is marked active=false. "
                f"Set active=true or change default_llm to a different LLM."
            )

    def _validate_fallback(self) -> None:
        """Q11 — same validation shape as default_llm."""
        if self._fallback_llm_id not in self._llms:
            raise LLMRegistryError(
                f"{self.config_path.name}: fallback_llm "
                f"{self._fallback_llm_id!r} is not declared under llms:. "
                f"Add an entry or omit fallback_llm to inherit "
                f"default_llm. Available: {sorted(self._llms)}."
            )
        entry = self._llms[self._fallback_llm_id]
        if not entry.active:
            raise LLMRegistryError(
                f"{self.config_path.name}: fallback_llm "
                f"{self._fallback_llm_id!r} is marked active=false. "
                f"Set active=true, change fallback_llm, or omit it "
                f"(inherits default_llm)."
            )

    def _format_available(self) -> str:
        lines = [f"Available LLMs ({self.config_path.name}):"]
        for entry in self.list_llms(include_inactive=True):
            tag = "  ← default" if entry.id == self._default_llm_id else ""
            status = "active" if entry.active else "inactive"
            lines.append(f"  - {entry.id:<24} ({status}){tag}")
        lines.append("")
        lines.append("To install a new LLM:")
        lines.append("  1. install/register it for its backend "
                     "(e.g. `ollama pull <tag>` for Ollama)")
        lines.append("  2. add an entry to config/llm_models.yaml")
        lines.append("  3. re-run with --llm <id>")
        return "\n".join(lines)


@functools.lru_cache(maxsize=4)
def _load_registry(
    config_path: Path,
) -> tuple[dict[str, LLMRegistryEntry], str, str | None]:
    if not config_path.exists():
        raise LLMRegistryError(f"llm_models.yaml not found at {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    llms_raw = data.get("llms")
    if not isinstance(llms_raw, dict) or not llms_raw:
        raise LLMRegistryError(
            f"{config_path}: expected top-level `llms:` mapping with at "
            f"least one entry"
        )
    default_llm = data.get("default_llm")
    if not isinstance(default_llm, str) or not default_llm.strip():
        raise LLMRegistryError(
            f"{config_path}: expected top-level `default_llm: <id>` string"
        )
    # Q11 — fallback_llm is optional. None means "mirror default_llm".
    fallback_raw = data.get("fallback_llm")
    if fallback_raw is None:
        fallback_llm: str | None = None
    elif isinstance(fallback_raw, str) and fallback_raw.strip():
        fallback_llm = fallback_raw.strip()
    else:
        raise LLMRegistryError(
            f"{config_path}: fallback_llm must be a non-empty string or "
            f"omitted (got {fallback_raw!r})"
        )
    llms = {
        llm_id: LLMRegistryEntry.from_dict(llm_id, llm_dict)
        for llm_id, llm_dict in llms_raw.items()
    }
    return llms, default_llm.strip(), fallback_llm

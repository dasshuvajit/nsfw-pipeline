"""LLM Registry — YAML-backed lookup over ``config/llm_models.yaml``.

Mirrors :mod:`src.memory.model_registry` and :mod:`src.memory.family_loader`
patterns: a single YAML file declares which Ollama-installed LLMs the
pipeline can use, plus a ``default_llm`` pointer for "use my pick when
no --llm flag and no per-role routing applies".

Construction validates the registry:

  * ``default_llm`` is present in the registry AND ``active: true``,
    otherwise :class:`LLMRegistryError` is raised at load time.
  * Each entry has the required fields (``ollama_id``, ``display_name``);
    missing required field raises :class:`LLMRegistryError`.

Read sites:

  * :class:`src.agents.llm_router.LLMRouter` (Phase 3) — resolves
    role/family → registry id → ``ollama_id`` for every LLM call.
  * :mod:`src.agents.llm_client` (Phase 1 bridge) — falls back to
    ``default_llm`` when ``pipeline.yaml::llm.model`` is absent.
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


@dataclass(frozen=True)
class LLMRegistryEntry:
    """One LLM entry from ``config/llm_models.yaml``.

    ``ollama_id`` is the tag passed to Ollama's HTTP API (``/api/generate``
    payload's ``model`` field). The registry ``id`` is the stable handle
    used by ``--llm <id>``, ``pipeline.yaml::llm.routing.*``, and
    ``prompts.llm_id`` / ``scene_facets.llm_id`` DB rows.
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

    @classmethod
    def from_dict(cls, llm_id: str, d: dict[str, Any]) -> "LLMRegistryEntry":
        required = {"ollama_id", "display_name"}
        missing = required - d.keys()
        if missing:
            raise LLMRegistryError(
                f"LLM {llm_id!r}: missing required keys {sorted(missing)}"
            )
        return cls(
            id=llm_id,
            ollama_id=str(d["ollama_id"]),
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
        lines.append("  1. ollama pull <ollama_tag>")
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

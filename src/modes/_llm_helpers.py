"""Shared LLM retry + validation helper for mode planners and scene generators.

Replaces four near-identical ``_generate_*`` / ``_attempt_*`` pairs in
``theme_mode``, ``niche_mode``, ``style_mode`` (and, by the same shape,
``series_planner`` + ``scene_generator``). Each of those sites used to
own ~30 LOC of identical retry-with-nudge logic plus a private
validator — any fix or behaviour change had to land four times.

Contract: the caller supplies a validator that either returns the
validated/transformed result (scenes filter-in-place is fine) or
``None`` to signal "retry this". On two failed attempts we raise the
caller-supplied error type.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING, TypeVar

from src.agents.llm_client import OllamaClient, OllamaJSONParseError

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRY_NUDGE = (
    "Return ONLY a raw JSON object or array, no markdown fences, no commentary."
)


_AESTHETIC_ANCHOR_FIELDS = (
    "color_palette",
    "photographer_ref",
    "art_movement",
)


def repair_colon_suffix_aesthetic_keys(plan: dict[str, Any]) -> None:
    """Defensive repair for the "key with trailing colon" LLM quirk.

    Round-5 finding (2026-05-19 LLM A/B run, Cydonia heretic-vision):
    constrained-decoding shape mishaps sometimes produce a plan where
    the schema-required aesthetic anchor key (e.g. ``color_palette``)
    is null, while an EXTRA key with a trailing colon (e.g.
    ``"color_palette:"``) carries the actual value the LLM picked.
    ``extra="allow"`` on the SeriesPlan schema lets the colon-suffixed
    extra through silently — Pydantic accepts it, ``warn_if_missing_
    aesthetic_anchors`` logs the canonical key as missing, and the
    composer / PNG metadata both lose the Phase 3 signature-look
    pinning even though the LLM successfully picked a tag.

    This helper salvages the value: for each aesthetic field, if the
    canonical key is empty AND a same-named-with-trailing-colon key
    is populated, copy the value over and delete the bad key.

    Call this BEFORE :func:`warn_if_missing_aesthetic_anchors` so the
    warning fires only when the LLM truly failed to pick a tag.
    """
    for fld in _AESTHETIC_ANCHOR_FIELDS:
        if not plan.get(fld):
            colon_key = f"{fld}:"
            val = plan.get(colon_key)
            if val:
                plan[fld] = val
                del plan[colon_key]
                logger.info(
                    "Repaired colon-suffixed aesthetic key %r → %r (LLM "
                    "shape quirk); value preserved.", colon_key, fld,
                )


def warn_if_missing_aesthetic_anchors(
    plan: dict[str, Any], *, mode_name: str,
) -> None:
    """Verifier round-4 IMPORTANT-5 — soft check for Phase 3 aesthetic
    anchors. SeriesPlan's ``color_palette`` / ``photographer_ref`` /
    ``art_movement`` are ``Optional[str]`` so Ollama's constrained
    decoding doesn't grammar-force them — a chatty LLM can legally
    skip all three. When that happens the series renders without a
    signature look (Phase 3's whole point), silently. Logging a
    WARNING here surfaces the degradation in run_log without rejecting
    the plan (back-compat for legacy series + Pony, which legitimately
    drops photographer_ref and art_movement).

    Call from each mode's ``_validate_plan`` AFTER the required-field
    check — only log when the LLM otherwise produced a valid plan.
    """
    missing = [k for k in _AESTHETIC_ANCHOR_FIELDS if not plan.get(k)]
    if len(missing) >= 3:
        logger.warning(
            "%s: series plan has NO aesthetic anchors populated "
            "(color_palette / photographer_ref / art_movement all None). "
            "The signature-look pinning is inert for this series — "
            "PNG metadata, composer threading, and reproducibility all "
            "lose the Phase 3 anchors. Consider re-running with a "
            "stricter LLM or bumping temperature.", mode_name,
        )
    elif "color_palette" in missing:
        logger.warning(
            "%s: series plan missing color_palette anchor. The Phase 3 "
            "palette pinning is inert for this series. Pony-only "
            "expected, every other family should populate this.",
            mode_name,
        )


def run_llm_with_retry(
    client: OllamaClient,
    *,
    system: str,
    user: str,
    validator: Callable[[Any], T | None],
    temperature: float,
    num_predict: int,
    mode_name: str,
    error_factory: Callable[[str], Exception],
    model: str,
    schema: "type[BaseModel] | None" = None,
    retry_nudge: str = DEFAULT_RETRY_NUDGE,
    max_retries: int = 1,
) -> T:
    """Call ``generate_json`` with one retry; validator decides validity.

    Parameters
    ----------
    client : OllamaClient
        Shared LLM client. Caller owns unload timing.
    system, user : str
        System + user prompts fed to ``generate_json``.
    validator : Callable[[Any], T | None]
        Consumes the parsed JSON. Returns the validated value (possibly
        transformed — e.g. a filtered scene list) or ``None`` to reject.
    temperature, num_predict : float, int
        Forwarded to ``generate_json``.
    mode_name : str
        Log-prefix so multi-mode runs are easy to grep.
    error_factory : Callable[[str], Exception]
        Factory producing the exception to raise on final failure —
        lets each caller keep its own error class (ThemeModeError etc.)
        without importing every mode's error module here.
    schema : type[BaseModel] | None
        Pydantic model used for grammar-constrained decoding (Phase 4b
        ``format: <schema>`` API). When set, Ollama enforces structural
        validity at decode time so a chatty / loose-aligned LLM (Venice,
        Magnum) can't produce free-form prose where JSON is expected.
        ``None`` (default) preserves the legacy free-form path used
        before 2026-05-06; safe for callers whose validator already
        tolerates malformed input.
    retry_nudge : str
        Appended to ``user`` on retry attempts.
    max_retries : int
        Number of extra attempts after the first. Default 1 preserves
        the pre-refactor behaviour (one first try + one retry = two
        total attempts).
    model : str
        Ollama tag to use for this call (REQUIRED). Resolved by the
        caller via :class:`LLMRouter` (override → routing → default).
    """
    current_user = user
    last_reason = "no LLM attempts made"
    for attempt in range(max_retries + 1):
        try:
            result = client.generate_json(
                system, current_user,
                temperature=temperature, num_predict=num_predict,
                model=model,
                schema=schema,
            )
        except OllamaJSONParseError as exc:
            last_reason = f"JSON parse error: {exc}"
            logger.warning(
                "%s: JSON parse error on attempt %d/%d: %s",
                mode_name, attempt + 1, max_retries + 1, exc,
            )
        else:
            validated = validator(result)
            if validated is not None:
                return validated
            last_reason = "validator rejected output"
            logger.warning(
                "%s: validator rejected attempt %d/%d",
                mode_name, attempt + 1, max_retries + 1,
            )

        # Prepare retry prompt — only appended once, not stacked every loop.
        if attempt == 0:
            current_user = f"{user}\n\nIMPORTANT: {retry_nudge}"

    raise error_factory(
        f"{mode_name}: failed to produce valid output after "
        f"{max_retries + 1} attempts ({last_reason})"
    )


def validate_dict_with_required_fields(
    required: set[str],
) -> Callable[[Any], dict | None]:
    """Factory: validator that accepts a dict iff all required keys present."""
    def _validate(result: Any) -> dict | None:
        if not isinstance(result, dict):
            return None
        if required - result.keys():
            return None
        return result
    return _validate


def validate_scene_list(required: set[str]) -> Callable[[Any], list[dict] | None]:
    """Factory: validator that returns a filtered list of valid scene dicts.

    Scenes missing any required field are dropped silently. If zero
    scenes remain the whole attempt is rejected (validator returns
    None → retry).
    """
    def _validate(result: Any) -> list[dict] | None:
        if not isinstance(result, list):
            return None
        valid = [
            s for s in result
            if isinstance(s, dict) and not (required - s.keys())
        ]
        return valid or None
    return _validate

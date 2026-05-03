"""Tests for ``OllamaClient.generate_json(schema=...)`` — Pydantic
validation path.

Audit found this path was untested. It's load-bearing for Phase B+
(every LLM JSON output goes through it). We mock the underlying HTTP
call (``OllamaClient.generate``) to return canned strings and exercise:

  - Plain dict / list returns when ``schema`` is omitted.
  - Pydantic ``BaseModel`` validation (success + failure paths).
  - Pydantic ``RootModel`` validation (list-of-models case used by
    ``SceneList = RootModel[list[Scene]]``).
  - Markdown-fence stripping happens for both paths.
  - JSON-parse failures vs schema-validation failures get distinct
    helpful error messages (both ``OllamaJSONParseError``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field, RootModel

from src.agents.llm_client import OllamaClient, OllamaJSONParseError


# ── test schemas (kept tiny + local; not from src/agents/schemas.py) ──


class _Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)


class _PersonList(RootModel[list[_Person]]):
    pass


# ── helper: stub the underlying generate() ────────────────────────


def _client_returning(text: str) -> OllamaClient:
    """Build a real OllamaClient with .generate() patched to return ``text``."""
    client = OllamaClient()
    return client


def _patch_generate(text: str):
    """Context-manager-style patch for ``OllamaClient.generate``."""
    return patch.object(OllamaClient, "generate", return_value=text)


# ── No schema: plain dict / list passthrough ────────────────────────


def test_no_schema_returns_plain_dict():
    with _patch_generate('{"name": "Elara", "age": 32}'):
        result = OllamaClient().generate_json("sys", "user")
    assert result == {"name": "Elara", "age": 32}


def test_no_schema_returns_plain_list():
    with _patch_generate('[{"name": "Elara"}, {"name": "Mira"}]'):
        result = OllamaClient().generate_json("sys", "user")
    assert isinstance(result, list)
    assert result[0]["name"] == "Elara"


def test_no_schema_strips_markdown_fences():
    fenced = '```json\n{"name": "Elara", "age": 32}\n```'
    with _patch_generate(fenced):
        result = OllamaClient().generate_json("sys", "user")
    assert result == {"name": "Elara", "age": 32}


def test_no_schema_raises_on_invalid_json():
    with _patch_generate("not json at all"):
        with pytest.raises(OllamaJSONParseError, match="Failed to parse"):
            OllamaClient().generate_json("sys", "user")


# ── BaseModel schema: validation success ────────────────────────────


def test_basemodel_schema_returns_dict_after_validation():
    with _patch_generate('{"name": "Elara", "age": 32}'):
        result = OllamaClient().generate_json("sys", "user", schema=_Person)
    assert result == {"name": "Elara", "age": 32}
    # Must be a plain dict (not a Pydantic instance).
    assert isinstance(result, dict)


def test_basemodel_schema_strips_markdown_fences_too():
    fenced = '```json\n{"name": "Elara", "age": 32}\n```'
    with _patch_generate(fenced):
        result = OllamaClient().generate_json("sys", "user", schema=_Person)
    assert result["name"] == "Elara"


def test_basemodel_schema_validates_field_constraints():
    """Pydantic's ge/le constraints are enforced; bad values → schema error."""
    with _patch_generate('{"name": "Elara", "age": 999}'):
        with pytest.raises(
            OllamaJSONParseError, match="schema validation against _Person",
        ):
            OllamaClient().generate_json("sys", "user", schema=_Person)


def test_basemodel_schema_rejects_missing_required_field():
    with _patch_generate('{"age": 32}'):
        with pytest.raises(
            OllamaJSONParseError, match=r"(?i)name|field required",
        ):
            OllamaClient().generate_json("sys", "user", schema=_Person)


def test_basemodel_schema_rejects_wrong_type():
    """``age`` is `int`; "thirty-two" should fail."""
    with _patch_generate('{"name": "Elara", "age": "thirty-two"}'):
        with pytest.raises(
            OllamaJSONParseError, match=r"(?i)age|integer",
        ):
            OllamaClient().generate_json("sys", "user", schema=_Person)


# ── RootModel schema: list-of-models ────────────────────────────────


def test_rootmodel_schema_returns_list_of_dicts():
    """Mirrors the SceneList = RootModel[list[Scene]] pattern."""
    payload = (
        '[{"name": "Elara", "age": 32}, '
        '{"name": "Mira", "age": 28}]'
    )
    with _patch_generate(payload):
        result = OllamaClient().generate_json(
            "sys", "user", schema=_PersonList,
        )
    assert isinstance(result, list)
    assert result == [
        {"name": "Elara", "age": 32},
        {"name": "Mira", "age": 28},
    ]


def test_rootmodel_schema_validates_each_element():
    """One bad row in the list → whole batch fails validation."""
    payload = (
        '[{"name": "Elara", "age": 32}, '
        '{"name": "Mira", "age": 999}]'  # over le=150
    )
    with _patch_generate(payload):
        with pytest.raises(
            OllamaJSONParseError, match="schema validation against _PersonList",
        ):
            OllamaClient().generate_json("sys", "user", schema=_PersonList)


def test_rootmodel_schema_rejects_non_list():
    with _patch_generate('{"name": "Elara", "age": 32}'):
        with pytest.raises(
            OllamaJSONParseError, match=r"(?i)list|input should be a valid",
        ):
            OllamaClient().generate_json("sys", "user", schema=_PersonList)


# ── Error message quality ───────────────────────────────────────────


def test_validation_error_includes_field_path():
    """The error message should help the operator see WHICH field failed."""
    with _patch_generate('{"name": "Elara", "age": "not a number"}'):
        with pytest.raises(OllamaJSONParseError) as exc_info:
            OllamaClient().generate_json("sys", "user", schema=_Person)
    msg = str(exc_info.value)
    assert "_Person" in msg          # which schema
    assert "age" in msg.lower()      # which field


def test_parse_error_includes_truncated_response_text():
    """JSON-parse failures should include the cleaned text for debugging."""
    with _patch_generate("definitely not json {[}"):
        with pytest.raises(OllamaJSONParseError) as exc_info:
            OllamaClient().generate_json("sys", "user", schema=_Person)
    msg = str(exc_info.value)
    assert "definitely not json" in msg


# ── Temperature / num_predict pass-through ──────────────────────────


def test_generate_json_forwards_temperature_and_num_predict():
    """Smoke: kwargs make it through to ``generate``."""
    with patch.object(
        OllamaClient, "generate", return_value='{"name": "x", "age": 1}',
    ) as mock_gen:
        OllamaClient().generate_json(
            "sys", "user",
            temperature=0.3,
            num_predict=2048,
            schema=_Person,
        )
    _, kwargs = mock_gen.call_args
    assert kwargs["temperature"] == 0.3
    assert kwargs["num_predict"] == 2048


def test_generate_json_default_temperature_is_lower_than_generate():
    """Structured output should default to a more deterministic temp."""
    with patch.object(
        OllamaClient, "generate", return_value="{}",
    ) as mock_gen:
        OllamaClient().generate_json("sys", "user")
    _, kwargs = mock_gen.call_args
    assert kwargs["temperature"] == 0.6  # current default in source


# ── Phase 4b: Ollama format:schema constrained decoding ────────────


def test_generate_json_passes_format_schema_when_schema_provided():
    """When a Pydantic schema is supplied, generate_json must thread
    ``model_json_schema()`` through to ``generate(format_schema=...)``."""
    with patch.object(
        OllamaClient, "generate", return_value='{"name": "x", "age": 1}',
    ) as mock_gen:
        OllamaClient().generate_json("sys", "user", schema=_Person)
    _, kwargs = mock_gen.call_args
    assert kwargs.get("format_schema") is not None
    fmt = kwargs["format_schema"]
    # The schema dict matches what Pydantic produces.
    assert fmt == _Person.model_json_schema()
    # Sanity — the schema declares the expected fields.
    assert "name" in fmt.get("properties", {})
    assert "age" in fmt.get("properties", {})


def test_generate_json_omits_format_schema_when_no_schema_provided():
    """Free-form JSON (no schema arg) skips constrained decoding."""
    with patch.object(
        OllamaClient, "generate", return_value='{"k": 1}',
    ) as mock_gen:
        OllamaClient().generate_json("sys", "user")
    _, kwargs = mock_gen.call_args
    # Either format_schema is absent or explicitly None
    assert kwargs.get("format_schema") is None


def test_generate_passes_format_to_payload():
    """The HTTP payload to /api/generate must include ``format`` when
    a schema is supplied — that's the field Ollama 0.5+ uses for
    grammar-constrained decoding."""
    import json as _json
    captured: dict = {}

    class _MockResp:
        status_code = 200

        def json(self):
            return {"response": '{"name": "x", "age": 1}'}

    def _capture_post(url, json=None, timeout=None):
        captured["payload"] = json
        return _MockResp()

    with patch("src.agents.llm_client.requests.post", side_effect=_capture_post):
        OllamaClient().generate(
            "sys", "user",
            format_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
    payload = captured["payload"]
    assert "format" in payload
    assert payload["format"] == {
        "type": "object", "properties": {"name": {"type": "string"}}
    }


def test_generate_omits_format_when_no_schema():
    """Free-form generation must NOT add ``format`` to the payload —
    the field's mere presence with an empty value can confuse older
    Ollama servers."""
    captured: dict = {}

    class _MockResp:
        status_code = 200

        def json(self):
            return {"response": "free text"}

    def _capture_post(url, json=None, timeout=None):
        captured["payload"] = json
        return _MockResp()

    with patch("src.agents.llm_client.requests.post", side_effect=_capture_post):
        OllamaClient().generate("sys", "user")
    payload = captured["payload"]
    assert "format" not in payload

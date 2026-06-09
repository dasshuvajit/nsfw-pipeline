"""Tests for the OpenAI-compatible API client (Agent Router backend).

Hermetic — `requests.post` is mocked, so no network / API key needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.agents.openai_client import (
    OpenAICompatibleClient,
    OpenAICompatibleConnectionError,
    OpenAICompatibleGenerateError,
    OpenAICompatibleJSONParseError,
)


class _Schema(BaseModel):
    prompt: str
    n: int = 0


def _resp(content: str, status: int = 200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {"choices": [{"message": {"content": content}}], "usage": {}}
    r.text = content
    return r


def test_no_key_constructs_but_is_unavailable_and_raises_on_use():
    c = OpenAICompatibleClient(api_key="")
    assert c.is_available() is False          # pool can still construct it
    with pytest.raises(OpenAICompatibleConnectionError):
        c.generate("s", "u", model="m")


def test_generate_json_builds_bearer_request_and_validates():
    c = OpenAICompatibleClient(api_key="sk-test", base_url="https://x/v1")
    with patch("src.agents.openai_client.requests.post") as post:
        post.return_value = _resp('{"prompt": "hello", "n": 3}')
        out = c.generate_json("sys", "usr", model="claude-x", schema=_Schema)
    assert out == {"prompt": "hello", "n": 3}
    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == "https://x/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert kwargs["json"]["model"] == "claude-x"
    assert kwargs["json"]["messages"][0]["role"] == "system"
    # schema + non-reasoning → light json_object nudge
    assert kwargs["json"]["response_format"] == {"type": "json_object"}


def test_skip_grammar_omits_response_format():
    c = OpenAICompatibleClient(api_key="sk-test")
    with patch("src.agents.openai_client.requests.post") as post:
        post.return_value = _resp('{"prompt": "x"}')
        c.generate_json("s", "u", model="m", schema=_Schema, skip_grammar_constraint=True)
    assert "response_format" not in post.call_args.kwargs["json"]


def test_strips_leading_fence_before_validation():
    c = OpenAICompatibleClient(api_key="sk-test")
    body = '```json\n{"prompt": "y", "n": 1}\n```'
    with patch("src.agents.openai_client.requests.post") as post:
        post.return_value = _resp(body)
        out = c.generate_json("s", "u", model="m", schema=_Schema)
    assert out == {"prompt": "y", "n": 1}


def test_skips_prose_preamble_before_the_json():
    c = OpenAICompatibleClient(api_key="sk-test")
    with patch("src.agents.openai_client.requests.post") as post:
        post.return_value = _resp('Here is the JSON: {"prompt": "z", "n": 2}')
        out = c.generate_json("s", "u", model="m", schema=_Schema)
    assert out == {"prompt": "z", "n": 2}


def test_schema_validation_failure_raises_parse_error():
    c = OpenAICompatibleClient(api_key="sk-test")
    with patch("src.agents.openai_client.requests.post") as post:
        post.return_value = _resp('{"not_prompt": 1}')   # missing required `prompt`
        with pytest.raises(OpenAICompatibleJSONParseError):
            c.generate_json("s", "u", model="m", schema=_Schema)


def test_non200_raises_generate_error_with_body():
    c = OpenAICompatibleClient(api_key="sk-test")
    with patch("src.agents.openai_client.requests.post") as post:
        post.return_value = _resp("invalid api key", status=401)
        with pytest.raises(OpenAICompatibleGenerateError):
            c.generate("s", "u", model="m")


def test_fallback_model_retries_once_on_parse_failure():
    c = OpenAICompatibleClient(api_key="sk-test")
    with patch("src.agents.openai_client.requests.post") as post:
        post.side_effect = [_resp("not json at all"),            # primary fails
                            _resp('{"prompt": "ok"}')]           # fallback succeeds
        out = c.generate_json("s", "u", model="bad", schema=_Schema,
                              fallback_model="good")
    assert out == {"prompt": "ok", "n": 0}    # n defaults via the schema
    assert post.call_count == 2


def test_unload_is_noop():
    c = OpenAICompatibleClient(api_key="sk-test")
    c.loaded_models.add("m")
    c.unload_all()
    assert c.loaded_models == set()

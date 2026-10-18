"""Tests for the single-key provider client design."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.provider_clients import get_api_key, make_generator_client, make_judge_client


def test_get_api_key_returns_stripped_value(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "  abc123  ")
    assert get_api_key("TEST_PROVIDER_KEY") == "abc123"


def test_get_api_key_fails_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("TEST_PROVIDER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_PROVIDER_KEY"):
        get_api_key("TEST_PROVIDER_KEY")


def test_get_api_key_fails_for_blank_value(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "   ")
    with pytest.raises(RuntimeError, match="TEST_PROVIDER_KEY"):
        get_api_key("TEST_PROVIDER_KEY")


def test_get_api_key_fails_for_comma_separated_value(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "first,second")
    with pytest.raises(RuntimeError, match="single key"):
        get_api_key("TEST_PROVIDER_KEY")


def test_make_generator_client_reads_google_key(monkeypatch) -> None:
    import google.genai as genai
    from src import config

    monkeypatch.setenv("GOOGLE_API_KEY", "googlekey-1")

    class FakeClient:
        def __init__(self, api_key: str, **kwargs) -> None:
            self.api_key = api_key
            self._httpx_client = SimpleNamespace(headers={})
            self.models = SimpleNamespace(
                generate_content=lambda **call: call
            )

    monkeypatch.setattr(genai, "Client", FakeClient)
    client = make_generator_client()
    client.models.generate_content(model="test-model", contents="hi")
    assert client.api_key == "googlekey-1"
    assert client._httpx_client.headers.get("x-goog-api-key") == "googlekey-1"
    assert config.GENERATOR_TIMEOUT_MS >= 180_000


def test_make_judge_client_reads_groq_key(monkeypatch) -> None:
    import src.provider_clients as providers

    monkeypatch.setenv("GROQ_API_KEY", "groqkey-1")
    seen: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)
            self.base_url = kwargs["base_url"]

    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)
    client = make_judge_client()
    assert seen["api_key"] == "groqkey-1"
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert seen["max_retries"] == 0
    assert seen["timeout"] == 60

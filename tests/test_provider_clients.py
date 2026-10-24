"""Tests for the single-key provider client design."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.provider_clients import get_api_key, make_generator_client, make_judge_client, make_safety_client


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
    seen = {}

    class FakeClient:
        def __init__(self, api_key: str, http_options, **kwargs) -> None:
            seen["api_key"] = api_key
            seen["timeout"] = http_options.timeout
            seen["attempts"] = http_options.retry_options.attempts
            self.models = SimpleNamespace(
                generate_content=lambda **call: call
            )

    monkeypatch.setattr(genai, "Client", FakeClient)
    client = make_generator_client()
    client.models.generate_content(model="test-model", contents="hi")
    assert seen == {
        "api_key": "googlekey-1",
        "timeout": config.GENERATOR_TIMEOUT_MS,
        "attempts": 1,
    }


def test_generator_timeout_defaults_to_30_seconds():
    project = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment.pop("GENERATOR_TIMEOUT_MS", None)
    # Import in a fresh process, and disable .env loading there: a developer's local override must
    # not decide whether the documented default is correct.
    code = (
        "import dotenv; dotenv.load_dotenv = lambda *a, **k: False; "
        "from src.config import GENERATOR_TIMEOUT_MS; print(GENERATOR_TIMEOUT_MS)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    assert result.stdout.strip() == "30000"


def test_make_safety_client_has_a_separate_short_timeout_and_no_sdk_retry(monkeypatch) -> None:
    import google.genai as genai

    monkeypatch.setenv("GOOGLE_API_KEY", "googlekey-1")
    seen = {}

    class FakeClient:
        def __init__(self, api_key, http_options):
            seen["api_key"] = api_key
            seen["timeout"] = http_options.timeout
            seen["attempts"] = http_options.retry_options.attempts
            self.models = SimpleNamespace(generate_content=lambda **call: call)

    monkeypatch.setattr(genai, "Client", FakeClient)
    make_safety_client()

    assert seen == {"api_key": "googlekey-1", "timeout": 15_000, "attempts": 1}


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

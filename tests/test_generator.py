"""Generation path tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generator import (
    SYSTEM_PROMPT,
    build_messages,
    citation_numbers,
    citations_are_valid,
    generate_text,
)


def test_citation_numbers_extract_bracketed_references() -> None:
    assert citation_numbers("Hello [1] and [2, 3] and [1-3]") == [1, 2, 3]


def test_citation_numbers_ignore_years() -> None:
    assert citation_numbers("[2024] and [1]", passage_count=3) == [1]


def test_citations_are_valid_for_fully_cited_answer() -> None:
    passages = [
        {"title": "P1", "source": "s1", "text": "a", "url": "u1"},
        {"title": "P2", "source": "s2", "text": "b", "url": "u2"},
    ]
    assert citations_are_valid("Answer [1] [2].", passages)


def test_citations_are_invalid_when_dangling() -> None:
    passages = [{"title": "P1", "source": "s1", "text": "a", "url": "u1"}]
    assert not citations_are_valid("Answer [2].", passages)


def test_build_messages_formats_passages() -> None:
    passages = [{"title": "T", "source": "s", "text": "body", "url": "u"}]
    messages = build_messages("q", passages)
    assert messages[1]["content"].startswith("Evidence passages:")
    assert "[1]" in messages[1]["content"]
    assert "Answer each part of the question" in SYSTEM_PROMPT


def test_generate_text_normalizes_stop_finish_reason(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GENERATOR_MODEL", "m")
    monkeypatch.setenv("GENERATOR_TEMPERATURE", "0")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    from src import config
    from src.provider_clients import make_generator_client

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.models = SimpleNamespace(
                generate_content=lambda model, contents, config=None: SimpleNamespace(
                    text="answer",
                    candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
                    prompt_feedback=SimpleNamespace(),
                )
            )

        def __call__(self, *args, **kwargs) -> SimpleNamespace:
            raise AssertionError("expected client.models.generate_content path")

    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GENERATOR_MODEL", "m")
    monkeypatch.setattr(config, "GENERATOR_TEMPERATURE", 0.0)
    monkeypatch.setattr(config, "GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setattr(config, "MAX_TOKENS", 2048)
    monkeypatch.setattr(
        "src.provider_clients.genai.Client", FakeClient
    )
    client = make_generator_client()
    answer, finish_reason = generate_text(client, "sys", "user")
    assert answer == "answer"
    assert finish_reason == "stop"


def test_generate_text_rejects_empty_answer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GENERATOR_MODEL", "m")
    monkeypatch.setenv("GENERATOR_TEMPERATURE", "0")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    from src import config
    from src.provider_clients import make_generator_client

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.models = SimpleNamespace(
                generate_content=lambda model, contents, config=None: SimpleNamespace(
                    text="",
                    candidates=[None],
                    prompt_feedback=SimpleNamespace(),
                )
            )

    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GENERATOR_MODEL", "m")
    monkeypatch.setattr(config, "GENERATOR_TEMPERATURE", 0.0)
    monkeypatch.setattr(config, "GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setattr(config, "MAX_TOKENS", 2048)
    monkeypatch.setattr(
        "src.provider_clients.genai.Client", FakeClient
    )
    client = make_generator_client()
    with pytest.raises(RuntimeError, match="empty answer"):
        generate_text(client, "sys", "user")


def test_generate_text_rejects_truncated_answer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GENERATOR_MODEL", "m")
    monkeypatch.setenv("GENERATOR_TEMPERATURE", "0")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    from src import config
    from src.provider_clients import make_generator_client

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.models = SimpleNamespace(
                generate_content=lambda model, contents, config=None: SimpleNamespace(
                    text="short",
                    candidates=[
                        SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))
                    ],
                    prompt_feedback=SimpleNamespace(),
                )
            )

    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GENERATOR_MODEL", "m")
    monkeypatch.setattr(config, "GENERATOR_TEMPERATURE", 0.0)
    monkeypatch.setattr(config, "GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setattr(config, "MAX_TOKENS", 2048)
    monkeypatch.setattr(
        "src.provider_clients.genai.Client", FakeClient
    )
    client = make_generator_client()
    with pytest.raises(RuntimeError, match="output token limit"):
        generate_text(client, "sys", "user")

"""Generation path tests."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from src.deadline import DeadlineExceeded
from src.generator import (
    GenerationEmptyResponseError,
    build_messages,
    citation_numbers,
    citations_are_valid,
    generate,
    generate_text,
    parse_citations,
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


def test_citations_are_invalid_when_an_impossible_id_accompanies_a_valid_one() -> None:
    passages = [{"title": "P1", "source": "s1", "text": "a", "url": "u1"}]
    assert not citations_are_valid("Answer [1] and [99].", passages)


@pytest.mark.parametrize(
    ("answer", "passage_count", "cited", "invalid"),
    [
        ("Fact [1].", 3, [1], []),
        ("Fact [1, 2].", 3, [1, 2], []),
        ("Fact [1;3].", 3, [1, 3], []),
        ("Fact [ 1 , 2 ].", 3, [1, 2], []),
        ("Fact [1-3].", 3, [1, 2, 3], []),
        ("Fact [1] then [2] then [1].", 3, [1, 2], []),
        ("In 2024 [2024] this changed.", 3, [], []),
        ("Fact [1] and a year [2024].", 3, [1], []),
        ("Fact [99].", 3, [], ["[99]"]),
        ("Fact [0].", 3, [], ["[0]"]),
        ("Fact [1] and [99].", 3, [1], ["[99]"]),
        ("Fact [1-9].", 3, [], ["[1-9]"]),
        ("No citation here.", 3, [], []),
        ("Fact [see above].", 3, [], []),
        ("Fact [1, x].", 3, [], []),
    ],
)
def test_citation_parsing_matrix(answer, passage_count, cited, invalid) -> None:
    parsed = parse_citations(answer, passage_count)
    assert parsed == {"cited": cited, "invalid": invalid}


@pytest.fixture
def gemini_client(monkeypatch):
    """Build a generator client through the real factory, with a fake SDK client behind it."""
    from src import config
    from src.provider_clients import genai, make_generator_client

    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GENERATOR_MODEL", "m")
    monkeypatch.setattr(config, "GENERATOR_TEMPERATURE", 0.0)
    monkeypatch.setattr(config, "GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setattr(config, "MAX_TOKENS", 2048)

    def build(text="answer", finish_reason="STOP", candidates=None):
        candidates = candidates if candidates is not None else [
            SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))
        ]

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.models = SimpleNamespace(
                    generate_content=lambda model, contents, config=None: SimpleNamespace(
                        text=text, candidates=candidates, prompt_feedback=SimpleNamespace()
                    )
                )

            def __call__(self, *args, **kwargs):
                raise AssertionError("expected client.models.generate_content path")

        monkeypatch.setattr(genai, "Client", FakeClient)
        return make_generator_client()

    return build


def test_build_messages_formats_passages() -> None:
    passages = [{"title": "T", "source": "s", "text": "body", "url": "u"}]
    messages = build_messages("q", passages)
    assert messages[1]["content"].startswith("Evidence passages:")
    assert "[1]" in messages[1]["content"]


def test_generate_text_normalizes_stop_finish_reason(gemini_client) -> None:
    answer, finish_reason = generate_text(gemini_client(), "sys", "user")
    assert answer == "answer"
    assert finish_reason == "stop"


def test_generate_text_rejects_empty_answer(gemini_client) -> None:
    client = gemini_client(text="", candidates=[None])
    with pytest.raises(RuntimeError, match="empty answer"):
        generate_text(client, "sys", "user")


def test_generate_retries_one_malformed_empty_provider_response(monkeypatch):
    from src import generator

    calls = []

    def fake_generate_once(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise GenerationEmptyResponseError("malformed empty response")
        return "grounded answer [1]"

    monkeypatch.setattr(generator, "_generate_once", fake_generate_once)
    monkeypatch.setattr(generator, "call_with_transient_rate_limit_retry", lambda call, **kwargs: call())
    assert generate(object(), "question", []) == "grounded answer [1]"
    assert len(calls) == 2


def test_generate_stops_at_an_expired_deadline() -> None:
    with pytest.raises(DeadlineExceeded):
        generate(object(), "question", [], deadline=time.monotonic() - 1)


def test_generate_once_forwards_the_deadline_to_the_provider_call(monkeypatch) -> None:
    """The reported defect: the deadline stopped at generate() and never reached the request."""
    from src import generator

    seen = {}

    def fake_generate_text(client, system, user, max_output_tokens=None, deadline=None):
        seen["deadline"] = deadline
        return "grounded answer [1]", "stop"

    monkeypatch.setattr(generator, "generate_text", fake_generate_text)
    deadline = time.monotonic() + 30

    assert generator._generate_once(object(), "q", [], deadline=deadline) == "grounded answer [1]"
    assert seen["deadline"] == deadline


def test_generate_forwards_the_same_deadline_to_the_truncation_retry(monkeypatch) -> None:
    from src import generator

    deadline = time.monotonic() + 30
    seen = []

    def fake_generate_once(*args, **kwargs):
        seen.append(kwargs.get("deadline"))
        if len(seen) == 1:
            raise generator.GenerationTruncatedError("hit the output limit")
        return "short answer [1]"

    monkeypatch.setattr(generator, "_generate_once", fake_generate_once)
    monkeypatch.setattr(generator, "call_with_transient_rate_limit_retry",
                        lambda call, **kwargs: call())

    assert generate(object(), "q", [], deadline=deadline) == "short answer [1]"
    assert seen == [deadline, deadline]


def test_an_empty_response_retry_keeps_the_same_deadline(monkeypatch) -> None:
    from src import generator

    deadline = time.monotonic() + 30
    seen = []

    def fake_generate_once(*args, **kwargs):
        seen.append(kwargs.get("deadline"))
        if len(seen) == 1:
            raise GenerationEmptyResponseError("empty")
        return "answer [1]"

    monkeypatch.setattr(generator, "_generate_once", fake_generate_once)
    monkeypatch.setattr(generator, "call_with_transient_rate_limit_retry",
                        lambda call, **kwargs: call())

    assert generate(object(), "q", [], deadline=deadline) == "answer [1]"
    assert seen == [deadline, deadline]


def test_the_compact_retry_after_truncation_is_bounded_by_the_deadline(monkeypatch) -> None:
    from src import config, generator

    timeouts = []

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def generate_content(self, model, contents, config):
            self.calls += 1
            timeouts.append(config.http_options.timeout)
            finish_reason = "MAX_TOKENS" if self.calls == 1 else "STOP"
            return SimpleNamespace(
                text="answer [1]",
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))],
            )

    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GENERATOR_TIMEOUT_MS", 30_000)

    passages = [{"title": "T", "source": "S", "text": "body", "url": ""}]
    assert generate(FakeClient(), "q", passages, deadline=time.monotonic() + 1) == "answer [1]"
    assert len(timeouts) == 2
    assert all(0 < timeout <= 1000 for timeout in timeouts)


def test_generate_text_caps_the_http_timeout_at_the_deadline(monkeypatch) -> None:
    from src import config

    seen = {}

    class FakeClient:
        def __init__(self) -> None:
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def generate_content(self, model, contents, config):
            seen["timeout"] = config.http_options.timeout
            return SimpleNamespace(
                text="answer",
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
            )

    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GENERATOR_TIMEOUT_MS", 30_000)

    generate_text(FakeClient(), "sys", "user", deadline=time.monotonic() + 1)

    assert 0 < seen["timeout"] <= 1000


def test_generate_text_rejects_truncated_answer(gemini_client) -> None:
    client = gemini_client(text="short", finish_reason="MAX_TOKENS")
    with pytest.raises(RuntimeError, match="output token limit"):
        generate_text(client, "sys", "user")

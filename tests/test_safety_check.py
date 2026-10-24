import json
import time
from types import SimpleNamespace

import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from openai import RateLimitError

from src import retry
from src.deadline import DeadlineExceeded
from src.safety_check import SafetyCheckError, SAFETY_ACTIONS, SAFETY_PROMPT, classify_for_safety


def rate_limit(message):
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    return RateLimitError("rate limited", response=response, body={"error": {"message": message}})


def server_error(code=503, message="This model is currently experiencing high demand."):
    return ServerError(code, {"error": {"code": code, "message": message, "status": "UNAVAILABLE"}})


class FakeSafetyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


def test_safety_check_parses_json_and_uses_a_fixed_schema():
    client = FakeSafetyClient(['{"action":"crisis"}'])

    result = classify_for_safety(client, "I want to die")

    assert result == {"action": "crisis"}
    config = client.calls[0]["config"]
    assert config.temperature == 0
    assert config.response_mime_type == "application/json"
    assert set(config.response_schema.properties["action"].enum) == set(SAFETY_ACTIONS)
    assert SAFETY_PROMPT in config.system_instruction


def test_safety_check_stops_when_the_deadline_has_passed():
    client = FakeSafetyClient([])

    with pytest.raises(DeadlineExceeded):
        classify_for_safety(client, "question", deadline=time.monotonic() - 1)

    assert client.calls == []


def test_safety_check_retries_invalid_json_once_then_fails_safely():
    client = FakeSafetyClient(["not json", "still not json"])

    with pytest.raises(SafetyCheckError):
        classify_for_safety(client, "question")

    assert len(client.calls) == 2


def test_safety_check_retries_a_timeout_once_then_fails_safely():
    client = FakeSafetyClient([TimeoutError("provider timed out"), TimeoutError("provider timed out")])

    with pytest.raises(SafetyCheckError):
        classify_for_safety(client, "question")

    assert len(client.calls) == 2


def test_safety_check_retries_a_transient_rate_limit_and_keeps_the_question(monkeypatch):
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    client = FakeSafetyClient([
        rate_limit("Rate limit reached for requests per minute. Please try again in 0.5s"),
        '{"action":"none"}',
    ])

    assert classify_for_safety(client, "what is asthma") == {"action": "none"}
    assert len(client.calls) == 2
    assert waits == [pytest.approx(0.5)]


def test_safety_check_retries_a_transient_server_error(monkeypatch):
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    client = FakeSafetyClient([server_error(503), '{"action":"redirect"}'])

    assert classify_for_safety(client, "question") == {"action": "redirect"}
    assert len(client.calls) == 2
    assert waits == [pytest.approx(2.0)]


def test_safety_check_does_not_repeat_the_transient_retry_budget(monkeypatch):
    """After the bounded retries the check fails closed instead of starting a second cycle."""
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    client = FakeSafetyClient([server_error(503), server_error(503), server_error(503)])

    with pytest.raises(SafetyCheckError):
        classify_for_safety(client, "question")

    assert len(client.calls) == 3


def test_safety_check_does_not_retry_a_hard_daily_quota(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("must not sleep for a daily limit"))
    quota = ClientError(429, {
        "message": "Quota exceeded for requests per day.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    })
    client = FakeSafetyClient([quota])

    with pytest.raises(SafetyCheckError):
        classify_for_safety(client, "question")

    assert len(client.calls) == 1

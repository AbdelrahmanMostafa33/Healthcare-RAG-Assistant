import json
import time
from types import SimpleNamespace

import pytest

from src.deadline import DeadlineExceeded
from src.safety_check import SafetyCheckError, SAFETY_ACTIONS, SAFETY_PROMPT, classify_for_safety


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

from types import SimpleNamespace

import pytest

from src.provider_clients import RotatingCredentialCall, api_keys
from src.retry import ProviderQuotaExceeded, call_with_transient_rate_limit_retry


class ProviderFailure(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"message": message}}


def make_rotator(operation, keys=("key-a", "key-b")):
    created = []

    def client_factory(key):
        created.append(key)
        return SimpleNamespace(key=key)

    return RotatingCredentialCall(list(keys), client_factory, operation, "TestProvider"), created


def test_api_keys_trims_ignores_empty_values_and_deduplicates(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", " first , , second,first,  ")
    assert api_keys("TEST_PROVIDER_KEY") == ["first", "second"]


def test_api_keys_fails_when_value_contains_no_keys(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", " , , ")
    with pytest.raises(RuntimeError, match="TEST_PROVIDER_KEY"):
        api_keys("TEST_PROVIDER_KEY")


def test_rotator_fails_over_once_after_transient_rate_limit_and_uses_working_key_next():
    calls = []

    def operation(client):
        calls.append(client.key)
        if client.key == "key-a":
            raise ProviderFailure(429, "Rate limit reached for requests per minute")
        return "ok"

    rotator, created = make_rotator(operation)
    assert rotator() == "ok"
    assert rotator() == "ok"
    assert calls == ["key-a", "key-b", "key-b"]
    assert created == ["key-a", "key-b"]


def test_transient_cooldown_matches_the_shared_retry_wait(monkeypatch):
    import src.provider_clients as providers
    from src import retry

    now = [0.0]
    waits = []
    calls = []
    monkeypatch.setattr(providers.time, "monotonic", lambda: now[0])

    def fake_sleep(delay):
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr(retry.time, "sleep", fake_sleep)

    def operation(client):
        calls.append(client.key)
        if len(calls) < 3:
            raise ProviderFailure(429, "Rate limit reached for requests per minute")
        return "ok"

    rotator, _created = make_rotator(operation)
    assert call_with_transient_rate_limit_retry(rotator, max_transient_retries=1) == "ok"
    assert calls == ["key-a", "key-b", "key-a"]
    assert waits == [2.0]


def test_rotator_skips_a_bad_authentication_key_for_later_calls():
    calls = []

    def operation(client):
        calls.append(client.key)
        if client.key == "key-a":
            raise ProviderFailure(401, "invalid API key")
        return "ok"

    rotator, _created = make_rotator(operation)
    assert rotator() == "ok"
    assert rotator() == "ok"
    assert calls == ["key-a", "key-b", "key-b"]


def test_rotator_turns_all_key_daily_quota_failures_into_a_bounded_stop():
    def operation(_client):
        raise ProviderFailure(429, "Requests per day quota exceeded")

    rotator, _created = make_rotator(operation)
    with pytest.raises(ProviderQuotaExceeded, match="all currently available credentials") as error:
        rotator()
    assert "key-a" not in str(error.value) and "key-b" not in str(error.value)


def test_rotator_does_not_switch_keys_for_unrelated_client_errors():
    calls = []

    def operation(client):
        calls.append(client.key)
        raise ProviderFailure(403, "model is not available")

    rotator, created = make_rotator(operation)
    with pytest.raises(ProviderFailure, match="model is not available"):
        rotator()
    assert calls == ["key-a"]
    assert created == ["key-a"]

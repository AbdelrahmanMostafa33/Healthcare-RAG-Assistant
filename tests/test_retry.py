import httpx
import pytest
from openai import RateLimitError

from src import retry


def rate_limit(message):
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    return RateLimitError("rate limited", response=response, body={"error": {"message": message}})


def test_daily_token_quota_is_not_retried():
    error = rate_limit("Limit reached: tokens per day quota exceeded")
    assert retry.is_tpd_exhaustion(error)


def test_tokens_per_minute_with_retry_after_is_treated_as_transient():
    error = rate_limit("Rate limit reached for tokens per minute. Please try again in 2.5s")
    assert not retry.is_tpd_exhaustion(error)
    assert retry.retry_after_seconds(error) == pytest.approx(2.5)


def test_transient_errors_retry_then_return_value(monkeypatch):
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    calls = []

    def call():
        calls.append(1)
        if len(calls) == 1:
            raise rate_limit("Rate limit reached. Please try again in 0.25s")
        return "ok"

    assert retry.call_with_transient_rate_limit_retry(call, max_transient_retries=2) == "ok"
    assert len(calls) == 2 and waits == [pytest.approx(0.25)]


def test_hard_quota_fails_fast_without_sleep(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("must not sleep for a hard quota"))
    calls = []

    def call():
        calls.append(1)
        raise rate_limit("You have exhausted tokens per day")

    with pytest.raises(RuntimeError, match="daily/token quota exhausted"):
        retry.call_with_transient_rate_limit_retry(call, max_transient_retries=5)
    assert len(calls) == 1


def test_transient_retry_budget_is_bounded(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    calls = []

    def call():
        calls.append(1)
        raise rate_limit("Please try again in 0s")

    with pytest.raises(RateLimitError):
        retry.call_with_transient_rate_limit_retry(call, max_transient_retries=2)
    assert len(calls) == 3


def test_negative_retry_budget_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        retry.call_with_transient_rate_limit_retry(lambda: None, max_transient_retries=-1)

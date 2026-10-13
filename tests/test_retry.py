import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from openai import RateLimitError

from src import retry


def rate_limit(message):
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    return RateLimitError("rate limited", response=response, body={"error": {"message": message}})


def server_error(code, message="This model is currently experiencing high demand."):
    """A Gemini-style failure: the status lives on ``code``, not ``status_code``."""
    return ServerError(code, {"error": {"code": code, "message": message, "status": "UNAVAILABLE"}})


def test_daily_token_quota_is_not_retried():
    error = rate_limit("Limit reached: tokens per day quota exceeded")
    assert retry.is_hard_quota_exhaustion(error)


def test_daily_request_limit_is_not_retried(monkeypatch):
    error = rate_limit("Rate limit reached for requests per day (RPD)")
    assert retry.is_hard_quota_exhaustion(error)
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("must not sleep for a daily limit"))
    with pytest.raises(retry.ProviderQuotaExceeded):
        retry.call_with_transient_rate_limit_retry(lambda: (_ for _ in ()).throw(error))


def test_tokens_per_minute_with_retry_after_is_treated_as_transient():
    error = rate_limit("Rate limit reached for tokens per minute. Please try again in 2.5s")
    assert not retry.is_hard_quota_exhaustion(error)
    assert retry.retry_after_seconds(error) == pytest.approx(2.5)


def test_retry_after_header_takes_precedence_over_provider_message():
    response = httpx.Response(
        429,
        headers={"retry-after": "3.25"},
        request=httpx.Request("POST", "https://example.test"),
    )
    error = RateLimitError(
        "rate limited", response=response,
        body={"error": {"message": "Rate limit reached for requests per minute"}},
    )
    assert retry.retry_after_seconds(error) == pytest.approx(3.25)


def test_retry_after_wait_honors_provider_delay_up_to_the_cap(monkeypatch):
    """The wait holds a worker thread, so an hour-long Retry-After must be truncated."""
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    response = httpx.Response(
        429, headers={"retry-after": "3600"}, request=httpx.Request("POST", "https://example.test")
    )
    error = RateLimitError("rate limited", response=response, body={"error": {"message": "try again"}})
    calls = []

    def call():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return "ok"

    assert retry.call_with_transient_rate_limit_retry(call, max_transient_retries=1) == "ok"
    assert retry.MAX_RETRY_AFTER_SECONDS == 60.0
    assert waits == [60.0]


def test_transient_server_errors_are_retried_with_bounded_backoff(monkeypatch):
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    calls = []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise server_error(503)
        return "ok"

    assert retry.call_with_transient_rate_limit_retry(call, max_transient_retries=3) == "ok"
    assert len(calls) == 3 and waits == [2.0, 2.0]


def test_a_client_timeout_deadline_is_treated_as_a_transient_failure():
    """The SDK reports its own 60 s client timeout as 504, so it must be retryable."""
    assert retry.is_transient_server_error(server_error(504, "deadline exceeded"))
    assert retry.status_code(server_error(504)) == 504


def test_an_exhausted_server_error_is_one_failed_row_not_a_batch_stop(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    calls = []

    def call():
        calls.append(1)
        raise server_error(503)

    with pytest.raises(retry.ProviderTransientFailure, match="stayed unavailable"):
        retry.call_with_transient_rate_limit_retry(call, max_transient_retries=2)
    assert len(calls) == 3


def test_client_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("must not retry a 400"))
    error = ClientError(400, {"error": {"code": 400, "message": "invalid request", "status": "INVALID_ARGUMENT"}})
    with pytest.raises(ClientError):
        retry.call_with_transient_rate_limit_retry(lambda: (_ for _ in ()).throw(error))


def test_status_code_reads_both_sdk_attribute_names():
    assert retry.status_code(server_error(503)) == 503
    assert retry.status_code(rate_limit("slow down")) == 429
    assert retry.status_code(RuntimeError("no status here")) is None


def test_gemini_minute_limit_retry_info_is_transient_and_honored(monkeypatch):
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    error = ClientError(429, {
        "message": "Quota exceeded for generate requests per minute.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [{
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "47s",
        }],
    })
    assert retry.is_rate_limit_error(error)
    assert not retry.is_hard_quota_exhaustion(error)
    assert retry.retry_after_seconds(error) == 47.0
    calls = []

    def call():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return "ok"

    assert retry.call_with_transient_rate_limit_retry(call, max_transient_retries=1) == "ok"
    assert waits == [47.0]


def test_gemini_daily_quota_is_stopped_without_retry(monkeypatch):
    error = ClientError(429, {
        "message": "Quota exceeded for requests per day.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    })
    assert retry.is_hard_quota_exhaustion(error)
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("must not sleep for a daily limit"))
    with pytest.raises(retry.ProviderQuotaExceeded):
        retry.call_with_transient_rate_limit_retry(lambda: (_ for _ in ()).throw(error))


def test_non_finite_retry_after_is_ignored():
    response = httpx.Response(
        429, headers={"retry-after": "inf"}, request=httpx.Request("POST", "https://example.test")
    )
    error = RateLimitError(
        "rate limited", response=response,
        body={"error": {"message": "Rate limit reached for requests per minute"}},
    )
    assert retry.retry_after_seconds(error) is None


def test_unclassified_429_fails_fast_without_claiming_it_is_a_daily_quota(monkeypatch):
    error = RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.test")),
        body={},
    )
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("must not retry an unknown limit"))
    with pytest.raises(retry.ProviderRateLimitRejected, match="unclassified"):
        retry.call_with_transient_rate_limit_retry(lambda: (_ for _ in ()).throw(error))


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

    with pytest.raises(RuntimeError, match="daily or account quota exhausted"):
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

"""Bounded retries for transient provider failures from Groq/OpenAI-compatible and Gemini APIs.

Two failure kinds are retried a bounded number of times: rate limits (429) and transient server
failures (5xx, including the client timeout that the Gemini SDK reports as 504). Hard daily or
account quota exhaustion is raised immediately instead of being retried.

This module is deliberately independent of generation and evaluation so both can share the
same small provider utility without introducing an import dependency between those layers.
"""
import math
import re
import time

from openai import RateLimitError


MAX_RETRY_DELAY_SECONDS = 30.0
INITIAL_RETRY_DELAY_SECONDS = 2.0
# A provider-requested delay is honoured, but never for longer than this. The wait holds a worker
# thread for its whole duration, so an uncapped Retry-After of an hour would block a request for
# an hour; beyond the cap the request is retried early and the retry budget absorbs the repeat.
MAX_RETRY_AFTER_SECONDS = 60.0
# 5xx means the provider failed or ran out of time rather than rejecting the request itself.
TRANSIENT_SERVER_STATUS = (500, 502, 503, 504)


HARD_QUOTA_SIGNALS = (
    "tokens per day", "tpd", "daily token", "requests per day", "rpd", "daily request",
    "daily quota", "daily limit", "monthly limit", "quota exceeded", "exceeded your quota", "insufficient_quota",
)
MINUTE_LIMIT_SIGNALS = ("per minute", "per second", "perminute", "persecond", "tpm", "rpm", "itpm", "otpm")


class ProviderRateLimitRejected(RuntimeError):
    """A 429 cannot be classified as a transient rate limit and must fail fast."""


class ProviderQuotaExceeded(ProviderRateLimitRejected):
    """A provider reported a daily or account quota that retries cannot resolve."""


class ProviderTransientFailure(RuntimeError):
    """A transient provider failure that outlived the retry budget.

    This is one failed request, not a reason to abandon the remaining work: the provider replied,
    it just did not have capacity for this attempt. Batch callers record the row and continue.
    """


def status_code(error: Exception) -> int | None:
    """Return the provider HTTP status for OpenAI- and Gemini-style errors, or None.

    The two SDKs disagree: ``openai`` sets ``status_code``, ``google-genai`` sets ``code`` and
    leaves ``status_code`` unset on the exception.
    """
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    response_code = getattr(getattr(error, "response", None), "status_code", None)
    try:
        return int(response_code) if response_code is not None else None
    except (TypeError, ValueError):
        return None


def provider_error_message(err: Exception) -> str:
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error = body.get("error") or {}
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"]).strip()
    if isinstance(err, RateLimitError):
        return ""
    message = getattr(err, "message", None)
    return str(message or err).strip()


def is_transient_server_error(err: Exception) -> bool:
    """Recognize a 5xx capacity failure (unavailable, overloaded, deadline exceeded).

    A Gemini 503 ("model is currently experiencing high demand") and a 504 from the client
    timeout both land here: the request was never rejected on its merits, so retrying is right.
    """
    return status_code(err) in TRANSIENT_SERVER_STATUS


def _google_quota_details(err: Exception) -> str:
    details = getattr(err, "details", None)
    if details is None:
        return ""
    try:
        return str(details).lower()
    except Exception:
        return ""


def is_rate_limit_error(err: Exception) -> bool:
    """Recognize provider 429 exceptions without treating unrelated API errors as throttles."""
    if isinstance(err, RateLimitError):
        return True
    return getattr(err, "code", None) == 429 or getattr(err, "status_code", None) == 429


def is_hard_quota_exhaustion(err: Exception) -> bool:
    """Identify hard daily/account quota exhaustion, not recoverable per-minute limits."""
    body = getattr(err, "body", None)
    error = body.get("error") or {} if isinstance(body, dict) else {}
    error_type = str(error.get("type") or "").lower() if isinstance(error, dict) else ""
    message = f"{provider_error_message(err)} {_google_quota_details(err)}".lower()
    if error_type == "insufficient_quota":
        return True
    if any(signal in message for signal in ("per day", "perday", "tokens per day", "requests per day")):
        return True
    # Gemini reports both minute and daily exhaustion as RESOURCE_EXHAUSTED/429. Preserve
    # minute-limited retries when the quota metadata identifies a rolling limit.
    if any(signal in message for signal in MINUTE_LIMIT_SIGNALS):
        return False
    return any(signal in message for signal in HARD_QUOTA_SIGNALS)


def retry_after_seconds(err: Exception) -> float | None:
    """Parse provider Retry-After headers/details and text such as ``retry in 2.5s``."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", {})
    header_value = headers.get("retry-after") if headers is not None else None
    if header_value is not None:
        try:
            seconds = float(header_value)
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None and math.isfinite(seconds) and seconds >= 0:
            return seconds
    details = getattr(err, "details", None)
    if isinstance(details, dict):
        details = details.get("details", [])
    if not isinstance(details, (list, tuple)):
        details = []
    for detail in details:
        if not isinstance(detail, dict) or "RetryInfo" not in str(detail.get("@type", "")):
            continue
        value = detail.get("retryDelay")
        if isinstance(value, str):
            match = re.fullmatch(r"\s*([\d.]+)s\s*", value)
            if match:
                try:
                    seconds = float(match.group(1))
                except ValueError:
                    seconds = None
                if seconds is not None and math.isfinite(seconds) and seconds >= 0:
                    return seconds
    message = provider_error_message(err)
    match = re.search(r"(?:try\s+again|retry)\s+in\s+([\d.]+)\s*s", message, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        seconds = float(match.group(1))
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _retry_wait(error: Exception, retries: int) -> float:
    """Seconds to wait before the next attempt: the provider's own delay when it is reasonable,
    otherwise bounded exponential backoff."""
    wait = retry_after_seconds(error)
    if wait is None:
        return min(INITIAL_RETRY_DELAY_SECONDS * (2 ** max(retries - 1, 0)), MAX_RETRY_DELAY_SECONDS)
    return min(wait, MAX_RETRY_AFTER_SECONDS)


def call_with_transient_rate_limit_retry(call, max_transient_retries: int = 5):
    """Call a no-argument operation, retrying transient 429s and 5xx replies a bounded number of times.

    The project client disables SDK retries so this helper owns the retry budget. Hard daily
    quota errors are raised immediately; transient rate limits and server failures receive
    bounded backoff. Exhausting the budget on a 5xx raises ProviderTransientFailure, which batch
    callers treat as one failed row rather than the end of the run.
    """
    if not isinstance(max_transient_retries, int) or max_transient_retries < 0:
        raise ValueError("max_transient_retries must be a non-negative integer")

    retries = 0
    while True:
        try:
            return call()
        except Exception as error:
            rate_limited = is_rate_limit_error(error)
            if not rate_limited and not is_transient_server_error(error):
                raise
            if rate_limited and is_hard_quota_exhaustion(error):
                raise _hard_quota_error(error) from error
            if rate_limited and not provider_error_message(error) and retry_after_seconds(error) is None:
                raise ProviderRateLimitRejected(
                    "Provider returned an unclassified rate-limit response; stopping instead of retrying blindly."
                ) from error
            if retries >= max_transient_retries:
                if rate_limited:
                    if isinstance(error, RateLimitError):
                        raise
                    raise ProviderRateLimitRejected("Provider rate-limit retries exhausted.") from error
                raise ProviderTransientFailure(
                    f"Provider stayed unavailable after {retries} transient retries."
                ) from error
            time.sleep(_retry_wait(error, retries))
            retries += 1


def _hard_quota_error(original: Exception) -> ProviderQuotaExceeded:
    """Build a clear error when the provider reports a hard quota limit."""
    message = provider_error_message(original)
    if message:
        return ProviderQuotaExceeded(f"Provider daily or account quota exhausted. {message}")
    return ProviderQuotaExceeded("Provider quota is exhausted; the request cannot succeed until quota is available.")

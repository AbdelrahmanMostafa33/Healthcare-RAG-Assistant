"""Bounded retries for transient rate limits from Groq/OpenAI-compatible and Gemini APIs.

This module is deliberately independent of generation and evaluation so both can share the
same small provider utility without introducing an import dependency between those layers.
"""
import math
import re
import time

from openai import RateLimitError


MAX_RETRY_DELAY_SECONDS = 30.0
INITIAL_RETRY_DELAY_SECONDS = 2.0


HARD_QUOTA_SIGNALS = (
    "tokens per day", "tpd", "daily token", "requests per day", "rpd", "daily request",
    "daily quota", "daily limit", "monthly limit", "quota exceeded", "exceeded your quota", "insufficient_quota",
)
MINUTE_LIMIT_SIGNALS = ("per minute", "per second", "perminute", "persecond", "tpm", "rpm", "itpm", "otpm")


class ProviderRateLimitRejected(RuntimeError):
    """A 429 cannot be classified as a transient rate limit and must fail fast."""


class ProviderQuotaExceeded(ProviderRateLimitRejected):
    """A provider reported a daily or account quota that retries cannot resolve."""


def _error_message(err: Exception) -> str:
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error = body.get("error") or {}
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"]).strip()
    if isinstance(err, RateLimitError):
        return ""
    message = getattr(err, "message", None)
    return str(message or err).strip()


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
    message = f"{_error_message(err)} {_google_quota_details(err)}".lower()
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
    message = _error_message(err)
    match = re.search(r"(?:try\s+again|retry)\s+in\s+([\d.]+)\s*s", message, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        seconds = float(match.group(1))
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def call_with_transient_rate_limit_retry(call, max_transient_retries: int = 5):
    """Call a no-argument operation, retrying transient 429s a bounded number of times.

    The project client disables SDK retries so this helper owns the retry budget. Hard daily
    quota errors are raised immediately; transient rate limits receive bounded backoff.
    """
    if not isinstance(max_transient_retries, int) or max_transient_retries < 0:
        raise ValueError("max_transient_retries must be a non-negative integer")

    retries = 0
    while True:
        try:
            return call()
        except Exception as error:
            if not is_rate_limit_error(error):
                raise
            if is_hard_quota_exhaustion(error):
                raise _hard_quota_error(error) from error
            if not _error_message(error) and retry_after_seconds(error) is None:
                raise ProviderRateLimitRejected(
                    "Provider returned an unclassified rate-limit response; stopping instead of retrying blindly."
                ) from error
            if retries >= max_transient_retries:
                if isinstance(error, RateLimitError):
                    raise
                raise ProviderRateLimitRejected("Provider rate-limit retries exhausted.") from error
            wait = retry_after_seconds(error)
            if wait is None:
                wait = INITIAL_RETRY_DELAY_SECONDS * (2 ** max(retries - 1, 0))
                wait = min(wait, MAX_RETRY_DELAY_SECONDS)
            time.sleep(wait)
            retries += 1


def _hard_quota_error(original: Exception) -> ProviderQuotaExceeded:
    """Build a clear error when the provider reports a hard quota limit."""
    message = _error_message(original)
    if message:
        return ProviderQuotaExceeded(f"Provider daily or account quota exhausted. {message}")
    return ProviderQuotaExceeded("Provider quota is exhausted; the request cannot succeed until quota is available.")

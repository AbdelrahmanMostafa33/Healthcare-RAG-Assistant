"""Bounded retries for transient OpenAI-compatible provider rate limits.

This module is deliberately independent of generation and evaluation so both can share the
same small provider utility without introducing an import dependency between those layers.
"""
import re
import time

from openai import RateLimitError


HARD_QUOTA_SIGNALS = (
    "tokens per day", "tpd", "daily token", "daily quota", "daily limit",
    "monthly limit", "quota exceeded", "exceeded your quota", "insufficient_quota",
)


def _error_message(err: RateLimitError) -> str:
    body = getattr(err, "body", None)
    if not isinstance(body, dict):
        return ""
    error = body.get("error") or {}
    if not isinstance(error, dict):
        return ""
    return str(error.get("message") or "").strip()


def is_tpd_exhaustion(err: RateLimitError) -> bool:
    """Identify hard daily/quota exhaustion, not recoverable per-minute limits."""
    body = getattr(err, "body", None)
    if not isinstance(body, dict):
        return True  # unknown 429 body: fail fast rather than retrying blindly
    error = body.get("error") or {}
    if not isinstance(error, dict):
        return True
    message = str(error.get("message") or "").lower()
    if not message:
        return True
    return any(signal in message for signal in HARD_QUOTA_SIGNALS)


def retry_after_seconds(err: RateLimitError) -> float | None:
    """Parse provider text such as ``Please try again in 2.5s`` when no header is available."""
    message = _error_message(err)
    match = re.search(r"try\s+again\s+in\s+([\d.]+)\s*s", message, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        seconds = float(match.group(1))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def call_with_transient_rate_limit_retry(call, max_transient_retries: int = 5):
    """Call a no-argument operation, retrying transient 429s a bounded number of times.

    The OpenAI SDK may already have retried before surfacing the error. This outer retry handles
    short provider-imposed cooldowns after the SDK's retry budget is exhausted. Hard daily quota
    errors are raised immediately with an actionable message.
    """
    if not isinstance(max_transient_retries, int) or max_transient_retries < 0:
        raise ValueError("max_transient_retries must be a non-negative integer")

    retries = 0
    while True:
        try:
            return call()
        except RateLimitError as error:
            if is_tpd_exhaustion(error):
                raise _hard_quota_error(error) from error
            if retries >= max_transient_retries:
                raise
            wait = retry_after_seconds(error)
            if wait is None:
                wait = 2 ** max(retries, 1)
            time.sleep(wait)
            retries += 1


def _hard_quota_error(original: RateLimitError) -> RuntimeError:
    """Build a clear error when the provider reports a hard quota limit."""
    message = _error_message(original)
    if message:
        return RuntimeError(f"Generator daily/token quota exhausted. {message}")
    return RuntimeError(f"Provider returned a non-retryable rate-limit error: {original}")

"""Small provider clients with bounded failover across configured API keys."""
from __future__ import annotations

import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Callable

from google import genai
from google.genai import types
from openai import OpenAI

from src import config
from src.retry import (
    INITIAL_RETRY_DELAY_SECONDS,
    ProviderQuotaExceeded,
    is_hard_quota_exhaustion,
    retry_after_seconds,
)

logger = logging.getLogger(__name__)


def api_keys(env_name: str) -> list[str]:
    """Return unique, non-empty credentials from a comma-separated environment value."""
    values = list(dict.fromkeys(value.strip() for value in os.getenv(env_name, "").split(",") if value.strip()))
    if not values:
        raise RuntimeError(f"{env_name} is not set. Copy .env.example to .env and add an API key.")
    return values


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(error, "code", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class RotatingCredentialCall:
    """Call one provider operation, failing over once across keys on auth/quota errors.

    Credentials are tried lazily. Invalid keys are skipped for the lifetime of this object;
    rate-limited keys receive a cooldown. The request is never duplicated after a successful
    response, and unrelated errors such as missing-model or malformed-request errors propagate.
    """

    def __init__(
        self,
        keys: list[str],
        client_factory: Callable[[str], object],
        operation: Callable[..., object],
        provider_name: str,
    ):
        if not keys:
            raise ValueError("At least one provider credential is required.")
        self._keys = keys
        self._client_factory = client_factory
        self._operation = operation
        self._provider_name = provider_name
        self._clients: list[object | None] = [None] * len(keys)
        self._invalid: set[int] = set()
        self._cooldown_until = [0.0] * len(keys)
        self._next_index = 0
        self._last_error: Exception | None = None
        self._lock = threading.Lock()

    @property
    def credential_count(self) -> int:
        """Number of distinct credentials configured, without exposing their values."""
        return len(self._keys)

    def _client_for(self, index: int):
        with self._lock:
            if self._clients[index] is None:
                self._clients[index] = self._client_factory(self._keys[index])
            return self._clients[index]

    def _next_available(self, attempted: set[int]) -> int | None:
        now = time.monotonic()
        with self._lock:
            for offset in range(len(self._keys)):
                index = (self._next_index + offset) % len(self._keys)
                if index in attempted or index in self._invalid or self._cooldown_until[index] > now:
                    continue
                return index
        return None

    def __call__(self, *args, **kwargs):
        attempted: set[int] = set()
        errors: list[Exception] = []
        for _ in range(len(self._keys)):
            index = self._next_available(attempted)
            if index is None:
                break
            attempted.add(index)
            try:
                result = self._operation(self._client_for(index), *args, **kwargs)
            except Exception as error:
                status = _status_code(error)
                hard_quota = is_hard_quota_exhaustion(error)
                should_rotate = status in (401, 429) or (status == 403 and hard_quota)
                if not should_rotate:
                    raise

                errors.append(error)
                with self._lock:
                    self._last_error = error
                    self._next_index = (index + 1) % len(self._keys)
                    if status == 401:
                        self._invalid.add(index)
                    else:
                        delay = retry_after_seconds(error)
                        if delay is None:
                            delay = 24 * 60 * 60 if hard_quota else INITIAL_RETRY_DELAY_SECONDS
                        self._cooldown_until[index] = time.monotonic() + max(0.1, delay)
                logger.warning(
                    "%s credential %d/%d returned HTTP %s; trying another configured key.",
                    self._provider_name,
                    index + 1,
                    len(self._keys),
                    status,
                )
                continue

            with self._lock:
                self._next_index = index
                self._last_error = None
            return result

        if errors:
            transient_limits = [
                error for error in errors
                if _status_code(error) == 429 and not is_hard_quota_exhaustion(error)
            ]
            if transient_limits:
                raise transient_limits[-1]
            if any(is_hard_quota_exhaustion(error) for error in errors):
                raise ProviderQuotaExceeded(
                    "Provider quota is exhausted for all currently available credentials."
                ) from errors[-1]
            raise errors[-1]

        with self._lock:
            previous_error = self._last_error
        if previous_error is not None:
            raise previous_error
        raise RuntimeError(f"No {self._provider_name} API credentials are currently available.")


def make_generator_client():
    if config.GENERATOR_PROVIDER != "gemini":
        raise RuntimeError(f"Unsupported GENERATOR_PROVIDER: {config.GENERATOR_PROVIDER!r}.")

    def create_gemini_client(key: str):
        return genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=60_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    generate_content = RotatingCredentialCall(
        api_keys("GOOGLE_API_KEY"),
        create_gemini_client,
        lambda client, **kwargs: client.models.generate_content(**kwargs),
        "Gemini",
    )
    return SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content),
        credential_count=generate_content.credential_count,
    )


def make_judge_client():
    if config.JUDGE_PROVIDER != "groq":
        raise RuntimeError(f"Unsupported JUDGE_PROVIDER: {config.JUDGE_PROVIDER!r}.")

    def create_groq_client(key: str):
        return OpenAI(
            api_key=key,
            base_url=config.GROQ_BASE_URL,
            max_retries=0,
            timeout=60,
        )

    create_completion = RotatingCredentialCall(
        api_keys("GROQ_API_KEY"),
        create_groq_client,
        lambda client, **kwargs: client.chat.completions.create(**kwargs),
        "Groq",
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)),
        credential_count=create_completion.credential_count,
    )

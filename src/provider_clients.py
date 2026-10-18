"""Provider clients.

This file builds the two provider clients used by the project:

- make_generator_client() -> google-genai Client (Gemini generation)
- make_judge_client() -> openai.OpenAI client (Groq judging)

Each client reads a single key from the matching env var:

- GOOGLE_API_KEY
- GROQ_API_KEY

Both must contain one key. A comma-separated value is an error.

On this machine the Gemini generateContent endpoint only accepts the API key when it is
sent as the `x-goog-api-key` header, not as a Bearer credential. The generator client
sets that header explicitly so the call is not rejected with 401 CREDENTIALS_MISSING.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from openai import OpenAI

from src import config


def get_api_key(env_name: str) -> str:
    value = os.getenv(env_name)
    if value is None:
        raise RuntimeError(f"{env_name} is not set. Copy .env.example to .env and add an API key.")
    value = value.strip()
    if not value:
        raise RuntimeError(f"{env_name} is not set. Copy .env.example to .env and add an API key.")
    if "," in value:
        raise RuntimeError(f"{env_name} must contain a single key, not a comma-separated list.")
    return value


def _patch_gemini_api_key(client: genai.Client) -> None:
    # On this machine the generateContent endpoint only accepts the key in this header, not as a
    # Bearer credential; without it the call is rejected with 401 CREDENTIALS_MISSING.
    try:
        client._httpx_client.headers["x-goog-api-key"] = client.api_key
    except Exception:
        try:
            transport = getattr(client, "_transport", None)
            if transport is not None and hasattr(transport, "headers"):
                transport.headers["x-goog-api-key"] = client.api_key
        except Exception:
            pass


def make_generator_client() -> genai.Client:
    if config.GENERATOR_PROVIDER != "gemini":
        raise RuntimeError(f"Unsupported GENERATOR_PROVIDER: {config.GENERATOR_PROVIDER!r}.")

    key = get_api_key("GOOGLE_API_KEY")
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=config.GENERATOR_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    _patch_gemini_api_key(client)
    return client


def make_judge_client() -> OpenAI:
    if config.JUDGE_PROVIDER != "groq":
        raise RuntimeError(f"Unsupported JUDGE_PROVIDER: {config.JUDGE_PROVIDER!r}.")

    key = get_api_key("GROQ_API_KEY")
    return OpenAI(
        api_key=key,
        base_url=config.GROQ_BASE_URL,
        max_retries=0,
        timeout=60,
    )

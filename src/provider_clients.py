"""Provider clients.

This file builds the two provider clients used by the project:

- make_generator_client() -> google-genai Client (Gemini generation)
- make_safety_client() -> separate google-genai Client (short safety checks)
- make_judge_client() -> OpenAI-compatible client for Groq or Cerebras judging

Each client reads a single key from the matching env var:

- GOOGLE_API_KEY
- GROQ_API_KEY or CEREBRAS_API_KEY, depending on JUDGE_PROVIDER

Both must contain one key. A comma-separated value is an error.

The Gemini SDK sends the API key as the `x-goog-api-key` header.
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
    return client


def make_safety_client() -> genai.Client:
    """Build a separate Gemini client with a short timeout and no SDK retries."""
    if config.GENERATOR_PROVIDER != "gemini":
        raise RuntimeError(f"Unsupported GENERATOR_PROVIDER: {config.GENERATOR_PROVIDER!r}.")

    key = get_api_key("GOOGLE_API_KEY")
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=config.SAFETY_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def make_judge_client() -> OpenAI:
    providers = {
        "groq": ("GROQ_API_KEY", config.GROQ_BASE_URL),
        "cerebras": ("CEREBRAS_API_KEY", config.CEREBRAS_BASE_URL),
    }
    if config.JUDGE_PROVIDER not in providers:
        supported = ", ".join(sorted(providers))
        raise RuntimeError(
            f"Unsupported JUDGE_PROVIDER: {config.JUDGE_PROVIDER!r}. Choose {supported}."
        )

    key_name, base_url = providers[config.JUDGE_PROVIDER]
    key = get_api_key(key_name)
    return OpenAI(
        api_key=key,
        base_url=base_url,
        max_retries=0,
        timeout=60,
    )

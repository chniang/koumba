"""Thin wrapper around the OpenAI SDK for configurable LLM providers.

Provider settings come from the environment or .env:

- KOUMBA_BASE_URL: OpenAI-compatible base URL
- KOUMBA_API_KEY_ENV: name of the environment variable holding the key
- KOUMBA_MODEL: model name

The tests replace the client with a scripted fake, so they never touch the
network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_API_KEY_ENV = "MODEL_API_KEY"
DEFAULT_MODEL = "muse-spark-1.3"
DEFAULT_TIMEOUT_SECONDS = 30.0


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _env_value(name: str) -> str | None:
    return _clean(os.environ.get(name))


def provider_label(base_url: str, model: str) -> str:
    """Return a friendly provider name for the app banner."""

    normalized = (base_url or "").strip().rstrip("/")
    parsed = urlparse(normalized if "://" in normalized else f"//{normalized}")
    host = (parsed.hostname or "").lower()
    model_name = (model or "").strip().lower()

    if host in {"meta.ai", "api.meta.ai"} or host.endswith(".meta.ai"):
        return "Meta Model API"
    if host in {"fireworks.ai", "api.fireworks.ai"} or host.endswith(
        ".fireworks.ai"
    ):
        return "Fireworks"
    if "baseten" in host or model_name.startswith("meta-models/"):
        return "Baseten"
    if host:
        return host
    return "Custom provider"


@dataclass(frozen=True)
class ProviderConfig:
    """The provider settings needed to build a client or label the UI."""

    base_url: str
    api_key_env: str
    model: str

    @property
    def provider(self) -> str:
        return provider_label(self.base_url, self.model)


def load_config() -> ProviderConfig:
    """Read provider settings from .env and the environment."""

    load_dotenv()
    return ProviderConfig(
        base_url=_env_value("KOUMBA_BASE_URL") or DEFAULT_BASE_URL,
        api_key_env=_env_value("KOUMBA_API_KEY_ENV") or DEFAULT_API_KEY_ENV,
        model=_env_value("KOUMBA_MODEL") or DEFAULT_MODEL,
    )


def api_key_configured(config: ProviderConfig | None = None) -> bool:
    """Check the configured key without returning or printing it."""

    config = config or load_config()
    return _env_value(config.api_key_env) is not None


class MissingAPIKeyError(RuntimeError):
    """Raised when the configured API key environment variable is empty."""


class ModelRequestError(RuntimeError):
    """Raised after the model call still fails on its one retry."""


class ModelClient:
    """Small, explicit wrapper so the agent does not depend on SDK details."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        config = load_config()

        self.base_url = _clean(base_url) or config.base_url
        self.api_key_env = _clean(api_key_env) or config.api_key_env
        self.model = _clean(model) or config.model
        self.provider = provider_label(self.base_url, self.model)
        self.timeout = timeout

        if api_key is None:
            api_key = _env_value(self.api_key_env)
        api_key = _clean(api_key)

        # A fake client is allowed without a key.  This is what the tests use.
        if not api_key and client is None:
            raise MissingAPIKeyError(
                f"{self.api_key_env} is not set. Add it to your environment "
                "or to a .env file (see .env.example), then try again."
            )

        self.api_key = api_key

        if client is None:
            # max_retries=0 because complete() owns the single retry below.
            client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=timeout,
                max_retries=0,
            )
        self._client = client

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> Any:
        """Run one Chat Completions call, retrying once on failure."""

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if tools is not None:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return self._client.chat.completions.create(**request)
            except Exception as exc:
                # Keep the original error for the final message.
                last_error = exc
                if attempt == 0:
                    continue

        raise ModelRequestError(
            f"The {self.provider} model request failed after one retry. "
            f"Check {self.api_key_env}, the network connection, and "
            f"model {self.model}."
        ) from last_error

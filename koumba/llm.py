"""Thin wrapper around the OpenAI SDK for the Meta Model API.

The model is only called from the Gradio app when a key is configured.
The tests replace this client with a scripted fake, so they never touch the
network.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
DEFAULT_TIMEOUT_SECONDS = 30.0


class MissingAPIKeyError(RuntimeError):
    """Raised when MODEL_API_KEY is not configured."""


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
    ) -> None:
        if api_key is None:
            # Load .env if the caller did not provide a key.  We never create
            # .env and we never write the key anywhere.
            load_dotenv()
            api_key = os.environ.get("MODEL_API_KEY")

        api_key = (api_key or "").strip() or None

        # A fake client is allowed without a key.  This is what the tests use.
        if not api_key and client is None:
            raise MissingAPIKeyError(
                "MODEL_API_KEY is not set. Add it to your environment or to "
                "a .env file (see .env.example), then try again."
            )

        configured_model = model or os.environ.get("KOUMBA_MODEL") or DEFAULT_MODEL
        self.api_key = api_key
        self.model = configured_model.strip() or DEFAULT_MODEL
        self.timeout = timeout

        if client is None:
            # max_retries=0 because complete() owns the single retry below.
            client = OpenAI(
                api_key=api_key,
                base_url=BASE_URL,
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
            "The model request failed after one retry. Check the API key, "
            "network connection, and model name."
        ) from last_error

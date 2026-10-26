"""Provider configuration tests; no network and no real keys."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from koumba.llm import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MissingAPIKeyError,
    ModelClient,
    api_key_configured,
    load_config,
    provider_label,
)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/muse-spark-open"


def test_defaults_when_no_provider_env_is_set(llm_env) -> None:
    config = load_config()

    assert config.base_url == DEFAULT_BASE_URL
    assert config.api_key_env == DEFAULT_API_KEY_ENV
    assert config.model == DEFAULT_MODEL
    assert config.provider == "Meta Model API"
    assert not api_key_configured(config)


def test_fireworks_settings_and_key_env_are_used(llm_env, monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_BASE_URL", FIREWORKS_BASE_URL)
    monkeypatch.setenv("KOUMBA_MODEL", FIREWORKS_MODEL)
    monkeypatch.setenv("KOUMBA_API_KEY_ENV", "FIREWORKS_API_KEY")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-fireworks-key")

    config = load_config()
    assert config.provider == "Fireworks"
    assert api_key_configured(config)

    client = ModelClient(client=object())

    assert client.base_url == FIREWORKS_BASE_URL
    assert client.model == FIREWORKS_MODEL
    assert client.api_key_env == "FIREWORKS_API_KEY"
    assert client.provider == "Fireworks"
    assert client.api_key == "fake-fireworks-key"


def test_baseten_is_detected_from_the_model_name(llm_env, monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("KOUMBA_MODEL", "meta-models/muse-spark-open-1026")

    config = load_config()

    assert config.provider == "Baseten"


def test_unknown_base_url_falls_back_to_the_hostname() -> None:
    assert (
        provider_label("https://llm.example.com/v1", "custom-model")
        == "llm.example.com"
    )


def test_missing_key_error_names_the_configured_env_var(
    llm_env,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KOUMBA_API_KEY_ENV", "FIREWORKS_API_KEY")

    with pytest.raises(MissingAPIKeyError, match="FIREWORKS_API_KEY"):
        ModelClient()


def test_sdk_receives_the_configured_base_url_and_model(
    llm_env,
    monkeypatch,
) -> None:
    class RecordingOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=object())

    monkeypatch.setenv("KOUMBA_BASE_URL", FIREWORKS_BASE_URL)
    monkeypatch.setenv("KOUMBA_MODEL", FIREWORKS_MODEL)
    monkeypatch.setenv("KOUMBA_API_KEY_ENV", "FIREWORKS_API_KEY")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-fireworks-key")
    monkeypatch.setattr("koumba.llm.OpenAI", RecordingOpenAI)

    client = ModelClient()

    sdk = client._client
    assert isinstance(sdk, RecordingOpenAI)
    assert sdk.kwargs["base_url"] == FIREWORKS_BASE_URL
    assert sdk.kwargs["api_key"] == "fake-fireworks-key"
    assert sdk.kwargs["timeout"] == 30.0
    assert sdk.kwargs["max_retries"] == 0
    assert client.provider == "Fireworks"

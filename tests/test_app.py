"""The Gradio app must import and build without a key or a model call."""

from __future__ import annotations


def test_app_builds_without_api_key(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()

    assert demo is not None


def test_provider_banner_names_the_configured_provider(
    llm_env,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KOUMBA_BASE_URL",
        "https://api.fireworks.ai/inference/v1",
    )
    monkeypatch.setenv(
        "KOUMBA_MODEL",
        "accounts/fireworks/models/muse-spark-open",
    )
    monkeypatch.setenv("KOUMBA_API_KEY_ENV", "FIREWORKS_API_KEY")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake-fireworks-key")

    import koumba.app as app

    text, is_warning = app._provider_banner()

    assert not is_warning
    assert "Fireworks" in text
    assert "accounts/fireworks/models/muse-spark-open" in text


def test_provider_banner_warns_when_the_key_is_missing(llm_env) -> None:
    import koumba.app as app

    text, is_warning = app._provider_banner()

    assert is_warning
    assert "Meta Model API key not configured" in text
    assert "MODEL_API_KEY" in text


def test_respond_without_key_shows_a_clear_message(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    def should_not_run():
        raise AssertionError("the agent must not start without a key")

    monkeypatch.setattr(app, "_agent", should_not_run)

    history, message, _trips, _thoughts = app._respond("Maa ngi Sahm", [])

    assert message == ""
    assert history[-1]["role"] == "assistant"
    assert "Meta Model API key not configured" in history[-1]["content"]
    assert "MODEL_API_KEY" in history[-1]["content"]


def test_missing_key_message_names_the_custom_provider(
    llm_env,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KOUMBA_BASE_URL",
        "https://api.fireworks.ai/inference/v1",
    )
    monkeypatch.setenv("KOUMBA_API_KEY_ENV", "FIREWORKS_API_KEY")

    import koumba.app as app

    message = app._missing_key_message()

    assert "Fireworks key not configured" in message
    assert "FIREWORKS_API_KEY" in message


def test_trip_cards_include_the_useful_fields() -> None:
    from koumba.app import _trip_cards

    html = _trip_cards(
        [
            {
                "transfers": 0,
                "total_minutes": 40.0,
                "legs": [
                    {
                        "operator": "Dakar Dem Dikk",
                        "line": "Ligne 7",
                        "board_stop": "Ouakam",
                        "alight_stop": "Palais 2",
                        "direction": "Palais 2",
                        "num_stops": 7,
                        "minutes": 40.0,
                    }
                ],
                "walks": [],
            }
        ]
    )

    assert "Dakar Dem Dikk Ligne 7" in html
    assert "Ouakam" in html
    assert "Palais 2" in html
    assert "40 min" in html

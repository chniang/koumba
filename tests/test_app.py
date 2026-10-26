"""The Gradio app must import and build without a key or a model call."""

from __future__ import annotations


def test_app_builds_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    import koumba.app as app

    monkeypatch.setattr(app, "_api_key_configured", lambda: False)

    demo = app.build_app()

    assert demo is not None


def test_respond_without_key_shows_a_clear_message(monkeypatch) -> None:
    import koumba.app as app

    monkeypatch.setattr(app, "_api_key_configured", lambda: False)

    def should_not_run():
        raise AssertionError("the agent must not start without a key")

    monkeypatch.setattr(app, "_agent", should_not_run)

    history, message, _trips, _thoughts = app._respond("Maa ngi Sahm", [])

    assert message == ""
    assert history[-1]["role"] == "assistant"
    assert "Meta API key not configured" in history[-1]["content"]


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

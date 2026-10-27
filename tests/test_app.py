"""The Gradio app must import and build without a key or a model call."""

from __future__ import annotations

from koumba.agent import AgentResult


class _FakeAgent:
    def __init__(self, results: list[AgentResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, list[dict] | None, str | None]] = []

    def run(
        self,
        text: str,
        history=None,
        image_path: str | None = None,
    ) -> AgentResult:
        self.calls.append((text, history, image_path))
        return self.results.pop(0)


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

    history, message, _trips, _thoughts, image = app._respond(
        "Maa ngi Sahm", []
    )

    assert message == ""
    assert history[-1]["role"] == "assistant"
    assert "Meta Model API key not configured" in history[-1]["content"]
    assert "MODEL_API_KEY" in history[-1]["content"]
    assert image is None


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


def test_respond_passes_history_and_uses_the_spoken_answer(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    spoken = (
        "Yaangi Sahm, duggal ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "Grand Yoff. Lu tollu ci 45 minit."
    )
    tool_calls = [{"round": 1, "name": "find_trips"}]
    result = AgentResult(
        reply="This is the long model answer. Yoff-Village walla Grand Yoff?",
        language="Wolof",
        spoken=spoken,
        trips=[],
        tool_calls=tool_calls,
    )
    fake = _FakeAgent([result])
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    history = [
        {"role": "user", "content": "Maa ngi Sahm"},
        {"role": "assistant", "content": "Yoff-Village walla Grand Yoff ?"},
    ]

    chatbot, message, _trips, thoughts, image = app._respond(
        "Grand Yoff", history
    )

    assert message == ""
    assert fake.calls == [("Grand Yoff", history, None)]
    assert chatbot[-1] == {
        "role": "assistant",
        "content": f"{spoken} Yoff-Village walla Grand Yoff?",
    }
    assert thoughts == tool_calls
    assert image is None


def test_respond_passes_the_bus_photo_and_clears_it(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    fake = _FakeAgent(
        [
            AgentResult(
                reply="Waaw, Tata 38.",
                language="Wolof",
                spoken="Waaw, Tata 38 bi dafay dem PAI.",
            )
        ]
    )
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    chatbot, _message, _trips, _thoughts, image = app._respond(
        "Is this my bus?",
        [],
        "C:/tmp/bus.jpg",
    )

    assert fake.calls == [("Is this my bus?", [], "C:/tmp/bus.jpg")]
    assert chatbot[-1]["content"] == "Waaw, Tata 38 bi dafay dem PAI."
    assert image is None


def test_bus_photo_direct_check_shows_the_trip_card(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    trip = {
        "transfers": 0,
        "total_minutes": 46.8,
        "legs": [
            {
                "operator": "Tata AFTU",
                "line": "Ligne 3",
                "board_stop": "Avenue Blaise Diagne",
                "alight_stop": "Yoff-Village",
                "direction": "Yoff-Village",
                "num_stops": 11,
                "minutes": 39.1,
            }
        ],
        "walks": [],
    }
    result = AgentResult(
        reply="Waaw, Tata 3.",
        language="Wolof",
        spoken="Waaw, Tata 3 bi dafay dem Yoff-Village.",
        bus_check={
            "serves_trip": True,
            "operator": "Tata AFTU",
            "line": "Ligne 3",
            "trip": trip,
        },
    )
    fake = _FakeAgent([result])
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    _chatbot, _message, cards, _thoughts, image = app._respond(
        "Is this my bus?",
        [],
        "C:/tmp/bus.jpg",
    )

    assert "Tata AFTU Ligne 3" in cards
    assert "Avenue Blaise Diagne" in cards
    assert app.HONEST_NOTE_EN in cards
    assert image is None


def test_bus_photo_negative_check_shows_the_alternative_card(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    alternative = {
        "transfers": 0,
        "total_minutes": 40,
        "legs": [
            {
                "operator": "Dakar Dem Dikk",
                "line": "Ligne 7",
                "board_stop": "Ouakam",
                "alight_stop": "Palais 2",
                "direction": "Palais 2",
                "num_stops": 7,
                "minutes": 40,
            }
        ],
        "walks": [],
    }
    result = AgentResult(
        reply="Déedéet, Tata 5.",
        language="Wolof",
        spoken="Déedéet, Tata 5 bi du dem Palais.",
        bus_check={
            "serves_trip": False,
            "operator": "Tata AFTU",
            "line": "Ligne 5",
            "best_alternative": alternative,
            "can_help_part_way": True,
        },
    )
    fake = _FakeAgent([result])
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    _chatbot, _message, cards, _thoughts, _image = app._respond(
        "Is this my bus?",
        [],
        "C:/tmp/bus.jpg",
    )

    assert "Dakar Dem Dikk Ligne 7" in cards
    assert "Part of the way with this bus" in cards
    assert app.HONEST_NOTE_EN in cards


def test_trip_cards_use_the_french_honest_note(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    fake = _FakeAgent(
        [
            AgentResult(
                reply="Voici la ligne 7.",
                language="French",
                spoken="À Ouakam, prends le Dem Dikk 7.",
            )
        ]
    )
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    _chatbot, _message, cards, _thoughts, _image = app._respond(
        "Je suis à Ouakam, je vais au Palais",
        [],
    )

    assert app.HONEST_NOTE_FR in cards
    assert app.HONEST_NOTE_EN not in cards


def test_thinking_panel_refreshes_with_only_the_current_turn(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    first_calls = [{"round": 1, "name": "resolve_place"}]
    fake = _FakeAgent(
        [
            AgentResult(
                reply="First turn",
                language="Wolof",
                tool_calls=first_calls,
            ),
            AgentResult(
                reply="Second turn",
                language="Wolof",
                tool_calls=[],
            ),
        ]
    )
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    history, _message, _trips, first_thoughts, _image = app._respond(
        "Maa ngi Sahm",
        [],
    )
    _history, _message, _trips, second_thoughts, _image = app._respond(
        "Grand Yoff",
        history,
    )

    assert first_thoughts == first_calls
    assert second_thoughts == []
    assert len(fake.calls) == 2


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
    assert (
        "Routes based on official AFTU / Dakar Dem Dikk data and field "
        "corrections. Check with the apprenti."
    ) in html

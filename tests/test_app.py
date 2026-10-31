"""The Gradio app must import and build without a key or a model call."""

from __future__ import annotations

import numpy as np

from koumba.agent import AgentResult


def _component(demo, elem_id: str) -> dict:
    for item in demo.config.get("components", []):
        if item.get("props", {}).get("elem_id") == elem_id:
            return item
    raise AssertionError(f"component {elem_id!r} was not built")


class _FakeAgent:
    def __init__(self, results: list[AgentResult]) -> None:
        self.results = list(results)
        self.calls: list[
            tuple[str, list[dict] | None, str | None, list[str] | None]
        ] = []

    def run(
        self,
        text: str,
        history=None,
        image_path: str | None = None,
        hints: list[str] | None = None,
    ) -> AgentResult:
        self.calls.append((text, history, image_path, hints))
        return self.results.pop(0)


def test_app_builds_without_api_key(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()

    assert demo is not None


def test_mobile_header_replaces_the_provider_banner(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    header = _component(demo, "app-header")["props"]["value"]

    assert "Koumba" in header
    assert "Dalal ak jàmm ! Fan nga bëgga dem ?" in header
    assert "Your Dakar bus assistant" in header
    assert not any(
        "provider" in item.get("props", {}).get("elem_classes", [])
        or "banner" in item.get("props", {}).get("elem_classes", [])
        for item in demo.config.get("components", [])
    )


def test_ui_language_switch_defaults_to_english(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    switch = _component(demo, "ui-language")

    assert switch["type"] == "radio"
    assert switch["props"]["value"] == "EN"
    assert [choice[0] for choice in switch["props"]["choices"]] == ["EN", "FR"]


def test_language_switch_updates_only_interface_labels(llm_env) -> None:
    import koumba.app as app

    updates = app._switch_language("FR")

    assert len(updates) == 13
    assert "Votre assistant bus à Dakar" in updates[0]["value"]
    assert updates[1]["placeholder"] == (
        "Appuie sur le micro et dis où tu es et où tu vas."
    )
    assert updates[3]["label"] == "Parle à Koumba"
    assert updates[5]["label"] == "Où vas-tu ?"
    assert updates[5]["placeholder"] == (
        "Écris : Maa ngi Sahm, dama bëgga dem Malika"
    )
    assert updates[6]["value"] == "Envoyer"
    assert updates[7]["value"] == "Effacer"
    assert updates[8]["label"] == "C'est mon bus ? (photo)"
    assert updates[9]["value"] == "Les photos ne sont pas conservées."
    assert "<strong>Essaie un exemple</strong>" in updates[10]["value"]


def test_empty_chat_is_short_and_voice_player_starts_hidden(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    chatbot = _component(demo, "chatbot")
    voice = _component(demo, "voice-player")

    assert chatbot["props"]["height"] == 160
    assert chatbot["props"]["min_height"] == 160
    assert chatbot["props"]["max_height"] == 420
    assert chatbot["props"]["placeholder"] == (
        "Tap the mic and say where you are and where you are going."
    )
    assert voice["props"]["visible"] is False
    assert app.UI_TEXT["English"]["placeholder"] == (
        "Type: Maa ngi Sahm, dama bëgga dem Malika"
    )


def test_css_keeps_mic_clean_and_hides_the_default_footer() -> None:
    import koumba.app as app

    css = app.CSS
    assert "#1F5FAD" in css
    assert "leg-tata" in css
    assert "leg-ddd" in css
    assert ".leg-tata::before" in css
    assert ".leg-ddd::before" in css
    assert "border-left: 7px solid #1F5FAD" not in css
    assert "border-left: 6px solid #1F5FAD" not in css
    assert "#F3FAF5" in css
    assert "#mic .mic-select" in css
    assert "#mic .icon-button-wrapper" in css
    assert 'content: "🎤"' in css
    assert 'content: "■"' in css
    assert "@keyframes koumba-mic-pulse" in css
    assert "#voice-player:has(.empty)" in css
    assert "#chatbot:has(.placeholder-content)" in css
    assert "#ui-language label:has(input:checked)" in css
    assert "footer {" in css
    assert "display: none !important;" in css


def test_voice_player_is_revealed_only_when_audio_exists(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    monkeypatch.setattr(app, "speak", lambda _text: b"RIFF-reply")

    shown, timing, job = app._synthesize_voice_ui(
        {"text": "Maa ngi Sahm", "timings": {}}
    )
    assert shown["value"] == b"RIFF-reply"
    assert shown["visible"] is True
    assert "voice " in timing
    assert job is None

    hidden, _timing, _job = app._synthesize_voice_ui(None)
    assert hidden["value"] is None
    assert hidden["visible"] is False

    empty, _timing, _job = app._synthesize_voice_ui(
        {"text": "", "timings": {}}
    )
    assert empty["visible"] is False


def test_voice_player_stays_hidden_when_tts_fails(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    def fail(_text: str) -> bytes:
        raise app.SpeakError("Voice reply service is unavailable.")

    monkeypatch.setattr(app, "speak", fail)
    monkeypatch.setattr(app.gr, "Warning", lambda *_args, **_kwargs: None)

    update, _timing, job = app._synthesize_voice_ui(
        {"text": "Maa ngi Sahm", "timings": {}}
    )

    assert update["value"] is None
    assert update["visible"] is False
    assert job is None


def test_clear_hides_the_voice_player(llm_env) -> None:
    import koumba.app as app

    values = app._clear_voice_ui("EN")

    assert values[6]["value"] is None
    assert values[6]["visible"] is False


def test_mobile_app_component_order_and_labels(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    order = [
        "app-header",
        "chatbot",
        "voice-player",
        "mic",
        "message",
        "photo-button",
        "trip-cards",
        "technical-panel",
        "footer",
    ]
    positions = [
        next(
            index
            for index, item in enumerate(demo.config.get("components", []))
            if item.get("props", {}).get("elem_id") == elem_id
        )
        for elem_id in order
    ]

    assert positions == sorted(positions)
    assert (
        _component(demo, "photo-button")["props"]["label"]
        == "Is this my bus? (photo)"
    )
    assert _component(demo, "technical-panel")["props"]["open"] is False
    assert _component(demo, "technical-panel")["props"]["label"] == (
        "How Koumba thought"
    )
    assert _component(demo, "models-used")["props"]["value"] == (
        "Meta Omnilingual ASR (speech) · Muse Spark Open (agent, vision) · "
        "Oolel-Voices (Wolof voice)"
    )
    assert app.FOOTER_LINE in _component(demo, "footer")["props"]["value"]


def test_examples_include_three_tap_targets(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    values = [
        item["props"].get("value")
        for item in demo.config.get("components", [])
        if item["type"] == "button"
        and "example-chip" in item.get("props", {}).get("elem_classes", [])
    ]

    assert values == [
        "Maa ngi Sahm, dama bëgga dem Malika",
        "Maa ngi Médina, dama bëgga dem Yoff",
        "Je suis à Ouakam, je vais au Palais",
    ]


def test_app_auto_sends_only_from_the_microphone_stop_event(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    dependencies = demo.config.get("dependencies", [])
    auto_send = [
        item
        for item in dependencies
        if item.get("api_name") == "_respond_to_recording"
    ]

    assert len(auto_send) == 1
    assert [target[1] for target in auto_send[0]["targets"]] == [
        "stop_recording"
    ]


def test_each_answer_event_chains_the_voice_phase(llm_env) -> None:
    import koumba.app as app

    demo = app.build_app()
    dependencies = demo.config.get("dependencies", [])
    voice_phases = [
        item
        for item in dependencies
        if str(item.get("api_name") or "").startswith("_synthesize_voice")
    ]

    assert len(voice_phases) == 6
    assert all(
        [target[1] for target in item["targets"]] == ["then"]
        for item in voice_phases
    )


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
        "Grand Yoff."
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
    assert fake.calls == [("Grand Yoff", history, None, None)]
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

    assert fake.calls == [("Is this my bus?", [], "C:/tmp/bus.jpg", None)]
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
    assert "op-tata" in cards
    assert ">TATA<" in cards
    assert "leg-tata" in cards
    assert "livery-tata" not in cards
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


def test_trip_cards_follow_the_ui_switch_not_the_answer_language(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    result = AgentResult(
        reply="Voici la ligne 7.",
        language="French",
        spoken="À Ouakam, prends le Dem Dikk 7.",
    )
    fake = _FakeAgent([result, result])
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    _chatbot, _message, cards, _thoughts, _image = app._respond(
        "Je suis à Ouakam, je vais au Palais",
        [],
    )

    assert app.HONEST_NOTE_EN in cards
    assert app.HONEST_NOTE_FR not in cards

    _chatbot, _message, cards, _thoughts, _image = app._respond(
        "Je suis à Ouakam, je vais au Palais",
        [],
        None,
        "FR",
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
    assert "7 stops" in html
    assert "~40 min without traffic" in html
    assert "about 40 min" not in html
    assert "best-trip" in html
    assert ">Best<" in html
    assert "op-dem" in html
    assert ">DEM DIKK<" in html
    assert "leg-ddd" in html
    assert "livery-ddd" not in html
    assert "Board" in html
    assert "Alight" in html
    assert (
        "Routes based on official AFTU / Dakar Dem Dikk data and field "
        "corrections. Check with the apprenti."
    ) in html


def test_transfer_trip_uses_each_operator_livery() -> None:
    from koumba.app import _trip_cards

    html = _trip_cards(
        [
            {
                "transfers": 1,
                "total_minutes": 42,
                "legs": [
                    {
                        "operator": "Tata AFTU",
                        "line": "Ligne 3",
                        "board_stop": "Sahm",
                        "alight_stop": "Médina",
                        "direction": "Médina",
                        "num_stops": 6,
                    },
                    {
                        "operator": "Dakar Dem Dikk",
                        "line": "Ligne 7",
                        "board_stop": "Médina",
                        "alight_stop": "Malika",
                        "direction": "Malika",
                        "num_stops": 9,
                    },
                ],
                "walks": [],
            }
        ]
    )

    assert "livery-tata" not in html
    assert "livery-ddd" not in html
    assert "leg-tata" in html
    assert "leg-ddd" in html
    assert ">TATA<" in html
    assert ">DEM DIKK<" in html
    assert ">Best<" in html


def test_trip_cards_use_french_stop_count_and_time_hint() -> None:
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
                    }
                ],
                "walks": [],
            }
        ],
        language="French",
    )

    assert "7 arrêts" in html
    assert "~40 min hors embouteillages" in html
    assert "7 stops" not in html


def test_same_name_walk_uses_the_destination_phrase() -> None:
    from koumba.app import _trip_cards

    trip = {
        "transfers": 0,
        "total_minutes": 46.8,
        "legs": [],
        "walks": [
            {
                "kind": "egress",
                "from_place": "Yoff-Village",
                "to_place": "Yoff-Village",
                "minutes": 5.8,
            }
        ],
    }

    french = _trip_cards([trip], language="French")
    english = _trip_cards([trip], language="English")

    assert "sortie : à pied jusqu'à destination (6 min)" in french
    assert "Yoff-Village &rarr; Yoff-Village" not in french
    assert "exit: walk to destination (6 min)" in english


def test_non_same_walk_keeps_the_two_places() -> None:
    from koumba.app import _trip_cards

    trip = {
        "transfers": 0,
        "total_minutes": 46.8,
        "legs": [],
        "walks": [
            {
                "kind": "access",
                "from_place": "Médina",
                "to_place": "Avenue Blaise Diagne",
                "minutes": 1.9,
            }
        ],
    }

    french = _trip_cards([trip], language="French")

    assert "entrée : Médina &rarr; Avenue Blaise Diagne (2 min)" in french


def test_voice_flow_uses_raw_asr_and_shows_hints(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    raw = "Maa ngi Sam, bëgg naa dem Malika"
    hints = ["Sahm (from 'Sam')"]
    transcription = {
        "text": raw,
        "raw_text": raw,
        "language": "Wolof",
        "asr_lang": "wol_Latn",
        "hints": hints,
        "asr_model": "omniASR_CTC_1B",
        "models": {"transcription": "Omnilingual ASR on Modal"},
        "timings": {"asr": 1.25},
    }
    tool_calls = [{"round": 1, "name": "find_trips"}]
    fake = _FakeAgent(
        [
            AgentResult(
                reply="Long answer.",
                language="Wolof",
                spoken="Yaangi Sahm, duggal ci Tata 38 bi jëm Malika.",
                tool_calls=tool_calls,
                llm_calls=1,
            )
        ]
    )
    monkeypatch.setenv("KOUMBA_ASR_URL", "https://example.modal.run")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_transcribe_audio", lambda _audio: transcription)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(app, "speak", lambda _text: b"RIFF-reply")

    (
        chatbot,
        message,
        _cards,
        thoughts,
        image,
        audio,
        reply_audio,
    ) = app._respond_with_audio(
        "",
        [],
        "C:/tmp/bus.jpg",
        (16_000, np.zeros(16_000, dtype=np.int16)),
    )

    assert chatbot[-2] == {"role": "user", "content": f"🎤 {raw}"}
    assert chatbot[-1]["role"] == "assistant"
    assert fake.calls == [(raw, [], "C:/tmp/bus.jpg", hints)]
    assert message == ""
    assert image is None
    assert audio is None
    assert reply_audio == b"RIFF-reply"

    voice = thoughts[0]["voice"]
    assert voice["raw_asr_text"] == raw
    assert voice["hints"] == hints
    assert voice["transcription_model"] == "Omnilingual ASR on Modal"
    assert "cleaned_text" not in voice
    assert "cleanup_model" not in voice
    assert thoughts[1] == tool_calls[0]


def test_voice_flow_keeps_the_textbox_after_transcription_failure(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    def fail(_audio):
        raise ValueError(
            "The recording is too short. Please speak for at least 0.5 "
            "seconds."
        )

    monkeypatch.setenv("KOUMBA_ASR_URL", "https://example.modal.run")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_transcribe_audio", fail)

    (
        chatbot,
        message,
        _cards,
        _thoughts,
        image,
        audio,
        reply_audio,
    ) = app._respond_with_audio(
        "Type this instead",
        [],
        None,
        (16_000, np.zeros(100, dtype=np.int16)),
    )

    assert chatbot[-1]["role"] == "assistant"
    assert "too short" in chatbot[-1]["content"]
    assert message == "Type this instead"
    assert image is None
    assert audio is None
    assert reply_audio is None


def test_voice_flow_reports_missing_modal_configuration(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    monkeypatch.setenv("KOUMBA_ASR_URL", "")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "")

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("the agent must not start without voice config")

    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_transcribe_audio", should_not_run)
    monkeypatch.setattr(app, "_agent", should_not_run)

    (
        chatbot,
        message,
        _cards,
        thoughts,
        image,
        audio,
        reply_audio,
    ) = app._respond_with_audio(
        "Type this instead",
        [],
        None,
        (16_000, np.zeros(16_000, dtype=np.int16)),
    )

    assert "Voice not configured" in chatbot[-1]["content"]
    assert message == "Type this instead"
    assert thoughts == []
    assert image is None
    assert audio is None
    assert reply_audio is None


def test_typed_wolof_reply_returns_audio_for_autoplay(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    spoken = "Yaangi Sahm, duggal ci Tata 38 bi jëm Malika."
    trip = {
        "total_minutes": 40,
        "legs": [
            {
                "operator": "Tata AFTU",
                "line": "Ligne 38",
                "board_stop": "Sahm",
                "alight_stop": "Malika",
                "direction": "Malika",
            }
        ],
        "walks": [],
    }
    fake = _FakeAgent(
        [
            AgentResult(
                reply="Long model answer.",
                language="Wolof",
                origin="Sahm",
                spoken=spoken,
                trips=[trip],
            )
        ]
    )
    calls: list[str] = []

    def fake_speak(text: str) -> bytes:
        calls.append(text)
        return b"RIFF-reply"

    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(app, "speak", fake_speak)

    values = app._respond_with_audio("Maa ngi Sahm", [], None, None)

    assert len(values) == 7
    assert values[0][-1]["content"] == spoken
    assert values[5] is None
    assert values[-1] == b"RIFF-reply"
    assert calls == [
        "Yaangi Saam, duggal ci Tataa fanweer ak juróom ñett bi jëm Malika."
    ]


def test_timing_line_names_each_step_and_marks_a_failed_voice() -> None:
    import koumba.app as app

    line = app._format_timings(
        {
            "asr": 9.06,
            "agent": 6.0,
            "agent_calls": 1,
            "voice": 8.44,
        }
    )

    assert line == "ASR 9.1 s · agent 6.0 s (1 LLM call) · voice 8.4 s"
    assert (
        app._format_timings({"voice": 8.44, "voice_failed": True})
        == "voice 8.4 s (failed)"
    )


def test_wake_voice_services_pings_both_health_endpoints(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    monkeypatch.setenv(
        "KOUMBA_ASR_URL",
        "https://asr.example.modal.run/",
    )
    monkeypatch.setenv(
        "KOUMBA_TTS_URL",
        "https://tts.example.modal.run",
    )
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")
    calls: list[dict] = []

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})

    monkeypatch.setattr(app.httpx, "get", fake_get)

    app._wake_voice_services()

    assert [call["url"] for call in calls] == [
        "https://asr.example.modal.run/health",
        "https://tts.example.modal.run/health",
    ]
    assert all(
        call["headers"] == {"X-Koumba-Key": "test-secret"}
        and call["follow_redirects"] is True
        for call in calls
    )


def test_wake_voice_services_ignores_a_sleeping_service(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    monkeypatch.setenv(
        "KOUMBA_ASR_URL",
        "https://asr.example.modal.run",
    )
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")

    def fail(_url, **_kwargs):
        raise app.httpx.ConnectError("still asleep")

    monkeypatch.setattr(app.httpx, "get", fail)

    app._wake_voice_services()


def test_text_phase_returns_before_tts_and_then_adds_voice(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    fake = _FakeAgent(
        [
            AgentResult(
                reply="Long model answer.",
                language="Wolof",
                spoken="Yaangi Sahm, duggal ci Tata 38 bi jëm Malika.",
            )
        ]
    )
    calls: list[str] = []

    def fake_speak(text: str) -> bytes:
        calls.append(text)
        return b"RIFF-reply"

    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(app, "speak", fake_speak)

    values = app._respond_phase_one("Maa ngi Sahm", [], None, None)

    assert len(values) == 9
    _chatbot, _message, _cards, _thoughts, _image = values[:5]
    audio, reply_audio, timing, job = values[5:]
    assert audio is None
    assert reply_audio is None
    assert timing.startswith("agent ")
    assert "voice" not in timing
    assert calls == []
    assert job is not None
    job_text = job["text"]

    reply_audio, timing, job = app._synthesize_voice(job)

    assert reply_audio == b"RIFF-reply"
    assert "voice " in timing
    assert job is None
    assert calls == [job_text]


def test_voice_phase_reports_asr_and_agent_call_count(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    transcription = {
        "text": "Maa ngi Sam, bëgg naa dem Malika",
        "raw_text": "Maa ngi Sam, bëgg naa dem Malika",
        "language": "Wolof",
        "asr_lang": "wol_Latn",
        "hints": ["Sahm (from 'Sam')"],
        "timings": {"asr": 1.25},
    }
    fake = _FakeAgent(
        [
            AgentResult(
                reply="Long answer.",
                language="Wolof",
                spoken="Yaangi Sahm, duggal ci Tata 38 bi jëm Malika.",
                llm_calls=1,
            )
        ]
    )
    monkeypatch.setenv("KOUMBA_ASR_URL", "https://example.modal.run")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_transcribe_audio", lambda _audio: transcription)
    monkeypatch.setattr(app, "_agent", lambda: fake)

    values = app._respond_phase_one(
        "",
        [],
        None,
        (16_000, np.zeros(16_000, dtype=np.int16)),
    )

    assert values[7].startswith("ASR 1.2 s · agent ")
    assert "(1 LLM call)" in values[7]
    assert "cleanup" not in values[7]
    assert values[8] is not None


def test_french_reply_does_not_request_audio(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    fake = _FakeAgent(
        [
            AgentResult(
                reply="Prends le Tata 38.",
                language="French",
                spoken="À Sahm, prends le Tata 38.",
            )
        ]
    )
    calls: list[str] = []
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(
        app,
        "speak",
        lambda text: calls.append(text) or b"RIFF-reply",
    )

    values = app._respond_with_audio(
        "Je suis à Sahm, je vais à Malika",
        [],
        None,
        None,
    )

    assert values[5] is None
    assert values[-1] is None
    assert calls == []


def test_tts_failure_keeps_the_text_answer(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    spoken = "Yaangi Sahm, duggal ci Tata 38 bi jëm Malika."
    fake = _FakeAgent(
        [
            AgentResult(
                reply="Long model answer.",
                language="Wolof",
                spoken=spoken,
            )
        ]
    )
    warnings: list[str] = []

    def fail(_text: str) -> bytes:
        raise app.SpeakError("Voice reply service is unavailable.")

    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(app, "speak", fail)
    monkeypatch.setattr(
        app.gr,
        "Warning",
        lambda message, **_kwargs: warnings.append(message),
    )

    values = app._respond_with_audio("Maa ngi Sahm", [], None, None)

    assert values[0][-1]["content"] == spoken
    assert values[-1] is None
    assert warnings == ["Voice reply service is unavailable."]


def test_missing_tts_configuration_shows_a_clear_warning(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app
    from koumba.speak import SpeakNotConfigured

    fake = _FakeAgent(
        [
            AgentResult(
                reply="Long model answer.",
                language="Wolof",
                spoken="Yaangi Sahm, duggal ci Tata 38 bi jëm Malika.",
            )
        ]
    )
    warnings: list[str] = []

    def fail(_text: str) -> bytes:
        raise SpeakNotConfigured(
            "Voice reply not configured. Add KOUMBA_TTS_URL and "
            "KOUMBA_ASR_KEY to your environment or .env file."
        )

    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(app, "speak", fail)
    monkeypatch.setattr(
        app.gr,
        "Warning",
        lambda message, **_kwargs: warnings.append(message),
    )

    values = app._respond_with_audio("Maa ngi Sahm", [], None, None)

    assert values[0][-1]["content"].startswith("Yaangi Sahm")
    assert values[-1] is None
    assert warnings == [
        "Voice reply not configured. Add KOUMBA_TTS_URL and "
        "KOUMBA_ASR_KEY to your environment or .env file."
    ]


def test_recording_stop_auto_sends_once_per_waveform(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    calls = []
    statuses: list[str] = []
    job = {"text": "Yaangi Saam", "timings": {"agent": 1.0}}

    def fake_phase_one(
        message_text,
        history,
        image_path=None,
        audio=None,
        ui_language=None,
    ):
        calls.append((message_text, history, image_path, audio, ui_language))
        return (
            [{"role": "user", "content": "🎤 Maa ngi Sahm"}],
            "",
            app._trip_cards(None),
            [{"voice": {"raw_asr_text": "Maa ngi Sam"}}],
            None,
            None,
            None,
            "agent 1.0 s",
            job,
        )

    monkeypatch.setattr(app, "_respond_phase_one", fake_phase_one)
    monkeypatch.setattr(
        app.gr,
        "Info",
        lambda message, **_kwargs: statuses.append(message),
    )
    first_audio = (16_000, np.zeros(16_000, dtype=np.int16))
    second_audio = (16_000, np.ones(16_000, dtype=np.int16))

    first = app._respond_to_recording([], None, first_audio, None)
    duplicate = app._respond_to_recording([], None, first_audio, first[-1])
    second = app._respond_to_recording([], None, second_audio, first[-1])

    assert len(calls) == 2
    assert calls[0][0] == ""
    assert calls[0][3] is first_audio
    assert calls[1][3] is second_audio
    assert statuses == [app.LISTENING_STATUS, app.LISTENING_STATUS]
    assert len(first) == 10
    assert first[-2] is job
    assert first[-1] == app._audio_fingerprint(first_audio)
    assert duplicate[-2] is None
    assert duplicate[-1] == first[-1]
    assert second[-1] == app._audio_fingerprint(second_audio)


def test_voice_thoughts_refresh_for_each_voice_turn(
    llm_env,
    monkeypatch,
) -> None:
    import koumba.app as app

    first_raw = "mangisam dama bagadam malika"
    second_raw = "dama nekk medina begga dem yoff"
    first_hints = [
        "Sahm (from 'mangisam')",
        "Malika (from 'malika')",
    ]
    second_hints = [
        "Médina (from 'medina')",
        "Yoff (from 'yoff')",
    ]
    transcriptions = iter(
        [
            {
                "text": first_raw,
                "raw_text": first_raw,
                "language": "Wolof",
                "asr_lang": "wol_Latn",
                "hints": first_hints,
            },
            {
                "text": second_raw,
                "raw_text": second_raw,
                "language": "Wolof",
                "asr_lang": "wol_Latn",
                "hints": second_hints,
            },
        ]
    )
    fake = _FakeAgent(
        [
            AgentResult(
                reply="First turn",
                language="Wolof",
                spoken="First turn",
            ),
            AgentResult(
                reply="Second turn",
                language="Wolof",
                spoken="Second turn",
            ),
        ]
    )
    monkeypatch.setenv("KOUMBA_ASR_URL", "https://example.modal.run")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")
    monkeypatch.setattr(app, "_api_key_configured", lambda: True)
    monkeypatch.setattr(
        app,
        "_transcribe_audio",
        lambda _audio: next(transcriptions),
    )
    monkeypatch.setattr(app, "_agent", lambda: fake)
    monkeypatch.setattr(app, "speak", lambda _text: b"RIFF-reply")

    history, _message, _cards, first_thoughts, _image, _audio, _reply = (
        app._respond_with_audio(
            "",
            [],
            None,
            (16_000, np.zeros(16_000, dtype=np.int16)),
        )
    )
    (
        _history,
        _message,
        _cards,
        second_thoughts,
        _image,
        _audio,
        _reply,
    ) = app._respond_with_audio(
        "",
        history,
        None,
        (16_000, np.ones(16_000, dtype=np.int16)),
    )

    assert first_thoughts[0]["voice"]["raw_asr_text"] == first_raw
    assert first_thoughts[0]["voice"]["hints"] == first_hints
    assert second_thoughts[0]["voice"]["raw_asr_text"] == second_raw
    assert second_thoughts[0]["voice"]["hints"] == second_hints
    assert "cleaned_text" not in first_thoughts[0]["voice"]
    assert "cleanup_model" not in first_thoughts[0]["voice"]
    assert [call[0] for call in fake.calls] == [first_raw, second_raw]

"""Agent tests use a scripted fake client, never a real provider."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import koumba.agent as agent_module
from koumba.agent import Agent, detect_language
from koumba.llm import MissingAPIKeyError, ModelClient


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def _response(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls or [],
                )
            )
        ]
    )


class FakeClient:
    """Returns pre-scripted responses and records every model call."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        messages,
        tools=None,
        tool_choice="auto",
        temperature=0.2,
    ):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
            }
        )
        if not self.responses:
            return _response("No more scripted responses.")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def test_wolof_request_runs_the_real_engine_and_returns_trips(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "find_trips",
                        {"origin": "Sahm", "destination": "Malika"},
                    )
                ]
            ),
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("Maa ngi Sahm, bëgg naa dem Malika")

    assert result.language == "Wolof"
    assert result.origin == "Sahm"
    assert result.destination == "Malika"
    assert result.trips
    assert result.trips[0]["legs"]
    assert result.trips[0]["legs"][0]["operator"] in {
        "Dakar Dem Dikk",
        "Tata AFTU",
    }
    assert result.tool_calls[0]["name"] == "find_trips"
    assert result.tool_calls[0]["result"]["trips"] == result.trips
    assert result.spoken.startswith("Yaangi Sahm, duggal ci Tata 38 bi jëm")
    assert result.spoken.endswith("wàcci Malika.")
    assert result.reply == result.spoken
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_raw_asr_spoken_form_resolves_to_the_real_origin(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "find_trips",
                        {"origin": "saam", "destination": "malika"},
                    )
                ]
            ),
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("ma ngi saam dama bëgga dem malika")

    assert result.origin == "Sahm"
    assert result.destination == "Malika"
    assert result.trips
    assert result.spoken.startswith("Yaangi Sahm")
    assert result.llm_calls == 1


def test_unknown_place_returns_suggestions_and_asks_a_question(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "find_trips",
                        {"origin": "Atlantis", "destination": "Malika"},
                    )
                ]
            ),
            _response(
                "Je ne trouve pas Atlantis. Voulez-vous dire Palais ?"
            ),
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("Je suis à Atlantis, je vais à Malika")

    assert result.language == "French"
    assert result.trips is None
    assert result.tool_calls
    tool_result = result.tool_calls[0]["result"]
    assert tool_result["error"] == "place_not_found"
    assert tool_result["suggestions"]
    assert "?" in result.reply
    assert result.llm_calls == 2
    assert client.call_count == 2


def test_agent_stops_after_five_tool_rounds(engine) -> None:
    responses = [
        _response(
            tool_calls=[
                _tool_call(
                    f"call_{index}",
                    "resolve_place",
                    {"query": "Sahm"},
                )
            ]
        )
        for index in range(5)
    ]
    client = FakeClient(responses)
    agent = Agent(client, engine)

    result = agent.run("Maa ngi Sahm")

    assert client.call_count == 5
    assert result.rounds == 5
    assert result.llm_calls == 5
    assert result.reply
    assert len(result.tool_calls) == 5


@pytest.mark.parametrize(
    "text",
    [
        "dama nekk medina beugg dem yoff",
        "maa ngi Sahm begg naa dem Malika",
        "dama bëgga dem",
        "fan la",
        "nekk",
        "beugg",
        "dama",
        "dem",
        "bëgga",
    ],
)
def test_informal_wolof_is_detected(text: str) -> None:
    assert detect_language(text) == "Wolof"


def test_history_keeps_the_last_six_text_messages_only(engine) -> None:
    client = FakeClient([_response("Short answer.")])
    agent = Agent(client, engine)
    history = [
        {"role": "system", "content": "drop the system message"},
        {"role": "user", "content": "old 1"},
        {"role": "assistant", "content": "old 2"},
        {"role": "tool", "content": "drop the tool message"},
        {"role": "user", "content": "old 3"},
        {"role": "assistant", "content": "old 4"},
        {"role": "user", "content": "old 5"},
        {"role": "assistant", "content": "old 6"},
        {"role": "user", "content": "old 7"},
        {"role": "assistant", "content": "old 8"},
        {"role": "user", "content": None},
        {"role": "assistant", "content": ["not", "text"]},
    ]

    agent.run("Grand Yoff", history=history)

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:-1] == [
        {"role": "user", "content": "old 3"},
        {"role": "assistant", "content": "old 4"},
        {"role": "user", "content": "old 5"},
        {"role": "assistant", "content": "old 6"},
        {"role": "user", "content": "old 7"},
        {"role": "assistant", "content": "old 8"},
    ]
    assert messages[-1] == {"role": "user", "content": "Grand Yoff"}


def test_place_hints_are_added_to_the_agent_context(engine) -> None:
    client = FakeClient([_response("Short answer.")])
    agent = Agent(client, engine)

    result = agent.run(
        "Maa ngi Sam, bëgg naa dem Malika",
        hints=["Sahm (from 'Sam')"],
    )

    content = client.calls[0]["messages"][-1]["content"]
    assert content == (
        "Maa ngi Sam, bëgg naa dem Malika\n\n"
        "possible places heard: Sahm (from 'Sam')"
    )
    assert result.language == "Wolof"
    assert result.llm_calls == 1


def test_ambiguous_resolve_place_uses_the_clarification_template(
    engine,
    monkeypatch,
) -> None:
    def fake_execute_tool(name, arguments, _engine, _place_index):
        assert name == "resolve_place"
        return {
            "query": arguments["query"],
            "candidates": [
                {
                    "name": "Yoff-Village",
                    "normalized": "yoff village",
                    "score": 92.0,
                },
                {
                    "name": "Grand Yoff",
                    "normalized": "grand yoff",
                    "score": 91.0,
                },
            ],
        }

    monkeypatch.setattr(agent_module, "execute_tool", fake_execute_tool)
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "resolve_place",
                        {"query": "Yoff"},
                    )
                ]
            )
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("Maa ngi Yoff, bëgg naa dem")

    assert result.reply == "Yoff-Village nga bëgga dem walla Grand Yoff ?"
    assert result.spoken == result.reply
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_ambiguous_tool_result_asks_the_validated_question(
    engine,
    monkeypatch,
) -> None:
    def fake_execute_tool(name, arguments, _engine, _place_index):
        assert name == "find_trips"
        return {
            "error": "ambiguous_place",
            "ambiguous": True,
            "query": arguments["origin"],
            "options": ["Sam Notaire", "Sahm"],
        }

    monkeypatch.setattr(agent_module, "execute_tool", fake_execute_tool)
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "find_trips",
                        {"origin": "sam", "destination": "Malika"},
                    )
                ]
            )
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("ma ngi sam dama bëgga dem malika")

    assert result.reply == "Sam Notaire nga bëgga dem walla Sahm ?"
    assert result.spoken == result.reply
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_follow_up_uses_history_and_keeps_the_earlier_language(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "find_trips",
                        {"origin": "Sahm", "destination": "Grand Yoff"},
                    )
                ]
            ),
        ]
    )
    history = [
        {
            "role": "user",
            "content": "Maa ngi Sahm, bëgg naa dem Yoff-Village walla Grand Yoff?",
        },
        {"role": "assistant", "content": "Yoff-Village walla Grand Yoff ?"},
    ]
    agent = Agent(client, engine)

    result = agent.run("Grand Yoff", history=history)

    first_messages = client.calls[0]["messages"]
    assert first_messages[1:3] == history
    assert first_messages[-1] == {"role": "user", "content": "Grand Yoff"}
    assert result.language == "Wolof"
    assert result.destination == "Grand Yoff"
    assert result.trips
    assert result.spoken.startswith("Yaangi Sahm, duggal ci Tata 38")
    assert result.spoken.endswith(".")
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_official_place_name_is_used_in_the_spoken_answer(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "find_trips",
                        {"origin": "medina", "destination": "Yoff"},
                    )
                ]
            ),
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("Maa ngi medina, bëgg naa dem Yoff")

    assert result.origin == "Médina"
    assert result.spoken.startswith(
        "Yaangi Médina, doxal ba Avenue Blaise Diagne, duggal ci Tata 3"
    )
    assert result.trips[0]["legs"][0]["board_stop"] == "Avenue Blaise Diagne"
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_bus_check_yes_uses_the_short_spoken_answer(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "check_bus_line",
                        {
                            "operator": "Tata",
                            "line": "38",
                            "origin": "Sahm",
                            "destination": "PAI",
                        },
                    )
                ]
            ),
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("Maa ngi Sahm, Tata 38 bi dafay dem PAI?")

    assert result.tool_calls[0]["result"]["serves_trip"] is True
    assert result.bus_check["trip"]["legs"][0]["line"] == "Ligne 38"
    assert result.spoken == (
        "Waaw, Tata 38 bi dafay dem PAI. Duggal ci, wàcci PAI."
    )
    assert result.reply == result.spoken
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_bus_check_no_speaks_the_alternative(engine) -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "call_1",
                        "check_bus_line",
                        {
                            "operator": "Tata",
                            "line": "38",
                            "origin": "Sahm",
                            "destination": "Malika",
                        },
                    )
                ]
            ),
        ]
    )
    agent = Agent(client, engine)

    result = agent.run("Maa ngi Sahm, Tata 38 bi dafay dem Malika?")

    assert result.spoken.startswith("Déedéet, Tata 38 bi du dem Malika.")
    assert "Tata 80" in result.spoken
    assert result.spoken.endswith("wàcci Malika.")
    assert "minit." not in result.spoken
    assert result.bus_check["best_alternative"]["legs"]
    assert result.reply == result.spoken
    assert result.llm_calls == 1
    assert client.call_count == 1


def test_missing_model_api_key_has_a_clear_error(llm_env) -> None:
    with pytest.raises(MissingAPIKeyError, match="MODEL_API_KEY"):
        ModelClient()


def test_blank_model_api_key_counts_as_missing(llm_env, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "   ")

    with pytest.raises(MissingAPIKeyError, match="MODEL_API_KEY"):
        ModelClient()


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.last_request: dict | None = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_request = kwargs
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return "ok"


class _FakeOpenAI:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_model_client_retries_once_and_passes_the_timeout(llm_env) -> None:
    fake = _FakeOpenAI()
    client = ModelClient(
        api_key="test-key",
        client=fake,
        timeout=5.0,
    )

    result = client.complete([{"role": "user", "content": "hello"}])

    assert result == "ok"
    assert fake.completions.calls == 2
    assert fake.completions.last_request is not None
    assert fake.completions.last_request["timeout"] == 5.0
    assert fake.completions.last_request["model"] == "muse-spark-1.3"

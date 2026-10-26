"""Agent tests use a scripted fake client, never the real Meta API."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from koumba.agent import Agent
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
            _response(
                "Maa ngi jàngal: Tata AFTU Ligne 38 + Ligne 80, "
                "1 transfer, about 80 min."
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
    assert client.call_count == 2

    # The second model call must contain the real tool output.
    second_call = client.calls[1]["messages"]
    tool_messages = [
        message for message in second_call if message.get("role") == "tool"
    ]
    assert tool_messages
    assert tool_messages[-1]["tool_call_id"] == "call_1"
    assert "trips" in tool_messages[-1]["content"]


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
    assert result.reply
    assert len(result.tool_calls) == 5


def test_missing_model_api_key_has_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr("koumba.llm.load_dotenv", lambda *args, **kwargs: None)

    with pytest.raises(MissingAPIKeyError, match="MODEL_API_KEY"):
        ModelClient()


def test_blank_model_api_key_counts_as_missing(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "   ")
    monkeypatch.setattr("koumba.llm.load_dotenv", lambda *args, **kwargs: None)

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


def test_model_client_retries_once_and_passes_the_timeout(monkeypatch) -> None:
    monkeypatch.delenv("KOUMBA_MODEL", raising=False)
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

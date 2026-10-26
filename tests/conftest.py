import sys
from pathlib import Path

import pytest

# Make the repository root importable when tests are run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koumba.data import load_data
from koumba.network import Network
from koumba.places import PlaceIndex
from koumba.router import RouteEngine


@pytest.fixture(scope="module")
def engine() -> RouteEngine:
    """A real-data engine shared by the agent and tool tests."""

    data = load_data()
    network = Network.from_data(data)
    place_index = PlaceIndex.from_data(data)
    return RouteEngine(network, place_index)


_LLM_ENV_VARS = (
    "MODEL_API_KEY",
    "KOUMBA_BASE_URL",
    "KOUMBA_API_KEY_ENV",
    "KOUMBA_MODEL",
    "BASETEN_API_KEY",
    "FIREWORKS_API_KEY",
)


@pytest.fixture
def llm_env(monkeypatch) -> None:
    """Isolate provider tests from the developer's environment and .env."""

    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("koumba.llm.load_dotenv", lambda *args, **kwargs: None)

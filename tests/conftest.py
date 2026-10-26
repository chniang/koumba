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

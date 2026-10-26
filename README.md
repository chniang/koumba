# Koumba

Koumba is a small bus assistant for Dakar. It finds direct trips first, then
trips with up to two transfers, using the Dakar Dem Dikk and Tata AFTU lines
from the public dataset. A Muse Spark agent can understand a request in Wolof,
French, or English and call the route engine as tools.

## Built during the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026)

This repository contains:

- `koumba/data.py` - cached download and loading of the four public CSVs.
- `koumba/places.py` - accent-insensitive, abbreviation-aware fuzzy matching
  for stops and neighbourhoods.
- `koumba/network.py` - the `(line, direction, stop)` graph, including
  same-name and 300 m walking transfers.
- `koumba/router.py` - Dijkstra with at most two transfers, direct preference,
  detour control, and distinct line combinations.
- `koumba/cli.py` - the readable command line output.
- `koumba/llm.py` - a small OpenAI-SDK wrapper for the Meta Model API.
- `koumba/tools.py` - JSON tool schemas and handlers for place resolution and
  trip search.
- `koumba/agent.py` - the Muse Spark tool-calling loop.
- `koumba/app.py` - the Gradio demo UI.
- `.env.example` - the two settings the app needs, with no real key.
- `tests/` - route tests, tool tests, and fake-client agent tests.

## Pre-existing

Only the dataset [TIJAANI/dakar-bus-network](https://huggingface.co/datasets/TIJAANI/dakar-bus-network)
was compiled before the hackathon. All code in this repository was written
from scratch during the hackathon.

## Requirements

- Python 3.12 (tested with Python 3.12.5 on Windows)
- `pandas==3.1.0`
- `rapidfuzz==3.14.6`
- `huggingface_hub==2.3.0`
- `pytest==9.1.1` for the tests
- `openai==3.39.0`
- `python-dotenv==1.2.4`
- `gradio==6.30.0`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the agent and the app, copy `.env.example` to `.env` and fill in your own
key:

```powershell
Copy-Item .env.example .env
# Then edit .env:
# MODEL_API_KEY=...
# KOUMBA_MODEL=muse-spark-1.3
```

`.env` is git-ignored. Never commit a real key. The route engine and CLI work
without a key; only the agent and the Gradio app need one.

## Usage

### Route engine CLI

Find up to three distinct trips:

```powershell
python -m koumba.cli "Rond-Point Sahm" "Malika"
python -m koumba.cli "Ouakam" "Palais"
python -m koumba.cli "Parcelles Assainies" "UCAD"
python -m koumba.cli "Guediawaye" "Palais"
```

Other options:

```powershell
python -m koumba.cli --help
python -m koumba.cli --schema
python -m koumba.cli "Ouakam" "Palais" --max-trips 5
```

The output shows the matched place names, the operator and line for each leg,
the boarding and alighting stops, the direction shown on the bus, the number
of stops, estimated minutes, transfer walks, and the total distance when the
coordinates are complete.

### Agent from Python

```python
from koumba.agent import Agent
from koumba.data import load_data
from koumba.llm import ModelClient
from koumba.network import Network
from koumba.places import PlaceIndex
from koumba.router import RouteEngine

data = load_data()
engine = RouteEngine(Network.from_data(data), PlaceIndex.from_data(data))
agent = Agent(ModelClient(), engine)

result = agent.run("Maa ngi Sahm, bëgg naa dem Malika")
print(result.reply)
print(result.language, result.origin, result.destination)
print(result.trips)
```

This example uses the real API only when `MODEL_API_KEY` is set. The test
suite uses a scripted fake client instead and never calls the Meta API.

### Gradio app

```powershell
python -m koumba.app
```

The app has a “Where are you going?” text box, Wolof and French example
buttons, a chat area, mobile-friendly trip cards, and a “How Koumba thought”
accordion with the tool calls. If `MODEL_API_KEY` is missing, it shows the
banner “Meta API key not configured” instead of crashing. Importing
`koumba.app` or calling `build_app()` only builds the UI; it does not start a
server.

## Data source and cache

The engine downloads these four files from the Hugging Face dataset
`TIJAANI/dakar-bus-network` on first use:

- `lignes_dakar.csv` - line metadata: operator, line name, termini, category.
- `itineraires_dakar.csv` - the ordered stops for every line.
- `arrets_geo.csv` - stop coordinates, neighbourhood, commune, and confidence.
- `lieux_geo.csv` - additional neighbourhood and landmark coordinates.

Files are cached in `data_cache/` and reused without downloading again. Set
`KOUMBA_DATA_DIR` to use a different cache folder. Missing coordinates do not
stop the route search; those segments simply have an unknown distance.

## Architecture

1. `data.py` loads and validates the four CSV tables.
2. `places.py` normalizes names (accents, punctuation, common abbreviations)
   and returns *all* plausible stop and place candidates with scores.
3. `network.py` builds a directed graph. A node is a line, a direction, and a
   stop position. Ride edges connect consecutive stops in both directions.
   Transfer edges connect different lines at the same normalized stop name,
   or at stops within 300 m of each other.
4. `router.py` runs Dijkstra over `(node, transfers used)` states. Each label
   remembers the line combination and the steps taken. The search keeps a
   small set of labels per state and never rides the same line twice. An
   optional `operator` argument keeps every leg on one company.
5. `cli.py` turns the result into plain-English legs and walks.
6. `tools.py` exposes `resolve_place` and `find_trips` as JSON tools and turns
   route errors into JSON errors.
7. `agent.py` sends the user message to the model, runs requested tools
   against the real engine, and repeats for at most five tool rounds.
8. `app.py` renders the chat, trip cards, and tool-call accordion.

Ride time uses 250 m/min (15 km/h); a segment with a missing coordinate uses
3 minutes as a fallback. Transfers add a 6-minute wait, and walking uses
80 m/min. Distances are straight-line (haversine) distances between stop
coordinates.

## Route algorithm

- Direct trips are preferred over trips with transfers, even when a transfer
  is slightly faster.
- At most two transfers are allowed, and each result must use a distinct line
  combination.
- The search first visits access stops within 700 m of the origin and last
  visits egress stops within 700 m of the destination. Exact normalized stop
  names board at the stop itself.
- A trip is marked as a detour when its total distance is more than 1.8 times
  the straight-line distance between the origin and destination. Detours are
  rejected when a non-detour alternative exists.
- An extra downtown guard rejects trips that cross Plateau/Palais-style
  downtown stops as a shortcut between two outer neighbourhoods. A trip is
  allowed to start or end downtown.
- A trip more than three times slower than the best remaining option is not
  shown as a useful alternative. This also catches bad routes whose
  coordinates are missing, so the distance-based detour check cannot run.

## Agent and tools

`ModelClient` uses the OpenAI SDK with:

- base URL `https://api.meta.ai/v1`
- `MODEL_API_KEY` from the environment or `.env`
- `KOUMBA_MODEL`, defaulting to `muse-spark-1.3`
- a 30-second timeout and one retry
- a clear `MissingAPIKeyError` when no key is configured

The two tools are:

- `resolve_place(query)` - returns the best stop/place candidates with scores.
- `find_trips(origin, destination, max_trips=3, operator=null)` - returns
  trips as plain JSON with legs, walks, transfers, and total minutes. The
  optional operator must be `Dakar Dem Dikk`, `Tata AFTU`, or null. Place and
  route errors come back as JSON with suggestions instead of crashing.

The agent system prompt tells Muse Spark to understand Wolof, French, English,
and mixed speech; to use the tools; never to invent a line or stop; to ask a
short question when a place is ambiguous; and to answer briefly in the user’s
language. Each result contains `reply`, `language`, `origin`, `destination`,
`trips`, `tool_calls`, and the number of `rounds` used. The language badge is
a small keyword heuristic; the model still writes the reply.

## Assumptions

- Same normalized stop names are the same physical stop.
- A transfer is possible at the same stop or within a 300 m walk.
- The synthetic tests use straight-line distances and simple line geometry.
- Missing coordinates are treated as unknown, not as zero.
- The user wants the best few sensible options, not every mathematical path.
- The model is allowed to choose tool arguments; the handlers validate them.
- Place matching is conservative on purpose.

## Known limitations

- The dataset is incomplete. Many lines only list their termini, and 145 stops
  have no coordinates. Some coordinates are marked `estimee` or `douteuse`.
- The model uses straight-line distances and fixed average speeds, not live
  traffic, schedules, or fares.
- Proximity transfers can create unrealistic links when stop coordinates are
  wrong. The detour, downtown, and slow-trip filters reduce this, but they do
  not repair the underlying data.
- Place matching is deliberately conservative. An unusual spelling may return
  no match rather than a wrong stop.
- The agent loop is tested with a scripted fake client. Live Muse Spark
  behavior is untested until the Meta account is active.
- The language badge uses a small heuristic and can mislabel very short or
  heavily mixed messages.
- No local machine-learning model is downloaded and there is no Streamlit app.
  The optional agent uses the Meta Model API only when a key is configured.

## Tests

```powershell
python -m pytest -s
```

The suite contains synthetic unit tests for direct trips, transfers, transfer
rejection, detour rejection, and unknown places. The real-data integration
tests cover Sahm to Malika (no downtown/Palais shortcut), Ouakam to Palais,
Parcelles Assainies to UCAD, and Guediawaye to Palais. Tool tests check the
JSON shapes and error paths. Agent tests use a scripted `FakeClient` to cover
a Wolof request with a real engine call, an unknown place with suggestions,
the five-round limit, and the missing-key error. App tests build the UI
without a key and check the trip-card fields. The tests never call the Meta
API.

## Project layout

```text
koumba/       package code
tests/        route, tool, agent, and app tests
data_cache/   downloaded CSVs (ignored by git)
.env.example  configuration template (the real .env is ignored)
```

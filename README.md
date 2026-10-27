# Koumba

Koumba is a small bus assistant for Dakar. It finds direct trips first, then
trips with up to two transfers, using the Dakar Dem Dikk and Tata AFTU lines
from the public dataset. A Muse Spark agent can understand a request in Wolof,
French, or English, call the route engine as tools, check a bus photo with its
vision input, and transcribe a spoken request with Muse Spark Open.

## Built during the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026)

This repository contains:

- `koumba/data.py` - cached download and loading of the four public CSVs.
- `koumba/places.py` - accent-insensitive, abbreviation-aware fuzzy matching
  for stops and neighbourhoods.
- `koumba/network.py` - the `(line, direction, stop)` graph, including
  same-name and 300 m walking transfers.
- `koumba/corrections.py` - loads and applies reviewed field corrections to
  line itineraries.
- `koumba/router.py` - Dijkstra with at most two transfers, direct preference,
  detour control, and distinct line combinations.
- `koumba/cli.py` - the readable command line output.
- `koumba/llm.py` - a small OpenAI-SDK wrapper with configurable base URL,
  key env var, and model.
- `koumba/tools.py` - JSON tool schemas and handlers for place resolution,
  trip search, bus-line checks, and line information.
- `koumba/agent.py` - the Muse Spark tool-calling loop with short conversation
  memory and optional image input.
- `koumba/vision.py` - in-memory image resize and JPEG/base64 encoding.
- `koumba/listen.py` - in-memory microphone WAV preparation and Muse Spark
  Open transcription, with no audio stored.
- `koumba/phrases.py` - deterministic short spoken answers in Wolof, French,
  and English, without travel-time guesses.
- `koumba/app.py` - the Gradio demo UI with a provider banner, voice and
  bus-photo inputs, bus-check cards, and the honest data note.
- `scripts/try_voice.py` - manual real-provider WAV transcription test.
- `data/corrections.json` - human-reviewed field corrections. It starts empty.
- `.env.example` - provider presets and key placeholders, with no real key.
- `tests/` - route, correction, tool, agent, provider, vision, voice, and app
  tests.

## Pre-existing

Only the dataset [TIJAANI/dakar-bus-network](https://huggingface.co/datasets/TIJAANI/dakar-bus-network)
was compiled before the hackathon. All code in this repository was written
from scratch during the hackathon.

## Requirements

- Python 3.12 (tested with Python 3.12.5 on Windows)
- `numpy==1.26.4`
- `pandas==3.1.0`
- `rapidfuzz==3.14.6`
- `huggingface_hub==2.3.0`
- `pytest==9.1.1` for the tests
- `openai==3.39.0`
- `python-dotenv==1.2.4`
- `gradio==6.30.0`
- `Pillow==11.3.0`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the agent and the app, copy `.env.example` to `.env` and fill in your own
key. The default preset is the Meta Model API:

```powershell
Copy-Item .env.example .env
# Then edit .env:
# KOUMBA_BASE_URL=https://api.meta.ai/v1
# KOUMBA_API_KEY_ENV=MODEL_API_KEY
# KOUMBA_MODEL=muse-spark-1.3
# MODEL_API_KEY=...
```

`.env.example` also contains commented presets for Baseten and Fireworks.
Baseten still needs its OpenAI-compatible base URL pasted from the Baseten
docs (marked TODO). `.env` is git-ignored. Never commit a real key. The route
engine and CLI work without a key; only the agent and the Gradio app need one.

## Usage

### Route engine CLI

Find up to three distinct trips:

```powershell
python -m koumba.cli "Rond-Point Sahm" "Malika"
python -m koumba.cli "Médina" "Yoff"
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
print(result.spoken)
print(result.reply)
print(result.language, result.origin, result.destination)
print(result.trips)

bus_photo = agent.run(
    "Is this my bus?",
    image_path="C:/photos/bus.jpg",
)
print(bus_photo.spoken)
```

`Agent.run(user_text, history=None, image_path=None)` accepts an optional local
photo. The photo is resized in memory, JPEG-encoded, and sent to the model as an
OpenAI-compatible `image_url` content part. Earlier turns stay text-only, so old
photos are never re-uploaded. The image is not written to disk by Koumba.

`run()` accepts earlier turns as a list of
`{"role": "user"|"assistant", "content": str}` messages. It keeps the last six
text messages as context, ignores tool messages, and sends them before the new
user message. The Gradio app passes its chat history automatically. The short
`spoken` answer is generated from the best trip, so a follow-up such as
"Grand Yoff" can reuse the earlier origin.

This example uses the real provider only when the configured key env var is
set. The default is `MODEL_API_KEY`. The test suite uses a scripted fake client
instead and never calls a real provider.

### Gradio app

```powershell
python -m koumba.app
```

The app has a “Where are you going?” text box, a “Speak to Koumba (Wolof,
French, English)” microphone/upload box, an optional “Photo of the bus
(optional)” upload/camera box, Wolof and French example buttons, a chat area,
mobile-friendly trip cards, and a “How Koumba thought” accordion with the tool
calls. Voice recordings and photos are not stored. A recording is converted to
a mono 16 kHz, 16-bit WAV in memory, transcribed by Muse Spark Open, shown in
the chat as `🎤 {transcript}`, and passed to the normal agent flow. The audio
input is cleared after the answer.

The app resizes the photo and sends it only for the current turn, then clears
the image input. After a bus photo, `check_bus_line` fills the cards with the
direct trip or with the best alternative; a partial alternative is labelled
“Part of the way with this bus”.

Each bus leg shows its stop count prominently (`11 arrêts` in French, `11
stops` in English). Travel time is no longer spoken, because Dakar traffic
makes estimates misleading. It remains in the JSON for ranking and appears on
the cards only as a small grey hint, such as `~45 min hors embouteillages` or
`~45 min without traffic`.

Small text under the cards says: “Routes based on official AFTU / Dakar Dem
Dikk data and field corrections. Check with the apprenti.” French turns show
the equivalent French note.

Its banner names the configured provider and model, such as “Provider:
Fireworks”; if the configured key env var is missing, it shows a warning such as
“Meta Model API key not configured” instead of crashing. Importing
`koumba.app` or calling `build_app()` only builds the UI; it does not start a
server.

### Manual voice test

```powershell
python scripts/try_voice.py path\to\file.wav
```

This is not part of pytest. It uses the provider configured in `.env`, prints
the transcript, detected language, and the agent's deterministic spoken answer.
The configured model must accept `audio_url` content, such as Muse Spark Open
on Baseten or Fireworks. The script accepts 8-bit, 16-bit, or 32-bit PCM WAV
files and still sends a mono 16 kHz, 16-bit WAV to the provider.

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

## Field corrections

`data/corrections.json` holds small itinerary fixes from a Dakar rider. It is
layered on top of the downloaded CSVs when the network is built, before nodes
and coordinates are prepared. The file starts as an empty list.

Supported operations are:

```json
[
  {
    "op": "remove_stop",
    "operator": "Tata AFTU",
    "line": "Ligne 5",
    "stop": "Rue 39",
    "reason": "The bus no longer serves this stop."
  },
  {
    "op": "replace_stop",
    "operator": "Tata AFTU",
    "line": "Ligne 3",
    "stop": "Avenue Cheikh Antadiop",
    "by": "Avenue Cheikh Anta Diop",
    "reason": "Correct the street spelling."
  },
  {
    "op": "insert_stop_after",
    "operator": "Tata AFTU",
    "line": "Ligne 12",
    "after": "Palais 1",
    "stop": "Example stop",
    "reason": "The bus stops here before turning."
  },
  {
    "op": "truncate_after",
    "operator": "Tata AFTU",
    "line": "Ligne 5",
    "stop": "Petersen",
    "reason": "The bus leaves this street after this stop."
  }
]
```

Every entry needs a `reason`. A correction is skipped with a clear warning if
its line or stop is unknown, its fields are incomplete, its operation is
unsupported, or it would leave a line with fewer than two stops. The loader
never crashes on a bad correction. At startup the console prints:

```text
[corrections] 2 applied, 1 skipped
```

## Architecture

1. `data.py` loads and validates the four CSV tables.
2. `places.py` normalizes names (accents, punctuation, common abbreviations)
   and returns *all* plausible stop and place candidates with scores.
3. `network.py` applies `data/corrections.json`, then builds a directed graph.
   A node is a line, a direction, and a stop position. Ride edges connect
   consecutive stops in both directions. Transfer edges connect different
   lines at the same normalized stop name, or at stops within 300 m of each
   other.
4. `router.py` runs Dijkstra over `(node, transfers used)` states. Each label
   remembers the line combination and the steps taken. The search keeps a
   small set of labels per state and never rides the same line twice. An
   optional `operator` argument keeps every leg on one company.
5. `cli.py` turns the result into plain-English legs and walks.
6. `tools.py` exposes `resolve_place`, `find_trips`, `check_bus_line`, and
   `line_info` as JSON tools and turns route errors into JSON errors.
7. `agent.py` sends the last six history messages and the new user message to
   the model, optionally with one image content part, runs requested tools
   against the real engine, and repeats for at most five tool rounds.
8. `vision.py` opens a local image with Pillow, applies EXIF orientation,
   limits the longest side to 1024 px, and encodes JPEG quality 85 as a base64
   data URL without saving a copy.
9. `listen.py` mixes stereo to mono, resamples to 16 kHz with numpy, trims to
   30 seconds, writes a 16-bit PCM WAV in memory, and sends it to Muse Spark
   Open with an `audio_url` content part. The prompt includes the cached list
   of every dataset stop and place name for spelling.
10. `phrases.py` turns the best trip or bus check into a short deterministic
    spoken answer, including a walk to the first boarding stop, and can build
    the validated Wolof clarification question. No template includes a travel
    time.
11. `app.py` renders the chat, voice and bus-photo inputs, trip cards, the
    check-bus cards, the honest data note, and the current turn's tool calls.

Ride time uses 250 m/min (15 km/h); a segment with a missing coordinate uses
3 minutes as a fallback. Every bus leg is billed for at least 2 minutes.
Transfers add a 6-minute wait, and walking uses 80 m/min. Distances are
straight-line (haversine) distances between stop coordinates.

## Route algorithm

- Trips are scored as `total_minutes + 15 * transfers`. A direct trip is shown
  first when its score is within 15 points of the best trip, so a slightly
  slower direct ride wins without hiding a much faster transfer.
- At most two transfers are allowed, and each result must use a distinct line
  combination.
- Exact normalized stop names board at the stop itself. If a query exactly
  matches a known stop or place, fuzzy candidates are discarded so "Yoff"
  cannot become "Grand Yoff"; spelling variants of the exact name remain
  available.
- Each exact access/egress stop also adds every stop within 600 m as a walking
  option. If the query is only a place, not an exact stop, the older 700 m
  access/egress search remains as a fallback. Duplicate coordinates with
  different names are skipped when they are within 1 m.
- When at least one exact-clean trip with no endpoint walk exists, nearby-walk
  transfer variants are hidden unless they are direct. This keeps exact routes
  such as Sahm to Malika stable while still allowing a nearby direct boarding
  such as Médina to Yoff.
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

`ModelClient` uses the OpenAI SDK with these settings from `.env` or the
environment:

- `KOUMBA_BASE_URL` - OpenAI-compatible base URL, default
  `https://api.meta.ai/v1`
- `KOUMBA_API_KEY_ENV` - name of the variable holding the key, default
  `MODEL_API_KEY`
- `KOUMBA_MODEL` - model name, default `muse-spark-1.3`
- a 30-second timeout and one retry
- a clear `MissingAPIKeyError` naming the configured key env var

The `.env.example` presets cover the Meta Model API, Baseten
(`meta-models/muse-spark-open-1026`, base URL TODO), and Fireworks
(`https://api.fireworks.ai/inference/v1`,
`accounts/fireworks/models/muse-spark-open`). The app banner derives the
provider name from the base URL and model so a switch does not need code
changes.

The four tools are:

- `resolve_place(query)` - returns the best stop/place candidates with scores.
- `find_trips(origin, destination, max_trips=3, operator=null)` - returns
  trips as plain JSON with legs, walks, transfers, and total minutes. The
  optional operator must be `Dakar Dem Dikk`, `Tata AFTU`, or null. Operator
  and line spellings are normalized (`Tata`, `L38`).
- `check_bus_line(operator, line, origin, destination)` - reports whether the
  requested bus serves the trip directly. A direct answer includes the
  `board_stop`, `alight_stop`, `direction`, `minutes`, and full `trip`; a
  negative answer includes the best normal `best_alternative` and whether the
  requested bus can help part of the way.
- `line_info(operator, line)` - returns the ordered stops, both termini, and
  the stop sequence in each direction. Unknown lines return `line_not_found`
  with the closest line names.

Place and route errors come back as JSON with suggestions instead of crashing.

The agent system prompt tells Muse Spark to understand Wolof, French, English,
and mixed speech; to use the tools; never to invent a line or stop; to ask a
short question only when a place is genuinely ambiguous; never to repeat the
user or list every trip in text; and to answer in at most two short sentences.
For a bus photo it must read the operator and line first, ask if they are
unreadable instead of guessing, and use `line_info` when the destination is
unknown.

Each result contains `reply`, `language`, `spoken`, `origin`, `destination`,
`trips`, `tool_calls`, `bus_check`, and the number of `rounds` used. `origin` and
`destination` use the official dataset spelling, such as `Médina` for
`medina`. The language badge is a small keyword heuristic that also recognizes
informal Wolof forms such as `dama`, `nekk`, `beugg`, `bëgga`, `fan la`, and
`dem`.

The app shows the deterministic `spoken` sentence for the best trip or bus
check instead of the model's longer reply. It appends only the model's
clarification question when one was asked. Bus names are shortened for speech
(`Tata 38`, `Dem Dikk 7`). The accordion shows only the tool calls from the
current turn and clears on the next message.

The native-speaker decision is that Dakar traffic makes travel-time estimates
misleading, so no spoken answer includes minutes. The time remains in the JSON
for ranking trips and is shown on cards only as a small grey hint.

The bus-check answers use the validated Wolof forms:

```text
Waaw, Tata 38 bi dafay dem PAI. Duggal ci, wàcci PAI.
Déedéet, Tata 38 bi du dem Malika. Yaangi Sahm, duggal ci Tata 38 ...
```

Normal trip answers use the validated `Yaangi ...` direct/transfer templates.
The sentence ends after the final `wàcci {stop}.` with no time sentence. When
the first leg needs a walk of at least one minute to a differently named stop,
the answer includes it:

```text
Yaangi Médina, doxal ba Avenue Blaise Diagne, duggal ci Tata 3 bi jëm ...
Tu es à Médina : marche jusqu'à Avenue Blaise Diagne, prends le Tata 3 ...
You are at Médina: walk to Avenue Blaise Diagne, take Tata 3 ...
```

The same walk wording is used inside the transfer template and inside the “no,
take this instead” bus-check alternative. A walk under one minute, or a board
stop with the same name as the origin, is not mentioned. French and English
bus-check answers follow the same direct/alternative shape.

## Assumptions

- Same normalized stop names are the same physical stop.
- A transfer is possible at the same stop or within a 300 m walk.
- An exact stop has walking access/egress options within 600 m; a place-only
  query uses 700 m.
- The synthetic tests use straight-line distances and simple line geometry.
- Missing coordinates are treated as unknown, not as zero.
- The user wants the best few sensible options, not every mathematical path.
- The model is allowed to choose tool arguments; the handlers validate them.
- Place matching is conservative on purpose.
- A bus photo is sent to the configured model provider for that turn only and
  is not stored by Koumba.
- A voice recording is sent to the configured provider only for transcription
  and is not stored by Koumba.

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
- Field corrections are only as reliable as their review. Invalid corrections
  are skipped with a warning, but a wrong accepted correction still changes the
  itinerary.
- The agent loop, vision content, and voice transcription are tested with
  scripted fake clients. They have not been tested against a live provider,
  live camera images, or live microphone audio.
- Voice transcription depends on the provider. Background noise, code-switching,
  or a strong accent may produce the wrong place spelling; the user can type or
  retry the recording.
- Photo quality depends on the model; glare, motion blur, or an unreadable
  destination can make the operator or line ambiguous.
- The Baseten preset leaves the base URL as a TODO until it is copied from the
  Baseten docs.
- The language badge uses a small heuristic and can mislabel very short or
  heavily mixed messages.
- No local machine-learning model is downloaded and there is no Streamlit app.
  The optional agent calls the configured OpenAI-compatible provider only when
  its key env var is set.

## Tests

```powershell
python -m pytest -s
```

The suite contains synthetic unit tests for direct trips, transfers, transfer
rejection, detour rejection, score-based direct promotion, the two-minute
minimum bus leg, nearby-stop access, and unknown places. Correction tests cover
all four operations, the empty repository file, unknown lines/stops, missing
reasons, unsupported operations, and a correction applied while a network is
built. The real-data integration tests cover Sahm to Malika (38 then 80),
MǸdina to Yoff (direct nearby boarding), Ouakam to Palais, Parcelles Assainies
to UCAD, Guediawaye to Palais, and exact Yoff/MǸdina matching without Grand
Yoff. Tool tests check the JSON shapes, exact-first place candidates, bus
yes/no answers, closest-line errors, line information, normalization, and error
paths. Agent tests use a scripted `FakeClient` to cover a Wolof request with a
real engine call, official origin spelling, bus-check speech and cards data,
history limits, a follow-up answer, informal Wolof detection, the five-round
limit, and the missing-key error. Vision tests create a synthetic Pillow image
and check resize, JPEG/base64 encoding, and the agent content part. Voice tests
cover stereo mixing, 44.1 kHz to 16 kHz resampling, rejecting recordings under
half a second, the 30-second trim, the `audio_url` data request, and parsing
fenced JSON after model reasoning. Phrase tests check the validated `Yaangi`
and bus-check templates without travel time, the first-stop walk wording,
short bus names, and chat composition. Provider tests check the defaults, the
Fireworks settings, Baseten detection, a custom key env var, and the SDK base
URL/model plumbing. App tests build the UI, check the provider banner, verify
history, photo and voice passing, transcript display, photo/audio clearing,
textbox recovery after transcription failure, bus-photo cards, stop counts,
time hints, the partial-route label, both honest notes, the spoken chat message,
and that the thinking panel only shows the current turn. The tests never call a
real provider.

## Project layout

```text
koumba/       package code
scripts/      manual real-provider voice test
data/         human-reviewed corrections.json
tests/        route, correction, tool, agent, provider, vision, voice, app tests
data_cache/   downloaded CSVs (ignored by git)
.env.example  configuration template (the real .env is ignored)
```

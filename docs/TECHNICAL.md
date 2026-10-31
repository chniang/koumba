# Koumba technical notes

This file holds the developer detail kept out of the main README. For the
judge-facing overview, see [`../README.md`](../README.md).

## Repository layout

- `koumba/data.py` - cached download and loading of the four public CSVs.
- `koumba/places.py` - accent-insensitive, abbreviation-aware fuzzy matching
  for stops and neighbourhoods.
- `koumba/network.py` - the `(line, direction, stop)` graph, including
  same-name and 300 m walking transfers.
- `koumba/corrections.py` - loads and applies reviewed field corrections to
  line itineraries.
- `koumba/router.py` - Dijkstra with at most two transfers, direct preference,
  detour control, and distinct line combinations.
- `koumba/cli.py` - readable command-line route output.
- `koumba/llm.py` - a small OpenAI-SDK wrapper with configurable base URL, key
  environment variable, and model.
- `koumba/tools.py` - JSON tool schemas and handlers for place resolution, trip
  search, bus-line checks, and line information.
- `koumba/agent.py` - the Muse Spark tool-calling loop with short conversation
  memory and optional image input.
- `koumba/vision.py` - in-memory image resize and JPEG/base64 encoding.
- `koumba/listen.py` - in-memory microphone WAV preparation, Modal ASR HTTP
  calls, and local place hints, with no audio stored.
- `koumba/speak.py` - the Modal Oolel-Voices HTTP client for short spoken
  replies.
- `koumba/phrases.py` - deterministic short spoken answers in Wolof, French,
  and English, without travel-time guesses.
- `koumba/voice_text.py` - keeps the displayed Wolof sentence intact and swaps
  bus and place names for their reviewed TTS spelling.
- `koumba/app.py` - the Gradio demo UI, with voice and bus-photo inputs, trip
  cards, spoken Wolof replies, and the honest data note.
- `modal_app/web.py` - the Modal-hosted Gradio demo and its bus-data cache.
- `modal_app/asr.py` - the Modal Omnilingual ASR service and its cache volume.
- `modal_app/tts.py` - the Modal Oolel-Voices service and its cache volume.
- `modal_app/README.md` - secret creation, deployment, and endpoint checks.
- `scripts/try_voice.py` - manual WAV -> Modal ASR -> agent test.
- `data/corrections.json` - human-reviewed field corrections. It starts empty.
- `.env.example` - provider presets and voice endpoint placeholders.
- `tests/` - route, correction, tool, agent, provider, vision, voice, TTS, and
  app tests.

The public dataset `TIJAANI/dakar-bus-network` existed before the hackathon. All
code in this repository was written during the event.

## Requirements

- Python 3.12 (tested with Python 3.12.5 on Windows)
- `numpy==2.5.4`
- `pandas==3.1.0`
- `rapidfuzz==3.14.6`
- `huggingface_hub==2.3.0`
- `pytest==9.1.1` for the tests
- `openai==3.39.0`
- `httpx==0.28.1`
- `python-dotenv==1.2.4`
- `gradio==6.30.0`
- `Pillow==12.3.0`

The `modal` package is needed only to deploy the services in `modal_app/`. The
local Koumba app uses plain HTTP and does not import Modal.

## Configuration

`ModelClient` reads these values from `.env` or the environment:

- `KOUMBA_BASE_URL` - OpenAI-compatible base URL. The default is
  `https://api.meta.ai/v1`; the live demo uses
  `https://inference.baseten.co/v1`.
- `KOUMBA_API_KEY_ENV` - name of the variable holding the key. The default is
  `MODEL_API_KEY`; the live demo uses `BASETEN_API_KEY`.
- `KOUMBA_MODEL` - model name. The default is `muse-spark-1.3`; the live demo
  uses `meta-models/muse-spark-open-1026`.

The client has a 30-second timeout and one retry. A missing key raises
`MissingAPIKeyError` naming the configured key variable.

Voice settings:

- `KOUMBA_ASR_URL` and `KOUMBA_ASR_KEY` - Modal Omnilingual ASR endpoint and
  shared secret.
- `KOUMBA_TTS_URL` - Modal Oolel-Voices endpoint. TTS reuses
  `KOUMBA_ASR_KEY`.
- `KOUMBA_DATA_DIR` - optional alternate cache folder for the CSV files.

Never commit a real `.env`. Only `.env.example` belongs in the repository.

## Data source and cache

The engine downloads these four files from the Hugging Face dataset
`TIJAANI/dakar-bus-network` on first use:

- `lignes_dakar.csv` - line metadata: operator, line name, termini, category.
- `itineraires_dakar.csv` - the ordered stops for every line.
- `arrets_geo.csv` - stop coordinates, neighbourhood, commune, and confidence.
- `lieux_geo.csv` - additional neighbourhood and landmark coordinates.

Files are cached in `data_cache/` and reused without downloading again. Set
`KOUMBA_DATA_DIR` to use a different cache folder.

Missing coordinates do not stop the route search. Those segments simply have an
unknown distance. Coordinates marked `estimee` (estimated) or `douteuse`
(doubtful) are still usable, but they are less reliable.

## Field corrections

`data/corrections.json` holds small itinerary fixes from a Dakar rider. It is
layered on top of the downloaded CSVs when the network is built, before nodes
and coordinates are prepared. The file starts as an empty list.

Supported operations:

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
never crashes on a bad correction. At startup the console prints a summary such
as:

```text
[corrections] 2 applied, 1 skipped
```

## Architecture

1. `data.py` loads and validates the four CSV tables.
2. `places.py` normalizes names (accents, punctuation, common abbreviations)
   and returns all plausible stop and place candidates with scores.
3. `network.py` applies `data/corrections.json`, then builds a directed graph.
   A node is a line, a direction, and a stop position. Ride edges connect
   consecutive stops in both directions. Transfer edges connect different lines
   at the same normalized stop name, or at stops within 300 m of each other.
4. `router.py` runs Dijkstra over `(node, transfers used)` states. Each label
   remembers the line combination and the steps taken. The search keeps a small
   set of labels per state and never rides the same line twice. An optional
   `operator` argument keeps every leg on one company.
5. `cli.py` turns the result into plain-language legs and walks.
6. `tools.py` exposes `resolve_place`, `find_trips`, `check_bus_line`, and
   `line_info` as JSON tools and turns route errors into JSON errors.
7. `agent.py` sends the last six history messages and the new user message to
   the model, optionally with one image content part, runs requested tools
   against the real engine, and repeats for at most five tool rounds. When a
   successful tool result already fills a validated phrase template, it returns
   that answer without a second model call.
8. `vision.py` opens a local image with Pillow, applies EXIF orientation, limits
   the longest side to 1024 px, and encodes JPEG quality 85 as a base64 data URL
   without saving a copy.
9. `listen.py` mixes stereo to mono, resamples to 16 kHz with NumPy, trims to
   30 seconds, writes a 16-bit PCM WAV in memory, and posts it to the Modal
   Omnilingual ASR endpoint with the `X-Koumba-Key` shared secret. It returns
   the raw text plus local fuzzy place hints.
10. `phrases.py` turns the best trip or bus check into a short deterministic
    spoken answer, including a walk to the first boarding stop, and can build
    the validated Wolof clarification question. No template includes a travel
    time.
11. `app.py` renders the chat, voice and bus-photo inputs, trip cards, bus-check
    cards, the honest data note, the current turn's tool calls, and the raw
    voice transcript with hints. The UI has a small EN | FR switch for its own
    labels; it has no provider banner.

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
  matches a known stop or place, fuzzy candidates are discarded so "Yoff" cannot
  become "Grand Yoff"; spelling variants of the exact name remain available.
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
- An extra downtown guard rejects trips that cross Plateau/Palais-style downtown
  stops as a shortcut between two outer neighbourhoods. A trip is allowed to
  start or end downtown.
- A trip more than three times slower than the best remaining option is not
  shown as a useful alternative. This also catches bad routes whose coordinates
  are missing, so the distance-based detour check cannot run.

## Agent and tools

The four tools are:

- `resolve_place(query)` - returns the best stop or place candidates with
  scores.
- `find_trips(origin, destination, max_trips=3, operator=null)` - returns trips
  as plain JSON with legs, walks, transfers, and total minutes. The optional
  operator must be `Dakar Dem Dikk`, `Tata AFTU`, or null. Operator and line
  spellings are normalized (`Tata`, `L38`).
- `check_bus_line(operator, line, origin, destination)` - reports whether the
  requested bus serves the trip directly. A direct answer includes the
  `board_stop`, `alight_stop`, `direction`, `minutes`, and full `trip`; a
  negative answer includes the best normal `best_alternative` and whether the
  requested bus can help part of the way.
- `line_info(operator, line)` - returns the ordered stops, both termini, and the
  stop sequence in each direction. Unknown lines return `line_not_found` with
  the closest line names.

Place and route errors come back as JSON with suggestions instead of crashing.

The agent system prompt tells Muse Spark to understand Wolof, French, English,
and mixed speech; to use the tools; never to invent a line or stop; to ask a
short question only when a place is genuinely ambiguous; never to repeat the
user or list every trip in text; and to answer in at most two short sentences.
For a bus photo it must read the operator and line first, ask if they are
unreadable instead of guessing, and use `line_info` when the destination is
unknown.

Each result contains `reply`, `language`, `spoken`, `origin`, `destination`,
`trips`, `tool_calls`, `bus_check`, and the number of `rounds` used. `origin`
and `destination` use the official dataset spelling, such as `Médina` for
`medina`. The language badge is a small keyword heuristic that also recognizes
informal Wolof forms such as `dama`, `nekk`, `beugg`, `bëgga`, `fan la`, and
`dem`.

The app shows the deterministic `spoken` sentence for the best trip or bus
check instead of the model's longer reply. It appends only the model's
clarification question when one was asked. Bus names are shortened for speech
(`Tata 38`, `Dem Dikk 7`). The accordion shows only the tool calls from the
current turn and clears on the next message.

The native-speaker decision is that Dakar traffic makes travel-time estimates
misleading, so no spoken answer includes minutes. Time remains in the JSON for
ranking trips and is shown on cards only as a small grey hint.

The bus-check answers use the validated Wolof forms:

```text
Waaw, Tata 38 bi dafay dem PAI. Duggal ci, wàcci PAI.
Déedéet, Tata 38 bi du dem Malika. Yaangi Sahm, duggal ci Tata 38 ...
```

Normal trip answers use the validated `Yaangi ...` direct and transfer
templates. The sentence ends after the final `wàcci {stop}.` with no time
sentence. When the first leg needs a walk of at least one minute to a differently
named stop, the answer includes it:

```text
Yaangi Médina, doxal ba Avenue Blaise Diagne, duggal ci Tata 3 bi jëm ...
Tu es à Médina : marche jusqu'à Avenue Blaise Diagne, prends le Tata 3 ...
You are at Médina: walk to Avenue Blaise Diagne, take Tata 3 ...
```

The same walk wording is used inside the transfer template and inside the "no,
take this instead" bus-check alternative. A walk under one minute, or a board
stop with the same name as the origin, is not mentioned. French and English
bus-check answers follow the same direct/alternative shape.

## Voice path

The ASR service runs Meta Omnilingual ASR on Modal CPU by default. The model is
`omniASR_CTC_1B`; `KOUMBA_ASR_MODEL=omniASR_CTC_300M` is the faster CPU option.
The model is loaded once per container and cached in the `koumba-asr-cache`
Volume. The temporary WAV is deleted immediately after inference.

`find_place_hints` compares the raw ASR words with official stop and place names
using rapidfuzz. The agent receives the raw transcript plus a
`possible places heard: ...` context line. The raw transcript is shown in the
chat and kept, with the hints, in the "How Koumba thought" panel.

For a Wolof answer, `voice_text.py` keeps the displayed sentence and swaps bus
and place names for the lexicon's reviewed `say_as` spelling. The app posts that
text to `KOUMBA_TTS_URL/speak` with `X-Koumba-Key: KOUMBA_ASR_KEY`, a
**150-second** client timeout, and redirect following for cold containers.
French and English answers do not request audio.

Oolel-Voices runs on CPU with the official female voice prompt and returns
`audio/wav`. Full sentences are synthesized, not concatenated word clips. The
model is baked into the image and loaded once during the build. The voice prompt
and generated WAVs are cached in the `koumba-tts-cache` Volume. The endpoint
accepts at most 400 characters.

Koumba does not store the TTS text or audio. The TTS service keeps a WAV cache
under `/cache/speech` and never logs the sentence. If TTS is unavailable, the
text answer is still shown.

The public web demo and the three Modal services are described in
[`../modal_app/README.md`](../modal_app/README.md).

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
- A bus photo is sent to the configured model provider for that turn only and is
  not stored by Koumba.
- A voice recording is sent to the Modal ASR endpoint only for transcription.
  The raw text and local place hints go to the agent. The audio is not stored.
- A Wolof text answer is sent to the Modal TTS endpoint only for synthesis.
  Koumba does not store the sentence or audio.

## Test suite

```powershell
python -m pytest -q
```

There are **281 tests**. They use fakes only and never call a real provider,
the live Modal endpoints, or a real microphone.

The suite covers:

- synthetic routes: direct trips, transfers, transfer rejection, detour
  rejection, score-based direct promotion, the two-minute minimum bus leg,
  nearby-stop access, and unknown places;
- corrections: all four operations, the empty repository file, unknown
  lines/stops, missing reasons, unsupported operations, and a correction applied
  while a network is built;
- real-data integration: Sahm to Malika (38 then 80), Médina to Yoff (direct
  nearby boarding), Ouakam to Palais, Parcelles Assainies to UCAD, Guediawaye to
  Palais, and exact Yoff/Médina matching without Grand Yoff;
- tools: JSON shapes, exact-first place candidates, bus yes/no answers,
  closest-line errors, line information, normalization, and error paths;
- agent: a scripted fake client covers a Wolof request with a real engine call,
  official origin spelling, bus-check speech and card data, history limits, a
  follow-up answer, informal Wolof detection, the five-round limit, and the
  missing-key error;
- vision: a synthetic Pillow image checks resize, JPEG/base64 encoding, and the
  agent content part;
- voice input: stereo mixing, 44.1 kHz to 16 kHz resampling, rejecting
  recordings under half a second, the 30-second trim, the Modal URL and
  `X-Koumba-Key` header, raw WAV body, 401 handling, missing voice
  configuration, raw-text retention, local place hints, and inference-time
  propagation;
- TTS client: endpoint URL, JSON body, shared key, 150-second timeout, missing
  configuration, authentication, HTTP and network failures, and empty audio;
- app: Wolof autoplay audio, French silence, warning on missing TTS
  configuration, text-answer survival when TTS fails, UI building, history,
  photo and voice passing, raw transcript display, raw text and hints in the
  thinking panel, missing voice configuration, photo/audio clearing, textbox
  recovery after transcription failure, bus-photo cards, stop counts, time
  hints, the partial-route label, both honest notes, the spoken chat message,
  and that the thinking panel only shows the current turn;
- phrases: validated `Yaangi` and bus-check templates without travel time, the
  first-stop walk wording, short bus names, and chat composition;
- provider settings: defaults, Fireworks settings, Baseten detection, a custom
  key environment variable, and the SDK base URL/model plumbing.

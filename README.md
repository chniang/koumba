# Koumba

Koumba is a small bus route engine for Dakar. It finds direct trips first,
then trips with up to two transfers, using the Dakar Dem Dikk and Tata AFTU
lines from the public dataset.

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
- `tests/` - synthetic unit tests and real-data integration tests.

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

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

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
   small set of labels per state and never rides the same line twice.
5. `cli.py` turns the result into plain-English legs and walks.

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

## Assumptions

- Same normalized stop names are the same physical stop.
- A transfer is possible at the same stop or within a 300 m walk.
- The synthetic tests use straight-line distances and simple line geometry.
- Missing coordinates are treated as unknown, not as zero.
- The user wants the best few sensible options, not every mathematical path.

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
- No machine-learning model, no Streamlit, and no external routing API are
  used.

## Tests

```powershell
python -m pytest -s
```

The suite contains synthetic unit tests for direct trips, transfers, transfer
rejection, detour rejection, and unknown places. The real-data integration
tests cover Sahm to Malika (no downtown/Palais shortcut), Ouakam to Palais,
Parcelles Assainies to UCAD, and Guediawaye to Palais.

## Project layout

```text
koumba/       package code
tests/        synthetic and real-data tests
data_cache/   downloaded CSVs (ignored by git)
```

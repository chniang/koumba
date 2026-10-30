"""Download and load the Dakar bus dataset.

The four CSVs come from the public Hugging Face dataset
``TIJAANI/dakar-bus-network`` (repo_type="dataset").  We only cache the
files here; no model is downloaded.

The real files have these columns:

* lignes_dakar.csv      -> compagnie, ligne, terminus_a, terminus_b,
                           categorie, source
* itineraires_dakar.csv -> compagnie, ligne, ordre, arret
* arrets_geo.csv        -> arret, lat, lon, quartier, commune, confiance
* lieux_geo.csv         -> lieu, lat, lon

The ``confiance`` column is useful: some coordinates are marked
"introuvable" (missing) or "douteuse" (rough).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_ID = "TIJAANI/dakar-bus-network"
FILES = (
    "lignes_dakar.csv",
    "itineraires_dakar.csv",
    "arrets_geo.csv",
    "lieux_geo.csv",
)


@dataclass
class NetworkData:
    """The four data frames, kept separate so each one is easy to inspect."""

    lignes: pd.DataFrame
    itineraires: pd.DataFrame
    arrets: pd.DataFrame
    lieux: pd.DataFrame


def data_dir() -> Path:
    """Where the CSVs live.

    ``KOUMBA_DATA_DIR`` wins, which is handy for tests or for moving the cache.
    Otherwise we use ``<repo>/data_cache``.
    """

    override = os.environ.get("KOUMBA_DATA_DIR")
    if override:
        folder = Path(override).expanduser()
    else:
        folder = Path(__file__).resolve().parent.parent / "data_cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def download_data(force: bool = False) -> dict[str, Path]:
    """Return local paths for the four files, downloading only when needed.

    If a file is already in our cache we do not touch the network, so the
    engine keeps working offline after the first run.
    """

    folder = data_dir()
    paths: dict[str, Path] = {}
    for filename in FILES:
        path = folder / filename
        if path.exists() and not force:
            paths[filename] = path
            continue

        try:
            downloaded = hf_hub_download(
                repo_id=DATASET_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(folder),
            )
        except Exception as exc:  # pragma: no cover - depends on the network
            raise RuntimeError(
                f"Could not download {filename} from {DATASET_ID}. "
                f"Check your connection or put the CSV in {folder}."
            ) from exc
        paths[filename] = Path(downloaded)
    return paths


def _read_csv(path: Path) -> pd.DataFrame:
    # The dataset is UTF-8.  Reading the columns explicitly in the docstring
    # above is deliberate: if the public dataset changes, loading fails fast.
    return pd.read_csv(path, encoding="utf-8")


def load_data(force: bool = False) -> NetworkData:
    """Download (if necessary), read, and lightly validate the dataset."""

    paths = download_data(force=force)
    data = NetworkData(
        lignes=_read_csv(paths["lignes_dakar.csv"]),
        itineraires=_read_csv(paths["itineraires_dakar.csv"]),
        arrets=_read_csv(paths["arrets_geo.csv"]),
        lieux=_read_csv(paths["lieux_geo.csv"]),
    )

    # Reviewed name fixes are applied to the frames, not to data_cache, so
    # every consumer (graph, place index, hints) sees the same stop name.
    from .corrections import canonicalize_data_stop_names

    canonicalize_data_stop_names(data)
    _validate(data)
    return data


def _validate(data: NetworkData) -> None:
    required = {
        "lignes": ("compagnie", "ligne", "terminus_a", "terminus_b"),
        "itineraires": ("compagnie", "ligne", "ordre", "arret"),
        "arrets": ("arret", "lat", "lon", "confiance"),
        "lieux": ("lieu", "lat", "lon"),
    }
    frames = {
        "lignes": data.lignes,
        "itineraires": data.itineraires,
        "arrets": data.arrets,
        "lieux": data.lieux,
    }

    for name, columns in required.items():
        frame = frames[name]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} is empty")

    data.itineraires["ordre"] = pd.to_numeric(
        data.itineraires["ordre"], errors="coerce"
    )
    if data.itineraires["ordre"].isna().any():
        raise ValueError("itineraires has non-numeric order values")

    data.itineraires = data.itineraires.sort_values(
        ["compagnie", "ligne", "ordre"]
    ).reset_index(drop=True)

    for column in ("lat", "lon"):
        data.arrets[column] = pd.to_numeric(data.arrets[column], errors="coerce")
        data.lieux[column] = pd.to_numeric(data.lieux[column], errors="coerce")


def describe_data(data: NetworkData) -> str:
    """A short, human-readable schema summary."""

    frames = (
        ("lignes_dakar.csv", data.lignes),
        ("itineraires_dakar.csv", data.itineraires),
        ("arrets_geo.csv", data.arrets),
        ("lieux_geo.csv", data.lieux),
    )
    lines = []
    for filename, frame in frames:
        lines.append(
            f"{filename}: {len(frame)} rows; columns: {', '.join(frame.columns)}"
        )
        missing = {
            column: int(frame[column].isna().sum())
            for column in frame.columns
            if frame[column].isna().any()
        }
        if missing:
            detail = ", ".join(f"{column}={count}" for column, count in missing.items())
            lines.append(f"  missing values: {detail}")

    if "compagnie" in data.lignes.columns:
        counts = data.lignes["compagnie"].value_counts().to_dict()
        lines.append(f"operators: {counts}")
    if "confiance" in data.arrets.columns:
        counts = data.arrets["confiance"].value_counts(dropna=False).to_dict()
        lines.append(f"coordinate confidence: {counts}")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download and inspect the bus data")
    parser.add_argument("--force", action="store_true", help="re-download the CSVs")
    args = parser.parse_args()

    data = load_data(force=args.force)
    print(describe_data(data))
    print(f"cache: {data_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

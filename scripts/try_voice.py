"""Manual voice test for a real WAV file.

Usage:
    python scripts/try_voice.py path\to\file.wav

This is intentionally not collected by pytest.  It sends real audio to the
Modal ASR endpoint, runs the Muse Spark cleanup, and asks the agent for a
route.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koumba.agent import Agent
from koumba.data import load_data
from koumba.listen import prepare_audio, transcribe, voice_config
from koumba.llm import ModelClient
from koumba.network import Network
from koumba.places import PlaceIndex
from koumba.router import RouteEngine


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8)
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2")
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError("Use an 8-bit, 16-bit, or 32-bit PCM WAV file.")

    if channels > 1:
        samples = samples.reshape(-1, channels)
    return sample_rate, samples


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python scripts/try_voice.py path\\to\\file.wav")
        return 2

    path = Path(args[0]).expanduser()
    if not path.exists():
        print(f"Audio file not found: {path}")
        return 2

    try:
        sample_rate, samples = _read_wav(path)
        wav_bytes = prepare_audio(sample_rate, samples)
    except (ValueError, wave.Error) as exc:
        print(f"Could not read that recording: {exc}")
        return 2

    try:
        voice_config()
        client = ModelClient()
        transcript = transcribe(client, wav_bytes)
    except Exception as exc:
        print(f"Transcription failed: {exc}")
        return 1

    print(f"Raw ASR:    {transcript['raw_text']}")
    print(f"Cleaned:    {transcript['text']}")
    print(f"Language:   {transcript['language']}")
    if transcript.get("places_fixed"):
        print(f"Fixed:      {', '.join(transcript['places_fixed'])}")

    try:
        data = load_data()
        engine = RouteEngine(
            Network.from_data(data),
            PlaceIndex.from_data(data),
        )
        result = Agent(client, engine).run(transcript["text"])
    except Exception as exc:
        print(f"Agent failed: {exc}")
        return 1

    print(f"Spoken:     {result.spoken or '(no spoken answer)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

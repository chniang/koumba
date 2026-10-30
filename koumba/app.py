"""A small Gradio demo for the Koumba agent.

Run it manually with ``python -m koumba.app``.  Importing this module or
calling ``build_app()`` only builds the UI; it never starts a server and
never calls the model.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from functools import lru_cache
from html import escape
from typing import Any

# Keep Gradio from sending usage telemetry.  This is set before the import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import httpx
import numpy as np

from .agent import Agent
from .data import load_data
from .listen import (
    ASR_MODEL_LABEL,
    LISTENING_STATUS,
    VoiceError,
    prepare_audio,
    transcribe,
    voice_config,
)
from .llm import ModelClient, ProviderConfig, api_key_configured, load_config
from .network import Network
from .phrases import compose_chat_message
from .places import PlaceIndex, normalize_name
from .router import RouteEngine
from .speak import SpeakError, speak
from .voice_text import tts_text

EXAMPLE_WOLOF = "Maa ngi Sahm, bëgg naa dem Malika"
EXAMPLE_FRENCH = "Je suis à Ouakam, je vais au Palais"

HONEST_NOTE_EN = (
    "Routes based on official AFTU / Dakar Dem Dikk data and field "
    "corrections. Check with the apprenti."
)
HONEST_NOTE_FR = (
    "Itinéraires basés sur les données officielles AFTU / Dakar Dem Dikk et "
    "des corrections de terrain. Vérifie avec l'apprenti."
)

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
.trip-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 12px;
}
.trip-card {
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.trip-head { font-weight: 700; margin-bottom: 8px; }
.pill {
  display: inline-block;
  font-size: 12px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: #eef2ff;
  color: #3730a3;
  margin-left: 4px;
}
.leg { border-top: 1px solid #f1f5f9; padding-top: 8px; margin-top: 8px; }
.line { font-weight: 650; }
.stops { font-weight: 700; font-size: 14px; margin-top: 4px; }
.muted { color: #64748b; font-size: 13px; margin-top: 2px; }
.time-hint { font-size: 12px; }
.honest-note {
  color: #64748b;
  font-size: 12px;
  line-height: 1.35;
  margin-top: 8px;
}
.empty {
  color: #64748b;
  padding: 12px;
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
}
.banner {
  border: 1px solid #fca5a5;
  background: #fef2f2;
  color: #991b1b;
  padding: 10px 12px;
  border-radius: 10px;
}
.provider {
  border: 1px solid #bfdbfe;
  background: #eff6ff;
  color: #1e3a8a;
  padding: 10px 12px;
  border-radius: 10px;
}
.reply-audio { max-width: 420px; }
.timings { color: #475569; font-size: 13px; margin-bottom: 8px; }
@media (max-width: 640px) {
  .trip-grid { grid-template-columns: 1fr; }
}
"""


def _provider_config() -> ProviderConfig:
    return load_config()


def _api_key_configured() -> bool:
    """Check the configured key env var without printing the key."""

    return api_key_configured(_provider_config())


def _missing_key_message() -> str:
    config = _provider_config()
    return (
        f"{config.provider} key not configured. Add "
        f"{config.api_key_env} to your environment or .env file, then try "
        "again."
    )


def _provider_banner() -> tuple[str, bool]:
    """Return the banner text and whether it is a warning."""

    config = _provider_config()
    if api_key_configured(config):
        return (
            f"**Provider:** {config.provider} \u00b7 "
            f"**Model:** `{config.model}`",
            False,
        )
    return (
        f"### {config.provider} key not configured\n"
        f"Add `{config.api_key_env}` to a local `.env` file "
        "(see `.env.example`) and restart the app.",
        True,
    )


@lru_cache(maxsize=1)
def _client() -> ModelClient:
    return ModelClient()


@lru_cache(maxsize=1)
def _agent() -> Agent:
    data = load_data()
    network = Network.from_data(data)
    place_index = PlaceIndex.from_data(data)
    engine = RouteEngine(network, place_index)
    return Agent(_client(), engine, place_index)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _is_french(language: str | None) -> bool:
    lowered = (language or "").strip().lower()
    return lowered in {"french", "français", "francais"}


def _is_wolof(language: str | None) -> bool:
    lowered = (language or "").strip().lower()
    return lowered in {"wolof", "wo", "wol_latn"}


def _minutes(value: Any) -> str:
    try:
        return f"{float(value):.0f} min"
    except (TypeError, ValueError):
        return "unknown time"


def _format_timings(timings: dict[str, Any] | None) -> str:
    """Show each turn step without ever including the user's words."""

    labels = (("asr", "ASR"), ("agent", "agent"), ("voice", "voice"))
    parts: list[str] = []
    for key, label in labels:
        try:
            seconds = float((timings or {}).get(key))
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None and (
            not math.isfinite(seconds) or seconds < 0
        ):
            seconds = None

        if key == "agent":
            try:
                calls = int((timings or {}).get("agent_calls"))
            except (TypeError, ValueError):
                calls = 0
            noun = "call" if calls == 1 else "calls"
            if calls > 0 and seconds is not None:
                parts.append(
                    f"agent {seconds:.1f} s ({calls} LLM {noun})"
                )
                continue
            if calls > 0:
                parts.append(f"agent: {calls} LLM {noun}")
                continue

        if seconds is None:
            continue
        suffix = (
            " (failed)"
            if key == "voice" and (timings or {}).get("voice_failed")
            else ""
        )
        parts.append(f"{label} {seconds:.1f} s{suffix}")
    return " · ".join(parts)


def _wake_voice_services() -> None:
    """Ping the ASR and TTS health endpoints so cold containers start early.

    Modal may answer a cold request with a redirect after a long wait, so the
    client follows redirects.  Any failure is ignored: the text answer must
    always work even when voice is down.
    """

    key = (os.environ.get("KOUMBA_ASR_KEY") or "").strip()
    headers = {"X-Koumba-Key": key} if key else {}

    for name in ("KOUMBA_ASR_URL", "KOUMBA_TTS_URL"):
        url = (os.environ.get(name) or "").strip().rstrip("/")
        if not url:
            continue
        try:
            httpx.get(
                f"{url}/health",
                headers=headers,
                timeout=150.0,
                follow_redirects=True,
            )
        except Exception:
            # Waking is best effort.  Never surface it in the UI.
            pass


def _start_voice_wakeup() -> None:
    """Start the wake-up in the background so the UI never waits for it."""

    threading.Thread(
        target=_wake_voice_services,
        name="koumba-voice-wakeup",
        daemon=True,
    ).start()


def _same_place(left: Any, right: Any) -> bool:
    first = normalize_name(str(left or ""))
    second = normalize_name(str(right or ""))
    return bool(first and second and first == second)


def _walk_note(walk: dict[str, Any], language: str | None) -> str:
    kind = str(walk.get("kind") or "walk")
    from_place = str(walk.get("from_place") or "")
    to_place = str(walk.get("to_place") or "")
    french = _is_french(language)
    labels = {
        "access": ("entry", "entrée"),
        "transfer": ("transfer", "correspondance"),
        "egress": ("exit", "sortie"),
    }
    english_label, french_label = labels.get(kind, ("walk", "marche"))

    if _same_place(from_place, to_place):
        if kind == "access":
            note = (
                "entrée : à pied jusqu'à l'arrêt"
                if french
                else "entry: walk to the stop"
            )
        elif kind == "transfer":
            note = (
                "correspondance : à pied jusqu'à l'arrêt"
                if french
                else "transfer: walk to the stop"
            )
        else:
            note = (
                "sortie : à pied jusqu'à destination"
                if french
                else "exit: walk to destination"
            )
    else:
        arrow = f"{escape(from_place)} &rarr; {escape(to_place)}"
        note = (
            f"{french_label} : {arrow}"
            if french
            else f"{english_label}: {arrow}"
        )
    return f"{note} ({_minutes(walk.get('minutes'))})"


def _stop_count(value: Any, language: str | None) -> str:
    try:
        count = max(0, int(value or 0))
    except (TypeError, ValueError):
        count = 0
    if _is_french(language):
        return f"{count} arrêt" if count == 1 else f"{count} arrêts"
    return f"{count} stop" if count == 1 else f"{count} stops"


def _time_hint(value: Any, language: str | None) -> str:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(minutes) or minutes <= 0:
        return ""
    suffix = (
        "hors embouteillages"
        if _is_french(language)
        else "without traffic"
    )
    return f"~{minutes:.0f} min {suffix}"


def _honest_note(language: str | None) -> str:
    if _is_french(language):
        return HONEST_NOTE_FR
    return HONEST_NOTE_EN


def _trip_cards(
    trips: list[dict[str, Any]] | None,
    language: str = "English",
    part_way: bool = False,
) -> str:
    """Render the trips as simple mobile-friendly cards."""

    note = f"<div class='honest-note'>{_honest_note(language)}</div>"
    if not trips:
        return (
            "<div class='empty'>No trip cards yet. Ask Koumba where you "
            "are and where you want to go.</div>"
            f"{note}"
        )

    part_way_pill = (
        "<span class='pill'>Part of the way with this bus</span>"
        if part_way
        else ""
    )
    cards: list[str] = []
    for index, trip in enumerate(trips, start=1):
        legs_html: list[str] = []
        for leg in trip.get("legs", []):
            operator = escape(str(leg.get("operator", "")))
            line = escape(str(leg.get("line", "")))
            board = escape(str(leg.get("board_stop", "")))
            alight = escape(str(leg.get("alight_stop", "")))
            direction = escape(str(leg.get("direction", "")))
            stops = _stop_count(leg.get("num_stops"), language)
            legs_html.append(
                "<div class='leg'>"
                f"<div class='line'>{operator} {line}</div>"
                f"<div>{board} &rarr; {alight}</div>"
                f"<div class='stops'>{stops}</div>"
                f"<div class='muted'>towards {direction}</div>"
                "</div>"
            )

        walk_notes = [
            _walk_note(walk, language)
            for walk in trip.get("walks", [])
            if float(walk.get("minutes") or 0) > 0
        ]
        walk_html = (
            f"<div class='muted'>{'; '.join(walk_notes)}</div>"
            if walk_notes
            else ""
        )

        transfers = int(trip.get("transfers", 0))
        time_hint = _time_hint(trip.get("total_minutes"), language)
        time_html = (
            f"<div class='muted time-hint'>{time_hint}</div>"
            if time_hint
            else ""
        )
        cards.append(
            "<div class='trip-card'>"
            f"<div class='trip-head'>Trip {index}"
            f"<span class='pill'>{_plural(transfers, 'transfer')}</span>"
            f"{part_way_pill}"
            "</div>"
            f"{time_html}"
            f"{''.join(legs_html)}"
            f"{walk_html}"
            "</div>"
        )

    return (
        "<div class='trip-grid'>"
        + "".join(cards)
        + "</div>"
        + note
    )


def _cards_for_result(result: Any) -> str:
    """Pick cards from a bus check first, then from normal trip findings."""

    check = getattr(result, "bus_check", None)
    language = getattr(result, "language", "English")
    if isinstance(check, dict):
        if check.get("serves_trip") is True:
            trip = check.get("trip")
            return _trip_cards(
                [trip] if isinstance(trip, dict) else [],
                language,
            )
        alternative = check.get("best_alternative")
        return _trip_cards(
            [alternative] if isinstance(alternative, dict) else [],
            language,
            part_way=bool(check.get("can_help_part_way")),
        )
    return _trip_cards(getattr(result, "trips", None), language)


def _history_as_dicts(history) -> list[dict[str, Any]]:
    """Normalize old and new Gradio chat history formats."""

    result: list[dict[str, Any]] = []
    for item in history or []:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            result.append({"role": "user", "content": str(item[0])})
            result.append({"role": "assistant", "content": str(item[1])})
        else:
            result.append(
                {
                    "role": getattr(item, "role", "user"),
                    "content": getattr(item, "content", ""),
                }
            )
    return result


def _respond_turn(
    message_text: str,
    history,
    image_path: str | None = None,
):
    """Handle one typed chat submission and also return its agent result."""

    text = (message_text or "").strip()
    previous = _history_as_dicts(history)
    current = list(previous)
    if not text:
        return (
            current,
            "",
            gr.update(),
            gr.update(),
            image_path,
            None,
            {},
        )

    current.append({"role": "user", "content": text})

    if not _api_key_configured():
        current.append(
            {
                "role": "assistant",
                "content": _missing_key_message(),
            }
        )
        return current, "", _trip_cards(None), [], None, None, {}

    started = time.perf_counter()
    try:
        result = _agent().run(text, previous, image_path)
    except Exception as exc:
        current.append(
            {
                "role": "assistant",
                "content": (
                    "Koumba could not start the route engine. "
                    f"{exc}"
                ),
            }
        )
        return current, "", _trip_cards(None), [], None, None, {}
    timings = {
        "agent": round(time.perf_counter() - started, 3),
        "agent_calls": getattr(result, "llm_calls", 0)
        or getattr(result, "rounds", 0),
    }

    current.append(
        {
            "role": "assistant",
            "content": compose_chat_message(result.spoken, result.reply),
        }
    )
    return (
        current,
        "",
        _cards_for_result(result),
        result.tool_calls,
        None,
        result,
        timings,
    )


def _respond(message_text: str, history, image_path: str | None = None):
    """Handle one chat submission and return the five UI values."""

    chatbot, message, cards, thoughts, image, _result, _timings = _respond_turn(
        message_text,
        history,
        image_path,
    )
    return chatbot, message, cards, thoughts, image


def _tts_text_for_result(result: Any) -> str:
    """Use the displayed sentence, with reviewed bus and place spellings."""

    text = str(
        getattr(result, "spoken", "") or getattr(result, "reply", "") or ""
    ).strip()
    if not text:
        return ""

    trips = getattr(result, "trips", None)
    best_trip = (
        trips[0]
        if isinstance(trips, list) and trips and isinstance(trips[0], dict)
        else None
    )
    check = getattr(result, "bus_check", None)
    alternative = (
        check.get("best_alternative")
        if isinstance(check, dict)
        else None
    )
    return tts_text(
        text,
        trip=best_trip,
        origin=getattr(result, "origin", None),
        check=check if isinstance(check, dict) else None,
        destination=getattr(result, "destination", None),
        alternative=alternative if isinstance(alternative, dict) else None,
    )


def _tts_job_for_result(
    result: Any,
    timings: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep the speech text in state until the second UI phase asks for it."""

    if result is None or not _is_wolof(getattr(result, "language", None)):
        return None
    try:
        text = _tts_text_for_result(result)
    except Exception:
        return None
    if not text:
        return None
    return {"text": text, "timings": dict(timings or {})}


def _synthesize_voice(job: Any):
    """Second Gradio phase: fill the autoplay player without touching chat."""

    if not isinstance(job, dict):
        return gr.update(), gr.update(), None

    timings = dict(job.get("timings") or {})
    text = str(job.get("text") or "").strip()
    if not text:
        return gr.update(), gr.update(), None

    started = time.perf_counter()
    failed = False
    audio: bytes | None = None
    try:
        audio = speak(text)
    except SpeakError as exc:
        gr.Warning(str(exc))
        failed = True
    except Exception:
        failed = True

    timings["voice"] = round(time.perf_counter() - started, 3)
    if failed:
        timings["voice_failed"] = True
    return audio, _format_timings(timings), None


def _audio_present(audio: Any) -> bool:
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        return False
    _sample_rate, samples = audio
    if samples is None:
        return False
    try:
        return len(samples) > 0
    except TypeError:
        return False


def _audio_fingerprint(audio: Any) -> str | None:
    """Identify one recording so stop_recording and change cannot both run."""

    if not _audio_present(audio):
        return None
    sample_rate, samples = audio
    try:
        array = np.asarray(samples)
        digest = hashlib.sha1()
        digest.update(str(float(sample_rate)).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    except (TypeError, ValueError):
        return None
    return digest.hexdigest()


def _transcribe_audio(audio: Any) -> dict[str, Any]:
    sample_rate, samples = audio
    wav_bytes = prepare_audio(sample_rate, samples)
    voice_config()
    return transcribe(wav_bytes)


def _transcription_error(exc: Exception) -> str:
    message = str(exc or "").strip()
    if isinstance(exc, VoiceError):
        return message
    if isinstance(exc, ValueError) and "too short" in message.lower():
        return message
    return (
        "I could not transcribe that recording. Please try again or type your "
        "message."
    )


def _voice_thoughts(
    transcription: dict[str, Any],
    tool_calls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep the raw transcript and local place hints in the thinking panel."""

    voice = {
        "raw_asr_text": transcription.get("raw_text") or "",
        "language": transcription.get("language") or "",
        "asr_lang": transcription.get("asr_lang") or "",
        "hints": transcription.get("hints") or [],
        "transcription_model": ASR_MODEL_LABEL,
    }
    return [{"voice": voice}, *(tool_calls or [])]


def _respond_phase_one(
    message_text: str,
    history,
    image_path: str | None = None,
    audio: Any = None,
):
    """Return text, cards, and timings before the slower speech call."""

    if not _audio_present(audio):
        (
            chatbot,
            message,
            cards,
            thoughts,
            image,
            result,
            timings,
        ) = _respond_turn(
            message_text,
            history,
            image_path,
        )
        job = _tts_job_for_result(result, timings)
        return (
            chatbot,
            message,
            cards,
            thoughts,
            image,
            None,
            None,
            _format_timings(timings),
            job,
        )

    previous = _history_as_dicts(history)
    current = list(previous)
    try:
        voice_config()
    except VoiceError as exc:
        current.append({"role": "assistant", "content": str(exc)})
        return (
            current,
            message_text or "",
            _trip_cards(None),
            [],
            image_path,
            None,
            None,
            "",
            None,
        )

    if not _api_key_configured():
        current.append(
            {"role": "assistant", "content": _missing_key_message()}
        )
        return (
            current,
            message_text or "",
            _trip_cards(None),
            [],
            image_path,
            None,
            None,
            "",
            None,
        )

    try:
        transcription = _transcribe_audio(audio)
    except Exception as exc:
        current.append(
            {"role": "assistant", "content": _transcription_error(exc)}
        )
        return (
            current,
            message_text or "",
            _trip_cards(None),
            [],
            image_path,
            None,
            None,
            "",
            None,
        )

    timings = dict(transcription.get("timings") or {})
    text = str(
        transcription.get("raw_text")
        or transcription.get("text")
        or ""
    ).strip()
    hints = transcription.get("hints") or []
    if not text:
        current.append(
            {
                "role": "assistant",
                "content": (
                    "I could not hear any words in that recording. Please try "
                    "again or type your message."
                ),
            }
        )
        return (
            current,
            message_text or "",
            _trip_cards(None),
            [],
            image_path,
            None,
            None,
            _format_timings(timings),
            None,
        )

    current.append({"role": "user", "content": f"🎤 {text}"})
    started = time.perf_counter()
    try:
        result = _agent().run(text, previous, image_path, hints=hints)
    except Exception as exc:
        timings["agent"] = round(time.perf_counter() - started, 3)
        current.append(
            {
                "role": "assistant",
                "content": f"Koumba could not start the route engine. {exc}",
            }
        )
        return (
            current,
            "",
            _trip_cards(None),
            [],
            None,
            None,
            None,
            _format_timings(timings),
            None,
        )
    timings["agent"] = round(time.perf_counter() - started, 3)
    timings["agent_calls"] = getattr(result, "llm_calls", 0) or getattr(
        result, "rounds", 0
    )

    current.append(
        {
            "role": "assistant",
            "content": compose_chat_message(result.spoken, result.reply),
        }
    )
    return (
        current,
        "",
        _cards_for_result(result),
        _voice_thoughts(transcription, result.tool_calls),
        None,
        None,
        None,
        _format_timings(timings),
        _tts_job_for_result(result, timings),
    )


def _respond_with_audio(
    message_text: str,
    history,
    image_path: str | None = None,
    audio: Any = None,
):
    """Synchronous helper for tests; the UI uses two phases."""

    (
        chatbot,
        message,
        cards,
        thoughts,
        image,
        _audio_input,
        _previous_reply,
        _timings,
        job,
    ) = _respond_phase_one(
        message_text,
        history,
        image_path,
        audio,
    )
    if job is None:
        return chatbot, message, cards, thoughts, image, None, None
    reply_audio, _timings, _job = _synthesize_voice(job)
    return chatbot, message, cards, thoughts, image, None, reply_audio


def _respond_to_recording(
    history,
    image_path: str | None,
    audio: Any,
    last_audio_key: str | None,
):
    """Auto-send a finished microphone recording exactly once."""

    audio_key = _audio_fingerprint(audio)
    if audio_key is None or audio_key == last_audio_key:
        # Another event already handled this waveform.  Leave every component
        # alone instead of adding the same question to the chat twice.  A null
        # job also stops the chained speech phase from replaying old audio.
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            last_audio_key,
        )

    gr.Info(LISTENING_STATUS, duration=4)
    result = _respond_phase_one("", history, image_path, audio)
    return (*result, audio_key)


def _clear():
    return [], "", _trip_cards(None), [], None


def _clear_voice():
    return (*_clear(), None, None, "", None, None)


def build_app() -> gr.Blocks:
    """Build the Gradio UI without launching it."""

    with gr.Blocks(title="Koumba") as demo:
        gr.Markdown(
            "# Koumba\n"
            "Your Dakar bus assistant. Ask in Wolof, French, or English."
        )

        banner_text, is_warning = _provider_banner()
        # The banner call above loads .env, so the voice URLs are ready now.
        _start_voice_wakeup()
        gr.Markdown(
            banner_text,
            elem_classes=["banner" if is_warning else "provider"],
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    label="Koumba",
                    height=360,
                    layout="bubble",
                )
                reply_audio = gr.Audio(
                    label="Koumba speaks (Wolof)",
                    autoplay=True,
                    interactive=False,
                    elem_classes=["reply-audio"],
                )
                message = gr.Textbox(
                    label="Where are you going?",
                    placeholder=(
                        "Maa ngi Sahm, bëgg naa dem Malika / "
                        "Je suis à Ouakam, je vais au Palais"
                    ),
                    lines=2,
                )
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Speak to Koumba (Wolof, French, English)",
                )
                voice_guard = gr.State(None)
                voice_job = gr.State(None)
                gr.Markdown("Voice recordings are not stored.")
                image = gr.Image(
                    label="Photo of the bus (optional)",
                    type="filepath",
                    sources=["upload", "webcam"],
                    height=220,
                )
                gr.Markdown("Photos are not stored.")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")

                gr.Markdown("**Try an example**")
                with gr.Row():
                    example_wolof = gr.Button(EXAMPLE_WOLOF, size="sm")
                    example_french = gr.Button(EXAMPLE_FRENCH, size="sm")

            with gr.Column(scale=2):
                trips = gr.HTML(
                    value=_trip_cards(None),
                    label="Trip cards",
                    container=True,
                )
                with gr.Accordion("How Koumba thought", open=False):
                    timings = gr.Markdown(value="", elem_classes=["timings"])
                    thoughts = gr.JSON(
                        value=[],
                        label="Tool calls and voice details",
                    )

        voice_outputs = [
            chatbot,
            message,
            trips,
            thoughts,
            image,
            audio,
            reply_audio,
            timings,
            voice_job,
        ]

        def add_voice_phase(event):
            return event.then(
                _synthesize_voice,
                inputs=[voice_job],
                outputs=[reply_audio, timings, voice_job],
            )

        add_voice_phase(
            send.click(
                _respond_phase_one,
                inputs=[message, chatbot, image, audio],
                outputs=voice_outputs,
            )
        )
        add_voice_phase(
            message.submit(
                _respond_phase_one,
                inputs=[message, chatbot, image, audio],
                outputs=voice_outputs,
            )
        )
        # Only the microphone stop event auto-sends.  Do not also bind change:
        # Gradio emits change when recording ends and again when we clear the
        # audio component, which would send the same question twice.
        add_voice_phase(
            audio.stop_recording(
                _respond_to_recording,
                inputs=[chatbot, image, audio, voice_guard],
                outputs=[*voice_outputs, voice_guard],
            )
        )
        add_voice_phase(
            example_wolof.click(
                lambda history, audio: _respond_phase_one(
                    EXAMPLE_WOLOF,
                    history,
                    None,
                    audio,
                ),
                inputs=[chatbot, audio],
                outputs=voice_outputs,
            )
        )
        add_voice_phase(
            example_french.click(
                lambda history, audio: _respond_phase_one(
                    EXAMPLE_FRENCH,
                    history,
                    None,
                    audio,
                ),
                inputs=[chatbot, audio],
                outputs=voice_outputs,
            )
        )
        clear.click(
            _clear_voice,
            outputs=[*voice_outputs, voice_guard],
        )

    return demo


if __name__ == "__main__":
    # Gradio 6 moved the CSS argument from Blocks to launch().
    build_app().launch(css=CSS)

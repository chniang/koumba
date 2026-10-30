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

EXAMPLE_WOLOF = "Maa ngi Sahm, dama bëgga dem Malika"
EXAMPLE_WOLOF_MEDINA = "Maa ngi Médina, dama bëgga dem Yoff"
EXAMPLE_FRENCH = "Je suis à Ouakam, je vais au Palais"

HONEST_NOTE_EN = (
    "Routes based on official AFTU / Dakar Dem Dikk data and field "
    "corrections. Check with the apprenti."
)
HONEST_NOTE_FR = (
    "Itinéraires basés sur les données officielles AFTU / Dakar Dem Dikk et "
    "des corrections de terrain. Vérifie avec l'apprenti."
)

MODELS_LINE = (
    "Meta Omnilingual ASR (speech) · Muse Spark Open (agent, vision) · "
    "Oolel-Voices (Wolof voice)"
)
FOOTER_LINE = (
    "Built during the Meta Global AI Developer Hackathon · Data: AFTU / "
    "Dakar Dem Dikk public lines · Check with the apprenti"
)

# All visible UI labels live here.  Koumba's answers stay in the language of
# the question; only the small EN | FR switch reads from this table.
UI_TEXT = {
    "English": {
        "tagline": "Your Dakar bus assistant",
        "chat_label": "Chat with Koumba",
        "voice_label": "Koumba's voice",
        "mic_label": "Speak to Koumba",
        "message_label": "Where are you going?",
        "placeholder": (
            "Maa ngi Sahm, dama bëgga dem Malika / "
            "Je suis à Ouakam, je vais au Palais"
        ),
        "send": "Send",
        "clear": "Clear",
        "photo_label": "Is this my bus? (photo)",
        "photo_note": "Photos are not stored.",
        "examples_label": "Try an example",
        "thinking_label": "How Koumba thought",
        "tools_label": "Tool calls and voice details",
        "trips_label": "Trip cards",
        "empty_trips": (
            "No trip cards yet. Ask Koumba where you are and where you "
            "want to go."
        ),
        "part_way": "Part of the way with this bus",
        "best": "Best",
        "trip": "Trip",
        "board": "Board",
        "alight": "Alight",
        "towards": "towards",
        "transfer": "transfer",
        "voice_note": "Voice recordings are not stored.",
    },
    "French": {
        "tagline": "Votre assistant bus à Dakar",
        "chat_label": "Discussion avec Koumba",
        "voice_label": "La voix de Koumba",
        "mic_label": "Parle à Koumba",
        "message_label": "Où vas-tu ?",
        "placeholder": (
            "Maa ngi Sahm, dama bëgga dem Malika / "
            "Je suis à Ouakam, je vais au Palais"
        ),
        "send": "Envoyer",
        "clear": "Effacer",
        "photo_label": "C'est mon bus ? (photo)",
        "photo_note": "Les photos ne sont pas conservées.",
        "examples_label": "Essaie un exemple",
        "thinking_label": "Comment Koumba a réfléchi",
        "tools_label": "Appels d'outils et détails vocaux",
        "trips_label": "Itinéraires",
        "empty_trips": (
            "Pas encore d'itinéraire. Dis à Koumba où tu es et où tu veux "
            "aller."
        ),
        "part_way": "Une partie du trajet avec ce bus",
        "best": "Meilleur",
        "trip": "Trajet",
        "board": "Monter",
        "alight": "Descendre",
        "towards": "vers",
        "transfer": "correspondance",
        "voice_note": "Les enregistrements vocaux ne sont pas conservés.",
    },
}

CSS = """
:root {
  --koumba-green: #00853F;
  --koumba-green-soft: #e8f6ee;
  --koumba-yellow: #FDEF42;
  --koumba-red: #E31B23;
  --koumba-ink: #102018;
  --koumba-muted: #5f6f65;
  --koumba-line: #e3ece6;
}

html, body {
  background: #ffffff !important;
}

.gradio-container {
  max-width: 480px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  background: #ffffff !important;
  color: var(--koumba-ink) !important;
  font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif !important;
  font-size: 15px !important;
}

.gradio-container * { box-sizing: border-box; }

.app-shell {
  width: 100%;
  max-width: 480px;
  margin: 0 auto;
  padding: 16px 16px 28px;
  gap: 14px;
  background: #ffffff;
}

.hero-row {
  align-items: flex-start;
  gap: 8px;
}
.hero-card { padding-top: 4px; }
.koumba-title {
  margin: 0;
  color: var(--koumba-green);
  font-size: 34px;
  font-weight: 850;
  letter-spacing: -0.8px;
  line-height: 1.05;
}
.koumba-greeting {
  margin: 8px 0 2px;
  color: var(--koumba-ink);
  font-size: 19px;
  font-weight: 750;
  line-height: 1.25;
}
.koumba-tagline {
  margin: 0;
  color: var(--koumba-muted);
  font-size: 13.5px;
  line-height: 1.4;
}

#ui-language {
  min-width: 92px !important;
  border: 1px solid var(--koumba-line);
  border-radius: 999px;
  padding: 2px;
  background: #f8fbf9;
}
#ui-language .wrap {
  flex-direction: row !important;
  gap: 2px !important;
  justify-content: center;
}
#ui-language label {
  margin: 0 !important;
  padding: 4px 8px !important;
  border-radius: 999px !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  color: var(--koumba-muted) !important;
}
#ui-language label:has(input:checked) {
  background: var(--koumba-green) !important;
  color: #ffffff !important;
}

#chatbot {
  border: 1px solid var(--koumba-line) !important;
  border-radius: 20px !important;
  background: #fbfdfb !important;
  overflow: hidden;
}
#chatbot .message-wrap { font-size: 14.5px !important; line-height: 1.45 !important; }

#voice-player {
  border: 1px solid var(--koumba-line) !important;
  border-radius: 16px !important;
  background: #f7fbf8 !important;
  padding: 5px !important;
}
#voice-player audio { width: 100% !important; }

#mic {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 128px;
  border: 0 !important;
  background: transparent !important;
}
#mic .audio-container {
  justify-content: center !important;
  align-items: center !important;
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
#mic .record-button {
  width: 104px !important;
  height: 104px !important;
  min-width: 104px !important;
  border-radius: 50% !important;
  background: var(--koumba-green) !important;
  border: 8px solid var(--koumba-green-soft) !important;
  box-shadow: 0 10px 24px rgba(0, 133, 63, 0.28) !important;
  color: #ffffff !important;
}
#mic .record-button svg {
  width: 40px !important;
  height: 40px !important;
  color: #ffffff !important;
  fill: #ffffff !important;
}
#mic .stop-button {
  border-radius: 50% !important;
  background: var(--koumba-red) !important;
  color: #ffffff !important;
}
.mic-note {
  margin: -8px 0 0;
  text-align: center;
  color: var(--koumba-muted);
  font-size: 11.5px;
}

.composer { gap: 8px; align-items: stretch; }
#message {
  border-radius: 16px !important;
  font-size: 15px !important;
}
#message textarea {
  min-height: 52px !important;
  border-radius: 16px !important;
}
.send-btn, .send-btn button {
  background: var(--koumba-green) !important;
  border-color: var(--koumba-green) !important;
  color: #ffffff !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}
.clear-btn, .clear-btn button {
  background: #ffffff !important;
  border: 1px solid var(--koumba-line) !important;
  color: var(--koumba-muted) !important;
  border-radius: 14px !important;
}

.photo-button, .photo-button button {
  width: 100% !important;
  border: 1px solid #e7df93 !important;
  border-radius: 14px !important;
  background: #fffdf0 !important;
  color: var(--koumba-ink) !important;
  font-weight: 700 !important;
}
.privacy-note {
  margin: -6px 0 0;
  color: var(--koumba-muted);
  font-size: 11.5px;
}

.examples-heading {
  margin: 0 0 -4px;
  color: var(--koumba-muted);
  font-size: 12.5px;
  font-weight: 750;
}
.examples-row { flex-wrap: wrap; gap: 8px; }
.example-chip, .example-chip button {
  flex: 1 1 140px;
  border: 1px solid #d7e8dd !important;
  border-radius: 999px !important;
  background: #f5fbf7 !important;
  color: #17452b !important;
  font-size: 12px !important;
  font-weight: 650 !important;
}

#trip-cards {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}
.trip-grid { display: flex; flex-direction: column; gap: 12px; }
.trip-card {
  border: 1px solid var(--koumba-line);
  border-radius: 20px;
  padding: 15px;
  background: #ffffff;
  box-shadow: 0 2px 10px rgba(16, 32, 24, 0.05);
}
.trip-card.best-trip {
  border: 2px solid var(--koumba-green);
  background: linear-gradient(180deg, #f2fbf6 0%, #ffffff 46%);
}
.trip-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 5px;
}
.trip-rank { color: var(--koumba-green); font-size: 13px; font-weight: 800; }
.pill {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--koumba-green-soft);
  color: #145c35;
  font-size: 11px;
  font-weight: 700;
}
.best-pill {
  background: var(--koumba-yellow) !important;
  color: #3b3300 !important;
  font-weight: 850 !important;
}
.leg {
  border-top: 1px solid var(--koumba-line);
  padding-top: 10px;
  margin-top: 10px;
}
.leg-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.bus-name {
  color: var(--koumba-ink);
  font-size: 17px;
  font-weight: 800;
  line-height: 1.25;
}
.op-tag {
  flex: 0 0 auto;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 10.5px;
  font-weight: 850;
  text-transform: uppercase;
  letter-spacing: 0.2px;
}
.op-tata { background: var(--koumba-yellow); color: #3b3300; }
.op-dem { background: var(--koumba-green); color: #ffffff; }
.board-alight {
  margin-top: 6px;
  color: var(--koumba-ink);
  font-size: 14px;
  line-height: 1.45;
}
.board-label, .alight-label {
  color: var(--koumba-muted);
  font-size: 10.5px;
  font-weight: 800;
  letter-spacing: 0.2px;
  text-transform: uppercase;
}
.stops {
  margin-top: 5px;
  color: var(--koumba-ink);
  font-size: 14px;
  font-weight: 750;
}
.muted {
  margin-top: 2px;
  color: var(--koumba-muted);
  font-size: 12.5px;
  line-height: 1.4;
}
.time-hint { font-size: 12px; }
.honest-note {
  margin-top: 8px;
  color: var(--koumba-muted);
  font-size: 11.5px;
  line-height: 1.4;
}
.empty {
  padding: 14px;
  border: 1px dashed #bfd9c9;
  border-radius: 16px;
  color: var(--koumba-muted);
  font-size: 13px;
}

.technical-panel {
  border: 1px solid var(--koumba-line) !important;
  border-radius: 16px !important;
  background: #fbfdfb !important;
}
.timings { margin-bottom: 8px; color: #475569; font-size: 12.5px; }
.models-used {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--koumba-line);
  color: var(--koumba-muted);
  font-size: 11.5px;
  line-height: 1.5;
}

.footer {
  padding: 8px 4px 2px;
  color: var(--koumba-muted);
  font-size: 11px;
  line-height: 1.5;
  text-align: center;
}

@media (max-width: 520px) {
  .gradio-container { max-width: 100% !important; }
  .app-shell { padding: 14px 14px 24px; }
  .koumba-title { font-size: 30px; }
  .koumba-greeting { font-size: 17px; }
  #mic .record-button { width: 96px !important; height: 96px !important; }
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
    return lowered in {"fr", "french", "français", "francais"}


def _is_wolof(language: str | None) -> bool:
    lowered = (language or "").strip().lower()
    return lowered in {"wolof", "wo", "wol_latn"}


def _ui_language(value: str | None) -> str:
    """Map the compact switch value to the UI label language."""

    if (value or "").strip().lower() in {"fr", "french", "français", "francais"}:
        return "French"
    return "English"


def _ui_copy(language: str | None) -> dict[str, str]:
    """Return the visible labels for one UI language."""

    return UI_TEXT[_ui_language(language)]


def _header_html(language: str | None) -> str:
    copy = _ui_copy(language)
    return (
        "<div class='hero-card'>"
        "<div class='koumba-title'>Koumba</div>"
        "<div class='koumba-greeting'>"
        "Dalal ak jàmm ! Fan nga bëgga dem ?"
        "</div>"
        f"<div class='koumba-tagline'>{escape(copy['tagline'])}</div>"
        "</div>"
    )


def _footer_html() -> str:
    return f"<div class='footer'>{escape(FOOTER_LINE)}</div>"


def _switch_language(value: str | None):
    """Update only interface labels when EN or FR is selected."""

    copy = _ui_copy(value)
    return (
        gr.update(value=_header_html(value)),
        gr.update(label=copy["chat_label"]),
        gr.update(label=copy["voice_label"]),
        gr.update(label=copy["mic_label"]),
        gr.update(value=copy["voice_note"]),
        gr.update(
            label=copy["message_label"],
            placeholder=copy["placeholder"],
        ),
        gr.update(value=copy["send"]),
        gr.update(value=copy["clear"]),
        gr.update(label=copy["photo_label"]),
        gr.update(value=copy["photo_note"]),
        gr.update(value=f"**{copy['examples_label']}**"),
        gr.update(label=copy["trips_label"]),
        gr.update(label=copy["tools_label"]),
    )


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


def _operator_tag(operator: str) -> tuple[str, str]:
    """Give each company a short, recognizable color tag."""

    lowered = operator.lower()
    if "dem dikk" in lowered:
        return "Dem Dikk", "op-dem"
    if "tata" in lowered:
        return "Tata", "op-tata"
    return operator or "Bus", "op-tata"


def _transfer_count(value: Any, language: str | None) -> str:
    try:
        count = max(0, int(value or 0))
    except (TypeError, ValueError):
        count = 0
    if _is_french(language):
        label = "correspondance" if count == 1 else "correspondances"
        return f"{count} {label}"
    return _plural(count, "transfer")


def _trip_cards(
    trips: list[dict[str, Any]] | None,
    language: str = "English",
    part_way: bool = False,
) -> str:
    """Render the trips as simple mobile-friendly cards."""

    language = _ui_language(language)
    copy = _ui_copy(language)
    note = f"<div class='honest-note'>{_honest_note(language)}</div>"
    if not trips:
        return f"<div class='empty'>{copy['empty_trips']}</div>{note}"

    part_way_pill = (
        f"<span class='pill'>{copy['part_way']}</span>" if part_way else ""
    )
    cards: list[str] = []
    for index, trip in enumerate(trips, start=1):
        is_best = index == 1
        rank_html = (
            f"<span class='pill best-pill'>{copy['best']}</span>"
            if is_best
            else f"<span class='trip-rank'>{copy['trip']} {index}</span>"
        )

        legs_html: list[str] = []
        for leg in trip.get("legs", []):
            operator = escape(str(leg.get("operator", "")))
            line = escape(str(leg.get("line", "")))
            board = escape(str(leg.get("board_stop", "")))
            alight = escape(str(leg.get("alight_stop", "")))
            direction = escape(str(leg.get("direction", "")))
            stops = _stop_count(leg.get("num_stops"), language)
            tag, tag_class = _operator_tag(str(leg.get("operator", "")))
            legs_html.append(
                "<div class='leg'>"
                "<div class='leg-head'>"
                f"<div class='bus-name'>{operator} {line}</div>"
                f"<span class='op-tag {tag_class}'>{escape(tag)}</span>"
                "</div>"
                "<div class='board-alight'>"
                f"<span class='board-label'>{copy['board']}</span> "
                f"{board} &rarr; "
                f"<span class='alight-label'>{copy['alight']}</span> "
                f"{alight}"
                "</div>"
                f"<div class='stops'>{stops}</div>"
                f"<div class='muted'>{copy['towards']} {direction}</div>"
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
            f"<div class='trip-card{' best-trip' if is_best else ''}'>"
            "<div class='trip-top'>"
            f"{rank_html}"
            f"<span class='pill'>"
            f"{_transfer_count(transfers, language)}"
            "</span>"
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


def _cards_for_result(
    result: Any,
    language: str = "English",
) -> str:
    """Pick cards from a bus check first, then from normal trip findings.

    Card labels follow the UI switch, not the language of Koumba's answer.
    """

    check = getattr(result, "bus_check", None)
    language = _ui_language(language)
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
    ui_language: str | None = None,
):
    """Handle one typed chat submission and also return its agent result."""

    language = _ui_language(ui_language)
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
        return current, "", _trip_cards(None, language), [], None, None, {}

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
        return current, "", _trip_cards(None, language), [], None, None, {}
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
        _cards_for_result(result, language),
        result.tool_calls,
        None,
        result,
        timings,
    )


def _respond(
    message_text: str,
    history,
    image_path: str | None = None,
    ui_language: str | None = None,
):
    """Handle one chat submission and return the five UI values."""

    chatbot, message, cards, thoughts, image, _result, _timings = _respond_turn(
        message_text,
        history,
        image_path,
        ui_language,
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
    ui_language: str | None = None,
):
    """Return text, cards, and timings before the slower speech call."""

    language = _ui_language(ui_language)
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
            language,
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
            _trip_cards(None, language),
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
            _trip_cards(None, language),
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
            _trip_cards(None, language),
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
            _trip_cards(None, language),
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
            _trip_cards(None, language),
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
        _cards_for_result(result, language),
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
    ui_language: str | None = None,
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
        ui_language,
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
    ui_language: str | None = None,
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
    result = _respond_phase_one("", history, image_path, audio, ui_language)
    return (*result, audio_key)


def _clear(ui_language: str | None = None):
    return (
        [],
        "",
        _trip_cards(None, _ui_language(ui_language)),
        [],
        None,
    )


def _clear_voice(ui_language: str | None = None):
    return (*_clear(ui_language), None, None, "", None, None)


def build_app() -> gr.Blocks:
    """Build the Gradio UI without launching it."""

    copy = _ui_copy("English")
    with gr.Blocks(title="Koumba") as demo:
        # Loading the provider config also reads .env, so the voice URLs are
        # ready before the background wake-up starts.  There is no banner:
        # the demo should look like a phone app, not a developer console.
        _provider_config()
        _start_voice_wakeup()

        with gr.Column(elem_classes=["app-shell"]):
            with gr.Row(elem_classes=["hero-row"]):
                with gr.Column(scale=5, min_width=250):
                    header = gr.HTML(
                        value=_header_html("English"),
                        elem_id="app-header",
                        container=False,
                    )
                with gr.Column(scale=1, min_width=92):
                    ui_language = gr.Radio(
                        choices=["EN", "FR"],
                        value="EN",
                        show_label=False,
                        container=False,
                        scale=1,
                        min_width=92,
                        elem_id="ui-language",
                        elem_classes=["lang-switch"],
                    )

            chatbot = gr.Chatbot(
                value=[],
                label=copy["chat_label"],
                show_label=False,
                height=300,
                max_height=420,
                layout="bubble",
                placeholder=copy["message_label"],
                elem_id="chatbot",
            )
            reply_audio = gr.Audio(
                label=copy["voice_label"],
                show_label=False,
                autoplay=True,
                interactive=False,
                elem_id="voice-player",
                elem_classes=["voice-player"],
            )
            audio = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label=copy["mic_label"],
                show_label=False,
                elem_id="mic",
                elem_classes=["big-mic"],
                waveform_options={
                    "waveform_color": "#9bd4b5",
                    "waveform_progress_color": "#00853F",
                    "trim_region_color": "#FDEF42",
                },
            )
            voice_note = gr.Markdown(
                copy["voice_note"],
                elem_classes=["mic-note"],
            )

            with gr.Row(elem_classes=["composer"]):
                message = gr.Textbox(
                    label=copy["message_label"],
                    placeholder=copy["placeholder"],
                    show_label=False,
                    container=False,
                    lines=1,
                    max_lines=5,
                    scale=5,
                    elem_id="message",
                )
                send = gr.Button(
                    copy["send"],
                    variant="primary",
                    size="lg",
                    scale=1,
                    min_width=84,
                    elem_id="send",
                    elem_classes=["send-btn"],
                )
                clear = gr.Button(
                    copy["clear"],
                    variant="secondary",
                    size="lg",
                    scale=0,
                    min_width=74,
                    elem_id="clear",
                    elem_classes=["clear-btn"],
                )

            photo = gr.UploadButton(
                copy["photo_label"],
                file_types=["image"],
                file_count="single",
                type="filepath",
                variant="secondary",
                size="lg",
                elem_id="photo-button",
                elem_classes=["photo-button"],
            )
            photo_note = gr.Markdown(
                copy["photo_note"],
                elem_classes=["privacy-note"],
            )

            examples_heading = gr.Markdown(
                f"**{copy['examples_label']}**",
                elem_classes=["examples-heading"],
            )
            with gr.Row(elem_classes=["examples-row"]):
                example_wolof = gr.Button(
                    EXAMPLE_WOLOF,
                    size="sm",
                    elem_classes=["example-chip"],
                )
                example_medina = gr.Button(
                    EXAMPLE_WOLOF_MEDINA,
                    size="sm",
                    elem_classes=["example-chip"],
                )
                example_french = gr.Button(
                    EXAMPLE_FRENCH,
                    size="sm",
                    elem_classes=["example-chip"],
                )

            trips = gr.HTML(
                value=_trip_cards(None, "English"),
                label=copy["trips_label"],
                show_label=False,
                container=False,
                elem_id="trip-cards",
            )

            with gr.Accordion(
                copy["thinking_label"],
                open=False,
                elem_id="technical-panel",
                elem_classes=["technical-panel"],
            ):
                timings = gr.Markdown(value="", elem_classes=["timings"])
                thoughts = gr.JSON(
                    value=[],
                    label=copy["tools_label"],
                )
                gr.Markdown(
                    MODELS_LINE,
                    elem_id="models-used",
                    elem_classes=["models-used"],
                )

            gr.HTML(
                _footer_html(),
                elem_id="footer",
                container=False,
            )

            voice_guard = gr.State(None)
            voice_job = gr.State(None)

        language_outputs = [
            header,
            chatbot,
            reply_audio,
            audio,
            voice_note,
            message,
            send,
            clear,
            photo,
            photo_note,
            examples_heading,
            trips,
            thoughts,
        ]
        ui_language.change(
            _switch_language,
            inputs=[ui_language],
            outputs=language_outputs,
        )

        voice_outputs = [
            chatbot,
            message,
            trips,
            thoughts,
            photo,
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
                inputs=[message, chatbot, photo, audio, ui_language],
                outputs=voice_outputs,
            )
        )
        add_voice_phase(
            message.submit(
                _respond_phase_one,
                inputs=[message, chatbot, photo, audio, ui_language],
                outputs=voice_outputs,
            )
        )
        # Only the microphone stop event auto-sends.  Do not also bind change:
        # Gradio emits change when recording ends and again when we clear the
        # audio component, which would send the same question twice.
        add_voice_phase(
            audio.stop_recording(
                _respond_to_recording,
                inputs=[chatbot, photo, audio, voice_guard, ui_language],
                outputs=[*voice_outputs, voice_guard],
            )
        )
        add_voice_phase(
            example_wolof.click(
                lambda history, audio, ui_language: _respond_phase_one(
                    EXAMPLE_WOLOF,
                    history,
                    None,
                    audio,
                    ui_language,
                ),
                inputs=[chatbot, audio, ui_language],
                outputs=voice_outputs,
            )
        )
        add_voice_phase(
            example_medina.click(
                lambda history, audio, ui_language: _respond_phase_one(
                    EXAMPLE_WOLOF_MEDINA,
                    history,
                    None,
                    audio,
                    ui_language,
                ),
                inputs=[chatbot, audio, ui_language],
                outputs=voice_outputs,
            )
        )
        add_voice_phase(
            example_french.click(
                lambda history, audio, ui_language: _respond_phase_one(
                    EXAMPLE_FRENCH,
                    history,
                    None,
                    audio,
                    ui_language,
                ),
                inputs=[chatbot, audio, ui_language],
                outputs=voice_outputs,
            )
        )
        clear.click(
            _clear_voice,
            inputs=[ui_language],
            outputs=[*voice_outputs, voice_guard],
        )

    return demo


if __name__ == "__main__":
    # Gradio 6 moved the CSS argument from Blocks to launch().
    build_app().launch(css=CSS)

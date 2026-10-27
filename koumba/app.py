"""A small Gradio demo for the Koumba agent.

Run it manually with ``python -m koumba.app``.  Importing this module or
calling ``build_app()`` only builds the UI; it never starts a server and
never calls the model.
"""

from __future__ import annotations

import os
from functools import lru_cache
from html import escape
from typing import Any

# Keep Gradio from sending usage telemetry.  This is set before the import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from .agent import Agent
from .data import load_data
from .llm import ModelClient, ProviderConfig, api_key_configured, load_config
from .network import Network
from .phrases import compose_chat_message
from .places import PlaceIndex
from .router import RouteEngine

EXAMPLE_WOLOF = "Maa ngi Sahm, bëgg naa dem Malika"
EXAMPLE_FRENCH = "Je suis à Ouakam, je vais au Palais"

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
.muted { color: #64748b; font-size: 13px; margin-top: 2px; }
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
def _agent() -> Agent:
    data = load_data()
    network = Network.from_data(data)
    place_index = PlaceIndex.from_data(data)
    engine = RouteEngine(network, place_index)
    return Agent(ModelClient(), engine, place_index)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _minutes(value: Any) -> str:
    try:
        return f"{float(value):.0f} min"
    except (TypeError, ValueError):
        return "unknown time"


def _trip_cards(trips: list[dict[str, Any]] | None) -> str:
    """Render the trips as simple mobile-friendly cards."""

    if not trips:
        return (
            "<div class='empty'>No trip cards yet. Ask Koumba where you "
            "are and where you want to go.</div>"
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
            stops = _plural(int(leg.get("num_stops", 0)), "stop")
            minutes = _minutes(leg.get("minutes"))
            legs_html.append(
                "<div class='leg'>"
                f"<div class='line'>{operator} {line}</div>"
                f"<div>{board} &rarr; {alight}</div>"
                f"<div class='muted'>towards {direction} &middot; "
                f"{stops} &middot; {minutes}</div>"
                "</div>"
            )

        walk_notes = [
            f"{escape(str(walk.get('kind', 'walk')))}: "
            f"{escape(str(walk.get('from_place', '')))} &rarr; "
            f"{escape(str(walk.get('to_place', '')))} "
            f"({_minutes(walk.get('minutes'))})"
            for walk in trip.get("walks", [])
            if float(walk.get("minutes") or 0) > 0
        ]
        walk_html = (
            f"<div class='muted'>Walk: {'; '.join(walk_notes)}</div>"
            if walk_notes
            else ""
        )

        transfers = int(trip.get("transfers", 0))
        cards.append(
            "<div class='trip-card'>"
            f"<div class='trip-head'>Trip {index}"
            f"<span class='pill'>{_plural(transfers, 'transfer')}</span>"
            f"<span class='pill'>about {_minutes(trip.get('total_minutes'))}</span>"
            "</div>"
            f"{''.join(legs_html)}"
            f"{walk_html}"
            "</div>"
        )

    return "<div class='trip-grid'>" + "".join(cards) + "</div>"


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


def _respond(message_text: str, history, image_path: str | None = None):
    """Handle one chat submission and return the five UI values."""

    text = (message_text or "").strip()
    previous = _history_as_dicts(history)
    current = list(previous)
    if not text:
        return current, "", gr.update(), gr.update(), image_path

    current.append({"role": "user", "content": text})

    if not _api_key_configured():
        current.append(
            {
                "role": "assistant",
                "content": _missing_key_message(),
            }
        )
        return current, "", _trip_cards(None), [], None

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
        return current, "", _trip_cards(None), [], None

    current.append(
        {
            "role": "assistant",
            "content": compose_chat_message(result.spoken, result.reply),
        }
    )
    return (
        current,
        "",
        _trip_cards(result.trips),
        result.tool_calls,
        None,
    )


def _clear():
    return [], "", _trip_cards(None), [], None


def build_app() -> gr.Blocks:
    """Build the Gradio UI without launching it."""

    with gr.Blocks(title="Koumba") as demo:
        gr.Markdown(
            "# Koumba\n"
            "Your Dakar bus assistant. Ask in Wolof, French, or English."
        )

        banner_text, is_warning = _provider_banner()
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
                message = gr.Textbox(
                    label="Where are you going?",
                    placeholder=(
                        "Maa ngi Sahm, bëgg naa dem Malika / "
                        "Je suis à Ouakam, je vais au Palais"
                    ),
                    lines=2,
                )
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
                    thoughts = gr.JSON(value=[], label="Tool calls")

        outputs = [chatbot, message, trips, thoughts, image]
        send.click(
            _respond,
            inputs=[message, chatbot, image],
            outputs=outputs,
        )
        message.submit(
            _respond,
            inputs=[message, chatbot, image],
            outputs=outputs,
        )
        example_wolof.click(
            lambda history: _respond(EXAMPLE_WOLOF, history),
            inputs=[chatbot],
            outputs=outputs,
        )
        example_french.click(
            lambda history: _respond(EXAMPLE_FRENCH, history),
            inputs=[chatbot],
            outputs=outputs,
        )
        clear.click(_clear, outputs=outputs)

    return demo


if __name__ == "__main__":
    # Gradio 6 moved the CSS argument from Blocks to launch().
    build_app().launch(css=CSS)

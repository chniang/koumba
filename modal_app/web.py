"""Modal service for the public Koumba Gradio demo.

This is a thin wrapper around ``koumba.app``.  It runs the same UI as the local
app, keeps the bus CSV cache in a Volume, and uses the ASR/TTS services only
through their configured HTTPS URLs.
"""

from __future__ import annotations

import os

import gradio as gr
import modal
from fastapi import FastAPI

from koumba.app import CSS, build_app

APP_NAME = "koumba-web"
CACHE_PATH = "/root/data_cache"
DATA_PATH = "/root/data"
WEB_CACHE_NAME = "koumba-web-cache"
SECRET_NAME = "koumba-app"
SECRET_KEYS = (
    "BASETEN_API_KEY",
    "KOUMBA_BASE_URL",
    "KOUMBA_API_KEY_ENV",
    "KOUMBA_MODEL",
    "KOUMBA_ASR_URL",
    "KOUMBA_ASR_KEY",
    "KOUMBA_TTS_URL",
)
MAX_CONTAINERS = 1
MAX_INPUTS = 100
CPU = 1.0
MEMORY = 2048
SCALEDOWN_WINDOW = 600
TIMEOUT = 600


def _read_keep_warm() -> int:
    value = (os.environ.get("KOUMBA_KEEP_WARM") or "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("KOUMBA_KEEP_WARM must be '0' or '1'.")
    return int(value)


MIN_CONTAINERS = _read_keep_warm()

app = modal.App(APP_NAME)
web_cache = modal.Volume.from_name(
    WEB_CACHE_NAME,
    create_if_missing=True,
)
web_secret = modal.Secret.from_name(
    SECRET_NAME,
    required_keys=list(SECRET_KEYS),
)

# The same pins as the local app keep Gradio and the data stack predictable.
# The local data folder is separate from the Volume-backed CSV cache.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("koumba")
    .add_local_dir("data", DATA_PATH)
)


def create_asgi_app():
    """Build the FastAPI wrapper around the real Gradio UI."""

    # build_app also starts the background ASR/TTS wake-up thread.  Keeping
    # that call here means the voice services warm up with the web container.
    demo = build_app()
    return gr.mount_gradio_app(
        FastAPI(title="Koumba"),
        demo,
        path="/",
        css=CSS,
        footer_links=[],
    )


@app.function(
    image=image,
    cpu=CPU,
    memory=MEMORY,
    volumes={CACHE_PATH: web_cache},
    secrets=[web_secret],
    env={"KOUMBA_DATA_DIR": CACHE_PATH},
    max_containers=MAX_CONTAINERS,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def web():
    """Return the Gradio app mounted at the site root."""

    return create_asgi_app()

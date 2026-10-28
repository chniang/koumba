"""Modal service for Kiriku Wolof text-to-speech.

Deploy this file with Modal.  The Koumba app only calls its HTTPS endpoint;
the local app never imports Modal or loads the TTS model.  The image is kept
separate from the ASR image so the two dependency trees cannot disturb each
other.  The model is downloaded once into a Modal Volume and the Synthesizer
is loaded once per container.
"""

import contextlib
import io
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path

import modal
from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

APP_NAME = "koumba-tts"
MODEL_ID = "AIHubSN/Kiriku-Wolof-TTS"
MODEL_NAME = "Kiriku-Wolof-TTS"
CACHE_ROOT = "/cache"
MODEL_DIR = f"{CACHE_ROOT}/kiriku-wolof-tts"
MAX_TEXT_CHARS = 400

# These pins match TTS 0.19.0's Python 3.10 era.  torch and torchaudio must
# stay on the same release, and TTS itself pins NumPy 1.22 on this Python.
TORCH_VERSION = "2.1.2"
NUMPY_VERSION = "1.22.0"
TRANSFORMERS_VERSION = "4.46.3"

logger = logging.getLogger("koumba-tts")


class SpeakIn(BaseModel):
    """The validated JSON body for one speech request."""

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name(
    "koumba-tts-cache",
    create_if_missing=True,
)
hf_secret = modal.Secret.from_name(
    "koumba-hf",
    required_keys=["HF_TOKEN"],
)
auth_secret = modal.Secret.from_name(
    "koumba-asr",
    required_keys=["KOUMBA_ASR_KEY"],
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install(
        f"torch=={TORCH_VERSION}+cpu",
        f"torchaudio=={TORCH_VERSION}+cpu",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        f"numpy=={NUMPY_VERSION}",
        "TTS==0.19.0",
        "fastapi[standard]",
    )
    # TTS 0.19.0's metadata pins transformers 4.33.*, but the Kiriku model
    # card asks for 4.46.3.  Install it in a separate layer so pip upgrades
    # its shared dependencies instead of trying to solve both pins at once.
    .pip_install(f"transformers=={TRANSFORMERS_VERSION}")
    .run_commands(
        'python -c "import numpy, torch, torchaudio, transformers; '
        "from TTS.utils.synthesizer import Synthesizer; "
        "print('numpy', numpy.__version__, 'torch', torch.__version__, "
        "'torchaudio', torchaudio.__version__, 'transformers', "
        "transformers.__version__); "
        "assert numpy.__version__ == '1.22.0'; "
        "assert torch.__version__.split('+')[0] == '2.1.2'; "
        "assert torchaudio.__version__.split('+')[0] == '2.1.2'; "
        "assert transformers.__version__ == '4.46.3'\""
    )
)


def create_api(
    synthesize: Callable[[str], bytes],
    authorize: Callable[[str | None], bool],
) -> FastAPI:
    """Build the ASGI app with FastAPI annotations resolved at runtime."""

    api = FastAPI(title="Koumba TTS", version="1.0")

    @api.get("/health")
    def health(
        x_koumba_key: str | None = Header(
            default=None,
            alias="X-Koumba-Key",
        ),
    ) -> dict[str, bool | str]:
        if not authorize(x_koumba_key):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"ok": True, "model": MODEL_NAME}

    @api.post("/speak")
    async def speak(
        payload: SpeakIn,
        x_koumba_key: str | None = Header(
            default=None,
            alias="X-Koumba-Key",
        ),
    ) -> Response:
        if not authorize(x_koumba_key):
            raise HTTPException(status_code=401, detail="Unauthorized")

        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is empty")

        # Keep Wolof letters and digits, but the model speaks lower case.
        text = text.lower()
        try:
            audio = await run_in_threadpool(synthesize, text)
        except Exception as exc:
            logger.warning(
                "Speech synthesis failed: %s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail="Speech synthesis failed",
            ) from exc

        return Response(content=audio, media_type="audio/wav")

    return api


@app.cls(
    image=image,
    cpu=4.0,
    memory=8192,
    volumes={CACHE_ROOT: model_cache},
    secrets=[hf_secret, auth_secret],
    scaledown_window=300,
    timeout=120,
    startup_timeout=900,
)
class KoumbaTTS:
    @modal.enter()
    def load(self) -> None:
        """Download the gated model once, then load the Synthesizer."""

        from huggingface_hub import snapshot_download
        from TTS.utils.synthesizer import Synthesizer

        token = (os.environ.get("HF_TOKEN") or "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is missing from the koumba-hf secret.")

        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = model_dir / "model.pth"
        config = model_dir / "config.json"
        cached = (
            checkpoint.is_file()
            and checkpoint.stat().st_size > 0
            and config.is_file()
            and config.stat().st_size > 0
        )
        if not cached:
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=str(model_dir),
                token=token,
            )
            # Commit only after a complete download.  A later crash cannot
            # leave the volume with a half-written checkpoint.
            model_cache.commit()

        self.synthesizer = Synthesizer(
            tts_checkpoint=str(checkpoint),
            tts_config_path=str(config),
        )
        # The web workers share four CPUs, so serialize model calls.
        self.synthesis_lock = threading.Lock()
        logger.info(
            "Loaded %s at %s (sample_rate=%s)",
            MODEL_NAME,
            model_dir,
            self.synthesizer.output_sample_rate,
        )

    def _authorized(self, supplied_key: str | None) -> bool:
        expected = (os.environ.get("KOUMBA_ASR_KEY") or "").strip()
        if not expected or not supplied_key:
            return False
        return secrets.compare_digest(expected, supplied_key)

    def _synthesize(self, text: str) -> bytes:
        """Return WAV bytes without writing text or audio to disk."""

        import numpy as np
        import soundfile as sf

        started = time.perf_counter()
        with self.synthesis_lock:
            # TTS prints the sentence while it splits text.  Redirect that
            # chatter so user text never reaches the service logs.
            with contextlib.redirect_stdout(io.StringIO()):
                waveform = self.synthesizer.tts(text=text)
        elapsed = time.perf_counter() - started

        sample_rate = int(self.synthesizer.output_sample_rate)
        with io.BytesIO() as buffer:
            sf.write(
                buffer,
                np.asarray(waveform, dtype=np.float32),
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            audio = buffer.getvalue()

        logger.info(
            "Synthesis time: %.3fs (chars=%d, sample_rate=%d)",
            elapsed,
            len(text),
            sample_rate,
        )
        return audio

    @modal.asgi_app()
    def web(self) -> FastAPI:
        """Return the small FastAPI app that Modal exposes over HTTPS."""

        return create_api(self._synthesize, self._authorized)

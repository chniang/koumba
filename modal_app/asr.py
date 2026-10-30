"""Modal service for Meta Omnilingual ASR.

Deploy this file with Modal; the Koumba app only calls its HTTP endpoint.
The model is loaded once per container and its weights stay in a Modal
Volume, so a cold container does not download them again.
"""

import io
import logging
import os
import secrets
import tempfile
import threading
import time
import wave
from pathlib import Path

import modal

APP_NAME = "koumba-asr"
DEFAULT_MODEL = "omniASR_CTC_1B"
ALLOWED_MODELS = {"omniASR_CTC_1B", "omniASR_CTC_300M"}
DEFAULT_LANG = "wol_Latn"
CACHE_PATH = "/cache/fairseq2"

ASR_DEVICE = (os.environ.get("KOUMBA_ASR_DEVICE") or "cpu").strip().lower()
if ASR_DEVICE not in {"cpu", "gpu"}:
    raise ValueError("KOUMBA_ASR_DEVICE must be 'cpu' or 'gpu'.")

ASR_MODEL = (
    os.environ.get("KOUMBA_ASR_MODEL") or DEFAULT_MODEL
).strip()
if ASR_MODEL not in ALLOWED_MODELS:
    allowed = ", ".join(sorted(ALLOWED_MODELS))
    raise ValueError(f"KOUMBA_ASR_MODEL must be one of: {allowed}.")

if ASR_DEVICE == "gpu":
    hardware = {"gpu": ["L4", "A10G"]}
else:
    hardware = {"cpu": 8.0, "memory": 16384}


def _read_keep_warm() -> int:
    value = (os.environ.get("KOUMBA_KEEP_WARM") or "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("KOUMBA_KEEP_WARM must be '0' or '1'.")
    return int(value)


MIN_CONTAINERS = _read_keep_warm()

logger = logging.getLogger("koumba-asr")

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("koumba-asr-cache", create_if_missing=True)
asr_secret = modal.Secret.from_name(
    "koumba-asr",
    required_keys=["KOUMBA_ASR_KEY"],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "omnilingual-asr==0.1.0",
        "fastapi[standard]",
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "numpy==1.26.4",
        "pyarrow==25.0.1",
        "pandas==2.3.3",
        "polars==1.44.2",
    )
    .run_commands(
        'python -c "import numpy, pyarrow, torch, torchaudio, fairseq2n; '
        "from omnilingual_asr.models.inference.pipeline import "
        "ASRInferencePipeline; "
        "from omnilingual_asr.models.wav2vec2_llama.lang_ids import "
        "supported_langs; "
        "print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__, "
        "'torch', torch.__version__); "
        "print('wol_Latn' in supported_langs); "
        "assert torch.__version__ == '2.8.0+cu128'; "
        "assert torchaudio.__version__ == '2.8.0+cu128'; "
        "assert 'wol_Latn' in supported_langs\""
    )
)


@app.cls(
    image=image,
    **hardware,
    volumes={"/cache": model_cache},
    secrets=[asr_secret],
    # Deploy-time choices must also reach the container's module globals.
    env={
        "FAIRSEQ2_CACHE_DIR": CACHE_PATH,
        "KOUMBA_ASR_DEVICE": ASR_DEVICE,
        "KOUMBA_ASR_MODEL": ASR_MODEL,
    },
    scaledown_window=300,
    timeout=120,
    startup_timeout=900,
    max_containers=1,
    min_containers=MIN_CONTAINERS,
)
class KoumbaASR:
    @modal.enter()
    def load(self) -> None:
        """Load the model and language list once per container."""

        import torch
        from omnilingual_asr.models.inference.pipeline import (
            ASRInferencePipeline,
        )
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import (
            supported_langs,
        )

        Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)
        self.supported_langs = set(supported_langs)
        self.languages = {
            "wol_Latn": "wol_Latn" in self.supported_langs,
            "fra_Latn": "fra_Latn" in self.supported_langs,
        }
        logger.info(
            "Omnilingual language support: wol_Latn=%s fra_Latn=%s",
            self.languages["wol_Latn"],
            self.languages["fra_Latn"],
        )
        if not self.languages["wol_Latn"]:
            logger.error("This Omnilingual ASR build does not list wol_Latn")

        device = "cuda" if ASR_DEVICE == "gpu" else "cpu"
        if device == "cuda":
            self.pipeline = ASRInferencePipeline(
                model_card=ASR_MODEL,
                device=device,
            )
        else:
            self.pipeline = ASRInferencePipeline(
                model_card=ASR_MODEL,
                device=device,
                dtype=torch.float32,
            )
        # Commit the downloaded weights now, so a crash later cannot lose them.
        model_cache.commit()
        # The web workers share one CPU allocation or GPU, so serialize calls.
        self.inference_lock = threading.Lock()
        logger.info(
            "Loaded %s: torch=%s, device=%s, wol_Latn=%s",
            ASR_MODEL,
            torch.__version__,
            device,
            self.languages["wol_Latn"],
        )

    def _authorized(self, supplied_key: str | None) -> bool:
        expected = (os.environ.get("KOUMBA_ASR_KEY") or "").strip()
        if not expected or not supplied_key:
            return False
        return secrets.compare_digest(expected, supplied_key)

    def _wav_seconds(self, audio_bytes: bytes) -> float:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
        except (wave.Error, EOFError) as exc:
            raise ValueError("The request body must be a PCM WAV file.") from exc
        if rate <= 0:
            raise ValueError("The WAV file has no usable sample rate.")
        return round(frames / rate, 3)

    def _transcribe(
        self,
        audio_bytes: bytes,
        lang: str,
    ) -> tuple[str, float, float]:
        seconds = self._wav_seconds(audio_bytes)
        path = ""

        # The pipeline wants a path, so use a short-lived temp file and erase
        # it even when inference fails.
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as audio_file:
                path = audio_file.name
                audio_file.write(audio_bytes)

            with self.inference_lock:
                started = time.perf_counter()
                texts = self.pipeline.transcribe([path], lang=[lang])
                infer_seconds = round(time.perf_counter() - started, 3)

            text = str(texts[0] if texts else "").strip()
            if not text:
                raise ValueError("The model returned an empty transcript.")
            logger.info(
                "Transcription time: %.3fs (lang=%s, audio=%.3fs, "
                "model=%s, device=%s)",
                infer_seconds,
                lang,
                seconds,
                ASR_MODEL,
                ASR_DEVICE,
            )
            return text, seconds, infer_seconds
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    @modal.asgi_app()
    def web(self):
        """Return the small FastAPI app that Modal exposes over HTTPS."""

        from fastapi import FastAPI, Header, HTTPException, Query, Request
        from fastapi.concurrency import run_in_threadpool

        api = FastAPI(title="Koumba ASR", version="1.0")

        @api.get("/languages")
        def languages() -> dict[str, bool | str]:
            return {**self.languages, "model": ASR_MODEL}

        @api.post("/transcribe")
        async def transcribe(
            request: Request,
            lang: str = Query(default=DEFAULT_LANG),
            x_koumba_key: str | None = Header(
                default=None,
                alias="X-Koumba-Key",
            ),
        ) -> dict[str, str | float]:
            if not self._authorized(x_koumba_key):
                raise HTTPException(status_code=401, detail="Unauthorized")
            if lang not in self.supported_langs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {lang}",
                )

            audio_bytes = await request.body()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio body")

            try:
                text, seconds, infer_seconds = await run_in_threadpool(
                    self._transcribe,
                    audio_bytes,
                    lang,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Omnilingual ASR failed")
                raise HTTPException(
                    status_code=500,
                    detail="Transcription failed",
                ) from exc

            return {
                "text": text,
                "lang": lang,
                "seconds": seconds,
                "infer_seconds": infer_seconds,
                "model": ASR_MODEL,
            }

        return api

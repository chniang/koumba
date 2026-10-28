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
import wave
from pathlib import Path

import modal

APP_NAME = "koumba-asr"
MODEL_CARD = "omniASR_CTC_1B"
DEFAULT_LANG = "wol_Latn"
CACHE_PATH = "/cache/fairseq2"

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
    .pip_install("omnilingual-asr==0.1.0", "fastapi[standard]")
)


@app.cls(
    image=image,
    gpu=["L4", "A10G"],
    volumes={"/cache": model_cache},
    secrets=[asr_secret],
    env={"FAIRSEQ2_CACHE_DIR": CACHE_PATH},
    scaledown_window=300,
    timeout=120,
    startup_timeout=900,
    max_containers=1,
)
class KoumbaASR:
    @modal.enter()
    def load(self) -> None:
        """Load the model and language list once per container."""

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

        self.pipeline = ASRInferencePipeline(
            model_card=MODEL_CARD,
            device="cuda",
        )
        # Commit the downloaded weights now, so a crash later cannot lose them.
        model_cache.commit()
        # One GPU is shared by the web workers, so serialize model calls.
        self.gpu_lock = threading.Lock()
        logger.info("Loaded %s on the Modal GPU", MODEL_CARD)

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

    def _transcribe(self, audio_bytes: bytes, lang: str) -> tuple[str, float]:
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

            with self.gpu_lock:
                texts = self.pipeline.transcribe([path], lang=[lang])

            text = str(texts[0] if texts else "").strip()
            if not text:
                raise ValueError("The model returned an empty transcript.")
            return text, seconds
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
        def languages() -> dict[str, bool]:
            return dict(self.languages)

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
                text, seconds = await run_in_threadpool(
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
                "model": MODEL_CARD,
            }

        return api

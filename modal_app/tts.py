"""Modal service for Oolel-Voices Wolof text-to-speech.

The app sends one complete Wolof sentence and gets one WAV back.  Native review
found that short, isolated words sound much worse than full sentences with their
normal punctuation, so this service does not concatenate lexicon clips.  The
model is baked into the image and loaded during the build once, so a missing
runtime module fails the deploy instead of crash-looping every container.  The
voice prompt and finished sentences are cached in a Modal Volume.

Deploy this file with Modal.  The local app only calls the HTTPS endpoint and
never imports Modal or loads the model.  The image is separate from the ASR
image so the two dependency trees cannot disturb each other.
"""

import contextlib
import hashlib
import io
import logging
import os
import random
import re
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
MODEL_ID = "soynade-research/Oolel-Voices"
MODEL_NAME = "Oolel-Voices"
VOICE_REPO_ID = "soynade-research/Oolel-Voices-Demo"
VOICE_FILENAME = "8_1_c.wav"
CACHE_ROOT = "/cache"
MODEL_DIR = "/opt/oolel-voices"
VOICE_PROMPT_PATH = f"{CACHE_ROOT}/{VOICE_FILENAME}"
SPEECH_DIR = f"{CACHE_ROOT}/speech"
MAX_TEXT_CHARS = 400
SEED = 2

TORCH_VERSION = "2.8.0"
TRANSFORMERS_VERSION = "4.46.3"
DIFFUSERS_VERSION = "0.29.0"
CONFORMER_VERSION = "0.3.2"
TORCHCODEC_VERSION = "0.7.0"

logger = logging.getLogger("koumba-tts")

_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*\b")


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


def _preload_oolel_model() -> None:
    """Download and import the model while the image is being built."""

    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the koumba-hf secret.")

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_dir),
        token=token,
    )
    # The remote code imports the actual runtime modules.  Loading here makes a
    # missing dependency fail this build step instead of every container start.
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    del model
    print(f"Oolel build check passed: {model_dir}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchaudio=={TORCH_VERSION}",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        f"transformers=={TRANSFORMERS_VERSION}",
        f"diffusers=={DIFFUSERS_VERSION}",
        f"conformer=={CONFORMER_VERSION}",
        "s3tokenizer",
        "librosa>=0.10.2",
        "numpy>=2.0",
        "num2words",
        "soundfile",
        f"torchcodec=={TORCHCODEC_VERSION}",
        "huggingface_hub",
        "fastapi[standard]",
        "omegaconf",
        "einops",
        "safetensors",
        "scipy",
        "resampy",
        "pyloudnorm",
        "resemble-perth",
    )
    .run_commands(
        "python -c \"import transformers, diffusers, s3tokenizer; "
        "print('transformers', transformers.__version__); "
        "print('diffusers', diffusers.__version__); "
        "print('s3tokenizer', getattr(s3tokenizer, '__version__', 'unknown'))\""
    )
    .run_function(
        _preload_oolel_model,
        secrets=[hf_secret],
        cpu=8.0,
        memory=16384,
        timeout=900,
    )
)


def convert_numbers_to_french(text: str) -> str:
    """Replace digits with French words, as the official Oolel demo does."""

    from num2words import num2words

    def replace_match(match: re.Match[str]) -> str:
        raw = match.group(0)
        normalized = raw.replace(",", ".")
        parts = normalized.split(".")
        try:
            if len(parts) == 2:
                number: float | int = float(normalized)
            else:
                number = int(parts[0])
            return num2words(number, lang="fr")
        except (ValueError, NotImplementedError):
            return raw

    return _NUMBER_PATTERN.sub(replace_match, str(text or ""))


def set_seed(seed: int = SEED) -> None:
    """Seed the three random sources used by the model."""

    import numpy as np
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    cpu=8.0,
    memory=16384,
    volumes={CACHE_ROOT: model_cache},
    secrets=[hf_secret, auth_secret],
    scaledown_window=300,
    timeout=180,
    startup_timeout=900,
)
class KoumbaTTS:
    @modal.enter()
    def load(self) -> None:
        """Load the baked-in model and fetch the voice prompt if needed."""

        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        token = (os.environ.get("HF_TOKEN") or "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is missing from the koumba-hf secret.")

        model_dir = Path(MODEL_DIR)
        if not (model_dir / "config.json").is_file():
            raise RuntimeError(
                f"{MODEL_NAME} is missing from the image at {model_dir}."
            )

        voice_path = Path(VOICE_PROMPT_PATH)
        downloaded = False
        if not voice_path.is_file():
            hf_hub_download(
                repo_id=VOICE_REPO_ID,
                filename=VOICE_FILENAME,
                repo_type="space",
                local_dir=CACHE_ROOT,
                token=token,
            )
            downloaded = True

        Path(SPEECH_DIR).mkdir(parents=True, exist_ok=True)
        if downloaded:
            model_cache.commit()

        self.model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self.voice_prompt = str(voice_path)
        # The web workers share the container, so serialize CPU generation.
        self.synthesis_lock = threading.Lock()
        logger.info(
            "Loaded %s at %s (sample_rate=%s)",
            MODEL_NAME,
            model_dir,
            self.model.sr,
        )

    def _authorized(self, supplied_key: str | None) -> bool:
        expected = (os.environ.get("KOUMBA_ASR_KEY") or "").strip()
        if not expected or not supplied_key:
            return False
        return secrets.compare_digest(expected, supplied_key)

    def _synthesize(self, text: str) -> bytes:
        """Return cached or freshly generated WAV bytes for one sentence."""

        import numpy as np
        import soundfile as sf
        import torch

        final_text = convert_numbers_to_french(text)
        digest = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        cache_path = Path(SPEECH_DIR) / f"{digest}.wav"

        with self.synthesis_lock:
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                logger.info(
                    "Speech cache hit (chars=%d)",
                    len(final_text),
                )
                return cache_path.read_bytes()

            started = time.perf_counter()
            set_seed()
            # The model may print the sentence while preparing it.  Keep user
            # text out of the service logs.
            with contextlib.redirect_stdout(io.StringIO()):
                with torch.inference_mode():
                    waveform = self.model.generate(
                        final_text,
                        audio_prompt_path=self.voice_prompt,
                        exaggeration=0.3,
                        temperature=0.05,
                        cfg_weight=0.5,
                    )
            elapsed = time.perf_counter() - started

            samples = waveform.squeeze(0).detach().cpu().float().numpy()
            with io.BytesIO() as buffer:
                sf.write(
                    buffer,
                    np.asarray(samples, dtype=np.float32),
                    int(self.model.sr),
                    format="WAV",
                    subtype="PCM_16",
                )
                audio = buffer.getvalue()

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_bytes(audio)
            os.replace(temporary, cache_path)

        logger.info(
            "Synthesis time: %.3fs (chars=%d, sample_rate=%d)",
            elapsed,
            len(final_text),
            int(self.model.sr),
        )
        return audio

    @modal.asgi_app()
    def web(self) -> FastAPI:
        """Return the small FastAPI app that Modal exposes over HTTPS."""

        return create_api(self._synthesize, self._authorized)

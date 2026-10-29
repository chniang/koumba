# Koumba ASR and TTS on Modal

This folder contains two separate Modal services: Meta Omnilingual ASR for
voice input and Oolel-Voices for spoken Wolof replies. They are separate from
the Koumba app: the app only posts WAV bytes or text to the deployed HTTPS URLs.
The local app never imports Modal or loads either model.

The ASR service uses Meta's Apache-2.0 Omnilingual ASR and defaults to CPU
because Modal GPU functions require a payment method. CPU mode uses 8 cores and
16 GiB
of RAM; GPU mode uses an L4 with A10G fallback. The model defaults to
`omniASR_CTC_1B`; set `omniASR_CTC_300M` for faster CPU inference. Both choices
are read at deploy time, and the pipeline is loaded once per container in
`@modal.enter`. The `koumba-asr-cache` Volume caches the model weights. Both
device modes install matching `torch`/`torchaudio` 2.8.0 CUDA 12.8 wheels,
as required by the `fairseq2` 0.6 runtime; those wheels also run on CPU-only
containers. The data stack is pinned to NumPy 1.26, PyArrow 25, pandas 2.3,
and polars 1.44 to keep the runtime compatible with fairseq2 0.6.

## 1. Create the shared secret

Run this from the repository root. Use any long random value.

```powershell
.\.venv\Scripts\python.exe -m modal secret create koumba-asr "KOUMBA_ASR_KEY=<random>"
```

Keep that value for `.env`. Do not commit it.

## 2. Deploy

CPU is the default and does not require a payment method:

```powershell
$env:KOUMBA_ASR_DEVICE="cpu"
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
```

For faster CPU inference, select the smaller model before deploying. Leave
`KOUMBA_ASR_MODEL` unset (or empty) to use the default `omniASR_CTC_1B`:

```powershell
$env:KOUMBA_ASR_MODEL="omniASR_CTC_300M"
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
```

To switch to GPU later, after adding a payment method to Modal:

```powershell
$env:KOUMBA_ASR_DEVICE="gpu"
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
```

Deploy-time choices are captured in the deployment. Change the PowerShell
variables and redeploy to apply a different device or model. To return to the
CPU default explicitly:

```powershell
$env:KOUMBA_ASR_DEVICE="cpu"
Remove-Item Env:KOUMBA_ASR_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
```

Modal prints the web URL. Put it in `.env` without `/transcribe`:

```text
KOUMBA_ASR_URL=https://your-workspace--koumba-asr-web.modal.run
KOUMBA_ASR_KEY=<the same random value>
```

The first container downloads the model into the Volume. Later containers
reuse the cached weights. The first request after the idle window is slower
because the container must start again; CPU inference is also slower than GPU
inference, especially with the 1B model.

## 3. Test

Load the same values into the current PowerShell session:

```powershell
$env:KOUMBA_ASR_URL = "https://your-workspace--koumba-asr-web.modal.run"
$env:KOUMBA_ASR_KEY = "<the same random value>"
```

Check the supported languages:

```powershell
curl.exe "$env:KOUMBA_ASR_URL/languages"
```

Expected:

```json
{"wol_Latn":true,"fra_Latn":true,"model":"omniASR_CTC_1B"}
```

The `model` value reflects `KOUMBA_ASR_MODEL` from the deployment.

Transcribe a PCM WAV:

```powershell
curl.exe -X POST "$env:KOUMBA_ASR_URL/transcribe?lang=wol_Latn" `
  -H "X-Koumba-Key: $env:KOUMBA_ASR_KEY" `
  -H "Content-Type: audio/wav" `
  --data-binary "@sample.wav"
```

Expected shape:

```json
{"text":"...","lang":"wol_Latn","seconds":2.4,"infer_seconds":1.8,"model":"omniASR_CTC_1B"}
```

`seconds` is the audio duration; `infer_seconds` is the measured transcription
time. Each request also logs the language, audio duration, inference time,
model, and device. A missing or wrong key returns HTTP 401. Requests without a
WAV body return HTTP 400. The temporary WAV is deleted after inference; no
audio is retained.

For the full local flow (Modal ASR, Muse Spark cleanup, then the route agent):

```powershell
.\.venv\Scripts\python.exe scripts\try_voice.py path\to\sample.wav
```

## Wolof TTS service

`modal_app/tts.py` serves
[`soynade-research/Oolel-Voices`](https://huggingface.co/soynade-research/Oolel-Voices),
a Wolof TTS model. The service uses the official demo's female voice prompt,
`8_1_c.wav`. Oolel-Voices is licensed
[AGPL-3.0](https://huggingface.co/soynade-research/Oolel-Voices); keep that
license in mind if you redistribute the service. The model is downloaded into
the image and loaded once during the build, so a missing runtime dependency
fails the deploy instead of crash-looping the service. The voice prompt and
finished speech are cached in the `koumba-tts-cache` Volume.

The service runs on CPU with 8 cores and 16 GiB of RAM. Native review found
that short, isolated words sound much worse than complete sentences, so the app
sends the whole displayed Wolof sentence with its normal punctuation. The
service does not concatenate lexicon clips. Generation is deterministic: it
seeds `torch`, `random`, and NumPy with `2`, then calls `generate` with
exaggeration `0.3`, temperature `0.05`, and CFG weight `0.5`. Digits are
converted to French words with `num2words`, matching the official demo. A
sentence takes roughly 6 seconds on two CPU cores.

The TTS endpoint uses the same `koumba-asr` secret and `KOUMBA_ASR_KEY` as the
ASR endpoint. Create a separate Hugging Face secret for the downloads:

```powershell
.\.venv\Scripts\python.exe -m modal secret create koumba-hf "HF_TOKEN=<your-token>"
```

Deploy it:

```powershell
.\.venv\Scripts\python.exe -m modal deploy modal_app/tts.py
```

Modal prints the web URL. Add it to `.env` without `/speak`:

```text
KOUMBA_TTS_URL=https://your-workspace--koumba-tts-web.modal.run
KOUMBA_ASR_KEY=<the same random value>
```

Load the values into the current PowerShell session and check the service:

```powershell
$env:KOUMBA_TTS_URL = "https://your-workspace--koumba-tts-web.modal.run"
$env:KOUMBA_ASR_KEY = "<the same random value>"

curl.exe "$env:KOUMBA_TTS_URL/health" `
  -H "X-Koumba-Key: $env:KOUMBA_ASR_KEY"
```

Expected:

```json
{"ok":true,"model":"Oolel-Voices"}
```

Generate a WAV from a Wolof sentence:

```powershell
curl.exe -X POST "$env:KOUMBA_TTS_URL/speak" `
  -H "X-Koumba-Key: $env:KOUMBA_ASR_KEY" `
  -H "Content-Type: application/json" `
  --data-raw '{"text":"Yaangi Saahm, duggal ci Tataa fanweer ak juróom ñett bi jëm Malika."}' `
  --output reply.wav
```

The image build downloads Oolel-Voices into `/opt/oolel-voices` and loads it
once, so missing dependencies fail the build. The first container downloads
only the voice prompt into the Volume, then loads the baked-in model in
`@modal.enter`; the startup allowance is 900 seconds. Later containers reuse the
cached prompt. The endpoint accepts up to 400
characters, returns `audio/wav`, and times out after 180 seconds. Requests with a missing or wrong key return HTTP 401; empty,
malformed, or overlong text returns HTTP 400. Each new sentence is cached as
`/cache/speech/<sha256>.wav` and later requests for the same final text serve
that file. The service logs cache hits, synthesis time, and character count,
but never logs the sentence. The first request after the 300-second idle window
pays the container start and model load cost.

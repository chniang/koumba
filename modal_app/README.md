# Koumba web, ASR, and TTS on Modal

This folder contains three separate Modal services: the public Gradio web app,
Meta Omnilingual ASR for voice input, and Oolel-Voices for spoken Wolof replies.
The voice services are separate from the app: the app only posts WAV bytes or
text to the deployed HTTPS URLs. The local app never imports Modal or loads
either voice model.

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

## Public web demo

`modal_app/web.py` serves the Gradio UI as the `koumba-web` Modal app. It runs
the same `koumba.app.build_app()` UI as the local app on one CPU container with
2 GiB of RAM. Gradio keeps session state in memory, so the app is capped at one
container while that container can handle up to 100 concurrent inputs. The bus
CSVs are downloaded from `TIJAANI/dakar-bus-network` once and cached in the
`koumba-web-cache` Volume through `KOUMBA_DATA_DIR`.

Create the app secret from the local `.env` without printing any values:

```powershell
.\.venv\Scripts\python.exe -m modal secret create koumba-app --from-dotenv .env --force
```

The secret must provide `BASETEN_API_KEY`, `KOUMBA_BASE_URL`,
`KOUMBA_API_KEY_ENV`, `KOUMBA_MODEL`, `KOUMBA_ASR_URL`, `KOUMBA_ASR_KEY`, and
`KOUMBA_TTS_URL`. Deploy the web app from the repository root:

```powershell
.\.venv\Scripts\python.exe -m modal deploy modal_app/web.py
```

Modal prints the public web URL after the deploy. It normally ends in
`--koumba-web-web.modal.run`; the same URL is in the dashboard under Apps →
`koumba-web` → `web`. Paste that URL into the root `README.md` live-demo link.

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

### Keep containers warm (optional)

All three services read `KOUMBA_KEEP_WARM` at deploy time. The default is `0`,
which lets Modal stop the container after the idle window. Set it to `1` to
pass `min_containers=1` and keep one container warm:

```powershell
$env:KOUMBA_KEEP_WARM="1"
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
.\.venv\Scripts\python.exe -m modal deploy modal_app/tts.py
.\.venv\Scripts\python.exe -m modal deploy modal_app/web.py
```

Return to the cheaper cold-start default with:

```powershell
Remove-Item Env:KOUMBA_KEEP_WARM -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
.\.venv\Scripts\python.exe -m modal deploy modal_app/tts.py
.\.venv\Scripts\python.exe -m modal deploy modal_app/web.py
```

**Cost warning:** `KOUMBA_KEEP_WARM=1` keeps one container running for each
service, even when nobody is using the app. Modal bills for that idle capacity.
In CPU mode ASR and TTS use 8 cores and 16 GiB of RAM each, while the web app
uses 1 core and 2 GiB; if ASR is deployed with `KOUMBA_ASR_DEVICE=gpu`, that GPU
stays warm too. Warm containers make the first reply much faster, but leave the
default at `0` unless you need the demo to answer immediately.

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

For the full local flow (Modal ASR, local place hints, then the route agent):

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
  --data-raw '{"text":"Yaangi Saam, duggal ci Tataa fanweer ak juróom ñett bi jëm Malika."}' `
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
pays the container start and model load cost. Set `KOUMBA_KEEP_WARM=1` at
deploy time when that cold start is too slow; see the warm-container section
above for the cost warning.

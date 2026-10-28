# Koumba ASR on Modal

This folder contains the Modal service for Meta Omnilingual ASR. It is separate
from the Koumba app: the app only posts WAV bytes to the deployed HTTPS URL.
The local app never imports Modal or loads the ASR model.

The service uses Meta's Apache-2.0 Omnilingual ASR and defaults to CPU because
Modal GPU functions require a payment method. CPU mode uses 8 cores and 16 GiB
of RAM; GPU mode uses an L4 with A10G fallback. The model defaults to
`omniASR_CTC_1B`; set `omniASR_CTC_300M` for faster CPU inference. Both choices
are read at deploy time, and the pipeline is loaded once per container in
`@modal.enter`. The `koumba-asr-cache` Volume caches the model weights.

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

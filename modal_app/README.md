# Koumba ASR on Modal

This folder contains the Modal service for Meta Omnilingual ASR. It is separate
from the Koumba app: the app only posts WAV bytes to the deployed HTTPS URL.
The local app never imports Modal or loads the ASR model.

The service uses Meta's Apache-2.0 `omniASR_CTC_1B` model, an L4 GPU with A10G
fallback, a 300-second scale-down window, and the `koumba-asr-cache` Volume for
model weights. The pipeline is loaded once per container in `@modal.enter`.

## 1. Create the shared secret

Run this from the repository root. Use any long random value.

```powershell
.\.venv\Scripts\python.exe -m modal secret create koumba-asr "KOUMBA_ASR_KEY=<random>"
```

Keep that value for `.env`. Do not commit it.

## 2. Deploy

```powershell
.\.venv\Scripts\python.exe -m modal deploy modal_app/asr.py
```

Modal prints the web URL. Put it in `.env` without `/transcribe`:

```text
KOUMBA_ASR_URL=https://your-workspace--koumba-asr-web.modal.run
KOUMBA_ASR_KEY=<the same random value>
```

The first container downloads the model into the Volume. Later containers
reuse the cached weights. The first request after the idle window is slower
because the GPU container must start again.

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
{"wol_Latn":true,"fra_Latn":true}
```

Transcribe a PCM WAV:

```powershell
curl.exe -X POST "$env:KOUMBA_ASR_URL/transcribe?lang=wol_Latn" `
  -H "X-Koumba-Key: $env:KOUMBA_ASR_KEY" `
  -H "Content-Type: audio/wav" `
  --data-binary "@sample.wav"
```

Expected shape:

```json
{"text":"...","lang":"wol_Latn","seconds":2.4,"model":"omniASR_CTC_1B"}
```

A missing or wrong key returns HTTP 401. Requests without a WAV body return
HTTP 400. The temporary WAV is deleted after inference; no audio is retained.

For the full local flow (Modal ASR, Muse Spark cleanup, then the route agent):

```powershell
.\.venv\Scripts\python.exe scripts\try_voice.py path\to\sample.wav
```

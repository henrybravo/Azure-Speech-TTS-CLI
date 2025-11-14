# Azure Speech TTS CLI

<p align="center">
	<a href="https://www.python.org/">
		<img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white">
	</a>
	<a href="https://learn.microsoft.com/azure/ai-services/speech-service/">
		<img alt="Azure Speech" src="https://img.shields.io/badge/Azure%20Speech-API-blueviolet?logo=microsoftazure&logoColor=white">
	</a>
	<a href="https://github.com/astral-sh/uv">
		<img alt="uv" src="https://img.shields.io/badge/Package%20Manager-uv-ff69b4?logo=python&logoColor=white">
	</a>
	<a href="#license">
		<img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
	</a>
	<a href="https://github.com/henrybravo/TTS/issues">
		<img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
	</a>
</p>

<p align="center">
	<em>Chunked, inspectable, SSML-powered text‑to‑speech pipeline in one Python file.</em>
</p>

A single-file Python CLI (`TTS.py`) for converting large text files into speech using **Azure AI Services Speech** by AI Foundry.

It adds:
- Automatic safe chunking (sentence-aware) for long input files
- Retry logic with exponential backoff on transient failures
- Flexible audio format selection via friendly aliases (`--list-formats`)
- Voice discovery & filtering (`--list-voices`, locale/name filtering, JSON output)
- SSML prosody & expressive style controls (rate, pitch, volume, style, style-degree)
- Optional per-chunk SSML export (`--write-ssml`) and chunk text snapshots (`--debug-chunks`)
- Robust WAV merging (multi-chunk RIFF concatenation with header repair)
- Duration verification (`--verify-duration`) to validate merged output
- Graceful handling of missing SDK dependency with helpful remediation

> Goal: Practical, inspectable, dependency-light tool for narration prep, rapid prototyping, or batch generation pipelines.

---
## 1. Quick Start

```powershell
# Clone the project from GitHub
git clone https://github.com/henrybravo/Azure-Speech-TTS-CLI.git
cd .\Azure-Speech-TTS-CLI

# (Recommended) Create & activate a virtual environment (using uv for speed)
uv venv
. .\.venv\Scripts\Activate.ps1

# Install Azure Speech SDK
uv pip install azure-cognitiveservices-speech

# Set required environment variables (replace placeholders)
$env:AZURE_SPEECH_KEY = "<YOUR_KEY>"
$env:AZURE_SPEECH_REGION = "eastus"   # Or your resource region

# Basic synthesis
uv run python .\TTS.py -i input-text.txt -o notes.wav -v en-US-AriaNeural
```

If run with **no arguments**, the tool prints its help and exits.

---
## 2. Installation & Environment

### 2.1 Requirements
- Python 3.9+ (tested with modern CPython versions)
- Azure Cognitive Services Speech resource (key + region OR a custom endpoint)
- Internet connectivity

### 2.2 Using `uv` (fast installer & resolver)
`uv` is a Rust-based drop-in replacement for pip/pip-tools style workflows.

Install (if not already):
```powershell
# Windows PowerShell example (script from official repo)
powershell -ExecutionPolicy Bypass -c "iwr https://astral.sh/uv/install.ps1 | iex"
```

Then inside your project directory:
```powershell
uv venv
. .\.venv\Scripts\Activate.ps1
uv pip install azure-cognitiveservices-speech
```

### 2.3 Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_SPEECH_KEY` | Yes | Speech resource key |
| `AZURE_SPEECH_REGION` | Yes (unless endpoint set) | Region (e.g. `eastus`, `westeurope`) |
| `AZURE_SPEECH_ENDPOINT` | Optional | Custom endpoint URL; if set, region can be omitted |

Set in PowerShell for the current session:
```powershell
$env:AZURE_SPEECH_KEY = "<key>"
$env:AZURE_SPEECH_REGION = "eastus"
```

> Never commit keys; prefer scripts, .env (excluded), or managed identity for production.

---
## 3. Core Usage

```powershell
# If run with no arguments, the tool prints its help and exits
uv run python .\TTS.py                                                           
usage: TTS.py [-h] --input INPUT [--output OUTPUT] [--voice VOICE] [--format FORMAT] [--max-chars MAX_CHARS] [--list-voices] [--locale-filter LOCALE_FILTER] [--contains CONTAINS] [--json]
              [--list-formats] [--rate RATE] [--pitch PITCH] [--volume VOLUME] [--style STYLE] [--style-degree STYLE_DEGREE] [--debug-chunks] [--write-ssml] [--verify-duration]

Azure Speech Text-to-Speech from a text file.

options:
  -h, --help            show this help message and exit
  --input, -i INPUT     Input text file (required).
  --output, -o OUTPUT   Output WAV filename (default: <input-stem>.wav)
  --voice, -v VOICE     Voice name (default: en-US-AriaNeural)
  --format, -f FORMAT   Speech synthesis output format (friendly alias or enum). Use --list-formats to inspect.
  --max-chars MAX_CHARS
                        Maximum characters per chunk (default: 5000).
  --list-voices         List available voices for the configured region and exit.
  --locale-filter LOCALE_FILTER
                        Optional locale filter (e.g. en-GB, en-US).
  --contains CONTAINS   Substring filter applied to voice name (case-insensitive).
  --json                Output voice listing as JSON when used with --list-voices.
  --list-formats        List supported friendly audio format aliases and exit.
  --rate RATE           Prosody speaking rate (e.g. +20%, -10%, fast, x-slow).
  --pitch PITCH         Prosody pitch (e.g. +2st, -3st, +5%).
  --volume VOLUME       Prosody volume adjustment (e.g. +0%, +10%, -5%).
  --style STYLE         Expressive style name (voice must support it, e.g. newscast, chat, cheerful).
  --style-degree STYLE_DEGREE
                        Style degree (0.0–2.0 typical).
  --debug-chunks        Print chunk summaries and write chunk_<n>.txt files before synthesis.
  --write-ssml          Write SSML per chunk to ssml_chunk_<n>.xml (neutral SSML if not using prosody/style).
  --verify-duration     After merge, parse WAV header and report duration & per-segment declared sizes.

# Basic (defaults to voice Aria)
uv run python .\TTS.py -i input-text.txt -o narration.wav

# Explicit input/output and voice
uv run python .\TTS.py -i input-text.txt -o narration.wav -v en-US-GuyNeural

# Choose a format (friendly alias)
uv run python .\TTS.py -i input-text.txt -f riff-24khz-16bit-mono-pcm

# Faster MP3 output (smaller file)
uv run python .\TTS.py -i input-text.txt -f mp3-24k-96 -o notes.mp3
```

If the chosen output format isn't RIFF/WAV, the tool writes raw bytes exactly as returned by the SDK (no merging logic needed if single chunk).

---
## 4. Listing Voices & Formats

### 4.1 Voices
```powershell
# All voices (may be large)
uv run python .\TTS.py --list-voices

# Filter by locale
uv run python .\TTS.py --list-voices --locale-filter en-GB

# Fuzzy name containment
uv run python .\TTS.py --list-voices --contains aria

# JSON output (automation)
uv run python .\TTS.py --list-voices --locale-filter en-US --json > voices.json
```

### 4.2 Formats
```powershell
uv run python .\TTS.py --list-formats
```
Shows alias -> enum mapping (only those present in the SDK version you installed).

---
## 5. Prosody & Expressive Styles (SSML)

The tool automatically switches to SSML mode when any of: `--rate`, `--pitch`, `--volume`, `--style` is provided.

| Flag | Examples | Notes |
|------|----------|-------|
| `--rate` | `+20%`, `-10%`, `fast`, `x-slow` | Percentage or keyword values supported by Azure SSML |
| `--pitch` | `+2st`, `-3st`, `+5%` | Semitones (`st`) or percent |
| `--volume` | `+0%`, `+10%`, `-5%` | Typical useful range -50% to +50% |
| `--style` | `newscast`, `cheerful`, `chat` | Depends on the chosen voice |
| `--style-degree` | `0.7`, `1.0` | Usually 0.0–2.0 (voice-specific) |

Examples:
```powershell
# Faster, slightly higher pitch
uv run python .\TTS.py -i input-text.txt -o fast.wav -v en-US-GuyNeural --rate +25% --pitch +2st

# Slower, deeper
uv run python .\TTS.py -i input-text.txt -o slow.wav -v en-US-AriaNeural --rate -15% --pitch -2st

# Expressive style
uv run python .\TTS.py -i input-text.txt -o cast.wav -v en-US-DavisNeural --style newscast --style-degree 1.0
```

### 5.1 Inspect Generated SSML
```powershell
uv run python .\TTS.py -i input-text.txt --write-ssml --rate +10% --pitch +2st
```
Produces `ssml_chunk_<n>.xml` per chunk (neutral SSML if no prosody/style flags are active).

---
## 6. Chunking Strategy
- Splits on sentence boundaries using a regex that detects punctuation followed by whitespace.
- Ensures each chunk ≤ `--max-chars` (default 5000, below Azure plain-text limit margin).
- Extremely long sentences are hard-split.

Diagnostics:
```powershell
uv run python .\TTS.py -i input-text.txt --debug-chunks
```
Creates `chunk_<n>.txt` and prints a 160-char preview for each.

---
## 7. Robust WAV Merging
For multi-chunk RIFF outputs the tool:
1. Extracts the `data` chunk start & declared size for each segment.
2. Concatenates actual payload bytes.
3. Recomputes RIFF size & `data` size fields.
4. Logs segment declared vs actual sizes to catch inconsistencies.

If a non-standard layout prevents patching, a warning is emitted and a naive concatenation is used as fallback.

---
## 8. Duration Verification
After synthesis, you can verify the merged length:
```powershell
uv run python .\TTS.py -i input-text.txt --verify-duration
```
Outputs `[VERIFY] channels=... sample_rate=... data_size=... duration=XX.XXs`.

Useful when the audible playback time appears shorter than expected: confirms whether the header and data sizes agree.

---
## 9. Error Handling & Retries
- Each chunk synthesis retries up to 3 times with exponential backoff (2s, 4s, 8s delays) on exceptions or cancellations.
- On repeated failure of a chunk the process aborts with a fatal error code.
- Distinguishes cancellation vs unexpected result reasons for clarity.

Exit codes:
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input file error |
| 2 | Config / environment error (missing key/region) |
| 3 | Synthesis failure after retries |
| 10 | Missing SDK dependency |
| 11/12 | Voice listing retrieval issues |

---
## 10. Troubleshooting
| Symptom | Possible Cause | Action |
|---------|----------------|--------|
| Immediate exit w/ help | No args | Supply arguments or accept defaults |
| `[FATAL] azure.cognitiveservices.speech SDK not found` | Package not installed in active venv | Activate venv; install with `uv pip install azure-cognitiveservices-speech` |
| Short audio vs text length | Incorrect WAV header or playback ended early | Use `--verify-duration` & inspect segment logs |
| Some chunks silent | Canceled synth or style unsupported | Remove style or inspect cancellation details |
| Voice not found | Exact name mismatch | Use `--list-voices --contains <part>` to locate canonical name |
| Unknown format warning | Alias not mapped | Run `--list-formats` and pick a listed alias |

---
## 11. Design Notes
- **Single file**: Easier to audit & drop into existing repos.
- **No extra dependencies**: Only the Azure Speech SDK required.
- **Transparency**: Debug flags produce traceable artifacts (text chunks + SSML).
- **Resilience**: Conservative chunk size and retries reduce failure surface.
- **Extensibility**: Future additions (presets, manifest JSON, parallel synthesis) can be layered without breaking interface.

---
## 12. Possible Future Enhancements
- JSON manifest summarizing voice, format, chunk boundaries, durations
- Parallel chunk synthesis (respecting Azure throttling limits)
- Automatic output extension inference from format
- Built-in `.env` loader (optional)
- Dry-run mode (generate & validate SSML only)

---
## 13. Security & Compliance
- Secrets stay in environment variables; never hardcode.
- Consider rotating keys and/or using managed identity in hosted settings.
- Logs avoid printing the key or endpoint.

---
## 14. Reference Commands (Copy/Paste)
```powershell
# List voices (US English JSON)
uv run python .\TTS.py --list-voices --locale-filter en-US --json

# Generate energetic narration with diagnostics
uv run python .\TTS.py -i input-text.txt -o energetic.wav -v en-US-JennyNeural --style cheerful --style-degree 0.7 --rate +10% --debug-chunks --write-SSML --verify-duration

# MP3 output faster download
uv run python .\TTS.py -i input-text.txt -o notes.mp3 -f mp3-24k-96
```

---
## 15. Sources & Further Reading
- Azure Speech SDK (Python) Docs: https://learn.microsoft.com/azure/ai-services/speech-service/ 
- SSML Reference (Prosody & Styles): https://learn.microsoft.com/azure/ai-services/speech-service/speech-synthesis-markup 
- Azure Speech Voices List: https://learn.microsoft.com/azure/ai-services/speech-service/language-support#text-to-speech 
- Azure Speech GitHub Samples: https://github.com/Azure-Samples/cognitive-services-speech-sdk 
- uv Project (Python package manager): https://github.com/astral-sh/uv 

---
## License

Released under the [MIT License](LICENSE). You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies subject to the terms in the license. Attribution in derivative works appreciated but not required.

Happy narrating! 🎤

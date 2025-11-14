
"""Text-to-Speech utility for Azure Cognitive Services.

Features:
  * Reads a UTF-8 text file (default: notes.txt) and chunks text safely.
  * Supports multiple audio output formats with friendly aliases (see --list-formats).
  * Can list available neural voices (see --list-voices) with filtering.
  * Retries transient failures with exponential backoff.
  * Optional SSML prosody/style controls (rate, pitch, volume, style, style degree).
  * Merges WAV chunks when using RIFF outputs; raw formats returned as-is.

Authentication (env vars - never hardcode):
  AZURE_SPEECH_KEY      Speech resource key
  AZURE_SPEECH_REGION   Region (e.g. eastus, westeurope) unless using endpoint
  AZURE_SPEECH_ENDPOINT Optional custom endpoint URL

Basic usage:
  python TTS.py -i notes.txt -o notes.wav -v "en-US-AriaNeural"
  python TTS.py --list-voices --locale-filter en-GB --contains ollie
  python TTS.py --format riff-24khz-16bit-mono-pcm -v "Microsoft Server Speech Text to Speech Voice (en-GB, OllieMultilingualNeural)"

Prosody / style examples:
  python TTS.py -i notes.txt -o fast.wav -v "en-US-GuyNeural" --rate +25%
  python TTS.py -i notes.txt -o slow.wav -v "en-US-AriaNeural" --rate -15% --pitch +2%
  python TTS.py -i notes.txt -o newscast.wav -v "en-US-DavisNeural" --style newscast --style-degree 1.0
  python TTS.py -i notes.txt -o energetic.wav -v "en-US-JennyNeural" --style cheerful --style-degree 0.7 --rate +10%

Notes:
  * rate accepts percentage (e.g. +20%, -10%) or keywords (x-slow, slow, medium, fast, x-fast).
  * pitch can use semitone adjustments (+2st, -3st) or percentages (+5%).
  * volume (e.g. +0%, +10%, -5%) typical safe range: -50% to +50%.
  * style/style-degree only apply to voices supporting styles; degree in [0.0, 2.0] typically.
  * For more samples: https://github.com/Azure-Samples/cognitive-services-speech-sdk
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

try:
  import azure.cognitiveservices.speech as speechsdk  # type: ignore
except ModuleNotFoundError:
  print("[FATAL] azure.cognitiveservices.speech SDK not found in current interpreter.")
  print("[HINT] Ensure your uv/venv is activated and install the package, e.g.:")
  print("       uv pip install azure-cognitiveservices-speech")
  print("       (Or: pip install azure-cognitiveservices-speech)")
  print("[HINT] If already installed, confirm VS Code Python interpreter matches the venv.")
  sys.exit(10)


DEFAULT_MAX_CHARS = 5000  # Conservative chunk size (< 5000 char plain text limit)
FRIENDLY_FORMAT_MAP = {
  # Friendly alias -> Actual SpeechSynthesisOutputFormat enum member name
  # Reference: inspect speechsdk.SpeechSynthesisOutputFormat.__members__ at runtime for full list.
  "riff-16khz-16bit-mono-pcm": "Riff16Khz16BitMonoPcm",
  "riff-24khz-16bit-mono-pcm": "Riff24Khz16BitMonoPcm",
  "riff-48khz-16bit-mono-pcm": "Riff48Khz16BitMonoPcm",
  "raw-16khz-16bit-mono-pcm": "Raw16Khz16BitMonoPcm",
  "raw-24khz-16bit-mono-pcm": "Raw24Khz16BitMonoPcm",
  "raw-48khz-16bit-mono-pcm": "Raw48Khz16BitMonoPcm",
  # Common MP3 bitrates (names can vary by SDK version; these are typical)
  "mp3-24k-96": "Audio24Khz96KBitRateMonoMp3",
  "mp3-24k-48": "Audio24Khz48KBitRateMonoMp3",
  "mp3-48k-96": "Audio48Khz96KBitRateMonoMp3",
  # Ogg/Opus
  "ogg-24k": "Ogg24Khz16BitMonoOpus",
  "ogg-48k": "Ogg48Khz16BitMonoOpus",
}
HEADER_MAX_SEARCH = 512   # Safety bound when scanning WAV header size

def load_text(path: Path) -> str:
  if not path.exists():
    raise FileNotFoundError(f"Input file not found: {path}")
  content = path.read_text(encoding="utf-8").strip()
  if not content:
    raise ValueError("Input file is empty.")
  return content

_sentence_split_regex = re.compile(r"(?<=[.!?])\s+")

def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
  """Split text into chunks not exceeding max_chars, trying sentence boundaries.

  Falls back to hard split if a single sentence exceeds max_chars.
  """
  sentences = _sentence_split_regex.split(text)
  chunks: List[str] = []
  current: List[str] = []
  current_len = 0
  for sent in sentences:
    sent = sent.strip()
    if not sent:
      continue
    if len(sent) > max_chars:
      # Hard split this long sentence
      for i in range(0, len(sent), max_chars):
        segment = sent[i : i + max_chars]
        if current:
          chunks.append(" ".join(current))
          current = []
          current_len = 0
        chunks.append(segment)
      continue
    prospective_len = current_len + (1 if current else 0) + len(sent)
    if prospective_len > max_chars and current:
      chunks.append(" ".join(current))
      current = [sent]
      current_len = len(sent)
    else:
      current.append(sent)
      current_len = prospective_len
  if current:
    chunks.append(" ".join(current))
  return chunks


def resolve_output_format(output_format: str) -> str | None:
  """Resolve a user string to a SpeechSynthesisOutputFormat enum member name.

  Strategy:
    1. Match friendly alias in FRIENDLY_FORMAT_MAP
    2. Case-insensitive direct member name match
    3. Fuzzy normalization: strip non-alnum, ignore case
  Returns the exact enum member name or None.
  """
  if not output_format:
    return None
  candidate = output_format.strip().lower()
  # Alias mapping
  if candidate in FRIENDLY_FORMAT_MAP:
    return FRIENDLY_FORMAT_MAP[candidate]
  # Build lowercase map of enum members for flexible match
  members = speechsdk.SpeechSynthesisOutputFormat.__members__
  lowered = {name.lower(): name for name in members.keys()}
  if candidate in lowered:
    return lowered[candidate]
  # Normalized attempt
  normalized = re.sub(r"[^a-z0-9]+", "", candidate)
  for key_lower, original in lowered.items():
    if re.sub(r"[^a-z0-9]+", "", key_lower) == normalized:
      return original
  return None

def create_speech_config(voice: str, output_format: str) -> speechsdk.SpeechConfig:
  key = os.getenv("AZURE_SPEECH_KEY")
  region = os.getenv("AZURE_SPEECH_REGION")
  endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
  if not key:
    raise EnvironmentError("AZURE_SPEECH_KEY env var is required.")
  if endpoint:
    speech_config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
  else:
    if not region:
      raise EnvironmentError("AZURE_SPEECH_REGION env var is required when AZURE_SPEECH_ENDPOINT is not set.")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
  speech_config.speech_synthesis_voice_name = voice
  # Set audio format if provided (validate against SDK enumerations if needed)
  resolved = resolve_output_format(output_format)
  if resolved:
    try:
      speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat[resolved])
    except KeyError:
      print(f"[WARN] Resolved format name '{resolved}' not present in SDK members; using default.")
  else:
    print(f"[WARN] Unknown/unsupported format '{output_format}', using SDK default. Use --list-formats to view supported aliases.")
  return speech_config

def escape_ssml(text: str) -> str:
  return (
    text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
  )

def build_ssml(chunk: str, voice: str, locale: Optional[str], rate: Optional[str], pitch: Optional[str], volume: Optional[str], style: Optional[str], style_degree: Optional[float]) -> str:
  # Locale fallback: attempt extraction from voice name pattern (... (locale, VoiceName)) else env region locale part.
  if not locale:
    match = re.search(r"\(([^,]+),", voice)
    locale = match.group(1) if match else "en-US"
  prosody_attrs = []
  if rate:
    prosody_attrs.append(f'rate="{rate}"')
  if pitch:
    prosody_attrs.append(f'pitch="{pitch}"')
  if volume:
    prosody_attrs.append(f'volume="{volume}"')
  prosody_attr_str = " ".join(prosody_attrs)
  style_open = style_close = ""
  if style:
    degree_attr = f' styledegree="{style_degree}"' if style_degree is not None else ""
    style_open = f'<mstts:express-as style="{style}"{degree_attr}>'
    style_close = '</mstts:express-as>'
  ssml = f"""<speak version='1.0' xml:lang='{locale}' xmlns:mstts='http://www.w3.org/2001/mstts'>
  <voice name="{voice}">
    {style_open}<prosody {prosody_attr_str}>{escape_ssml(chunk)}</prosody>{style_close}
  </voice>
</speak>"""
  return ssml

def synthesize_chunks(chunks: Iterable[str], speech_config: speechsdk.SpeechConfig, voice: str, rate: Optional[str], pitch: Optional[str], volume: Optional[str], style: Optional[str], style_degree: Optional[float], write_ssml: bool) -> List[bytes]:
  use_ssml = any([rate, pitch, volume, style])
  synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
  audio_segments: List[bytes] = []
  for idx, chunk in enumerate(chunks, start=1):
    attempt = 0
    while True:
      attempt += 1
      try:
        print(f"[INFO] Synthesizing chunk {idx} (len={len(chunk)} chars) attempt {attempt}... mode={'SSML' if use_ssml else 'text'}")
        ssml_used: Optional[str] = None
        if use_ssml:
          ssml_used = build_ssml(chunk, voice, None, rate, pitch, volume, style, style_degree)
          result = synthesizer.speak_ssml_async(ssml_used).get()
        else:
          result = synthesizer.speak_text_async(chunk).get()
          if write_ssml:
            # Provide neutral SSML for inspection when plain text mode
            ssml_used = build_ssml(chunk, voice, None, None, None, None, None, None)
        if write_ssml and ssml_used:
          out_ssml = Path(f"ssml_chunk_{idx}.xml")
          out_ssml.write_text(ssml_used, encoding="utf-8")
          print(f"[DEBUG] Wrote SSML: {out_ssml}")
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
          audio_segments.append(result.audio_data)
          print(f"[OK] Chunk {idx} synthesized ({len(result.audio_data)} bytes).")
          break
        elif result.reason == speechsdk.ResultReason.Canceled:
          details = result.cancellation_details
          print(f"[ERROR] Chunk {idx} canceled: {details.reason} - {details.error_details}")
          if attempt >= 3:
            raise RuntimeError(f"Synthesis canceled for chunk {idx}: {details.reason}")
        else:
          print(f"[ERROR] Unexpected result for chunk {idx}: {result.reason}")
          if attempt >= 3:
            raise RuntimeError(f"Unexpected synthesis result for chunk {idx}")
      except Exception as ex:  # noqa: BLE001
        print(f"[WARN] Exception on chunk {idx}: {ex}")
        if attempt >= 3:
          raise
      sleep_for = 2 ** (attempt - 1)
      print(f"[INFO] Retrying chunk {idx} after {sleep_for}s...")
      time.sleep(sleep_for)
  return audio_segments

def merge_wav_segments(segments: List[bytes]) -> bytes:
  if not segments:
    raise ValueError("No audio segments to merge.")
  if len(segments) == 1:
    return segments[0]
  # Robust RIFF concatenation: parse "data" chunk start & sizes for each segment.
  def parse_segment(data: bytes) -> tuple[int, int, int]:
    """Return (payload_start, data_chunk_size, header_total_length_up_to_payload).

    payload_start: index where audio sample bytes begin.
    data_chunk_size: size declared in the 'data' chunk for this segment.
    header_len: number of bytes before payload (including 'data' + size field).
    """
    # Seek 'data' chunk; RIFF may contain extra chunks (fact, LIST, JUNK, etc.)
    search_limit = min(len(data), 10_000)  # safety
    idx = data.find(b'data', 0, search_limit)
    if idx < 0:
      # Fallback heuristic: assume standard 44-byte header
      return (44, len(data) - 44, 44)
    if idx + 8 > len(data):
      raise ValueError("Corrupt WAV: incomplete data chunk header.")
    declared_size = int.from_bytes(data[idx + 4: idx + 8], 'little')
    payload_start = idx + 8
    header_len = payload_start  # everything before payload
    return (payload_start, declared_size, header_len)

  first = segments[0]
  first_payload_start, first_declared_size, first_header_len = parse_segment(first)
  payloads: List[bytes] = [first[first_payload_start:]]
  declared_sizes = [first_declared_size]
  header_bytes = first[:first_payload_start]  # preserve original header (incl 'data'+size field)
  for i, seg in enumerate(segments[1:], start=2):
    payload_start, declared_size, _ = parse_segment(seg)
    declared_sizes.append(declared_size)
    payloads.append(seg[payload_start:])
    print(f"[DEBUG] Segment {i} declared data size={declared_size} actual bytes={len(seg) - payload_start}")
  combined_payload = b''.join(payloads)

  # Patch RIFF chunk sizes in header.
  # RIFF size field bytes[4:8] = file_size_minus_8
  # data subchunk size is at last 4 bytes of header (just before first payload) if classic layout.
  # We attempt to locate 'data' again in header to patch correctly.
  data_marker_index = header_bytes.find(b'data')
  if data_marker_index < 0:
    print("[WARN] Unable to patch header sizes (no 'data' marker in header); returning naive concatenation.")
    return header_bytes + combined_payload

  # Build merged body
  merged = header_bytes + combined_payload

  # Compute new sizes
  total_data_size = sum(len(p) for p in payloads)
  riff_size = total_data_size + (first_payload_start - 8)  # file size - 8

  # Patch RIFF size
  merged = (
    merged[:4]
    + riff_size.to_bytes(4, 'little')
    + merged[8:]
  )
  # Patch data chunk size
  size_field_index = data_marker_index + 4
  merged = (
    merged[:size_field_index]
    + total_data_size.to_bytes(4, 'little')
    + merged[size_field_index + 4:]
  )
  print(f"[INFO] Merged WAV: segments={len(segments)} total_data_bytes={total_data_size}")
  return merged

def write_bytes(path: Path, data: bytes) -> None:
  path.write_bytes(data)
  print(f"[INFO] Wrote audio file: {path} ({len(data)} bytes)")

def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Azure Speech Text-to-Speech from a text file.")
  parser.add_argument("--input", "-i", required=True, help="Input text file (required).")
  parser.add_argument("--output", "-o", default=None, help="Output WAV filename (default: <input-stem>.wav)")
  parser.add_argument("--voice", "-v", default="en-US-AriaNeural", help="Voice name (default: en-US-AriaNeural)")
  parser.add_argument(
    "--format",
    "-f",
    default="riff-16khz-16bit-mono-pcm",
    help="Speech synthesis output format (friendly alias or enum). Use --list-formats to inspect.",
  )
  parser.add_argument(
    "--max-chars",
    type=int,
    default=DEFAULT_MAX_CHARS,
    help=f"Maximum characters per chunk (default: {DEFAULT_MAX_CHARS}).",
  )
  # Voice listing / discovery options
  parser.add_argument(
    "--list-voices",
    action="store_true",
    help="List available voices for the configured region and exit.")
  parser.add_argument(
    "--locale-filter",
    default=None,
    help="Optional locale filter (e.g. en-GB, en-US).")
  parser.add_argument(
    "--contains",
    default=None,
    help="Substring filter applied to voice name (case-insensitive).")
  parser.add_argument(
    "--json",
    action="store_true",
    help="Output voice listing as JSON when used with --list-voices.")
  parser.add_argument(
    "--list-formats",
    action="store_true",
    help="List supported friendly audio format aliases and exit.")
  # SSML / prosody controls
  # NOTE: Percent signs must be escaped as '%%' for argparse help formatting
  parser.add_argument("--rate", default=None, help="Prosody speaking rate (e.g. +20%%, -10%%, fast, x-slow).")
  parser.add_argument("--pitch", default=None, help="Prosody pitch (e.g. +2st, -3st, +5%%).")
  parser.add_argument("--volume", default=None, help="Prosody volume adjustment (e.g. +0%%, +10%%, -5%%).")
  parser.add_argument("--style", default=None, help="Expressive style name (voice must support it, e.g. newscast, chat, cheerful).")
  parser.add_argument("--style-degree", type=float, default=None, help="Style degree (0.0–2.0 typical).")
  parser.add_argument("--debug-chunks", action="store_true", help="Print chunk summaries and write chunk_<n>.txt files before synthesis.")
  parser.add_argument("--write-ssml", action="store_true", help="Write SSML per chunk to ssml_chunk_<n>.xml (neutral SSML if not using prosody/style).")
  parser.add_argument("--verify-duration", action="store_true", help="After merge, parse WAV header and report duration & per-segment declared sizes.")
  if not argv:
    # No arguments supplied: show help and exit early.
    parser.print_help()
    sys.exit(0)
  return parser.parse_args(argv)

def list_formats(as_json: bool) -> int:
  """List friendly format aliases and their resolved enum names."""
  items = [
    {"alias": alias, "enum": enum_name} for alias, enum_name in FRIENDLY_FORMAT_MAP.items()
    if enum_name in speechsdk.SpeechSynthesisOutputFormat.__members__
  ]
  if as_json:
    print(json.dumps(items, indent=2))
  else:
    print("Available format aliases (alias -> enum):")
    for item in items:
      print(f"  {item['alias']} -> {item['enum']}")
    print(f"[INFO] Total aliases: {len(items)}")
  return 0

def list_voices(speech_config: speechsdk.SpeechConfig, locale_filter: str | None, contains: str | None, as_json: bool) -> int:
  """Retrieve and display available neural voices.

  Filters:
    locale_filter: exact match on locale (e.g. en-GB)
    contains: substring case-insensitive search in voice name
  """
  print("[INFO] Fetching voices...")
  synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
  try:
    result = synthesizer.get_voices_async().get()
  except Exception as e:  # noqa: BLE001
    print(f"[FATAL] Unable to retrieve voices: {e}")
    return 11
  if result.reason != speechsdk.ResultReason.VoicesListRetrieved:
    print(f"[FATAL] Unexpected result reason: {result.reason}")
    return 12
  voices = result.voices or []
  # Apply filters
  if locale_filter:
    voices = [v for v in voices if (v.locale or '').lower() == locale_filter.lower()]
  if contains:
    needle = contains.lower().strip()
    def _gender_name(gender_obj):
      if gender_obj is None:
        return ''
      if hasattr(gender_obj, 'name') and isinstance(getattr(gender_obj, 'name'), str):
        try:
          return gender_obj.name.lower()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
          return str(gender_obj).lower()
      return str(gender_obj).lower()
    def _matches(v):
      gender_val = _gender_name(getattr(v, 'gender', None))
      # Exact gender handling to avoid 'male' matching 'female'.
      if needle in ('male', 'female'):
        return gender_val == needle
      # Build token list from name, short_name, locale, styles, and gender.
      raw_tokens = [
        v.name or '',
        getattr(v, 'short_name', '') or '',
        v.locale or '',
        gender_val,
      ] + (getattr(v, 'style_list', []) or [])
      tokens: list[str] = []
      for t in raw_tokens:
        if not t:
          continue
        # Split on non-alphanumeric to get atomic tokens
        tokens.extend([x for x in re.split(r'[^a-zA-Z0-9]+', t) if x])
      tokens_lower = [t.lower() for t in tokens]
      # Prefer exact token match; fallback to substring in original fields if no exact match.
      if needle in tokens_lower:
        return True
      # Substring fallback across original joined lower text (excluding gender to avoid male/female confusion)
      haystack_parts = [p.lower() for p in raw_tokens if p]
      # Remove gender part for substring search when searching generic term containing 'male'/'female'
      if needle in ('male', 'female'):
        haystack_parts = [p for p in haystack_parts if p != gender_val]
      return any(needle in p for p in haystack_parts if p)
    voices = [v for v in voices if _matches(v)]
  if not voices:
    print("[INFO] No voices matched filters.")
    return 0
  if as_json:
    def _normalize(val):
      """Return a JSON-serializable representation of SDK values.

      Converts Enums (with .name) to their name, leaves primitives untouched,
      falls back to str(val) for unknown object types.
      """
      if val is None:
        return None
      # Enum-like: has .name and .value attributes typically
      if hasattr(val, "name") and isinstance(getattr(val, "name"), str):
        try:
          return val.name  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
          pass
      # Primitives pass through
      if isinstance(val, (str, int, float, bool, list, dict)):
        return val
      return str(val)

    payload = []
    for v in voices:
      gender_raw = getattr(v, 'gender', None)
      styles_raw = getattr(v, 'style_list', None)
      payload.append({
        "name": v.name,
        "locale": v.locale,
        "gender": _normalize(gender_raw),
        "shortName": getattr(v, 'short_name', None),
        "styles": [_normalize(s) for s in styles_raw] if styles_raw else None,
      })
    print(json.dumps(payload, indent=2))
  else:
    print("Available voices (name | locale | gender | styles):")
    for v in voices:
      gender_val = getattr(v, 'gender', '?')
      if gender_val != '?' and hasattr(gender_val, 'name'):
        try:
          gender_val = gender_val.name  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
          gender_val = str(gender_val)
      styles_list = getattr(v, 'style_list', []) or []
      styles = ",".join(styles_list)
      print(f"  {v.name} | {v.locale} | {gender_val} | {styles}")
    print(f"[INFO] Total voices listed: {len(voices)}")
  return 0

def main(argv: List[str]) -> int:
  args = parse_args(argv)
  # Early listing operations shouldn't require reading input file
  try:
    speech_config = create_speech_config(args.voice, args.format)
  except Exception as e:  # noqa: BLE001
    print(f"[FATAL] Config error: {e}")
    return 2

  if args.list_formats:
    return list_formats(args.json)
  if args.list_voices:
    return list_voices(speech_config, args.locale_filter, args.contains, args.json)

  in_path = Path(args.input)
  if not in_path.exists():
    print(f"[FATAL] Input file not found: {in_path}. Supply a valid path with --input.")
    return 1
  out_path = Path(args.output) if args.output else Path(in_path.stem + ".wav")

  try:
    text = load_text(in_path)
  except Exception as e:  # noqa: BLE001
    print(f"[FATAL] Unable to read input: {e}")
    return 1

  chunks = chunk_text(text, args.max_chars)
  print(f"[INFO] Total text length: {len(text)} chars -> {len(chunks)} chunk(s).")
  if getattr(args, 'debug_chunks', False):
    print("[DEBUG] Listing chunk summaries (first 160 chars escaped)...")
    for i, c in enumerate(chunks, 1):
      preview = c[:160].replace("\n", "\\n")
      print(f"[CHUNK {i}] len={len(c)} preview=" + preview)
      Path(f"chunk_{i}.txt").write_text(c, encoding="utf-8")
    print(f"[DEBUG] Wrote {len(chunks)} chunk_<n>.txt files.")

  try:
    segments = synthesize_chunks(
      chunks,
      speech_config,
      args.voice,
      args.rate,
      args.pitch,
      args.volume,
      args.style,
      args.style_degree,
      getattr(args, 'write_ssml', False),
    )
    # Log per-segment raw byte lengths for diagnostics
    for idx, seg in enumerate(segments, start=1):
      print(f"[DEBUG] Raw segment {idx} length={len(seg)} bytes")
    merged = merge_wav_segments(segments)
    write_bytes(out_path, merged)
    if getattr(args, 'verify_duration', False):
      # Basic PCM header parsing for duration (works for standard PCM)
      if merged[:4] == b'RIFF' and merged[8:12] == b'WAVE':
        channels = int.from_bytes(merged[22:24], 'little')
        sample_rate = int.from_bytes(merged[24:28], 'little')
        bits_per_sample = int.from_bytes(merged[34:36], 'little')
        data_idx = merged.find(b'data')
        if data_idx >= 0:
          data_size = int.from_bytes(merged[data_idx+4:data_idx+8], 'little')
          bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
          if bytes_per_second > 0:
            duration_sec = data_size / bytes_per_second
            print(f"[VERIFY] channels={channels} sample_rate={sample_rate} bits={bits_per_sample} data_size={data_size} -> duration={duration_sec:.2f}s")
          else:
            print("[VERIFY] Unable to compute duration (bytes_per_second=0).")
        else:
          print("[VERIFY] 'data' chunk not found in merged WAV header.")
      else:
        print("[VERIFY] Not a RIFF/WAVE header; duration check skipped.")
  except Exception as e:  # noqa: BLE001
    print(f"[FATAL] Synthesis failed: {e}")
    return 3

  print("[SUCCESS] Synthesis complete.")
  return 0

if __name__ == "__main__":  # pragma: no cover
  sys.exit(main(sys.argv[1:]))

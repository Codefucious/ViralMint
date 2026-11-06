# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025-2026 ViralMint Contributors
"""Speech-to-text through OpenRouter's /audio/transcriptions endpoint.

The opt-in alternative to local faster-whisper: `whisper_service.transcribe()`
routes here when the user has picked a model in Settings, and everything
downstream keeps receiving the same {text, language, segments[words]} shape.

Three things about the endpoint decide how this file is built:

  * Uploads are capped at 25 MB and upstream providers give up after 60 s, so
    audio is re-encoded to small mono MP3 and sent in chunks.
  * Captions need word timestamps, but a model's listing doesn't say whether
    it can produce them: some reject the request with a 400, and others could
    answer with less than was asked for. All of these raise
    ExternalTranscriptionError, and the caller falls back to local Whisper. The
    first chunk runs alone so that a model which can't do this costs one small
    request, not a whole batch.
  * Providers disagree on the response. Words may arrive without segments, all
    in a single segment, or with the language missing or given as a name
    ("english") rather than a code. `_normalise_chunk` is where those
    differences stop.
"""
import asyncio
import logging
import math
import tempfile
import time
from pathlib import Path

import httpx

from backend.core.ai_provider import _OPENROUTER_HEADERS
from backend.core.concurrency import _ffmpeg_semaphore
from backend.core.exceptions import ExternalTranscriptionError
from backend.services.video_utils import run_ffmpeg

logger = logging.getLogger(__name__)

_API_BASE = "https://openrouter.ai/api/v1"
_TRANSCRIBE_URL = f"{_API_BASE}/audio/transcriptions"
_MODELS_URL = f"{_API_BASE}/models?output_modalities=transcription"

# 5 minutes of 64 kbps mono MP3 is ~2.4 MB: far below the 25 MB cap, and short
# enough for a provider to finish inside its 60-second limit.
CHUNK_SECONDS = 300.0
_MAX_PARALLEL = 3
_MAX_ATTEMPTS = 3
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_REQUEST_TIMEOUT = 120.0
_MODELS_CACHE_SECONDS = 3600.0

# Only used to rebuild segments for providers that return nothing but words.
_SEGMENT_GAP_SECONDS = 1.0
_SEGMENT_MAX_WORDS = 30
_SENTENCE_END = (".", "?", "!", "。", "？", "！")

# Whisper-family providers report a language name; everything else in the app
# stores the ISO code that faster-whisper gives.
_LANGUAGE_NAMES = {
    "arabic": "ar", "chinese": "zh", "dutch": "nl", "english": "en",
    "french": "fr", "german": "de", "hindi": "hi", "indonesian": "id",
    "italian": "it", "japanese": "ja", "korean": "ko", "polish": "pl",
    "portuguese": "pt", "russian": "ru", "spanish": "es", "thai": "th",
    "turkish": "tr", "vietnamese": "vi",
}

_models_cache: dict = {"at": 0.0, "models": None}


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT, connect=15.0))


# ── Model list ───────────────────────────────────────────────────────────────

async def list_transcription_models() -> tuple[list[dict], bool]:
    """OpenRouter's transcription models as ([{id, name}], is_stale).

    The list is public, so no key is needed. It is cached for an hour, and a
    failed refresh serves the previous list rather than emptying the dropdown.
    Raises ExternalTranscriptionError only when there is nothing to serve.
    """
    cached = _models_cache["models"]
    if cached is not None and time.monotonic() - _models_cache["at"] < _MODELS_CACHE_SECONDS:
        return cached, False
    try:
        async with _http_client() as client:
            resp = await client.get(_MODELS_URL, headers=_OPENROUTER_HEADERS)
        resp.raise_for_status()
        rows = resp.json().get("data") or []
    except (httpx.HTTPError, ValueError, AttributeError) as e:
        logger.warning("Could not fetch OpenRouter transcription models: %s", e)
        if cached is not None:
            return cached, True
        raise ExternalTranscriptionError(
            "Couldn't load OpenRouter's transcription models. "
            "Check your internet connection and try again."
        ) from e

    models = sorted(
        (
            {"id": m["id"], "name": m.get("name") or m["id"]}
            for m in rows
            if m.get("id")
            and "transcription" in ((m.get("architecture") or {}).get("output_modalities") or [])
        ),
        key=lambda m: m["name"].lower(),
    )
    _models_cache.update(at=time.monotonic(), models=models)
    return models, False


# ── Audio preparation ────────────────────────────────────────────────────────

def _chunk_plan(duration: float) -> list[tuple[float, float | None]]:
    """(start, length) per chunk. Equal lengths, because a short leftover tail
    is exactly what providers reject as too short. The last chunk runs to end
    of file, and unknown duration (0) means a single whole-file chunk."""
    if duration <= 0:
        return [(0.0, None)]
    count = max(1, math.ceil(duration / CHUNK_SECONDS))
    size = duration / count
    return [(i * size, None if i == count - 1 else size) for i in range(count)]


def _encode_chunk(source: Path, target: Path, start: float, length: float | None) -> None:
    """Cut one chunk as 16 kHz mono MP3. Input may be a video file."""
    args = ["-ss", f"{start:.3f}"] if start else []
    args += ["-i", str(source)]
    if length is not None:
        args += ["-t", f"{length:.3f}"]
    args += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k", str(target)]
    result = run_ffmpeg(args, timeout=180)
    if result.returncode != 0:
        raise ExternalTranscriptionError(
            f"Couldn't prepare the audio for upload: {(result.stderr or '').strip()[:200]}"
        )


# ── API call ─────────────────────────────────────────────────────────────────

def _error_detail(resp: httpx.Response) -> str:
    try:
        error = resp.json().get("error")
        message = error.get("message") if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        message = None
    return str(message or resp.text or "no details")[:200]


def _explain_failure(resp: httpx.Response, model: str) -> str:
    detail = _error_detail(resp)
    if resp.status_code == 400:
        return f"{model} rejected the request, possibly because it can't return timestamps ({detail})"
    if resp.status_code in (401, 403):
        return f"OpenRouter rejected the API key ({detail})"
    if resp.status_code == 402:
        return "your OpenRouter account is out of credits"
    if resp.status_code == 404:
        return f"OpenRouter no longer offers {model}"
    return f"OpenRouter returned HTTP {resp.status_code} ({detail})"


async def _post_chunk(
    client: httpx.AsyncClient, path: Path, *, model: str, api_key: str, language: str | None,
) -> dict:
    fields: dict = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language:
        fields["language"] = language
    audio = path.read_bytes()
    headers = {**_OPENROUTER_HEADERS, "Authorization": f"Bearer {api_key}"}

    failure = "no attempt made"
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = await client.post(
                _TRANSCRIBE_URL, headers=headers, data=fields,
                files={"file": (path.name, audio, "audio/mpeg")},
            )
        except httpx.TransportError as e:
            failure = f"couldn't reach OpenRouter ({type(e).__name__})"
        else:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as e:
                    raise ExternalTranscriptionError("OpenRouter sent an unreadable reply") from e
                if not isinstance(payload, dict) or payload.get("error"):
                    raise ExternalTranscriptionError(f"{model} failed: {_error_detail(resp)}")
                return payload
            failure = _explain_failure(resp, model)
            if resp.status_code not in _RETRYABLE_STATUS:
                raise ExternalTranscriptionError(failure)
        if attempt + 1 < _MAX_ATTEMPTS:
            await asyncio.sleep(2 ** attempt)
    raise ExternalTranscriptionError(failure)


# ── Response normalisation ───────────────────────────────────────────────────

def _language_code(value) -> str | None:
    name = str(value or "").strip().lower().replace("_", "-").split("-")[0]
    if not name:
        return None
    return _LANGUAGE_NAMES.get(name, name[:10])


def _is_cjk(text: str) -> bool:
    return any("぀" <= ch <= "鿿" or "가" <= ch <= "힣" for ch in text)


def _timed_word(raw: dict, offset: float) -> dict | None:
    text = str(raw.get("word") or raw.get("text") or "")
    if not text.strip():
        return None
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (KeyError, TypeError, ValueError):
        return None
    confidence = raw.get("confidence")
    return {
        "word": text,
        "start": start + offset,
        "end": max(start, end) + offset,
        "probability": float(confidence) if isinstance(confidence, (int, float)) else 1.0,
    }


def _timed_segment(raw: dict, offset: float) -> dict | None:
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    try:
        start = float(raw["start"])
        end = float(raw["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"start": start + offset, "end": max(start, end) + offset, "text": text, "words": []}


def _segments_from_words(words: list[dict]) -> list[dict]:
    groups: list[list[dict]] = []
    for word in words:
        current = groups[-1] if groups else None
        if (
            current is None
            or len(current) >= _SEGMENT_MAX_WORDS
            or word["start"] - current[-1]["end"] >= _SEGMENT_GAP_SECONDS
            or current[-1]["word"].strip().endswith(_SENTENCE_END)
        ):
            groups.append([word])
        else:
            current.append(word)
    segments = []
    for group in groups:
        pieces = [w["word"].strip() for w in group]
        separator = "" if _is_cjk("".join(pieces)) else " "
        segments.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": separator.join(pieces),
            "words": group,
        })
    return segments


def _normalise_chunk(payload: dict, offset: float, model: str) -> list[dict]:
    """One provider response as faster-whisper-shaped segments on the source
    audio's timeline. Raises when speech came back with no timing at all."""
    raw_segments = [s for s in (payload.get("segments") or []) if isinstance(s, dict)]
    raw_words = payload.get("words") or [
        w for s in raw_segments for w in (s.get("words") or [])
    ]
    words = sorted(
        filter(None, (_timed_word(w, offset) for w in raw_words if isinstance(w, dict))),
        key=lambda w: w["start"],
    )
    segments = sorted(
        filter(None, (_timed_segment(s, offset) for s in raw_segments)),
        key=lambda s: s["start"],
    )

    if not words:
        if segments or str(payload.get("text") or "").strip():
            raise ExternalTranscriptionError(f"{model} returned no word timestamps")
        return []  # genuine silence
    if not segments:
        return _segments_from_words(words)

    index = 0
    for word in words:
        middle = (word["start"] + word["end"]) / 2
        while index + 1 < len(segments) and middle >= segments[index]["end"]:
            index += 1
        segments[index]["words"].append(word)
    # Some providers return one segment per request, which on a 5-minute chunk
    # is too coarse for clip selection to find a boundary in.
    if any(len(segment["words"]) > _SEGMENT_MAX_WORDS for segment in segments):
        return _segments_from_words(words)
    return segments


# ── Entry point ──────────────────────────────────────────────────────────────

async def transcribe_via_openrouter(
    audio_path: str | Path,
    *,
    model: str,
    api_key: str,
    language: str | None = None,
    duration: float = 0.0,
    on_progress=None,
) -> dict:
    """Transcribe with an OpenRouter model. Same return shape as
    `whisper_service.transcribe()`, except `language_probability` is None
    because providers don't report one.

    Raises ExternalTranscriptionError when the result can't be used.
    `on_progress(fraction)` fires as each chunk completes.
    """
    source = Path(audio_path)
    plan = _chunk_plan(duration)
    outcomes: list[tuple[list[dict], str | None] | None] = [None] * len(plan)
    finished = 0
    slots = asyncio.Semaphore(_MAX_PARALLEL)

    with tempfile.TemporaryDirectory(prefix="viralmint-stt-") as tmp:
        async with _http_client() as client:
            async def _run_chunk(index: int) -> None:
                nonlocal finished
                start, length = plan[index]
                target = Path(tmp) / f"chunk-{index:04d}.mp3"
                async with slots:
                    async with _ffmpeg_semaphore:
                        await asyncio.to_thread(_encode_chunk, source, target, start, length)
                    payload = await _post_chunk(
                        client, target, model=model, api_key=api_key, language=language)
                    target.unlink(missing_ok=True)
                outcomes[index] = (
                    _normalise_chunk(payload, start, model),
                    _language_code(payload.get("language")),
                )
                finished += 1
                if on_progress is not None:
                    try:
                        on_progress(min(0.99, finished / len(plan)))
                    except Exception:
                        pass  # progress UI must never break transcription

            await _run_chunk(0)
            tasks = [asyncio.create_task(_run_chunk(i)) for i in range(1, len(plan))]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    segments = [segment for outcome in outcomes for segment in outcome[0]]
    detected = next((outcome[1] for outcome in outcomes if outcome[1]), None)
    return {
        "text": " ".join(segment["text"] for segment in segments),
        "language": _language_code(language) or detected,
        "language_probability": None,
        "segments": segments,
    }

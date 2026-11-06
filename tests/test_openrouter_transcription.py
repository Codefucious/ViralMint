# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025-2026 ViralMint Contributors
"""OpenRouter speech-to-text: chunking, response shapes and engine routing.

What these pin: captions and clip cutting read `segments[].words[]` on the
SOURCE audio's timeline, so every provider's reply has to land in that shape,
and a selected model that can't deliver must never cost the user their
transcription. No network or ffmpeg runs here.
"""
import asyncio
import types
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.core.exceptions import ExternalTranscriptionError
from backend.services import openrouter_transcription as stt
from backend.services import whisper_service as ws_mod
from backend.services.whisper_service import whisper_service

MODEL = "openai/whisper-large-v3"


def _fake_encode(_source, target, _start, _length):
    Path(target).write_bytes(b"mp3")


@pytest.fixture
def api(monkeypatch):
    """Route the module's HTTP through a handler; record requests."""
    state = types.SimpleNamespace(requests=[], respond=None)

    def _handle(request):
        state.requests.append(request)
        return state.respond(request, len(state.requests))

    monkeypatch.setattr(
        stt, "_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(_handle)),
    )
    monkeypatch.setattr(stt, "_encode_chunk", _fake_encode)
    monkeypatch.setattr(stt.asyncio, "sleep", AsyncMock())
    return state


def _reply(start=0.0, text="hello world"):
    return httpx.Response(200, json={
        "text": text,
        "language": "english",
        "segments": [{"start": start, "end": start + 2.0, "text": text}],
        "words": [
            {"word": "hello", "start": start, "end": start + 1.0},
            {"word": "world", "start": start + 1.0, "end": start + 2.0, "confidence": 0.5},
        ],
    })


async def _run(duration=10.0, **kw):
    return await stt.transcribe_via_openrouter(
        "in.mp4", model=MODEL, api_key="k", duration=duration, **kw)


# ── chunk plan ───────────────────────────────────────────────────────────────

def test_short_audio_is_one_chunk_to_end_of_file():
    assert stt._chunk_plan(90.0) == [(0.0, None)]


def test_unknown_duration_is_one_whole_file_chunk():
    assert stt._chunk_plan(0.0) == [(0.0, None)]


def test_long_audio_splits_evenly_with_no_tiny_tail():
    plan = stt._chunk_plan(601.0)
    assert len(plan) == 3
    assert plan[1][0] == pytest.approx(601.0 / 3)
    assert plan[0][1] == pytest.approx(601.0 / 3)
    assert plan[-1][1] is None


# ── response normalisation ───────────────────────────────────────────────────

def test_words_are_placed_in_their_segments_on_the_source_timeline():
    segments = stt._normalise_chunk({
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ],
        "words": [
            {"word": "one", "start": 0.1, "end": 0.9},
            {"word": "two", "start": 1.1, "end": 1.9},
        ],
    }, 300.0, MODEL)
    assert [s["start"] for s in segments] == [300.0, 301.0]
    assert [[w["word"] for w in s["words"]] for s in segments] == [["one"], ["two"]]
    assert segments[1]["words"][0]["start"] == pytest.approx(301.1)


def test_words_nested_inside_segments_are_used():
    segments = stt._normalise_chunk({"segments": [{
        "start": 0.0, "end": 1.0, "text": "hi",
        "words": [{"word": "hi", "start": 0.0, "end": 1.0}],
    }]}, 0.0, MODEL)
    assert segments[0]["words"][0]["word"] == "hi"


def test_segments_are_rebuilt_when_a_provider_returns_only_words():
    segments = stt._normalise_chunk({"words": [
        {"word": "First.", "start": 0.0, "end": 0.5},
        {"word": "Second", "start": 0.6, "end": 1.0},
        {"word": "part", "start": 1.0, "end": 1.4},
        {"word": "later", "start": 5.0, "end": 5.5},
    ]}, 0.0, MODEL)
    assert [s["text"] for s in segments] == ["First.", "Second part", "later"]
    assert segments[1]["start"] == 0.6 and segments[1]["end"] == 1.4


def test_rebuilt_cjk_segments_have_no_spaces():
    segments = stt._normalise_chunk({"words": [
        {"word": "你", "start": 0.0, "end": 0.2},
        {"word": "好", "start": 0.2, "end": 0.4},
    ]}, 0.0, MODEL)
    assert segments[0]["text"] == "你好"


def test_one_giant_segment_is_rebuilt_from_its_words():
    """Gemini's shape: every word of the request inside a single segment."""
    words = [
        {"word": f"w{i}.", "start": float(i), "end": i + 0.5}
        for i in range(stt._SEGMENT_MAX_WORDS + 1)
    ]
    segments = stt._normalise_chunk({
        "segments": [{"start": 0.0, "end": 40.0, "text": "everything"}],
        "words": words,
    }, 0.0, MODEL)
    assert len(segments) == len(words)


@pytest.mark.parametrize("payload", [
    {"text": "spoken words"},
    {"text": "hi", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]},
])
def test_speech_without_word_timing_is_a_failure_not_silently_worse_captions(payload):
    with pytest.raises(ExternalTranscriptionError, match="no word timestamps"):
        stt._normalise_chunk(payload, 0.0, MODEL)


def test_silence_is_an_empty_transcript_not_a_failure():
    assert stt._normalise_chunk({"text": ""}, 0.0, MODEL) == []


@pytest.mark.parametrize("raw,code", [
    ("english", "en"), ("en", "en"), ("en-US", "en"), ("zh_CN", "zh"), ("", None), (None, None),
])
def test_language_becomes_a_code(raw, code):
    assert stt._language_code(raw) == code


# ── transcription ────────────────────────────────────────────────────────────

async def test_result_has_the_shape_downstream_code_reads(api):
    api.respond = lambda *_: _reply()
    result = await _run()
    assert result["text"] == "hello world"
    assert result["language"] == "en"
    word = result["segments"][0]["words"][1]
    assert word == {"word": "world", "start": 1.0, "end": 2.0, "probability": 0.5}


async def test_request_asks_for_word_timestamps_and_sends_the_key(api):
    api.respond = lambda *_: _reply()
    await _run(language="ja")
    request = api.requests[0]
    body = request.read()
    assert request.headers["authorization"] == "Bearer k"
    assert b"verbose_json" in body and MODEL.encode() in body
    assert body.count(b'name="timestamp_granularities[]"') == 2
    assert b'name="language"' in body and b"ja" in body


async def test_chunks_join_in_order_at_their_offsets(api, monkeypatch):
    monkeypatch.setattr(stt, "CHUNK_SECONDS", 10.0)
    api.respond = lambda *_: _reply()
    progress = []
    result = await _run(duration=30.0, on_progress=progress.append)
    assert [s["start"] for s in result["segments"]] == [0.0, 10.0, 20.0]
    assert len(api.requests) == 3
    assert progress[-1] == 0.99 and progress == sorted(progress)


async def test_a_model_without_timestamps_costs_one_request_not_every_chunk(api, monkeypatch):
    monkeypatch.setattr(stt, "CHUNK_SECONDS", 10.0)
    api.respond = lambda *_: httpx.Response(
        400, json={"error": {"message": "verbose_json unsupported"}})
    with pytest.raises(ExternalTranscriptionError, match="verbose_json unsupported"):
        await _run(duration=60.0)
    assert len(api.requests) == 1


async def test_rate_limit_is_retried(api):
    api.respond = lambda _r, n: httpx.Response(429, json={}) if n == 1 else _reply()
    assert (await _run())["text"] == "hello world"
    assert len(api.requests) == 2


async def test_retries_stop_with_a_readable_reason(api):
    api.respond = lambda *_: httpx.Response(503, json={"error": {"message": "overloaded"}})
    with pytest.raises(ExternalTranscriptionError, match="overloaded"):
        await _run()
    assert len(api.requests) == stt._MAX_ATTEMPTS


@pytest.mark.parametrize("status,reason", [
    (401, "API key"), (402, "out of credits"), (404, "no longer offers"),
])
async def test_account_errors_are_not_retried(api, status, reason):
    api.respond = lambda *_: httpx.Response(status, json={})
    with pytest.raises(ExternalTranscriptionError, match=reason):
        await _run()
    assert len(api.requests) == 1


async def test_error_body_in_a_200_is_a_failure(api):
    api.respond = lambda *_: httpx.Response(200, json={"error": {"message": "upstream died"}})
    with pytest.raises(ExternalTranscriptionError, match="upstream died"):
        await _run()


# ── model list ───────────────────────────────────────────────────────────────

@pytest.fixture
def empty_cache(monkeypatch):
    monkeypatch.setattr(stt, "_models_cache", {"at": 0.0, "models": None})


def _catalog():
    row = lambda i, n, out: {"id": i, "name": n, "architecture": {"output_modalities": out}}
    return httpx.Response(200, json={"data": [
        row("z/zed", "Zed", ["transcription"]),
        row("a/alpha", "Alpha", ["transcription"]),
        row("t/text", "Text", ["text"]),
    ]})


async def test_models_are_filtered_sorted_and_cached(api, empty_cache):
    api.respond = lambda *_: _catalog()
    models, stale = await stt.list_transcription_models()
    assert [m["id"] for m in models] == ["a/alpha", "z/zed"] and not stale
    await stt.list_transcription_models()
    assert len(api.requests) == 1


async def test_failed_refresh_serves_the_previous_list(api, monkeypatch):
    cached = [{"id": "a/alpha", "name": "Alpha"}]
    monkeypatch.setattr(stt, "_models_cache", {"at": -1e9, "models": cached})
    api.respond = lambda *_: httpx.Response(500)
    assert await stt.list_transcription_models() == (cached, True)


async def test_no_list_at_all_raises(api, empty_cache):
    api.respond = lambda *_: httpx.Response(500)
    with pytest.raises(ExternalTranscriptionError):
        await stt.list_transcription_models()


# ── engine routing in whisper_service ────────────────────────────────────────

LOCAL_RESULT = {"text": "local", "language": "en", "segments": []}
REMOTE_RESULT = {"text": "remote", "language": "en", "segments": []}


@pytest.fixture
def routing(monkeypatch):
    """Stub both engines; each test sets the saved model and remote behaviour."""
    state = types.SimpleNamespace(
        user=types.SimpleNamespace(transcription_model=MODEL),
        key="k",
        remote=AsyncMock(return_value=REMOTE_RESULT),
        warn=AsyncMock(),
        ensure_model=AsyncMock(),
    )

    class _Model:
        def transcribe(self, *_a, **_k):
            return iter([]), types.SimpleNamespace(language="en", language_probability=1.0)

    state.ensure_model.return_value = _Model()
    monkeypatch.setattr(
        "backend.services.ffmpeg_service.has_audio_stream", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "backend.services.render_post.load_user_settings", AsyncMock(side_effect=lambda _u: state.user))
    monkeypatch.setattr(
        "backend.core.ai_provider.resolve_openrouter_key", lambda _s: state.key)
    monkeypatch.setattr(stt, "transcribe_via_openrouter", state.remote)
    monkeypatch.setattr("backend.core.ws_manager.ws_manager.send_constraint_warning", state.warn)
    monkeypatch.setattr(ws_mod.WhisperService, "ensure_model", state.ensure_model)
    monkeypatch.setattr("backend.services.video_utils.probe_duration", lambda _p: 12.0)
    return state


async def test_selected_model_does_the_work_and_never_loads_whisper(routing):
    assert await whisper_service.transcribe("a.mp3") == REMOTE_RESULT
    routing.ensure_model.assert_not_called()
    assert routing.remote.call_args.kwargs["model"] == MODEL
    assert routing.remote.call_args.kwargs["duration"] == 12.0


async def test_no_selection_stays_local_and_silent(routing):
    routing.user.transcription_model = None
    await whisper_service.transcribe("a.mp3")
    routing.remote.assert_not_called()
    routing.ensure_model.assert_called_once()
    routing.warn.assert_not_called()


async def test_no_settings_row_stays_local(routing):
    routing.user = None
    await whisper_service.transcribe("a.mp3")
    routing.remote.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"timing_only": True}, {"local_only": True}])
async def test_own_narration_never_leaves_the_machine(routing, kwargs):
    await whisper_service.transcribe("a.mp3", **kwargs)
    routing.remote.assert_not_called()


async def test_failure_falls_back_to_whisper_and_says_why(routing):
    routing.remote.side_effect = ExternalTranscriptionError("your account is out of credits")
    result = await whisper_service.transcribe("a.mp3")
    assert result["language"] == "en"
    routing.ensure_model.assert_called_once()
    message = routing.warn.call_args.kwargs["message"]
    assert MODEL in message and "out of credits" in message


async def test_missing_key_falls_back_and_says_why(routing):
    routing.key = ""
    await whisper_service.transcribe("a.mp3")
    routing.remote.assert_not_called()
    assert "no OpenRouter API key" in routing.warn.call_args.kwargs["message"]


async def test_cancelling_a_job_does_not_start_whisper(routing):
    routing.remote.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await whisper_service.transcribe("a.mp3")
    routing.ensure_model.assert_not_called()

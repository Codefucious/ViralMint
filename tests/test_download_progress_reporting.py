# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025-2026 ViralMint Contributors
"""A running download must report how far along it is.

Reported on a 75-minute YouTube video: the job sat at 10% for the whole
transfer and looked hung, while it was pulling ~1.8 MB/s the entire time. The
yt-dlp hook measured progress, but only the stall watchdog ever read it.

These tests pin the path from yt-dlp's hook to the job's progress callback, and
that reporting can never harm the download it is reporting on.
"""
import asyncio

import pytest

from backend.services import ytdlp_service
from backend.services.ytdlp_service import (
    _DownloadProgress,
    _await_with_stall_guard,
    _download_fraction,
    _make_progress_hook,
)


# ── reading a fraction out of a yt-dlp event ────────────────────────────

def test_fragment_counts_win_over_a_byte_estimate():
    """HLS `total_bytes_estimate` drifts as fragments land; the count does not."""
    assert _download_fraction({
        "fragment_index": 450, "fragment_count": 900,
        "downloaded_bytes": 10, "total_bytes_estimate": 1000,
    }) == pytest.approx(0.5)


def test_fraction_from_exact_total_bytes():
    assert _download_fraction(
        {"downloaded_bytes": 250, "total_bytes": 1000}
    ) == pytest.approx(0.25)


def test_fraction_from_estimated_total_bytes():
    assert _download_fraction(
        {"downloaded_bytes": 750, "total_bytes_estimate": 1000}
    ) == pytest.approx(0.75)


def test_no_known_total_is_none_not_zero():
    """Unknown must stay distinguishable from 0%, so the UI can show MB only."""
    assert _download_fraction({"downloaded_bytes": 500}) is None


@pytest.mark.parametrize("event", [
    {"fragment_index": 3, "fragment_count": 0},
    {"downloaded_bytes": 5, "total_bytes": 0},
    {"downloaded_bytes": "5", "total_bytes": "many"},
    {},
])
def test_unusable_events_are_none(event):
    assert _download_fraction(event) is None


def test_fraction_is_clamped_to_one():
    """An estimate can undershoot the real size; the bar must not pass 100%."""
    assert _download_fraction(
        {"downloaded_bytes": 1500, "total_bytes_estimate": 1000}
    ) == 1.0


# ── the tracker ─────────────────────────────────────────────────────────

def test_fraction_starts_unknown():
    assert _DownloadProgress().fraction is None


def test_fraction_never_moves_backward():
    """A DASH audio stream or a fallback format restarts its count at 0."""
    p = _DownloadProgress()
    p.note_fraction(0.9)
    p.note_fraction(0.02)
    assert p.fraction == pytest.approx(0.9)


def test_reset_clears_the_fraction():
    p = _DownloadProgress()
    p.note_fraction(0.9)
    p.reset()
    assert p.fraction is None


def test_hook_feeds_the_fraction():
    p = _DownloadProgress()
    _make_progress_hook(p)({
        "status": "downloading", "downloaded_bytes": 400, "total_bytes": 1000,
    })
    assert p.fraction == pytest.approx(0.4)
    assert p.bytes_downloaded == 400


def test_hook_survives_a_malformed_event():
    """The hook runs inside yt-dlp; raising there would abort the download."""
    p = _DownloadProgress()
    _make_progress_hook(p)({
        "status": "downloading", "fragment_index": object(), "fragment_count": 5,
    })
    assert p.fraction is None


# ── reporting from the guard ────────────────────────────────────────────

async def _feed(tracker, steps, delay=0.03):
    """Stand in for a download: move the tracker, then return a result."""
    for downloaded, fraction in steps:
        tracker.note_bytes(downloaded)
        if fraction is not None:
            tracker.note_fraction(fraction)
        await asyncio.sleep(delay)
    return "done"


def _guard(awaitable, tracker, on_progress):
    return _await_with_stall_guard(
        awaitable, tracker, hard_timeout=30, stall_timeout=30,
        url="https://example.com/v", on_progress=on_progress,
    )


@pytest.fixture
def fast_guard(monkeypatch):
    monkeypatch.setattr(ytdlp_service, "_STALL_POLL_S", 0.01)
    monkeypatch.setattr(ytdlp_service, "_PROGRESS_REPORT_S", 0.0)


async def test_progress_reaches_the_callback(fast_guard):
    seen = []

    async def on_progress(fraction, downloaded):
        seen.append((fraction, downloaded))

    p = _DownloadProgress()
    result = await _guard(_feed(p, [(100, 0.1), (500, 0.5)]), p, on_progress)

    assert result == "done"
    assert (0.1, 100) in seen and (0.5, 500) in seen


async def test_unknown_total_still_reports_bytes(fast_guard):
    seen = []

    async def on_progress(fraction, downloaded):
        seen.append((fraction, downloaded))

    p = _DownloadProgress()
    await _guard(_feed(p, [(100, None)]), p, on_progress)
    assert (None, 100) in seen


async def test_unchanged_progress_is_not_resent(fast_guard):
    """Every report is a DB write, so an idle transfer must cost nothing."""
    seen = []

    async def on_progress(fraction, downloaded):
        seen.append((fraction, downloaded))

    p = _DownloadProgress()
    await _guard(_feed(p, [(100, 0.1)], delay=0.15), p, on_progress)
    assert seen == [(0.1, 100)]


async def test_nothing_is_reported_before_the_first_byte(fast_guard):
    seen = []

    async def on_progress(fraction, downloaded):
        seen.append((fraction, downloaded))

    async def silent():
        await asyncio.sleep(0.1)
        return "done"

    await _guard(silent(), _DownloadProgress(), on_progress)
    assert seen == []


async def test_reports_are_throttled(monkeypatch):
    monkeypatch.setattr(ytdlp_service, "_STALL_POLL_S", 0.01)
    monkeypatch.setattr(ytdlp_service, "_PROGRESS_REPORT_S", 60.0)
    seen = []

    async def on_progress(fraction, downloaded):
        seen.append(downloaded)

    p = _DownloadProgress()
    steps = [(n * 100, n / 20) for n in range(1, 11)]
    await _guard(_feed(p, steps), p, on_progress)
    assert len(seen) == 1


async def test_a_failing_callback_does_not_fail_the_download(fast_guard):
    async def on_progress(fraction, downloaded):
        raise RuntimeError("database is locked")

    p = _DownloadProgress()
    assert await _guard(_feed(p, [(100, 0.1)]), p, on_progress) == "done"


async def test_a_hung_callback_does_not_block_the_watchdog(fast_guard, monkeypatch):
    monkeypatch.setattr(ytdlp_service, "_PROGRESS_CALLBACK_TIMEOUT_S", 0.02)

    async def on_progress(fraction, downloaded):
        await asyncio.sleep(60)

    p = _DownloadProgress()
    result = await asyncio.wait_for(
        _guard(_feed(p, [(100, 0.1)]), p, on_progress), timeout=5,
    )
    assert result == "done"


async def test_no_callback_is_the_default():
    p = _DownloadProgress()
    result = await _await_with_stall_guard(
        _feed(p, [(100, 0.1)]), p, hard_timeout=30, stall_timeout=30,
        url="https://example.com/v",
    )
    assert result == "done"


# ── what the job shows ──────────────────────────────────────────────────

@pytest.fixture
def sent(monkeypatch):
    from backend.core.ws_manager import ws_manager

    calls = []

    async def fake_send(job_id, pct, step, user_id="local"):
        calls.append((job_id, pct, step, user_id))

    monkeypatch.setattr(ws_manager, "send_progress", fake_send)
    return calls


@pytest.mark.parametrize("fraction,pct", [(0.0, 10), (0.5, 37.5), (1.0, 65)])
async def test_download_fills_the_band_before_analysis(sent, fraction, pct):
    """10% is where the transfer starts and 70% is where analysis starts."""
    from backend.core.task_runner import _download_progress_reporter

    await _download_progress_reporter("job", "user")(fraction, 1024 * 1024)
    assert sent[0][1] == pytest.approx(pct)


async def test_step_text_shows_percent_and_size(sent):
    from backend.core.task_runner import _download_progress_reporter

    await _download_progress_reporter("job", "user")(0.43, 325 * 1024 * 1024)
    assert sent == [("job", pytest.approx(33.65),
                     "Downloading video... 43% (325 MB)", "user")]


async def test_unknown_total_shows_size_without_inventing_a_percent(sent):
    from backend.core.task_runner import _download_progress_reporter

    await _download_progress_reporter("job", "user")(None, 325 * 1024 * 1024)
    assert sent == [("job", 10, "Downloading video... 325 MB", "user")]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025-2026 ViralMint Contributors
"""Speaker framing: the crop path, and when it must step aside for blur-fill.

The path is what viewers see, so these pin its behaviour rather than its maths:
it holds still, follows a walking speaker, jumps only on a real cut, and never
leaves the frame. The real model runs once, on a blank frame.
"""
import asyncio
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from backend.core.exceptions import JobCancelledError
from backend.services import clip_extractor as CE
from backend.services import ffmpeg_service as fs
from backend.services import speaker_crop as sc
from backend.services.speaker_crop import CropSegment, Face, plan_crop

WIDTH = 1280
CROP = 404


def _samples(positions, step=0.5):
    """One sample per position: a face at x, or none for None, or a crowd for a list."""
    out = []
    for i, x in enumerate(positions):
        xs = [] if x is None else (x if isinstance(x, list) else [x])
        out.append((i * step, [Face(float(v), 100.0) for v in xs]))
    return out


def _x_at(segments, t):
    for s in segments:
        if s.start <= t < s.end:
            return s.x_from + (s.x_to - s.x_from) * (t - s.start) / (s.end - s.start)
    return segments[-1].x_to


def _center(x):
    return x - CROP / 2


def test_no_faces_anywhere_means_blur_not_a_blind_crop():
    assert plan_crop(_samples([None] * 6), [], 3.0, WIDTH, CROP) is None


def test_speaker_is_centred_in_the_crop():
    segments = plan_crop(_samples([300] * 10), [], 5.0, WIDTH, CROP)
    assert _x_at(segments, 2.0) == pytest.approx(_center(300))


def test_small_movement_does_not_move_the_crop():
    wobble = [640, 655, 630, 650, 640, 625, 650, 640]
    segments = plan_crop(_samples(wobble), [], 4.0, WIDTH, CROP)
    assert len({(s.x_from, s.x_to) for s in segments}) == 1


def test_crop_follows_a_walking_speaker_without_jumping():
    walk = [300 + 40 * i for i in range(20)]
    segments = plan_crop(_samples(walk), [], 10.0, WIDTH, CROP)
    assert _x_at(segments, 9.9) > _x_at(segments, 0.0) + 300
    steps = [abs(_x_at(segments, t / 10 + 0.1) - _x_at(segments, t / 10)) for t in range(99)]
    assert max(steps) < 30


def test_a_camera_cut_jumps_the_crop_straight_to_the_new_position():
    samples = _samples([300] * 6 + [900] * 6)
    segments = plan_crop(samples, [3.0], 6.0, WIDTH, CROP)
    assert _x_at(segments, 2.9) == pytest.approx(_center(300))
    assert _x_at(segments, 3.0) == pytest.approx(_center(900))


def test_a_shot_with_no_face_holds_the_centre():
    samples = _samples([300] * 4 + [None] * 4)
    segments = plan_crop(samples, [2.0], 4.0, WIDTH, CROP)
    assert _x_at(segments, 3.0) == pytest.approx((WIDTH - CROP) / 2)


def test_an_audience_shot_holds_the_centre_instead_of_hopping_between_faces():
    crowd = [[100, 500, 1100]] * 6
    segments = plan_crop(_samples([300] * 4 + crowd), [2.0], 5.0, WIDTH, CROP)
    assert _x_at(segments, 4.0) == pytest.approx((WIDTH - CROP) / 2)


def test_one_stray_detection_cannot_pull_the_crop():
    samples = _samples([300, 300, 300, 1200, 300, 300, 300])
    segments = plan_crop(samples, [], 3.5, WIDTH, CROP)
    assert all(_x_at(segments, t / 10) == pytest.approx(_center(300)) for t in range(35))


@pytest.mark.parametrize("x", [5, WIDTH - 5])
def test_crop_never_leaves_the_frame(x):
    segments = plan_crop(_samples([x] * 6), [], 3.0, WIDTH, CROP)
    for s in segments:
        assert 0 <= s.x_from <= WIDTH - CROP and 0 <= s.x_to <= WIDTH - CROP


def _gray(levels):
    return np.array([np.full((36, 64), v, np.uint8) for v in levels])


def test_a_sudden_change_is_a_cut_at_the_right_time():
    assert sc.cuts_from_frames(_gray([50] * 50 + [120] * 50), 10.0) == [5.0]


def test_steady_motion_is_not_a_cut():
    """Large but not sudden: every frame changes by the same big amount."""
    assert sc.cuts_from_frames(_gray([i * 10 for i in range(25)]), 10.0) == []


def test_a_slow_dissolve_is_not_a_cut():
    assert sc.cuts_from_frames(_gray([50 + i for i in range(100)]), 10.0) == []


def test_a_cut_at_the_very_edge_of_a_clip_is_ignored():
    assert sc.cuts_from_frames(_gray([50] * 2 + [120] * 98), 10.0) == []


def test_flash_shorter_than_a_shot_counts_once():
    frames = _gray([50] * 50 + [200] * 2 + [50] * 48)
    assert sc.cuts_from_frames(frames, 10.0) == [5.0]


def test_expression_covers_the_whole_clip_and_stays_flat():
    expression = sc._x_expression([
        CropSegment(0.0, 2.0, 100.0, 100.0),
        CropSegment(2.0, 4.0, 100.0, 200.0),
    ])
    assert expression == (
        "gte(t,0.000)*lt(t,2.000)*100.0+gte(t,2.000)*(100.0+50.000*(t-2.000))"
    )
    assert "if(" not in expression


def test_real_model_finds_nothing_in_a_blank_frame():
    assert sc._detect(np.zeros((640, 640, 3), np.uint8)) == []


def test_model_ships_with_the_backend():
    assert sc.MODEL_PATH.stat().st_size > 200_000


def test_a_source_too_narrow_to_crop_uses_blur():
    assert sc.speaker_zoom_filter(Path("x.mp4"), 0, 5, 400, 720) is None


def test_detection_failure_uses_blur_rather_than_losing_the_clip(monkeypatch):
    def _boom(*a):
        raise RuntimeError("model broke")
    monkeypatch.setattr(sc, "_sample_faces", _boom)
    assert sc.speaker_zoom_filter(Path("x.mp4"), 0, 5, 1280, 720) is None


# ── extract_clip ─────────────────────────────────────────────────────────────

@pytest.fixture
def ffmpeg_calls(monkeypatch):
    calls = []

    def _fake_run(cmd, **_):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="1280,720\n", stderr="")
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 2000)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(fs.subprocess, "run", _fake_run)
    return calls


def _filter(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


async def test_zoom_uses_the_speaker_crop(ffmpeg_calls, monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "speaker_zoom_filter", lambda *a: "crop=404:720:x='1':y=0")
    await fs.extract_clip(tmp_path / "s.mp4", 1, 6, tmp_path / "o.mp4", vertical_fill="zoom")
    assert _filter(ffmpeg_calls[-1]).startswith("crop=404")


async def test_zoom_with_nothing_to_frame_falls_back_to_blur(ffmpeg_calls, monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "speaker_zoom_filter", lambda *a: None)
    await fs.extract_clip(tmp_path / "s.mp4", 1, 6, tmp_path / "o.mp4", vertical_fill="zoom")
    assert "boxblur" in _filter(ffmpeg_calls[-1])


async def test_blur_never_runs_detection(ffmpeg_calls, monkeypatch, tmp_path):
    def _unexpected(*a):
        raise AssertionError("blur must not run detection")
    monkeypatch.setattr(sc, "speaker_zoom_filter", _unexpected)
    await fs.extract_clip(tmp_path / "s.mp4", 1, 6, tmp_path / "o.mp4", vertical_fill="blur")
    assert "boxblur" in _filter(ffmpeg_calls[-1])


# ── cancelling during the cut ────────────────────────────────────────────────

async def test_cancelled_job_stops_cutting_and_frees_the_encoders(monkeypatch, tmp_path):
    """The bug: a cancelled 58-clip job kept encoding every clip, and a newer
    job sat behind it for the shared ffmpeg slots."""
    started = []

    async def _extract(src, start, end, out, **kw):
        started.append(start)
        Path(out).write_bytes(b"clip")

    async def _cancelled_after_two(_job_id):
        return len(started) >= 2

    async def _progress(*a, **k):
        return None

    monkeypatch.setattr(fs, "extract_clip", _extract)
    monkeypatch.setattr("backend.agents.job_helper.job_cancelled", _cancelled_after_two)
    monkeypatch.setattr(CE.ws_manager, "send_progress", _progress)
    monkeypatch.setattr(CE, "settings", types.SimpleNamespace(GENERATED_DIR=tmp_path))
    monkeypatch.setattr(CE, "_ffmpeg_semaphore", asyncio.Semaphore(1))

    video = type("V", (), {"id": "abcdefgh", "video_path": str(tmp_path / "s.mp4"), "title": "t"})()
    with pytest.raises(JobCancelledError):
        await CE._process_clips_parallel(
            video=video,
            clip_windows=[{"start": float(i), "end": i + 1.0} for i in range(10)],
            segments=[], caption_style="none", user_settings=None, job_id="job",
        )
    assert len(started) == 2
    assert list(tmp_path.glob("clip_*")) == []

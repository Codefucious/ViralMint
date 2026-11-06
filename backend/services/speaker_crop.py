# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025-2026 ViralMint Contributors
"""Frame a landscape clip on its speaker for a 9:16 crop.

A centre crop isn't enough for real footage: the speaker stands in a different
place in every shot, camera cuts move them instantly, and they walk. So:

  1. Tiny grey frames from ffmpeg reveal the exact camera cuts in the clip.
  2. YuNet finds faces in low-res samples (2 per second).
  3. Each shot gets its own crop path. Inside a shot the crop holds still for
     small movements and glides after a speaker who walks. It only jumps on a
     real cut, where a jump is invisible.
  4. The path becomes one `crop` filter whose x depends on time.

The result is None when no face appears anywhere, because cropping slides or a
screen recording would throw away content. Callers then use blur-fill.

onnxruntime and numpy arrive with faster-whisper, so this adds no dependency.
The model is YuNet (MIT, Shiqi Yu) from OpenCV Zoo: backend/assets, 232 KB.
"""
import logging
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np

from backend.services.video_utils import run_ffmpeg

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "face_detection_yunet_2023mar.onnx"

_INPUT_SIZE = 640          # the model's fixed input
_STRIDES = (8, 16, 32)
_MIN_SCORE = 0.6
_NMS_IOU = 0.3

_SAMPLE_FPS = 2.0
_CUT_FRAME_WIDTH = 64
_CUT_FRAME_HEIGHT = 36
_CUT_MIN_CHANGE = 6.0      # mean grey-level change; real cuts measured 14-25
_CUT_SPIKE_RATIO = 4.0     # how far above the surrounding frames a cut must be
_MIN_SHOT_SECONDS = 0.5    # a "cut" closer than this is a flash, not a shot
_CROWD_FACES = 3           # this many faces per sample: an audience shot
_DEAD_ZONE = 0.12          # share of crop width the speaker may drift unfollowed
_FOLLOW_RATE = 0.5         # share of the excess distance covered per sample
_DETECT_TIMEOUT_S = 180


@dataclass(frozen=True)
class Face:
    center_x: float
    area: float


@dataclass(frozen=True)
class CropSegment:
    start: float
    end: float
    x_from: float
    x_to: float


# ── Face detection ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _session():
    import onnxruntime
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2  # clips run in parallel; don't oversubscribe
    return onnxruntime.InferenceSession(
        str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = boxes[:, 2] * boxes[:, 3]
    while order.size:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 0] + boxes[best, 2], boxes[rest, 0] + boxes[rest, 2])
        y2 = np.minimum(boxes[best, 1] + boxes[best, 3], boxes[rest, 1] + boxes[rest, 3])
        overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = overlap / (areas[best] + areas[rest] - overlap + 1e-6)
        order = rest[iou <= _NMS_IOU]
    return keep


def _decode_faces(outputs: dict[str, np.ndarray]) -> list[Face]:
    """YuNet's per-stride grids → faces, in 640×640 input pixels."""
    boxes, scores = [], []
    for stride in _STRIDES:
        cells = _INPUT_SIZE // stride
        cls = outputs[f"cls_{stride}"].reshape(-1)
        obj = outputs[f"obj_{stride}"].reshape(-1)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        hits = np.nonzero(score >= _MIN_SCORE)[0]
        if not hits.size:
            continue
        rows, cols = hits // cells, hits % cells
        center_x = (cols + bbox[hits, 0]) * stride
        center_y = (rows + bbox[hits, 1]) * stride
        width = np.exp(bbox[hits, 2]) * stride
        height = np.exp(bbox[hits, 3]) * stride
        boxes.append(np.stack([center_x - width / 2, center_y - height / 2, width, height], axis=1))
        scores.append(score[hits])
    if not boxes:
        return []
    all_boxes, all_scores = np.concatenate(boxes), np.concatenate(scores)
    return [
        Face(float(all_boxes[i, 0] + all_boxes[i, 2] / 2), float(all_boxes[i, 2] * all_boxes[i, 3]))
        for i in _non_max_suppression(all_boxes, all_scores)
    ]


def _detect(frame: np.ndarray) -> list[Face]:
    """Faces in a 640×640 BGR frame."""
    session = _session()
    tensor = frame.astype(np.float32).transpose(2, 0, 1)[None]
    names = [o.name for o in session.get_outputs()]
    values = session.run(names, {session.get_inputs()[0].name: tensor})
    return _decode_faces(dict(zip(names, values)))


def _sample_faces(video: Path, start: float, duration: float, scale: float) -> list[tuple[float, list[Face]]]:
    """(time, faces) through the clip, with x in source pixels. `scale` is
    source pixels per model pixel. Sources here are always landscape, so the
    frame fits the model's square input at full width, padded below."""
    filters = f"fps={_SAMPLE_FPS},scale={_INPUT_SIZE}:-2,pad={_INPUT_SIZE}:{_INPUT_SIZE}:0:0"
    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(video),
         "-t", str(duration), "-an", "-vf", filters, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    watchdog = threading.Timer(_DETECT_TIMEOUT_S, process.kill)
    watchdog.start()
    frame_bytes = _INPUT_SIZE * _INPUT_SIZE * 3
    samples: list[tuple[float, list[Face]]] = []
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(_INPUT_SIZE, _INPUT_SIZE, 3)
            faces = [Face(f.center_x * scale, f.area) for f in _detect(frame)]
            samples.append((len(samples) / _SAMPLE_FPS, faces))
    finally:
        watchdog.cancel()
        process.kill()
        process.stdout.close()
        process.wait()
    return samples


def cuts_from_frames(frames: np.ndarray, duration: float) -> list[float]:
    """Cut times from a (count, height, width) stack of tiny grey frames.

    A cut is a change that is both large and sudden: a one-frame spike well
    above the frames around it. Fast motion is large but not sudden, so a
    speaker waving a hand stays one shot. A slow dissolve has no spike and
    isn't a cut, which suits the crop: it glides through instead of jumping.
    """
    if len(frames) < 2:
        return []
    change = np.abs(frames[1:].astype(np.int16) - frames[:-1]).mean(axis=(1, 2))
    seconds_per_frame = duration / len(frames)
    cuts: list[float] = []
    for index in np.nonzero(change > _CUT_MIN_CHANGE)[0]:
        nearby = np.r_[change[max(0, index - 6):index], change[index + 1:index + 7]]
        if nearby.size and change[index] <= _CUT_SPIKE_RATIO * np.median(nearby):
            continue
        cut = float((index + 1) * seconds_per_frame)
        if _MIN_SHOT_SECONDS <= cut <= duration - _MIN_SHOT_SECONDS:
            if not cuts or cut - cuts[-1] >= _MIN_SHOT_SECONDS:
                cuts.append(cut)
    return cuts


def _find_cuts(video: Path, start: float, duration: float) -> list[float]:
    """Camera cut times in seconds from the start of the clip.

    Measured here rather than with ffmpeg's `scene` score, whose timestamps
    came back past the end of the clip, so real cuts were thrown away.
    """
    result = run_ffmpeg(
        ["-ss", str(start), "-i", str(video), "-t", str(duration), "-an",
         "-vf", f"scale={_CUT_FRAME_WIDTH}:{_CUT_FRAME_HEIGHT},format=gray",
         "-f", "rawvideo", "pipe:1"],
        timeout=_DETECT_TIMEOUT_S, binary_stdout=True,
    )
    frame_bytes = _CUT_FRAME_WIDTH * _CUT_FRAME_HEIGHT
    count = len(result.stdout) // frame_bytes
    frames = np.frombuffer(result.stdout[:count * frame_bytes], np.uint8)
    return cuts_from_frames(frames.reshape(count, _CUT_FRAME_HEIGHT, _CUT_FRAME_WIDTH), duration)


# ── Planning the crop path ───────────────────────────────────────────────────

def _speaker_track(samples: list[tuple[float, list[Face]]]) -> list[tuple[float, float]]:
    """(time, x) of the likely speaker: the biggest face in each sample, with a
    5-sample median so one misdetection or a passing face can't pull the crop."""
    raw = [(t, max(faces, key=lambda f: f.area).center_x) for t, faces in samples if faces]
    track = []
    for i, (t, _) in enumerate(raw):
        window = [x for _, x in raw[max(0, i - 2):i + 3]]
        track.append((t, median(window)))
    return track


def _plan_shot(
    start: float, end: float, samples: list[tuple[float, list[Face]]],
    source_width: int, crop_width: int,
) -> list[CropSegment]:
    """The crop path for one shot, as left-edge x positions."""
    highest = source_width - crop_width

    def left_edge(center: float) -> float:
        return float(min(highest, max(0.0, center - crop_width / 2)))

    track = _speaker_track(samples)
    crowd = bool(samples) and median(len(faces) for _, faces in samples) >= _CROWD_FACES
    if not track or crowd:
        centered = left_edge(source_width / 2)
        return [CropSegment(start, end, centered, centered)]

    dead_zone = crop_width * _DEAD_ZONE
    position = median(x for _, x in track[:3])
    points = [(start, position)]
    for t, x in track:
        distance = x - position
        if abs(distance) > dead_zone:
            excess = abs(distance) - dead_zone
            position += (excess if distance > 0 else -excess) * _FOLLOW_RATE
        if t > points[-1][0]:
            points.append((t, position))
    points.append((end, position))

    segments: list[CropSegment] = []
    for (t0, c0), (t1, c1) in zip(points, points[1:]):
        x0, x1 = left_edge(c0), left_edge(c1)
        if segments and segments[-1].x_from == segments[-1].x_to == x0 == x1:
            segments[-1] = CropSegment(segments[-1].start, t1, x0, x1)
        elif t1 > t0:
            segments.append(CropSegment(t0, t1, x0, x1))
    return segments


def plan_crop(
    samples: list[tuple[float, list[Face]]], cuts: list[float], duration: float,
    source_width: int, crop_width: int,
) -> list[CropSegment] | None:
    """The whole clip's crop path, or None when no face was ever seen."""
    if not any(faces for _, faces in samples):
        return None
    edges = [0.0, *cuts, duration]
    segments: list[CropSegment] = []
    for start, end in zip(edges, edges[1:]):
        in_shot = [s for s in samples if start <= s[0] < end]
        segments += _plan_shot(start, end, in_shot, source_width, crop_width)
    return segments


def _x_expression(segments: list[CropSegment]) -> str:
    """Time → x as a flat sum of gated terms. Not nested if()s: ffmpeg's
    expression parser has a depth limit that a long clip would reach."""
    terms = []
    for index, s in enumerate(segments):
        last = index == len(segments) - 1
        gate = f"gte(t,{s.start:.3f})" if last else f"gte(t,{s.start:.3f})*lt(t,{s.end:.3f})"
        if s.x_from == s.x_to:
            value = f"{s.x_from:.1f}"
        else:
            slope = (s.x_to - s.x_from) / (s.end - s.start)
            value = f"({s.x_from:.1f}+{slope:.3f}*(t-{s.start:.3f}))"
        terms.append(f"{gate}*{value}")
    return "+".join(terms)


def speaker_zoom_filter(
    video: Path, start: float, duration: float, source_width: int, source_height: int,
    target_width: int = 1080, target_height: int = 1920,
) -> str | None:
    """A filter that zooms in on the speaker, or None to use blur-fill instead:
    no face in the clip, or detection failed. Never raises, because framing
    must not cost the user a clip."""
    crop_width = int(source_height * target_width / target_height) // 2 * 2
    if crop_width >= source_width:
        return None
    try:
        scale = source_width / _INPUT_SIZE
        samples = _sample_faces(video, start, duration, scale)
        segments = plan_crop(samples, _find_cuts(video, start, duration),
                             duration, source_width, crop_width)
    except Exception:
        logger.warning("Speaker detection failed for %s; using blur-fill", video, exc_info=True)
        return None
    if segments is None:
        logger.info("No faces in %s at %.1fs; using blur-fill", video.name, start)
        return None
    return (
        f"crop={crop_width}:{source_height}:x='{_x_expression(segments)}':y=0,"
        f"scale={target_width}:{target_height}:flags=lanczos"
    )

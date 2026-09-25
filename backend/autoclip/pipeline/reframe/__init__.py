"""Reframe stage — 16:9 source to a 9:16 crop path that follows the speaker.

Pipeline for one clip:

1. Split into shots. Crop paths never interpolate across a cut.
2. Sample face landmarks across the clip.
3. Link observations into per-person tracks.
4. Map diarized speakers onto tracks once, then let turn boundaries drive framing.
5. Per shot, pick a strategy — TRACK, WIDE, or GENERAL.
6. Build a raw crop path, drive it through the lazy-follow (a hysteresis
   dead-band that parks the camera while the speaker stays in frame centre
   and only chases — slowly — when they genuinely leave; while parked it
   settles toward the subject at an invisible ~1 px/s so the resting frame
   ends centred rather than wherever the last chase happened to stop), and
   emit segments.

The quality bar is "never jarring": no visible jitter, no cut-off faces, the
speaker on screen for essentially all of their speaking time. Every default here
is biased toward stillness — a locked frame that is slightly off-centre beats a
frame that is always correct and always moving.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

from .. import ffmpeg
from ..transcript import Transcript
from .croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
    centre_crop,
    target_crop_size,
)
from .faces import FaceDetectionUnavailable, FaceObservation, sample_faces
from .scenes import Shot, detect_shots
from .smoothing import SmoothingConfig, lazy_follow
from .speaker import assign_speakers, map_tracks_to_speakers
from .tracker import FaceTrack, build_tracks

log = logging.getLogger(__name__)

__all__ = [
    "CropPath",
    "CropSegment",
    "ReframeConfig",
    "Strategy",
    "build_crop_path",
]

#: Where the eye line sits within the crop, as a fraction of crop height.
#: Portrait convention puts the eyes near the upper third; 0.38 reads as
#: composed without leaving the chin tight to the bottom edge.
EYE_LINE_RATIO = 0.38

#: A subject whose ideal framing stays within this span for the whole shot is
#: genuinely static and is better locked than tracked. Kept just above the
#: dead zone so detection noise on a still subject reads as "barely moves",
#: not "never moves".
LOCK_IF_SPREAD_BELOW_PX = 40.0

#: Fraction of samples trimmed from each end when measuring a shot's spread,
#: so one or two mis-detections on an otherwise still subject cannot inflate
#: the spread and wrongly force a tracked (moving) framing.
SPREAD_TRIM_FRACTION = 0.1

#: If detected faces span more than this fraction of the crop width, no crop can
#: hold them and we fall back to a fitted (blurred-background) frame.
FIT_IF_SPREAD_EXCEEDS = 0.95

#: Slow push-in applied to shots with no face at all, to stop a static wide from
#: feeling like a still image. Off by default — see the note in export._zoom_filter.
GENERAL_ZOOM = 0.0


@dataclass
class ReframeConfig:
    aspect_w: int = 9
    aspect_h: int = 16
    sample_fps: float = 5.0
    smoothing: SmoothingConfig | None = None
    #: Skip face detection entirely and centre-crop everything.
    centre_only: bool = False


def build_crop_path(
    video: Path,
    *,
    start_s: float,
    end_s: float,
    transcript: Transcript | None = None,
    config: ReframeConfig | None = None,
) -> CropPath:
    """Compute the crop path for one clip.

    Falls back to a centred crop whenever face detection is unavailable or finds
    nothing — a centre crop is an acceptable result, an exception is not.

    Preconditions:
        video exists; 0 <= start_s < end_s <= source duration.
    """
    config = config or ReframeConfig()
    duration = end_s - start_s

    info = ffmpeg.probe(video)
    if not info.has_video or not info.width or not info.height:
        raise ValueError(f"{video.name} has no video stream to reframe.")

    source_w, source_h = info.width, info.height
    crop_w, crop_h = target_crop_size(source_w, source_h, config.aspect_w, config.aspect_h)

    if config.centre_only:
        return centre_crop(
            source_w, source_h, duration, aspect_w=config.aspect_w, aspect_h=config.aspect_h
        )

    try:
        observations = sample_faces(
            video, start_s=start_s, end_s=end_s, sample_fps=config.sample_fps
        )
    except FaceDetectionUnavailable as exc:
        log.warning("Face detection unavailable (%s); using a centre crop.", exc)
        return centre_crop(
            source_w,
            source_h,
            duration,
            aspect_w=config.aspect_w,
            aspect_h=config.aspect_h,
            zoom=GENERAL_ZOOM,
        )

    if not observations:
        log.info("No faces detected in %s; using a centre crop.", video.name)
        return centre_crop(
            source_w,
            source_h,
            duration,
            aspect_w=config.aspect_w,
            aspect_h=config.aspect_h,
            zoom=GENERAL_ZOOM,
        )

    tracks = build_tracks(observations)
    shots = detect_shots(video, start_s=start_s, end_s=end_s)

    turns = _clip_relative_turns(transcript, start_s, end_s)
    speaker_map = map_tracks_to_speakers(tracks, turns) if turns else {}

    segments: list[CropSegment] = []
    for shot in shots:
        segments.extend(
            _segments_for_shot(
                shot,
                tracks=tracks,
                turns=turns,
                speaker_map=speaker_map,
                source_w=source_w,
                source_h=source_h,
                crop_w=crop_w,
                crop_h=crop_h,
                config=config,
            )
        )

    if not segments:
        return centre_crop(
            source_w, source_h, duration, aspect_w=config.aspect_w, aspect_h=config.aspect_h
        )

    # Guarantee the segments tile the clip exactly; a gap or overlap would
    # desync the concatenated render from the audio.
    segments[0].start_s = 0.0
    segments[-1].end_s = duration
    for index in range(len(segments) - 1):
        segments[index].end_s = segments[index + 1].start_s

    segments = [s for s in segments if s.duration_s > 0.01]

    log.info(
        "Reframed %s into %d segment(s): %s",
        video.name,
        len(segments),
        ", ".join(s.strategy.value for s in segments),
    )
    return CropPath(source_width=source_w, source_height=source_h, segments=segments)


def _clip_relative_turns(
    transcript: Transcript | None, start_s: float, end_s: float
) -> list[tuple[str, float, float]]:
    """Rebase diarization turns onto the clip's timeline, dropping non-overlapping ones."""
    if transcript is None or not transcript.has_diarization:
        return []

    turns: list[tuple[str, float, float]] = []
    for speaker, turn_start, turn_end in transcript.speaker_turns():
        overlap_start = max(turn_start, start_s)
        overlap_end = min(turn_end, end_s)
        if overlap_end > overlap_start:
            turns.append((speaker, overlap_start - start_s, overlap_end - start_s))
    return turns


def _segments_for_shot(
    shot: Shot,
    *,
    tracks: list[FaceTrack],
    turns: list[tuple[str, float, float]],
    speaker_map: dict[str, FaceTrack],
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> list[CropSegment]:
    assignments = assign_speakers(
        tracks,
        shot_start_s=shot.start_s,
        shot_end_s=shot.end_s,
        turns=turns,
        speaker_map=speaker_map,
    )

    segments: list[CropSegment] = []
    for assignment in assignments:
        visible = [
            t for t in tracks if t.observations_between(assignment.start_s, assignment.end_s)
        ]

        if assignment.track is not None:
            segments.append(
                _track_segment(
                    assignment.track,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    config=config,
                )
            )
        elif visible:
            segments.append(
                _wide_segment(
                    visible,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )
        else:
            segments.append(
                _general_segment(
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )

    return segments


def _track_segment(
    track: FaceTrack,
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> CropSegment:
    """Frame a single subject, tracking them or locking on if they barely move.

    Close-ups are *not* locked. A locked close-up is the worst off-centre
    offender: the mean position averages out the subject's drift, so a talking
    head that shifts while speaking gets frozen slightly off-centre with no way
    to correct it. Tracking a close-up costs nothing — the lazy-follow's dead
    band keeps the frame parked while the subject stays put.
    """
    observations = track.observations_between(start_s, end_s)
    if not observations:
        return _general_segment(
            start_s=start_s,
            end_s=end_s,
            source_w=source_w,
            source_h=source_h,
            crop_w=crop_w,
            crop_h=crop_h,
        )

    raw = [
        (
            o.t,
            _clamp(o.cx - crop_w / 2, 0, source_w - crop_w),
            _clamp(o.eye_y - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
        )
        for o in observations
    ]

    # Spread is measured robustly (a trimmed range, not max-min): a single
    # mis-detection on a still subject must not inflate the spread and wrongly
    # force a tracked framing. The same outlier-robustness the median lock
    # position buys has to apply to the lock *decision* too.
    spread_x = _trimmed_spread([x for _, x, _ in raw])
    spread_y = _trimmed_spread([y for _, _, y in raw])

    # A subject who barely moves is better locked than tracked — but locked at
    # the median of where they actually were, not the mean. The median is
    # robust to detection outliers, which otherwise pull a mean-locked frame
    # off the subject for the whole shot.
    should_lock = (
        spread_x < LOCK_IF_SPREAD_BELOW_PX and spread_y < LOCK_IF_SPREAD_BELOW_PX
    ) or len(raw) < 3

    if should_lock:
        x = statistics.median(item[1] for item in raw)
        y = statistics.median(item[2] for item in raw)
        keyframes = [CropKeyframe(t=start_s, x=x, y=y)]
    else:
        smoothing = config.smoothing or SmoothingConfig()
        # Lazy follow: the crop parks while the subject stays inside its band,
        # so pacing around a centre won't re-frame every step. The band scales
        # with the crop's tight dimension.
        xs = lazy_follow([(t, x) for t, x, _ in raw], smoothing, reference_px=crop_w)
        ys = lazy_follow([(t, y) for t, _, y in raw], smoothing, reference_px=crop_w)
        keyframes = [
            CropKeyframe(
                t=t,
                x=_clamp(x, 0.0, float(source_w - crop_w)),
                y=_clamp(y, 0.0, float(source_h - crop_h)),
            )
            for (t, x), (_, y) in zip(xs, ys, strict=True)
        ]
        # Anchor the ends so the expression covers the whole segment.
        if keyframes[0].t > start_s:
            keyframes.insert(0, CropKeyframe(start_s, keyframes[0].x, keyframes[0].y))
        if keyframes[-1].t < end_s:
            keyframes.append(CropKeyframe(end_s, keyframes[-1].x, keyframes[-1].y))

    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=keyframes,
        strategy=Strategy.TRACK,
    )


def _wide_segment(
    visible: list[FaceTrack],
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> CropSegment:
    """Hold several faces at once — locked, never panning.

    Panning between two people who are both on screen is the most obviously
    robotic thing an auto-reframer does. If they fit, lock on their centroid; if
    they don't, fit the whole frame rather than cutting someone out.
    """
    observations: list[FaceObservation] = [
        o for track in visible for o in track.observations_between(start_s, end_s)
    ]
    if not observations:
        return _general_segment(
            start_s=start_s,
            end_s=end_s,
            source_w=source_w,
            source_h=source_h,
            crop_w=crop_w,
            crop_h=crop_h,
        )

    left = min(o.left for o in observations)
    right = max(o.right for o in observations)
    spread = right - left

    if spread > crop_w * FIT_IF_SPREAD_EXCEEDS:
        return CropSegment(
            start_s=start_s,
            end_s=end_s,
            width=source_w,
            height=source_h,
            keyframes=[CropKeyframe(t=start_s, x=0.0, y=0.0)],
            strategy=Strategy.WIDE,
            fit=True,
        )

    centre_x = (left + right) / 2
    eye_y = sum(o.eye_y for o in observations) / len(observations)

    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[
            CropKeyframe(
                t=start_s,
                x=_clamp(centre_x - crop_w / 2, 0, source_w - crop_w),
                y=_clamp(eye_y - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
            )
        ],
        strategy=Strategy.WIDE,
    )


def _general_segment(
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> CropSegment:
    """No usable face — centre crop."""
    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[CropKeyframe(t=start_s, x=(source_w - crop_w) / 2, y=(source_h - crop_h) / 2)],
        strategy=Strategy.GENERAL,
        zoom=GENERAL_ZOOM,
    )


def _trimmed_spread(values: list[float]) -> float:
    """Range of the middle ``1 - 2*SPREAD_TRIM_FRACTION`` of ``values``.

    Max-minus-min is hostage to a single bad detection: one frame where the
    landmarker jumps reads as "the subject moved" and downgrades a lockable
    still shot to a tracked one. Trimming 10% off each end tolerates exactly
    that noise. Fewer than three samples carry no shape information; they
    report zero spread, which reads as lockable.
    """
    if len(values) < 3:
        return 0.0
    ordered = sorted(values)
    trim = int(SPREAD_TRIM_FRACTION * (len(ordered) - 1))
    return ordered[-1 - trim] - ordered[trim]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high)) if high > low else max(0.0, low)

"""Silence tightening — compress dead air instead of cutting it.

Retention on short-form platforms dies in the pauses: a speaker who stops to
think for two seconds reads as broken on a feed, but the pause itself is what
makes the sentence land. Cutting the silence outright produces audible jumps —
room tone vanishes, breaths clip, mouth positions teleport. Speeding the
silent spans up instead (say 4x) keeps every frame and every sample, just
briefly, which reads as brisk pacing rather than an edit.

The whole stage is a pure timeline transform: it maps a list of detected
silences and a clip range to an ordered list of :class:`Span` s — each a
source-time interval and a playback speed — that tile the clip exactly. The
export stage renders each span at its speed and concatenates; nothing else
about the pipeline changes. Word times and crop-path keyframe times are
remapped through the same plan so captions stay lip-synced and the camera
keeps tracking through the fast-forwarded stretches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .prepare import Silence

log = logging.getLogger(__name__)

#: Playback speed inside tightened silence. 4x compresses a 2s pause to 0.5s —
#: clearly faster, still visible motion, no perceptible cut.
DEFAULT_SILENCE_SPEED = 4.0

#: Silences shorter than this stay at 1x. Sub-half-second gaps are the natural
#: rhythm of speech; tightening them makes the speaker sound machine-gunned.
DEFAULT_MIN_SILENCE_S = 0.5

#: Breathing room left at normal speed on each side of a tightened gap, so the
#: transition into and out of fast-forward never sits flush against a word.
DEFAULT_KEEP_SILENCE_S = 0.12

#: The hook needs its run-up: no tightening this close to the clip start.
HOOK_PROTECTION_S = 0.75

#: Same for the outro beat after the last word.
OUTRO_PROTECTION_S = 0.3

#: If the plan would compress the clip below this fraction of its original
#: length, the silence data is probably wrong (bad VAD, music bed) — skip
#: tightening rather than ship a mangled clip.
MIN_OUTPUT_FRACTION = 0.6


class TightenError(RuntimeError):
    """The clip cannot be tightened safely; render it untightened instead."""


@dataclass(frozen=True)
class Span:
    """A source-time interval played back at ``speed``.

    Times are clip-relative seconds — the same timeline the crop path and the
    transcript words use.
    """

    start_s: float
    end_s: float
    speed: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def output_duration_s(self) -> float:
        return self.duration_s / self.speed


@dataclass(frozen=True)
class TightenPlan:
    """Spans tiling a clip exactly, plus the timeline transforms they define."""

    start_s: float
    end_s: float
    spans: tuple[Span, ...]

    @property
    def output_duration_s(self) -> float:
        return sum(span.output_duration_s for span in self.spans)

    @property
    def saved_s(self) -> float:
        return (self.end_s - self.start_s) - self.output_duration_s

    def output_span(self, span: Span) -> tuple[float, float]:
        """Start and end of ``span`` on the tightened output timeline."""
        elapsed = 0.0
        for candidate in self.spans:
            if candidate is span:
                return elapsed, elapsed + span.output_duration_s
            elapsed += candidate.output_duration_s
        raise ValueError("Span is not part of this plan.")

    def remap(self, t: float) -> float:
        """Source timeline -> tightened output timeline.

        Times inside a tightened span compress proportionally; times in
        untouched spans pass through with the accumulated offsets.
        """
        if t <= self.start_s:
            return t - self.start_s

        elapsed = 0.0
        for span in self.spans:
            if t < span.end_s or span is self.spans[-1]:
                clamped = min(t, span.end_s)
                return elapsed + (clamped - span.start_s) / span.speed
            elapsed += span.output_duration_s
        return elapsed

    def unmap(self, t: float) -> float:
        """Tightened output timeline -> source timeline (inverse of remap)."""
        elapsed = 0.0
        for span in self.spans:
            output = span.output_duration_s
            if t < elapsed + output or span is self.spans[-1]:
                return span.start_s + (t - elapsed) * span.speed
            elapsed += output
        return self.end_s


def build_plan(
    start_s: float,
    end_s: float,
    silences: list[Silence],
    *,
    speed: float = DEFAULT_SILENCE_SPEED,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    keep_silence_s: float = DEFAULT_KEEP_SILENCE_S,
) -> TightenPlan:
    """Build the tighten plan for one clip.

    ``silences`` are in the same absolute source timeline as ``start_s`` /
    ``end_s`` (which is what :meth:`Silence` carries); the returned spans are
    clip-relative, matching the crop path and word timelines.

    Preconditions:
        start_s < end_s; silences may be unsorted and may extend past the
        clip range — anything outside is ignored.
    """
    if speed <= 1.0:
        raise ValueError("speed must be greater than 1.0")

    duration = end_s - start_s
    if duration <= 0:
        raise ValueError("clip range is empty")

    # Shrink each silence by the keep margins before deciding whether it is
    # worth tightening: a 0.5s pause with 0.12s kept on each side has only
    # 0.26s of compressible air, which is not worth a speed change.
    tightened: list[tuple[float, float]] = []
    for silence in silences:
        inner_start = max(silence.start + keep_silence_s, start_s)
        inner_end = min(silence.end - keep_silence_s, end_s)
        if inner_end - inner_start < min_silence_s:
            continue
        tightened.append((inner_start, inner_end))

    # Merge overlaps so one long pause split into two VAD events becomes one
    # span, and sort so the tiling below is ordered.
    tightened.sort()
    merged: list[tuple[float, float]] = []
    for start, end in tightened:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    spans: list[Span] = []
    cursor = start_s
    for silence_start, silence_end in merged:
        # Protect the hook: gaps starting before the protection window ends
        # are only tightened from where the protection runs out.
        silence_start = max(silence_start, start_s + HOOK_PROTECTION_S)
        if silence_end - silence_start < min_silence_s:
            continue

        if silence_start > cursor:
            spans.append(Span(cursor, silence_start, 1.0))
        # The tail margin of the gap plays at normal speed and is part of the
        # next speech span, so the transition out lands on kept air.
        spans.append(Span(silence_start, silence_end, speed))
        cursor = silence_end

    spans.append(Span(cursor, end_s, 1.0))

    plan = TightenPlan(start_s, end_s, tuple(spans))
    if plan.output_duration_s < duration * MIN_OUTPUT_FRACTION:
        raise TightenError(
            f"Tightening would compress the clip to "
            f"{plan.output_duration_s:.1f}s of {duration:.1f}s; the silence "
            "data looks wrong, so the clip is rendered untightened."
        )
    return plan


def atempo_chain(speed: float) -> str:
    """The ``atempo`` filter chain for ``speed``.

    ffmpeg's atempo accepts 0.5-2.0 per instance; anything faster chains
    instances. 4.0 becomes ``atempo=2.0,atempo=2.0``.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    remaining = speed
    factors: list[float] = []
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(round(remaining, 6))
    return ",".join(f"atempo={factor}" for factor in factors)

"""Crop-path smoothing.

Raw per-frame face positions are far too noisy to drive a crop directly —
detection jitter of a few pixels reads as a shaking camera. Two styles of
controller live here:

**``smooth_series``** — the old re-centering chain, in three stages:

1. **One Euro Filter** — adaptive low-pass. At low speed it filters hard (kills
   jitter); at high speed it filters lightly (keeps up with a real pan). A plain
   EMA has to choose one or the other, so it either shakes or lags.
2. **Dead zone** — below a movement threshold, don't move at all. A locked frame
   reads as intentional; a frame that micro-corrects reads as broken.
3. **Velocity clamp** — cap how fast the crop can travel, so a detection glitch
   can never whip the frame across the shot.

**``lazy_follow``** — a hysteresis dead-band follow that replaces it for
speaker tracking. Re-centering on the person is what makes a camera feel glued
to them: the crop chases every drift, so the frame never rests. ``lazy_follow``
holds the camera parked while the subject stays inside a margin, chases with a
decelerating ease once they pass it, and parks again the instant they return to
the near-centre band — the same bank-and-hold rhythm an operator uses.

Reference: Casiez, Roussel & Vogel, "1€ Filter" (CHI 2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Lower cutoff filters harder at rest. Tuned for face tracking at 5-10 Hz
#: sampling, where detection noise dominates below ~1 Hz.
DEFAULT_MIN_CUTOFF = 0.6
#: How aggressively the cutoff opens up with speed. Higher means less lag on
#: fast movement, at the cost of letting more jitter through.
DEFAULT_BETA = 0.02
DEFAULT_DERIVATIVE_CUTOFF = 1.0


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class LowPass:
    """First-order low-pass filter with a settable per-sample alpha."""

    def __init__(self) -> None:
        self.value: float | None = None

    def __call__(self, sample: float, alpha: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """Adaptive low-pass filter trading jitter against lag by signal speed."""

    def __init__(
        self,
        *,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        derivative_cutoff: float = DEFAULT_DERIVATIVE_CUTOFF,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._value = LowPass()
        self._derivative = LowPass()
        self._last_sample: float | None = None
        self._last_time: float | None = None

    def __call__(self, sample: float, timestamp: float) -> float:
        if self._last_time is None or timestamp <= self._last_time:
            self._last_time = timestamp
            self._last_sample = sample
            return self._value(sample, 1.0)

        dt = timestamp - self._last_time
        derivative = (sample - (self._last_sample or sample)) / dt
        smoothed_derivative = self._derivative(derivative, _alpha(self.derivative_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        result = self._value(sample, _alpha(cutoff, dt))

        self._last_time = timestamp
        self._last_sample = sample
        return result


@dataclass
class SmoothingConfig:
    min_cutoff: float = DEFAULT_MIN_CUTOFF
    beta: float = DEFAULT_BETA
    #: Movement smaller than this (pixels) is ignored entirely.
    dead_zone_px: float = 26.0
    #: Ceiling on crop travel, in pixels per second.
    max_velocity_px_s: float = 220.0
    #: Lazy-follow trigger. The camera stays parked until the subject's ideal
    #: framing drifts further than this *fraction of the crop width* from the
    #: camera centre. Pacing/waving stays inside the band, so the frame rests.
    #: Raised to ~0.28 so small head tilts / shoulder turns stay inside the
    #: parked band — the camera only wakes for a genuinely big move.
    follow_margin_ratio: float = 0.28
    #: Lazy-follow hysteresis. Once chasing, the camera stops again the moment
    #: the subject is back inside this *fraction of the crop width* of centre.
    #: Narrower than ``follow_margin_ratio`` so the trigger never oscillates at
    #: the boundary.
    hold_margin_ratio: float = 0.06
    #: Time constant of the chase's ease-in/out (seconds). Slower reads calmer;
    #: this is what makes the camera lag the walker instead of gluing to them.
    #: Kept slow (~0.60s) so a triggered chase reads as a calm operator pan,
    #: not a snap. ``max_velocity_px_s`` is the hard ceiling on top of it.
    follow_tau_s: float = 0.60


def smooth_series(
    samples: list[tuple[float, float]], config: SmoothingConfig | None = None
) -> list[tuple[float, float]]:
    """Smooth ``(timestamp, value)`` samples through the full three-stage chain.

    Preconditions:
        samples are sorted by timestamp.
    """
    config = config or SmoothingConfig()
    if len(samples) <= 1:
        return list(samples)

    one_euro = OneEuroFilter(min_cutoff=config.min_cutoff, beta=config.beta)

    output: list[tuple[float, float]] = []
    held: float | None = None
    previous_time: float | None = None

    for timestamp, raw in samples:
        filtered = one_euro(raw, timestamp)

        if held is None:
            held = filtered
        else:
            # Dead zone: hold position until the target has moved meaningfully.
            if abs(filtered - held) >= config.dead_zone_px:
                target = filtered
                if previous_time is not None:
                    dt = max(1e-6, timestamp - previous_time)
                    max_step = config.max_velocity_px_s * dt
                    delta = target - held
                    if abs(delta) > max_step:
                        target = held + math.copysign(max_step, delta)
                held = target

        output.append((timestamp, held))
        previous_time = timestamp

    return output


def lazy_follow(
    samples: list[tuple[float, float]],
    config: SmoothingConfig | None = None,
    *,
    reference_px: float,
) -> list[tuple[float, float]]:
    """Hysteresis dead-band follow: move less, and when you move, don't chase.

    Drives a camera axis from ``(timestamp, target)`` samples where ``target``
    is where the camera *would* sit to centre the subject perfectly. Instead of
    re-centring on every drift, the camera:

    1. starts at the first target and stays parked while the subject's ideal
       framing is inside the trigger band (``follow_margin_ratio`` of
       ``reference_px``);
    2. once the subject leaves that band, chases with an exponential ease
       (``follow_tau_s``) that decelerates as it closes in, capped by
       ``max_velocity_px_s``;
    3. stops dead the instant the subject is back inside the narrower hold band
       (``hold_margin_ratio``), and stays parked there no matter how much they
       wander inside it — the wider trigger band means it does not wake up and
       chase again until they genuinely leave.

    The hysteresis between the trigger and hold bands keeps a subject pacing
    on the boundary from toggling the camera on and off.

    Preconditions:
        samples are sorted by timestamp, ``reference_px`` is the tight crop
        dimension (pixels) the margins scale against.
    """
    config = config or SmoothingConfig()

    follow_margin = max(config.follow_margin_ratio * reference_px, config.dead_zone_px)
    hold_margin = max(config.hold_margin_ratio * reference_px, 1.0)
    if hold_margin >= follow_margin:
        hold_margin = follow_margin * 0.5

    if len(samples) <= 1:
        return list(samples)

    output: list[tuple[float, float]] = []
    camera: float | None = None
    chasing = False
    previous_time: float | None = None

    for timestamp, target in samples:
        if camera is None:
            camera = float(target)
        else:
            dt = max(1e-6, timestamp - previous_time) if previous_time is not None else 1e-6
            error = target - camera

            if chasing:
                # We stop as soon as the subject is back in the safe inner
                # band — the camera parks where it is instead of re-centring.
                if abs(error) < hold_margin:
                    chasing = False
            elif abs(error) > follow_margin:
                chasing = True

            if chasing:
                # Exponential ease: large steps at the far edge of the chase,
                # gentle creep as we close in. This is the "don't chase" feel.
                alpha = 1.0 - math.exp(-dt / config.follow_tau_s)
                step = error * alpha
                max_step = config.max_velocity_px_s * dt
                if abs(step) > max_step:
                    step = math.copysign(max_step, step)
                camera += step

        output.append((timestamp, camera))
        previous_time = timestamp

    return output

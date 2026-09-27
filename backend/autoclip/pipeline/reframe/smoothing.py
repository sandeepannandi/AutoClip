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
decelerating ease once they pass it, and returns to the near-centre band — the
same bank-and-hold rhythm an operator uses. Beyond a large error threshold the
chase switches to a *burst*: a shorter ease and a much higher velocity ceiling
that recovers a subject who jumped or lunged out of frame in a few
decelerating steps, reading as the camera jumping with them rather than
snapping after. The vertical axis runs the same controller with a tighter wake
band — a cut-off forehead is worse than an off-centre body.

The park is not a freeze. While parked the camera *settles*: it eases toward
the subject at a rate several times slower than a chase (tau 3.0s vs 0.6s). A
couple of pixels per frame is invisible as motion but, over a second or two,
puts the subject back on the centre line — the frame the viewer actually sees
is composed, while the micro-movement that gets it there never registers. This
is what keeps a subject who drifts and stops from standing near the frame edge
for the rest of the shot.

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
    #: camera centre. Small head tilts / shoulder turns stay inside the band,
    #: so the frame rests, but ordinary repositioning (~15% of the frame) now
    #: wakes it instead of leaving the subject parked visibly off-centre for
    #: the rest of the shot.
    follow_margin_ratio: float = 0.14
    #: Lazy-follow hysteresis. Once chasing, the camera stops again the moment
    #: the subject is back inside this *fraction of the crop width* of centre.
    #: Narrower than ``follow_margin_ratio`` so the trigger never oscillates at
    #: the boundary.
    hold_margin_ratio: float = 0.04
    #: Vertical wake band, as a fraction of the crop *height*. Tighter than the
    #: horizontal band because a cut-off forehead reads as broken while an
    #: off-centre body merely reads as composed differently. The vertical axis
    #: gets its own band but shares the hold band, settle, and burst settings.
    follow_margin_ratio_y: float = 0.10
    #: Burst chase. When the subject's ideal framing is further than this
    #: fraction of the reference dimension from the camera — a jump, a lean out
    #: of frame, a detection dropout and reappearance — the chase switches from
    #: the calm operator pan to a fast recovery: shorter tau and a much higher
    #: velocity ceiling. The ease is still exponential, so the recovery
    #: decelerates as it closes; it reads as the camera jumping *with* the
    #: subject, not snapping after them.
    burst_error_ratio: float = 0.30
    #: Burst exit hysteresis, as a fraction of the burst entry error. The burst
    #: disengages once the remaining error falls below ``burst_exit_ratio`` of
    #: the entry threshold, so a subject near the boundary doesn't toggle rates.
    burst_exit_ratio: float = 0.6
    #: Time constant of the burst chase's ease (seconds). Faster than the calm
    #: chase, still an ease — never a per-frame teleport.
    burst_tau_s: float = 0.35
    #: Velocity ceiling during a burst, in pixels per second. The calm ceiling
    #: would leave the subject out of frame for over a second after a big jump;
    #: this closes the gap in a few decelerating steps instead.
    #: Velocity ceiling during a burst, in pixels per second. The calm ceiling
    #: would leave the subject out of frame for over a second after a big jump;
    #: this closes the gap in a few decelerating steps instead.
    burst_velocity_px_s: float = 700.0
    #: Time constant of the chase's ease-in/out (seconds). Slower reads calmer;
    #: this is what makes the camera lag the walker instead of gluing to them.
    #: Kept slow (~0.60s) so a triggered chase reads as a calm operator pan,
    #: not a snap. ``max_velocity_px_s`` is the hard ceiling on top of it.
    follow_tau_s: float = 0.60
    #: Time constant of the settle (seconds). While parked inside the band the
    #: camera keeps easing toward the subject at this much slower rate, so the
    #: resting frame ends centred without the settle registering as motion —
    #: a 40px offset closes at ~13 px/s initially and decays exponentially,
    #: which still reads as settling rather than a camera move. Tightened from
    #: 8.0 (τ=8 left a drifted subject near the frame edge for ~10s); 3.0
    #: re-centres within a couple of seconds while keeping every per-sample
    #: step under ~3px on a phone-width crop. Off-centre parking is the
    #: complaint this fixes: a subject who drifts and stops must not be framed
    #: off-centre for the rest of the shot.
    settle_tau_s: float = 3.0
    #: Fraction of ``reference_px`` below which the settle is suspended. At rest
    #: the dead zone absorbs detection noise anyway; this is a floor for
    #: reference sizes so small the settle step would be indivisible.
    settle_dead_zone_ratio: float = 0.01


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
    vertical: bool = False,
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
    3. leaves the chase the instant the subject is back inside the narrower
       hold band (``hold_margin_ratio``), and will not wake into a full chase
       again until they genuinely leave the wide trigger band.

    While parked — including after a chase ends — the camera *settles*: it
    eases toward the subject at the much slower ``settle_tau_s`` rate, so a
    subject who drifts and stops is re-centred over the following second or
    two instead of being framed off-centre for the rest of the shot. The settle
    is deliberately ~2-4 px/s on a phone-width crop: invisible as motion, but
    it is what puts the resting frame on the centre line. Movement below
    ``settle_dead_zone_ratio`` of ``reference_px`` is not corrected at all —
    that margin is where detection noise lives.

    The hysteresis between the trigger and hold bands still keeps a subject
    pacing on the boundary from toggling the camera between chase and rest.

    Beyond ``burst_error_ratio`` of ``reference_px`` the chase switches to a
    fast recovery (``burst_tau_s`` ease, ``burst_velocity_px_s`` ceiling) —
    what a subject jumping or lunging out of frame needs — and eases back to
    the calm chase as the gap closes, exiting the burst at
    ``burst_exit_ratio`` of the entry threshold.

    Preconditions:
        samples are sorted by timestamp, ``reference_px`` is the crop dimension
        (pixels) the margins scale against — width for x, height for y.
    """
    config = config or SmoothingConfig()

    wake_ratio = config.follow_margin_ratio_y if vertical else config.follow_margin_ratio
    follow_margin = max(wake_ratio * reference_px, config.dead_zone_px)
    hold_margin = max(config.hold_margin_ratio * reference_px, 1.0)
    if hold_margin >= follow_margin:
        hold_margin = follow_margin * 0.5
    settle_dead_zone = max(config.settle_dead_zone_ratio * reference_px, 1.0)

    if len(samples) <= 1:
        return list(samples)

    output: list[tuple[float, float]] = []
    camera: float | None = None
    chasing = False
    burst_active = False
    previous_time: float | None = None

    for timestamp, target in samples:
        if camera is None:
            camera = float(target)
        else:
            dt = max(1e-6, timestamp - previous_time) if previous_time is not None else 1e-6
            error = target - camera

            if chasing:
                # The chase ends inside the safe inner band; the settle takes
                # over from there and finishes the centring invisibly.
                if abs(error) < hold_margin:
                    chasing = False
            elif abs(error) > follow_margin:
                chasing = True

            if chasing:
                # Burst selection: a huge error means the subject jumped or
                # lunged out of frame — recover fast, then hand back to the
                # calm chase well before closing, so the final approach keeps
                # the operator-pan feel. Entry/exit hysteresis prevents rate
                # flapping when the subject sits near the boundary.
                burst_threshold = config.burst_error_ratio * reference_px
                burst_exit_limit = config.burst_exit_ratio * burst_threshold
                if burst_active:
                    burst_active = abs(error) >= burst_exit_limit
                else:
                    burst_active = abs(error) >= burst_threshold

                tau = config.burst_tau_s if burst_active else config.follow_tau_s
                ceiling = config.burst_velocity_px_s if burst_active else config.max_velocity_px_s

                # Exponential ease: large steps at the far edge of the chase,
                # gentle creep as we close in. This is the "don't chase" feel.
                alpha = 1.0 - math.exp(-dt / tau)
                step = error * alpha
                max_step = ceiling * dt
                if abs(step) > max_step:
                    step = math.copysign(max_step, step)
                camera += step
            elif abs(error) > settle_dead_zone:
                # Settle: the same ease at a far slower time constant. A few
                # pixels per frame reads as stillness but re-centres within a
                # second or two, so the resting frame ends composed.
                alpha = 1.0 - math.exp(-dt / config.settle_tau_s)
                step = error * alpha
                camera += step

        output.append((timestamp, camera))
        previous_time = timestamp

    return output

"""Crop path construction, smoothing, and tracking.

These are the acceptance-bar mechanics from PRD 6.4 expressed as unit tests:
jitter suppression, no interpolation across cuts, and correct crop geometry.
"""

from __future__ import annotations

import math

import pytest
from autoclip.pipeline.reframe import croppath, smoothing
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropSegment,
    Strategy,
    axis_expression,
    centre_crop,
    decimate,
    segment_crop_filter,
    target_crop_size,
)
from autoclip.pipeline.reframe.faces import FaceObservation
from autoclip.pipeline.reframe.tracker import build_tracks


class TestTargetCropSize:
    def test_landscape_to_vertical(self) -> None:
        # 1080 * 9/16 = 607.5, rounded to the nearest even width.
        assert target_crop_size(1920, 1080, 9, 16) == (608, 1080)

    def test_square_output(self) -> None:
        assert target_crop_size(1920, 1080, 1, 1) == (1080, 1080)

    def test_landscape_source_to_landscape_output_is_full_frame(self) -> None:
        assert target_crop_size(1920, 1080, 16, 9) == (1920, 1080)

    def test_vertical_source_to_vertical_output_is_full_frame(self) -> None:
        assert target_crop_size(1080, 1920, 9, 16) == (1080, 1920)

    @pytest.mark.parametrize(
        ("sw", "sh"), [(1920, 1080), (1280, 720), (3840, 2160), (1440, 1080), (1920, 800)]
    )
    def test_dimensions_are_always_even(self, sw: int, sh: int) -> None:
        # Odd dimensions fail an h264 yuv420p encode outright.
        width, height = target_crop_size(sw, sh, 9, 16)

        assert width % 2 == 0
        assert height % 2 == 0

    def test_crop_never_exceeds_the_source(self) -> None:
        width, height = target_crop_size(1920, 1080, 9, 16)

        assert width <= 1920
        assert height <= 1080


class TestCentreCrop:
    def test_produces_one_static_segment(self) -> None:
        path = centre_crop(1920, 1080, 30.0)

        assert len(path.segments) == 1
        assert path.segments[0].is_static
        assert path.segments[0].strategy is Strategy.GENERAL

    def test_is_horizontally_centred(self) -> None:
        path = centre_crop(1920, 1080, 30.0)
        segment = path.segments[0]

        assert segment.keyframes[0].x == pytest.approx((1920 - segment.width) / 2)

    def test_spans_the_full_duration(self) -> None:
        path = centre_crop(1920, 1080, 42.5)

        assert path.segments[0].end_s == 42.5
        assert path.duration_s == 42.5


class TestOneEuroFilter:
    def test_suppresses_stationary_jitter(self) -> None:
        # A still subject with +/-5px detection noise must not move the crop.
        samples = [(i * 0.2, 500 + (5 if i % 2 else -5)) for i in range(40)]

        smoothed = smoothing.smooth_series(samples, smoothing.SmoothingConfig())
        values = [v for _, v in smoothed[5:]]

        assert max(values) - min(values) < 3.0

    def test_follows_a_genuine_pan(self) -> None:
        # A real 400px move over 8s must actually be followed.
        samples = [(i * 0.2, 300 + i * 10) for i in range(40)]

        smoothed = smoothing.smooth_series(samples, smoothing.SmoothingConfig())

        assert smoothed[-1][1] > 600

    def test_dead_zone_holds_small_movements(self) -> None:
        config = smoothing.SmoothingConfig(dead_zone_px=20.0)
        samples = [(i * 0.2, 500 + i * 0.5) for i in range(10)]

        smoothed = smoothing.smooth_series(samples, config)

        assert smoothed[0][1] == pytest.approx(smoothed[-1][1], abs=1.0)

    def test_velocity_is_clamped(self) -> None:
        # A detection glitch must never whip the frame across the shot.
        config = smoothing.SmoothingConfig(max_velocity_px_s=100.0, dead_zone_px=1.0)
        samples = [(0.0, 0.0), (0.2, 0.0), (0.4, 5000.0), (0.6, 5000.0)]

        smoothed = smoothing.smooth_series(samples, config)

        for (t0, v0), (t1, v1) in zip(smoothed, smoothed[1:], strict=False):
            assert abs(v1 - v0) <= 100.0 * (t1 - t0) + 1e-6

    def test_single_sample_passes_through(self) -> None:
        assert smoothing.smooth_series([(0.0, 100.0)]) == [(0.0, 100.0)]

    def test_empty_input(self) -> None:
        assert smoothing.smooth_series([]) == []

    def test_filter_is_deterministic(self) -> None:
        samples = [(i * 0.2, 300 + math.sin(i) * 50) for i in range(30)]

        first = smoothing.smooth_series(samples)
        second = smoothing.smooth_series(samples)

        assert first == second


class TestLazyFollow:
    CONFIG = smoothing.SmoothingConfig()
    # With reference_px=600: follow_margin = 0.28*600 = 168px; hold = 0.06*600 = 36px.
    REFERENCE = 600.0

    def test_parks_while_the_subject_stays_inside_the_band(self) -> None:
        # Wandering, waving, weight-shifting — none of it leaves the 168px
        # band, so the camera never wakes into a chase. The settle creeps
        # toward the wander's midline at sub-pixel-per-step speed: over the
        # whole series it drifts a few dozen pixels, never visibly.
        samples = [(i * 0.2, 500 + (50 if i % 2 else -50)) for i in range(40)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        assert followed[0][1] == pytest.approx(450.0)  # starts on the first target
        assert max(v for _, v in followed) - min(v for _, v in followed) < 40.0
        # Settle steps stay invisible: a fraction of a percent of the ceiling.
        assert max(abs(b - a) for (_, a), (_, b) in zip(followed, followed[1:], strict=False)) < 3.0

    def test_follows_a_genuine_drift_past_the_band(self) -> None:
        # A fast sustained departure wakes the camera after the first band
        # crossing; the ramp to 300 + 40*29 = 1460 is real tracking.
        samples = [(i * 0.2, 300 + i * 40) for i in range(30)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        assert followed[-1][1] > 800

    def test_hysteresis_stops_it_waking_again_on_small_movement(self) -> None:
        # Once the subject returns inside the narrow hold band the camera stops
        # chasing; the settle then creeps the last ~30px out over seconds. The
        # wobble must not re-arm the chase: per-step movement stays far under
        # the velocity ceiling, and the residual offset shrinks monotonically.
        samples = [
            (0.0, 500.0),
            (0.2, 700.0),  # far out: chase
            (0.4, 550.0),  # back inside the hold band: chase ends
            (0.6, 560.0),  # small wobble inside the band
            (0.8, 575.0),
            (1.0, 545.0),
        ]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)
        values = [v for _, v in followed]

        assert followed[1][1] == pytest.approx(544.0, abs=1.0)
        steps = [abs(b - a) for (_, a), (_, b) in zip(followed[2:], followed[3:], strict=False)]
        assert max(steps) < 10.0
        assert values[-1] > values[2]  # settling toward the subject, not away

    def test_parks_through_an_8_percent_drift(self) -> None:
        # 8% of the 600px crop width is 48px — well inside the 168px follow
        # band. The camera starts on the subject, who sways +-48 around 520 (a
        # head tilt whose midline sits 20px off). The chase must never wake;
        # per-sample movement stays at settle scale (~1.2px/step measured).
        samples = [(i * 0.2, 520 + 48 * math.sin(i * 0.5)) for i in range(40)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        assert followed[0][1] == pytest.approx(520.0)
        assert max(v for _, v in followed) - min(v for _, v in followed) < 10.0
        assert max(abs(b - a) for (_, a), (_, b) in zip(followed, followed[1:], strict=False)) < 5.0

    def test_settle_recentres_a_drifted_subject(self) -> None:
        # THE off-centre regression: the camera starts centred on the subject
        # at 500; the subject steps to 590 (+90px, inside the 168px follow band
        # — no chase) and holds. The old controller froze them off-centre
        # forever; the settle must close most of that gap invisibly.
        samples = [(i * 0.2, 500.0) for i in range(3)] + [(i * 0.2, 590.0) for i in range(3, 53)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)
        values = [v for _, v in followed]

        assert values[2] == pytest.approx(500.0)  # still centred before the drift
        assert values[-1] == pytest.approx(590.0, abs=30.0)  # >2/3 re-centred
        # And the correction is invisible: far under the velocity ceiling.
        assert max(abs(b - a) for (_, a), (_, b) in zip(followed, followed[1:], strict=False)) < 3.0

    def test_settle_is_disabled_below_its_dead_zone(self) -> None:
        # Sub-settle-dead-zone offsets (1% of 600 = 6px) are detection-noise
        # territory and must not move the camera at all.
        samples = [(i * 0.2, 504.0) for i in range(30)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        assert all(v == pytest.approx(504.0) for _, v in followed)

    def test_chases_and_settles_inside_the_hold_band_after_a_35_percent_crossing(
        self,
    ) -> None:
        # A crossing of 35% of the crop width (210px) wakes the camera, which
        # chases slowly (velocity-clamped ease), stops inside the 36px hold
        # band, and the settle then closes the remaining distance invisibly.
        samples = [(i * 0.2, 500.0) for i in range(3)] + [(i * 0.2, 710.0) for i in range(3, 33)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)
        values = [v for _, v in followed]

        assert all(v == pytest.approx(500.0) for v in values[:3])
        assert max(values) > 550.0  # it actually chased instead of giving up
        # The settle finishes the centring the chase parked short of.
        assert abs(values[-1] - 710.0) <= 20.0
        assert abs(values[-1] - values[-2]) < 2.0  # and it is calm

    def test_boundary_pacing_never_oscillates_the_camera(self) -> None:
        # Drifting in/out at +-120px around centre stays inside the 168px
        # band: the camera must never chase (per-sample movement stays at
        # settle scale, ~3px/step measured — an order under the ceiling) and
        # stays near the pacing's midline rather than oscillating with it.
        samples = [(0.0, 500.0)] + [(i * 0.2, 500 + (120 if i % 2 else -120)) for i in range(1, 60)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        assert max(v for _, v in followed) - min(v for _, v in followed) < 10.0
        steps = [abs(b - a) for (_, a), (_, b) in zip(followed, followed[1:], strict=False)]
        assert max(steps) < 10.0

    def test_chase_velocity_is_clamped(self) -> None:
        # A detection glitch must never whip the frame across the shot.
        samples = [(0.0, 0.0), (0.2, 5000.0), (0.4, 5000.0), (0.6, 5000.0)]

        followed = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        for (t0, v0), (t1, v1) in zip(followed, followed[1:], strict=False):
            assert abs(v1 - v0) <= self.CONFIG.max_velocity_px_s * (t1 - t0) + 1e-6
        assert followed[-1][1] < 1000

    def test_single_sample_passes_through(self) -> None:
        assert smoothing.lazy_follow([(0.0, 100.0)], reference_px=600.0) == [(0.0, 100.0)]

    def test_empty_input(self) -> None:
        assert smoothing.lazy_follow([], reference_px=600.0) == []

    def test_filter_is_deterministic(self) -> None:
        samples = [(i * 0.2, 300 + math.sin(i) * 50) for i in range(30)]

        first = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)
        second = smoothing.lazy_follow(samples, self.CONFIG, reference_px=self.REFERENCE)

        assert first == second


class TestAxisExpression:
    def test_single_keyframe_is_a_constant(self) -> None:
        assert axis_expression([CropKeyframe(0.0, 100.0, 50.0)], "x") == "100.0"

    def test_two_keyframes_produce_a_ramp(self) -> None:
        frames = [CropKeyframe(0.0, 0.0, 0.0), CropKeyframe(2.0, 200.0, 0.0)]

        expression = axis_expression(frames, "x")

        assert "if(lt(t,2.0)" in expression
        assert "100.0" in expression  # slope: 200px over 2s

    def test_offset_rebases_times(self) -> None:
        frames = [CropKeyframe(10.0, 0.0, 0.0), CropKeyframe(12.0, 200.0, 0.0)]

        expression = axis_expression(frames, "x", offset_s=10.0)

        assert "t-0.0" in expression
        assert "lt(t,2.0)" in expression

    def test_expression_evaluates_correctly_at_keyframes(self) -> None:
        frames = [
            CropKeyframe(0.0, 0.0, 0.0),
            CropKeyframe(1.0, 100.0, 0.0),
            CropKeyframe(2.0, 50.0, 0.0),
        ]

        expression = axis_expression(frames, "x")

        assert _evaluate(expression, 0.0) == pytest.approx(0.0, abs=0.1)
        assert _evaluate(expression, 0.5) == pytest.approx(50.0, abs=0.1)
        assert _evaluate(expression, 1.5) == pytest.approx(75.0, abs=0.1)

    def test_empty_keyframes_give_zero(self) -> None:
        assert axis_expression([], "x") == "0"


class TestDecimate:
    def test_collinear_keyframes_are_removed(self) -> None:
        frames = [CropKeyframe(float(i), float(i * 10), 0.0) for i in range(20)]

        assert len(decimate(frames)) == 2

    def test_direction_changes_are_kept(self) -> None:
        frames = [
            CropKeyframe(0.0, 0.0, 0.0),
            CropKeyframe(1.0, 100.0, 0.0),
            CropKeyframe(2.0, 0.0, 0.0),
        ]

        assert len(decimate(frames)) == 3

    def test_respects_the_limit(self) -> None:
        frames = [CropKeyframe(float(i), float(i % 2) * 100, 0.0) for i in range(200)]

        result = decimate(frames, limit=10)

        assert len(result) <= 10

    def test_endpoints_survive(self) -> None:
        frames = [CropKeyframe(float(i), float(i % 3) * 60, 0.0) for i in range(100)]

        result = decimate(frames, limit=8)

        assert result[0].t == 0.0
        assert result[-1].t == 99.0

    def test_times_stay_strictly_increasing(self) -> None:
        frames = [CropKeyframe(float(i), float(i % 5) * 40, 0.0) for i in range(150)]

        result = decimate(frames, limit=12)

        assert all(b.t > a.t for a, b in zip(result, result[1:], strict=False))


class TestSegmentFilter:
    def test_static_segment_uses_constants(self) -> None:
        segment = CropSegment(
            start_s=0.0,
            end_s=5.0,
            width=606,
            height=1080,
            keyframes=[CropKeyframe(0.0, 657.0, 0.0)],
        )

        assert segment_crop_filter(segment) == "crop=w=606:h=1080:x='657.0':y='0.0'"

    def test_moving_segment_uses_expressions(self) -> None:
        segment = CropSegment(
            start_s=0.0,
            end_s=4.0,
            width=606,
            height=1080,
            keyframes=[
                CropKeyframe(0.0, 100.0, 0.0),
                CropKeyframe(2.0, 300.0, 0.0),
                CropKeyframe(4.0, 200.0, 0.0),
            ],
        )

        result = segment_crop_filter(segment)

        # Commas are backslash-escaped for the filtergraph parser, which strips
        # them before the expression parser sees the string.
        assert "if(lt(t\\," in result

    def test_commas_are_escaped_for_the_filtergraph(self) -> None:
        segment = CropSegment(
            start_s=0.0,
            end_s=4.0,
            width=606,
            height=1080,
            keyframes=[
                CropKeyframe(0.0, 100.0, 0.0),
                CropKeyframe(2.0, 300.0, 0.0),
                CropKeyframe(4.0, 200.0, 0.0),
            ],
        )

        result = segment_crop_filter(segment)

        # An unescaped comma would terminate the filter and break the graph.
        assert ",'" not in result.replace("\\,", "")

    def test_near_static_path_collapses_to_a_constant(self) -> None:
        # Sub-pixel drift is not movement.
        segment = CropSegment(
            start_s=0.0,
            end_s=4.0,
            width=606,
            height=1080,
            keyframes=[CropKeyframe(float(i), 100.0 + i * 0.1, 0.0) for i in range(5)],
        )

        assert segment.is_static
        assert "if(" not in segment_crop_filter(segment)


class TestSerialisation:
    def test_crop_path_round_trips(self, tmp_path) -> None:
        original = croppath.CropPath(
            source_width=1920,
            source_height=1080,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=606,
                    height=1080,
                    keyframes=[CropKeyframe(0.0, 100.0, 0.0), CropKeyframe(5.0, 200.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
                CropSegment(
                    start_s=5.0,
                    end_s=9.0,
                    width=1920,
                    height=1080,
                    keyframes=[CropKeyframe(5.0, 0.0, 0.0)],
                    strategy=Strategy.WIDE,
                    fit=True,
                ),
            ],
        )

        restored = croppath.CropPath.from_dict(original.to_dict())

        assert restored.source_width == 1920
        assert len(restored.segments) == 2
        assert restored.segments[0].strategy is Strategy.TRACK
        assert restored.segments[1].fit is True

    def test_saves_and_loads_from_disk(self, tmp_path) -> None:
        path = centre_crop(1920, 1080, 12.0)
        target = path.save(tmp_path / "crop.json")

        assert croppath.CropPath.load(target).duration_s == 12.0


class TestTrackSegment:
    """Framing decisions for a single subject: when to lock, where to lock."""

    CROP_W, CROP_H, SOURCE_W, SOURCE_H = 608, 1080, 1920, 1080

    def _observation(self, t: float, cx: float, cy: float = 400.0) -> FaceObservation:
        return FaceObservation(t=t, cx=cx, cy=cy, width=200, height=260, eye_y=cy - 40, mar=0.05)

    def _segment(self, observations: list[FaceObservation]) -> CropSegment:
        from autoclip.pipeline.reframe import ReframeConfig, _track_segment
        from autoclip.pipeline.reframe.tracker import FaceTrack

        track = FaceTrack(id=0, observations=observations)
        return _track_segment(
            track,
            start_s=observations[0].t,
            end_s=observations[-1].t + 0.2,
            source_w=self.SOURCE_W,
            source_h=self.SOURCE_H,
            crop_w=self.CROP_W,
            crop_h=self.CROP_H,
            config=ReframeConfig(),
        )

    def test_static_subject_locks_on_the_median(self) -> None:
        # A still subject (plus one 80px detection outlier) locks. The lock
        # must sit on the median — the outlier must not drag it off-centre the
        # way a mean would.
        observations = [self._observation(i * 0.2, 800.0) for i in range(10)]
        observations.append(self._observation(2.0, 880.0))

        segment = self._segment(observations)

        assert segment.strategy is Strategy.TRACK
        assert len(segment.keyframes) == 1
        expected_x = 800.0 - self.CROP_W / 2
        assert segment.keyframes[0].x == pytest.approx(expected_x, abs=1.0)

    def test_close_up_is_tracked_not_mean_locked(self) -> None:
        # THE off-centre regression: a close-up (400px face on a 608px crop,
        # past the old 0.55 width lock) whose subject steps 60px mid-shot used
        # to be frozen at the drift's MEAN — off the subject at both ends. Now
        # it is tracked: the path starts on the subject and follows the drift.
        observations = [
            self._observation(i * 0.2, 900.0 + (60.0 if i >= 10 else 0.0)) for i in range(20)
        ]
        observations = [
            FaceObservation(
                t=o.t, cx=o.cx, cy=o.cy, width=400, height=500, eye_y=o.eye_y, mar=o.mar
            )
            for o in observations
        ]

        segment = self._segment(observations)

        assert segment.strategy is Strategy.TRACK
        # Not a single frozen keyframe: the subject is followed, not averaged.
        assert len(segment.keyframes) > 1
        # Starts centred on where the subject actually was (900 - 304 = 596),
        # not on the drift's mean (930 - 304 = 626) as the old mean-lock did.
        assert segment.keyframes[0].x == pytest.approx(596.0, abs=1.0)
        # And the path moves in the drift's direction.
        x_values = [k.x for k in segment.keyframes]
        assert x_values[-1] > x_values[0] + 5.0

    def test_moving_subject_is_tracked(self) -> None:
        observations = [self._observation(i * 0.2, 700.0 + i * 15.0) for i in range(20)]

        segment = self._segment(observations)

        assert segment.strategy is Strategy.TRACK
        assert len(segment.keyframes) > 1

    def test_lock_and_track_stay_within_the_source_bounds(self) -> None:
        # A subject hugging the right edge of the source: the crop cannot
        # centre them without leaving the frame, and it must not try.
        observations = [self._observation(i * 0.2, 1850.0) for i in range(10)]

        segment = self._segment(observations)

        assert 0.0 <= segment.keyframes[0].x <= self.SOURCE_W - self.CROP_W


class TestTracking:
    def _observation(self, t: float, cx: float, cy: float = 400.0) -> FaceObservation:
        return FaceObservation(t=t, cx=cx, cy=cy, width=200, height=260, eye_y=cy - 40, mar=0.05)

    def test_a_moving_face_becomes_one_track(self) -> None:
        observations = [self._observation(i * 0.2, 500 + i * 5) for i in range(20)]

        tracks = build_tracks(observations)

        assert len(tracks) == 1
        assert len(tracks[0].observations) == 20

    def test_two_separated_faces_become_two_tracks(self) -> None:
        observations = [
            obs
            for i in range(20)
            for obs in (self._observation(i * 0.2, 400), self._observation(i * 0.2, 1400))
        ]

        tracks = build_tracks(observations)

        assert len(tracks) == 2

    def test_a_long_gap_splits_a_track(self) -> None:
        early = [self._observation(i * 0.2, 500) for i in range(10)]
        late = [self._observation(20 + i * 0.2, 500) for i in range(10)]

        tracks = build_tracks(early + late)

        assert len(tracks) == 2

    def test_transient_detections_are_dropped(self) -> None:
        stable = [self._observation(i * 0.2, 500) for i in range(20)]
        blip = [self._observation(1.0, 1700)]

        tracks = build_tracks(stable + blip)

        assert len(tracks) == 1

    def test_tracks_rank_by_prominence(self) -> None:
        big = [
            FaceObservation(t=i * 0.2, cx=500, cy=400, width=400, height=500, eye_y=360, mar=0.1)
            for i in range(20)
        ]
        small = [
            FaceObservation(t=i * 0.2, cx=1500, cy=400, width=90, height=110, eye_y=380, mar=0.1)
            for i in range(20)
        ]

        tracks = build_tracks(big + small)

        assert tracks[0].mean_area > tracks[1].mean_area

    def test_no_observations_gives_no_tracks(self) -> None:
        assert build_tracks([]) == []

    def test_mouth_activity_distinguishes_talking_from_still(self) -> None:
        talking = build_tracks(
            [
                FaceObservation(
                    t=i * 0.2,
                    cx=500,
                    cy=400,
                    width=200,
                    height=260,
                    eye_y=360,
                    mar=0.05 + (0.15 if i % 2 else 0.0),
                )
                for i in range(20)
            ]
        )[0]
        still = build_tracks(
            [
                FaceObservation(
                    t=i * 0.2, cx=1500, cy=400, width=200, height=260, eye_y=360, mar=0.05
                )
                for i in range(20)
            ]
        )[0]

        assert talking.mouth_activity(0.0, 4.0) > still.mouth_activity(0.0, 4.0)


def _evaluate(expression: str, t: float) -> float:
    """Evaluate an ffmpeg-style expression in Python, for test assertions only."""
    import re

    python_expression = re.sub(r"\blt\(", "_lt(", expression)
    python_expression = re.sub(r"\bif\(", "_if(", python_expression)
    return eval(  # noqa: S307 - test-only evaluation of expressions we generated
        python_expression,
        {
            "_if": lambda cond, a, b: a if cond else b,
            "_lt": lambda a, b: a < b,
            "t": t,
        },
    )

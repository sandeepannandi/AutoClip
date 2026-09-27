"""Silence tightening — span construction, timeline remapping, filtergraph shape.

These are pure-logic tests: the span builder and its transforms are where
A/V sync bugs and timeline drift would be born, and they're cheap to pin.
The one real render (slow) proves the whole ffmpeg path end to end.
"""

from __future__ import annotations

import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import export, tighten
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
)


def _silences(*bounds: tuple[float, float]) -> list[Silence]:
    return [Silence(start=s, end=e) for s, e in bounds]


class TestBuildPlan:
    def test_gaps_shorter_than_the_minimum_stay_at_1x(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((10.0, 10.3)))

        assert len(plan.spans) == 1
        assert plan.spans[0].speed == 1.0
        assert plan.output_duration_s == pytest.approx(30.0)

    def test_long_gaps_are_sped_and_the_rest_is_untouched(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)))

        assert [(s.start_s, s.end_s, s.speed) for s in plan.spans] == [
            (0.0, 10.12, 1.0),
            (10.12, 11.88, 4.0),
            (11.88, 30.0, 1.0),
        ]
        # 1.76s of gap plays at 4x = 0.44s: 1.32s saved.
        assert plan.output_duration_s == pytest.approx(30.0 - 1.32)
        assert plan.saved_s == pytest.approx(1.32)

    def test_keep_margins_leave_air_around_each_gap(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)))

        assert plan.spans[1].start_s == pytest.approx(10.0 + tighten.DEFAULT_KEEP_SILENCE_S)
        assert plan.spans[1].end_s == pytest.approx(12.0 - tighten.DEFAULT_KEEP_SILENCE_S)

    def test_hook_protection_keeps_the_first_moment_intact(self) -> None:
        # A gap opening inside the first 0.75s only tightens past it.
        plan = tighten.build_plan(0.0, 30.0, _silences((0.0, 5.0)))

        assert plan.spans[0].speed == 1.0
        assert plan.spans[0].end_s == pytest.approx(tighten.HOOK_PROTECTION_S)
        assert plan.spans[1].speed == 4.0

    def test_overlapping_silences_merge_into_one_span(self) -> None:
        plain = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0), (11.0, 14.0)))
        split = tighten.build_plan(0.0, 30.0, _silences((10.0, 14.0)))

        assert [(s.start_s, s.end_s, s.speed) for s in plain.spans] == [
            (s.start_s, s.end_s, s.speed) for s in split.spans
        ]

    def test_silences_outside_the_clip_are_ignored(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((-5.0, -1.0), (35.0, 40.0)))

        assert len(plan.spans) == 1
        assert plan.spans[0].speed == 1.0

    def test_spans_tile_the_clip_exactly(self) -> None:
        plan = tighten.build_plan(
            0.0, 40.0, _silences((5.0, 6.0), (9.0, 13.0), (20.0, 24.0), (30.0, 30.7))
        )

        cursor = 0.0
        for span in plan.spans:
            assert span.start_s == pytest.approx(cursor)
            cursor = span.end_s
        assert cursor == pytest.approx(40.0)

    def test_unsafe_compression_is_rejected(self) -> None:
        # 20 minutes of "silence" in a 30s clip is a broken VAD, not a script.
        with pytest.raises(tighten.TightenError):
            tighten.build_plan(0.0, 30.0, _silences((0.5, 29.9)))

    def test_speed_must_exceed_one(self) -> None:
        with pytest.raises(ValueError):
            tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)), speed=1.0)


class TestRemap:
    def test_remap_compresses_only_the_tightened_span(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)))

        # Before the gap: identity.
        assert plan.remap(5.0) == pytest.approx(5.0)
        # Inside: 10.12 + (t - 10.12) / 4.
        assert plan.remap(11.0) == pytest.approx(10.12 + 0.22)
        # After: the 1.32s the gap shed is removed.
        assert plan.remap(20.0) == pytest.approx(20.0 - 1.32)

    def test_unmap_is_the_inverse(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)))

        for t in (0.0, 5.0, 10.5, 11.0, 12.5, 20.0, 29.9, 30.0):
            assert plan.unmap(plan.remap(t)) == pytest.approx(t, abs=1e-9)

    def test_output_duration_matches_remap_of_the_end(self) -> None:
        plan = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)))

        assert plan.remap(30.0) == pytest.approx(plan.output_duration_s)


class TestAtempoChain:
    def test_two_times_is_a_single_filter(self) -> None:
        assert tighten.atempo_chain(2.0) == "atempo=2.0"

    def test_four_times_chains_two(self) -> None:
        assert tighten.atempo_chain(4.0) == "atempo=2.0,atempo=2.0"

    def test_one_times_is_an_empty_chain(self) -> None:
        assert tighten.atempo_chain(1.0) == "atempo=1.0"

    def test_speed_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            tighten.atempo_chain(0.0)


def _request(silences: list[Silence] | None = None, *, segments: int = 1) -> export.ExportRequest:
    width, height = 404, 720
    crop_segments = [
        CropSegment(
            start_s=0.0,
            end_s=30.0,
            width=width,
            height=height,
            keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
            strategy=Strategy.TRACK,
        )
    ]
    return export.ExportRequest(
        source=None,  # type: ignore[arg-type] - filtergraph tests never touch the file
        destination=None,  # type: ignore[arg-type]
        start_s=0.0,
        end_s=30.0,
        crop_path=CropPath(source_width=1280, source_height=720, segments=crop_segments),
        words=[],
        style=None,  # type: ignore[arg-type]
        burn_captions=False,
        silences=silences,
    )


class TestTightenedFiltergraph:
    SETTINGS = ExportSettings(tighten_silences=True)
    PLAN = tighten.build_plan(0.0, 30.0, _silences((10.0, 12.0)))

    def test_disabled_settings_render_untightened(self) -> None:
        request = _request(_silences((10.0, 12.0)))
        settings = ExportSettings(tighten_silences=False)

        assert export._plan_for_request(request, settings) is None

    def test_no_silences_renders_untightened(self) -> None:
        assert export._plan_for_request(_request(), self.SETTINGS) is None

    def test_plan_is_built_when_enabled(self) -> None:
        request = _request(_silences((10.0, 12.0)))

        assert export._plan_for_request(request, self.SETTINGS) is not None

    def test_unsafe_plan_falls_back_to_untightened(self) -> None:
        request = _request(_silences((0.5, 29.9)))

        assert export._plan_for_request(request, self.SETTINGS) is None

    def test_video_graph_spans_carry_their_speed(self) -> None:
        graph = export.build_video_filtergraph(
            _request(_silences((10.0, 12.0))),
            subtitle_name=None,
            fonts_name="fonts",
            plan=self.PLAN,
        )

        # The fast span's setpts divides by the speed.
        assert "setpts=(PTS-STARTPTS)/4.000000" in graph
        assert "setpts=(PTS-STARTPTS)/1.000000" in graph
        assert "concat=n=3" in graph

    def test_audio_graph_chains_atempo_and_loudnorm(self) -> None:
        graph = export.build_tightened_audio_filtergraph(
            _request(_silences((10.0, 12.0))), self.PLAN, self.SETTINGS
        )

        assert "atrim=start=10.1200:end=11.8800" in graph
        assert "atempo=2.0,atempo=2.0" in graph
        assert "concat=n=3:v=0:a=1" in graph
        assert graph.endswith("[acat]loudnorm=I=-14.0:TP=-1.5:LRA=11.0[aout]")

    def test_untightened_audio_graph_is_the_historical_one(self) -> None:
        request = _request()
        plan = export._plan_for_request(request, self.SETTINGS)

        assert plan is None
        assert export.build_audio_filtergraph(self.SETTINGS) == (
            "loudnorm=I=-14.0:TP=-1.5:LRA=11.0"
        )

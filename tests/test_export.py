"""Export filtergraph construction — pure string unit tests, no ffmpeg needed.

:func:`build_video_filtergraph` is the piece of the export stage that decides
the order filters run in, and ordering mistakes here are invisible to the
render tests until a frame is produced. These tests pin the shape of the
graph, especially the colour-grade chain which must sit BEFORE the ``ass``
filter so burned-in captions stay un-graded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from autoclip.pipeline import captions, export, punch
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
)
from autoclip.pipeline.tighten import build_plan
from autoclip.pipeline.transcript import Word

TRACKING_GRADES: list[str] = ["warm", "punchy", "cool", "film"]


class TestEncoderArgs:
    """Quality presets: slow x264, p4 NVENC — clips are short, spend the time."""

    @staticmethod
    def _patch_report(monkeypatch, nvenc_works: bool) -> None:
        from types import SimpleNamespace

        fake = SimpleNamespace(ffmpeg=SimpleNamespace(nvenc_works=nvenc_works))
        monkeypatch.setattr(export, "report", lambda: fake)

    def test_software_encoder_uses_the_slow_preset(self, monkeypatch) -> None:
        from autoclip.config import ExportSettings

        self._patch_report(monkeypatch, nvenc_works=False)

        args = export.encoder_args(ExportSettings(prefer_hardware_encoder=True))

        assert "libx264" in args
        assert args[args.index("-preset") + 1] == "slow"

    def test_nvenc_uses_p4_when_available(self, monkeypatch) -> None:
        from autoclip.config import ExportSettings

        self._patch_report(monkeypatch, nvenc_works=True)

        args = export.encoder_args(ExportSettings(prefer_hardware_encoder=True))

        assert "h264_nvenc" in args
        assert args[args.index("-preset") + 1] == "p4"

    def test_hardware_encoder_is_skipped_when_disabled(self, monkeypatch) -> None:
        from autoclip.config import ExportSettings

        self._patch_report(monkeypatch, nvenc_works=True)

        args = export.encoder_args(ExportSettings(prefer_hardware_encoder=False))

        assert "libx264" in args


def _request(
    color_grade: str = "none", *, segments: int = 1, burn_captions: bool = True
) -> export.ExportRequest:
    width, height = 404, 720
    crop_segments = [
        CropSegment(
            start_s=0.0,
            end_s=2.5,
            width=width,
            height=height,
            keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
            strategy=Strategy.TRACK,
        )
    ]
    if segments > 1:
        crop_segments.append(
            CropSegment(
                start_s=2.5,
                end_s=5.0,
                width=width,
                height=height,
                keyframes=[CropKeyframe(2.5, 400.0, 0.0)],
                strategy=Strategy.TRACK,
            )
        )
    return export.ExportRequest(
        source=Path("source.mp4"),
        destination=Path("out.mp4"),
        start_s=0.0,
        end_s=5.0,
        crop_path=CropPath(source_width=1280, source_height=720, segments=crop_segments),
        words=[],
        style=captions.get_style("bold_pop"),
        burn_captions=burn_captions,
        color_grade=color_grade,
    )


class TestGrades:
    def test_none_adds_no_grade_filters(self) -> None:
        graph = export.build_video_filtergraph(_request(), subtitle_name="captions.ass")

        assert "colorbalance" not in graph
        assert "curves" not in graph
        assert "eq=saturation" not in graph
        assert graph.endswith("[v0]ass=filename=captions.ass:fontsdir=fonts[vout]")

    @pytest.mark.parametrize("grade", TRACKING_GRADES)
    def test_grade_chain_runs_before_captions(self, grade: str) -> None:
        graph = export.build_video_filtergraph(
            _request(color_grade=grade), subtitle_name="captions.ass"
        )

        # The grade chain sits between the scaled footage and the ass filter,
        # via its own [vgrade] label so captions burn in on the treated image.
        assert f"[v0]{export.grade_filters(grade)}[vgrade]" in graph
        assert graph.endswith("[vgrade]ass=filename=captions.ass:fontsdir=fonts[vout]")

    def test_grade_applies_without_captions(self) -> None:
        graph = export.build_video_filtergraph(
            _request(color_grade="warm", burn_captions=False),
            subtitle_name="captions.ass",
        )

        assert (
            "[v0]colortemperature=temperature=5000,eq=saturation=1.12:contrast=1.03[vgrade]"
            in graph
        )
        assert graph.endswith("[vgrade]null[vout]")

    def test_grade_applies_after_multi_segment_concatenation(self) -> None:
        graph = export.build_video_filtergraph(
            _request(color_grade="cool", segments=2), subtitle_name="captions.ass"
        )

        assert "[vcat]colortemperature=temperature=7500,eq=saturation=1.06[vgrade]" in graph
        assert "[vgrade]ass=filename=captions.ass" in graph

    def test_unknown_grade_raises_export_error(self) -> None:
        with pytest.raises(export.ExportError, match="Unknown color grade"):
            export.build_video_filtergraph(
                _request(color_grade="mucky"), subtitle_name="captions.ass"
            )

    def test_grade_filters_rejects_unknown_names(self) -> None:
        with pytest.raises(export.ExportError, match="Unknown color grade"):
            export.grade_filters("mucky")

    def test_grade_filters_returns_noop_for_none(self) -> None:
        assert export.grade_filters("none") == ""


class TestPunches:
    """The punch-in zoompan sits after the grade, before the captions."""

    @staticmethod
    def _request_with_punches(**kwargs) -> export.ExportRequest:
        request = _request(**kwargs)
        request.punches = [
            punch.Punch(
                start_s=1.0,
                zoom_in_end_s=1.3,
                hold_end_s=1.8,
                end_s=2.2,
                zoom=punch.DEFAULT_ZOOM,
            )
        ]
        return request

    def test_no_punches_adds_no_zoompan(self) -> None:
        graph = export.build_video_filtergraph(_request(), subtitle_name="captions.ass")

        assert "vpunch" not in graph
        assert "zoompan" not in graph

    def test_punch_chain_runs_after_grade_before_captions(self) -> None:
        graph = export.build_video_filtergraph(
            self._request_with_punches(color_grade="warm"), subtitle_name="captions.ass"
        )

        assert "[vgrade]fps=30,zoompan=z='" in graph
        # The peak zoom flows into the expression as 1 + zoom.
        assert "1.0800" in graph
        assert graph.endswith("[vpunch]ass=filename=captions.ass:fontsdir=fonts[vout]")


class TestWordRemap:
    """Caption words must land on the clip-relative output timeline exactly once."""

    @staticmethod
    def _request_with_words() -> export.ExportRequest:
        base = 3045.75  # a mid-video clip start, like real jobs produce
        request = _request()
        request.start_s = base
        request.end_s = base + 5.0
        phrase = "nobody tells you this but most agencies die".split()  # noqa: SIM905
        request.words = [
            Word(text=token, start=base + i * 0.5, end=base + i * 0.5 + 0.45, speaker=None)
            for i, token in enumerate(phrase)
        ]
        return request

    def test_untightened_words_are_clip_relative(self) -> None:
        words = export._remap_words(self._request_with_words(), None)

        assert words[0].start == pytest.approx(0.0)
        assert words[-1].end == pytest.approx(3.95)

    def test_tightened_words_are_not_shifted_twice(self) -> None:
        # Regression: remap already returns clip-relative output times, and the
        # subtitle writer subtracted the clip start again — every event clamped
        # to 0:00:00.00 and captions burned in invisibly.
        request = self._request_with_words()
        plan = build_plan(request.start_s, request.end_s, [Silence(start=3050.0, end=3051.0)])

        words = export._remap_words(request, plan)

        assert words[0].start == pytest.approx(0.0, abs=0.01)
        assert words[-1].end <= plan.output_duration_s + 0.01
        # The tighten plan compresses later words forward, it does not push
        # them off the front of the video.
        assert all(word.end > word.start for word in words)

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
from autoclip.pipeline import captions, export
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
)

TRACKING_GRADES: list[str] = ["warm", "punchy", "cool", "film"]


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

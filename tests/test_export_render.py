"""End-to-end render tests against real ffmpeg.

These are the tests that catch what unit tests structurally cannot: a
filtergraph that parses but produces the wrong thing, a font libass can't
resolve, a path escaping bug that only appears on Windows, an encoder flag the
local build rejects. Marked ``slow`` — they shell out and encode video.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import captions, export, ffmpeg
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
    centre_crop,
)
from autoclip.pipeline.transcript import Word

pytestmark = pytest.mark.slow

SOURCE_W, SOURCE_H = 1280, 720
SOURCE_DURATION = 10.0


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    """A synthetic 1280x720 test clip with a tone, generated once per session."""
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={SOURCE_W}x{SOURCE_H}:rate=30:duration={SOURCE_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={SOURCE_DURATION}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def words() -> list[Word]:
    """Words spanning 2.0-7.0s of the source, matching the clip under test."""
    texts = [
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "caption",
        "rendering",
        "pipeline",
        "right",
        "now",
    ]
    step = 5.0 / len(texts)
    return [
        Word(text=text, start=2.0 + i * step, end=2.0 + i * step + step * 0.85)
        for i, text in enumerate(texts)
    ]


def make_request(source: Path, destination: Path, crop_path: CropPath, words, **kwargs):
    return export.ExportRequest(
        source=source,
        destination=destination,
        start_s=2.0,
        end_s=7.0,
        crop_path=crop_path,
        words=words,
        style=kwargs.pop("style", captions.get_style("bold_pop")),
        **kwargs,
    )


class TestSingleSegmentRender:
    def test_renders_a_vertical_clip(self, source_video, words, tmp_path) -> None:
        destination = tmp_path / "out.mp4"
        request = make_request(
            source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.exists()
        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.has_audio
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    @pytest.mark.parametrize("ratio", ["9:16", "1:1", "16:9"])
    def test_every_ratio_renders(self, source_video, words, tmp_path, ratio: str) -> None:
        destination = tmp_path / f"out_{ratio.replace(':', 'x')}.mp4"
        aspect = tuple(int(part) for part in ratio.split(":"))
        crop_path = centre_crop(SOURCE_W, SOURCE_H, 5.0, aspect_w=aspect[0], aspect_h=aspect[1])

        export.export_clip(
            make_request(source_video, destination, crop_path, words, ratio=ratio),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == export.ratio_dimensions(ratio)

    @pytest.mark.parametrize("style_key", list(captions.PRESETS))
    def test_every_caption_style_burns_in(
        self, source_video, words, tmp_path, style_key: str
    ) -> None:
        # A style whose font libass can't resolve still "succeeds" but renders
        # in a substituted face, so this checks the render completes for each.
        destination = tmp_path / f"out_{style_key}.mp4"
        request = make_request(
            source_video,
            destination,
            centre_crop(SOURCE_W, SOURCE_H, 5.0),
            words,
            style=captions.get_style(style_key),
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.stat().st_size > 1000

    def test_captions_visibly_change_the_output(self, source_video, words, tmp_path) -> None:
        """Burned captions must actually alter the pixels.

        Without this, a silently-failing `ass` filter would leave every other
        assertion passing while shipping clips with no captions on them.
        """
        with_captions = tmp_path / "with.mp4"
        without_captions = tmp_path / "without.mp4"

        export.export_clip(
            make_request(source_video, with_captions, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )
        export.export_clip(
            make_request(
                source_video,
                without_captions,
                centre_crop(SOURCE_W, SOURCE_H, 5.0),
                words,
                burn_captions=False,
            ),
            work_dir=tmp_path / "work",
        )

        assert _frame_signature(with_captions, 2.5) != _frame_signature(without_captions, 2.5)


class TestMultiSegmentRender:
    def test_concatenated_segments_render(self, source_video, words, tmp_path) -> None:
        crop_w, crop_h = 404, 720
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=2.5,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(0.0, 100.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
                CropSegment(
                    start_s=2.5,
                    end_s=5.0,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(2.5, 700.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
            ],
        )
        destination = tmp_path / "multi.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    def test_segments_actually_show_different_regions(self, source_video, words, tmp_path) -> None:
        """A crop that doesn't move would make the two segments identical."""
        crop_w, crop_h = 404, 720
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=2.5,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
                ),
                CropSegment(
                    start_s=2.5,
                    end_s=5.0,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(2.5, 876.0, 0.0)],
                ),
            ],
        )
        destination = tmp_path / "regions.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words, burn_captions=False),
            work_dir=tmp_path / "work",
        )

        assert _frame_signature(destination, 1.0) != _frame_signature(destination, 4.0)

    def test_animated_crop_expression_renders(self, source_video, words, tmp_path) -> None:
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=404,
                    height=720,
                    keyframes=[
                        CropKeyframe(0.0, 0.0, 0.0),
                        CropKeyframe(2.5, 400.0, 0.0),
                        CropKeyframe(5.0, 876.0, 0.0),
                    ],
                    strategy=Strategy.TRACK,
                )
            ],
        )
        destination = tmp_path / "animated.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words, burn_captions=False),
            work_dir=tmp_path / "work",
        )

        # A broken expression would evaluate to a constant, leaving these equal.
        assert _frame_signature(destination, 0.5) != _frame_signature(destination, 4.5)

    def test_fit_segment_renders_with_blurred_background(
        self, source_video, words, tmp_path
    ) -> None:
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=SOURCE_W,
                    height=SOURCE_H,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
                    strategy=Strategy.WIDE,
                    fit=True,
                )
            ],
        )
        destination = tmp_path / "fit.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)


class TestEncoding:
    def test_software_encoder_path(self, source_video, words, tmp_path) -> None:
        settings = ExportSettings(prefer_hardware_encoder=False)
        destination = tmp_path / "x264.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        assert ffmpeg.probe(destination).video_codec == "h264"

    def test_output_is_yuv420p(self, source_video, words, tmp_path) -> None:
        # Anything else fails to play on a surprising number of phones.
        destination = tmp_path / "pixfmt.mp4"
        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )

        result = subprocess.run(
            [
                ffmpeg.ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "csv=p=0",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "yuv420p"

    def test_srt_sidecar_is_written_when_requested(self, source_video, words, tmp_path) -> None:
        settings = ExportSettings(write_srt=True)
        destination = tmp_path / "sidecar.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        assert destination.with_suffix(".srt").exists()

    def test_progress_reaches_completion(self, source_video, words, tmp_path) -> None:
        seen: list[float] = []

        export.export_clip(
            make_request(
                source_video, tmp_path / "progress.mp4", centre_crop(SOURCE_W, SOURCE_H, 5.0), words
            ),
            work_dir=tmp_path / "work",
            on_progress=seen.append,
        )

        assert seen
        assert seen[-1] == 1.0
        assert all(0.0 <= value <= 1.0 for value in seen)


class TestPathsWithSpecialCharacters:
    def test_directory_with_spaces_and_punctuation(self, source_video, words, tmp_path) -> None:
        """The escaping test that actually matters on Windows.

        A drive letter plus a space plus an apostrophe is the combination that
        breaks naive filtergraph quoting.
        """
        awkward = tmp_path / "My Clips (2026)" / "it's here"
        awkward.mkdir(parents=True)
        destination = awkward / "out.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=awkward / "work",
        )

        assert destination.exists()
        assert ffmpeg.probe(destination).width == 1080


@pytest.mark.parametrize("grade", ["warm", "punchy", "cool", "film"])
class TestColorGrading:
    """Each grade must clearly shift pixels versus an ungraded clip.

    A grade that parses but barely does anything — a typo in a filter
    expression, an option the local ffmpeg ignores, or an over-timid preset —
    would pass a hash-equality check and ship clips indistinguishable from
    ungraded ones. The mean-abs threshold instead requires a perceptible move.
    """

    #: Floor on the mean abs RGB shift (0..255) below which a grade doesn't
    #: visibly read on screen. The presets sit at ~2-8, well above it.
    MIN_MEAN_DELTA = 2.0

    def test_grade_visibly_changes_the_output(
        self, source_video, words, tmp_path, grade: str
    ) -> None:
        plain = tmp_path / "plain.mp4"
        graded = tmp_path / f"grade_{grade}.mp4"

        export.export_clip(
            make_request(source_video, plain, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )
        export.export_clip(
            make_request(
                source_video,
                graded,
                centre_crop(SOURCE_W, SOURCE_H, 5.0),
                words,
                color_grade=grade,
            ),
            work_dir=tmp_path / "work",
        )

        delta = _mean_abs_delta(graded, plain, timestamps=(1.0, 2.5, 4.0))
        assert delta >= self.MIN_MEAN_DELTA


class TestCaptionColourOverride:
    """The per-clip primary colour must reach the ASS the renderer burns in."""

    def test_override_surfaces_in_the_captions_ass(self, source_video, words, tmp_path) -> None:
        destination = tmp_path / "out.mp4"
        request = make_request(
            source_video,
            destination,
            centre_crop(SOURCE_W, SOURCE_H, 5.0),
            words,
            primary_color="#FFE500",
            style=captions.get_style("clean_lower"),
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.exists()
        ass_path = tmp_path / "work" / destination.stem / "captions.ass"
        assert ass_path.exists()
        style_line = next(
            line
            for line in ass_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(f"Style: {captions.STYLE_NAME},")
        )
        assert _ass_colour(255, 229, 0) in style_line

    def test_preset_primary_is_written_when_no_override(
        self, source_video, words, tmp_path
    ) -> None:
        destination = tmp_path / "out.mp4"
        export.export_clip(
            make_request(
                source_video,
                destination,
                centre_crop(SOURCE_W, SOURCE_H, 5.0),
                words,
                style=captions.get_style("clean_lower"),
            ),
            work_dir=tmp_path / "work",
        )

        assert destination.exists()
        ass_path = tmp_path / "work" / destination.stem / "captions.ass"
        style_line = next(
            line
            for line in ass_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(f"Style: {captions.STYLE_NAME},")
        )
        assert _ass_colour(255, 255, 255) in style_line


def _ass_colour(r: int, g: int, b: int, alpha: int = 0) -> str:
    """Format RGB as the ASS style-field colour, ``&HAABBGGRR``."""
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _frame_signature(video: Path, timestamp: float) -> str:
    """Hash one frame's pixels, for comparing rendered output."""
    result = subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "hash",
            "-hash",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _frame_rgb(video: Path, timestamp: float) -> np.ndarray:
    """Decode one frame as a ``(h, w, 3)`` uint8 RGB array."""
    probe = ffmpeg.probe(video)
    width, height = int(probe.width), int(probe.height)
    result = subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    pixels = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = width * height * 3
    if pixels.size != expected:
        raise AssertionError(
            f"decoded {pixels.size} bytes for {video}@t={timestamp}, wanted {expected}"
        )
    return pixels.reshape(height, width, 3)


def _mean_abs_delta(
    left: Path, right: Path, timestamps: tuple[float, ...] = (1.0, 2.5, 4.0)
) -> float:
    """Mean per-pixel abs RGB difference between two renders, over sample frames.

    Frame-correlated content (subtitles, the scene itself) is identical in both
    videos and cancels out; only the grade's own effect is measured.
    """
    if not timestamps:
        raise AssertionError("need at least one timestamp to compare")
    total = 0.0
    for timestamp in timestamps:
        a = _frame_rgb(left, timestamp)
        b = _frame_rgb(right, timestamp)
        if a.shape != b.shape:
            raise AssertionError(f"dimension mismatch at t={timestamp}: {a.shape} vs {b.shape}")
        total += float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    return total / len(timestamps)

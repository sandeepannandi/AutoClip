"""Export stage — crop path plus captions plus audio normalisation to an MP4.

The whole render is a single ffmpeg pass. Per-shot crop segments are trimmed out
of the source, cropped independently, scaled to a common output size, and
concatenated in the filtergraph — which is what allows a wide two-shot and a
tight single to sit in one clip without a mid-stream frame-size change.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import ExportSettings
from ..db.models import CaptionPosition
from ..system import report
from . import captions as captions_module
from . import ffmpeg, tighten
from . import punch as punch_module
from .captions import CaptionStyle
from .prepare import Silence
from .reframe.croppath import CropKeyframe, CropPath, CropSegment, segment_crop_filter
from .transcript import Word

log = logging.getLogger(__name__)

#: Output dimensions per aspect ratio. 1080-wide is the platform sweet spot:
#: high enough to avoid re-encode mush, low enough to upload quickly.
RATIOS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

#: Colour grading presets, each an ffmpeg filter chain applied to the footage
#: before subtitles are burned in. "none" inserts nothing.
#:
#: Calibrated numerically (mean abs delta vs an ungraded frame: warm ~4%,
#: cool ~3%, punchy ~2%, film ~8% on a test pattern) so each reads clearly on a
#: phone screen. The render suite gates every preset on a minimum pixel shift,
#: so a future tweak can't silently fade back to imperceptible.
GRADES: dict[str, str] = {
    "none": "",
    "warm": "colortemperature=temperature=5000,eq=saturation=1.12:contrast=1.03",
    "punchy": "eq=contrast=1.10:saturation=1.35,curves=master='0/0.02 0.5/0.5 1/0.98'",
    "cool": "colortemperature=temperature=7500,eq=saturation=1.06",
    "film": "curves=master='0/0.05 0.5/0.5 1/0.95',eq=saturation=0.92",
}

#: Loudness targets. -14 LUFS integrated with -1.5 dBTP is what every major
#: platform normalises toward, so hitting it ourselves avoids their processing.
LOUDNESS_TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11.0

#: The crop-path ``headroom`` value at which padding engages. The reframe stage
#: emits exactly 0.0 or a positive fraction, so this is a single decision point
#: that keeps the two stages decoupled.
HEADROOM_ENGAGE = 0.02

#: Frame rate the punch zoompan runs at. A fixed rational rate keeps the
#: expression's frame counter (``on``) deterministic across inputs.
PUNCH_FPS = 30

AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 48_000


class ExportError(RuntimeError):
    """Rendering a clip failed."""


@dataclass
class ExportRequest:
    source: Path
    destination: Path
    start_s: float
    end_s: float
    crop_path: CropPath
    words: list[Word]
    style: CaptionStyle
    ratio: str = "9:16"
    burn_captions: bool = True
    caption_position: CaptionPosition | None = None
    #: Colour grade preset name, a key of :data:`GRADES`. "none" leaves the
    #: footage untouched.
    color_grade: str = "none"
    #: Per-clip caption primary colour override (#RRGGBB). None keeps the
    #: preset's own primary.
    primary_color: str | None = None
    #: Detected silences on the source timeline, when the caller has them.
    #: With tightening enabled these become fast-forward spans; None or empty
    #: simply renders untightened.
    silences: list[Silence] | None = None
    #: Punch-in zoom events, on the same source-absolute timeline as ``words``.
    #: The export stage converts them to the output timeline (through the
    #: tighten plan when tightening) and renders them after the concat/grade,
    #: before captions — the text stays static and legible while the footage
    #: moves. Empty renders nothing extra.
    punches: list[punch_module.Punch] | None = None
    #: Leave um/uh and other hesitation sounds out of the burned-in captions.
    #: Captions only — footage and audio are untouched.
    filler_word_cleanup: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def ratio_dimensions(ratio: str) -> tuple[int, int]:
    if ratio not in RATIOS:
        raise ExportError(f"Unknown ratio {ratio!r}. Available: {', '.join(RATIOS)}")
    return RATIOS[ratio]


def grade_filters(color_grade: str) -> str:
    """Return the ffmpeg filter chain for a grade preset name.

    Raises :class:`ExportError` for unknown names so callers can surface a 400
    before any rendering work starts.
    """
    try:
        return GRADES[color_grade]
    except KeyError:
        raise ExportError(
            f"Unknown color grade {color_grade!r}. Available: {', '.join(GRADES)}"
        ) from None


def slugify_title(title: str, *, max_length: int = 50) -> str:
    """Filename-safe slug for a clip title."""
    slug = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


def output_filename(title: str, ratio: str) -> str:
    return f"{slugify_title(title)}_{ratio.replace(':', 'x')}.mp4"


# --------------------------------------------------------------------------
# Filtergraph construction
# --------------------------------------------------------------------------


def _plan_for_request(
    request: ExportRequest, settings: ExportSettings
) -> tighten.TightenPlan | None:
    """The tighten plan for this clip, or None to render untightened.

    Disabled in settings, no silences provided, or unsafely compressible data
    (TightenError) all mean the ordinary 1x timeline. A plan failure must never
    fail an export — the untightened clip is always an acceptable result.
    """
    if not settings.tighten_silences or not request.silences:
        return None
    try:
        return tighten.build_plan(
            request.start_s,
            request.end_s,
            request.silences,
            speed=settings.silence_speed,
            min_silence_s=settings.min_silence_s,
            keep_silence_s=settings.keep_silence_s,
        )
    except (tighten.TightenError, ValueError) as exc:
        log.warning(
            "Tightening skipped for the clip %.1f-%.1fs: %s",
            request.start_s,
            request.end_s,
            exc,
        )
        return None


def _tighten_spans(request: ExportRequest, plan: tighten.TightenPlan) -> list[tighten.Span]:
    """Render spans: the plan's spans split at crop-segment boundaries.

    Two timelines meet here, and keeping them straight is the whole game:

    - the **tighten plan** and the **silences** are source-absolute — they
      come from audio analysis of the whole file;
    - the **crop path** is clip-relative (0 is the clip's first frame), which
      is also what the ffmpeg input looks like after ``-ss`` seeking.

    So: lift each crop segment into source time, intersect with the plan's
    spans, and the resulting render spans stay source-absolute. Every consumer
    (trim filters, keyframe rebasing, atrim) converts back to clip-relative at
    the point of use.

    A well-formed crop path tiles the clip exactly (``build_crop_path``
    guarantees it), so the intersections cover everything. A malformed one
    must never silently drop footage: any range the intersections leave
    uncovered is filled with a 1x span and a warning.
    """
    render_spans: list[tighten.Span] = []
    for segment in request.crop_path.segments:
        # Lift the clip-relative segment into source time.
        seg_start = segment.start_s + request.start_s
        seg_end = segment.end_s + request.start_s
        if seg_end <= seg_start:
            continue
        for span in plan.spans:
            start = max(seg_start, span.start_s)
            end = min(seg_end, span.end_s)
            if end - start <= 0:
                continue
            render_spans.append(tighten.Span(start, end, span.speed))

    render_spans.sort(key=lambda s: s.start_s)

    # Fill any hole in the tiling at normal speed rather than dropping it.
    tiled: list[tighten.Span] = []
    cursor = request.start_s
    for span in render_spans:
        if span.start_s > cursor + 1e-9:
            log.warning(
                "Crop path left %.2fs untiled (%.2f-%.2f); filling at 1x.",
                span.start_s - cursor,
                cursor,
                span.start_s,
            )
            tiled.append(tighten.Span(cursor, span.start_s, 1.0))
        tiled.append(span)
        cursor = max(cursor, span.end_s)
    if cursor < request.end_s - 1e-9:
        log.warning(
            "Crop path ended at %.2f before the clip end %.2f; filling at 1x.",
            cursor,
            request.end_s,
        )
        tiled.append(tighten.Span(cursor, request.end_s, 1.0))

    return tiled


def _rebased_segment(
    segment: CropSegment, rel_start: float, rel_end: float
) -> CropSegment:
    """CropSegment rebased onto a render span's clip-relative local timeline.

    ``rel_start``/``rel_end`` are the render span's clip-relative bounds (0 is
    the clip's first frame — the same timeline the crop path lives on). A
    static segment collapses to one keyframe at zero; a moving one keeps its
    shape with keyframe times shifted so the crop motion stays synchronised
    with the footage inside the fast-forwarded spans.
    """
    if len(segment.keyframes) <= 1:
        keyframes = [CropKeyframe(0.0, segment.keyframes[0].x, segment.keyframes[0].y)]
    else:
        keyframes = [
            CropKeyframe(k.t - rel_start, k.x, k.y)
            for k in segment.keyframes
            if rel_start <= k.t <= rel_end
        ]
        if not keyframes:
            keyframes = [CropKeyframe(0.0, segment.keyframes[0].x, segment.keyframes[0].y)]
        keyframes[0] = CropKeyframe(0.0, keyframes[0].x, keyframes[0].y)
        keyframes[-1] = CropKeyframe(rel_end - rel_start, keyframes[-1].x, keyframes[-1].y)

    return CropSegment(
        start_s=0.0,
        end_s=rel_end - rel_start,
        width=segment.width,
        height=segment.height,
        keyframes=keyframes,
        strategy=segment.strategy,
        zoom=segment.zoom,
        fit=segment.fit,
        headroom=segment.headroom,
    )


def build_video_filtergraph(
    request: ExportRequest,
    *,
    subtitle_name: str | None,
    fonts_name: str = "fonts",
    plan: tighten.TightenPlan | None = None,
) -> str:
    """Build the ``-filter_complex`` video chain for one clip.

    Without a plan this is the historical graph: crop segments trimmed from the
    seeked input, cropped, scaled, concatenated, graded, subtitled. With a plan
    the trim list is the (crop segment × tighten span) intersections instead,
    each also sped by its span's factor — the concat output is the tightened
    clip, on which captions land already remapped.
    """
    out_w, out_h = ratio_dimensions(request.ratio)
    segments = request.crop_path.segments
    if not segments:
        raise ExportError("The crop path has no segments.")

    # (source interval, speed) pairs to render, in output order.
    intervals: list[tighten.Span]
    if plan is not None:
        intervals = _tighten_spans(request, plan)
        if not intervals:
            raise ExportError("Tightening produced no render spans for this clip.")
    else:
        # Untightened: one 1x span per crop segment, preserving the historical
        # multi-segment trim/concat shape.
        intervals = [
            tighten.Span(
                max(segment.start_s, request.start_s),
                min(segment.end_s, request.end_s),
                1.0,
            )
            for segment in segments
            if min(segment.end_s, request.end_s) > max(segment.start_s, request.start_s)
        ]
        if not intervals:
            intervals = [tighten.Span(request.start_s, request.end_s, 1.0)]

    # Render spans are source-absolute; the trim filters and the crop keyframes
    # need clip-relative times (the input is -ss seeked to the clip start).
    def segment_for(span: tighten.Span) -> tuple[CropSegment, float, float]:
        """The covering crop segment plus the span's clip-relative bounds."""
        for segment in segments:
            seg_start = segment.start_s + request.start_s
            seg_end = segment.end_s + request.start_s
            if seg_start <= span.start_s < seg_end:
                return segment, span.start_s - request.start_s, span.end_s - request.start_s
        return (
            segments[-1],
            span.start_s - request.start_s,
            span.end_s - request.start_s,
        )

    parts: list[str] = []
    labels: list[str] = []

    for index, span in enumerate(intervals):
        label = f"v{index}"
        labels.append(f"[{label}]")

        rel_start = span.start_s - request.start_s
        rel_end = span.end_s - request.start_s

        source_label = f"[s{index}]"
        parts.append(
            f"[0:v]trim=start={rel_start:.4f}:end={rel_end:.4f},"
            f"setpts=(PTS-STARTPTS)/{span.speed:.6f}{source_label}"
        )

        segment, k_start, k_end = segment_for(span)
        segment = _rebased_segment(segment, k_start, k_end)

        if segment.fit:
            parts.extend(_fit_chain(source_label, index, out_w, out_h))
            continue

        if segment.headroom >= HEADROOM_ENGAGE:
            # Tight close-up whose crown overflows the top edge: shrink the
            # sharp content and expose a blurred pad band at the top (drawn
            # from the same footage) so the whole head stays visible. The pad
            # fraction scales with the overflow; the engagement floor stops a
            # one-off detection flicker from padding a whole segment.
            parts.extend(
                _headroom_chain(
                    source_label, index, max(segment.headroom, 0.05), out_w, out_h
                )
            )
            continue

        # HEADROOM_ENGAGE keeps the two stages decoupled: reframe emits
        # headroom=0.0 (or None in old crop JSONs) for "ordinary crop".

        chain = [segment_crop_filter(segment)]
        if segment.zoom > 0:
            chain.append(_zoom_filter(segment.zoom, out_w, out_h))
        chain.append(f"scale={out_w}:{out_h}:flags=lanczos")
        chain.append("setsar=1,format=yuv420p")

        parts.append(f"{source_label}{','.join(chain)}[{label}]")

    if len(intervals) == 1:
        current = "[v0]"
    else:
        parts.append(f"{''.join(labels)}concat=n={len(intervals)}:v=1:a=0[vcat]")
        current = "[vcat]"

    grade = grade_filters(request.color_grade)
    if grade:
        # Grade the footage before captions are burned in, so the text stays
        # un-graded and maximally legible on top of the treated image.
        parts.append(f"{current}{grade}[vgrade]")
        current = "[vgrade]"

    punch_chain = _punch_filter(request, plan, out_w, out_h)
    if punch_chain:
        # Punch-ins run on the assembled clip so the move reads against the
        # finished frame, before captions — text stays static and legible.
        parts.append(f"{current}{punch_chain}[vpunch]")
        current = "[vpunch]"

    if request.burn_captions and subtitle_name is not None:
        # Bare relative names — ffmpeg runs with its cwd set to the render
        # workspace, so there is nothing here that needs escaping.
        parts.append(f"{current}ass=filename={subtitle_name}:fontsdir={fonts_name}[vout]")
    else:
        parts.append(f"{current}null[vout]")

    return ";".join(parts)


def _fit_chain(source_label: str, index: int, out_w: int, out_h: int) -> list[str]:
    """Fit the whole frame into the output over a blurred copy of itself.

    Used when the subjects are spread wider than any crop can hold. Plain black
    bars would be honest but look cheap on a phone; a blurred fill is what
    viewers are used to seeing and keeps the frame full-bleed.
    """
    return [
        f"{source_label}split=2[fitbg{index}][fitfg{index}]",
        (
            f"[fitbg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=24:2[bg{index}]"
        ),
        (
            f"[fitfg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos[fg{index}]"
        ),
        (f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v{index}]"),
    ]


def _headroom_chain(
    source_label: str, index: int, pad: float, out_w: int, out_h: int
) -> list[str]:
    """Shrink the content and pad the top band with a blurred copy of itself.

    The input is the segment's full-height crop — the case where no ``y``
    shift can free headroom, because the window already rides the source's top
    edge. Content is scaled to ``1 - pad`` of the output size and pinned to
    the BOTTOM of the frame (that is where a speaker's chest sits), so the
    exposed band at the top becomes usable headroom. The pad is drawn from a
    blurred copy of the same crop, so it reads as an extension of the scene —
    the standard treatment for close-ups that no reframing can fix.
    """
    content_h = max(2, round(out_h * (1.0 - pad)))
    content_w = max(2, round(out_w * (1.0 - pad)))
    return [
        f"{source_label}split=2[hrbg{index}][hrfg{index}]",
        (
            f"[hrbg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=20:2[hrb{index}]"
        ),
        f"[hrfg{index}]scale={content_w}:{content_h}:flags=lanczos[hrs{index}]",
        (
            f"[hrb{index}][hrs{index}]overlay=x=(W-w)/2:y=H-h,"
            f"scale={out_w}:{out_h}:flags=lanczos,setsar=1,format=yuv420p[v{index}]"
        ),
    ]


def _punch_times(
    punches: list[punch_module.Punch],
    plan: tighten.TightenPlan | None,
    *,
    start_s: float,
    output_end_s: float,
) -> list[tuple[float, float, float, float, float]]:
    """Punches lifted onto the output timeline.

    Returns ``(start, ease_end, hold_end, end, zoom)`` per punch. Source-
    absolute word times are lifted to clip-relative, remapped through the
    tighten plan when one is active, and clamped to the rendered output. A
    punch compressed below :data:`punch.MIN_OUTPUT_DURATION_S` of output time
    — mostly tightened away by silence speeding — is dropped rather than
    flickered, and its eased phases scale with whatever compression it did
    survive so the move stays smooth inside a shorter window.
    """
    lifted: list[tuple[float, float, float, float, float]] = []
    for punch in punches:
        rel_start = punch.start_s - start_s
        rel_end = punch.end_s - start_s
        if plan is not None:
            out_start = plan.remap(plan.start_s + rel_start)
            out_end = plan.remap(plan.start_s + rel_end)
        else:
            out_start, out_end = rel_start, rel_end
        out_end = min(out_end, output_end_s)
        if out_end - out_start < punch_module.MIN_OUTPUT_DURATION_S:
            continue
        # Scale the eased phases with the compression so the shape of the
        # move survives: a punch that lost 20% of its window eases 20% faster.
        scale = (out_end - out_start) / punch.duration_s
        lifted.append(
            (
                out_start,
                out_start + (punch.zoom_in_end_s - punch.start_s) * scale,
                out_start + (punch.hold_end_s - punch.start_s) * scale,
                out_end,
                punch.zoom,
            )
        )
    return lifted


def _punch_filter(
    request: ExportRequest,
    plan: tighten.TightenPlan | None,
    out_w: int,
    out_h: int,
) -> str:
    """The punch-in zoompan chain for this clip, or '' for none.

    One zoompan renders every punch as a piecewise ``z`` expression over the
    frame counter (``on``): smoothstep ease-in, hold at peak, smoothstep ease-
    back, 1.0 elsewhere. Smoothstep's zero-velocity endpoints are what make
    the move read as a camera push rather than a jump. Centering ``x``/``y``
    on the frame keeps the zoom framed on the subject instead of drifting.
    """
    punches = request.punches or []
    if not punches:
        return ""

    output_end_s = plan.output_duration_s if plan is not None else request.duration_s
    times = _punch_times(punches, plan, start_s=request.start_s, output_end_s=output_end_s)
    if not times:
        return ""

    fps = PUNCH_FPS

    def smoothstep(u: str) -> str:
        return f"(3*pow({u},2)-2*pow({u},3))"

    # One nested-if branch per punch, built in reverse so each else-branch is
    # the already-built expression for later time. Every phase boundary is a
    # frame count: zoompan's ``on`` increments once per output frame.
    expression = "1"
    for out_start, ease_end, hold_end, out_end, zoom in reversed(times):
        in_frames = max(1, round((ease_end - out_start) * fps))
        hold_frames = max(1, round((hold_end - ease_end) * fps))
        out_frames = max(1, round((out_end - hold_end) * fps))
        start_frame = out_start * fps
        end_frame = out_end * fps
        peak = 1.0 + zoom

        expression = (
            f"if(lt(on,{start_frame + in_frames:.0f}),"
            f"1+{zoom:.4f}*{smoothstep(f'(on-{start_frame:.3f})/{in_frames}')},"
            f"if(lt(on,{start_frame + in_frames + hold_frames:.0f}),"
            f"{peak:.4f},"
            f"if(lt(on,{end_frame:.0f}),"
            f"1+{zoom:.4f}*(1-"
            f"{smoothstep(f'(on-{start_frame + in_frames + hold_frames:.3f})/{out_frames}')}),"
            f"{expression})))"
        )

    return (
        f"fps={fps},zoompan=z='{expression}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={out_w}x{out_h}:fps={fps}"
    )


def _zoom_filter(zoom: float, out_w: int, out_h: int) -> str:
    """A slow push-in over the segment.

    Uses ``zoompan``, which is the only filter that can change apparent scale
    while holding a fixed output size. Off by default — at very small
    increments it can step visibly, and a locked frame beats a stuttering one.
    """
    end_zoom = 1.0 + zoom
    return f"zoompan=z='min(zoom+{zoom / 240:.6f},{end_zoom:.4f})':d=1:s={out_w}x{out_h}:fps=30"


def build_audio_filtergraph(settings: ExportSettings) -> str:
    return f"loudnorm=I={settings.loudness_lufs}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}"


def build_tightened_audio_filtergraph(
    request: ExportRequest,
    plan: tighten.TightenPlan,
    settings: ExportSettings,
) -> str:
    """Per-span ``atempo`` chains followed by the ordinary loudness stage.

    Span boundaries are converted to clip-relative times, matching the seeked
    input the video chain trims from — identical boundaries on both streams is
    what keeps them in sync frame for frame.
    """
    spans = _tighten_spans(request, plan)
    if not spans:
        return build_audio_filtergraph(settings)

    parts: list[str] = []
    labels: list[str] = []
    for index, span in enumerate(spans):
        label = f"a{index}"
        labels.append(f"[{label}]")
        rel_start = span.start_s - request.start_s
        rel_end = span.end_s - request.start_s
        chain = [
            f"atrim=start={rel_start:.4f}:end={rel_end:.4f}",
            "asetpts=PTS-STARTPTS",
        ]
        if span.speed != 1.0:
            chain.append(tighten.atempo_chain(span.speed))
        parts.append(f"[0:a]{','.join(chain)}[{label}]")

    if len(spans) == 1:
        current = "[a0]"
    else:
        parts.append(f"{''.join(labels)}concat=n={len(spans)}:v=0:a=1[acat]")
        current = "[acat]"

    parts.append(f"{current}{build_audio_filtergraph(settings)}[aout]")
    return ";".join(parts)


def encoder_args(settings: ExportSettings) -> list[str]:
    """Pick the video encoder and its quality settings.

    NVENC is materially faster where available. Its quality knob is ``-cq``
    rather than ``-crf``, and the two scales are close enough that reusing the
    configured value keeps output consistent between machines.

    x264 uses ``slow`` rather than ``medium``: clips are short (about a
    minute), so the ~2x encode time is acceptable, and the same CRF buys
    visibly fewer artefacts on fine detail — exactly what a downscaled 4K
    crop is full of. NVENC steps p5 → p4 for the same reason.
    """
    if settings.prefer_hardware_encoder and report().ffmpeg.nvenc_works:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            str(settings.crf),
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(settings.crf),
        "-profile:v",
        "high",
    ]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def export_clip(
    request: ExportRequest,
    *,
    work_dir: Path,
    settings: ExportSettings | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Render one clip to an MP4 and return the output path.

    Preconditions:
        request.source exists; request.crop_path segments span the clip duration
        in clip-relative seconds.
    """
    settings = settings or ExportSettings()
    out_w, out_h = ratio_dimensions(request.ratio)

    plan = _plan_for_request(request, settings)

    # Each clip renders in its own workspace so concurrent exports can't collide
    # on the shared `captions.ass` name that the relative-path scheme requires.
    workspace = work_dir / request.destination.stem
    workspace.mkdir(parents=True, exist_ok=True)

    subtitle_name: str | None = None
    fonts_name = "fonts"
    render_cwd: Path | None = None

    if request.burn_captions and request.words:
        words = _remap_words(
            request,
            plan,
            drop_fillers=request.filler_word_cleanup,
        )
        ass_path = captions_module.write_ass(
            workspace / "captions.ass",
            words,
            request.style,
            width=out_w,
            height=out_h,
            position=request.caption_position,
            primary_override=request.primary_color,
        )
        render_cwd, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(
            ass_path, captions_module.FONT_DIR
        )

    filtergraph = build_video_filtergraph(
        request, subtitle_name=subtitle_name, fonts_name=fonts_name, plan=plan
    )

    # Output timeline length: the tightened duration when tightening, the
    # clip's own duration otherwise. Drives ffmpeg progress reporting.
    output_duration = (
        plan.output_duration_s if plan is not None else request.duration_s
    )

    request.destination.parent.mkdir(parents=True, exist_ok=True)

    # Input and output paths are ordinary argv arguments, so they stay absolute
    # and need no escaping regardless of the working directory.
    args = [
        # Seeking before -i decodes from the nearest keyframe and is far faster
        # than an output-side seek; modern ffmpeg keeps it frame-accurate.
        "-ss",
        f"{request.start_s:.4f}",
        "-t",
        f"{request.duration_s:.4f}",
        "-i",
        str(request.source),
        "-filter_complex",
    ]

    if plan is not None:
        # Tightened audio rides in the SAME filter_complex: per-span atrim +
        # atempo + concat alongside the video chain, so the streams cannot
        # drift apart. One graph, one -filter_complex.
        full_graph = ";".join(
            [filtergraph, build_tightened_audio_filtergraph(request, plan, settings)]
        )
        args.append(full_graph)
        args += ["-map", "[vout]", "-map", "[aout]"]
    else:
        args.append(filtergraph)
        args += ["-map", "[vout]", "-map", "0:a?", "-af", build_audio_filtergraph(settings)]

    args += [
        *encoder_args(settings),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-movflags",
        "+faststart",
        str(request.destination),
    ]

    try:
        ffmpeg.run(
            args,
            total_duration_s=output_duration,
            on_progress=on_progress,
            cancelled=cancelled,
            cwd=render_cwd,
        )
    except ffmpeg.FFmpegError as exc:
        raise ExportError(f"Could not render {request.destination.name}.\n{exc}") from exc

    if not request.destination.exists() or request.destination.stat().st_size == 0:
        raise ExportError(f"ffmpeg reported success but {request.destination.name} is empty.")

    if settings.write_srt and request.words:
        captions_module.write_srt(
            request.destination.with_suffix(".srt"),
            captions_module.drop_filler_words(
                _remap_words(request, plan),
                enabled=request.filler_word_cleanup,
            ),
        )

    saved = plan.saved_s if plan is not None else 0.0
    log.info(
        "Exported %s (%.1fs, %s%s)",
        request.destination.name,
        output_duration,
        request.ratio,
        f", tightened {saved:.1f}s of silence" if saved > 0.05 else "",
    )
    return request.destination


def _remap_words(
    request: ExportRequest,
    plan: tighten.TightenPlan | None,
    *,
    drop_fillers: bool = False,
) -> list[Word]:
    """Words on the clip-relative output timeline, ready for the subtitle file.

    The plan's remap maps source-absolute times straight onto the tightened,
    clip-relative output (``t <= plan.start_s`` becomes ``t - plan.start_s``),
    and without a plan the same shift is applied by hand — either way the
    result is already clip-relative, so the ``write_ass``/``write_srt`` calls
    below pass ``time_offset_s=0``. Subtracting the clip start a second time
    there clamped every caption to a zero-length event and burned nothing.
    Fillers still drop when ``drop_fillers`` is set — the caption list
    shrinks, the timeline does not.
    """
    if plan is None:
        shifted = [
            Word(
                text=word.text,
                start=max(0.0, word.start - request.start_s),
                end=max(1e-3, word.end - request.start_s),
                speaker=word.speaker,
            )
            for word in request.words
        ]
        return captions_module.drop_filler_words(shifted, enabled=drop_fillers)
    remapped = []
    for word in request.words:
        remapped.append(
            Word(
                text=word.text,
                start=plan.remap(word.start),
                end=max(plan.remap(word.start) + 1e-3, plan.remap(word.end)),
                speaker=word.speaker,
            )
        )
    return captions_module.drop_filler_words(remapped, enabled=drop_fillers)

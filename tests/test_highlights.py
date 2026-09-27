"""Highlight detection: window aggregation and failure reporting.

When every window errors, the job must fail with the providers' real errors —
not the misleading "no clips found" answer reserved for a model that actually
answered and found nothing.
"""

from __future__ import annotations

import pytest
from autoclip.pipeline import highlights
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import (
    ClipCandidate,
    ClipCandidates,
    DetectionConfig,
    LLMProvider,
    ProviderError,
    ProviderStatus,
)


def make_transcript(word_count: int = 60) -> Transcript:
    return Transcript(
        words=[Word(text=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(word_count)]
    )


class FlakyProvider(LLMProvider):
    """Returns queued candidates until the script runs dry, then errors."""

    name = "flaky"
    requires_key = False

    def __init__(self, responses: list[list[ClipCandidate]], error: str) -> None:
        super().__init__("flaky-model")
        self.responses = list(responses)
        self.error = error

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        raise AssertionError("detect_highlights is overridden; _complete is unused")

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)

    async def detect_highlights(self, window, config) -> ClipCandidates:
        if self.responses:
            return ClipCandidates(clips=self.responses.pop(0))
        raise ProviderError(self.error, provider=self.name)


def candidate(start: int, end: int) -> ClipCandidate:
    return ClipCandidate(
        start_word_index=start, end_word_index=end, title="T", score=80, reason="R"
    )


class TestEffectiveMaxClips:
    """The clip budget scales with video length; the setting is a ceiling."""

    def test_short_video_gets_the_short_budget(self) -> None:
        # 610 words at 0.5s each ~ 5 minutes of speech.
        assert highlights.effective_max_clips(make_transcript(610), 10) == 4

    def test_long_video_gets_the_full_budget(self) -> None:
        # 3650 words at 0.5s each ~ 30.4 minutes.
        assert highlights.effective_max_clips(make_transcript(3650), 10) == 10

    def test_just_under_the_threshold_is_still_short(self) -> None:
        # 3600 words ~ 29.9 minutes: under the 30-minute mark.
        assert highlights.effective_max_clips(make_transcript(3600), 10) == 4

    def test_configured_limit_is_a_ceiling_not_a_floor(self) -> None:
        long_video = make_transcript(3650)
        short_video = make_transcript(610)

        assert highlights.effective_max_clips(long_video, 3) == 3
        assert highlights.effective_max_clips(short_video, 3) == 3

    async def test_detection_truncates_to_the_duration_scaled_budget(self) -> None:
        # A model that proposes six candidates for a five-minute video gets
        # cut to the short-video budget of four.
        provider = FlakyProvider(
            [[candidate(i, i + 8) for i in range(0, 48, 8)]], error="unused"
        )

        clips = await highlights.detect(
            make_transcript(610),
            provider,
            DetectionConfig(min_duration_s=1.0, max_duration_s=60.0, max_clips=10),
            job_id="j1",
        )

        assert len(clips) == 4


class TestWindowFailureReporting:
    async def test_all_windows_failing_reports_the_real_errors(self) -> None:
        provider = FlakyProvider([], error="400 json_validate_failed: output truncated")

        with pytest.raises(highlights.HighlightError) as excinfo:
            await highlights.detect(
                # Long enough to split into two overlapping windows.
                make_transcript(1200),
                provider,
                DetectionConfig(min_duration_s=1.0, max_duration_s=60.0),
                job_id="j1",
            )

        message = str(excinfo.value)
        assert "all 2 window(s)" in message
        assert "json_validate_failed" in message

    async def test_error_lists_the_failed_word_ranges(self) -> None:
        provider = FlakyProvider([], error="boom")

        with pytest.raises(highlights.HighlightError) as excinfo:
            await highlights.detect(
                make_transcript(),
                provider,
                DetectionConfig(min_duration_s=1.0, max_duration_s=60.0),
                job_id="j1",
            )

        assert "words 0-" in str(excinfo.value)

    async def test_partial_failures_still_produce_clips(self) -> None:
        # First window answers, second fails: the job succeeds on what came back.
        provider = FlakyProvider([[candidate(0, 40)]], error="window two exploded")
        transcript = make_transcript(60)

        clips = await highlights.detect(
            transcript,
            provider,
            DetectionConfig(min_duration_s=1.0, max_duration_s=60.0),
            job_id="j1",
        )

        assert len(clips) == 1

    async def test_clean_no_clips_answer_is_not_reported_as_a_failure(self) -> None:
        provider = FlakyProvider([[], []], error="should never be raised")

        with pytest.raises(highlights.HighlightError, match="No clips were found"):
            await highlights.detect(
                make_transcript(),
                provider,
                DetectionConfig(min_duration_s=1.0, max_duration_s=60.0),
                job_id="j1",
            )

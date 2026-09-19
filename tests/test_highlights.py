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

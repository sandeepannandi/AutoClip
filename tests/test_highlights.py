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


def make_sentence_transcript(word_count: int = 200, words_per_sentence: int = 10) -> Transcript:
    """Words of 0.5s each with a sentence end every ``words_per_sentence`` words."""
    words = []
    for i in range(word_count):
        text = f"w{i}." if (i + 1) % words_per_sentence == 0 else f"w{i}"
        words.append(Word(text=text, start=i * 0.5, end=i * 0.5 + 0.4))
    return Transcript(words=words)


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


def candidate(start: int, end: int, **overrides) -> ClipCandidate:
    defaults = {
        "start_word_index": start,
        "end_word_index": end,
        "title": "T",
        "score": 80,
        "reason": "R",
    }
    defaults.update(overrides)
    return ClipCandidate(**defaults)


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


class TestEngagementRanking:
    """Ranking blends overall score with the model's hook-strength read."""

    def test_engagement_key_blends_score_and_hook(self) -> None:
        assert highlights.engagement_key(90, 40) == pytest.approx(
            highlights.QUALITY_WEIGHT * 90 + highlights.ENGAGEMENT_WEIGHT * 40
        )
        # A strong hook can outrank a better overall score.
        assert highlights.engagement_key(85, 95) > highlights.engagement_key(88, 40)

    async def test_equal_scores_rank_by_hook_strength(self) -> None:
        weak_hook = candidate(0, 40, title="Weak", score=80, hook_strength=40)
        strong_hook = candidate(50, 100, title="Strong", score=80, hook_strength=90)
        provider = FlakyProvider([[weak_hook, strong_hook]], error="unused")

        clips = await highlights.detect(
            make_sentence_transcript(120),
            provider,
            DetectionConfig(min_duration_s=1.0, max_duration_s=60.0),
            job_id="j1",
        )

        assert [c.title for c in clips] == ["Strong", "Weak"]

    def test_dedupe_keeps_the_higher_engagement_clip(self) -> None:
        # Overlapping spans are the same clip; the stronger hook survives even
        # though the other candidate has a higher overall score.
        weaker_hook = candidate(0, 60, title="Weaker hook", score=88, hook_strength=20)
        stronger_hook = candidate(10, 70, title="Stronger hook", score=80, hook_strength=95)

        kept = highlights.dedupe([weaker_hook, stronger_hook])

        assert [c.title for c in kept] == ["Stronger hook"]

    async def test_hook_strength_survives_to_the_clip(self) -> None:
        provider = FlakyProvider([[candidate(0, 40, hook_strength=77)]], error="unused")

        clips = await highlights.detect(
            make_sentence_transcript(60),
            provider,
            DetectionConfig(min_duration_s=1.0, max_duration_s=60.0),
            job_id="j1",
        )

        assert clips[0].hook_strength == 77


class TestHookStart:
    """The clip opens on the gripping line, not the flat lead-in."""

    async def test_hook_word_index_is_used_when_the_model_provides_one(self) -> None:
        # Candidate spans words 0-119 (60s). The model says the hook begins at
        # word 30 (a sentence start, 15s in) — the clip must open there, not
        # at word 0 where the plain snap lands.
        provider = FlakyProvider(
            [[candidate(0, 119, hook="w30", hook_word_index=30)]], error="unused"
        )

        clips = await highlights.detect(
            make_sentence_transcript(200),
            provider,
            DetectionConfig(min_duration_s=30.0, max_duration_s=70.0),
            job_id="j1",
        )

        assert clips[0].start_word == 30

    async def test_hook_quote_match_snaps_without_an_index(self) -> None:
        # v1-style response: verbatim quote but no index. The proposed start
        # is word 32 (mid-sentence); the plain snap prefers the earlier
        # sentence start at 30, but the hook quote sits at word 40 — within
        # the verification window — so the clip must open there instead.
        provider = FlakyProvider(
            [[candidate(32, 151, hook="w40.", hook_word_index=None)]], error="unused"
        )

        clips = await highlights.detect(
            make_sentence_transcript(200),
            provider,
            DetectionConfig(min_duration_s=30.0, max_duration_s=70.0),
            job_id="j1",
        )

        assert clips[0].start_word == 40

    async def test_hallucinated_hook_is_ignored(self) -> None:
        # The model quotes words that appear nowhere near the start; the
        # boundary falls back to the plain sentence snap (word 0) instead of
        # snapping to something bogus.
        provider = FlakyProvider(
            [[candidate(0, 119, hook="the moon is cheese", hook_word_index=None)]],
            error="unused",
        )

        clips = await highlights.detect(
            make_sentence_transcript(200),
            provider,
            DetectionConfig(min_duration_s=30.0, max_duration_s=70.0),
            job_id="j1",
        )

        assert clips[0].start_word == 0

    async def test_hook_outside_the_candidate_span_is_ignored(self) -> None:
        # A hook index before the clip's own start is the model pointing
        # elsewhere; the plain snap must win.
        provider = FlakyProvider(
            [[candidate(30, 149, hook="w10", hook_word_index=10)]], error="unused"
        )

        clips = await highlights.detect(
            make_sentence_transcript(200),
            provider,
            DetectionConfig(min_duration_s=30.0, max_duration_s=70.0),
            job_id="j1",
        )

        # Plain snap of word 30 lands on the sentence start 30 sits in.
        assert clips[0].start_word == 30

    async def test_hook_is_dropped_when_it_breaks_the_duration_band(self) -> None:
        # Snapping to the hook would leave 27s of clip — under the 30s floor.
        # The plain snap (word 0, 60s) must win over dropping the clip.
        provider = FlakyProvider(
            [[candidate(0, 119, hook="w100", hook_word_index=100)]], error="unused"
        )

        clips = await highlights.detect(
            make_sentence_transcript(200),
            provider,
            DetectionConfig(min_duration_s=30.0, max_duration_s=70.0),
            job_id="j1",
        )

        assert clips[0].start_word == 0
        assert clips[0].duration_s == pytest.approx(60.0, abs=1.0)
"""Clip boundary refinement — sentence snapping, duration clamping, silence alignment."""

from __future__ import annotations

import pytest
from autoclip.pipeline import boundaries
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.transcript import Transcript, Word


def make_transcript(sentences: list[list[str]], *, word_duration: float = 0.5) -> Transcript:
    """Build a transcript with evenly spaced words and clean sentence boundaries."""
    words: list[Word] = []
    t = 0.0
    for sentence in sentences:
        for index, text in enumerate(sentence):
            token = text + ("." if index == len(sentence) - 1 else "")
            words.append(Word(text=token, start=t, end=t + word_duration * 0.8))
            t += word_duration
    return Transcript(words=words)


@pytest.fixture
def transcript() -> Transcript:
    # Ten sentences of ten words: 100 words, 0.5s each, 50 seconds total.
    return make_transcript([[f"w{s}{i}" for i in range(10)] for s in range(10)])


class TestSentenceSnapping:
    def test_start_snaps_to_the_first_word_of_a_sentence(self, transcript: Transcript) -> None:
        # Index 13 is mid-second-sentence; sentence starts are 0, 10, 20, ...
        assert boundaries.snap_start_to_sentence(transcript, 13) == 10

    def test_start_prefers_moving_earlier_on_a_tie(self, transcript: Transcript) -> None:
        # Index 15 is equidistant from 10 and 20; starting wide beats cutting the hook.
        assert boundaries.snap_start_to_sentence(transcript, 15) == 10

    def test_end_snaps_to_the_last_word_of_a_sentence(self, transcript: Transcript) -> None:
        assert boundaries.snap_end_to_sentence(transcript, 27) == 29

    def test_end_prefers_moving_later_on_a_tie(self, transcript: Transcript) -> None:
        # Ending on the complete thought rather than truncating it.
        assert boundaries.snap_end_to_sentence(transcript, 24) == 29

    def test_index_zero_always_starts_a_sentence(self, transcript: Transcript) -> None:
        assert boundaries.snap_start_to_sentence(transcript, 0) == 0

    def test_final_index_always_ends_a_sentence(self, transcript: Transcript) -> None:
        last = len(transcript.words) - 1
        assert boundaries.snap_end_to_sentence(transcript, last) == last

    def test_unpunctuated_transcript_leaves_the_index_alone(self) -> None:
        transcript = Transcript(
            words=[Word(text=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(60)]
        )

        assert boundaries.snap_start_to_sentence(transcript, 30) == 30


class TestRefine:
    def test_produces_a_boundary_inside_the_duration_range(self, transcript: Transcript) -> None:
        result = boundaries.refine(transcript, 10, 69, min_duration_s=20.0, max_duration_s=90.0)

        assert result is not None
        assert 20.0 <= result.duration_s <= 90.0
        assert result.start_word == 10

    def test_over_long_range_is_trimmed_to_a_sentence_end(self, transcript: Transcript) -> None:
        # Words 0-99 span 50s; ask for a 10s ceiling.
        result = boundaries.refine(transcript, 0, 99, min_duration_s=5.0, max_duration_s=10.0)

        assert result is not None
        assert result.duration_s <= 10.0
        assert transcript.words[result.end_word].ends_sentence

    def test_too_short_range_is_extended(self, transcript: Transcript) -> None:
        # Words 0-9 span 5s; require at least 12s.
        result = boundaries.refine(transcript, 0, 9, min_duration_s=12.0, max_duration_s=40.0)

        assert result is not None
        assert result.duration_s >= 12.0

    def test_returns_none_when_no_valid_length_exists(self, transcript: Transcript) -> None:
        # The whole transcript is 50s; a 200s floor cannot be met.
        assert (
            boundaries.refine(transcript, 0, 99, min_duration_s=200.0, max_duration_s=300.0) is None
        )

    def test_inverted_range_is_rejected(self, transcript: Transcript) -> None:
        assert boundaries.refine(transcript, 50, 50) is None

    def test_out_of_range_indices_are_clamped(self, transcript: Transcript) -> None:
        result = boundaries.refine(transcript, -5, 9_999, min_duration_s=5.0, max_duration_s=90.0)

        assert result is not None
        assert 0 <= result.start_word < len(transcript.words)
        assert 0 <= result.end_word < len(transcript.words)

    def test_empty_transcript_returns_none(self) -> None:
        assert boundaries.refine(Transcript(), 0, 10) is None

    def test_start_never_goes_negative(self, transcript: Transcript) -> None:
        result = boundaries.refine(transcript, 0, 59, min_duration_s=5.0, max_duration_s=90.0)

        assert result is not None
        assert result.start_s >= 0.0


class TestDurationTarget:
    """A too-short candidate extends toward the band's middle, not its floor.

    Stopping at the first sentence end past the minimum makes every clip
    cluster just above the floor; the walk should continue while the next
    sentence end lands nearer the target in the middle of the band.
    """

    def test_short_candidate_extends_past_the_minimum_toward_the_middle(
        self, transcript: Transcript
    ) -> None:
        # Sentence ends from word 0 land at 4.9s, 9.9s, ..., 24.9s, ... With a
        # 20-62 band the first qualifying end is 24.9s, but 34.9/39.9s sit
        # nearer the 43.1s target — the clip must not stop at 24.9s.
        result = boundaries.refine(transcript, 0, 29, min_duration_s=20.0, max_duration_s=62.0)

        assert result is not None
        # 44.9s of speech + the 0.35s fallback tail.
        assert result.duration_s == pytest.approx(45.25)

    def test_walk_stops_once_the_target_is_reached(self, transcript: Transcript) -> None:
        # 20-40 band: target 31s. Ends run 24.9, 29.9, 34.9 — the clip takes
        # 34.9s, the first end at or past the target, not the last that fits.
        result = boundaries.refine(transcript, 0, 29, min_duration_s=20.0, max_duration_s=40.0)

        assert result is not None
        assert result.duration_s == pytest.approx(34.9 + 0.35)

    def test_target_never_pushes_a_clip_over_the_ceiling(self, transcript: Transcript) -> None:
        # 20-30 band: target 25.5s. The walk takes 29.9s — inside the band —
        # rather than the 24.9s floor-hugger; the tail pushes it to 30.25s and
        # the post-alignment ceiling check clamps it back to exactly 30s.
        result = boundaries.refine(transcript, 0, 29, min_duration_s=20.0, max_duration_s=30.0)

        assert result is not None
        assert result.duration_s == pytest.approx(30.0)

    def test_default_band_yields_a_clip_near_the_50_60s_spot(self, transcript: Transcript) -> None:
        # The 50s fixture against the new 48-62 defaults: the whole transcript
        # becomes one clip sitting just inside the band.
        result = boundaries.refine(transcript, 0, 99)

        assert result is not None
        assert 48.0 <= result.duration_s <= 62.0

    def test_returns_none_when_no_candidate_can_reach_the_band(
        self, transcript: Transcript
    ) -> None:
        # Starting at word 30 only 35s of material remain — under the 48s floor.
        assert boundaries.refine(transcript, 30, 99) is None


class TestSilenceAlignment:
    def test_start_lands_inside_a_nearby_silence(self) -> None:
        # Speech begins at 10.0; silence runs 9.4-10.0.
        silences = [Silence(start=9.4, end=10.0)]

        aligned = boundaries.align_start(10.0, silences)

        assert 9.4 < aligned < 10.0
        assert aligned == pytest.approx(10.0 - boundaries.SILENCE_LEAD_S)

    def test_start_never_precedes_the_silence(self) -> None:
        # A very short silence must not be overrun by the lead-in.
        silences = [Silence(start=9.95, end=10.0)]

        aligned = boundaries.align_start(10.0, silences)

        assert aligned >= 9.95

    def test_end_lands_inside_the_following_silence(self) -> None:
        silences = [Silence(start=30.0, end=31.0)]

        aligned = boundaries.align_end(30.0, silences)

        assert 30.0 <= aligned < 31.0
        assert aligned == pytest.approx(30.0 + boundaries.SILENCE_TAIL_S)

    def test_end_never_overruns_the_silence(self) -> None:
        silences = [Silence(start=30.0, end=30.1)]

        aligned = boundaries.align_end(30.0, silences)

        assert aligned <= 30.1

    def test_falls_back_to_fixed_padding_without_silences(self) -> None:
        assert boundaries.align_start(10.0, []) == pytest.approx(10.0 - boundaries.FALLBACK_LEAD_S)
        assert boundaries.align_end(30.0, []) == pytest.approx(30.0 + boundaries.FALLBACK_TAIL_S)

    def test_distant_silences_are_ignored(self) -> None:
        # A silence five seconds away says nothing about this boundary.
        silences = [Silence(start=2.0, end=5.0)]

        assert boundaries.align_start(10.0, silences) == pytest.approx(
            10.0 - boundaries.FALLBACK_LEAD_S
        )

    def test_alignment_only_ever_widens_a_clip(self) -> None:
        silences = [Silence(start=9.0, end=10.0), Silence(start=30.0, end=31.0)]

        start = boundaries.align_start(10.0, silences)
        end = boundaries.align_end(30.0, silences)

        assert start <= 10.0
        assert end >= 30.0

    def test_refine_respects_the_ceiling_after_alignment(self, transcript: Transcript) -> None:
        # Generous silences either side would push the clip over the limit.
        silences = [Silence(start=0.0, end=5.0), Silence(start=20.0, end=26.0)]

        result = boundaries.refine(
            transcript, 10, 39, silences=silences, min_duration_s=5.0, max_duration_s=12.0
        )

        assert result is not None
        assert result.duration_s <= 12.0

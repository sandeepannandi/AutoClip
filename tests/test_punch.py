"""Punch-in zoom events built from the model's emphasis words."""

from __future__ import annotations

import pytest
from autoclip.pipeline import punch
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.tighten import build_plan
from autoclip.pipeline.transcript import Word

TEXT = "nobody tells you this but most agencies die at ten clients and here is why"


def _words(
    text: str = TEXT, start_s: float = 0.0, step: float = 0.5
) -> list[Word]:        return [
            Word(
                text=token,
                start=start_s + i * step,
                end=start_s + i * step + step - 0.05,
                speaker=None,
            )
            for i, token in enumerate(text.split())
        ]


class TestBuildPunches:
    def test_finds_the_phrase_and_times_the_move(self) -> None:
        punches = punch.build_punches(_words(), ["agencies die"])

        assert len(punches) == 1
        # "agencies" is word 6 of the sentence (0-indexed), at 6 * 0.5s.
        start = punches[0].start_s
        assert start == pytest.approx(3.0)
        assert punches[0].zoom_in_end_s == pytest.approx(start + punch.DEFAULT_EASE_IN_S)
        assert punches[0].hold_end_s == pytest.approx(
            start + punch.DEFAULT_EASE_IN_S + punch.DEFAULT_HOLD_S
        )
        assert punches[0].end_s == pytest.approx(
            start + punch.DEFAULT_EASE_IN_S + punch.DEFAULT_HOLD_S + punch.DEFAULT_EASE_OUT_S
        )
        assert punches[0].zoom == pytest.approx(punch.DEFAULT_ZOOM)

    def test_matching_ignores_case_and_punctuation(self) -> None:
        # The model quotes verbatim; Whisper's transcript may differ in casing
        # and trailing punctuation. Both directions must still match.
        assert punch.build_punches(_words(), ["Agencies Die,"])
        assert punch.build_punches(
            _words("Nobody tells you this but most AGENCIES DIE at ten clients."),
            ["agencies die"],
        )

    def test_unmatched_phrase_yields_nothing(self) -> None:
        assert punch.build_punches(_words(), ["quantum revenue streams"]) == []

    def test_empty_inputs_yield_nothing(self) -> None:
        assert punch.build_punches([], ["agencies die"]) == []
        assert punch.build_punches(_words(), []) == []

    def test_at_most_two_punches(self) -> None:
        words = _words("one two three four five six seven eight nine ten " * 4)
        punches = punch.build_punches(
            words, ["three four", "six seven", "one two", "eight nine"]
        )

        assert len(punches) <= punch.DEFAULT_MAX_PUNCHES

    def test_minimum_gap_between_punches(self) -> None:
        words = _words("one two three four five six seven eight nine ten " * 4)
        punches = punch.build_punches(words, ["three four", "five six"])

        assert len(punches) == 1  # too close together; the first wins

    def test_distant_phrases_both_get_punches(self) -> None:
        # Unique tokens: a repeating text would re-match "two three" later in
        # the clip, inside the min gap of the first punch.
        nato = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "\
            "kilo lima mike november oscar papa quebec romeo sierra tango "\
            "uniform victor whiskey xray yankee zulu"
        words = _words(nato)
        punches = punch.build_punches(words, ["bravo charlie", "sierra tango"])

        assert [p.start_s for p in punches] == sorted(p.start_s for p in punches)
        assert len(punches) == 2

    def test_hook_run_up_is_protected(self) -> None:
        # The phrase starts 0.25s in — inside the protection window. The punch
        # must not begin before the window ends.
        words = _words(start_s=0.0)
        punches = punch.build_punches(words, ["nobody tells"])

        assert punches
        assert punches[0].start_s >= punch.HOOK_PROTECTION_S

    def test_punch_that_would_overrun_the_clip_is_dropped(self) -> None:
        # "is why" starts at 6.5s; the 1.2s ease-back would spill past the
        # last word's end at 7.45s, so the punch is skipped.
        punches = punch.build_punches(_words(), ["is why"])

        assert punches == []

    def test_longest_phrase_wins_when_overlapping(self) -> None:
        words = _words()
        punches = punch.build_punches(words, ["agencies die", "most agencies die"])

        assert len(punches) == 1
        assert punches[0].start_s == pytest.approx(words[5].start)  # "most"


class TestTightenedTimeline:
    """Punches must land where the phrase is SEEN, not spoken in the source."""

    def test_punch_times_survive_remap(self) -> None:
        words = _words()  # 0..9s
        punches = punch.build_punches(words, ["agencies die"])
        assert punches

        # Tighten a 1.5s silence after the punch; the punch itself sits in
        # normal-speed speech, so its output times shift by the saved amount.
        plan = build_plan(0.0, 9.0, [Silence(start=6.0, end=7.5)])

        lifted_start = plan.remap(plan.start_s + punches[0].start_s)
        assert lifted_start == pytest.approx(punches[0].start_s, abs=0.01)

    def test_punch_inside_tightened_silence_is_dropped(self) -> None:
        # A punch anchored at 6.25s sits inside a 6.0-7.5s 4x span: compressed
        # below the minimum output duration, it must be dropped.
        plan = build_plan(0.0, 9.0, [Silence(start=6.0, end=7.5)])
        request_punches = [
            punch.Punch(
                start_s=6.25,
                zoom_in_end_s=6.55,
                hold_end_s=7.05,
                end_s=7.45,
                zoom=0.08,
            )
        ]

        from autoclip.pipeline import export as export_module

        lifted = export_module._punch_times(
            request_punches, plan, start_s=0.0, output_end_s=plan.output_duration_s
        )

        # 1.2s of source at 4x is 0.3s of output — above the floor, so it
        # survives but compressed to roughly a quarter of its length.
        assert lifted
        assert lifted[0][3] - lifted[0][0] == pytest.approx(0.3, abs=0.06)

    def test_punch_compressed_below_the_floor_is_dropped(self) -> None:
        from autoclip.pipeline import export as export_module

        tiny = [
            punch.Punch(start_s=6.0, zoom_in_end_s=6.1, hold_end_s=6.2, end_s=6.3, zoom=0.08)
        ]
        plan = build_plan(0.0, 9.0, [Silence(start=5.9, end=7.4)])

        lifted = export_module._punch_times(
            tiny, plan, start_s=0.0, output_end_s=plan.output_duration_s
        )

        assert lifted == []

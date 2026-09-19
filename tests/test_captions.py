"""Caption grouping and ASS generation."""

from __future__ import annotations

import pysubs2
import pytest
from autoclip.pipeline import captions
from autoclip.pipeline.transcript import Word


def words_from(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(text=t, start=s, end=e) for t, s, e in spec]


def evenly_spaced(texts: list[str], *, step: float = 0.4) -> list[Word]:
    return [
        Word(text=text, start=i * step, end=i * step + step * 0.8) for i, text in enumerate(texts)
    ]


class TestGrouping:
    def test_respects_the_word_ceiling(self) -> None:
        groups = captions.group_words(evenly_spaced([f"w{i}" for i in range(12)]), max_words=4)

        assert all(len(g.words) <= 4 for g in groups)
        assert sum(len(g.words) for g in groups) == 12

    def test_breaks_on_a_long_pause(self) -> None:
        words = words_from([("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 2.0, 2.3)])

        groups = captions.group_words(words, max_words=10, max_gap_s=0.4)

        assert len(groups) == 2
        assert groups[1].words[0].text == "three"

    def test_breaks_after_sentence_punctuation(self) -> None:
        words = words_from([("Hello.", 0.0, 0.3), ("Next", 0.4, 0.7), ("thing", 0.8, 1.0)])

        groups = captions.group_words(words, max_words=10)

        assert len(groups) == 2
        assert groups[0].text == "Hello."

    def test_trailing_sentence_end_does_not_make_an_empty_group(self) -> None:
        words = words_from([("Hello", 0.0, 0.3), ("world.", 0.4, 0.7)])

        groups = captions.group_words(words, max_words=10)

        assert len(groups) == 1
        assert all(g.words for g in groups)

    def test_every_word_appears_exactly_once(self) -> None:
        words = evenly_spaced([f"w{i}" for i in range(37)])

        grouped = [w for g in captions.group_words(words, max_words=4) for w in g.words]

        assert [w.text for w in grouped] == [w.text for w in words]

    def test_empty_input_gives_no_groups(self) -> None:
        assert captions.group_words([]) == []

    def test_group_times_span_its_words(self) -> None:
        groups = captions.group_words(evenly_spaced(["a", "b", "c"]), max_words=3)

        assert groups[0].start == 0.0
        assert groups[0].end == pytest.approx(0.8 + 0.32, abs=0.01)


class TestColour:
    def test_hex_converts_to_rgb(self) -> None:
        colour = captions.hex_to_ass("#FFE500")

        assert (colour.r, colour.g, colour.b) == (255, 229, 0)

    def test_leading_hash_is_optional(self) -> None:
        assert captions.hex_to_ass("FFFFFF").r == 255

    def test_bad_hex_raises(self) -> None:
        with pytest.raises(ValueError):
            captions.hex_to_ass("#FFF")


class TestPresets:
    def test_all_four_presets_exist(self) -> None:
        assert set(captions.PRESETS) == {
            "bold_pop",
            "karaoke_fill",
            "clean_lower",
            "boxed",
        }

    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_every_preset_ships_its_font(self, key: str) -> None:
        # A missing font file means libass silently substitutes, and the caption
        # looks nothing like the preview.
        assert captions.get_style(key).font_path.exists()

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown caption style"):
            captions.get_style("neon-explosion")

    def test_clean_lower_is_set_in_inter(self) -> None:
        # Inter is the whole reason this preset reads as professional rather than
        # as a shout; if the family name drifts, libass substitutes silently and
        # the caption stops matching the style list it was picked from.
        style = captions.get_style("clean_lower")

        assert (style.font, style.font_file) == ("Inter", "Inter-Variable.ttf")

    def test_clean_lower_is_the_lightest_preset(self) -> None:
        style = captions.get_style("clean_lower")

        # It has to stay the smallest and quietest preset, not just smaller than
        # whichever ones happen to exist today.
        assert style.size_ratio <= 0.045
        assert style.size_ratio == min(other.size_ratio for other in captions.PRESETS.values())
        assert style.outline_width <= 1.5
        assert style.shadow <= 0.5

    @pytest.mark.parametrize("key", ["bold_pop", "boxed"])
    def test_high_energy_presets_entrance_animate(self, key: str) -> None:
        assert captions.get_style(key).entrance is True

    @pytest.mark.parametrize("key", ["karaoke_fill", "clean_lower"])
    def test_calm_presets_do_not_entrance_animate(self, key: str) -> None:
        assert captions.get_style(key).entrance is False


class TestAssGeneration:
    @pytest.fixture
    def words(self) -> list[Word]:
        return evenly_spaced(["the", "quick", "brown", "fox", "jumps", "over."])

    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_every_preset_produces_events(self, key: str, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style(key), width=1080, height=1920)

        assert len(subs.events) > 0
        assert captions.STYLE_NAME in subs.styles

    def test_resolution_is_recorded(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert subs.info["PlayResX"] == "1080"
        assert subs.info["PlayResY"] == "1920"

    def test_font_size_scales_with_height(self, words: list[Word]) -> None:
        tall = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)
        square = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1080)

        assert (
            tall.styles[captions.STYLE_NAME].fontsize > square.styles[captions.STYLE_NAME].fontsize
        )

    def test_time_offset_rebases_to_clip_relative(self, words: list[Word]) -> None:
        offset = captions.build_ass(
            words,
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
            time_offset_s=100.0,
        )

        # Absolute times were 0-2.7s, so everything clamps to zero rather than
        # going negative.
        assert offset.events[0].start == 0

    def test_offset_preserves_relative_spacing(self) -> None:
        words = [Word(text=t, start=100.0 + i, end=100.5 + i) for i, t in enumerate("abc")]

        subs = captions.build_ass(
            words,
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
            time_offset_s=100.0,
        )

        assert subs.events[0].start == pytest.approx(0, abs=20)

    def test_all_caps_is_applied(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert "QUICK" in " ".join(e.text for e in subs.events)

    def test_clean_lower_preserves_case(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("clean_lower"), width=1080, height=1920)

        assert "quick" in " ".join(e.text for e in subs.events)

    def test_clean_lower_render_is_set_in_inter(self, words: list[Word]) -> None:
        # The preset is the only one in Inter; the generated styles are what
        # libass actually reads, so assert on those rather than the dataclass.
        subs = captions.build_ass(words, captions.get_style("clean_lower"), width=1080, height=1920)

        assert subs.styles[captions.STYLE_NAME].fontname == "Inter"

    def test_karaoke_emits_kf_tags(self, words: list[Word]) -> None:
        subs = captions.build_ass(
            words, captions.get_style("karaoke_fill"), width=1080, height=1920
        )

        assert any("\\kf" in event.text for event in subs.events)

    def test_bold_pop_emits_one_event_per_word(self, words: list[Word]) -> None:
        # Per-word highlighting has no per-glyph timeline in libass, so each
        # word needs its own event showing the whole group.
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert len(subs.events) == len(words)

    def test_clean_lower_emits_one_event_per_group(self, words: list[Word]) -> None:
        style = captions.get_style("clean_lower")
        groups = captions.group_words(words, max_words=style.max_words)

        subs = captions.build_ass(words, style, width=1080, height=1920)

        assert len(subs.events) == len(groups)

    def test_boxed_uses_an_opaque_border_style(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("boxed"), width=1080, height=1920)

        assert subs.styles[captions.STYLE_NAME].borderstyle == 3

    def test_bottom_placement_is_the_default(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("bold_pop"), width=1080, height=1920)

        assert subs.styles[captions.STYLE_NAME].alignment == pysubs2.Alignment.BOTTOM_CENTER
        assert subs.styles[captions.STYLE_NAME].marginv == round(1920 * 0.24)

    @pytest.mark.parametrize(
        ("position", "alignment", "marginv"),
        [
            ("bottom", pysubs2.Alignment.BOTTOM_CENTER, round(1920 * 0.24)),
            ("middle", pysubs2.Alignment.MIDDLE_CENTER, 0),
            ("top", pysubs2.Alignment.TOP_CENTER, round(1920 * 0.10)),
        ],
    )
    def test_position_parameter_places_captions(
        self, words: list[Word], position: str, alignment, marginv: int
    ) -> None:
        subs = captions.build_ass(
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            position=position,
        )

        ass_style = subs.styles[captions.STYLE_NAME]
        assert ass_style.alignment == alignment
        assert ass_style.marginv == marginv

    def test_bold_pop_first_word_entrances(self, words: list[Word]) -> None:
        style = captions.get_style("bold_pop")
        groups = captions.group_words(words, max_words=style.max_words)
        subs = captions.build_ass(words, style, width=1080, height=1920)

        assert "\\move(" in subs.events[0].text
        assert "\\fad(" in subs.events[0].text
        # Each caption group entrances once, on its first word; later word
        # events for the same group stay put so the line doesn't jitter.
        assert sum("\\move(" in event.text for event in subs.events) == len(groups)

    def test_clean_lower_emits_no_entrance_tags(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("clean_lower"), width=1080, height=1920)

        assert all(
            "\\move(" not in event.text and "\\fad(" not in event.text for event in subs.events
        )

    def test_events_never_start_before_zero(self, words: list[Word]) -> None:
        subs = captions.build_ass(
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            time_offset_s=1.0,
        )

        assert all(event.start >= 0 for event in subs.events)

    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_events_never_overlap(self, key: str, words: list[Word]) -> None:
        # Only one caption line may ever be on screen at a time, for every
        # preset and its animation mode.
        subs = captions.build_ass(words, captions.get_style(key), width=1080, height=1920)

        events = list(subs.events)
        assert all(next_.start >= prev.end for prev, next_ in zip(events, events[1:], strict=False))

    def test_overlapping_groups_are_serialised(self) -> None:
        # A sentence break lands mid-overlap (the next word starts before the
        # sentence-ending word finishes). The following group must wait for the
        # current line to clear rather than stacking a second line.
        words = words_from([("Hello.", 0.0, 1.2), ("world", 0.9, 1.6)])

        subs = captions.build_ass(words, captions.get_style("clean_lower"), width=1080, height=1920)

        first, second = list(subs.events)
        assert first.start == 0
        assert first.end == pysubs2.make_time(s=1.2)
        assert second.start == first.end
        assert second.end == pysubs2.make_time(s=1.6)

    def test_srt_cues_never_overlap(self, tmp_path) -> None:
        words = words_from([("Hello.", 0.0, 1.2), ("world", 0.9, 1.6)])

        path = captions.write_srt(tmp_path / "out.srt", words)

        cues = list(pysubs2.load(str(path), encoding="utf-8").events)
        assert all(next_.start >= prev.end for prev, next_ in zip(cues, cues[1:], strict=False))

    def test_written_file_is_valid_ass(self, words: list[Word], tmp_path) -> None:
        path = captions.write_ass(
            tmp_path / "out.ass",
            words,
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
        )

        reloaded = pysubs2.load(str(path), encoding="utf-8")

        assert len(reloaded.events) > 0

    def test_srt_sidecar_is_written(self, words: list[Word], tmp_path) -> None:
        path = captions.write_srt(tmp_path / "out.srt", words)

        assert path.exists()
        assert len(pysubs2.load(str(path), encoding="utf-8").events) > 0

    def test_primary_override_replaces_the_style_primary(self, words: list[Word]) -> None:
        subs = captions.build_ass(
            words,
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
            primary_override="#FFE500",
        )

        colour = subs.styles[captions.STYLE_NAME].primarycolor
        assert (colour.r, colour.g, colour.b) == (255, 229, 0)

    def test_without_override_the_preset_primary_is_emitted(self, words: list[Word]) -> None:
        subs = captions.build_ass(words, captions.get_style("clean_lower"), width=1080, height=1920)

        colour = subs.styles[captions.STYLE_NAME].primarycolor
        assert (colour.r, colour.g, colour.b) == (255, 255, 255)

    def test_secondary_stays_preset_controlled(self, words: list[Word]) -> None:
        # clean_lower has no accent, so the secondary colour falls back to the
        # preset primary, not the override — the override only replaces primary.
        style = captions.get_style("clean_lower")
        subs = captions.build_ass(words, style, width=1080, height=1920, primary_override="#FFE500")

        ass_style = subs.styles[captions.STYLE_NAME]
        assert ass_style.secondarycolor == captions.hex_to_ass(style.primary)

    def test_primary_override_never_mutates_the_preset(self, words: list[Word], tmp_path) -> None:
        style = captions.get_style("clean_lower")
        before = style.primary

        captions.build_ass(words, style, width=1080, height=1920, primary_override="#FFE500")
        captions.write_ass(
            tmp_path / "never_mutates.ass",
            words,
            style,
            width=1080,
            height=1920,
            primary_override="#FFE500",
        )

        assert captions.get_style("clean_lower").primary == before

    def test_written_file_carries_the_primary_override(self, words: list[Word], tmp_path) -> None:
        path = captions.write_ass(
            tmp_path / "coloured.ass",
            words,
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
            primary_override="#FFE500",
        )

        reloaded = pysubs2.load(str(path), encoding="utf-8")
        colour = reloaded.styles[captions.STYLE_NAME].primarycolor
        assert (colour.r, colour.g, colour.b) == (255, 229, 0)

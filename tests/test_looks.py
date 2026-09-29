"""Look presets — named caption-style/grade pairs."""

from __future__ import annotations

import pytest
from autoclip.pipeline import captions, export, looks


class TestLooks:
    def test_five_builtins_with_unique_keys(self) -> None:
        keys = [look.key for look in looks.all_looks()]

        assert len(keys) == len(set(keys)) == 5

    def test_every_look_references_real_presets(self) -> None:
        # validate_look runs at import; this pins the guarantee in tests too.
        for look in looks.all_looks():
            assert captions.get_style(look.caption_style) is captions.PRESETS[look.caption_style]
            export.grade_filters(look.color_grade)  # raises ExportError if unknown

    def test_get_look_is_case_sensitive_and_validating(self) -> None:
        assert looks.get_look("hormozi").caption_style == "bold_pop"

        with pytest.raises(ValueError, match="Unknown look"):
            looks.get_look("explosion")

    def test_looks_cover_distinct_caption_styles(self) -> None:
        # The point of a look is the pairing; two looks with the same pair
        # would be duplicate buttons.
        pairs = {(look.caption_style, look.color_grade) for look in looks.all_looks()}

        assert len(pairs) == 5

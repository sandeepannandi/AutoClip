"""Look presets — one named (caption style, colour grade) pair per button.

A look is the visual identity of an edit: the caption preset and the colour
grade it plays against. Picking them separately works but asks the user to
know the craft; a look encodes the pairing once, so one click gives a clip
(or a whole job) a coherent style. Looks never touch the crop, the audio,
or the words — they map exactly onto the two fields the captions endpoint
already persists, so applying one is an ordinary edit, not a new mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import captions, export


@dataclass(frozen=True)
class Look:
    """One named look: a caption preset plus the grade it pairs with."""

    key: str
    label: str
    description: str
    caption_style: str
    color_grade: str


LOOKS: dict[str, Look] = {
    "hormozi": Look(
        key="hormozi",
        label="Hormozi",
        description="Chunky yellow-accent captions on a punchy grade. Loud and direct.",
        caption_style="bold_pop",
        color_grade="punchy",
    ),
    "viral": Look(
        key="viral",
        label="Viral",
        description="Karaoke word-fill on a warm grade. Built for energy.",
        caption_style="karaoke_fill",
        color_grade="warm",
    ),
    "cinematic": Look(
        key="cinematic",
        label="Cinematic",
        description="Boxed captions on a filmic grade. Moody and composed.",
        caption_style="boxed",
        color_grade="film",
    ),
    "podcast": Look(
        key="podcast",
        label="Podcast",
        description="Clean lower-third captions, untouched footage. Professional.",
        caption_style="clean_lower",
        color_grade="none",
    ),
    "cool": Look(
        key="cool",
        label="Cool",
        description="Clean captions over a cool, composed grade. Sleek and modern.",
        caption_style="clean_lower",
        color_grade="cool",
    ),
}


def get_look(key: str) -> Look:
    look = LOOKS.get(key)
    if look is None:
        raise ValueError(f"Unknown look {key!r}. Available: {', '.join(LOOKS)}")
    return look


def all_looks() -> list[Look]:
    """Every built-in look, in a stable order."""
    return list(LOOKS.values())


def validate_look(look: Look) -> None:
    """A look's parts must reference real presets.

    Checked eagerly rather than at render time, so a typo in the table above
    surfaces as a loud error instead of a broken export months later.
    """
    captions.get_style(look.caption_style)
    try:
        export.grade_filters(look.color_grade)
    except export.ExportError as exc:  # pragma: no cover - guards the table
        raise ValueError(f"Look {look.key!r} grades with an unknown preset: {exc}") from exc


for look in LOOKS.values():
    validate_look(look)

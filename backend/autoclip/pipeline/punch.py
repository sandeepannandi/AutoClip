"""Punch-in zooms on the emphasized words.

The kinetic-cut effect every viral editor does by hand: when the payoff line
is spoken, the frame eases in ~8%, holds briefly, eases back. Done right it is
felt rather than seen — a smooth ease over a third of a second, never a snap,
and only on the one or two phrases the model flagged as the quotable moment.

The model already returns ``emphasis_words`` for every clip; this module turns
those phrases into timed zoom events. Punches are built on the same timeline
as the words they are anchored to (source-absolute, like ``ExportRequest.
words``); the export stage converts them to the tightened output timeline.
Nothing here touches the crop path, the audio, or the captions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .transcript import Word

log = logging.getLogger(__name__)

#: Peak zoom of a punch-in, as a fraction above 1.0. 8% reads as a deliberate
#: camera move on a phone; 15%+ starts to feel like a mistake.
DEFAULT_ZOOM = 0.08

#: The ease in, hold, and ease back durations. The ease is long enough to read
#: as a camera move rather than a jump cut, and the hold is short enough that
#: the frame never feels stuck.
DEFAULT_EASE_IN_S = 0.30
DEFAULT_HOLD_S = 0.50
DEFAULT_EASE_OUT_S = 0.40

#: At most two punches per clip. More turns an edit into a drum solo.
DEFAULT_MAX_PUNCHES = 2

#: Minimum gap between the END of one punch and the START of the next, so two
#: flagged phrases close together become one calm edit, not a wobble.
DEFAULT_MIN_GAP_S = 3.0

#: The hook needs its run-up untouched — the opening beat is where a viewer
#: decides to stay, and a zoom there reads as a glitch. Same value the
#: silence-tightening stage protects.
HOOK_PROTECTION_S = 0.75

#: A punch compressed below this much output time (mostly tightened away by
#: silence speeding) is dropped rather than rendered as a flicker.
MIN_OUTPUT_DURATION_S = 0.2


@dataclass(frozen=True)
class Punch:
    """One punch-in zoom event.

    ``start_s`` is when the ease-in begins; the frame reaches full zoom at
    ``zoom_in_end_s``, holds to ``hold_end_s``, and is back to 1.0 by ``end_s``.
    """

    start_s: float
    zoom_in_end_s: float
    hold_end_s: float
    end_s: float
    zoom: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def build_punches(
    words: list[Word],
    emphasis_words: list[str],
    *,
    zoom: float = DEFAULT_ZOOM,
    ease_in_s: float = DEFAULT_EASE_IN_S,
    hold_s: float = DEFAULT_HOLD_S,
    ease_out_s: float = DEFAULT_EASE_OUT_S,
    max_punches: int = DEFAULT_MAX_PUNCHES,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
) -> list[Punch]:
    """Timed punch events for one clip, on the words' own timeline.

    ``words`` are the clip's words — the same slice the captions receive. The
    returned punches carry those times unchanged; the export stage lifts them
    onto the output timeline.

    Preconditions:
        words sorted by start; emphasis_words are short phrases (1-6 words).
    """
    if not words or not emphasis_words:
        return []

    clip_start = words[0].start
    clip_end = max(word.end for word in words)

    # Longest phrases first so "agencies die at 10" wins over "agencies die"
    # when one phrase contains another.
    ordered = sorted(dict.fromkeys(emphasis_words), key=len, reverse=True)

    anchors: list[float] = []
    seen: set[int] = set()
    for phrase in ordered:
        index = _find_phrase(words, phrase)
        if index is not None and index not in seen:
            seen.add(index)
            anchors.append(words[index].start)

    punches: list[Punch] = []
    for anchor in sorted(anchors):
        start = max(anchor, clip_start + HOOK_PROTECTION_S)
        end = start + ease_in_s + hold_s + ease_out_s
        if end > clip_end:
            # The payoff lands in the clip's last moments: skip rather than
            # truncate the ease-back into the outro.
            continue
        punches.append(
            Punch(
                start_s=start,
                zoom_in_end_s=start + ease_in_s,
                hold_end_s=start + ease_in_s + hold_s,
                end_s=end,
                zoom=zoom,
            )
        )

    return _select(punches, max_punches=max_punches, min_gap_s=min_gap_s)


def _select(punches: list[Punch], *, max_punches: int, min_gap_s: float) -> list[Punch]:
    """Cap the count and enforce the gap, keeping the earliest candidates.

    Anchors are already time-ordered, so "first come" keeps the punches that
    play earliest — the frame moves on the first flagged phrase, not a random
    later one.
    """
    kept: list[Punch] = []
    for punch in punches:
        if len(kept) >= max_punches:
            break
        if any(punch.start_s - other.end_s < min_gap_s for other in kept):
            continue
        kept.append(punch)
    return kept


def _find_phrase(words: list[Word], phrase: str) -> int | None:
    """The index of the first word of ``phrase`` inside ``words``, or None.

    Matching is on stripped tokens — case, punctuation, and apostrophe drift
    between the model's quote and Whisper's transcript must not break it.
    """
    def tokens(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()

    wanted_words = tokens(phrase).split()
    if not wanted_words:
        return None

    joined = [tokens(word.text) for word in words]
    count = len(wanted_words)

    for index in range(len(joined) - count + 1):
        if joined[index : index + count] == wanted_words:
            return index
    return None

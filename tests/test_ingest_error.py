"""Tests for yt-dlp error translation, including the cookie-copy case.

The cookie-copy failure ("Could not copy <Browser> cookie database") is the
most common YouTube download failure on Windows: Chrome's cookie SQLite database
is locked while the browser (including its system-tray process) is running, and
yt-dlp cannot copy it. Previously this fell through to the generic "update
yt-dlp" hint, which was unhelpful.
"""

from __future__ import annotations

import pytest
from autoclip.config import IngestSettings
from autoclip.pipeline.ingest import _translate_ytdlp_error

_COOKIE_ERROR = "ERROR: Could not copy Chrome cookie database"


@pytest.mark.parametrize(
    "browser,expected_in_message,expected_in_hint",
    [
        ("chrome", "from chrome", "Chrome"),
        ("firefox", "from firefox", "Firefox"),
        ("edge", "from edge", "Edge"),
        ("", "", ""),
    ],
)
def test_cookie_copy_error_gives_actionable_hint(
    browser: str, expected_in_message: str, expected_in_hint: str
) -> None:
    exc = Exception(_COOKIE_ERROR)
    err = _translate_ytdlp_error(exc, IngestSettings(cookies_from_browser=browser))

    msg = str(err)
    assert "Could not copy cookies" in msg
    if expected_in_message:
        assert expected_in_message in msg

    hint = err.hint
    assert "locked while the browser is running" in hint
    assert "system tray" in hint
    assert "signed in to YouTube" in hint

    if expected_in_hint:
        assert expected_in_hint in hint


def test_bot_check_error_still_translated() -> None:
    exc = Exception("sign in to confirm you are not a bot")
    err = _translate_ytdlp_error(exc, IngestSettings(cookies_from_browser="chrome"))

    assert "bot check" in str(err).lower()
    assert "cookies are already being read" in err.hint.lower()


def test_generic_ytdlp_error_still_falls_through() -> None:
    exc = Exception("something completely unexpected happened")
    err = _translate_ytdlp_error(exc, IngestSettings(cookies_from_browser="chrome"))

    assert "yt-dlp could not download this video" in str(err).lower()
    assert "autoclip update-ytdlp" in err.hint

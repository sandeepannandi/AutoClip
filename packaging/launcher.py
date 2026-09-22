"""PyInstaller entry point for the AutoClip Windows executable.

Behaviour is delegated entirely to the normal CLI so the exe and a source
install cannot drift:

* no arguments  -> ``autoclip serve`` (web app + browser, the default UX)
* any arguments -> passed through to the Typer app unchanged

The only frozen-specific duty is putting the bundled ffmpeg/ffprobe on PATH
before anything probes for them (``pipeline.ffmpeg`` and ``system.probe_ffmpeg``
both resolve via ``shutil.which``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _force_utf8_io() -> None:
    # When stdout/stderr are redirected (or the console mix is other than
    # UTF-8), Python reports a charset like cp1252 that cannot encode rich's
    # Unicode box/spinner glyphs. Frozen builds must never die with a
    # UnicodeEncodeError inside a status spinner, so pin the standard streams
    # to UTF-8.
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _prepare_environment() -> None:
    if not getattr(sys, "frozen", False):
        return

    _force_utf8_io()

    # sys._MEIPASS is the bundle root: ``_internal`` in onedir, the extraction
    # temp dir in onefile. The spec collects packaging/ffmpeg -> ffmpeg/.
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    ffmpeg_dir = bundle_root / "ffmpeg"
    if ffmpeg_dir.is_dir():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    _prepare_environment()

    if len(sys.argv) == 1:
        sys.argv.append("serve")

    from autoclip.cli import app

    app()


if __name__ == "__main__":
    main()

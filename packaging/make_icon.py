"""Render the AutoClip favicon (frontend/index.html) into packaging/icon.ico.

The favicon is an inline SVG: a 32x32 dark rounded square with an orange play
triangle. Pillow can't rasterize SVG, so the same geometry is redrawn at high
supersampling and downscaled for crisp edges.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

BG = (26, 21, 18)   # #1a1512
FG = (255, 143, 66)  # #ff8f42

SIZES = (16, 32, 48, 64, 96, 128, 256)


def draw_canvas(size: int) -> Image.Image:
    # Supersample by 8 for smooth rounded corners and triangle edges.
    scale = 8
    full = size * scale
    img = Image.new("RGBA", (full, full), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded square with rx=6 on the 32x32 viewBox.
    rx = round(6 / 32 * full)
    d.rounded_rectangle((0, 0, full - 1, full - 1), radius=rx, fill=BG)

    # Play triangle: (11,8) (11,24) (23,16) on the 32x32 viewBox.
    pts = [(round(x / 32 * full), round(y / 32 * full)) for x, y in [(11, 8), (11, 24), (23, 16)]]
    d.polygon(pts, fill=FG)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "icon").with_suffix(".ico")
    # Pillow renders the size ladder from a single largest source image.
    draw_canvas(max(SIZES)).save(
        target,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
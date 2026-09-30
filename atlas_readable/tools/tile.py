"""Slice full-page screenshots into viewport-sized tiles for review: tile.py DIR [TILE_HEIGHT]."""
import sys
from pathlib import Path
from PIL import Image

src = Path(sys.argv[1])
th = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
out = src / "tiles"
out.mkdir(exist_ok=True)
for png in sorted(src.glob("*.png")):
    im = Image.open(png)
    w, h = im.size
    for i, top in enumerate(range(0, h, th)):
        im.crop((0, top, w, min(h, top + th))).save(out / f"{png.stem}_{i:02d}.png")
    print(png.name, im.size, "tiles:", (h + th - 1) // th)

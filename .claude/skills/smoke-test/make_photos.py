"""Generate synthetic test photos from InsightFace's bundled sample image. Runs inside the image.

Usage: python make_photos.py /dest
"""
import sys
from pathlib import Path

import insightface
from PIL import Image

dest = Path(sys.argv[1])
(dest / "sub").mkdir(parents=True, exist_ok=True)
src = Path(insightface.__file__).parent / "data/images/t1.jpg"  # 6 people
im = Image.open(src).convert("RGB")
big = im.resize((im.width * 5, im.height * 5), Image.BICUBIC)  # ~6400px, exercises draft decode

im.save(dest / "group.jpg")
im.resize((im.width * 2 // 3, im.height * 2 // 3)).save(dest / "group_small.jpeg")
im.transpose(Image.FLIP_LEFT_RIGHT).save(dest / "sub" / "group.JPG")  # subfolder + upper-case ext
big.save(dest / "big.jpg", quality=92)

# Pixels stored rotated + EXIF orientation 6: only correct if EXIF transpose is applied.
exif = Image.Exif()
exif[0x0112] = 6
big.transpose(Image.ROTATE_90).save(dest / "rotated.jpg", quality=92, exif=exif)

Image.new("RGB", (800, 600), (90, 140, 200)).save(dest / "sky.jpg")  # no faces
(dest / "broken.jpg").write_text("not an image")  # must be reported, not crash
print(f"wrote test photos to {dest}")

"""Generate synthetic dark-particle image frames; no experimental data used."""

from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / "synthetic_batch" / "demo-3000pa" / "segments" / "segment_001" / "frames_enhanced"
root.mkdir(parents=True, exist_ok=True)

for frame in range(32):
    image = Image.new("L", (220, 140), 225)
    draw = ImageDraw.Draw(image)
    phase = frame / 31
    height = 65 * 4 * phase * (1 - phase)
    particles = [
        (55 + 1.2 * frame, 115 - height),
        (155 - 0.8 * frame, 116 - 0.82 * height),
    ]
    for x, y in particles:
        draw.ellipse((round(x - 4), round(y - 4), round(x + 4), round(y + 4)), fill=25)
    image.save(root / f"frame_{frame:03d}.png")

print(f"Synthetic frames: {len(list(root.glob('*.png')))} in {root}")

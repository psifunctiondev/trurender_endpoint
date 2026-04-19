#!/usr/bin/env python3
"""Create a comparison grid of all 12 blind test renders (A-L), 3 columns x 4 rows."""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs"
GRID_PATH = os.path.join(OUTPUT_DIR, "v5_r3_blind_test.png")

LETTERS = list("ABCDEFGHIJKL")
COLS = 3
ROWS = 4

# Target thumbnail size (maintaining 16:9 aspect)
THUMB_W = 960
THUMB_H = 540

PADDING = 20
LABEL_HEIGHT = 40

# Calculate grid dimensions
grid_w = COLS * THUMB_W + (COLS + 1) * PADDING
grid_h = ROWS * (THUMB_H + LABEL_HEIGHT) + (ROWS + 1) * PADDING

print(f"Creating grid: {grid_w}x{grid_h}")

grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
draw = ImageDraw.Draw(grid)

# Try to get a nice font
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
except Exception:
    font = ImageFont.load_default()

for i, letter in enumerate(LETTERS):
    col = i % COLS
    row = i // COLS

    x = PADDING + col * (THUMB_W + PADDING)
    y = PADDING + row * (THUMB_H + LABEL_HEIGHT + PADDING)

    # Load and resize
    img_path = os.path.join(OUTPUT_DIR, f"v5_r3_blind_{letter}_fullres.png")
    if not os.path.exists(img_path):
        print(f"  WARNING: {img_path} not found, skipping")
        continue

    img = Image.open(img_path)
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

    # Draw label
    draw.text((x + 10, y + 5), letter, fill=(255, 255, 255), font=font)

    # Paste thumbnail below label
    grid.paste(img, (x, y + LABEL_HEIGHT))

    print(f"  Added {letter}")

grid.save(GRID_PATH, "PNG", optimize=True)
size_mb = os.path.getsize(GRID_PATH) / (1024 * 1024)
print(f"\nSaved grid to {GRID_PATH} ({size_mb:.1f}MB)")

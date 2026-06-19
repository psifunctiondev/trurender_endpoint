#!/usr/bin/env python3
"""Build a 4-up comparison contact sheet for the Tier C #2 nudge sweep.

Cells (left to right):
  - DS-0.1 — depth_strength=0.1  (tier_c_2_nudges/DS-0.1.png)
  - DS-0.2 — depth_strength=0.2  (tier_c_2_nudges/DS-0.2.png)
  - DS-0.3 — depth_strength=0.3  (tier_c_2_depth/DS-0.3.png) — ANCHOR from prior probe
  - DS-0.4 — depth_strength=0.4  (tier_c_2_nudges/DS-0.4.png)

Layout: 1x4 horizontal strip. Width sized for the 4 cells.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint")
NUDGES = ROOT / "outputs/tier_c_2_nudges"
TIER_C2 = ROOT / "outputs/tier_c_2_depth"
METRICS = json.load(open(NUDGES / "metrics.json"))

CELL_INFO = {
    "DS-0.1": {
        "title": "DS-0.1",
        "caption": "Lightest touch. Just barely above baseline.",
        "path": NUDGES / "DS-0.1.png",
        "key": "DS-0.1",
        "depth": "0.1",
    },
    "DS-0.2": {
        "title": "DS-0.2",
        "caption": "Tighter nudge. Geometry starting to feel anchored.",
        "path": NUDGES / "DS-0.2.png",
        "key": "DS-0.2",
        "depth": "0.2",
    },
    "DS-0.3": {
        "title": "DS-0.3 — anchor (Tier C #2 winner)",
        "caption": "Tier C #2 winner. TR/TL=0.149 (-56% vs v7.2 baseline).",
        "path": TIER_C2 / "DS-0.3.png",
        "key": "DS-0.3 (anchor)",
        "depth": "0.3",
    },
    "DS-0.4": {
        "title": "DS-0.4",
        "caption": "Heavier anchor. Past the sweet spot?",
        "path": NUDGES / "DS-0.4.png",
        "key": "DS-0.4",
        "depth": "0.4",
    },
}

# Layout — 1x4 horizontal strip
TILE_W, TILE_H = 800, 446          # big enough to see right-edge noise
PAD = 18
LABEL_H = 145
GRID_W = 4 * TILE_W + 5 * PAD
GRID_H = TILE_H + LABEL_H + 3 * PAD + 80

sheet = Image.new("RGB", (GRID_W, GRID_H), color=(26, 36, 51))   # navy 900
draw = ImageDraw.Draw(sheet)


def get_font(size):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


font_title = get_font(20)
font_sub = get_font(13)
font_big = get_font(32)
font_meta = get_font(12)

# Title bar at top
draw.text((PAD, 8),
          "TruRender v7.2 — Tier C #2 Nudges: depth_strength sweep around DS-0.3 (4 cells)",
          font=font_big, fill=(244, 246, 248))
draw.text((PAD, 52),
          "Input: enscape_input.png (ManKitNoLines) · Seed 42 · CFG 4.0 · euler/simple · fp8mixed · "
          "shift 3.1 · cfgnorm 1.0 · P2 prompt · DS-0.3 anchor from outputs/tier_c_2_depth/",
          font=font_sub, fill=(255, 180, 143))   # glow 300

# Place each cell
order = ["DS-0.1", "DS-0.2", "DS-0.3", "DS-0.4"]
winner_key = None
results = METRICS["results"]
if results:
    winner_key = min(results, key=lambda k: results[k]["tr_tl_ratio"])

for i, code in enumerate(order):
    info = CELL_INFO[code]
    x = PAD + i * (TILE_W + PAD)
    y = 88 + PAD

    # Cell background card
    draw.rectangle([x, y, x + TILE_W, y + TILE_H + LABEL_H], fill=(46, 47, 54))

    # Place the rendered image
    png = Image.open(info["path"]).convert("RGB")
    png.thumbnail((TILE_W, TILE_H), Image.LANCZOS)
    px = x + (TILE_W - png.width) // 2
    py = y + 28 + (TILE_H - 28 - png.height) // 2
    sheet.paste(png, (px, py))

    # Title strip on top
    draw.rectangle([x, y, x + TILE_W, y + 28], fill=(26, 36, 51))
    draw.text((x + 8, y + 4), info["title"], font=font_title, fill=(244, 246, 248))

    # Metrics strip below image
    m = METRICS["results"].get(info["key"])
    if m:
        accent = (255, 180, 143)   # glow 300
        normal = (108, 125, 148)   # steel 500
        is_winner = (code == winner_key)
        mtxt = (
            f"TR std: {m['tr_r_std']:.2f}  ·  TL std: {m['tl_r_std']:.2f}  ·  "
            f"TR/TL: {m['tr_tl_ratio']:.3f}  ·  "
            f"size: {m['size_kb']:.1f} KB  ·  {m['image_w']}x{m['image_h']}"
        )
        col = accent if is_winner else normal
        draw.text((x + 8, y + TILE_H + 10), mtxt, font=font_sub, fill=col)
    draw.text((x + 8, y + TILE_H + 34), info["caption"], font=font_sub, fill=(255, 180, 143))
    draw.text((x + 8, y + TILE_H + 64),
              f"depth_strength = {info['depth']}", font=font_meta, fill=(108, 125, 148))

# Footer
footer_y = 88 + PAD + TILE_H + LABEL_H + 8
draw.text((PAD, footer_y),
          "Reading the metrics: TR/TL ratio measures top-right vs top-left 100x100 corner R-channel std-dev. "
          "Lower TR std = less right-edge noise. v7.2 baseline ratio is ~0.34. DS-0.3 anchor collapsed "
          "TR std to 3.80 (vs 7.04 baseline). We're testing whether 0.1/0.2/0.4 land tighter or looser.",
          font=font_sub, fill=(108, 125, 148))
draw.text((PAD, footer_y + 36),
          "Highlight color = best TR/TL ratio in this 4-cell probe (lower is better). "
          "DS-0.3 anchor pulled from outputs/tier_c_2_depth/ — not re-rendered.",
          font=font_meta, fill=(108, 125, 148))

sheet.save(NUDGES / "comparison_sheet.png", optimize=True)
print(f"Wrote {NUDGES / 'comparison_sheet.png'}")
print(f"Dimensions: {sheet.size}")
print(f"Winner by TR/TL: {winner_key}")
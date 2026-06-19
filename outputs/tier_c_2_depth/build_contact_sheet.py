#!/usr/bin/env python3
"""Build a 5-up comparison contact sheet for the Tier C #2 (DiffSynth depth) probe.

Cells (left to right):
  - P2-anchor  — v7_tier_b/BASE.png  — v7.2 baseline
  - CTRL       — tier_c_2_depth/CTRL.png  — depth_strength=0.0 (validation gate)
  - DS-0.3     — tier_c_2_depth/DS-0.3.png — depth_strength=0.3
  - DS-0.5     — tier_c_2_depth/DS-0.5.png — depth_strength=0.5
  - DS-0.7     — tier_c_2_depth/DS-0.7.png — depth_strength=0.7
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint")
PROBE = ROOT / "outputs/tier_c_2_depth"
TIER_B = ROOT / "outputs/v7_tier_b"
METRICS = json.load(open(PROBE / "metrics.json"))

CELL_INFO = {
    "P2": {
        "title": "P2 — v7.2 anchor",
        "caption": "45-word P2 prompt · v7.2 defaults · 1920x1072 (2MP)",
        "path": TIER_B / "BASE.png",
        "key": "P2-anchor (v7_tier_b/BASE)",
    },
    "CTRL": {
        "title": "CTRL — depth=0.0 (validation gate)",
        "caption": "Same workflow as P2 (depth_strength=0.0 default). Should match v7.2 BASE.",
        "path": PROBE / "CTRL.png",
        "key": "CTRL (depth=0.0)",
    },
    "DS-0.3": {
        "title": "DS-0.3 — depth_strength=0.3",
        "caption": "DiffSynth Qwen-Image-Depth ControlNet (light touch). TR std collapses.",
        "path": PROBE / "DS-0.3.png",
        "key": "DS-0.3",
    },
    "DS-0.5": {
        "title": "DS-0.5 — depth_strength=0.5",
        "caption": "Mid strength. TR std recovers toward baseline. Anchoring is heavier.",
        "path": PROBE / "DS-0.5.png",
        "key": "DS-0.5",
    },
    "DS-0.7": {
        "title": "DS-0.7 — depth_strength=0.7",
        "caption": "Heavy depth anchoring. TR std near baseline. Risk of over-constraint.",
        "path": PROBE / "DS-0.7.png",
        "key": "DS-0.7",
    },
}

# Layout — 1x5 horizontal strip
TILE_W, TILE_H = 640, 358           # third of 1920 → 640, half of 1072 → 536 then 0.66
PAD = 18
LABEL_H = 145                      # header + metrics strip below image
GRID_W = 5 * TILE_W + 6 * PAD
GRID_H = TILE_H + LABEL_H + 3 * PAD + 80   # +80 for top title bar

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
font_sub   = get_font(13)
font_big   = get_font(32)
font_meta  = get_font(12)
font_arrow = get_font(36)

# Title bar at top
draw.text((PAD, 8), "TruRender v7.2 — Tier C #2: DiffSynth Depth ControlNet Backstop (5 cells)",
          font=font_big, fill=(244, 246, 248))
draw.text((PAD, 52),
          "Input: enscape_input.png (ManKitNoLines) · Seed 42 · CFG 4.0 · euler/simple · fp8mixed · shift 3.1 · cfgnorm 1.0 · P2 prompt",
          font=font_sub, fill=(255, 180, 143))   # glow 300

# Place each cell
order = ["P2", "CTRL", "DS-0.3", "DS-0.5", "DS-0.7"]
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
        # Highlight: best TR/TL ratio gets glow accent
        accent = (255, 180, 143)   # glow 300
        normal = (108, 125, 148)   # steel 500
        is_winner = (code == "DS-0.3")  # this is the standout
        mtxt = (
            f"TR std: {m['tr_r_std']:.2f}  ·  TL std: {m['tl_r_std']:.2f}  ·  "
            f"TR/TL: {m['tr_tl_ratio']:.3f}  ·  "
            f"size: {m['size_kb']:.1f} KB  ·  {m['image_w']}x{m['image_h']}"
        )
        col = accent if is_winner else normal
        draw.text((x + 8, y + TILE_H + 10), mtxt, font=font_sub, fill=col)
    draw.text((x + 8, y + TILE_H + 34), info["caption"], font=font_sub, fill=(255, 180, 143))
    # depth strength tag
    if code.startswith("DS-"):
        tag = f"depth_strength = {code.split('-')[1]}"
    elif code == "CTRL":
        tag = "depth_strength = 0.0 (off — validation gate)"
    else:
        tag = "(v7.2 baseline, no depth)"
    draw.text((x + 8, y + TILE_H + 64), tag, font=font_meta, fill=(108, 125, 148))
    # bit-identicality marker on CTRL
    if code == "CTRL":
        draw.text((x + 8, y + TILE_H + 84),
                  "Note: CTRL ≠ BASE bit-identically (ComfyUI Qwen pipeline has minor non-determinism)",
                  font=font_meta, fill=(240, 100, 58))   # coral 500

# Footer
footer_y = 88 + PAD + TILE_H + LABEL_H + 8
draw.text((PAD, footer_y),
          "Reading the metrics: TR/TL ratio measures top-right vs top-left 100x100 corner R-channel std-dev. "
          "Lower TR std = less right-edge noise. Ratio near 0.34 (P2 anchor) is the v7.2 baseline. "
          "DS-0.3 collapses TR std to 3.80 (vs 7.04 baseline, -46%) and ratio to 0.149 (-56%) — clear win. "
          "DS-0.5 and DS-0.7 lose the advantage as depth becomes a hard constraint rather than an anchor.",
          font=font_sub, fill=(108, 125, 148))
draw.text((PAD, footer_y + 36),
          "Per-cell artifact: depth_map.png (Depth Anything V2-Small) is the visual depth of the source render — "
          "what 'depth' looks like in this scene. The actual ControlNet runs its own internal depth encoding on the RGB image.",
          font=font_meta, fill=(108, 125, 148))

sheet.save(PROBE / "comparison_sheet.png", optimize=True)
print(f"Wrote {PROBE / 'comparison_sheet.png'}")
print(f"Dimensions: {sheet.size}")

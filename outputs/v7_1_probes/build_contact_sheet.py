#!/usr/bin/env python3
"""Build a 3-up comparison contact sheet for the v7.1 probe renders."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROBE_DIR = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_1_probes")
METRICS = json.load(open(PROBE_DIR / "metrics.json"))

# Cell labels and descriptions
CELL_INFO = {
    "BASE":   ("Baseline (v7.1 defaults)",       "steps=40 · euler/simple · fp8=True · cfg=3.5 · seed=42"),
    "SAMPLER":("Sampler swap: dpmpp_2m/karras",   "steps=40 · dpmpp_2m/karras · fp8=True · cfg=3.5 · seed=42"),
    "STEP60": ("Steps 60 (vs 40)",                "steps=60 · euler/simple · fp8=True · cfg=3.5 · seed=42"),
    "BF16":   ("BF16 (FP8 disabled) — FAILED",    "use_fp8=False · model file not deployed in v7.1 container"),
}

# Build a 2x2 grid (4 cells, last is the BF16 failure card)
TILE_W, TILE_H = 960, 536          # half-resolution of 1920x1072
PAD = 24
LABEL_H = 110                     # header + caption strip below image
GRID_W = 2 * TILE_W + 3 * PAD
GRID_H = 2 * (TILE_H + LABEL_H) + 3 * PAD

sheet = Image.new("RGB", (GRID_W, GRID_H), color=(26, 36, 51))   # navy 900
draw = ImageDraw.Draw(sheet)

# Try to load a reasonable font, fall back to default
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

font_title = get_font(22)
font_sub   = get_font(16)
font_big   = get_font(40)

# Title bar at top
draw.text((PAD, 8), "TruRender v7.1 — Single-Variable Probe Sweep (4 cells)",
          font=font_big, fill=(244, 246, 248))
draw.text((PAD, 60),
          "Input: enscape_input.png · Prompt: P2 · Output: 1920x1072 (2MP) · Seed: 42",
          font=font_sub, fill=(255, 180, 143))   # glow 300

# Place each cell
positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
order = ["BASE", "SAMPLER", "STEP60", "BF16"]

for (col, row), code in zip(positions, order):
    x = PAD + col * (TILE_W + PAD)
    y = 96 + PAD + row * (TILE_H + LABEL_H + PAD)

    info = CELL_INFO[code]
    title, caption = info

    # Cell background card
    draw.rectangle([x, y, x + TILE_W, y + TILE_H + LABEL_H], fill=(46, 47, 54))

    if code != "BF16":
        # Place the rendered image
        png = Image.open(PROBE_DIR / f"{code}.png").convert("RGB")
        png.thumbnail((TILE_W, TILE_H), Image.LANCZOS)
        px = x + (TILE_W - png.width) // 2
        py = y + (TILE_H - png.height) // 2
        sheet.paste(png, (px, py))

        # Title strip on top
        draw.text((x + 12, y + 8), title, font=font_title, fill=(244, 246, 248))

        # Metrics strip below image
        m = next(r for r in METRICS["rows"] if r["code"] == code)
        mtxt = (
            f"render: {m['render_s']:.1f}s · wall: {m['wall_s']:.1f}s · "
            f"size: {m['size_bytes']//1024} KB · "
            f"std-dev TL/TR/BL/BR: "
            f"{m['tl_std']:.1f}/{m['tr_std']:.1f}/{m['bl_std']:.1f}/{m['br_std']:.1f} · "
            f"TR/TL={m['tr_tl']:.3f}"
        )
        draw.text((x + 12, y + TILE_H + 8),  mtxt, font=font_sub, fill=(108, 125, 148))
        draw.text((x + 12, y + TILE_H + 32), caption, font=font_sub, fill=(255, 180, 143))
        draw.text((x + 12, y + TILE_H + 56),
                  f"[{code}] {m['dims'][0]}×{m['dims'][1]}",
                  font=font_sub, fill=(108, 125, 148))
    else:
        # BF16 failure card
        draw.rectangle([x, y, x + TILE_W, y + TILE_H], fill=(20, 20, 24))
        draw.text((x + 24, y + 24), title, font=font_title, fill=(240, 100, 58))   # coral 500
        msg = [
            "Render FAILED: ComfyUI rejected the workflow.",
            "",
            "Error:",
            "  unet_name: 'qwen_image_edit_2511_bf16.safetensors'",
            "  not in ['qwen_image_edit_2511_fp8mixed.safetensors']",
            "",
            "Root cause:",
            "  The v7.1 Modal deployment only ships the fp8 model file",
            "  on the model volume. The bf16 variant would need to be",
            "  downloaded and added — that requires redeploying the",
            "  Modal app, which is out of scope for this probe.",
            "",
            "Status: 3/4 renders completed successfully.",
        ]
        ty = y + 60
        for line in msg:
            fill = (255, 180, 143) if line.startswith("Error") or line.startswith("Root cause") or line.startswith("Status") else (244, 246, 248)
            draw.text((x + 24, ty), line, font=font_sub, fill=fill)
            ty += 22
        draw.text((x + 12, y + TILE_H + 8),  caption, font=font_sub, fill=(108, 125, 148))
        draw.text((x + 12, y + TILE_H + 32), "[BF16] error caught, did not block other cells", font=font_sub, fill=(240, 100, 58))

sheet.save(PROBE_DIR / "comparison_sheet.png", optimize=True)
print(f"Wrote {PROBE_DIR / 'comparison_sheet.png'}")

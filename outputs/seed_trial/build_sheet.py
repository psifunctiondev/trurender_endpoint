#!/usr/bin/env python3
"""Build the seed-variation trial comparison contact sheet.

1 row × 5 columns:
    [Tier-B BASE (seed=42)] [S-1] [S-7] [S-100] [S-1234]

Visual style matches Tier-B / wide-shift sheets:
    - Dark navy background (#1A2433)
    - White cell code label at top of each cell
    - TR/TL ratio label at bottom of each cell
    - Layout: 1 row × 5 columns, ~2500×500

Output: outputs/seed_trial/comparison_sheet.png
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/seed_trial"

# Psi Function brand
NAVY = (26, 36, 51)
STEEL = (108, 125, 148)
WHITE = (244, 246, 248)
CORAL = (240, 100, 58)

CELLS = [
    {
        "code": "Tier-B BASE",
        "seed": 42,
        "label_subtitle": "(seed=42 reference)",
        "path": "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_tier_b/BASE.png",
    },
    {"code": "S-1",    "seed": 1,    "label_subtitle": "small int",      "path": os.path.join(OUT_DIR, "S-1.png")},
    {"code": "S-7",    "seed": 7,    "label_subtitle": "prime-ish",      "path": os.path.join(OUT_DIR, "S-7.png")},
    {"code": "S-100",  "seed": 100,  "label_subtitle": "round",          "path": os.path.join(OUT_DIR, "S-100.png")},
    {"code": "S-1234", "seed": 1234, "label_subtitle": "larger int",     "path": os.path.join(OUT_DIR, "S-1234.png")},
]

FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFNSText.ttf",
]


def get_font(size: int):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def main() -> None:
    metrics = json.load(open(os.path.join(OUT_DIR, "metrics.json")))
    # BASE ref + trial cells
    by_code = {c["code"]: c for c in metrics["trial"]["cells"]}
    base_ref = metrics["reference"]

    # Build a code → metrics dict for labels. BASE here uses "BASE" key in metrics.
    label_map = {"Tier-B BASE": base_ref}
    for c in CELLS[1:]:
        label_map[c["code"]] = by_code[c["code"]]

    # Cell image target height
    target_h = 380
    target_w = int(target_h * (1920 / 1072))  # ≈ 680

    # Layout
    n_cols = len(CELLS)
    pad_x = 8
    pad_outer = 16
    header_h = 56
    sub_header_h = 18
    footer_h = 22
    top_label_h = header_h + sub_header_h
    cell_block_h = top_label_h + target_h + footer_h

    sheet_w = pad_outer * 2 + sum(target_w + pad_x for _ in range(n_cols)) - pad_x
    sheet_h = pad_outer * 2 + 24 + cell_block_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), NAVY)
    draw = ImageDraw.Draw(sheet)

    title_font = get_font(22)
    sub_font = get_font(13)
    cell_label_font = get_font(20)
    cell_sub_font = get_font(12)
    metric_font = get_font(11)

    title = "TruRender v7.1 — Seed Variation Trial (P2 / 2MP / 40s euler simple / shift 3.1 / cfgnorm 1.0)"
    tw = draw.textlength(title, font=title_font)
    draw.text(((sheet_w - tw) / 2, pad_outer - 4), title, fill=WHITE, font=title_font)

    x = pad_outer
    y = pad_outer + 24
    for cell in CELLS:
        code = cell["code"]
        m = label_map[code]
        code_text = f"{code}"
        sub_text = f"seed={cell['seed']}  ·  {cell['label_subtitle']}"
        cw = draw.textlength(code_text, font=cell_label_font)
        sw = draw.textlength(sub_text, font=cell_sub_font)
        draw.text((x + (target_w - cw) / 2, y), code_text, fill=WHITE, font=cell_label_font)
        draw.text((x + (target_w - sw) / 2, y + 26), sub_text, fill=STEEL, font=cell_sub_font)

        img = Image.open(cell["path"]).convert("RGB")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        sheet.paste(img, (x, y + top_label_h))

        ratio = m.get("tr_tl")
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        prefix = f"TR std={m['tr_std']:.2f}  TL std={m['tl_std']:.2f}  ratio="
        metric_text = prefix + ratio_str
        prefix_w = draw.textlength(prefix, font=metric_font)
        total_w = draw.textlength(metric_text, font=metric_font)
        start_x = x + (target_w - total_w) / 2
        # Highlight ratio in coral if it deviates from BASE
        ratio_color = STEEL
        if ratio is not None and base_ref.get("tr_tl") is not None:
            if abs(ratio - base_ref["tr_tl"]) > 0.05:
                ratio_color = CORAL
        draw.text((start_x, y + top_label_h + target_h + 6), prefix, fill=STEEL, font=metric_font)
        draw.text(
            (start_x + prefix_w, y + top_label_h + target_h + 6),
            ratio_str,
            fill=ratio_color,
            font=metric_font,
        )

        x += target_w + pad_x

    out_path = os.path.join(OUT_DIR, "comparison_sheet.png")
    sheet.save(out_path, optimize=True)
    print(f"wrote {out_path}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
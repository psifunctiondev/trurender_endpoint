#!/usr/bin/env python3
"""Build the wide-shift comparison contact sheet.

1 row × 5 columns:
    [Tier-B BASE (shift=3.1)] [SH-1.0] [SH-2.0] [SH-6.0] [SH-8.0]

Visual style matches the Tier-B sheet:
    - Dark navy background (#1A2433)
    - White cell code label at top of each cell
    - TL/TR ratio label at bottom of each cell
    - Layout: 1 row × 5 columns, ~2500×500

Output: outputs/wide_shift/comparison_sheet.png
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/wide_shift"

# Psi Function brand: Navy 900 / Steel 500 / Coral 500 / White 50
NAVY = (26, 36, 51)
STEEL = (108, 125, 148)
WHITE = (244, 246, 248)
CORAL = (240, 100, 58)

CELLS = [
    {
        "code": "Tier-B BASE",
        "shift": 3.1,
        "label_subtitle": "(trained)",
        "path": "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_tier_b/BASE.png",
    },
    {"code": "SH-1.0", "shift": 1.0, "label_subtitle": "linear",  "path": os.path.join(OUT_DIR, "SH-1.0.png")},
    {"code": "SH-2.0", "shift": 2.0, "label_subtitle": "between", "path": os.path.join(OUT_DIR, "SH-2.0.png")},
    {"code": "SH-6.0", "shift": 6.0, "label_subtitle": "above",   "path": os.path.join(OUT_DIR, "SH-6.0.png")},
    {"code": "SH-8.0", "shift": 8.0, "label_subtitle": "far above","path": os.path.join(OUT_DIR, "SH-8.0.png")},
]

# Try to find a usable font; fall back to PIL default
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


def load_metrics() -> dict:
    """Load the metrics.json so we can label each cell with its TR/TL ratio."""
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        return json.load(f)


# Display code → metrics.json code (we use the human label in the sheet,
# the metrics file uses an internal uppercase key for the reference cell)
CODE_MAP = {
    "Tier-B BASE": "TIER-B-BASE",
}


def main() -> None:
    metrics = load_metrics()
    # Build a code → metrics map
    by_code = {c["code"]: c for c in metrics["cells"]}

    # Cell image target height (preserve aspect, all source images are 1920×1072)
    target_h = 380  # leave room for header/footer labels
    # 1920×1072 → aspect 0.5583 → width = 380 / 0.5583 ≈ 680
    target_w = int(target_h * (1920 / 1072))  # ≈ 680

    # Layout
    n_cols = len(CELLS)
    cell_total_w = target_w  # image plus side padding inside the cell
    pad_x = 8  # gap between cells
    pad_outer = 16
    header_h = 56
    sub_header_h = 18
    footer_h = 22
    top_label_h = header_h + sub_header_h
    cell_block_h = top_label_h + target_h + footer_h

    sheet_w = pad_outer * 2 + sum(cell_total_w + pad_x for _ in range(n_cols)) - pad_x
    sheet_h = pad_outer * 2 + 24 + cell_block_h  # 24 for sheet title

    sheet = Image.new("RGB", (sheet_w, sheet_h), NAVY)
    draw = ImageDraw.Draw(sheet)

    # Sheet title
    title_font = get_font(22)
    sub_font = get_font(13)
    cell_label_font = get_font(20)
    cell_sub_font = get_font(12)
    metric_font = get_font(11)

    title = "TruRender v7.1 — Wide model_sampling_shift Sweep (4 cells, P2 / seed 42 / euler/simple 40)"
    tw = draw.textlength(title, font=title_font)
    draw.text(((sheet_w - tw) / 2, pad_outer - 4), title, fill=WHITE, font=title_font)

    # Cells
    x = pad_outer
    y = pad_outer + 24
    for cell in CELLS:
        code = cell["code"]
        m = by_code[CODE_MAP.get(code, code)]
        # Top label (code) + sub label (subtitle)
        code_text = f"{code}"
        sub_text = f"shift={cell['shift']:.1f}  ·  {cell['label_subtitle']}"
        cw = draw.textlength(code_text, font=cell_label_font)
        sw = draw.textlength(sub_text, font=cell_sub_font)
        draw.text((x + (target_w - cw) / 2, y), code_text, fill=WHITE, font=cell_label_font)
        draw.text((x + (target_w - sw) / 2, y + 26), sub_text, fill=STEEL, font=cell_sub_font)

        # Image — paste at top_label_h
        img = Image.open(cell["path"]).convert("RGB")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        sheet.paste(img, (x, y + top_label_h))

        # Bottom metrics line
        ratio = m["tr_tl_ratio"]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        metric_text = f"TR std={m['tr_std']:.2f}  TL std={m['tl_std']:.2f}  ratio={ratio_str}"
        mw = draw.textlength(metric_text, font=metric_font)
        # TL/TR ratio in coral if high (>1.0) or to highlight deviation
        ratio_color = CORAL if ratio is not None and abs(ratio - 0.339) > 0.05 else STEEL
        # Use coral for the ratio number itself
        prefix = f"TR std={m['tr_std']:.2f}  TL std={m['tl_std']:.2f}  ratio="
        prefix_w = draw.textlength(prefix, font=metric_font)
        total_w = draw.textlength(metric_text, font=metric_font)
        start_x = x + (target_w - total_w) / 2
        draw.text((start_x, y + top_label_h + target_h + 6), prefix, fill=STEEL, font=metric_font)
        draw.text(
            (start_x + prefix_w, y + top_label_h + target_h + 6),
            ratio_str,
            fill=ratio_color,
            font=metric_font,
        )

        x += cell_total_w + pad_x

    out_path = os.path.join(OUT_DIR, "comparison_sheet.png")
    sheet.save(out_path, optimize=True)
    print(f"wrote {out_path}  ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()

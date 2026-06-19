#!/usr/bin/env python3
"""Build v7.4 probe artifacts: comparison sheet, metrics, selector output JSON.

Reads:
  trurender_endpoint/outputs/v7_4_probe/base.png
  trurender_endpoint/outputs/v7_4_probe/img_only.png
  trurender_endpoint/outputs/v7_4_probe/results_manifest.json
  trurender_endpoint/inputs/enscape_input.png  (source)

Writes:
  trurender_endpoint/outputs/v7_4_probe/comparison_sheet.png  (1980x544 2-up)
  trurender_endpoint/outputs/v7_4_probe/metrics.json
  trurender_endpoint/outputs/v7_4_probe/selector_output.json

Also loads the style-refs.json + runs the smart selector (mirrors the
remote code path) to capture exact score breakdown for the report.
"""
import json
import sys
import os
from pathlib import Path

from PIL import Image
import numpy as np

REPO_ROOT = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint")
PROBE_DIR = REPO_ROOT / "outputs" / "v7_4_probe"
BASE_PATH = PROBE_DIR / "base.png"
IMG_PATH = PROBE_DIR / "img_only.png"
SOURCE_PATH = REPO_ROOT / "inputs" / "enscape_input.png"
MANIFEST_PATH = PROBE_DIR / "results_manifest.json"
STYLE_REFS_JSON = REPO_ROOT / "assets" / "trurender" / "style-refs" / "style-refs.json"

# Add the pipeline dir to sys.path so we can import the smart selector
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
# Stub out modal so the import doesn't fail
import types
_modal_stub = types.ModuleType("modal")
class _Stub:
    """Stub that returns itself for any attribute access / call."""
    def __getattr__(self, name):
        return _Stub()
    def __call__(self, *a, **kw):
        return _Stub()
_modal_stub.Image = _Stub()
_modal_stub.App = _Stub()
_modal_stub.Volume = _Stub()
_modal_stub.Secret = _Stub()
sys.modules["modal"] = _modal_stub
# Stub modal.concurrent decorator too (used as @modal.concurrent(...))
_modal_stub.concurrent = lambda *a, **kw: (lambda f: f)
_modal_stub.method = lambda *a, **kw: (lambda f: f)
_modal_stub.enter = lambda *a, **kw: (lambda f: f)
_modal_stub.exit = lambda *a, **kw: (lambda f: f)
_modal_stub.cls = lambda *a, **kw: (lambda f: f)
from trurender_qwen_comfyui import select_style_ref, _score_ref  # noqa: E402

# --- 1. Build selector_output.json ---
with open(STYLE_REFS_JSON) as f:
    style_manifest = json.load(f)
with open(MANIFEST_PATH) as f:
    sweep_manifest = json.load(f)

# Use IMG cell's space_type=kitchen — that's the probe default
probe_pick = select_style_ref("kitchen")

# Get the IMG cell timing
img_cell = next(r for r in sweep_manifest["results"] if r["code"] == "IMG")
base_cell = next(r for r in sweep_manifest["results"] if r["code"] == "BASE")

selector_output = {
    "probe": "v7.4",
    "input_image": str(SOURCE_PATH),
    "space_type": "kitchen",
    "picked": {
        "filename": probe_pick["filename"],
        "source": probe_pick["source"],
        "score": probe_pick["score"],
        "tags": probe_pick["tags"],
        "weight": probe_pick["weight"],
        "local_path": probe_pick.get("local_path"),
        "why": probe_pick["why"],
    },
    "scored_candidates": probe_pick.get("scored_candidates", []),
    "tag_weights_table": style_manifest["tag_weights"]["kitchen"],
    "catherine_picks_available": ["style_ref_1_library.jpg",
                                  "style_ref_2_staircase.jpg",
                                  "style_ref_3_kitchen.jpg"],
    "all_refs_count": 19,
    "notes": (
        "Smart selector unified 16 canonical CTAI refs (local) + 3 Catherine picks "
        "(Modal volume). For 'kitchen', the canonical set wins (no Catherine pick "
        "scored higher; the highest Catherine pick would have been "
        "style_ref_3_kitchen.jpg with score 1.0 = 1.0 * 1.0 from the kitchen tag, "
        "but Locke-Feature-scaled.jpg scores 1.275 with both kitchen=0.85 AND "
        "living=0.425 contributions)."
    ),
}
with open(PROBE_DIR / "selector_output.json", "w") as f:
    json.dump(selector_output, f, indent=2)
print(f"[build_artifacts] wrote selector_output.json")
print(f"  picked: {probe_pick['filename']} (source={probe_pick['source']}, score={probe_pick['score']:.3f})")

# --- 2. Build metrics.json ---
def image_metrics(path: Path) -> dict:
    img = Image.open(path)
    w, h = img.size
    arr = np.array(img.convert("RGB"))
    # R-channel corner std-dev (TR/TL ratio from v7 sweep methodology)
    # Use ~5% corner patches
    pw, ph = max(int(w * 0.05), 8), max(int(h * 0.05), 8)
    tl = arr[:ph, :pw, 0]
    tr = arr[:ph, -pw:, 0]
    bl = arr[-ph:, :pw, 0]
    br = arr[-ph:, -pw:, 0]
    tl_std = float(tl.std())
    tr_std = float(tr.std())
    bl_std = float(bl.std())
    br_std = float(br.std())
    # Whole-image R std (signal-richness)
    whole_r_std = float(arr[:, :, 0].std())
    # Diff against source
    src = Image.open(SOURCE_PATH).convert("RGB")
    if src.size != img.size:
        src_r = src.resize(img.size, Image.LANCZOS)
    else:
        src_r = src
    src_arr = np.array(src_r)
    diff = np.abs(arr.astype(np.int32) - src_arr.astype(np.int32))
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff.astype(np.float64) ** 2).mean()))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "width": w,
        "height": h,
        "tl_r_std": tl_std,
        "tr_r_std": tr_std,
        "bl_r_std": bl_std,
        "br_r_std": br_std,
        "tr_over_tl_ratio": (tr_std / tl_std) if tl_std > 0 else None,
        "whole_r_std": whole_r_std,
        "mae_vs_source": mae,
        "rmse_vs_source": rmse,
    }

base_metrics = image_metrics(BASE_PATH)
img_metrics_ = image_metrics(IMG_PATH)
source_metrics = image_metrics(SOURCE_PATH)

metrics = {
    "probe": "v7.4",
    "input_image": str(SOURCE_PATH),
    "wall_time_s": sweep_manifest["total_sweep_s"],
    "ok_count": sweep_manifest["ok_count"],
    "fail_count": sweep_manifest["fail_count"],
    "cells": {
        "BASE": {
            **base_metrics,
            "render_s": base_cell["render_s"],
            "wall_s": base_cell["wall_s"],
            "style_image_name": base_cell.get("style_image_name"),
        },
        "IMG": {
            **img_metrics_,
            "render_s": img_cell["render_s"],
            "wall_s": img_cell["wall_s"],
            "style_image_name": img_cell.get("style_image_name"),
        },
    },
    "source": source_metrics,
    "notes": (
        "TR/TL std ratio is the v7.3 right-border-hallucination diagnostic. "
        "v7.3 baseline TR/TL was 0.68 (clean) at 2MP; v7.2 baseline was 1.18. "
        "Comparing the two cells shows the style-ref effect on border noise. "
        "MAE/RMSE measure divergence from the source Enscape render — "
        "lower = more faithful to source geometry/materials."
    ),
}
with open(PROBE_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[build_artifacts] wrote metrics.json")
print(f"  BASE: {base_metrics['width']}x{base_metrics['height']}, "
      f"TR/TL={base_metrics['tr_over_tl_ratio']:.3f}, "
      f"MAE={base_metrics['mae_vs_source']:.2f}, "
      f"size={base_metrics['size_bytes']//1024}KB")
print(f"  IMG:  {img_metrics_['width']}x{img_metrics_['height']}, "
      f"TR/TL={img_metrics_['tr_over_tl_ratio']:.3f}, "
      f"MAE={img_metrics_['mae_vs_source']:.2f}, "
      f"size={img_metrics_['size_bytes']//1024}KB")

# --- 3. Build comparison_sheet.png (1980x544 2-up, source on left + 2 cells) ---
# Actually per brief: 2-up side-by-side at 1980x544 — base | img_only
# Layout: left label strip (~40px) + 2 cells of ~960px each
TARGET_W = 1980
TARGET_H = 544
LABEL_W = 60
CELL_W = (TARGET_W - LABEL_W) // 2  # 960
CELL_H = TARGET_H  # 544

# Source image scaled to fit
def fit_image(img, w, h):
    src_w, src_h = img.size
    scale = min(w / src_w, h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

sheet = Image.new("RGB", (TARGET_W, TARGET_H), (20, 20, 24))
# Try a 3-up: SOURCE | BASE | IMG — that's more useful for visual diff
# Brief says 2-up. Let me follow brief: 2 cells side-by-side.
base_img = fit_image(Image.open(BASE_PATH), CELL_W, CELL_H)
img_img = fit_image(Image.open(IMG_PATH), CELL_W, CELL_H)
# Paste centered
sheet.paste(base_img, (LABEL_W + (CELL_W - base_img.width) // 2,
                       (CELL_H - base_img.height) // 2))
sheet.paste(img_img, (LABEL_W + CELL_W + (CELL_W - img_img.width) // 2,
                      (CELL_H - img_img.height) // 2))

# Add labels (vertical, on the left strip)
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(sheet)
try:
    # Try to find a system font
    for fp in ["/System/Library/Fonts/Helvetica.ttc",
               "/System/Library/Fonts/SFNSMono.ttf",
               "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(fp):
            font_lg = ImageFont.truetype(fp, 28)
            font_sm = ImageFont.truetype(fp, 14)
            break
    else:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()
except Exception:
    font_lg = ImageFont.load_default()
    font_sm = ImageFont.load_default()

# Label strip background
draw.rectangle([(0, 0), (LABEL_W, TARGET_H)], fill=(40, 40, 50))

# Cell labels above each cell
draw.text((LABEL_W + 8, 6), "BASE  (v7.3)", fill=(220, 220, 220), font=font_lg)
draw.text((LABEL_W + CELL_W + 8, 6), "IMG  (v7.4)", fill=(220, 220, 220), font=font_lg)

# Sub-labels
draw.text((LABEL_W + 8, 38), "no style ref", fill=(150, 150, 150), font=font_sm)
sub = f"style={img_cell.get('style_image_name','?')}"
draw.text((LABEL_W + CELL_W + 8, 38), sub, fill=(150, 150, 150), font=font_sm)

# Add small metadata at bottom
m1 = f"seed=42 cfg=4.0 40steps depth=0.3"
m2 = f"seed=42 cfg=4.0 40steps depth=0.3"
draw.text((LABEL_W + 8, TARGET_H - 22), m1, fill=(120, 120, 120), font=font_sm)
draw.text((LABEL_W + CELL_W + 8, TARGET_H - 22), m2, fill=(120, 120, 120), font=font_sm)

# Vertical text "v7.4 PROBE" on the left strip
v_text = "v7.4 PROBE"
try:
    # Render vertical: rotate text
    txt_img = Image.new("RGBA", (200, 30), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt_img)
    td.text((0, 0), v_text, fill=(180, 180, 180), font=font_lg)
    txt_img = txt_img.rotate(90, expand=True)
    sheet.paste(txt_img, (10, 200), txt_img)
except Exception:
    pass

# Save
out_sheet = PROBE_DIR / "comparison_sheet.png"
sheet.save(out_sheet, optimize=True)
print(f"[build_artifacts] wrote {out_sheet} ({out_sheet.stat().st_size//1024}KB)")

# --- 4. Build a 3-up "source | base | img" for richer visual evidence ---
T3_W = 1980
T3_H = 720
T3_LABEL_W = 60
T3_CELL_W = (T3_W - T3_LABEL_W) // 3
T3_CELL_H = T3_H
sheet3 = Image.new("RGB", (T3_W, T3_H), (20, 20, 24))
src_img = fit_image(Image.open(SOURCE_PATH), T3_CELL_W, T3_CELL_H)
base_img = fit_image(Image.open(BASE_PATH), T3_CELL_W, T3_CELL_H)
img_img = fit_image(Image.open(IMG_PATH), T3_CELL_W, T3_CELL_H)
sheet3.paste(src_img, (T3_LABEL_W + (T3_CELL_W - src_img.width)//2,
                       (T3_CELL_H - src_img.height)//2))
sheet3.paste(base_img, (T3_LABEL_W + T3_CELL_W + (T3_CELL_W - base_img.width)//2,
                        (T3_CELL_H - base_img.height)//2))
sheet3.paste(img_img, (T3_LABEL_W + 2*T3_CELL_W + (T3_CELL_W - img_img.width)//2,
                       (T3_CELL_H - img_img.height)//2))
draw3 = ImageDraw.Draw(sheet3)
draw3.rectangle([(0, 0), (T3_LABEL_W, T3_H)], fill=(40, 40, 50))
draw3.text((T3_LABEL_W + 8, 6), "SOURCE  (Enscape)", fill=(220, 220, 220), font=font_lg)
draw3.text((T3_LABEL_W + T3_CELL_W + 8, 6), "BASE  (v7.3)", fill=(220, 220, 220), font=font_lg)
draw3.text((T3_LABEL_W + 2*T3_CELL_W + 8, 6), "IMG  (v7.4)", fill=(220, 220, 220), font=font_lg)
draw3.text((T3_LABEL_W + 8, 38), "3D render", fill=(150, 150, 150), font=font_sm)
draw3.text((T3_LABEL_W + T3_CELL_W + 8, 38), "no style ref", fill=(150, 150, 150), font=font_sm)
draw3.text((T3_LABEL_W + 2*T3_CELL_W + 8, 38), sub, fill=(150, 150, 150), font=font_sm)
out_sheet3 = PROBE_DIR / "comparison_3up.png"
sheet3.save(out_sheet3, optimize=True)
print(f"[build_artifacts] wrote {out_sheet3} ({out_sheet3.stat().st_size//1024}KB)")
print("[build_artifacts] DONE")

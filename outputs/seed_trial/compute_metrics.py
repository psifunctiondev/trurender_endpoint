#!/usr/bin/env python3
"""Compute 100×100 corner R-channel std-dev asymmetry for seed trial + BASE reference.

Per task spec: TR std (top-right 100×100 R-channel), TL std (top-left 100×100 R-channel),
TR/TL ratio. Includes the v7_tier_b BASE cell (shift=3.1, seed=42) as reference.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/seed_trial")
MANIFEST = json.load(open(OUT_DIR / "results_manifest.json"))
BASE_PATH = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_tier_b/BASE.png")
BASE_METRICS = json.load(open("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_tier_b/metrics.json"))


def corner_r_std(img: Image.Image, corner: str, size: int = 100):
    """Return std-dev of R-channel in a `size`×`size` corner of the image."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    r = arr[:, :, 0]
    if corner == "TL":
        patch = r[0:size, 0:size]
    elif corner == "TR":
        patch = r[0:size, w - size:w]
    elif corner == "BL":
        patch = r[h - size:h, 0:size]
    elif corner == "BR":
        patch = r[h - size:h, w - size:w]
    else:
        raise ValueError(corner)
    return float(np.std(patch))


def fmt(n):
    if n is None:
        return "–"
    return f"{n:.3f}"


rows = []
# Process trial cells
for r in MANIFEST["results"]:
    p = OUT_DIR / r["output"]
    if not r["ok"] or not p.exists():
        rows.append({**r, "tl_std": None, "tr_std": None,
                     "bl_std": None, "br_std": None, "tr_tl": None,
                     "dims": None, "size_kb": r["size_bytes"] // 1024 if r.get("size_bytes") else None})
        continue
    img = Image.open(p)
    tl = corner_r_std(img, "TL")
    tr = corner_r_std(img, "TR")
    bl = corner_r_std(img, "BL")
    br = corner_r_std(img, "BR")
    ratio = tr / tl if tl > 0 else None
    rows.append({
        **r,
        "tl_std": tl, "tr_std": tr, "bl_std": bl, "br_std": br,
        "tr_tl": ratio,
        "dims": img.size,
        "size_kb": r["size_bytes"] // 1024,
    })

# Add BASE reference (v7_tier_b shift=3.1/seed=42) with same metric
base_img = Image.open(BASE_PATH)
base_tl = corner_r_std(base_img, "TL")
base_tr = corner_r_std(base_img, "TR")
base_bl = corner_r_std(base_img, "BL")
base_br = corner_r_std(base_img, "BR")
base_ratio = base_tr / base_tl if base_tl > 0 else None
base_row = {
    "code": "BASE",
    "output": "BASE.png",
    "seed": 42,
    "cfg": 4.0,
    "steps": 40,
    "sampler_name": "euler",
    "scheduler": "simple",
    "model_sampling_shift": 3.1,
    "cfgnorm_strength": 1.0,
    "use_fp8": True,
    "ok": True,
    "wall_s": None,
    "render_s": None,
    "size_bytes": BASE_PATH.stat().st_size,
    "tl_std": base_tl, "tr_std": base_tr, "bl_std": base_bl, "br_std": base_br,
    "tr_tl": base_ratio,
    "dims": base_img.size,
    "size_kb": BASE_PATH.stat().st_size // 1024,
    "note": "v7_tier_b BASE reference (shift=3.1, seed=42) — same metric, included for comparison",
}
all_rows = [base_row] + rows

# Print markdown table
print("| Code | seed | Dims | TL std | TR std | BL std | BR std | TR/TL | render_s | wall_s | size_KB |")
print("|------|------|------|--------|--------|--------|--------|-------|----------|--------|---------|")
for r in all_rows:
    if r.get("tl_std") is None:
        print(f"| {r['code']} | {r.get('seed', '?')} | FAILED | – | – | – | – | – | – | "
              f"{fmt(r.get('wall_s'))} | – |")
        continue
    render = fmt(r.get("render_s")) if r.get("render_s") else "–"
    wall = fmt(r.get("wall_s")) if r.get("wall_s") else "–"
    print(f"| {r['code']} | {r.get('seed', '?')} | {r['dims'][0]}x{r['dims'][1]} | "
          f"{fmt(r['tl_std'])} | {fmt(r['tr_std'])} | {fmt(r['bl_std'])} | "
          f"{fmt(r['br_std'])} | {fmt(r['tr_tl'])} | {render} | {wall} | "
          f"{r['size_kb']} |")

# Save metrics JSON
metrics = {
    "trial": {
        "cells": rows,
        "common": MANIFEST["common"],
        "metric": "100x100 corner R-channel std-dev (R extracted from RGB; np.std of patch)",
        "spec_path": str(OUT_DIR / "spec.json"),
    },
    "reference": {
        **base_row,
        "source_path": str(BASE_PATH),
        "source_metrics_path": str(Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_tier_b/metrics.json")),
    },
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\nMetrics written to {OUT_DIR / 'metrics.json'}")
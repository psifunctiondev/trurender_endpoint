#!/usr/bin/env python3
"""Compute std-dev TR/TL asymmetry for each probe render and build comparison sheet."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROBE_DIR = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_1_probes")
MANIFEST = json.load(open(PROBE_DIR / "results_manifest.json"))

def split_halves(img: Image.Image):
    """Split image into 4 quadrants (TL, TR, BL, BR) and return std-dev of each."""
    w, h = img.size
    half_w, half_h = w // 2, h // 2
    arr = np.array(img.convert("L"), dtype=np.float32)
    TL = arr[0:half_h, 0:half_w]
    TR = arr[0:half_h, half_w:w]
    BL = arr[half_h:h, 0:half_w]
    BR = arr[half_h:h, half_w:w]
    return (
        float(np.std(TL)),
        float(np.std(TR)),
        float(np.std(BL)),
        float(np.std(BR)),
    )

def fmt(n): return f"{n:.2f}"

rows = []
for r in MANIFEST["results"]:
    p = PROBE_DIR / r["output"]
    if not r["ok"] or not p.exists():
        rows.append({
            **r,
            "tl_std": None, "tr_std": None, "bl_std": None, "br_std": None,
            "tr_tl": None, "dims": None,
        })
        continue
    img = Image.open(p)
    tl, tr, bl, br = split_halves(img)
    ratio = tr / tl if tl > 0 else None
    rows.append({
        **r,
        "tl_std": tl, "tr_std": tr, "bl_std": bl, "br_std": br,
        "tr_tl": ratio,
        "dims": img.size,
    })

# Print markdown table
print("| Code | Dims | TL std | TR std | BL std | BR std | TR/TL | render_s | wall_s | size_KB |")
print("|------|------|--------|--------|--------|--------|-------|----------|--------|---------|")
for r in rows:
    if r["tl_std"] is None:
        print(f"| {r['code']} | FAILED | – | – | – | – | – | – | {fmt(r['wall_s'])} | – |")
        continue
    print(f"| {r['code']} | {r['dims'][0]}x{r['dims'][1]} | {fmt(r['tl_std'])} | {fmt(r['tr_std'])} | "
          f"{fmt(r['bl_std'])} | {fmt(r['br_std'])} | {fmt(r['tr_tl'])} | "
          f"{fmt(r['render_s'])} | {fmt(r['wall_s'])} | {r['size_bytes']//1024} |")

# Save metrics JSON for later use
metrics = {"rows": rows, "common": MANIFEST["common"]}
with open(PROBE_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\nMetrics written to {PROBE_DIR / 'metrics.json'}")

#!/usr/bin/env python3
"""Compute TR/TL std-dev metrics for the 4 depth cells + the v7_tier_b P2 anchor.

Same metric as v7_tier_b/metrics.json: 100x100 corner patches, R channel only.
TR std / TL std ratio. Closer to 1.0 = more uniform. Lower TR std = less right-edge noise.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint")
TIER_C2 = ROOT / "outputs/tier_c_2_depth"
V7_TIER_B = ROOT / "outputs/v7_tier_b"
METRICS_OUT = TIER_C2 / "metrics.json"

PATCH = 100
CHANNEL = "R"  # R-only, matches v7_tier_b

# Cells: (code, image path)
cells = [
    ("P2-anchor (v7_tier_b/BASE)", V7_TIER_B / "BASE.png"),
    ("CTRL (depth=0.0)", TIER_C2 / "CTRL.png"),
    ("DS-0.3", TIER_C2 / "DS-0.3.png"),
    ("DS-0.5", TIER_C2 / "DS-0.5.png"),
    ("DS-0.7", TIER_C2 / "DS-0.7.png"),
]


def corner_patch_std(img_arr: np.ndarray, corner: str, patch: int = PATCH) -> float:
    h, w = img_arr.shape[:2]
    if corner == "TR":
        patch_arr = img_arr[:patch, w - patch:, 0]  # R channel, top-right 100x100
    elif corner == "TL":
        patch_arr = img_arr[:patch, :patch, 0]
    else:
        raise ValueError(corner)
    return float(np.std(patch_arr))


def main() -> int:
    results = {}
    print(f"=== TR/TL std-dev metrics (R channel, {PATCH}x{PATCH} corner patches) ===")
    for code, path in cells:
        if not path.exists():
            print(f"  {code}: SKIP (missing: {path})", file=sys.stderr)
            continue
        img = np.array(Image.open(path).convert("RGB"))
        tr = corner_patch_std(img, "TR")
        tl = corner_patch_std(img, "TL")
        ratio = tr / tl if tl > 0 else float("inf")
        size_bytes = path.stat().st_size
        results[code] = {
            "tr_r_std": round(tr, 3),
            "tl_r_std": round(tl, 3),
            "tr_tl_ratio": round(ratio, 4),
            "patch_size": PATCH,
            "image_w": img.shape[1],
            "image_h": img.shape[0],
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 1),
        }
        print(f"  {code:35s}: TR={tr:7.3f} TL={tl:7.3f} ratio={ratio:.4f}  size={size_bytes:>9,} bytes")
    out = {
        "patch_size": PATCH,
        "channel": CHANNEL,
        "metric": "TR std-dev / TL std-dev of 100x100 corner patches",
        "results": results,
    }
    METRICS_OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {METRICS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

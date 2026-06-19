#!/usr/bin/env python3
"""Compute 100x100 corner R-channel std-dev asymmetry for the wide-shift sweep.

Same shape as the prior probes (Tier-B & v6 reference). We open each PNG,
extract the top-right (TR) and top-left (TL) 100x100 corner patches, take
the R-channel std-dev of each, and record the TR/TL ratio. This is the
asymmetry signal Quinn has been tracking — it correlates with subtle
top-right artifacts from the diffusion sampling path.
"""

import json
import os
from PIL import Image
import numpy as np

OUT_DIR = "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/wide_shift"

# (label, shift value, png filename) — matches the spec.json order
CELLS = [
    ("SH-1.0", 1.0, "SH-1.0.png"),
    ("SH-2.0", 2.0, "SH-2.0.png"),
    ("SH-6.0", 6.0, "SH-6.0.png"),
    ("SH-8.0", 8.0, "SH-8.0.png"),
]

# Reference: Tier-B BASE (shift=3.1)
REFERENCE = {
    "code": "TIER-B-BASE",
    "label": "Tier-B BASE",
    "shift": 3.1,
    "path": "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_tier_b/BASE.png",
}


def corner_metrics(path: str) -> dict:
    """Return std-dev of R-channel in TL and TR 100x100 corners."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    arr = np.array(im)  # (H, W, 3)
    # Top-Left 100x100
    tl = arr[0:100, 0:100, 0]  # R-channel
    # Top-Right 100x100
    tr = arr[0:100, w - 100:w, 0]
    return {
        "tl_std": float(np.std(tl)),
        "tr_std": float(np.std(tr)),
        "tr_tl_ratio": float(np.std(tr) / np.std(tl)) if np.std(tl) > 0 else None,
    }


def main() -> None:
    out = {
        "spec": "100x100 corner R-channel std-dev asymmetry (TL vs TR). "
                "TR/TL ratio >1.0 means the top-right corner is noisier than the top-left.",
        "patch_size": 100,
        "channel": "R",
        "cells": [],
    }

    # Reference cell first
    ref_metrics = corner_metrics(REFERENCE["path"])
    out["cells"].append({
        "code": REFERENCE["code"],
        "label": REFERENCE["label"],
        "model_sampling_shift": REFERENCE["shift"],
        "path": REFERENCE["path"],
        **ref_metrics,
    })

    # Wide-shift cells
    for code, shift, fname in CELLS:
        path = os.path.join(OUT_DIR, fname)
        m = corner_metrics(path)
        out["cells"].append({
            "code": code,
            "label": code,
            "model_sampling_shift": shift,
            "path": path,
            **m,
        })

    # Save
    out_path = os.path.join(OUT_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")

    # Pretty print summary table
    print()
    print(f"{'code':<14} {'shift':>6}  {'TL std':>8}  {'TR std':>8}  {'TR/TL':>7}")
    for c in out["cells"]:
        ratio = c["tr_tl_ratio"]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "  N/A "
        print(f"{c['code']:<14} {c['model_sampling_shift']:>6.1f}  "
              f"{c['tl_std']:>8.2f}  {c['tr_std']:>8.2f}  {ratio_str:>7}")


if __name__ == "__main__":
    main()

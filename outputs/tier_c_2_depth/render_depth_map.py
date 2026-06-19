#!/usr/bin/env python3
"""Render a depth map of enscape_input.png using Depth Anything V2 (small).

Output: outputs/tier_c_2_depth/depth_map.png (grayscale, normalized 0-255).

This is purely for visual inspection by Quinn — the actual TruRender
workflow passes the source Enscape image directly to the DiffSynth
ControlNet's image input (QwenImageDiffsynthControlnet handles its
own internal depth extraction from that RGB image).

Model: depth-anything/Depth-Anything-V2-Small-hf (HuggingFace)
Why: small + fast + runs on Mac CPU in ~10 seconds. The actual
ControlNet sees a more complex depth encoding, but this preview
shows what the *content* of "depth" looks like for this scene.
"""
import sys
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

ROOT = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint")
INPUT = ROOT / "inputs/enscape_input.png"
OUTPUT_DIR = ROOT / "outputs/tier_c_2_depth"
OUTPUT = OUTPUT_DIR / "depth_map.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: input not found: {INPUT}", file=sys.stderr)
        return 1

    print(f"[depth_map] input: {INPUT}")
    print(f"[depth_map] loading Depth Anything V2 Small (HuggingFace)...")
    device = "cpu"  # safe; Mac CPU is fine for this small model
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )

    print(f"[depth_map] running inference on {INPUT}...")
    img = Image.open(INPUT).convert("RGB")
    result = depth_pipe(img)
    depth = np.array(result["depth"])  # H x W, uint8 or float
    if depth.dtype != np.uint8:
        # Normalize to 0-255
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = ((depth - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            depth = np.zeros_like(depth, dtype=np.uint8)

    # Apply a colormap so the depth is visually interpretable (closer = warmer)
    depth_color = Image.fromarray(depth).convert("RGB")
    # Resize to original resolution if needed
    if depth_color.size != img.size:
        depth_color = depth_color.resize(img.size, Image.BILINEAR)
    depth_color.save(OUTPUT)
    print(f"[depth_map] saved: {OUTPUT} ({depth_color.size[0]}x{depth_color.size[1]})")
    # Also save the raw grayscale version
    raw_out = OUTPUT_DIR / "depth_map_gray.png"
    Image.fromarray(depth).resize(img.size, Image.BILINEAR).save(raw_out)
    print(f"[depth_map] saved grayscale: {raw_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

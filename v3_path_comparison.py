#!/usr/bin/env python3
"""
TruRender v3 Path Comparison: Spatially-varying denoise vs Post-processing

Path A: Dual-pass inpainting with mask-based compositing
  - Pass 1: d0.93 + photo prompt (aggressive, transforms flat CG surfaces)
  - Pass 2: d0.83 + hybrid prompt (conservative, preserves objects/details)
  - Mask from edge detection separates flat areas from detailed areas
  - Composite: flat surfaces from Pass 1, objects from Pass 2

Path B: d0.83 render + photographic post-processing
  - Uses /render/photo endpoint (render + DoF, vignette, grain, warmth, CA)

Baseline: d0.83 render with hybrid prompt (no tricks)

Output:
  - outputs/v3_path_comparison.png  (4-column comparison)
  - outputs/v3_path_a_mask.png      (the compositing mask)
  - outputs/v3_path_a_composite.png (full-res Path A result)
  - outputs/v3_path_b_photo.png     (full-res Path B result)
  - Stats: mean pixel diff from source per path, broken down by zone
"""

import io
import os
import sys
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RENDER_URL = "https://psifunctiondev--trurender-comfyui-trurendercomfyui-web.modal.run/render"
PHOTO_URL = "https://psifunctiondev--trurender-comfyui-trurendercomfyui-web.modal.run/render/photo"
HEALTH_URL = "https://psifunctiondev--trurender-comfyui-trurendercomfyui-web.modal.run/health"

INPUT_PATH = "inputs/enscape_input.png"
OUTPUT_DIR = "outputs"

SEED = 2718
CN_STRENGTH = 0.65

# Timeouts — cold start can take ~2min, render ~30s
TIMEOUT = 300  # 5 min per request

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: call the render endpoint
# ---------------------------------------------------------------------------

def render(image_path: str, seed: int, denoise: float,
           controlnet_strength: float = CN_STRENGTH,
           prompt_style: str = "hybrid",
           timeout: int = TIMEOUT) -> bytes:
    """Call the /render endpoint. Returns PNG bytes."""
    with open(image_path, "rb") as f:
        files = {"image": ("input.png", f, "image/png")}
        data = {
            "seed": str(seed),
            "denoise": str(denoise),
            "controlnet_strength": str(controlnet_strength),
            "prompt_style": prompt_style,
        }
        print(f"  → Rendering: seed={seed}, d={denoise}, cn={controlnet_strength}, style={prompt_style}")
        t0 = time.time()
        resp = requests.post(RENDER_URL, files=files, data=data, timeout=timeout)
        elapsed = time.time() - t0
        resp.raise_for_status()
        print(f"  ← Done in {elapsed:.1f}s ({len(resp.content)/(1024*1024):.1f} MB)")
        return resp.content


def render_photo(image_path: str, seed: int, denoise: float,
                 controlnet_strength: float = CN_STRENGTH,
                 dof_strength: float = 0.6,
                 vignette_strength: float = 0.3,
                 chromatic_aberration: float = 0.4,
                 grain_strength: float = 0.15,
                 warmth: float = 0.2,
                 timeout: int = TIMEOUT) -> bytes:
    """Call the /render/photo endpoint. Returns PNG bytes."""
    with open(image_path, "rb") as f:
        files = {"image": ("input.png", f, "image/png")}
        data = {
            "seed": str(seed),
            "denoise": str(denoise),
            "controlnet_strength": str(controlnet_strength),
            "dof_strength": str(dof_strength),
            "vignette_strength": str(vignette_strength),
            "chromatic_aberration": str(chromatic_aberration),
            "grain_strength": str(grain_strength),
            "warmth": str(warmth),
        }
        print(f"  → Photo render: seed={seed}, d={denoise}, cn={controlnet_strength}")
        t0 = time.time()
        resp = requests.post(PHOTO_URL, files=files, data=data, timeout=timeout)
        elapsed = time.time() - t0
        resp.raise_for_status()
        print(f"  ← Done in {elapsed:.1f}s ({len(resp.content)/(1024*1024):.1f} MB)")
        return resp.content


# ---------------------------------------------------------------------------
# Mask generation: separate flat surfaces from objects/details
# ---------------------------------------------------------------------------

def generate_flat_surface_mask(image_path: str, output_size: tuple = None) -> Image.Image:
    """
    Generate a mask where WHITE = flat surfaces (walls, ceiling, floor)
    and BLACK = detailed objects (furniture, fixtures, textures).

    Strategy:
    1. Convert to grayscale
    2. Apply Canny-like edge detection (Sobel gradients)
    3. Dilate edges to cover object boundaries
    4. Invert: high-edge areas become black (keep conservative render),
       low-edge areas become white (use aggressive render)
    5. Gaussian blur for smooth blending at boundaries
    """
    img = Image.open(image_path).convert("L")

    # If output_size specified, resize to match render output
    if output_size:
        img = img.resize(output_size, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # --- Sobel edge detection ---
    # Horizontal and vertical Sobel filters
    from scipy.ndimage import sobel, gaussian_filter, maximum_filter

    gx = sobel(arr, axis=1)  # horizontal edges
    gy = sobel(arr, axis=0)  # vertical edges
    magnitude = np.sqrt(gx**2 + gy**2)

    # Normalize to 0-1
    magnitude = magnitude / (magnitude.max() + 1e-6)

    # --- Also use local variance as a texture detector ---
    # High local variance = detailed texture (furniture, patterns)
    # Low local variance = flat surface (walls, ceiling)
    local_mean = gaussian_filter(arr, sigma=8)
    local_sq_mean = gaussian_filter(arr**2, sigma=8)
    local_var = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
    local_var = local_var / (local_var.max() + 1e-6)

    # Combine edge magnitude and local variance
    # Both contribute to "detail" detection
    detail_map = np.maximum(magnitude, local_var * 0.7)

    # --- Dilate the detail regions ---
    # Use maximum filter to expand detail areas (catches object boundaries)
    detail_dilated = maximum_filter(detail_map, size=25)

    # --- Threshold: flat vs detailed ---
    # Low detail_dilated → flat surface (white in mask → use aggressive pass)
    # High detail_dilated → detailed area (black in mask → use conservative pass)
    threshold = 0.15  # tuned: below this is "flat"
    flat_mask = (detail_dilated < threshold).astype(np.float32)

    # --- Gaussian blur for smooth blending ---
    flat_mask = gaussian_filter(flat_mask, sigma=20)

    # Convert to PIL Image
    mask_uint8 = (flat_mask * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L")


# ---------------------------------------------------------------------------
# Zone masks for stats (ceiling / furniture / floor)
# ---------------------------------------------------------------------------

def get_zone_masks(h: int, w: int) -> dict:
    """
    Approximate zone masks for the architectural interior.
    Based on typical Enscape interior: ceiling ~top 25%, floor ~bottom 20%,
    furniture = everything else.
    These are rough approximations for stats purposes.
    """
    ceiling = np.zeros((h, w), dtype=bool)
    ceiling[:int(h * 0.25), :] = True

    floor = np.zeros((h, w), dtype=bool)
    floor[int(h * 0.80):, :] = True

    furniture = np.ones((h, w), dtype=bool)
    furniture[:int(h * 0.25), :] = False
    furniture[int(h * 0.80):, :] = False

    return {"ceiling": ceiling, "furniture": furniture, "floor": floor}


def compute_stats(source: np.ndarray, rendered: np.ndarray, label: str):
    """Compute and print mean pixel difference stats by zone."""
    h, w = source.shape[:2]
    zones = get_zone_masks(h, w)

    # Overall
    diff = np.abs(source.astype(np.float32) - rendered.astype(np.float32))
    mean_diff = diff.mean()
    print(f"\n  {label}:")
    print(f"    Overall mean pixel diff: {mean_diff:.2f}")

    for zone_name, zone_mask in zones.items():
        zone_diff = diff[zone_mask].mean()
        print(f"    {zone_name:12s}: {zone_diff:.2f}")

    return mean_diff


# ---------------------------------------------------------------------------
# Comparison image builder
# ---------------------------------------------------------------------------

def build_comparison(source: Image.Image, path_a: Image.Image,
                     path_b: Image.Image, baseline: Image.Image,
                     output_path: str, thumb_width: int = 480):
    """Build a 4-column comparison image with labels."""
    images = [source, path_a, path_b, baseline]
    labels = [
        "Source (Enscape)",
        "Path A: Dual-pass\n(d0.93 flat + d0.83 detail)",
        "Path B: Render+PostProcess\n(d0.83 + photo FX)",
        "Baseline: d0.83\n(hybrid, no tricks)",
    ]

    # Calculate dimensions
    aspect = source.height / source.width
    thumb_h = int(thumb_width * aspect)
    label_height = 60
    padding = 10
    header_height = 40

    total_w = 4 * thumb_width + 5 * padding
    total_h = header_height + thumb_h + label_height + 2 * padding

    # Create canvas
    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 25))
    draw = ImageDraw.Draw(canvas)

    # Try to get a decent font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

    # Title
    draw.text((padding, 8), "TruRender v3 — Path Comparison (seed 2718)",
              fill=(240, 100, 58), font=title_font)

    # Paste thumbnails and labels
    for i, (img, label) in enumerate(zip(images, labels)):
        x = padding + i * (thumb_width + padding)
        y = header_height + padding

        thumb = img.resize((thumb_width, thumb_h), Image.LANCZOS)
        canvas.paste(thumb, (x, y))

        # Label below
        for j, line in enumerate(label.split("\n")):
            draw.text((x + 5, y + thumb_h + 4 + j * 18), line,
                      fill=(200, 200, 200), font=font)

    canvas.save(output_path)
    print(f"\n✓ Comparison saved: {output_path} ({canvas.size[0]}x{canvas.size[1]})")
    return canvas


# ===========================================================================
# Main
# ===========================================================================

def main():
    total_start = time.time()

    print("=" * 70)
    print("TruRender v3 — Path Comparison")
    print("=" * 70)

    # --- Check health ---
    print("\n[1/7] Checking endpoint health...")
    try:
        resp = requests.get(HEALTH_URL, timeout=10)
        health = resp.json()
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  GPU: {health.get('gpu', 'unknown')}")
    except Exception as e:
        print(f"  Health check failed: {e}")
        print("  Endpoint may be cold-starting. Proceeding anyway (first render may take ~2min)...")

    # --- Load source image ---
    print(f"\n[2/7] Loading source image: {INPUT_PATH}")
    source_img = Image.open(INPUT_PATH)
    print(f"  Size: {source_img.size[0]}x{source_img.size[1]}")

    # The renders come back at 1536x864, so we need source at that size for comparison
    render_size = (1536, 864)
    source_resized = source_img.resize(render_size, Image.LANCZOS)

    # --- Generate mask for Path A ---
    print(f"\n[3/7] Generating flat-surface mask...")
    mask = generate_flat_surface_mask(INPUT_PATH, output_size=render_size)
    mask_path = os.path.join(OUTPUT_DIR, "v3_path_a_mask.png")
    mask.save(mask_path)
    print(f"  Mask saved: {mask_path}")
    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    pct_flat = (mask_arr > 0.5).mean() * 100
    print(f"  Flat surface coverage: {pct_flat:.1f}%")

    # --- Path A Pass 1: Aggressive render (d0.93, photo prompt) ---
    print(f"\n[4/7] Path A Pass 1 — aggressive render (d=0.93, photo prompt)...")
    aggressive_bytes = render(INPUT_PATH, seed=SEED, denoise=0.93,
                              prompt_style="photo")
    aggressive_img = Image.open(io.BytesIO(aggressive_bytes))
    aggressive_img.save(os.path.join(OUTPUT_DIR, "v3_path_a_pass1_aggressive.png"))
    print(f"  Pass 1 size: {aggressive_img.size}")

    # --- Path A Pass 2: Conservative render (d0.83, hybrid prompt) ---
    print(f"\n[5/7] Path A Pass 2 — conservative render (d=0.83, hybrid prompt)...")
    conservative_bytes = render(INPUT_PATH, seed=SEED, denoise=0.83,
                                prompt_style="hybrid")
    conservative_img = Image.open(io.BytesIO(conservative_bytes))
    conservative_img.save(os.path.join(OUTPUT_DIR, "v3_path_a_pass2_conservative.png"))
    print(f"  Pass 2 size: {conservative_img.size}")

    # --- Composite Path A ---
    print(f"\n  Compositing Path A...")
    # Ensure mask matches render size
    if mask.size != aggressive_img.size:
        mask = mask.resize(aggressive_img.size, Image.LANCZOS)
        mask_arr = np.array(mask, dtype=np.float32) / 255.0

    agg_arr = np.array(aggressive_img, dtype=np.float32)
    con_arr = np.array(conservative_img, dtype=np.float32)
    mask_3d = mask_arr[:, :, np.newaxis]

    # White mask = use aggressive (flat surfaces), Black = use conservative (details)
    composite_arr = agg_arr * mask_3d + con_arr * (1 - mask_3d)
    composite_arr = composite_arr.clip(0, 255).astype(np.uint8)
    path_a_img = Image.fromarray(composite_arr)
    path_a_path = os.path.join(OUTPUT_DIR, "v3_path_a_composite.png")
    path_a_img.save(path_a_path)
    print(f"  Composite saved: {path_a_path}")

    # --- Path B: Render + post-processing ---
    print(f"\n[6/7] Path B — render + post-processing (d=0.83, photo FX)...")
    photo_bytes = render_photo(INPUT_PATH, seed=SEED, denoise=0.83,
                               dof_strength=0.4,       # lighter DoF
                               vignette_strength=0.25,
                               chromatic_aberration=0.3,
                               grain_strength=0.12,
                               warmth=0.15)
    path_b_img = Image.open(io.BytesIO(photo_bytes))
    path_b_path = os.path.join(OUTPUT_DIR, "v3_path_b_photo.png")
    path_b_img.save(path_b_path)
    print(f"  Path B saved: {path_b_path}")

    # --- Baseline is the conservative render (d0.83, hybrid, no tricks) ---
    baseline_img = conservative_img  # already rendered in Pass 2

    # --- Build comparison ---
    print(f"\n[7/7] Building comparison image...")
    comparison_path = os.path.join(OUTPUT_DIR, "v3_path_comparison.png")
    build_comparison(source_resized, path_a_img, path_b_img, baseline_img,
                     comparison_path, thumb_width=480)

    # --- Stats ---
    print("\n" + "=" * 70)
    print("STATS: Mean pixel difference from source (per zone)")
    print("=" * 70)

    src_arr = np.array(source_resized)

    # Resize renders to match source_resized if needed
    def to_array(img):
        if img.size != render_size:
            img = img.resize(render_size, Image.LANCZOS)
        return np.array(img)

    compute_stats(src_arr, to_array(path_a_img), "Path A (dual-pass composite)")
    compute_stats(src_arr, to_array(path_b_img), "Path B (render + post-process)")
    compute_stats(src_arr, to_array(baseline_img), "Baseline (d0.83 hybrid)")

    # Aggressive vs conservative individual stats
    compute_stats(src_arr, to_array(aggressive_img), "Pass 1 alone (d0.93 photo)")
    compute_stats(src_arr, to_array(conservative_img), "Pass 2 alone (d0.83 hybrid)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    print(f"\nOutputs:")
    print(f"  {comparison_path}")
    print(f"  {mask_path}")
    print(f"  {path_a_path}")
    print(f"  {path_b_path}")


if __name__ == "__main__":
    main()

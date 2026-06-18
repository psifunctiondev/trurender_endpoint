"""
TruRender v6.0 - ComfyUI-on-Modal for architectural render → photorealistic conversion.

Uses the actual ComfyUI workflow with:
  - Flux.1-dev (UNET)
  - XLabs Depth ControlNet V3 (preserves spatial layout / 3D structure)
  - XLabs Canny ControlNet V3 (preserves edges / architectural lines)
  - XLabs HED ControlNet V3 (soft edges / gradients, optional)
  - Depth Anything ViT-L for depth extraction
  - Canny edge detection for line extraction
  - Single-pass KSampler with configurable denoise
  - DPM++ 2M Karras sampler, cfg 3.5
  - Triple ControlNet: Depth + Canny + HED (configurable)
  - InstantX/Shakker-Labs IP-Adapter for style transfer (128 tokens, 57 blocks)
  - SigLIP-so400m vision encoder for style conditioning

Deploy:   cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal deploy trurender_comfyui.py
Models:   cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_comfyui.py::download_models
Styles:   cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_comfyui.py::upload_styles
"""

import modal
import json
import io
import base64
import time
import uuid
import os
import threading
from pathlib import Path


# ---------------------------------------------------------------------------
# Flat-surface mask generator for spatially-varying denoise
# ---------------------------------------------------------------------------

def generate_flat_surface_mask(image_bytes: bytes, flat_denoise_strength: float = 0.3,
                                blur_radius: int = 15, variance_window: int = 16) -> bytes:
    """
    Generate a grayscale mask identifying flat/featureless regions in an image.

    Returns PNG bytes of a grayscale mask where:
      - White (255) = flat regions → will receive MORE diffusion noise
      - Black (0) = detail-rich regions → will receive LESS noise (protected)

    The mask is used with SetLatentNoiseMask in ComfyUI to apply spatially-varying
    denoise. flat_denoise_strength controls the overall intensity of the mask.

    Algorithm:
      1. Convert to grayscale
      2. Compute local variance in sliding windows (low variance = flat)
      3. Detect edges via Sobel-like gradient magnitude (high gradient = detail)
      4. Combine: flat_score = (1 - normalized_variance) * (1 - normalized_edges)
      5. Apply strength scaling and Gaussian blur for smooth transitions

    Args:
        image_bytes: Source image as PNG/JPEG bytes
        flat_denoise_strength: Overall mask intensity (0.0 = fully black/no-op, 1.0 = maximum)
        blur_radius: Gaussian blur radius for smoothing mask boundaries
        variance_window: Window size for local variance computation
    """
    import numpy as np
    from PIL import Image as PILImage, ImageFilter

    img = PILImage.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # --- 1. Local variance (sliding window) ---
    # Pad array for windowed computation
    pad = variance_window // 2
    # Use a strided approach for efficiency: compute mean and mean-of-squares
    # Then variance = E[x^2] - (E[x])^2
    from numpy.lib.stride_tricks import sliding_window_view

    # Pad with edge values to handle borders
    padded = np.pad(arr, pad, mode='edge')
    windows = sliding_window_view(padded, (variance_window, variance_window))
    # windows shape: (h, w, window, window)
    local_mean = windows.mean(axis=(-2, -1))
    local_var = windows.var(axis=(-2, -1))

    # Normalize variance to [0, 1] — use 95th percentile to avoid outlier domination
    # sliding_window_view on padded array produces (h+1, w+1) — trim back to (h, w)
    local_var = local_var[:h, :w]
    var_cap = np.percentile(local_var, 95)
    if var_cap > 0:
        norm_var = np.clip(local_var / var_cap, 0, 1)
    else:
        norm_var = np.zeros_like(local_var)

    # --- 2. Edge detection (Sobel-like gradient magnitude) ---
    # Horizontal and vertical gradients using numpy
    # Sobel kernels approximated via finite differences
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]  # horizontal gradient
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]  # vertical gradient
    gradient_mag = np.sqrt(gx**2 + gy**2)

    # Normalize gradient to [0, 1]
    grad_cap = np.percentile(gradient_mag, 95)
    if grad_cap > 0:
        norm_grad = np.clip(gradient_mag / grad_cap, 0, 1)
    else:
        norm_grad = np.zeros_like(gradient_mag)

    # --- 3. Combine: flat regions score high ---
    # flat_score = (1 - variance) * (1 - edges)
    # Both low variance AND low edges → flat surface
    flat_score = (1.0 - norm_var) * (1.0 - norm_grad)

    # --- 4. Scale by strength and convert to mask ---
    mask = (flat_score * flat_denoise_strength * 255).clip(0, 255).astype(np.uint8)

    # --- 5. Gaussian blur for smooth transitions ---
    mask_img = PILImage.fromarray(mask, mode='L')
    if blur_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Return as PNG bytes
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    return buf.getvalue()


def generate_flat_surface_mask_at_size(image_bytes: bytes, target_w: int, target_h: int,
                                       flat_denoise_strength: float = 0.3,
                                       blur_radius: int = 15,
                                       variance_window: int = 16) -> bytes:
    """
    Generate a flat-surface mask resized to exactly target_w × target_h.
    Calls generate_flat_surface_mask then resizes to match render target dimensions.
    This ensures the mask aligns with the latent space (target_w/8 × target_h/8).
    """
    from PIL import Image as PILImage
    mask_bytes = generate_flat_surface_mask(
        image_bytes,
        flat_denoise_strength=flat_denoise_strength,
        blur_radius=blur_radius,
        variance_window=variance_window,
    )
    mask_img = PILImage.open(io.BytesIO(mask_bytes)).convert('L')
    if mask_img.size != (target_w, target_h):
        mask_img = mask_img.resize((target_w, target_h), PILImage.LANCZOS)
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Prompts (preserved exactly from the working ComfyUI workflow)
# ---------------------------------------------------------------------------

# --- Prompt Library ---
# "preserve" = original conservative prompt (fights transformation on flat surfaces)
# "photo" = photographic realism (encourages real-world material character)
# "hybrid" = architectural accuracy + photographic materiality
PROMPT_LIBRARY = {
    "preserve": {
        "positive": (
            "photo-realistic architectural interior photograph, physically accurate lighting, "
            "real-world materials, natural reflections, correct scale and proportions, "
            "true-to-life textures, unchanged architecture, unchanged materials, unchanged colors, "
            "professional architectural photography, preserve original linework, "
            "preserve exact edges and contours"
        ),
        "negative": (
            "stylized, artistic, concept art, unrealistic lighting, overexposed, oversharpened, "
            "extra objects, people, cars, plants, decor changes, furniture changes, material changes, "
            "color shift, texture replacement, hallucinated details, blur, noise, grain, "
            "cinematic look, dramatic lighting, fantasy, contrast change, shadow change, "
            "lighting change, color grading, tone mapping, warm lighting, moody lighting, "
            "reinterpretation of shapes, softening of architectural lines"
        ),
    },
    "photo": {
        "positive": (
            "interior photograph shot on Hasselblad medium format camera, natural window light "
            "with subtle falloff and soft shadows, real plaster walls with faint texture variation, "
            "visible wood grain in furniture, natural fabric weave and subtle creasing in upholstery, "
            "real hardwood floor with micro-scratches and patina, slight dust motes in light beams, "
            "gentle lens vignetting, true optical depth of field, color captured through glass optics, "
            "subtle warm color cast from natural daylight, professional architectural photography "
            "for high-end design magazine, lived-in elegance"
        ),
        "negative": (
            "3d render, CGI, computer generated, perfect surfaces, uniform flat textures, "
            "plastic looking materials, impossible lighting, floating objects, extra furniture, "
            "people, animals, text, watermark, cartoon, painting, illustration, "
            "oversaturated, HDR tonemapping, extreme contrast, neon colors"
        ),
    },
    "hybrid": {
        "positive": (
            "professional architectural interior photograph, Hasselblad medium format, "
            "preserving exact room layout and furniture placement, same color palette, "
            "but with real-world material quality: visible wood grain, natural fabric texture "
            "with subtle creases and weave, plaster walls with faint imperfections, "
            "real glass reflections with slight impurity, hardwood with natural patina, "
            "soft natural window light with gentle falloff across walls, "
            "subtle lens characteristics, true-to-life material depth, "
            "photographed for Architectural Digest"
        ),
        "negative": (
            "3d render, CGI, computer generated, perfect uniform surfaces, plastic materials, "
            "extra objects, people, animals, decor changes, furniture changes, layout changes, "
            "different color scheme, cartoon, painting, illustration, text, watermark, "
            "oversaturated, extreme HDR, neon, fantasy lighting"
        ),
    },
}

# Default prompt style
DEFAULT_PROMPT_STYLE = "hybrid"
POSITIVE_PROMPT = PROMPT_LIBRARY[DEFAULT_PROMPT_STYLE]["positive"]
NEGATIVE_PROMPT = PROMPT_LIBRARY[DEFAULT_PROMPT_STYLE]["negative"]

# ---------------------------------------------------------------------------
# Modal image: ComfyUI + custom nodes + deps
# ---------------------------------------------------------------------------

comfyui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget", "curl")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
    )
    .pip_install(
        # diffusers with RMSNorm (required by Shakker-Labs IPAdapter-Flux attention_processor.py)
        # Must be installed BEFORE the custom node pip install to avoid version conflicts
        "diffusers>=0.30.0",
        "einops>=0.8.0",
        "transformers>=4.45.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.5",
    )
    .run_commands(
        # Install ComfyUI (pinned to known-good commit for IPAdapter-Flux compatibility)
        "git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui",
        "cd /comfyui && git checkout bda1482 || true",  # pin to commit recommended by Shakker-Labs
        "cd /comfyui && pip install -r requirements.txt",
        # Install comfyui_controlnet_aux (provides DepthAnythingPreprocessor)
        "cd /comfyui/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /comfyui/custom_nodes/comfyui_controlnet_aux && pip install -r requirements.txt",
        # Install Shakker-Labs IP-Adapter Flux nodes (InstantX implementation — 128 tokens, 57 blocks)
        # Pin to last known-good commit before June 2025 advanced node changes
        "cd /comfyui/custom_nodes && git clone https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux.git",
        "cd /comfyui/custom_nodes/ComfyUI-IPAdapter-Flux && git checkout 57ae7f9",  # 2025-05-16: stable clip_vision path update
        # Reinstall requirements after checkout to ensure pinned deps are current
        "cd /comfyui/custom_nodes/ComfyUI-IPAdapter-Flux && pip install -r requirements.txt",
        # Patch Shakker-Labs for ComfyUI bda1482 compatibility:
        # Patch 1: DoubleStreamBlock no longer has flipped_img_txt attribute (always False now)
        "sed -i 's/self.flipped_img_txt = original_block.flipped_img_txt/self.flipped_img_txt = getattr(original_block, \"flipped_img_txt\", False)/' /comfyui/custom_nodes/ComfyUI-IPAdapter-Flux/flux/layers.py",
        # NOTE: Patch 2 (adding **kwargs) was removed — it caused SyntaxError (kwargs before positional args)
        # bda1482 ComfyUI doesn't need this patch
        # NOTE: Build-time IPA import test removed — build containers have no GPU,
        # so model_management.py fails at import time (torch.cuda.current_device() with no driver).
        # Runtime import is validated via the startup diagnostic subprocess call.
        # Create model directories
        "mkdir -p /comfyui/models/unet /comfyui/models/vae /comfyui/models/clip /comfyui/models/controlnet",
        # IP-Adapter model directories (InstantX + SigLIP)
        "mkdir -p /comfyui/models/ipadapter-flux /comfyui/models/clip_vision/siglip-so400m-patch14-384",
    )
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "Pillow>=10.0.0",
        "aiohttp",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "COMFYUI_DIR": "/comfyui",
    })
)

# ---------------------------------------------------------------------------
# Modal app + volume
# ---------------------------------------------------------------------------

app = modal.App("trurender-comfyui")
model_volume = modal.Volume.from_name("trurender-model-cache", create_if_missing=True)

VOLUME_PATH = "/models"
COMFYUI_DIR = "/comfyui"
COMFYUI_PORT = 8188

# ---------------------------------------------------------------------------
# Model download specs
# ---------------------------------------------------------------------------

MODEL_SPECS = [
    {
        "name": "flux1-dev.safetensors",
        "repo": "black-forest-labs/FLUX.1-dev",
        "filename": "flux1-dev.safetensors",
        "dest": "comfyui_models/unet",
    },
    {
        "name": "ae.safetensors",
        "repo": "black-forest-labs/FLUX.1-dev",
        "filename": "ae.safetensors",
        "dest": "comfyui_models/vae",
    },
    {
        "name": "t5xxl_fp16.safetensors",
        "repo": "comfyanonymous/flux_text_encoders",
        "filename": "t5xxl_fp16.safetensors",
        "dest": "comfyui_models/clip",
    },
    {
        "name": "clip_l.safetensors",
        "repo": "comfyanonymous/flux_text_encoders",
        "filename": "clip_l.safetensors",
        "dest": "comfyui_models/clip",
    },
    {
        "name": "flux-depth-controlnet-v3.safetensors",
        "repo": "XLabs-AI/flux-controlnet-depth-v3",
        "filename": "flux-depth-controlnet-v3.safetensors",
        "dest": "comfyui_models/controlnet",
    },
    {
        "name": "flux-canny-controlnet-v3.safetensors",
        "repo": "XLabs-AI/flux-controlnet-canny-v3",
        "filename": "flux-canny-controlnet-v3.safetensors",
        "dest": "comfyui_models/controlnet",
    },
    {
        "name": "flux-hed-controlnet-v3.safetensors",
        "repo": "XLabs-AI/flux-controlnet-hed-v3",
        "filename": "flux-hed-controlnet-v3.safetensors",
        "dest": "comfyui_models/controlnet",
    },
    {
        "name": "depth_anything_vitl14.pth",
        "repo": "LiheYoung/Depth-Anything",
        "filename": "checkpoints/depth_anything_vitl14.pth",
        "dest": "annotator_ckpts",
        "dest_filename": "depth_anything_vitl14.pth",
        "repo_type": "space",
    },
    # InstantX IP-Adapter for Flux (Shakker-Labs, 128 tokens, 57 blocks)
    {
        "name": "ip-adapter.bin",
        "repo": "InstantX/FLUX.1-dev-IP-Adapter",
        "filename": "ip-adapter.bin",
        "dest": "comfyui_models/ipadapter-flux",
    },
    # SigLIP Vision Encoder (required by InstantX IP-Adapter)
    # Note: stored with siglip- prefix on volume; _setup_model_links symlinks them to original names
    {
        "name": "siglip-model.safetensors",
        "repo": "google/siglip-so400m-patch14-384",
        "filename": "model.safetensors",
        "dest": "comfyui_models/clip_vision/siglip-so400m-patch14-384",
    },
    {
        "name": "siglip-config.json",
        "repo": "google/siglip-so400m-patch14-384",
        "filename": "config.json",
        "dest": "comfyui_models/clip_vision/siglip-so400m-patch14-384",
    },
    {
        "name": "siglip-preprocessor_config.json",
        "repo": "google/siglip-so400m-patch14-384",
        "filename": "preprocessor_config.json",
        "dest": "comfyui_models/clip_vision/siglip-so400m-patch14-384",
    },
]


# ---------------------------------------------------------------------------
# One-time model download function
# ---------------------------------------------------------------------------

@app.function(
    image=comfyui_image,
    volumes={VOLUME_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=3600,  # models are large, allow 1 hour
)
def download_models():
    """Download all required models to the volume. Run once before first deploy."""
    from huggingface_hub import hf_hub_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[TruRender] Authenticated with HuggingFace")

    for spec in MODEL_SPECS:
        dest_dir = os.path.join(VOLUME_PATH, spec["dest"])
        dest_filename = spec.get("dest_filename", spec["name"])
        dest_path = os.path.join(dest_dir, dest_filename)

        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"[TruRender] ✓ {spec['name']} already exists ({size_mb:.0f} MB)")
            continue

        print(f"[TruRender] Downloading {spec['name']} from {spec['repo']}...")
        os.makedirs(dest_dir, exist_ok=True)

        kwargs = {
            "repo_id": spec["repo"],
            "filename": spec["filename"],
            "token": hf_token,
            "local_dir": f"/tmp/hf_download_{spec['name']}",
        }
        if "repo_type" in spec:
            kwargs["repo_type"] = spec["repo_type"]
        downloaded = hf_hub_download(**kwargs)
        # Move to destination
        import shutil
        shutil.move(downloaded, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[TruRender] ✓ {spec['name']} downloaded ({size_mb:.0f} MB)")

    model_volume.commit()
    print("[TruRender] All models downloaded and committed to volume.")

    # List volume contents
    for root, dirs, files in os.walk(VOLUME_PATH):
        for f in files:
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fpath} ({size_mb:.0f} MB)")


@app.function(
    image=comfyui_image,
    volumes={VOLUME_PATH: model_volume},
    timeout=300,
)
def upload_style_references(style_data: list[dict] = None):
    """Upload default style reference photos to the Modal volume.

    Args:
        style_data: list of {"name": str, "data_b64": str} dicts. If None, lists existing styles.

    Usage (from local entrypoint):
        modal run trurender_comfyui.py::upload_styles
    """
    style_dir = os.path.join(VOLUME_PATH, "style_references")
    os.makedirs(style_dir, exist_ok=True)

    if not style_data:
        # List existing style references
        existing = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(style_dir) else []
        print(f"[TruRender] Style references directory: {style_dir}")
        print(f"[TruRender] Existing styles: {len(existing)}")
        for f in existing:
            size_kb = os.path.getsize(os.path.join(style_dir, f)) / 1024
            print(f"  - {f} ({size_kb:.0f} KB)")
        return

    for item in style_data:
        name = item["name"]
        data = base64.b64decode(item["data_b64"])
        dest = os.path.join(style_dir, name)
        with open(dest, "wb") as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"[TruRender] \u2713 Uploaded {name} ({size_kb:.0f} KB)")

    model_volume.commit()
    print(f"[TruRender] {len(style_data)} style references uploaded and committed.")


@app.local_entrypoint(name="upload_styles")
def upload_styles():
    """Upload style reference photos from local disk to Modal volume.

    Uploads two sets:
    1. Legacy generic refs (style_ref_1/2/3) — used by use_default_style fallback
    2. CTAI Catherine-approved refs (ctai_*) — loaded from assets/trurender/style-refs/
       via the style-refs.json manifest

    Usage: modal run trurender_comfyui.py::upload_styles
    """
    import json

    # Workspace root (resolved from this script's location)
    workspace = Path(__file__).resolve().parent.parent  # up from pipeline/

    items = []

    # --- Legacy generic style refs ---
    style_files = {
        "style_ref_1_library.jpg":
            "trurender_endpoint/outputs/best_photo_1_library_9.jpg",
        "style_ref_2_staircase.jpg":
            "trurender_endpoint/outputs/best_photo_3_staircase_dining_85.jpg",
        "style_ref_3_kitchen.jpg":
            "trurender_endpoint/outputs/best_photo_5_kitchen_8.jpg",
    }
    for dest_name, rel_path in style_files.items():
        src = workspace / rel_path
        if not src.exists():
            print(f"WARNING (legacy): {src} not found, skipping")
            continue
        data = src.read_bytes()
        items.append({"name": dest_name, "data_b64": base64.b64encode(data).decode()})
        print(f"[legacy] {dest_name} ({len(data) / 1024:.0f} KB)")

    # --- CTAI Catherine-approved style refs from manifest ---
    manifest_path = workspace / "assets" / "trurender" / "style-refs" / "style-refs.json"
    style_refs_dir = manifest_path.parent
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for ref in manifest.get("refs", []):
            src = style_refs_dir / ref["filename"]
            if not src.exists():
                print(f"WARNING (ctai): {src} not found, skipping")
                continue
            dest_name = f"ctai_{ref['filename']}"
            data = src.read_bytes()
            items.append({"name": dest_name, "data_b64": base64.b64encode(data).decode()})
            print(f"[ctai] {dest_name} ({len(data) / 1024:.0f} KB) — {', '.join(ref['tags'][:3])}")
    else:
        print(f"WARNING: CTAI manifest not found at {manifest_path}, skipping CTAI refs")

    if not items:
        print("No style reference files found. Aborting.")
        return

    print(f"\nUploading {len(items)} style references to Modal volume...")
    upload_style_references.remote(style_data=items)
    print("Done!")


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(image_name: str, seed: int = 42,
                   steps: int = 40, cfg: float = 3.5,
                   denoise: float = 0.83,
                   controlnet_strength: float = 0.65,
                   canny_strength: float = 0.80,
                   hed_strength: float = 0.60,
                   canny_low: int = 50, canny_high: int = 150,
                   hed_safe: str = "enable",
                   hed_resolution: int = None,
                   use_canny: bool = True,
                   use_depth: bool = True,
                   use_hed: bool = False,
                   canny_first: bool = False,
                   target_width: int = 1024, target_height: int = 1024,
                   prompt_style: str = None,
                   positive_prompt: str = None,
                   negative_prompt: str = None,
                   use_ip_adapter: bool = False,
                   ip_adapter_strength: float = 0.4,
                   style_image_name: str = None,
                   flat_denoise_strength: float = 0.0,
                   mask_image_name: str = None) -> dict:
    """
    Build the ComfyUI workflow JSON with dynamic parameters.

    Dual ControlNet pipeline (v4):
      - Canny ControlNet: preserves edges and architectural lines
      - Depth ControlNet: preserves spatial layout and 3D structure
      - Canny has higher default strength (0.80 vs 0.65) per Quinn's directive:
        "if there's tension between the two the canny should win"

    Prompt control (in order of priority):
      1. positive_prompt/negative_prompt — raw text overrides
      2. prompt_style — key from PROMPT_LIBRARY ("preserve", "photo", "hybrid")
      3. DEFAULT_PROMPT_STYLE — module default

    ControlNet modes:
      - use_canny=True, use_depth=True: dual ControlNet (default)
      - use_canny=True, use_depth=False: canny-only
      - use_canny=False, use_depth=True: depth-only (v3 behavior)
      - use_canny=False, use_depth=False: no ControlNet (pure img2img)

    Spatially-varying denoise (SVD):
      - flat_denoise_strength > 0: uses KSamplerAdvanced + SetLatentNoiseMask
      - flat_denoise_strength == 0: standard KSampler (existing behavior)
      - mask_image_name: uploaded mask filename (generated by generate_flat_surface_mask)

    Images are resized to max_dimension on longest side before processing.
    """
    # Resolve prompts
    if positive_prompt is None or negative_prompt is None:
        style = prompt_style or DEFAULT_PROMPT_STYLE
        prompts = PROMPT_LIBRARY.get(style, PROMPT_LIBRARY[DEFAULT_PROMPT_STYLE])
        if positive_prompt is None:
            positive_prompt = prompts["positive"]
        if negative_prompt is None:
            negative_prompt = prompts["negative"]

    workflow = {
        "1": {
            "inputs": {"image": image_name, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "50": {
            "inputs": {
                "upscale_method": "lanczos",
                "width": target_width,
                "height": target_height,
                "crop": "disabled",
                "image": ["1", 0],
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Resize Image"},
        },
        "2": {
            "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"},
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        "3": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "4": {
            "inputs": {
                "clip_name1": "t5xxl_fp16.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
                "device": "default",
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "DualCLIPLoader"},
        },
        "5": {
            "inputs": {
                "text": positive_prompt,
                "clip": ["4", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive)"},
        },
        "6": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"},
        },
        "10": {
            "inputs": {
                "pixels": ["50", 0],
                "vae": ["3", 0],
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"},
        },
        "13": {
            "inputs": {
                "samples": ["11", 0],
                "vae": ["3", 0],
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "14": {
            "inputs": {
                "filename_prefix": "trurender_output",
                "images": ["13", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
    }

    # Build the ControlNet chain: conditioning flows through each CN in sequence.
    # Start from raw CLIP conditioning (nodes 5/6), apply ControlNets in order.
    # The LAST CN in the chain has highest priority (final conditioning authority).
    # The KSampler reads from whichever node is last in the chain.
    # Default chain order: Depth → HED → Canny (hard edges get final say).
    # canny_first=True swaps Canny before Depth (but tests show order is irrelevant for Flux).

    last_positive_source = ["5", 0]  # raw CLIP positive
    last_negative_source = ["6", 0]  # raw CLIP negative

    # Determine chain order: Depth (spatial) → HED (soft edges) → Canny (hard edges)
    if canny_first:
        cn_order = []
        if use_canny:
            cn_order.append("canny")
        if use_hed:
            cn_order.append("hed")
        if use_depth:
            cn_order.append("depth")
    else:
        cn_order = []
        if use_depth:
            cn_order.append("depth")
        if use_hed:
            cn_order.append("hed")
        if use_canny:
            cn_order.append("canny")

    for cn_type in cn_order:
        if cn_type == "depth":
            workflow["7"] = {
                "inputs": {"control_net_name": "flux-depth-controlnet-v3.safetensors"},
                "class_type": "ControlNetLoader",
                "_meta": {"title": "Load Depth ControlNet"},
            }
            workflow["8"] = {
                "inputs": {"image": ["50", 0]},
                "class_type": "DepthAnythingPreprocessor",
                "_meta": {"title": "Depth Anything"},
            }
            workflow["15"] = {
                "inputs": {
                    "filename_prefix": "trurender_depth",
                    "images": ["8", 0],
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Depth Map"},
            }
            workflow["41"] = {
                "inputs": {
                    "strength": controlnet_strength,
                    "start_percent": 0,
                    "end_percent": 1,
                    "positive": last_positive_source,
                    "negative": last_negative_source,
                    "control_net": ["7", 0],
                    "image": ["8", 0],
                    "vae": ["3", 0],
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {"title": "Apply Depth ControlNet"},
            }
            last_positive_source = ["41", 0]
            last_negative_source = ["41", 1]

        elif cn_type == "canny":
            workflow["20"] = {
                "inputs": {"control_net_name": "flux-canny-controlnet-v3.safetensors"},
                "class_type": "ControlNetLoader",
                "_meta": {"title": "Load Canny ControlNet"},
            }
            workflow["21"] = {
                "inputs": {
                    "low_threshold": canny_low,
                    "high_threshold": canny_high,
                    "resolution": max(target_width, target_height),  # match target resolution
                    "image": ["50", 0],
                },
                "class_type": "CannyEdgePreprocessor",
                "_meta": {"title": "Canny Edge Detection"},
            }
            workflow["25"] = {
                "inputs": {
                    "filename_prefix": "trurender_canny",
                    "images": ["21", 0],
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Canny Edge Map"},
            }
            workflow["22"] = {
                "inputs": {
                    "strength": canny_strength,
                    "start_percent": 0,
                    "end_percent": 1,
                    "positive": last_positive_source,
                    "negative": last_negative_source,
                    "control_net": ["20", 0],
                    "image": ["21", 0],
                    "vae": ["3", 0],
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {"title": "Apply Canny ControlNet"},
            }
            last_positive_source = ["22", 0]
            last_negative_source = ["22", 1]

        elif cn_type == "hed":
            workflow["30"] = {
                "inputs": {"control_net_name": "flux-hed-controlnet-v3.safetensors"},
                "class_type": "ControlNetLoader",
                "_meta": {"title": "Load HED ControlNet"},
            }
            workflow["31"] = {
                "inputs": {
                    "safe": hed_safe,
                    "resolution": hed_resolution if hed_resolution else max(target_width, target_height),
                    "image": ["50", 0],
                },
                "class_type": "HEDPreprocessor",
                "_meta": {"title": "HED Soft Edge Detection"},
            }
            workflow["35"] = {
                "inputs": {
                    "filename_prefix": "trurender_hed",
                    "images": ["31", 0],
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save HED Edge Map"},
            }
            workflow["32"] = {
                "inputs": {
                    "strength": hed_strength,
                    "start_percent": 0,
                    "end_percent": 1,
                    "positive": last_positive_source,
                    "negative": last_negative_source,
                    "control_net": ["30", 0],
                    "image": ["31", 0],
                    "vae": ["3", 0],
                },
                "class_type": "ControlNetApplyAdvanced",
                "_meta": {"title": "Apply HED ControlNet"},
            }
            last_positive_source = ["32", 0]
            last_negative_source = ["32", 1]

    # --- IP-Adapter nodes (60-62) ---
    # When active, wraps the model from UNETLoader with style conditioning
    # from a reference image. KSampler uses the IP-Adapter-modified model.
    model_source = ["2", 0]  # default: raw UNETLoader

    if use_ip_adapter and style_image_name:
        # Node 60: Load the style reference image
        workflow["60"] = {
            "inputs": {"image": style_image_name, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Style Reference Image"},
        }

        # Node 61: Load InstantX IP-Adapter + SigLIP vision encoder
        # (Shakker-Labs/ComfyUI-IPAdapter-Flux, 128 tokens, 57 blocks)
        workflow["61"] = {
            "inputs": {
                "ipadapter": "ip-adapter.bin",
                "clip_vision": "google/siglip-so400m-patch14-384",
                "provider": "cuda",
            },
            "class_type": "IPAdapterFluxLoader",
            "_meta": {"title": "Load InstantX IP-Adapter"},
        }

        # Node 62: Apply IP-Adapter to the model with style conditioning
        workflow["62"] = {
            "inputs": {
                "model": ["2", 0],  # UNETLoader output
                "ipadapter_flux": ["61", 0],
                "image": ["60", 0],
                "weight": ip_adapter_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            "class_type": "ApplyIPAdapterFlux",
            "_meta": {"title": "Apply InstantX IP-Adapter"},
        }

        # KSampler uses IP-Adapter-modified model instead of raw UNETLoader
        model_source = ["62", 0]

    # --- KSampler / KSamplerAdvanced + SVD noise mask ---
    if flat_denoise_strength > 0 and mask_image_name:
        # Spatially-varying denoise: use KSamplerAdvanced + SetLatentNoiseMask
        # The noise mask tells the sampler where to inject MORE noise (white = more noise).
        # KSamplerAdvanced doesn't have a 'denoise' param — instead we control
        # start_at_step to achieve the equivalent denoise level.
        # denoise 0.84 with 40 steps = start at step 6 (40 - 40*0.84 = 6.4 → 6)
        start_step = max(0, int(round(steps * (1.0 - denoise))))

        # Node 70: Load the flat-surface mask image
        workflow["70"] = {
            "inputs": {"image": mask_image_name, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Flat Surface Mask"},
        }

        # Node 71: Convert image to mask (red channel)
        workflow["71"] = {
            "inputs": {
                "channel": "red",
                "image": ["70", 0],
            },
            "class_type": "ImageToMask",
            "_meta": {"title": "Image to Mask (Flat Regions)"},
        }

        # Node 72: Apply noise mask to the latent
        # SetLatentNoiseMask auto-resizes the mask to latent dimensions (image_size/8)
        workflow["72"] = {
            "inputs": {
                "samples": ["10", 0],  # VAEEncode output
                "mask": ["71", 0],
            },
            "class_type": "SetLatentNoiseMask",
            "_meta": {"title": "Set Noise Mask (Flat Regions)"},
        }

        # Node 73: Save mask for debugging
        workflow["73"] = {
            "inputs": {
                "filename_prefix": "trurender_svd_mask",
                "images": ["70", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save SVD Mask (Debug)"},
        }

        # Node 11: KSamplerAdvanced with noise-masked latent
        workflow["11"] = {
            "inputs": {
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "start_at_step": start_step,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable",
                "model": model_source,
                "positive": last_positive_source,
                "negative": last_negative_source,
                "latent_image": ["72", 0],  # noise-masked latent
            },
            "class_type": "KSamplerAdvanced",
            "_meta": {"title": "KSamplerAdvanced (SVD)"},
        }
        print(f"[TruRender] SVD: KSamplerAdvanced start_step={start_step}/{steps} (denoise≈{denoise}), mask={mask_image_name}")
    else:
        # Standard KSampler — no mask, global denoise
        workflow["11"] = {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": model_source,
                "positive": last_positive_source,
                "negative": last_negative_source,
                "latent_image": ["10", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        }

    return workflow


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------

def postprocess_image(render_bytes: bytes, depth_bytes: bytes = None,
                      dof_strength: float = 0.6,
                      vignette_strength: float = 0.3,
                      chromatic_aberration: float = 0.4,
                      grain_strength: float = 0.15,
                      warmth: float = 0.2) -> bytes:
    """
    Apply photographic post-processing effects to a render.

    All strengths are 0.0 (off) to 1.0 (maximum). Defaults tuned for
    architectural interior photography look.

    Effects:
      1. Depth-of-field blur (requires depth map)
      2. Lens vignetting
      3. Chromatic aberration
      4. Film grain
      5. Warm tone mapping

    Returns PNG bytes.
    """
    import numpy as np
    from PIL import Image as PILImage, ImageFilter

    img = PILImage.open(io.BytesIO(render_bytes)).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # --- 1. Depth-of-field blur ---
    if depth_bytes and dof_strength > 0:
        depth_img = PILImage.open(io.BytesIO(depth_bytes)).convert('L')
        depth_img = depth_img.resize((w, h), PILImage.LANCZOS)
        depth = np.array(depth_img, dtype=np.float32) / 255.0

        # Focus on the mid-ground (depth ~0.4-0.6 is sharpest)
        # Create blur mask: further from mid-depth = more blur
        focus_depth = 0.5
        blur_mask = np.abs(depth - focus_depth) * 2.0  # 0 at focus, 1 at extremes
        blur_mask = np.clip(blur_mask * dof_strength, 0, 1)

        # Apply graduated Gaussian blur
        max_radius = int(8 * dof_strength)
        if max_radius >= 1:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=max_radius))
            blurred_arr = np.array(blurred, dtype=np.float32)
            mask_3d = blur_mask[:, :, np.newaxis]
            arr = arr * (1 - mask_3d) + blurred_arr * mask_3d

    # --- 2. Lens vignetting ---
    if vignette_strength > 0:
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        # Normalized distance from center (1.0 at corners)
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        dist = dist / dist.max()
        # Smooth falloff — starts at ~60% from center
        vignette = 1.0 - (dist ** 2) * vignette_strength * 0.5
        vignette = np.clip(vignette, 0, 1)
        arr *= vignette[:, :, np.newaxis]

    # --- 3. Chromatic aberration ---
    if chromatic_aberration > 0:
        shift = int(1 + chromatic_aberration * 2)  # 1-3 pixels
        # Shift red channel outward, blue channel inward
        r, g, b = arr[:, :, 0].copy(), arr[:, :, 1].copy(), arr[:, :, 2].copy()
        # Slight radial shift — approximate with horizontal/vertical shift
        arr[:, shift:, 0] = r[:, :-shift]   # red shifts right
        arr[:, :-shift, 2] = b[:, shift:]   # blue shifts left

    # --- 4. Film grain ---
    if grain_strength > 0:
        grain = np.random.normal(0, grain_strength * 12, arr.shape).astype(np.float32)
        # Grain is more visible in mid-tones
        luminance = np.mean(arr, axis=2, keepdims=True) / 255.0
        grain_mask = 4.0 * luminance * (1.0 - luminance)  # peaks at mid-gray
        arr += grain * grain_mask

    # --- 5. Warm tone mapping ---
    if warmth > 0:
        # Subtle warm shift — boost reds/yellows, cool blues slightly
        arr[:, :, 0] += warmth * 4    # red +
        arr[:, :, 1] += warmth * 2    # green slight +
        arr[:, :, 2] -= warmth * 3    # blue -

    # Clamp and convert back
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    result = PILImage.fromarray(arr)

    buf = io.BytesIO()
    result.save(buf, format='PNG')
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Contact sheet builder
# ---------------------------------------------------------------------------

def make_contact_sheet(images: list, seeds: list, cols: int = 4,
                       thumb_width: int = 512, label_height: int = 30):
    """Create a labeled grid of images."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    if not images:
        return None

    # Calculate thumbnail dimensions preserving aspect ratio
    sample = images[0]
    aspect = sample.height / sample.width
    thumb_h = int(thumb_width * aspect)

    rows = (len(images) + cols - 1) // cols
    sheet_w = cols * thumb_width
    sheet_h = rows * (thumb_h + label_height)

    sheet = PILImage.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (img, seed) in enumerate(zip(images, seeds)):
        row, col = divmod(i, cols)
        x = col * thumb_width
        y = row * (thumb_h + label_height)

        thumb = img.resize((thumb_width, thumb_h), PILImage.LANCZOS)
        sheet.paste(thumb, (x, y))

        label = f"seed: {seed}"
        draw.text((x + 5, y + thumb_h + 5), label, fill=(255, 255, 255), font=font)

    return sheet


# ---------------------------------------------------------------------------
# TruRender ComfyUI service
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A100-80GB",
    image=comfyui_image,
    volumes={VOLUME_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    scaledown_window=600,  # 10 min idle timeout
    timeout=900,  # 15 min per request max (batch can be long)
)
@modal.concurrent(max_inputs=10)  # route multiple HTTP requests to same container
class TruRenderComfyUI:

    @modal.enter()
    def start(self):
        """Setup model symlinks, start ComfyUI server, wait for ready."""
        import subprocess

        print("[TruRender] Setting up models...")
        self._setup_model_links()

        print("[TruRender] Starting ComfyUI server...")
        self.comfyui_process = subprocess.Popen(
            [
                "python", "main.py",
                "--listen", "127.0.0.1",
                "--port", str(COMFYUI_PORT),
                "--disable-auto-launch",
                "--verbose",  # enable verbose logging to capture custom node errors
            ],
            cwd=COMFYUI_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Start background thread to continuously print ComfyUI output
        import threading
        def _stream_comfyui_logs():
            for line in self.comfyui_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[ComfyUI] {line}")
        log_thread = threading.Thread(target=_stream_comfyui_logs, daemon=True)
        log_thread.start()
        self._log_thread = log_thread

        self._wait_for_comfyui(timeout=300)
        print("[TruRender] ComfyUI server ready!")

        # Check available node types for debugging
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/object_info")
            obj_info = json.loads(req.read())
            depth_nodes = [k for k in obj_info.keys() if 'depth' in k.lower() or 'Depth' in k]
            canny_nodes = [k for k in obj_info.keys() if 'canny' in k.lower() or 'Canny' in k]
            cn_nodes = [k for k in obj_info.keys() if 'controlnet' in k.lower() or 'ControlNet' in k]
            ipa_nodes = [k for k in obj_info.keys() if 'ipadapter' in k.lower() or 'IPAdapter' in k]
            print(f"[TruRender] Depth nodes: {depth_nodes}")
            print(f"[TruRender] Canny nodes: {canny_nodes}")
            print(f"[TruRender] ControlNet nodes: {cn_nodes}")
            print(f"[TruRender] IP-Adapter nodes: {ipa_nodes}")
        except Exception as e:
            print(f"[TruRender] Could not query node info: {e}")

        # Explicit import diagnostic for IPAdapter-Flux node
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c",
                 "import sys, importlib.util; "
                 "sys.path.insert(0, '/comfyui'); "
                 "import folder_paths; "
                 "spec = importlib.util.spec_from_file_location('ipadapter_flux', '/comfyui/custom_nodes/ComfyUI-IPAdapter-Flux/ipadapter_flux.py'); "
                 "m = importlib.util.module_from_spec(spec); "
                 "spec.loader.exec_module(m); "
                 "print('IPA import OK:', list(m.NODE_CLASS_MAPPINGS.keys()))"],
                capture_output=True, text=True, timeout=30
            )
            print(f"[TruRender] IPA diagnostic stdout: {result.stdout.strip()}")
            if result.stderr:
                print(f"[TruRender] IPA diagnostic stderr (last 2000 chars): {result.stderr.strip()[-2000:]}")
        except Exception as diag_e:
            print(f"[TruRender] IPA diagnostic failed: {diag_e}")

        # Warm up models by running a tiny dummy workflow
        print("[TruRender] Warming up models (first run loads into VRAM)...")
        self._warmup()

    def _warmup(self):
        """Run a tiny dummy workflow to force model loading into VRAM."""
        from PIL import Image as PILImage

        try:
            # Create a small 64x64 test image
            img = PILImage.new("RGB", (64, 64), (128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            warmup_start = time.time()
            output, elapsed = self._render_single(
                img_bytes, seed=1, steps=4,  # minimal steps for warmup
                denoise=0.83,
                use_canny=True,   # load canny CN into VRAM
                use_depth=True,   # load depth CN into VRAM
                use_hed=True,     # load HED CN into VRAM
                max_dim=64,  # tiny for fast warmup
            )
            total = time.time() - warmup_start
            print(f"[TruRender] Warmup complete in {total:.1f}s (render: {elapsed:.1f}s). All ControlNets loaded into VRAM.")
        except Exception as e:
            print(f"[TruRender] Warmup failed: {e}")
            print(f"[TruRender] First real render will be slow (model loading).")

        # Warm up IP-Adapter if models are available (InstantX + SigLIP)
        ip_adapter_path = f"{VOLUME_PATH}/comfyui_models/ipadapter-flux/ip-adapter.bin"
        clip_vision_path = f"{VOLUME_PATH}/comfyui_models/clip_vision/siglip-so400m-patch14-384/siglip-model.safetensors"
        if os.path.exists(ip_adapter_path) and os.path.exists(clip_vision_path):
            try:
                print("[TruRender] Warming up IP-Adapter...")
                # Use the same tiny image as both input and style reference
                img = PILImage.new("RGB", (64, 64), (200, 150, 100))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                style_bytes = buf.getvalue()

                ipa_start = time.time()
                output, elapsed = self._render_single(
                    img_bytes, seed=1, steps=4,
                    denoise=0.83,
                    use_canny=False, use_depth=False, use_hed=False,
                    max_dim=64,
                    use_ip_adapter=True,
                    ip_adapter_strength=0.4,
                    style_image_bytes=style_bytes,
                )
                ipa_total = time.time() - ipa_start
                print(f"[TruRender] IP-Adapter warmup complete in {ipa_total:.1f}s")
            except Exception as e:
                print(f"[TruRender] IP-Adapter warmup failed (non-fatal): {e}")
                print(f"[TruRender] IP-Adapter will load on first use.")
        else:
            print("[TruRender] IP-Adapter models not found, skipping warmup.")
            print("[TruRender]   Run: modal run trurender_comfyui.py::download_models")

    @modal.exit()
    def stop(self):
        """Terminate ComfyUI server."""
        if hasattr(self, "comfyui_process") and self.comfyui_process:
            self.comfyui_process.terminate()
            self.comfyui_process.wait(timeout=10)
            print("[TruRender] ComfyUI server stopped.")

    def _setup_model_links(self):
        """Symlink models from volume to ComfyUI directories."""
        links = {
            f"{VOLUME_PATH}/comfyui_models/unet/flux1-dev.safetensors":
                f"{COMFYUI_DIR}/models/unet/flux1-dev.safetensors",
            f"{VOLUME_PATH}/comfyui_models/vae/ae.safetensors":
                f"{COMFYUI_DIR}/models/vae/ae.safetensors",
            f"{VOLUME_PATH}/comfyui_models/clip/t5xxl_fp16.safetensors":
                f"{COMFYUI_DIR}/models/clip/t5xxl_fp16.safetensors",
            f"{VOLUME_PATH}/comfyui_models/clip/clip_l.safetensors":
                f"{COMFYUI_DIR}/models/clip/clip_l.safetensors",
            f"{VOLUME_PATH}/comfyui_models/controlnet/flux-depth-controlnet-v3.safetensors":
                f"{COMFYUI_DIR}/models/controlnet/flux-depth-controlnet-v3.safetensors",
            f"{VOLUME_PATH}/comfyui_models/controlnet/flux-canny-controlnet-v3.safetensors":
                f"{COMFYUI_DIR}/models/controlnet/flux-canny-controlnet-v3.safetensors",
            f"{VOLUME_PATH}/comfyui_models/controlnet/flux-hed-controlnet-v3.safetensors":
                f"{COMFYUI_DIR}/models/controlnet/flux-hed-controlnet-v3.safetensors",
            # InstantX IP-Adapter + SigLIP vision encoder
            f"{VOLUME_PATH}/comfyui_models/ipadapter-flux/ip-adapter.bin":
                f"{COMFYUI_DIR}/models/ipadapter-flux/ip-adapter.bin",
            # SigLIP files: stored with siglip- prefix on volume, symlinked to original names for loader
            f"{VOLUME_PATH}/comfyui_models/clip_vision/siglip-so400m-patch14-384/siglip-model.safetensors":
                f"{COMFYUI_DIR}/models/clip_vision/siglip-so400m-patch14-384/model.safetensors",
            f"{VOLUME_PATH}/comfyui_models/clip_vision/siglip-so400m-patch14-384/siglip-config.json":
                f"{COMFYUI_DIR}/models/clip_vision/siglip-so400m-patch14-384/config.json",
            f"{VOLUME_PATH}/comfyui_models/clip_vision/siglip-so400m-patch14-384/siglip-preprocessor_config.json":
                f"{COMFYUI_DIR}/models/clip_vision/siglip-so400m-patch14-384/preprocessor_config.json",
        }

        for src, dst in links.items():
            if not os.path.exists(src):
                print(f"[TruRender] WARNING: Model not found: {src}")
                print(f"[TruRender]   Run: modal run trurender_comfyui.py::download_models")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"[TruRender] ✓ Linked {os.path.basename(src)} ({size_mb:.0f} MB)")

        # Setup depth anything model for comfyui_controlnet_aux
        depth_src = f"{VOLUME_PATH}/annotator_ckpts/depth_anything_vitl14.pth"
        if os.path.exists(depth_src):
            # comfyui_controlnet_aux looks in its ckts/ directory
            # The DepthAnythingPreprocessor expects the model at a specific path
            ckts_dir = f"{COMFYUI_DIR}/custom_nodes/comfyui_controlnet_aux/ckts/LiheYoung/Depth-Anything/checkpoints"
            os.makedirs(ckts_dir, exist_ok=True)
            dst = os.path.join(ckts_dir, "depth_anything_vitl14.pth")
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(depth_src, dst)
            size_mb = os.path.getsize(depth_src) / (1024 * 1024)
            print(f"[TruRender] ✓ Linked depth_anything_vitl14.pth ({size_mb:.0f} MB)")
        else:
            print(f"[TruRender] WARNING: Depth model not found: {depth_src}")
            print(f"[TruRender]   Will auto-download on first use (slow)")

    def _wait_for_comfyui(self, timeout: int = 300):
        """Wait for ComfyUI server to be ready."""
        import urllib.request
        import urllib.error

        start = time.time()
        url = f"http://127.0.0.1:{COMFYUI_PORT}/system_stats"

        while time.time() - start < timeout:
            # Check if process died
            if self.comfyui_process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI process died with code {self.comfyui_process.returncode}. "
                    f"(see [ComfyUI] log lines above for details)"
                )

            try:
                req = urllib.request.urlopen(url, timeout=2)
                if req.status == 200:
                    return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass

            time.sleep(2)

        raise TimeoutError(f"ComfyUI did not start within {timeout}s")

    def _upload_image(self, image_bytes: bytes, filename: str) -> str:
        """Upload an image to ComfyUI's input directory."""
        import urllib.request

        # Detect content type from magic bytes and filename extension
        _ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        _mime = "image/jpeg" if _ext in ("jpg", "jpeg") else "image/png"

        # Build multipart form data
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {_mime}\r\n\r\n"
        ).encode() + image_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"true\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFYUI_PORT}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        return result.get("name", filename)

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        """Submit a workflow to ComfyUI and return the prompt_id."""
        import urllib.request

        payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{COMFYUI_PORT}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:2000]
            print(f"[TruRender] ComfyUI /prompt HTTP {e.code}: {err_body}")
            raise
        result = json.loads(resp.read())
        print(f"[TruRender] Queue response: {json.dumps(result)}")

        if "error" in result:
            raise RuntimeError(f"ComfyUI prompt error: {json.dumps(result['error'])}")
        if "node_errors" in result and result["node_errors"]:
            raise RuntimeError(f"ComfyUI node errors: {json.dumps(result['node_errors'], indent=2)}")

        return result["prompt_id"]

    def _poll_result(self, prompt_id: str, timeout: int = 840) -> dict:
        """Poll ComfyUI for workflow completion. Returns the history entry."""
        import urllib.request

        start = time.time()
        last_queue_log = 0
        while time.time() - start < timeout:
            try:
                # Check history first
                req = urllib.request.urlopen(
                    f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}"
                )
                history = json.loads(req.read())

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    print(f"[TruRender] Prompt {prompt_id[:8]} status: {json.dumps(status)}")
                    if status.get("completed", False) or status.get("status_str") == "success":
                        return entry
                    # Check for error
                    if status.get("status_str") == "error":
                        raise RuntimeError(
                            f"ComfyUI workflow failed: {json.dumps(status, indent=2)}"
                        )

                # Periodically check queue state for debugging
                elapsed = time.time() - start
                if elapsed - last_queue_log > 10:
                    try:
                        qreq = urllib.request.urlopen(
                            f"http://127.0.0.1:{COMFYUI_PORT}/queue"
                        )
                        queue = json.loads(qreq.read())
                        running = queue.get("queue_running", [])
                        pending = queue.get("queue_pending", [])
                        print(f"[TruRender] Queue: {len(running)} running, {len(pending)} pending ({elapsed:.0f}s elapsed)")
                    except Exception:
                        pass
                    last_queue_log = elapsed

            except (urllib.error.URLError, ConnectionRefusedError):
                pass

            time.sleep(2)

        # Final debug dump before timeout
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/queue")
            queue = json.loads(req.read())
            print(f"[TruRender] TIMEOUT - Final queue state: {json.dumps(queue)}")
            req2 = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}")
            hist = json.loads(req2.read())
            print(f"[TruRender] TIMEOUT - Final history: {json.dumps(hist)}")
        except Exception as e:
            print(f"[TruRender] TIMEOUT - Could not get debug info: {e}")

        raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout}s")

    def _get_output_image(self, history_entry: dict, node_id: str = "14") -> bytes:
        """Extract an output image from a completed workflow history entry."""
        import urllib.request
        import urllib.parse

        outputs = history_entry.get("outputs", {})
        save_node = outputs.get(node_id, {})
        images = save_node.get("images", [])

        if not images:
            raise RuntimeError(f"No output images for node {node_id}. Outputs: {json.dumps(outputs, indent=2)}")

        img_info = images[0]
        params = urllib.parse.urlencode({
            "filename": img_info["filename"],
            "subfolder": img_info.get("subfolder", ""),
            "type": img_info.get("type", "output"),
        })
        req = urllib.request.urlopen(
            f"http://127.0.0.1:{COMFYUI_PORT}/view?{params}"
        )
        return req.read()

    def _get_depth_map(self, history_entry: dict) -> bytes:
        """Extract the depth map from node 15 (SaveImage on DepthAnything output)."""
        try:
            return self._get_output_image(history_entry, node_id="15")
        except RuntimeError:
            return None  # depth map is optional

    def _get_canny_map(self, history_entry: dict) -> bytes:
        """Extract the Canny edge map from node 25 (SaveImage on Canny output)."""
        try:
            return self._get_output_image(history_entry, node_id="25")
        except RuntimeError:
            return None  # canny map is optional

    def _get_hed_map(self, history_entry: dict) -> bytes:
        """Extract the HED soft edge map from node 35 (SaveImage on HED output)."""
        try:
            return self._get_output_image(history_entry, node_id="35")
        except RuntimeError:
            return None  # hed map is optional

    def _load_default_style_reference(self) -> bytes:
        """Load a random default style reference image from the Modal volume.

        Returns the image bytes, or None if no default styles are available.
        """
        import random
        style_dir = f"{VOLUME_PATH}/style_references"
        if not os.path.exists(style_dir):
            print(f"[TruRender] No style_references directory at {style_dir}")
            return None
        styles = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not styles:
            print(f"[TruRender] No style reference images in {style_dir}")
            return None
        chosen = random.choice(styles)
        path = os.path.join(style_dir, chosen)
        print(f"[TruRender] Using default style reference: {chosen}")
        with open(path, "rb") as f:
            return f.read()

    def _render_single(self, image_bytes: bytes, seed: int = 42,
                       steps: int = 40, cfg: float = 3.5,
                       denoise: float = 0.83,
                       controlnet_strength: float = 0.65,
                       canny_strength: float = 0.80,
                       hed_strength: float = 0.60,
                       canny_low: int = 50, canny_high: int = 150,
                       hed_safe: str = "enable",
                       hed_resolution: int = None,
                       use_canny: bool = True,
                       use_depth: bool = True,
                       use_hed: bool = False,
                       canny_first: bool = False,
                       max_dim: int = 1536,
                       return_depth: bool = False,
                       return_canny: bool = False,
                       return_hed: bool = False,
                       prompt_style: str = None,
                       use_ip_adapter: bool = False,
                       ip_adapter_strength: float = 0.4,
                       style_image_bytes: bytes = None,
                       flat_denoise_strength: float = 0.0) -> tuple:
        """
        Run a single render through ComfyUI.

        Returns (output_image_bytes, elapsed_seconds) by default.
        With return_depth=True: (output, depth_bytes, elapsed)
        With return_canny=True: (output, canny_bytes, elapsed)
        With both: (output, depth_bytes, canny_bytes, elapsed)

        Dual ControlNet (v4 default):
          - Depth CN at controlnet_strength (default 0.65)
          - Canny CN at canny_strength (default 0.80)
          - Canny applied last (higher priority in the chain)

        max_dim: maximum dimension on longest side (default 1536).
        Images are resized preserving aspect ratio, dimensions rounded to 16.
        """
        from PIL import Image as PILImage
        start = time.time()
        client_id = uuid.uuid4().hex

        # Compute target dimensions preserving aspect ratio
        img = PILImage.open(io.BytesIO(image_bytes))
        w, h = img.size
        scale = min(max_dim / max(w, h), 1.0)  # don't upscale
        target_w = int(w * scale) // 16 * 16
        target_h = int(h * scale) // 16 * 16
        cn_desc = []
        if use_depth: cn_desc.append(f"depth@{controlnet_strength}")
        if use_hed: cn_desc.append(f"hed@{hed_strength}")
        if use_canny: cn_desc.append(f"canny@{canny_strength}")
        cn_str = " + ".join(cn_desc) if cn_desc else "none"
        ipa_str = f" | IPA@{ip_adapter_strength}" if use_ip_adapter and style_image_bytes else ""
        svd_str = f" | SVD@{flat_denoise_strength}" if flat_denoise_strength > 0 else ""
        print(f"[TruRender] Input: {w}x{h} \u2192 Target: {target_w}x{target_h} | CN: {cn_str}{ipa_str}{svd_str} | denoise: {denoise}")

        # Upload input image
        filename = f"trurender_input_{client_id[:8]}.png"
        uploaded_name = self._upload_image(image_bytes, filename)

        # Upload style reference image for IP-Adapter if provided
        style_uploaded_name = None
        if use_ip_adapter and style_image_bytes:
            # Detect format from magic bytes to preserve correct extension
            _style_ext = "jpg" if style_image_bytes[:2] == b"\xff\xd8" else "png"
            style_filename = f"trurender_style_{client_id[:8]}.{_style_ext}"
            try:
                style_uploaded_name = self._upload_image(style_image_bytes, style_filename)
                print(f"[TruRender] Style reference uploaded: {style_uploaded_name}")
            except Exception as e:
                print(f"[TruRender] WARNING: Failed to upload style image: {e}")
                print(f"[TruRender] Proceeding without IP-Adapter.")
                use_ip_adapter = False

        # Generate and upload flat-surface mask for SVD if enabled
        mask_uploaded_name = None
        if flat_denoise_strength > 0:
            print(f"[TruRender] Generating flat-surface mask (strength={flat_denoise_strength}, target={target_w}x{target_h})...")
            mask_bytes = generate_flat_surface_mask_at_size(
                image_bytes,
                target_w=target_w,
                target_h=target_h,
                flat_denoise_strength=flat_denoise_strength,
            )
            mask_filename = f"trurender_mask_{client_id[:8]}.png"
            try:
                mask_uploaded_name = self._upload_image(mask_bytes, mask_filename)
                print(f"[TruRender] Flat-surface mask uploaded: {mask_uploaded_name} ({len(mask_bytes)} bytes)")
            except Exception as e:
                print(f"[TruRender] WARNING: Failed to upload mask: {e}")
                print(f"[TruRender] Falling back to standard KSampler (no SVD).")
                flat_denoise_strength = 0.0

        # Build and submit workflow
        workflow = build_workflow(
            image_name=uploaded_name,
            seed=seed,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            controlnet_strength=controlnet_strength,
            canny_strength=canny_strength,
            hed_strength=hed_strength,
            canny_low=canny_low,
            canny_high=canny_high,
            hed_safe=hed_safe,
            hed_resolution=hed_resolution,
            use_canny=use_canny,
            use_depth=use_depth,
            use_hed=use_hed,
            canny_first=canny_first,
            target_width=target_w,
            target_height=target_h,
            prompt_style=prompt_style,
            use_ip_adapter=use_ip_adapter and style_uploaded_name is not None,
            ip_adapter_strength=ip_adapter_strength,
            style_image_name=style_uploaded_name,
            flat_denoise_strength=flat_denoise_strength,
            mask_image_name=mask_uploaded_name,
        )

        prompt_id = self._queue_prompt(workflow, client_id)
        print(f"[TruRender] Queued prompt {prompt_id} (seed={seed})")

        # Wait for completion
        history = self._poll_result(prompt_id)

        # Get output image
        output_bytes = self._get_output_image(history)
        elapsed = time.time() - start
        print(f"[TruRender] Render complete: seed={seed}, {elapsed:.1f}s")

        # Collect optional outputs
        extras = []
        if return_depth:
            depth_bytes = self._get_depth_map(history)
            extras.append(depth_bytes)
        if return_canny:
            canny_bytes = self._get_canny_map(history)
            extras.append(canny_bytes)
        if return_hed:
            hed_bytes = self._get_hed_map(history)
            extras.append(hed_bytes)

        if extras:
            return (output_bytes, *extras, elapsed)
        return output_bytes, elapsed

    @modal.asgi_app()
    def web(self):
        """FastAPI web endpoint."""
        from fastapi import FastAPI, File, UploadFile, Form
        from fastapi.responses import Response, JSONResponse, StreamingResponse
        from PIL import Image as PILImage
        from typing import Optional
        import random

        web_app = FastAPI(title="TruRender v6.0", version="6.0.0")

        from fastapi import Request as FastAPIRequest
        from fastapi.responses import JSONResponse as FastAPIJSONResponse
        import traceback

        @web_app.exception_handler(Exception)
        async def global_exception_handler(request: FastAPIRequest, exc: Exception):
            tb = traceback.format_exc()
            print(f"[TruRender] Unhandled exception: {exc}\n{tb}")
            return FastAPIJSONResponse(
                status_code=500,
                content={"error": str(exc), "traceback": tb[-2000:]},
            )

        # In-memory job store for async renders
        jobs = {}  # job_id -> {status, result_bytes, elapsed, seed, error}
        jobs_lock = threading.Lock()

        def _run_render_job(job_id, image_bytes, seed, steps, cfg, denoise,
                            controlnet_strength, canny_strength=0.80,
                            canny_low=50, canny_high=150,
                            use_canny=True, use_depth=True,
                            use_hed=False, hed_strength=0.60,
                            hed_safe="enable", hed_resolution=None,
                            canny_first=False, max_dim=1536,
                            prompt_style=None, output_format="png",
                            use_ip_adapter=False, ip_adapter_strength=0.4,
                            style_image_bytes=None,
                            flat_denoise_strength=0.0):
            """Background thread for async render."""
            from PIL import Image as PILImage
            try:
                output_bytes, elapsed = self._render_single(
                    image_bytes, seed=seed, steps=steps, cfg=cfg,
                    denoise=denoise,
                    controlnet_strength=controlnet_strength,
                    canny_strength=canny_strength,
                    canny_low=canny_low, canny_high=canny_high,
                    use_canny=use_canny, use_depth=use_depth,
                    use_hed=use_hed, hed_strength=hed_strength,
                    hed_safe=hed_safe, hed_resolution=hed_resolution,
                    canny_first=canny_first, max_dim=max_dim,
                    prompt_style=prompt_style,
                    use_ip_adapter=use_ip_adapter,
                    ip_adapter_strength=ip_adapter_strength,
                    style_image_bytes=style_image_bytes,
                    flat_denoise_strength=flat_denoise_strength,
                )
                if output_format.lower() == "jpeg":
                    img = PILImage.open(io.BytesIO(output_bytes))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    output_bytes = buf.getvalue()
                    media_type = "image/jpeg"
                else:
                    media_type = "image/png"

                cn_info = []
                if use_depth: cn_info.append(f"depth@{controlnet_strength}")
                if use_hed: cn_info.append(f"hed@{hed_strength}")
                if use_canny: cn_info.append(f"canny@{canny_strength}")
                headers = {
                    "X-TruRender-Time": f"{elapsed:.1f}s",
                    "X-TruRender-Seed": str(seed),
                    "X-TruRender-Version": "6.0.0",
                    "X-TruRender-ControlNets": ",".join(cn_info) if cn_info else "none",
                }
                if use_ip_adapter:
                    headers["X-TruRender-IPAdapter"] = f"instantx@{ip_adapter_strength}"

                with jobs_lock:
                    jobs[job_id] = {
                        "status": "completed",
                        "result_bytes": output_bytes,
                        "elapsed": elapsed,
                        "seed": seed,
                        "media_type": media_type,
                        "headers": headers,
                        "error": None,
                    }
            except Exception as e:
                with jobs_lock:
                    jobs[job_id] = {
                        "status": "failed",
                        "result_bytes": None,
                        "elapsed": None,
                        "seed": seed,
                        "media_type": None,
                        "headers": None,
                        "error": str(e),
                    }

        @web_app.post("/render")
        async def render(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=40),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
            canny_strength: float = Form(default=0.80),
            canny_low: int = Form(default=50),
            canny_high: int = Form(default=150),
            use_canny: bool = Form(default=True),
            use_depth: bool = Form(default=True),
            use_hed: bool = Form(default=False),
            hed_strength: float = Form(default=0.60),
            hed_safe: str = Form(default="enable"),
            hed_resolution: Optional[int] = Form(default=None),
            canny_first: bool = Form(default=False),
            prompt_style: str = Form(default=None),
            max_dim: int = Form(default=1536),
            output_format: str = Form(default="png"),
            style_image: Optional[UploadFile] = File(default=None),
            ip_adapter_strength: float = Form(default=0.4),
            use_ip_adapter: bool = Form(default=False),
            use_default_style: bool = Form(default=False),
            flat_denoise_strength: float = Form(default=0.0),
        ):
            """
            Single render: upload an architectural image, get photorealistic version back.

            Triple ControlNet (v5): Depth + HED soft edges + Canny hard edges.
            - controlnet_strength: Depth CN strength (default 0.65)
            - hed_strength: HED soft edge CN strength (default 0.60)
            - hed_safe: "enable" or "disable" (safe mode clips weak edges)
            - hed_resolution: preprocessor resolution (default: match target dim)
            - canny_strength: Canny CN strength (default 0.80, higher = more edge fidelity)
            - canny_low/canny_high: Canny edge detection thresholds
            - use_canny/use_depth/use_hed: enable/disable individual ControlNets
            - canny_first: if True, apply Canny first then Depth (Depth wins conflicts)
            - prompt_style: "preserve", "photo", or "hybrid" (default from server config)
            - max_dim: maximum dimension on longest side (default 1536)

            IP-Adapter (style transfer):
            - style_image: optional style reference photo (auto-enables IP-Adapter)
            - ip_adapter_strength: strength of style conditioning (default 0.4, range 0-1)
            - use_ip_adapter: explicitly enable IP-Adapter (auto-set True if style_image provided)
            - use_default_style: use a random default style reference from the volume

            Spatially-varying denoise (SVD):
            - flat_denoise_strength: extra noise for flat/featureless regions (0.0=off, 0.3-0.5 suggested)
            """
            image_bytes = await image.read()

            # Handle style image for IP-Adapter
            style_image_bytes = None
            if style_image is not None:
                style_image_bytes = await style_image.read()
                use_ip_adapter = True
            elif use_default_style or use_ip_adapter:
                # Load a random default style reference from the volume
                style_image_bytes = self._load_default_style_reference()
                if style_image_bytes:
                    use_ip_adapter = True
                else:
                    print("[TruRender] No default style references found, disabling IP-Adapter")
                    use_ip_adapter = False

            output_bytes, elapsed = self._render_single(
                image_bytes, seed=seed, steps=steps, cfg=cfg,
                denoise=denoise,
                controlnet_strength=controlnet_strength,
                canny_strength=canny_strength,
                hed_strength=hed_strength,
                canny_low=canny_low,
                canny_high=canny_high,
                hed_safe=hed_safe,
                hed_resolution=hed_resolution,
                use_canny=use_canny,
                use_depth=use_depth,
                use_hed=use_hed,
                canny_first=canny_first,
                max_dim=max_dim,
                prompt_style=prompt_style,
                use_ip_adapter=use_ip_adapter,
                ip_adapter_strength=ip_adapter_strength,
                style_image_bytes=style_image_bytes,
                flat_denoise_strength=flat_denoise_strength,
            )

            if output_format.lower() == "jpeg":
                img = PILImage.open(io.BytesIO(output_bytes))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                output_bytes = buf.getvalue()
                media_type = "image/jpeg"
            else:
                media_type = "image/png"

            cn_info = []
            if use_depth: cn_info.append(f"depth@{controlnet_strength}")
            if use_hed: cn_info.append(f"hed@{hed_strength}")
            if use_canny: cn_info.append(f"canny@{canny_strength}")
            headers = {
                "X-TruRender-Time": f"{elapsed:.1f}s",
                "X-TruRender-Seed": str(seed),
                "X-TruRender-Version": "6.0.0",
                "X-TruRender-ControlNets": ",".join(cn_info) if cn_info else "none",
            }
            if use_ip_adapter:
                headers["X-TruRender-IPAdapter"] = f"instantx@{ip_adapter_strength}"
            if flat_denoise_strength > 0:
                headers["X-TruRender-SVD"] = f"flat@{flat_denoise_strength}"

            return Response(
                content=output_bytes,
                media_type=media_type,
                headers=headers,
            )

        @web_app.post("/render/stream")
        async def render_stream(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=40),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
            canny_strength: float = Form(default=0.80),
            canny_low: int = Form(default=100),
            canny_high: int = Form(default=200),
            hed_strength: float = Form(default=0.60),
            hed_safe: str = Form(default="enable"),
            hed_resolution: Optional[int] = Form(default=None),
            use_canny: bool = Form(default=True),
            use_depth: bool = Form(default=True),
            use_hed: bool = Form(default=False),
            canny_first: bool = Form(default=False),
            prompt_style: str = Form(default=None),
            max_dim: int = Form(default=1536),
            output_format: str = Form(default="png"),
            style_image: Optional[UploadFile] = File(default=None),
            ip_adapter_strength: float = Form(default=0.4),
            use_ip_adapter: bool = Form(default=False),
            use_default_style: bool = Form(default=False),
            flat_denoise_strength: float = Form(default=0.0),
        ):
            """
            Streaming render endpoint. Sends keepalive newlines every 10s
            while rendering, then the full image as the final chunk.
            This prevents Modal's HTTP gateway from timing out on long renders.

            Response format: text lines (status updates) ending with a blank line,
            then the raw image bytes. Content-Type is application/octet-stream.
            Parse the image after the double-newline delimiter.
            """
            import asyncio

            image_bytes = await image.read()
            if seed == 0:
                seed = random.randint(0, 2**32 - 1)

            # Handle style image
            style_image_bytes = None
            if style_image is not None:
                style_image_bytes = await style_image.read()
                use_ip_adapter = True
            elif use_default_style or use_ip_adapter:
                style_image_bytes = self._load_default_style_reference()
                if style_image_bytes:
                    use_ip_adapter = True
                else:
                    use_ip_adapter = False

            render_result = {"done": False, "output": None, "elapsed": 0, "error": None}

            def do_render():
                try:
                    output_bytes, elapsed = self._render_single(
                        image_bytes, seed=seed, steps=steps, cfg=cfg,
                        denoise=denoise,
                        controlnet_strength=controlnet_strength,
                        canny_strength=canny_strength,
                        hed_strength=hed_strength,
                        canny_low=canny_low,
                        canny_high=canny_high,
                        hed_safe=hed_safe,
                        hed_resolution=hed_resolution,
                        use_canny=use_canny,
                        use_depth=use_depth,
                        use_hed=use_hed,
                        canny_first=canny_first,
                        max_dim=max_dim,
                        prompt_style=prompt_style,
                        use_ip_adapter=use_ip_adapter,
                        ip_adapter_strength=ip_adapter_strength,
                        style_image_bytes=style_image_bytes,
                        flat_denoise_strength=flat_denoise_strength,
                    )
                    render_result["output"] = output_bytes
                    render_result["elapsed"] = elapsed
                except Exception as e:
                    render_result["error"] = str(e)
                finally:
                    render_result["done"] = True

            thread = threading.Thread(target=do_render, daemon=True)
            thread.start()

            async def generate():
                tick = 0
                while not render_result["done"]:
                    tick += 1
                    yield f"status: rendering ({tick * 10}s)\n".encode()
                    await asyncio.sleep(10)

                if render_result["error"]:
                    yield f"error: {render_result['error']}\n".encode()
                    return

                elapsed = render_result["elapsed"]
                out = render_result["output"]
                if output_format.lower() == "jpeg":
                    img = PILImage.open(io.BytesIO(out))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    out = buf.getvalue()

                svd_str = f" svd=flat@{flat_denoise_strength}" if flat_denoise_strength > 0 else ""
                ipa_str = f" ipa=instantx@{ip_adapter_strength}" if use_ip_adapter else ""
                yield f"done: elapsed={elapsed:.1f}s seed={seed}{ipa_str}{svd_str} size={len(out)}\n".encode()
                yield b"\n"  # delimiter
                yield out

            return StreamingResponse(generate(), media_type="application/octet-stream")

        @web_app.post("/render/submit")
        async def render_submit(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=40),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
            canny_strength: float = Form(default=0.80),
            canny_low: int = Form(default=50),
            canny_high: int = Form(default=150),
            use_canny: bool = Form(default=True),
            use_depth: bool = Form(default=True),
            use_hed: bool = Form(default=False),
            hed_strength: float = Form(default=0.60),
            hed_safe: str = Form(default="enable"),
            hed_resolution: Optional[int] = Form(default=None),
            canny_first: bool = Form(default=False),
            prompt_style: str = Form(default=None),
            max_dim: int = Form(default=1536),
            output_format: str = Form(default="png"),
            style_image: Optional[UploadFile] = File(default=None),
            ip_adapter_strength: float = Form(default=0.4),
            use_ip_adapter: bool = Form(default=False),
            use_default_style: bool = Form(default=False),
            flat_denoise_strength: float = Form(default=0.0),
        ):
            """
            Submit a render job asynchronously. Returns a job_id immediately.
            Poll /render/status/{job_id} for results.
            Download result from /render/result/{job_id}.

            Accepts all the same parameters as /render.
            """
            image_bytes = await image.read()
            job_id = f"tr-{uuid.uuid4().hex[:12]}"

            # Handle style image for IP-Adapter
            style_image_bytes = None
            if style_image is not None:
                style_image_bytes = await style_image.read()
                use_ip_adapter = True
            elif use_default_style or use_ip_adapter:
                style_image_bytes = self._load_default_style_reference()
                if style_image_bytes:
                    use_ip_adapter = True
                else:
                    use_ip_adapter = False

            with jobs_lock:
                jobs[job_id] = {"status": "running", "seed": seed}

            thread = threading.Thread(
                target=_run_render_job,
                args=(job_id, image_bytes, seed, steps, cfg, denoise, controlnet_strength),
                kwargs=dict(
                    canny_strength=canny_strength,
                    canny_low=canny_low, canny_high=canny_high,
                    use_canny=use_canny, use_depth=use_depth,
                    use_hed=use_hed, hed_strength=hed_strength,
                    hed_safe=hed_safe, hed_resolution=hed_resolution,
                    canny_first=canny_first, max_dim=max_dim,
                    prompt_style=prompt_style, output_format=output_format,
                    use_ip_adapter=use_ip_adapter,
                    ip_adapter_strength=ip_adapter_strength,
                    style_image_bytes=style_image_bytes,
                    flat_denoise_strength=flat_denoise_strength,
                ),
                daemon=True,
            )
            thread.start()

            return {"job_id": job_id, "status": "running", "seed": seed}

        @web_app.get("/render/status/{job_id}")
        async def render_status(job_id: str):
            """Check status of an async render job."""
            with jobs_lock:
                job = jobs.get(job_id)

            if not job:
                return JSONResponse({"error": "Job not found"}, status_code=404)

            if job["status"] == "running":
                return {"job_id": job_id, "status": "running"}

            if job["status"] == "failed":
                return {"job_id": job_id, "status": "failed", "error": job["error"]}

            # Completed — return metadata (not the image, that's at /render/result)
            return {
                "job_id": job_id,
                "status": "completed",
                "seed": job["seed"],
                "elapsed_seconds": round(job["elapsed"], 1),
            }

        @web_app.get("/render/result/{job_id}")
        async def render_result(job_id: str):
            """Download the result image from a completed render job."""
            with jobs_lock:
                job = jobs.get(job_id)

            if not job:
                return JSONResponse({"error": "Job not found"}, status_code=404)
            if job["status"] != "completed":
                return JSONResponse({"error": f"Job status: {job['status']}"}, status_code=400)

            return Response(
                content=job["result_bytes"],
                media_type=job.get("media_type", "image/png"),
                headers=job.get("headers") or {
                    "X-TruRender-Time": f"{job['elapsed']:.1f}s",
                    "X-TruRender-Seed": str(job["seed"]),
                },
            )

        @web_app.post("/render/batch")
        async def render_batch(
            image: UploadFile = File(...),
            num_seeds: int = Form(default=8),
            base_seed: Optional[int] = Form(default=None),
            steps: int = Form(default=40),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
            canny_strength: float = Form(default=0.80),
            hed_strength: float = Form(default=0.60),
            use_canny: bool = Form(default=True),
            use_depth: bool = Form(default=True),
            use_hed: bool = Form(default=False),
            canny_first: bool = Form(default=False),
            max_dim: int = Form(default=1536),
            style_image: Optional[UploadFile] = File(default=None),
            ip_adapter_strength: float = Form(default=0.4),
            use_ip_adapter: bool = Form(default=False),
            use_default_style: bool = Form(default=False),
        ):
            """
            Batch render: generate multiple variations with different seeds.
            Triple ControlNet (v5) applied to each render.
            - canny_first: if True, apply Canny first then Depth (Depth wins conflicts)
            - max_dim: maximum dimension on longest side (default 1536)
            - style_image: optional style reference photo for IP-Adapter (same for all seeds)
            - ip_adapter_strength: strength of style conditioning (default 0.4)
            - use_default_style: use a random default style reference

            Returns a contact sheet image + per-seed metadata as JSON.
            """
            image_bytes = await image.read()

            # Handle style image for IP-Adapter
            style_image_bytes = None
            if style_image is not None:
                style_image_bytes = await style_image.read()
                use_ip_adapter = True
            elif use_default_style or use_ip_adapter:
                style_image_bytes = self._load_default_style_reference()
                if style_image_bytes:
                    use_ip_adapter = True
                else:
                    use_ip_adapter = False

            # Generate seeds
            if base_seed is None:
                base_seed = random.randint(0, 2**32 - 1)
            seeds = [base_seed + i * 7919 for i in range(num_seeds)]  # prime step for variety

            # Render all seeds sequentially (ComfyUI is single-GPU)
            results = []
            output_images = []
            batch_start = time.time()

            for seed in seeds:
                try:
                    out_bytes, elapsed = self._render_single(
                        image_bytes, seed=seed, steps=steps, cfg=cfg,
                        denoise=denoise,
                        controlnet_strength=controlnet_strength,
                        canny_strength=canny_strength,
                        use_canny=use_canny,
                        use_depth=use_depth,
                        use_hed=use_hed,
                        hed_strength=hed_strength,
                        canny_first=canny_first,
                        max_dim=max_dim,
                        use_ip_adapter=use_ip_adapter,
                        ip_adapter_strength=ip_adapter_strength,
                        style_image_bytes=style_image_bytes,
                    )
                    img = PILImage.open(io.BytesIO(out_bytes))
                    output_images.append(img)

                    # Encode individual image
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    results.append({
                        "seed": seed,
                        "image_base64": img_b64,
                        "size": f"{img.size[0]}x{img.size[1]}",
                        "elapsed_seconds": round(elapsed, 1),
                        "status": "success",
                    })
                except Exception as e:
                    results.append({
                        "seed": seed,
                        "image_base64": None,
                        "size": None,
                        "elapsed_seconds": None,
                        "status": f"error: {str(e)}",
                    })

            batch_elapsed = time.time() - batch_start

            # Build contact sheet
            contact_sheet_b64 = None
            if output_images:
                sheet = make_contact_sheet(
                    output_images,
                    [r["seed"] for r in results if r["status"] == "success"],
                )
                if sheet:
                    buf = io.BytesIO()
                    sheet.save(buf, format="PNG")
                    contact_sheet_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return JSONResponse({
                "id": f"trurender-batch-{uuid.uuid4().hex[:12]}",
                "version": "6.2.0",
                "num_seeds": num_seeds,
                "base_seed": base_seed,
                "total_elapsed_seconds": round(batch_elapsed, 1),
                "contact_sheet_base64": contact_sheet_b64,
                "renders": results,
                "parameters": {
                    "steps": steps,
                    "cfg": cfg,
                    "denoise": denoise,
                    "controlnet_strength": controlnet_strength,
                    "sampler": "dpmpp_2m",
                    "scheduler": "karras",
                    "ip_adapter": {
                        "enabled": use_ip_adapter,
                        "strength": ip_adapter_strength if use_ip_adapter else None,
                    },
                },
            })

        @web_app.post("/render/photo")
        async def render_photo(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=40),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
            canny_strength: float = Form(default=0.80),
            hed_strength: float = Form(default=0.60),
            hed_safe: str = Form(default="enable"),
            hed_resolution: Optional[int] = Form(default=None),
            use_canny: bool = Form(default=True),
            use_depth: bool = Form(default=True),
            use_hed: bool = Form(default=False),
            canny_first: bool = Form(default=False),
            max_dim: int = Form(default=1536),
            dof_strength: float = Form(default=0.6),
            vignette_strength: float = Form(default=0.3),
            chromatic_aberration: float = Form(default=0.4),
            grain_strength: float = Form(default=0.15),
            warmth: float = Form(default=0.2),
            output_format: str = Form(default="png"),
        ):
            """
            Full pipeline: render + post-processing in one call.
            Returns a photographic-quality image with DoF, vignetting,
            chromatic aberration, film grain, and warm toning applied.
            Triple ControlNet (v5) is used for the render step.
            """
            image_bytes = await image.read()
            render_bytes, depth_bytes, elapsed = self._render_single(
                image_bytes, seed=seed, steps=steps, cfg=cfg,
                denoise=denoise,
                controlnet_strength=controlnet_strength,
                canny_strength=canny_strength,
                hed_strength=hed_strength,
                hed_safe=hed_safe,
                hed_resolution=hed_resolution,
                use_canny=use_canny,
                use_depth=use_depth,
                use_hed=use_hed,
                canny_first=canny_first,
                max_dim=max_dim,
                return_depth=True,
            )

            # Apply post-processing
            pp_start = time.time()
            output_bytes = postprocess_image(
                render_bytes,
                depth_bytes=depth_bytes,
                dof_strength=dof_strength,
                vignette_strength=vignette_strength,
                chromatic_aberration=chromatic_aberration,
                grain_strength=grain_strength,
                warmth=warmth,
            )
            pp_elapsed = time.time() - pp_start

            if output_format.lower() == "jpeg":
                img = PILImage.open(io.BytesIO(output_bytes))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                output_bytes = buf.getvalue()
                media_type = "image/jpeg"
            else:
                media_type = "image/png"

            return Response(
                content=output_bytes,
                media_type=media_type,
                headers={
                    "X-TruRender-Time": f"{elapsed:.1f}s",
                    "X-TruRender-PostProcess-Time": f"{pp_elapsed:.2f}s",
                    "X-TruRender-Seed": str(seed),
                    "X-TruRender-Version": "6.0.0",
                    "X-TruRender-Pipeline": "render+postprocess",
                },
            )

        @web_app.get("/health")
        def health():
            """Health check - verifies ComfyUI is running."""
            import urllib.request
            try:
                req = urllib.request.urlopen(
                    f"http://127.0.0.1:{COMFYUI_PORT}/system_stats", timeout=5
                )
                stats = json.loads(req.read())
                # Check IP-Adapter model availability
                ip_adapter_available = (
                    os.path.exists(f"{COMFYUI_DIR}/models/ipadapter-flux/ip-adapter.bin")
                    and os.path.exists(f"{COMFYUI_DIR}/models/clip_vision/siglip-so400m-patch14-384/model.safetensors")
                )
                # Check default style references
                style_dir = f"{VOLUME_PATH}/style_references"
                style_count = len([f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(style_dir) else 0
                return {
                    "status": "ok",
                    "service": "trurender-comfyui",
                    "version": "6.2.0",
                    "comfyui": "running",
                    "gpu": stats.get("devices", [{}])[0].get("name", "unknown"),
                    "vram_total": stats.get("devices", [{}])[0].get("vram_total", 0),
                    "vram_free": stats.get("devices", [{}])[0].get("vram_free", 0),
                    "ip_adapter": {
                        "available": ip_adapter_available,
                        "version": "instantx" if ip_adapter_available else None,
                        "default_styles": style_count,
                    },
                }
            except Exception as e:
                return JSONResponse(
                    {"status": "error", "service": "trurender-comfyui", "error": str(e)},
                    status_code=503,
                )

        @web_app.post("/render/depth")
        async def render_depth(
            image: UploadFile = File(...),
            max_dim: int = Form(default=1536),
        ):
            """
            Extract a depth map from the input image using DepthAnything.
            Runs the render pipeline at minimum denoise just to get the depth map.
            Returns the depth map as a grayscale PNG.
            """
            image_bytes = await image.read()
            output_bytes, depth_bytes, elapsed = self._render_single(
                image_bytes, seed=42, denoise=0.01, max_dim=max_dim,
                return_depth=True,
                use_canny=False,  # don't need canny for depth extraction
                prompt_style="preserve",
            )
            if not depth_bytes:
                return JSONResponse(
                    {"error": "Failed to extract depth map"},
                    status_code=500,
                )
            return Response(
                content=depth_bytes,
                media_type="image/png",
                headers={
                    "X-TruRender-Time": f"{elapsed:.1f}s",
                    "X-TruRender-Version": "6.0.0",
                },
            )

        @web_app.post("/render/canny")
        async def render_canny(
            image: UploadFile = File(...),
            canny_low: int = Form(default=50),
            canny_high: int = Form(default=150),
            max_dim: int = Form(default=1536),
        ):
            """
            Extract a Canny edge map from the input image.
            Returns the edge map as a grayscale PNG.
            """
            image_bytes = await image.read()
            output_bytes, canny_bytes, elapsed = self._render_single(
                image_bytes, seed=42, denoise=0.01, max_dim=max_dim,
                return_canny=True,
                use_depth=False,  # don't need depth for canny extraction
                canny_low=canny_low,
                canny_high=canny_high,
                prompt_style="preserve",
            )
            if not canny_bytes:
                return JSONResponse(
                    {"error": "Failed to extract Canny edge map"},
                    status_code=500,
                )
            return Response(
                content=canny_bytes,
                media_type="image/png",
                headers={
                    "X-TruRender-Time": f"{elapsed:.1f}s",
                    "X-TruRender-Version": "6.0.0",
                    "X-TruRender-Canny-Thresholds": f"{canny_low}/{canny_high}",
                },
            )

        @web_app.post("/render/hed")
        async def render_hed(
            image: UploadFile = File(...),
            max_dim: int = Form(default=1536),
            hed_safe: str = Form(default="enable"),
            hed_resolution: Optional[int] = Form(default=None),
        ):
            """
            Extract a HED soft edge map from the input image.
            Returns the edge map as a PNG.
            HED captures soft edges and gradients that Canny misses.
            - hed_safe: "enable" or "disable" (safe mode clips weak edges)
            - hed_resolution: preprocessor resolution override
            """
            image_bytes = await image.read()
            output_bytes, hed_bytes, elapsed = self._render_single(
                image_bytes, seed=42, denoise=0.01, max_dim=max_dim,
                return_hed=True,
                use_depth=False,
                use_canny=False,
                use_hed=True,
                hed_safe=hed_safe,
                hed_resolution=hed_resolution,
                prompt_style="preserve",
            )
            if not hed_bytes:
                return JSONResponse(
                    {"error": "Failed to extract HED edge map"},
                    status_code=500,
                )
            return Response(
                content=hed_bytes,
                media_type="image/png",
                headers={
                    "X-TruRender-Time": f"{elapsed:.1f}s",
                    "X-TruRender-Version": "6.0.0",
                },
            )

        @web_app.get("/debug/nodes")
        def debug_nodes():
            """Debug: list available ComfyUI node types (useful for diagnosing IP-Adapter issues)."""
            import urllib.request
            try:
                req = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/object_info", timeout=10)
                obj_info = json.loads(req.read())
                ipa_nodes = {k: v.get("input", {}) for k, v in obj_info.items() if "ipadapter" in k.lower() or "IPAdapter" in k}
                flux_nodes = [k for k in obj_info.keys() if "flux" in k.lower() or "Flux" in k]
                return {
                    "total_nodes": len(obj_info),
                    "ipa_nodes": ipa_nodes,
                    "flux_nodes": flux_nodes,
                    "clip_vision_models": obj_info.get("IPAdapterFluxLoader", {}).get("input", {}).get("required", {}).get("clip_vision", None),
                }
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        @web_app.get("/")
        def root():
            return {
                "service": "trurender-comfyui",
                "version": "6.2.0",
                "description": (
                    "Architectural render → photorealistic conversion using ComfyUI. "
                    "Flux.1-dev + Triple ControlNet (Depth + HED + Canny) with single-pass KSampler. "
                    "Depth preserves spatial structure, HED captures soft edges/gradients, "
                    "Canny locks hard architectural lines. "
                    "Optional InstantX IP-Adapter style transfer from reference photos (128 tokens, SigLIP encoder)."
                ),
                "endpoints": {
                    "POST /render": "Single render with configurable ControlNets + optional IP-Adapter style transfer",
                    "POST /render/photo": "Full pipeline - render + post-processing (DoF, vignette, grain, etc.)",
                    "POST /render/batch": "Batch render - multiple seeds, contact sheet + JSON (supports IP-Adapter)",
                    "POST /render/depth": "Extract depth map only",
                    "POST /render/canny": "Extract Canny edge map only",
                    "POST /render/hed": "Extract HED soft edge map only",
                    "GET /health": "Health check (includes IP-Adapter status)",
                },
                "workflow": {
                    "sampler": "dpmpp_2m",
                    "scheduler": "karras",
                    "cfg": 3.5,
                    "steps": 40,
                    "denoise": 0.83,
                    "controlnets": {
                        "depth": {"model": "flux-depth-controlnet-v3 (XLabs)", "strength": 0.65},
                        "hed": {"model": "flux-hed-controlnet-v3 (XLabs)", "strength": 0.60, "note": "soft edges, opt-in via use_hed=true"},
                        "canny": {"model": "flux-canny-controlnet-v3 (XLabs)", "strength": 0.80},
                    },
                    "ip_adapter": {
                        "model": "InstantX/FLUX.1-dev-IP-Adapter (Shakker-Labs, 128 tokens, 57 blocks)",
                        "clip_vision": "google/siglip-so400m-patch14-384",
                        "default_strength": 0.4,
                        "note": "Optional style transfer from reference photo. Off by default.",
                    },
                    "depth_extractor": "Depth Anything ViT-L",
                    "edge_detectors": "Canny (50/150) + HED (soft edges)",
                },
            }

        return web_app


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Quick smoke test - prints deployment info."""
    print("=" * 60)
    print("TruRender v6.0 - ComfyUI on Modal")
    print("=" * 60)
    print()
    print("Setup:")
    print("  1. Download models (run once):")
    print("     modal run trurender_comfyui.py::download_models")
    print()
    print("  2. Upload style references (optional, for IP-Adapter):")
    print("     modal run trurender_comfyui.py::upload_styles")
    print()
    print("  3. Deploy:")
    print("     modal deploy trurender_comfyui.py")
    print()
    print("Endpoints:")
    print("  POST /render       - single image render (+ optional IP-Adapter style transfer)")
    print("  POST /render/batch - multi-seed batch render (+ optional IP-Adapter)")
    print("  POST /render/photo - render + post-processing")
    print("  GET  /health       - health check (includes IP-Adapter status)")
    print()
    print("Workflow: Flux.1-dev + Triple ControlNet + InstantX IP-Adapter")
    print("  Sampler:      DPM++ 2M Karras")
    print("  CFG:          3.5")
    print("  Steps:        40")
    print("  Denoise:      0.83 (configurable)")
    print("  ControlNets:  Depth@0.65 + Canny@0.80 + HED@0.60")
    print("  IP-Adapter:   InstantX (Shakker-Labs, 128 tokens), off by default, strength 0.4")
    print("=" * 60)

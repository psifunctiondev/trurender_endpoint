"""
TruRender v3 - ComfyUI-on-Modal for architectural render → photorealistic conversion.

Uses the actual ComfyUI workflow with:
  - Flux.1-dev (UNET)
  - XLabs Depth ControlNet V3 (flux-depth-controlnet-v3)
  - Depth Anything ViT-L for depth extraction
  - Single-pass KSampler with configurable denoise
  - DPM++ 2M Karras sampler, cfg 3.5
  - ControlNet strength 0.65 (reduced from 1.0 for visible transformation)

Deploy:   cd agents/tekton/modal && /Users/doxa/Library/Python/3.9/bin/modal deploy trurender_comfyui.py
Models:   cd agents/tekton/modal && /Users/doxa/Library/Python/3.9/bin/modal run trurender_comfyui.py::download_models
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
    .run_commands(
        # Install ComfyUI
        "git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui",
        "cd /comfyui && pip install -r requirements.txt",
        # Install comfyui_controlnet_aux (provides DepthAnythingPreprocessor)
        "cd /comfyui/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /comfyui/custom_nodes/comfyui_controlnet_aux && pip install -r requirements.txt",
        # Create model directories
        "mkdir -p /comfyui/models/unet /comfyui/models/vae /comfyui/models/clip /comfyui/models/controlnet",
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
        "name": "depth_anything_vitl14.pth",
        "repo": "LiheYoung/Depth-Anything",
        "filename": "checkpoints/depth_anything_vitl14.pth",
        "dest": "annotator_ckpts",
        "dest_filename": "depth_anything_vitl14.pth",
        "repo_type": "space",
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


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(image_name: str, seed: int = 42,
                   steps: int = 20, cfg: float = 3.5,
                   denoise: float = 0.83,
                   controlnet_strength: float = 0.65,
                   target_width: int = 1024, target_height: int = 1024,
                   prompt_style: str = None,
                   positive_prompt: str = None,
                   negative_prompt: str = None) -> dict:
    """
    Build the ComfyUI workflow JSON with dynamic parameters.

    Prompt control (in order of priority):
      1. positive_prompt/negative_prompt — raw text overrides
      2. prompt_style — key from PROMPT_LIBRARY ("preserve", "photo", "hybrid")
      3. DEFAULT_PROMPT_STYLE — module default

    Single-pass KSampler pipeline:
      img2img at denoise 0.83 (sweet spot for photorealism without hallucination).
      DPM++ 2M Karras sampler at cfg 3.5.
      ControlNet strength 0.65 (reduced from 1.0 to allow visible transformation).

    Images are resized to max_dimension on longest side before processing
    (Flux at 4K is extremely slow). Output is at the resized resolution.
    """
    # Resolve prompts
    if positive_prompt is None or negative_prompt is None:
        style = prompt_style or DEFAULT_PROMPT_STYLE
        prompts = PROMPT_LIBRARY.get(style, PROMPT_LIBRARY[DEFAULT_PROMPT_STYLE])
        if positive_prompt is None:
            positive_prompt = prompts["positive"]
        if negative_prompt is None:
            negative_prompt = prompts["negative"]

    return {
        "1": {
            "inputs": {"image": image_name, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        # Resize to max_dimension on longest side (keeps aspect ratio)
        # Width/height are pre-computed by caller and must be divisible by 16
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
        "7": {
            "inputs": {"control_net_name": "flux-depth-controlnet-v3.safetensors"},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load ControlNet Model"},
        },
        "8": {
            "inputs": {
                "image": ["50", 0],
            },
            "class_type": "DepthAnythingPreprocessor",
            "_meta": {"title": "Depth Anything"},
        },
        "10": {
            "inputs": {
                "pixels": ["50", 0],
                "vae": ["3", 0],
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"},
        },
        "11": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": ["2", 0],
                "positive": ["41", 0],
                "negative": ["41", 1],
                "latent_image": ["10", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
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
        "15": {
            "inputs": {
                "filename_prefix": "trurender_depth",
                "images": ["8", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Depth Map"},
        },
        "41": {
            "inputs": {
                "strength": controlnet_strength,
                "start_percent": 0,
                "end_percent": 1,
                "positive": ["5", 0],
                "negative": ["6", 0],
                "control_net": ["7", 0],
                "image": ["8", 0],
                "vae": ["3", 0],
            },
            "class_type": "ControlNetApplyAdvanced",
            "_meta": {"title": "Apply ControlNet"},
        },
    }


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
            ],
            cwd=COMFYUI_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        self._wait_for_comfyui(timeout=300)
        print("[TruRender] ComfyUI server ready!")

        # Check available node types for debugging
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/object_info")
            obj_info = json.loads(req.read())
            depth_nodes = [k for k in obj_info.keys() if 'depth' in k.lower() or 'Depth' in k]
            print(f"[TruRender] Depth-related nodes available: {depth_nodes}")
            if 'DepthAnythingPreprocessor' in obj_info:
                node_info = obj_info['DepthAnythingPreprocessor']
                input_info = node_info.get('input', {}).get('required', {})
                print(f"[TruRender] DepthAnythingPreprocessor inputs: {json.dumps(input_info)}")
        except Exception as e:
            print(f"[TruRender] Could not query node info: {e}")

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
                max_dim=64,  # tiny for fast warmup
            )
            total = time.time() - warmup_start
            print(f"[TruRender] Warmup complete in {total:.1f}s (render: {elapsed:.1f}s). Models loaded into VRAM.")
        except Exception as e:
            print(f"[TruRender] Warmup failed: {e}")
            print(f"[TruRender] First real render will be slow (model loading).")

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
                # Read remaining output
                stdout = self.comfyui_process.stdout.read()
                raise RuntimeError(
                    f"ComfyUI process died with code {self.comfyui_process.returncode}. "
                    f"Output:\n{stdout}"
                )

            try:
                req = urllib.request.urlopen(url, timeout=2)
                if req.status == 200:
                    return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass

            # Print any output from ComfyUI for debugging
            if self.comfyui_process.stdout:
                import select
                import sys
                # Non-blocking read
                try:
                    line = self.comfyui_process.stdout.readline()
                    if line:
                        print(f"[ComfyUI] {line.rstrip()}")
                except Exception:
                    pass

            time.sleep(2)

        raise TimeoutError(f"ComfyUI did not start within {timeout}s")

    def _upload_image(self, image_bytes: bytes, filename: str) -> str:
        """Upload an image to ComfyUI's input directory."""
        import urllib.request

        # Build multipart form data
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
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
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(f"[TruRender] Queue response: {json.dumps(result)}")

        if "error" in result:
            raise RuntimeError(f"ComfyUI prompt error: {json.dumps(result['error'])}")
        if "node_errors" in result and result["node_errors"]:
            raise RuntimeError(f"ComfyUI node errors: {json.dumps(result['node_errors'], indent=2)}")

        return result["prompt_id"]

    def _poll_result(self, prompt_id: str, timeout: int = 600) -> dict:
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

    def _render_single(self, image_bytes: bytes, seed: int = 42,
                       steps: int = 20, cfg: float = 3.5,
                       denoise: float = 0.83,
                       controlnet_strength: float = 0.65,
                       max_dim: int = 1536,
                       return_depth: bool = False,
                       prompt_style: str = None) -> tuple:
        """
        Run a single render through ComfyUI.
        Returns (output_image_bytes, elapsed_seconds) or
        (output_image_bytes, depth_map_bytes, elapsed_seconds) if return_depth=True.

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
        print(f"[TruRender] Input: {w}x{h} → Target: {target_w}x{target_h} (max_dim={max_dim})")

        # Upload input image
        filename = f"trurender_input_{client_id[:8]}.png"
        uploaded_name = self._upload_image(image_bytes, filename)

        # Build and submit workflow (single-pass)
        workflow = build_workflow(
            image_name=uploaded_name,
            seed=seed,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            controlnet_strength=controlnet_strength,
            target_width=target_w,
            target_height=target_h,
            prompt_style=prompt_style,
        )

        prompt_id = self._queue_prompt(workflow, client_id)
        print(f"[TruRender] Queued prompt {prompt_id} (seed={seed})")

        # Wait for completion
        history = self._poll_result(prompt_id)

        # Get output image
        output_bytes = self._get_output_image(history)
        elapsed = time.time() - start
        print(f"[TruRender] Render complete: seed={seed}, {elapsed:.1f}s")

        if return_depth:
            depth_bytes = self._get_depth_map(history)
            return output_bytes, depth_bytes, elapsed
        return output_bytes, elapsed

    @modal.asgi_app()
    def web(self):
        """FastAPI web endpoint."""
        from fastapi import FastAPI, File, UploadFile, Form
        from fastapi.responses import Response, JSONResponse
        from PIL import Image as PILImage
        from typing import Optional
        import random

        web_app = FastAPI(title="TruRender v3", version="3.0.0")

        # In-memory job store for async renders
        jobs = {}  # job_id -> {status, result_bytes, elapsed, seed, error}
        jobs_lock = threading.Lock()

        def _run_render_job(job_id, image_bytes, seed, steps, cfg, denoise, controlnet_strength):
            """Background thread for async render."""
            try:
                output_bytes, elapsed = self._render_single(
                    image_bytes, seed=seed, steps=steps, cfg=cfg,
                    denoise=denoise,
                    controlnet_strength=controlnet_strength,
                )
                with jobs_lock:
                    jobs[job_id] = {
                        "status": "completed",
                        "result_bytes": output_bytes,
                        "elapsed": elapsed,
                        "seed": seed,
                        "error": None,
                    }
            except Exception as e:
                with jobs_lock:
                    jobs[job_id] = {
                        "status": "failed",
                        "result_bytes": None,
                        "elapsed": None,
                        "seed": seed,
                        "error": str(e),
                    }

        @web_app.post("/render")
        async def render(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=20),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
            prompt_style: str = Form(default=None),
            output_format: str = Form(default="png"),
        ):
            """
            Single render: upload an architectural image, get photorealistic version back.
            prompt_style: "preserve", "photo", or "hybrid" (default from server config).
            NOTE: This is synchronous. For long renders, use /render/submit instead.
            """
            image_bytes = await image.read()
            output_bytes, elapsed = self._render_single(
                image_bytes, seed=seed, steps=steps, cfg=cfg,
                denoise=denoise,
                controlnet_strength=controlnet_strength,
                prompt_style=prompt_style,
            )

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
                    "X-TruRender-Seed": str(seed),
                    "X-TruRender-Version": "3.0.0",
                },
            )

        @web_app.post("/render/submit")
        async def render_submit(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=20),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
        ):
            """
            Submit a render job asynchronously. Returns a job_id immediately.
            Poll /render/status/{job_id} for results.
            """
            image_bytes = await image.read()
            job_id = f"tr-{uuid.uuid4().hex[:12]}"

            with jobs_lock:
                jobs[job_id] = {"status": "running", "seed": seed}

            thread = threading.Thread(
                target=_run_render_job,
                args=(job_id, image_bytes, seed, steps, cfg, denoise, controlnet_strength),
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
                media_type="image/png",
                headers={
                    "X-TruRender-Time": f"{job['elapsed']:.1f}s",
                    "X-TruRender-Seed": str(job["seed"]),
                },
            )

        @web_app.post("/render/batch")
        async def render_batch(
            image: UploadFile = File(...),
            num_seeds: int = Form(default=8),
            base_seed: Optional[int] = Form(default=None),
            steps: int = Form(default=20),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
        ):
            """
            Batch render: generate multiple variations with different seeds.

            Returns a contact sheet image + per-seed metadata as JSON.
            Designed for hallucination detection pipeline integration.
            """
            image_bytes = await image.read()

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
                "version": "3.0.0",
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
                },
            })

        @web_app.post("/render/photo")
        async def render_photo(
            image: UploadFile = File(...),
            seed: int = Form(default=42),
            steps: int = Form(default=20),
            cfg: float = Form(default=3.5),
            denoise: float = Form(default=0.83),
            controlnet_strength: float = Form(default=0.65),
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
            """
            image_bytes = await image.read()
            render_bytes, depth_bytes, elapsed = self._render_single(
                image_bytes, seed=seed, steps=steps, cfg=cfg,
                denoise=denoise,
                controlnet_strength=controlnet_strength,
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
                    "X-TruRender-Version": "3.0.0",
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
                return {
                    "status": "ok",
                    "service": "trurender-comfyui",
                    "version": "3.0.0",
                    "comfyui": "running",
                    "gpu": stats.get("devices", [{}])[0].get("name", "unknown"),
                    "vram_total": stats.get("devices", [{}])[0].get("vram_total", 0),
                    "vram_free": stats.get("devices", [{}])[0].get("vram_free", 0),
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
                    "X-TruRender-Version": "3.0.0",
                },
            )

        @web_app.get("/")
        def root():
            return {
                "service": "trurender-comfyui",
                "version": "3.0.0",
                "description": (
                    "Architectural render → photorealistic conversion using ComfyUI. "
                    "Flux.1-dev + Depth ControlNet V3 with single-pass KSampler "
                    "(DPM++ 2M Karras, denoise 0.83, ControlNet 0.65)."
                ),
                "endpoints": {
                    "POST /render": "Single render - upload image, get photorealistic version",
                    "POST /render/photo": "Full pipeline - render + post-processing (DoF, vignette, grain, etc.)",
                    "POST /render/batch": "Batch render - multiple seeds, contact sheet + JSON",
                    "GET /health": "Health check",
                },
                "workflow": {
                    "sampler": "dpmpp_2m",
                    "scheduler": "karras",
                    "cfg": 3.5,
                    "steps": 20,
                    "denoise": 0.83,
                    "controlnet_strength": 0.65,
                    "controlnet": "flux-depth-controlnet-v3 (XLabs)",
                    "depth": "Depth Anything ViT-L",
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
    print("TruRender v3 - ComfyUI on Modal")
    print("=" * 60)
    print()
    print("Setup:")
    print("  1. Download models (run once):")
    print("     modal run trurender_comfyui.py::download_models")
    print()
    print("  2. Deploy:")
    print("     modal deploy trurender_comfyui.py")
    print()
    print("Endpoints:")
    print("  POST /render       - single image render")
    print("  POST /render/batch - multi-seed batch render")
    print("  GET  /health       - health check")
    print()
    print("Workflow: Flux.1-dev + Depth ControlNet V3")
    print("  Sampler:  DPM++ 2M Karras")
    print("  CFG:      3.5")
    print("  Steps:    20")
    print("  Denoise:  0.83 (configurable)")
    print("  ControlNet: 0.65 (reduced for visible transformation)")
    print("=" * 60)

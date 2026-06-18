"""
TruRender v7.0 — Qwen-Image-Edit-2511 pipeline (edit-first paradigm).

Replaces v6's Flux1-dev + depth/canny ControlNet stack with Qwen-Image-Edit-2511,
an instruction-based edit model. The Enscape render is treated as an EDIT target:
geometry, layout, furniture, hardware, and material identity are preserved by the
model itself; the instruction only authorises a change in *rendering quality*.

Why v7:
  - **License**: Qwen-Image-Edit-2511 is Apache 2.0 — clean commercial use, no fee.
    Flux1-dev (v6) is non-commercial; using it inside paid client work would
    require a paid BFL license.
  - **Fidelity**: an instruction-based edit model preserves the source far better
    than high-denoise img2img, which is the product's #1 requirement.

Architecture (matches Comfy-Org's official image_qwen_image_edit_2511.json):
  LoadImage → TextEncodeQwenImageEditPlus (carries pixels via Qwen2.5-VL)
            → CFGNorm → KSampler (denoise=1.0, euler/simple, steps=40, cfg=4.0)
            → VAEDecode → SaveImage

Aspect preservation: Python-computed dimensions (Pillow) → EmptySD3LatentImage.
Output dimensions match the input aspect ratio at a ~1MP bucket, snapped to
multiples of 16. This replaces v6's ImageScale-to-1024x1024 (which stretched
16:9 input ~22% horizontally) and Quinn's original FluxKontextImageScale
(a BFL custom node we explicitly avoid).

Models staged on the existing trurender-model-cache volume (shared with v6):
  - qwen_image_edit_2511_fp8mixed.safetensors  (~20GB, default — cost lever)
  - qwen_image_edit_2511_bf16.safetensors      (~41GB, set use_fp8=False)
  - qwen_2.5_vl_7b_fp8_scaled.safetensors      (~9GB,  text+vision encoder)
  - qwen_image_vae.safetensors                  (~242MB)
  - Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors (~810MB, optional)
  - qwen_image_depth_diffsynth_controlnet.safetensors (~2GB, optional backstop, NOT wired)

Deploy:    cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal deploy trurender_qwen_comfyui.py
Models:    cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_qwen_comfyui.py::download_models
Render:    cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png
"""

import modal
import json
import io
import time
import uuid
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Default prompts (hardcoded — per brief, not user-configurable in v7)
# Instruction-style, not descriptive. The hard constraint ("do not add/move/
# restyle anything; change only rendering quality") is what protects fidelity.
# ---------------------------------------------------------------------------

DEFAULT_POSITIVE = (
    "Turn this 3D architectural rendering into a photorealistic interior "
    "photograph. Preserve the exact room layout, camera angle, and perspective, "
    "and keep every element in the scene in identical positions, shapes, "
    "proportions, and colors: all walls, windows and mullions, cabinetry, "
    "countertops, the kitchen island, appliances, pendant lights, dining table, "
    "chairs, plants, and all decor. Do not add, remove, relocate, resize, or "
    "restyle any object, and do not change any material or finish. Change ONLY "
    "the rendering quality to that of a real photograph: physically accurate "
    "natural daylight from the windows with soft directional shadows and gentle "
    "falloff; true material microtexture (visible wood grain, natural stone "
    "veining in the marble, fabric weave on the upholstery, brushed and polished "
    "metal on the hardware and faucet, real glass with subtle reflections and "
    "slight impurity, plaster walls with faint surface variation); realistic "
    "global illumination and contact shadows; and authentic camera optics and "
    "color response. Hasselblad medium-format quality, photographed for "
    "Architectural Digest, indistinguishable from a real photo."
)

DEFAULT_NEGATIVE = (
    "3d render, CGI, computer-generated, video game graphics, cartoon, "
    "illustration, painting, plastic or waxy surfaces, perfectly uniform "
    "textures, oversaturated colors, extreme HDR, neon, fantasy lighting, "
    "changed layout, moved or added or removed furniture, distorted geometry, "
    "warped lines, text, watermark, signature"
)


# ---------------------------------------------------------------------------
# Model file names (must match what download_models stages on the volume)
# ---------------------------------------------------------------------------

DIFFUSION_BF16 = "qwen_image_edit_2511_bf16.safetensors"
DIFFUSION_FP8 = "qwen_image_edit_2511_fp8mixed.safetensors"   # ~20GB cost lever (default)
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
DEPTH_DIFFSYNTH_BACKSTOP = "qwen_image_depth_diffsynth_controlnet.safetensors"  # TODO: depth backstop


# ---------------------------------------------------------------------------
# Aspect-preservation helper
# ---------------------------------------------------------------------------

def compute_aspect_preserving_dims(image_bytes: bytes, target_megapixels: float = 1.0,
                                   max_side: int = 1536) -> tuple:
    """Read image, compute (width, height) preserving aspect ratio at ~target_megapixels.

    Returns dimensions snapped to multiples of 16 (ComfyUI latent alignment).
    Replaces v6's ImageScale-to-1024x1024 (which stretched 16:9 ~22% horizontally)
    AND Quinn's original FluxKontextImageScale (a BFL custom node we explicitly
    avoid in v7 — see task brief design decision).
    """
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(image_bytes))
    w, h = img.size
    aspect = w / h
    target_area = target_megapixels * 1024 * 1024
    target_w_f = (target_area * aspect) ** 0.5
    target_h_f = target_area / target_w_f
    # Cap longest side to max_side (safety — large inputs)
    if max(target_w_f, target_h_f) > max_side:
        scale = max_side / max(target_w_f, target_h_f)
        target_w_f *= scale
        target_h_f *= scale
    # Snap to multiples of 16
    target_w = max(16, (int(target_w_f) // 16) * 16)
    target_h = max(16, (int(target_h_f) // 16) * 16)
    return target_w, target_h


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(image_name: str = "enscape_input.png",
                   seed: int = 42,
                   target_width: int = 1328,
                   target_height: int = 768,
                   *,
                   positive: str = DEFAULT_POSITIVE,
                   negative: str = DEFAULT_NEGATIVE,
                   # Full-quality path (default — per-image best result):
                   steps: int = 40,
                   cfg: float = 4.0,
                   sampler_name: str = "euler",
                   scheduler: str = "simple",
                   # Model-sampling / stabilisation (match Comfy-Org official template):
                   model_sampling_shift: float = 3.1,
                   cfgnorm_strength: float = 1.0,
                   # Cost / speed levers:
                   use_fp8: bool = True,
                   use_lightning_lora: bool = False,
                   filename_prefix: str = "trurender_qwen") -> dict:
    """Return an API-format ComfyUI prompt graph (dict keyed by node id).

    With use_lightning_lora=True the sampler drops to 4 steps / cfg 1.0 and a
    LoraLoaderModelOnly node is inserted after CFGNorm. Use that for cheap
    preview passes; leave it off for final client deliverables.

    With use_fp8=True (default) the fp8mixed weights are loaded (~20GB, fits
    A100-80GB with room for the 9GB text encoder; negligible quality loss).
    Set use_fp8=False for the bf16 weights (~41GB).
    """
    unet = DIFFUSION_FP8 if use_fp8 else DIFFUSION_BF16

    if use_lightning_lora:
        # Lightning 4-step preset: cheap preview
        steps, cfg = 4, 1.0

    g: dict = {}

    # 1: Load the input Enscape render (original resolution, no rescale)
    g["1"] = {
        "inputs": {"image": image_name, "upload": "image"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Enscape Render"},
    }

    # 3: UNETLoader — Qwen-Image-Edit-2511 (fp8mixed or bf16)
    g["3"] = {
        "inputs": {"unet_name": unet, "weight_dtype": "default"},
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Qwen-Image-Edit 2511"},
    }

    # 4: CLIPLoader — Qwen2.5-VL text+vision encoder
    g["4"] = {
        "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image", "device": "default"},
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load Qwen2.5-VL Text/Vision Encoder"},
    }

    # 5: VAELoader — Qwen VAE
    g["5"] = {
        "inputs": {"vae_name": VAE},
        "class_type": "VAELoader",
        "_meta": {"title": "Load Qwen VAE"},
    }

    # 6: ModelSamplingAuraFlow (shift 3.1) — works for Qwen despite the name
    g["6"] = {
        "inputs": {"shift": model_sampling_shift, "model": ["3", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "Model Sampling (shift)"},
    }

    # 7: CFGNorm — between ModelSamplingAuraFlow and KSampler
    g["7"] = {
        "inputs": {"strength": cfgnorm_strength, "model": ["6", 0]},
        "class_type": "CFGNorm",
        "_meta": {"title": "CFG Norm"},
    }

    # Model feeding the sampler. Default = CFGNorm output (node 7).
    model_ref = ["7", 0]
    if use_lightning_lora:
        g["20"] = {
            "inputs": {
                "lora_name": LIGHTNING_LORA,
                "strength_model": 1.0,
                "model": ["7", 0],
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "Lightning 4-step LoRA (preview only)"},
        }
        model_ref = ["20", 0]

    # 8/9: TextEncodeQwenImageEditPlus — the edit-model's encode node.
    # image1 carries the source render's pixels (read by Qwen2.5-VL encoder
    # inside the node). TextEncodeQwenImageEditPlus is what makes this an EDIT
    # rather than a regeneration.
    g["8"] = {
        "inputs": {
            "prompt": positive,
            "clip": ["4", 0],
            "vae": ["5", 0],
            "image1": ["1", 0],
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "Qwen Edit Encode (Positive / instruction)"},
    }
    g["9"] = {
        "inputs": {
            "prompt": negative,
            "clip": ["4", 0],
            "vae": ["5", 0],
            "image1": ["1", 0],
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "Qwen Edit Encode (Negative)"},
    }

    # 10: EmptySD3LatentImage — sets output dimensions (matches input aspect ratio).
    # width/height are Python-computed (not FluxKontextImageScale — BFL custom node).
    # denoise=1.0 is correct for the edit paradigm: the source is carried through
    # TextEncodeQwenImageEditPlus's image1, the latent just sets output dimensions.
    g["10"] = {
        "inputs": {"width": target_width, "height": target_height, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
        "_meta": {"title": "EmptySD3LatentImage (output dims, aspect-preserved)"},
    }

    # 11: KSampler
    g["11"] = {
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "model": model_ref,
            "positive": ["8", 0],
            "negative": ["9", 0],
            "latent_image": ["10", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    }

    # 12: VAEDecode
    g["12"] = {
        "inputs": {"samples": ["11", 0], "vae": ["5", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    }

    # 13: SaveImage
    g["13"] = {
        "inputs": {"filename_prefix": filename_prefix, "images": ["12", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    }

    return g


# ---------------------------------------------------------------------------
# Modal image: ComfyUI (main, for TextEncodeQwenImageEditPlus) + Python deps
# v6 image stays pinned to bda1482 (IPAdapter-Flux compat) — separate Modal app.
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
        "diffusers>=0.30.0",
        "einops>=0.8.0",
        "transformers>=4.45.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.5",
    )
    .run_commands(
        # ComfyUI main — has TextEncodeQwenImageEditPlus, ModelSamplingAuraFlow,
        # CFGNorm, EmptySD3LatentImage. v7 doesn't need IPAdapter-Flux or
        # comfyui_controlnet_aux (no IP-Adapter, no ControlNet in the v7 graph).
        "git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui",
        "cd /comfyui && pip install -r requirements.txt",
        # Create model directories ComfyUI will look in
        "mkdir -p /comfyui/models/unet /comfyui/models/vae /comfyui/models/clip",
        "mkdir -p /comfyui/models/model_patches /comfyui/models/loras",
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
# Modal app + volume (separate app from v6; shared volume with v6)
# ---------------------------------------------------------------------------

app = modal.App("trurender-qwen-comfyui")
model_volume = modal.Volume.from_name("trurender-model-cache", create_if_missing=True)

VOLUME_PATH = "/models"
COMFYUI_DIR = "/comfyui"
COMFYUI_PORT = 8188


# ---------------------------------------------------------------------------
# Model download specs
# ---------------------------------------------------------------------------

MODEL_SPECS = [
    # fp8mixed weights (default — cost lever, ~20GB)
    {
        "name": "qwen_image_edit_2511_fp8mixed.safetensors",
        "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "filename": "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "dest": "qwen_models/unet",
    },
    # NOTE: bf16 weights (~41GB) intentionally NOT staged by default. The brief
    # lists exactly 5 models and fp8mixed is the default quality. To enable bf16
    # later, set UNETLoader's unet_name to qwen_image_edit_2511_bf16.safetensors
    # and add the file to MODEL_SPECS. Quality difference vs fp8mixed is
    # negligible; cost difference is ~2x VRAM.
    # Qwen2.5-VL text+vision encoder
    {
        "name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "dest": "qwen_models/clip",
    },
    # Qwen VAE
    {
        "name": "qwen_image_vae.safetensors",
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "filename": "split_files/vae/qwen_image_vae.safetensors",
        "dest": "qwen_models/vae",
    },
    # Lightning 4-step LoRA (optional, for cheap preview passes)
    {
        "name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "dest": "qwen_models/loras",
    },
    # TODO: depth backstop — DiffSynth depth control patch. NOT wired into
    # build_workflow() by default. Per task brief: "Include the model download
    # but don't wire it into build_workflow(). Leave a # TODO: depth backstop
    # comment with a brief note." Stage it on the volume so it's available if
    # we ever need to wire it in.
    {
        "name": "qwen_image_depth_diffsynth_controlnet.safetensors",
        "repo": "Comfy-Org/Qwen-Image-DiffSynth-ControlNets",
        "filename": "split_files/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors",
        "dest": "qwen_models/model_patches",
    },
]


# ---------------------------------------------------------------------------
# One-time model download function
# ---------------------------------------------------------------------------

@app.function(
    image=comfyui_image,
    volumes={VOLUME_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=3600,  # ~28GB of models, allow 1 hour
)
def download_models():
    """Download all required Qwen models to the volume. Run once before first deploy.

    Stages the v7 model stack on the existing trurender-model-cache volume
    (shared with v6). v6 and v7 use disjoint subdirectories (qwen_models/ vs
    comfyui_models/), so they coexist on the same volume.
    """
    from huggingface_hub import hf_hub_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[TruRender v7] Authenticated with HuggingFace")

    for spec in MODEL_SPECS:
        dest_dir = os.path.join(VOLUME_PATH, spec["dest"])
        dest_filename = spec.get("dest_filename", spec["name"])
        dest_path = os.path.join(dest_dir, dest_filename)

        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            size_gb = size_mb / 1024
            print(f"[TruRender v7] ✓ {spec['name']} already exists ({size_gb:.2f} GB)")
            continue

        print(f"[TruRender v7] Downloading {spec['name']} from {spec['repo']}...")
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
        size_gb = size_mb / 1024
        print(f"[TruRender v7] ✓ {spec['name']} downloaded ({size_gb:.2f} GB)")

    model_volume.commit()
    print("[TruRender v7] All models downloaded and committed to volume.")

    # List volume contents (v6 + v7 subdirs)
    for root, dirs, files in os.walk(VOLUME_PATH):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            size_gb = size_mb / 1024
            if size_mb >= 1024:
                size_str = f"{size_gb:.2f} GB"
            else:
                size_str = f"{size_mb:.0f} MB"
            print(f"  {fpath} ({size_str})")


# ---------------------------------------------------------------------------
# TruRender v7 ComfyUI service
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A100-80GB",
    image=comfyui_image,
    volumes={VOLUME_PATH: model_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    scaledown_window=600,  # 10 min idle timeout
    timeout=900,  # 15 min per request max
)
@modal.concurrent(max_inputs=10)  # route multiple HTTP requests to same container
class TruRenderQwen:

    @modal.enter()
    def start(self):
        """Setup model symlinks, start ComfyUI server, wait for ready, warmup."""
        import subprocess

        print("[TruRender v7] Setting up model symlinks...")
        self._setup_model_links()

        print("[TruRender v7] Starting ComfyUI server...")
        self.comfyui_process = subprocess.Popen(
            [
                "python", "main.py",
                "--listen", "127.0.0.1",
                "--port", str(COMFYUI_PORT),
                "--disable-auto-launch",
                "--verbose",
            ],
            cwd=COMFYUI_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

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
        print("[TruRender v7] ComfyUI server ready!")

        # Check that the Qwen-Image-Edit nodes are available
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/object_info")
            obj_info = json.loads(req.read())
            qwen_nodes = [k for k in obj_info.keys() if 'qwen' in k.lower() or 'Qwen' in k]
            essential_nodes = [
                "UNETLoader", "CLIPLoader", "VAELoader", "VAEEncode", "VAEDecode",
                "KSampler", "SaveImage", "LoadImage", "EmptySD3LatentImage",
                "ModelSamplingAuraFlow", "CFGNorm", "TextEncodeQwenImageEditPlus",
                "LoraLoaderModelOnly",
            ]
            missing = [n for n in essential_nodes if n not in obj_info]
            print(f"[TruRender v7] Qwen-related nodes: {qwen_nodes}")
            if missing:
                print(f"[TruRender v7] WARNING: missing essential nodes: {missing}")
            else:
                print(f"[TruRender v7] ✓ All essential nodes present (including TextEncodeQwenImageEditPlus)")
        except Exception as e:
            print(f"[TruRender v7] Could not query node info: {e}")

        # Warm up models by running a tiny dummy workflow
        print("[TruRender v7] Warming up models (first run loads into VRAM)...")
        self._warmup()

    @modal.exit()
    def stop(self):
        """Terminate ComfyUI server."""
        if hasattr(self, "comfyui_process") and self.comfyui_process:
            self.comfyui_process.terminate()
            self.comfyui_process.wait(timeout=10)
            print("[TruRender v7] ComfyUI server stopped.")

    def _setup_model_links(self):
        """Symlink Qwen models from volume to ComfyUI directories."""
        # v7 symlinks (Qwen stack)
        links = {
            f"{VOLUME_PATH}/qwen_models/unet/qwen_image_edit_2511_fp8mixed.safetensors":
                f"{COMFYUI_DIR}/models/unet/qwen_image_edit_2511_fp8mixed.safetensors",
            f"{VOLUME_PATH}/qwen_models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors":
                f"{COMFYUI_DIR}/models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            f"{VOLUME_PATH}/qwen_models/vae/qwen_image_vae.safetensors":
                f"{COMFYUI_DIR}/models/vae/qwen_image_vae.safetensors",
            f"{VOLUME_PATH}/qwen_models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors":
                f"{COMFYUI_DIR}/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            f"{VOLUME_PATH}/qwen_models/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors":
                f"{COMFYUI_DIR}/models/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors",
        }

        for src, dst in links.items():
            if not os.path.exists(src):
                print(f"[TruRender v7] WARNING: Model not found: {src}")
                print(f"[TruRender v7]   Run: modal run trurender_qwen_comfyui.py::download_models")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"
            print(f"[TruRender v7] ✓ Linked {os.path.basename(src)} ({size_str})")

    def _wait_for_comfyui(self, timeout: int = 300):
        """Wait for ComfyUI server to be ready."""
        import urllib.request
        import urllib.error

        start = time.time()
        url = f"http://127.0.0.1:{COMFYUI_PORT}/system_stats"

        while time.time() - start < timeout:
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
            print(f"[TruRender v7] ComfyUI /prompt HTTP {e.code}: {err_body}")
            raise
        result = json.loads(resp.read())
        print(f"[TruRender v7] Queue response: {json.dumps(result)}")

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
                req = urllib.request.urlopen(
                    f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}"
                )
                history = json.loads(req.read())

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    print(f"[TruRender v7] Prompt {prompt_id[:8]} status: {json.dumps(status)}")
                    if status.get("completed", False) or status.get("status_str") == "success":
                        return entry
                    if status.get("status_str") == "error":
                        raise RuntimeError(
                            f"ComfyUI workflow failed: {json.dumps(status, indent=2)}"
                        )

                elapsed = time.time() - start
                if elapsed - last_queue_log > 10:
                    try:
                        qreq = urllib.request.urlopen(
                            f"http://127.0.0.1:{COMFYUI_PORT}/queue"
                        )
                        queue = json.loads(qreq.read())
                        running = queue.get("queue_running", [])
                        pending = queue.get("queue_pending", [])
                        print(f"[TruRender v7] Queue: {len(running)} running, {len(pending)} pending ({elapsed:.0f}s elapsed)")
                    except Exception:
                        pass
                    last_queue_log = elapsed

            except (urllib.error.URLError, ConnectionRefusedError):
                pass

            time.sleep(2)

        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/queue")
            queue = json.loads(req.read())
            print(f"[TruRender v7] TIMEOUT - Final queue state: {json.dumps(queue)}")
            req2 = urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}")
            hist = json.loads(req2.read())
            print(f"[TruRender v7] TIMEOUT - Final history: {json.dumps(hist)}")
        except Exception as e:
            print(f"[TruRender v7] TIMEOUT - Could not get debug info: {e}")

        raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout}s")

    def _get_output_image(self, history_entry: dict, node_id: str = "13") -> bytes:
        """Extract the output image from a completed workflow history entry."""
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

    def _warmup(self):
        """Run a tiny dummy workflow to force model loading into VRAM.

        Uses a small 64x64 test image and 4 steps so the warmup is fast but
        the model and text encoder still get loaded.
        """
        from PIL import Image as PILImage

        try:
            img = PILImage.new("RGB", (64, 64), (128, 128, 128))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            warmup_start = time.time()
            output, elapsed = self._render_single(
                img_bytes, seed=1, steps=4, target_width=128, target_height=128,
            )
            total = time.time() - warmup_start
            print(f"[TruRender v7] Warmup complete in {total:.1f}s (render: {elapsed:.1f}s). All models loaded into VRAM.")
        except Exception as e:
            print(f"[TruRender v7] Warmup failed: {e}")
            print(f"[TruRender v7] First real render will be slow (model loading).")

    @modal.method()
    def _render_single(self, image_bytes: bytes, seed: int = 42,
                       target_width: int = None, target_height: int = None,
                       steps: int = 40, cfg: float = 4.0,
                       sampler_name: str = "euler", scheduler: str = "simple",
                       use_fp8: bool = True,
                       use_lightning_lora: bool = False,
                       output_format: str = "png") -> tuple:
        """Run a single Qwen-Image-Edit render through ComfyUI.

        Returns (output_image_bytes, elapsed_seconds).

        If target_width/target_height are not provided, dimensions are computed
        from the input image with compute_aspect_preserving_dims() — preserving
        the source aspect ratio at a ~1MP bucket snapped to multiples of 16.
        """
        from PIL import Image as PILImage
        start = time.time()
        client_id = uuid.uuid4().hex

        # Compute target dimensions preserving aspect ratio
        if target_width is None or target_height is None:
            target_width, target_height = compute_aspect_preserving_dims(image_bytes)

        # Sanity-check input
        src_img = PILImage.open(io.BytesIO(image_bytes))
        src_w, src_h = src_img.size
        fp8_str = "fp8" if use_fp8 else "bf16"
        lora_str = f" | lora@1.0" if use_lightning_lora else ""
        print(f"[TruRender v7] Input: {src_w}x{src_h} → Output: {target_width}x{target_height} | "
              f"steps={steps} cfg={cfg} {fp8_str}{lora_str} | seed={seed}")

        # Upload input image
        filename = f"trurender_v7_input_{client_id[:8]}.png"
        uploaded_name = self._upload_image(image_bytes, filename)

        # Build and submit workflow
        workflow = build_workflow(
            image_name=uploaded_name,
            seed=seed,
            target_width=target_width,
            target_height=target_height,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            use_fp8=use_fp8,
            use_lightning_lora=use_lightning_lora,
            filename_prefix=f"trurender_v7_{client_id[:8]}",
        )

        prompt_id = self._queue_prompt(workflow, client_id)
        print(f"[TruRender v7] Queued prompt {prompt_id} (seed={seed})")

        # Wait for completion
        history = self._poll_result(prompt_id)

        # Get output image (node 13 = SaveImage in v7 workflow)
        output_bytes = self._get_output_image(history, node_id="13")
        elapsed = time.time() - start
        print(f"[TruRender v7] Render complete: seed={seed}, {elapsed:.1f}s, {len(output_bytes)} bytes")

        return output_bytes, elapsed


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="render")
def render(
    input_path: str,
    output_path: str = None,
    seed: int = 42,
    steps: int = 40,
    cfg: float = 4.0,
    use_fp8: bool = True,
    use_lightning_lora: bool = False,
    target_width: int = None,
    target_height: int = None,
):
    """Render an Enscape image with the v7 (Qwen-Image-Edit) pipeline.

    Usage:
        modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png
        modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png --output-path out.png --seed 1234
    """
    from datetime import datetime

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    image_bytes = input_path.read_bytes()
    print(f"[TruRender v7] Local: read {len(image_bytes)} bytes from {input_path}")

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"/tmp/trurender_v7_render_{ts}.png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Call the remote class
    output_bytes, elapsed = TruRenderQwen()._render_single.remote(
        image_bytes, seed=seed, target_width=target_width, target_height=target_height,
        steps=steps, cfg=cfg, use_fp8=use_fp8, use_lightning_lora=use_lightning_lora,
    )

    output_path.write_bytes(output_bytes)
    print(f"[TruRender v7] ✓ Rendered in {elapsed:.1f}s → {output_path} ({len(output_bytes)} bytes)")


@app.local_entrypoint()
def main():
    """Quick smoke test — prints deployment info."""
    print("=" * 60)
    print("TruRender v7.0 — Qwen-Image-Edit-2511 on Modal")
    print("=" * 60)
    print()
    print("Setup:")
    print("  1. Download models (run once, ~25-30 min for ~28GB):")
    print("     modal run trurender_qwen_comfyui.py::download_models")
    print()
    print("  2. Deploy:")
    print("     modal deploy trurender_qwen_comfyui.py")
    print()
    print("  3. Render:")
    print("     modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png")
    print()
    print("Architecture:")
    print("  Model:        Qwen-Image-Edit-2511 (fp8mixed default, bf16 optional)")
    print("  TextEncoder:  Qwen2.5-VL-7B (fp8 scaled)")
    print("  VAE:          Qwen VAE")
    print("  Sampler:      euler / simple / denoise=1.0 / cfg=4.0 / steps=40")
    print("  Aspect:       Python-computed dims → EmptySD3LatentImage (preserves aspect)")
    print("  License:      Apache 2.0 (commercial use OK)")
    print("=" * 60)

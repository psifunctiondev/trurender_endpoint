"""
TruRender — Modal serverless endpoint for architectural render → photorealistic image.

Uses Flux.1-dev with:
  - ControlNet Union Pro 2.0 (multi-control: canny + depth)
  - img2img pipeline (low denoise to preserve structure)
  - Two-pass refinement for quality

Pipeline: Enscape/SketchUp render → extract canny edges + depth map → 
          Flux img2img with dual ControlNet → photorealistic output

Deploy:   cd agents/tekton/modal && modal deploy trurender.py
Test:     cd agents/tekton/modal && modal run trurender.py
"""

import modal
import io
import base64
import time
import uuid

# ---------------------------------------------------------------------------
# Modal image with all dependencies
# ---------------------------------------------------------------------------

trurender_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV deps
    .pip_install(
        "torch==2.5.1",
        "diffusers==0.33.0",
        "transformers>=4.44.0,<5.0.0",
        "accelerate>=0.30.0,<1.0.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "hf_transfer",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy",
        "fastapi[standard]",
        "python-multipart",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("trurender")
model_volume = modal.Volume.from_name("trurender-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
CONTROLNET_UNION_ID = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ---------------------------------------------------------------------------
# TruRender service class
# ---------------------------------------------------------------------------


@app.cls(
    gpu="A100-80GB",
    image=trurender_image,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    scaledown_window=600,  # 10 min idle timeout
    timeout=600,  # 10 min per request max
)
class TruRender:
    @modal.enter()
    def load_models(self):
        """Download and load all models into GPU memory on container start."""
        import os
        import torch
        from huggingface_hub import snapshot_download, login
        from diffusers import (
            FluxControlNetImg2ImgPipeline,
            FluxControlNetModel,
        )
        from transformers import pipeline as hf_pipeline

        # Authenticate with HuggingFace
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("[TruRender] Authenticated with HuggingFace")
        else:
            print("[TruRender] WARNING: No HF_TOKEN found!")

        print("[TruRender] Downloading models...")

        # Download ControlNet Union
        cn_path = f"/models/{CONTROLNET_UNION_ID}"
        snapshot_download(CONTROLNET_UNION_ID, local_dir=cn_path, token=hf_token)

        # Download Flux base model (gated — requires token)
        flux_path = f"/models/{FLUX_MODEL_ID}"
        snapshot_download(FLUX_MODEL_ID, local_dir=flux_path, token=hf_token)

        # Download Depth Anything V2
        depth_path = f"/models/{DEPTH_MODEL_ID}"
        snapshot_download(DEPTH_MODEL_ID, local_dir=depth_path, token=hf_token)

        model_volume.commit()
        print("[TruRender] Models downloaded, loading pipeline...")

        # Load ControlNet Union Pro 2.0
        # Pro 2.0 removed mode embeddings, so we load as a single model
        self.controlnet = FluxControlNetModel.from_pretrained(
            cn_path, torch_dtype=torch.bfloat16
        )

        # Load the full pipeline: Flux + ControlNet + img2img
        # Use single controlnet (Union Pro 2.0 handles multiple control types internally)
        self.pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
            flux_path,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

        # Load Depth Anything V2 for depth map extraction
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model=depth_path,
            device="cuda",
            torch_dtype=torch.float16,
        )

        print("[TruRender] Pipeline loaded and ready.")

    def _extract_depth(self, image):
        """Extract depth map using Depth Anything V2."""
        from PIL import Image as PILImage

        result = self.depth_estimator(image)
        depth_map = result["depth"]  # PIL Image
        return depth_map.convert("RGB")

    def _extract_canny(self, image, low_threshold=50, high_threshold=150):
        """Extract canny edges from image."""
        import numpy as np
        import cv2
        from PIL import Image as PILImage

        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return PILImage.fromarray(edges_rgb)

    def _render(
        self,
        image,
        prompt: str = "",
        strength: float = 0.25,
        controlnet_conditioning_scale_canny: float = 0.7,
        controlnet_conditioning_scale_depth: float = 0.6,
        control_guidance_end: float = 0.8,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        seed: int = 42,
        use_dual_control: bool = True,
        second_pass: bool = True,
        second_pass_strength: float = 0.15,
        canny_low: int = 50,
        canny_high: int = 150,
        max_dim: int = 1024,
    ):
        """
        Core rendering function.

        Uses Flux.1-dev img2img with ControlNet Union Pro 2.0 to convert
        architectural renderings to photorealistic images.

        The key to fidelity: canny edges lock architectural lines and material
        boundaries; depth locks spatial structure; low img2img strength means
        the model starts from the original image and can only nudge toward
        photorealism, not reinvent.

        Args:
            image: PIL Image (the Enscape/SketchUp render)
            prompt: Positive prompt (default: optimized for arch. photorealism)
            strength: img2img denoising strength (lower = more faithful)
            controlnet_conditioning_scale_canny: Canny ControlNet strength
            controlnet_conditioning_scale_depth: Depth ControlNet strength
            control_guidance_end: When to stop ControlNet guidance (0-1)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            use_dual_control: Use both canny + depth (vs canny only)
            second_pass: Run a refinement pass at lower strength
            second_pass_strength: Denoising strength for refinement
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
        """
        import torch
        from PIL import Image as PILImage

        # Default prompt optimized for architectural interior photorealism
        if not prompt:
            prompt = (
                "professional architectural interior photograph, "
                "shot on Canon EOS R5 with 24mm tilt-shift lens, "
                "natural daylight through windows, physically accurate lighting, "
                "real-world materials and surface textures, "
                "subtle natural reflections on polished surfaces, "
                "correct scale and proportions, sharp architectural details, "
                "true-to-life colors and tones, high dynamic range, "
                "identical composition and layout to reference image, "
                "no changes to any furniture or materials or colors or objects, "
                "professional architectural photography, photorealistic, 8K detail"
            )

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Resize to Flux-friendly resolution while preserving aspect ratio
        # Must be divisible by 16 for VAE
        # Flux.1-dev works reliably up to ~3072 on longest side.
        # If max_dim > 3072, we render at 3072 and upscale output afterward.
        FLUX_MAX_DIM = 3072
        w, h = image.size
        render_dim = min(max_dim, FLUX_MAX_DIM)
        scale = render_dim / max(w, h)
        if scale > 1.0:
            scale = 1.0  # never upscale the input beyond its native res
        new_w = int(w * scale) // 16 * 16
        new_h = int(h * scale) // 16 * 16
        # Remember the requested output size for final upscale
        if max_dim <= FLUX_MAX_DIM:
            out_w, out_h = new_w, new_h
        else:
            out_scale = max_dim / max(w, h)
            if out_scale > 1.0:
                out_scale = 1.0
            out_w = int(w * out_scale) // 16 * 16
            out_h = int(h * out_scale) // 16 * 16
        image_resized = image.resize((new_w, new_h), PILImage.LANCZOS)

        # Extract control signals
        canny_image = self._extract_canny(image_resized, canny_low, canny_high)

        if use_dual_control:
            depth_image = self._extract_depth(image_resized)
            depth_image = depth_image.resize((new_w, new_h), PILImage.LANCZOS)

            # PASS 1a: Depth-guided pass (spatial structure)
            result = self.pipe(
                prompt=prompt,
                image=image_resized,
                control_image=depth_image,
                width=new_w,
                height=new_h,
                strength=strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale_depth,
                control_guidance_end=control_guidance_end,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

            # PASS 1b: Canny-guided refinement (edge preservation)
            generator2 = torch.Generator(device="cuda").manual_seed(seed + 1)
            canny_from_result = self._extract_canny(result, canny_low, canny_high)
            result = self.pipe(
                prompt=prompt,
                image=result,
                control_image=canny_from_result,
                width=new_w,
                height=new_h,
                strength=second_pass_strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale_canny,
                control_guidance_end=control_guidance_end,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator2,
            ).images[0]
        else:
            # Single control: canny only
            result = self.pipe(
                prompt=prompt,
                image=image_resized,
                control_image=canny_image,
                width=new_w,
                height=new_h,
                strength=strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale_canny,
                control_guidance_end=control_guidance_end,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

        # Optional third pass: final refinement with original canny edges
        if second_pass and not use_dual_control:
            generator3 = torch.Generator(device="cuda").manual_seed(seed + 2)
            canny_refined = self._extract_canny(result, canny_low, canny_high)
            result = self.pipe(
                prompt=prompt,
                image=result,
                control_image=canny_refined,
                width=new_w,
                height=new_h,
                strength=second_pass_strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale_canny * 0.8,
                control_guidance_end=control_guidance_end,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator3,
            ).images[0]

        # Upscale to requested output resolution
        if (result.size[0], result.size[1]) != (out_w, out_h):
            result = result.resize((out_w, out_h), PILImage.LANCZOS)

        return result

    @modal.asgi_app()
    def web(self):
        """FastAPI web endpoint."""
        from fastapi import FastAPI, File, UploadFile, Form
        from fastapi.responses import Response, JSONResponse
        from PIL import Image as PILImage
        from typing import Optional

        web_app = FastAPI(title="TruRender", version="1.0.0")

        @web_app.post("/render")
        async def render(
            image: UploadFile = File(...),
            prompt: Optional[str] = Form(default=""),
            strength: float = Form(default=0.25),
            controlnet_scale_canny: float = Form(default=0.7),
            controlnet_scale_depth: float = Form(default=0.6),
            control_guidance_end: float = Form(default=0.8),
            guidance_scale: float = Form(default=3.5),
            num_steps: int = Form(default=28),
            seed: int = Form(default=42),
            dual_control: bool = Form(default=True),
            second_pass: bool = Form(default=True),
            second_pass_strength: float = Form(default=0.15),
            canny_low: int = Form(default=50),
            canny_high: int = Form(default=150),
            output_format: str = Form(default="png"),
            max_dim: int = Form(default=1024),
        ):
            """
            Render an architectural image to photorealistic.

            Upload an image file, get back the photorealistic version.

            Key parameters:
            - strength: How much to transform (0.15-0.35 recommended)
            - controlnet_scale_canny: Edge preservation strength (0.5-0.9)
            - controlnet_scale_depth: Spatial structure strength (0.4-0.8)
            - dual_control: Use both canny+depth (recommended True)
            - second_pass: Refinement pass (recommended True)
            """
            start_time = time.time()

            image_bytes = await image.read()
            input_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

            result = self._render(
                image=input_image,
                prompt=prompt,
                strength=strength,
                controlnet_conditioning_scale_canny=controlnet_scale_canny,
                controlnet_conditioning_scale_depth=controlnet_scale_depth,
                control_guidance_end=control_guidance_end,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                seed=seed,
                use_dual_control=dual_control,
                second_pass=second_pass,
                second_pass_strength=second_pass_strength,
                canny_low=canny_low,
                canny_high=canny_high,
                max_dim=max_dim,
            )

            buf = io.BytesIO()
            fmt = "PNG" if output_format.lower() == "png" else "JPEG"
            quality = 95 if fmt == "JPEG" else None
            result.save(buf, format=fmt, quality=quality)
            buf.seek(0)

            elapsed = time.time() - start_time
            media_type = f"image/{output_format.lower()}"

            return Response(
                content=buf.getvalue(),
                media_type=media_type,
                headers={
                    "X-TruRender-Time": f"{elapsed:.1f}s",
                    "X-TruRender-Size": f"{result.size[0]}x{result.size[1]}",
                    "X-TruRender-Seed": str(seed),
                },
            )

        @web_app.post("/render/json")
        async def render_json(
            image: UploadFile = File(...),
            prompt: Optional[str] = Form(default=""),
            strength: float = Form(default=0.25),
            controlnet_scale_canny: float = Form(default=0.7),
            controlnet_scale_depth: float = Form(default=0.6),
            control_guidance_end: float = Form(default=0.8),
            guidance_scale: float = Form(default=3.5),
            num_steps: int = Form(default=28),
            seed: int = Form(default=42),
            dual_control: bool = Form(default=True),
            second_pass: bool = Form(default=True),
            second_pass_strength: float = Form(default=0.15),
            canny_low: int = Form(default=50),
            canny_high: int = Form(default=150),
            max_dim: int = Form(default=1024),
        ):
            """
            Render and return result as JSON with base64 image.
            Useful for programmatic API calls.
            """
            start_time = time.time()

            image_bytes = await image.read()
            input_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

            result = self._render(
                image=input_image,
                prompt=prompt,
                strength=strength,
                controlnet_conditioning_scale_canny=controlnet_scale_canny,
                controlnet_conditioning_scale_depth=controlnet_scale_depth,
                control_guidance_end=control_guidance_end,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                seed=seed,
                use_dual_control=dual_control,
                second_pass=second_pass,
                second_pass_strength=second_pass_strength,
                canny_low=canny_low,
                canny_high=canny_high,
                max_dim=max_dim,
            )

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            elapsed = time.time() - start_time

            return {
                "id": f"trurender-{uuid.uuid4().hex[:12]}",
                "image_base64": result_b64,
                "size": f"{result.size[0]}x{result.size[1]}",
                "elapsed_seconds": round(elapsed, 1),
                "seed": seed,
                "parameters": {
                    "strength": strength,
                    "controlnet_scale_canny": controlnet_scale_canny,
                    "controlnet_scale_depth": controlnet_scale_depth,
                    "control_guidance_end": control_guidance_end,
                    "guidance_scale": guidance_scale,
                    "num_steps": num_steps,
                    "dual_control": dual_control,
                    "second_pass": second_pass,
                    "second_pass_strength": second_pass_strength,
                },
            }

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "trurender",
                "version": "1.0.0",
                "model": FLUX_MODEL_ID,
                "controlnet": CONTROLNET_UNION_ID,
                "depth_model": DEPTH_MODEL_ID,
            }

        @web_app.get("/")
        def root():
            return {
                "service": "trurender",
                "version": "1.0.0",
                "description": (
                    "Architectural render → photorealistic image conversion. "
                    "Uses Flux.1-dev + ControlNet Union Pro 2.0 with dual "
                    "canny/depth control and two-pass refinement."
                ),
                "endpoints": {
                    "/render": "POST multipart - upload image, receive image back",
                    "/render/json": "POST multipart - upload image, receive JSON with base64",
                    "/health": "GET - health check",
                },
                "recommended_params": {
                    "strength": "0.20-0.30 (lower = more faithful to original)",
                    "controlnet_scale_canny": "0.6-0.8 (edge preservation)",
                    "controlnet_scale_depth": "0.5-0.7 (spatial structure)",
                    "dual_control": "true (use both canny + depth)",
                    "second_pass": "true (refinement for quality)",
                },
            }

        return web_app


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Quick smoke test — prints deployment info."""
    print("=" * 60)
    print("TruRender — Architectural Render → Photorealistic")
    print("=" * 60)
    print(f"Base model:  {FLUX_MODEL_ID}")
    print(f"ControlNet:  {CONTROLNET_UNION_ID}")
    print(f"Depth model: {DEPTH_MODEL_ID}")
    print()
    print("Endpoints:")
    print("  POST /render      — upload image file, get image back")
    print("  POST /render/json — upload image file, get JSON + base64")
    print("  GET  /health      — health check")
    print()
    print("Example curl:")
    print('  curl -X POST -F "image=@interior.png" -F "strength=0.25" \\')
    print("    https://<endpoint>/render -o output.png")
    print("=" * 60)

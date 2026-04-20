# TruRender Deploy Log

Rule: **Every `modal deploy` must be preceded by a git commit of the code being deployed.**

## ⚠️ CRITICAL: Two Separate Pipelines

There are TWO different TruRender implementations. Confusing them caused days of wasted work.

| Pipeline | Modal App | Code | Status |
|----------|-----------|------|--------|
| **ComfyUI** (XLabs ControlNet V3) | `trurender-comfyui` | `trurender_comfyui.py` | ✅ WORKING — all good renders came from here |
| **Diffusers** (Shakker-Labs Union Pro 2.0) | `trurender` | `trurender_diffusers_BROKEN.py` | ❌ BROKEN — produces hallucinated scenes |

**Correct endpoint:** `https://psifunctiondev--trurender-comfyui-trurendercomfyui-web.modal.run/render`
**Wrong endpoint:** `https://psifunctiondev--trurender-trurender-web.modal.run/render`

## Deployments

### ComfyUI Pipeline (the one that works)

#### 2026-04-18 ~11:27 EDT — v5 (Canny + Depth + HED)
- **Code:** `trurender_comfyui.py` (69KB, committed `15ab5b5`)
- **Modal app:** `trurender-comfyui` (`ap-bnSPBRBuDOueo0ecBD8QvZ`)
- **Endpoint:** `https://psifunctiondev--trurender-comfyui-trurendercomfyui-web.modal.run/render`
- **Pipeline:** ComfyUI + Flux.1-dev UNET + XLabs Canny ControlNet V3 + XLabs Depth ControlNet V3 + Depth Anything ViT-L
- **Sampler:** DPM++ 2M Karras, cfg 3.5
- **GPU:** A100 80GB
- **Result:** All approved round 2 renders. Blue chairs preserved, red brick buildings preserved, all colors/materials accurate.
- **Verified working:** 2026-04-20 (test render confirmed all scene elements correct)

#### 2026-04-17 ~09:46 EDT — v3 (Depth-only predecessor)
- **Code:** `trurender_comfyui_v3_ARCHIVED.py` (53KB)
- **Pipeline:** Depth ControlNet only (no canny). Predecessor to v5.

### Diffusers Pipeline (the broken one)

#### 2026-04-19 ~12:55 EDT — Resolution cap fix (still broken)
- **Commit:** `56b5820`
- **Code:** `trurender_diffusers_BROKEN.py` (renamed from trurender.py)
- **Modal app:** `trurender` (`ap-CM2D4zA7cTkVtDnhTCEGpo`)
- **Pipeline:** diffusers FluxControlNetImg2ImgPipeline + Shakker-Labs Union Pro 2.0
- **Problem:** Produces completely hallucinated scenes — preserves room geometry via ControlNet but replaces all colors, materials, furniture styles. Blue chairs become beige, red brick becomes greenery, etc.
- **Root cause:** This pipeline was never the one that produced good renders. It was a parallel experiment that doesn't work for our fidelity requirements.

#### 2026-04-14 ~16:36 EDT — Original diffusers deployment
- **Commit:** none (untracked, code lost)
- **Note:** May have worked differently at lower denoise; never properly evaluated.

## Lessons Learned

1. **Always know which endpoint you're hitting.** Two apps with similar names = disaster.
2. **Always commit before deploying.** (`15ab5b5` for comfyui, nothing for original diffusers = lost code)
3. **Validate renders visually, not just by HTTP status and file size.**
4. **The ComfyUI pipeline is the correct one.** XLabs ControlNet V3 + ComfyUI preserves scene fidelity. Diffusers + Union Pro 2.0 does not.

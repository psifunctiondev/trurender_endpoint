"""
TruRender v7.4 — Qwen-Image-Edit-2511 pipeline (edit-first paradigm).

v7.4 adds multi-image style-reference conditioning. The Qwen-Image-Edit-2511
model has a second image input (image2) on its TextEncodeQwenImageEditPlus
node that takes a style reference; the encoder sees both images and the
model can be instructed to apply the style of Figure 2 onto Figure 1. This
enables CTAI (Catherine-Approved) style guidance without IP-Adapter or any
other external conditioning network.

Backward compat: the style path is opt-in. `style_image_name=None` produces
a v7.3-BASE-bit-identical workflow (one LoadImage, image1 only on both
TextEncodeQwenImageEditPlus instances).

v7.3 added the DiffSynth depth ControlNet backstop (default strength 0.3,
validated 2026-06-19 as the noise-reduction sweet spot). v7.3 default
positive/negative are preserved verbatim.

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
Output dimensions match the input aspect ratio at a ~2MP bucket, snapped to
multiples of 16. This replaces v6's ImageScale-to-1024x1024 (which stretched
16:9 input ~22% horizontally) and Quinn's original FluxKontextImageScale
(a BFL custom node we explicitly avoid). 2MP is the smallest target that
suppresses the right-edge hallucination bug Qwen-Image-Edit shows on 16:9
input at ~1MP; see v7.1 release notes.

Production entrypoints:
  - `render(input_path, output_path=None, seed=DEFAULT_SEED, ...)` — single image.
  - `render_options(input_path, output_dir=None, num_options=5, anchor_seed=DEFAULT_SEED, ...)`
    — N seed-variation options in one warm Modal container. This is the end-user
    "give me options" path: 5 renders per run, first at DEFAULT_SEED (1234, the
    v7.2 chosen hero), next 4 at random seeds. Cost is ~$0.55 + $0.03 per option.
  - `sweep(spec_path)` — JSON-spec parameter sweep (for internal probing).

Models staged on the existing trurender-model-cache volume (shared with v6):
  - qwen_image_edit_2511_fp8mixed.safetensors  (~20GB, default — cost lever)
  - qwen_2.5_vl_7b_fp8_scaled.safetensors      (~9GB,  text+vision encoder)
  - qwen_image_vae.safetensors                  (~242MB)
  - Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors (~810MB, optional)
  - qwen_image_depth_diffsynth_controlnet.safetensors (~2GB, optional backstop, NOT wired)
  - (bf16 weights ~41GB intentionally NOT staged; v7.1 probe showed fp8mixed
    is a wash vs bf16 with ~15% speed penalty. See MODEL_SPECS comment.)

Deploy:    cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal deploy trurender_qwen_comfyui.py
Models:    cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_qwen_comfyui.py::download_models
Render:    cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png
Options:   cd pipeline && /Users/doxa/Library/Python/3.9/bin/modal run trurender_qwen_comfyui.py::render_options --input-path /path/to/enscape.png --num-options 5
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
    "3D render, plastic surfaces, distorted geometry, text, watermark, "
    "added furniture"
)

# ---------------------------------------------------------------------------
# v7.4: multi-image style-reference conditioning.
# When style_image_name is set, the workflow wires a second image as
# TextEncodeQwenImageEditPlus.image2 (the style reference). The positive
# prompt is swapped to a "match the style of Figure 2" trigger. The
# 6-item negative is preserved verbatim (v7.3 trim, locked).
# ---------------------------------------------------------------------------

DEFAULT_POSITIVE_STYLE = (
    "Change the style of Figure 1 to the style of Figure 2. "
    "Photorealistic interior photograph. Preserve exact room layout, "
    "camera angle, and object positions. Do not change any material or finish."
)


# ---------------------------------------------------------------------------
# Model file names (must match what download_models stages on the volume)
# ---------------------------------------------------------------------------

DIFFUSION_BF16 = "qwen_image_edit_2511_bf16.safetensors"  # noqa: kept for reference; not staged by default
DIFFUSION_FP8 = "qwen_image_edit_2511_fp8mixed.safetensors"   # ~20GB cost lever (default)
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
DEPTH_DIFFSYNTH_CONTROLNET = "qwen_image_depth_diffsynth_controlnet.safetensors"  # DiffSynth depth ControlNet (~2.11 GB, optional backstop)
DEPTH_ANYTHING_MODEL = "depth_anything_vitl14.pth"  # Used by DepthAnythingPreprocessor
# v6 trurender_comfyui.py stages the same depth_anything_vitl14.pth on the volume

# ---------------------------------------------------------------------------
# Default seed (anchor for the "5 options per run" production path)
# ---------------------------------------------------------------------------
# Seed 1234 was selected as the v7.2 default after the seed variation trial
# (outputs/seed_trial/ on 2026-06-18, 4 cells: 1, 7, 100, 1234). It was the only
# cell with a clearly visible blue sky through the right windows, the most
# "shippable bright daylight" render in the set, and the only seed in the trial
# whose exposure balance read as intentional rather than lucky. End users get
# this seed as the first option, then 4 random seeds as variations.
DEFAULT_SEED = 1234


# ---------------------------------------------------------------------------
# Aspect-preservation helper
# ---------------------------------------------------------------------------

def compute_aspect_preserving_dims(image_bytes: bytes, target_megapixels: float = 2.0,
                                   max_side: int = 2048) -> tuple:
    """Read image, compute (width, height) preserving aspect ratio at ~target_megapixels.

    Defaults: target_megapixels=2.0, max_side=2048. Both bumped from the original
    (1.0, 1536) defaults after round-1 blind testing revealed Qwen-Image-Edit's
    vision encoder hallucinates structured content along the right border at ~1MP
    for 16:9 inputs. Diagnostic (`outputs/v7_diag_borders/` on 2026-06-18) showed:
        - 1MP 1360x768  → TR/TL std ratio 1.18 (right-edge spike, visible artifact)
        - 2MP 1920x1072 → TR/TL std ratio 0.68 (right cleaner than left, bug gone)
        - 1:1 square    → TR/TL std ratio 0.20 (full suppression but warps geometry)
    2MP is the smallest target that suppresses the bug while preserving 16:9 aspect.

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
# Smart style-ref selector (v7.4)
# Unifies the 16 canonical CTAI refs (local assets/trurender/style-refs/) with
# Catherine's 3 personal picks (Modal volume /models/style_references/) and
# returns the top-1 reference for a given room type. Scoring per the
# style-refs.json manifest: score = base_weight * tag_weights[space_type][ref_tag],
# summed across the ref's tags. Highest score wins.
# ---------------------------------------------------------------------------

STYLE_REFS_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "assets" / "trurender" / "style-refs" / "style-refs.json"

# Catherine's 3 picks — hardcoded tags per brief (filenames → tags). The volume
# copies are uploaded via pipeline/upload_style_refs.py to /models/style_references/.
CATHERINE_PICKS = [
    {
        "filename": "style_ref_1_library.jpg",
        "tags": ["interior", "library"],
        "weight": 1.0,
        "source": "catherine",
        "notes": "Catherine pick #1 — library interior",
    },
    {
        "filename": "style_ref_2_staircase.jpg",
        "tags": ["interior", "entry", "stair"],
        "weight": 1.0,
        "source": "catherine",
        "notes": "Catherine pick #2 — staircase / entry",
    },
    {
        "filename": "style_ref_3_kitchen.jpg",
        "tags": ["interior", "kitchen"],
        "weight": 1.0,
        "source": "catherine",
        "notes": "Catherine pick #3 — kitchen",
    },
]


def _load_canonical_refs() -> list:
    """Load the 16 canonical CTAI refs from the local style-refs.json manifest.

    Returns a list of dicts with keys: filename, tags, weight, source, notes,
    local_path. local_path is the local absolute path to the JPG (sibling of
    style-refs.json in assets/trurender/style-refs/).
    """
    import json as _json
    if not STYLE_REFS_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"style-refs.json manifest not found at {STYLE_REFS_MANIFEST_PATH}. "
            "v7.4 smart selector needs the CTAI canonical 16."
        )
    with open(STYLE_REFS_MANIFEST_PATH) as f:
        manifest = _json.load(f)
    refs_dir = STYLE_REFS_MANIFEST_PATH.parent
    out = []
    for r in manifest["refs"]:
        out.append({
            "filename": r["filename"],
            "tags": list(r.get("tags", [])),
            "weight": float(r.get("weight", manifest.get("default_weight", 0.3))),
            "source": "canonical",
            "notes": r.get("notes", ""),
            "local_path": str(refs_dir / r["filename"]),
        })
    return out


def _load_catherine_picks() -> list:
    """Return Catherine's 3 picks with their hardcoded tags.

    These live on the Modal volume at /models/style_references/. Tag mapping
    is hardcoded per the brief (option b) since no metadata is stored on the
    volume alongside the images.
    """
    return [{
        "filename": p["filename"],
        "tags": list(p["tags"]),
        "weight": float(p["weight"]),
        "source": "catherine",
        "notes": p["notes"],
        "local_path": None,  # not local — fetched from Modal volume at render time
    } for p in CATHERINE_PICKS]


def _all_refs() -> list:
    """Return the unified 19 references (16 canonical + 3 Catherine picks)."""
    return _load_canonical_refs() + _load_catherine_picks()


def _score_ref(ref: dict, space_type: str, tag_weights: dict) -> float:
    """Score a single ref against the target space_type.

    score = base_weight * sum(tag_weights[space_type][tag] for tag in ref.tags)
    Returns 0.0 if the space_type is not in tag_weights (selector would skip).
    """
    if space_type not in tag_weights:
        return 0.0
    multipliers = tag_weights[space_type]
    score = 0.0
    for tag in ref["tags"]:
        m = multipliers.get(tag, 0.0)
        score += ref["weight"] * float(m)
    return score


def select_style_ref(space_type: str, override_filename: str = None) -> dict:
    """Pick the best style reference for the given room type.

    Args:
        space_type: One of "kitchen", "living", "bath", "exterior", "entry",
                    "dining". Must be a key in tag_weights.
        override_filename: If set, return this specific ref (skip scoring).
                           Must match a filename in the unified 19.

    Returns:
        Dict with keys: filename, tags, weight, source, score, local_path,
        and a `why` string explaining the pick.

    Raises:
        FileNotFoundError: if style-refs.json manifest is missing.
        ValueError: if space_type is unknown or override_filename doesn't match.
    """
    import json as _json
    if not STYLE_REFS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"style-refs.json manifest not found at {STYLE_REFS_MANIFEST_PATH}")
    with open(STYLE_REFS_MANIFEST_PATH) as f:
        manifest = _json.load(f)
    tag_weights = manifest["tag_weights"]

    refs = _all_refs()

    if override_filename:
        for r in refs:
            if r["filename"] == override_filename:
                score = _score_ref(r, space_type, tag_weights) if space_type in tag_weights else r["weight"]
                return {
                    "filename": r["filename"],
                    "tags": r["tags"],
                    "weight": r["weight"],
                    "source": r["source"],
                    "score": score,
                    "local_path": r.get("local_path"),
                    "why": f"explicit override ({override_filename}); space_type={space_type} score={score:.3f}",
                }
        raise ValueError(
            f"override_filename={override_filename!r} not found in unified 19 refs. "
            f"Available: {[r['filename'] for r in refs]}"
        )

    if space_type not in tag_weights:
        raise ValueError(
            f"unknown space_type={space_type!r}. Valid: {list(tag_weights.keys())}"
        )

    scored = [(r, _score_ref(r, space_type, tag_weights)) for r in refs]
    # Filter out refs with score 0 (no tag overlap with this space_type)
    scored = [(r, s) for r, s in scored if s > 0]
    if not scored:
        raise RuntimeError(
            f"smart selector found zero refs with positive score for space_type={space_type!r}. "
            "This is a bug — every space_type should match at least the base 'interior' tag."
        )
    scored.sort(key=lambda rs: rs[1], reverse=True)
    top_ref, top_score = scored[0]

    # Build "why" string: top 3 contributors
    multipliers = tag_weights[space_type]
    contribs = []
    for tag in top_ref["tags"]:
        m = multipliers.get(tag, 0.0)
        if m > 0:
            contribs.append((tag, top_ref["weight"] * m))
    contribs.sort(key=lambda c: c[1], reverse=True)
    why_parts = [
        f"ref={top_ref['filename']} ({top_ref['source']}) score={top_score:.3f}",
        f"top_tag_contribs: " + ", ".join(f"{t}={s:.3f}" for t, s in contribs[:3]),
        f"space_type={space_type} matched against {sum(1 for r, s in scored if s > 0)} refs",
    ]
    return {
        "filename": top_ref["filename"],
        "tags": top_ref["tags"],
        "weight": top_ref["weight"],
        "source": top_ref["source"],
        "score": top_score,
        "local_path": top_ref.get("local_path"),
        "why": " | ".join(why_parts),
        "scored_candidates": [
            {"filename": r["filename"], "source": r["source"], "score": s}
            for r, s in scored[:6]
        ],
    }


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
                   # v7.4: optional style-reference wiring. style_image_name=None
                   # means the v7.3 BASE workflow (no second LoadImage, image1
                   # only on both TextEncodeQwenImageEditPlus instances).
                   # Setting style_image_name adds a second LoadImage (node 100)
                   # and wires its output to image2 on BOTH encode nodes. The
                   # caller is expected to upload the style image to ComfyUI's
                   # input dir first (handled in _render_single).
                   style_image_name: str = None,
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
                   # Optional DiffSynth depth ControlNet backstop (Tier C #2, v7.3+):
                   # Default 0.3 (v7.3 production) — applies the DiffSynth depth
                   # ControlNet at strength 0.3, validated as the noise-reduction
                   # sweet spot (-56% right-edge TR/TL noise vs v7.2 baseline).
                   # Set to 0.0 to disable (preserves v7.2 behavior bit-identically).
                   # Range matches the QwenImageDiffsynthControlnet node: -10.0 to 10.0,
                   # practical range 0.0-1.0. Negative strengths subtract the depth
                   # signal (rarely useful, but supported for A/B experiments).
                   depth_strength: float = 0.3,
                   filename_prefix: str = "trurender_qwen") -> dict:
    """Return an API-format ComfyUI prompt graph (dict keyed by node id).

    With use_lightning_lora=True the sampler drops to 4 steps / cfg 1.0 and a
    LoraLoaderModelOnly node is inserted after CFGNorm. Use that for cheap
    preview passes; leave it off for final client deliverables.

    With use_fp8=True (default) the fp8mixed weights are loaded (~20GB, fits
    A100-80GB with room for the 9GB text encoder; negligible quality loss).
    Set use_fp8=False for the bf16 weights (~41GB).

    With depth_strength > 0 (default 0.3 in v7.3), a DiffSynth Qwen-Image-Depth
    ControlNet is loaded (qwen_image_depth_diffsynth_controlnet.safetensors,
    ~2.11 GB) and inserted into the model chain between the last
    model-conditioning node (CFGNorm or Lightning LoRA) and the KSampler. The
    ControlNet's image input is the source Enscape render — DiffSynth's
    control net handles its own internal depth extraction from that image.
    depth_strength=0.0 disables the backstop (preserves v7.2 behavior
    bit-identically, no new nodes are inserted).
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

    # Optional DiffSynth depth ControlNet (Tier C #2 backstop, v7.3+ default).
    # When depth_strength > 0, we add two nodes:
    #   30: ModelPatchLoader — loads qwen_image_depth_diffsynth_controlnet.safetensors
    #   33: QwenImageDiffsynthControlnet — applies the depth patch to the model
    # The patched model replaces model_ref for the KSampler.
    # depth_strength=0.0 skips this block entirely (preserves v7.2 behavior
    # bit-identically). v7.3 default is 0.3, validated 2026-06-19.
    if depth_strength > 0:
        g["30"] = {
            "inputs": {"name": DEPTH_DIFFSYNTH_CONTROLNET},
            "class_type": "ModelPatchLoader",
            "_meta": {"title": "Load DiffSynth Depth ControlNet"},
        }
        g["33"] = {
            "inputs": {
                "model": model_ref,
                "model_patch": ["30", 0],
                "vae": ["5", 0],
                # The ControlNet's image input is the source Enscape render.
                # DiffSynth's Qwen-Image-Depth control net extracts its own
                # depth features internally from this RGB image — no external
                # preprocessor is required (QwenImageDiffsynthControlnet's
                # implementation handles the VAE encoding and the patch
                # conditioning inside the node).
                "image": ["1", 0],
                "strength": float(depth_strength),
            },
            "class_type": "QwenImageDiffsynthControlnet",
            "_meta": {"title": f"Apply DiffSynth Depth ControlNet (strength={depth_strength})"},
        }
        model_ref = ["33", 0]

    # 8/9: TextEncodeQwenImageEditPlus — the edit-model's encode node.
    # image1 carries the source render's pixels (read by Qwen2.5-VL encoder
    # inside the node). TextEncodeQwenImageEditPlus is what makes this an EDIT
    # rather than a regeneration.
    #
    # v7.4: when style_image_name is set, a second LoadImage (node 100) is added
    # and its output is wired to BOTH encode nodes' image2 input. The model
    # treats image1 as "Figure 1" (the source to edit) and image2 as
    # "Figure 2" (the style reference). The positive prompt in that case
    # should be DEFAULT_POSITIVE_STYLE (or equivalent) telling the model to
    # match Figure 2's style onto Figure 1.
    style_image2_ref = None
    if style_image_name:
        g["100"] = {
            "inputs": {"image": style_image_name, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Style Reference (Figure 2)"},
        }
        style_image2_ref = ["100", 0]
    image1_ref = ["1", 0]
    image2_ref = style_image2_ref  # None when v7.3 BASE behavior

    encode_positive_inputs = {
        "prompt": positive,
        "clip": ["4", 0],
        "vae": ["5", 0],
        "image1": image1_ref,
    }
    encode_negative_inputs = {
        "prompt": negative,
        "clip": ["4", 0],
        "vae": ["5", 0],
        "image1": image1_ref,
    }
    if image2_ref is not None:
        encode_positive_inputs["image2"] = image2_ref
        encode_negative_inputs["image2"] = image2_ref

    g["8"] = {
        "inputs": encode_positive_inputs,
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "Qwen Edit Encode (Positive / instruction)"},
    }
    g["9"] = {
        "inputs": encode_negative_inputs,
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
    # NOTE: bf16 weights (~41GB) intentionally NOT staged by default. fp8mixed is
    # the production default; the BF16 sub-agent's 2026-06-18 probe showed
    # fp8mixed is functionally a wash vs BF16 on this model (TR/TL ratio
    # 0.41 vs 0.42, perceptually interchangeable, 15% slower on BF16). The
    # ~$0.30 download + 5 min redeploy is not worth the dormant precision.
    # To re-enable: add the entry below and the matching symlink in
    # TruRenderQwen._setup_model_links(), then redeploy the Modal app.
    #
    #   {"name": "qwen_image_edit_2511_bf16.safetensors",
    #    "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
    #    "filename": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
    #    "dest": "qwen_models/unet"},
    #
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
                print("[TruRender v7] ✓ All essential nodes present (including TextEncodeQwenImageEditPlus)")
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
                print("[TruRender v7]   Run: modal run trurender_qwen_comfyui.py::download_models")
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
            print("[TruRender v7] First real render will be slow (model loading).")

    @modal.method()
    def _render_single(self, image_bytes: bytes, seed: int = 42,
                       target_width: int = None, target_height: int = None,
                       steps: int = 40, cfg: float = 4.0,
                       sampler_name: str = "euler", scheduler: str = "simple",
                       model_sampling_shift: float = 3.1,
                       cfgnorm_strength: float = 1.0,
                       use_fp8: bool = True,
                       use_lightning_lora: bool = False,
                       depth_strength: float = 0.3,
                       output_format: str = "png",
                       positive: str = None,
                       negative: str = None,
                       # v7.4 multi-image style-reference conditioning.
                       # style_image_bytes=None → v7.3 BASE behavior (image1
                       # only). Set to enable the v7.4 path: the bytes get
                       # uploaded as a second LoadImage and wired to image2
                       # on both TextEncodeQwenImageEditPlus instances. The
                       # positive prompt in that case is expected to be
                       # DEFAULT_POSITIVE_STYLE (or compatible).
                       style_image_bytes: bytes = None,
                       style_image_name: str = None) -> tuple:
        """Run a single Qwen-Image-Edit render through ComfyUI.

        Returns (output_image_bytes, elapsed_seconds).

        If target_width/target_height are not provided, dimensions are computed
        from the input image with compute_aspect_preserving_dims() — preserving
        the source aspect ratio at a ~2MP bucket snapped to multiples of 16.
        (2MP default is required to suppress Qwen-Image-Edit's right-border
        hallucination on 16:9 inputs — see compute_aspect_preserving_dims docstring.)

        If positive/negative are not provided, DEFAULT_POSITIVE/DEFAULT_NEGATIVE
        (the v7 hardcoded defaults) are used. These were made into kwargs so the
        local `render` entrypoint can override them for parameter sweeps without
        touching build_workflow internals (which stay at the documented defaults).

        depth_strength (default 0.3 in v7.3) is the optional DiffSynth depth
        ControlNet backstop, validated 2026-06-19. 0.0 = OFF, preserves v7.2
        behavior bit-identically. 0.3 = the noise-reduction sweet spot (-56%
        right-edge TR/TL noise vs v7.2 baseline). Range: -10.0 to 10.0 (matches
        QwenImageDiffsynthControlnet node's range). Practical range 0.0-1.0.

        v7.4: style_image_bytes (raw image bytes) and style_image_name
        (pre-uploaded filename on the ComfyUI input dir, optional) — if either
        is set, the workflow is built with a second LoadImage and image2 wiring.
        style_image_bytes is preferred; style_image_name is for callers that
        pre-staged the file.
        """
        from PIL import Image as PILImage
        start = time.time()
        client_id = uuid.uuid4().hex

        # Compute target dimensions preserving aspect ratio
        if target_width is None or target_height is None:
            target_width, target_height = compute_aspect_preserving_dims(image_bytes)

        # Resolve prompts (None → v7 default)
        positive_prompt = positive if positive is not None else DEFAULT_POSITIVE
        negative_prompt = negative if negative is not None else DEFAULT_NEGATIVE

        # Sanity-check input
        src_img = PILImage.open(io.BytesIO(image_bytes))
        src_w, src_h = src_img.size
        fp8_str = "fp8" if use_fp8 else "bf16"
        lora_str = " | lora@1.0" if use_lightning_lora else ""
        depth_str = f" | depth@{depth_strength}" if depth_strength > 0 else ""
        print(f"[TruRender v7] Input: {src_w}x{src_h} → Output: {target_width}x{target_height} | "
              f"steps={steps} cfg={cfg} shift={model_sampling_shift} cfgnorm={cfgnorm_strength} "
              f"{fp8_str}{lora_str}{depth_str} | seed={seed}")
        print(f"[TruRender v7] positive prompt len: {len(positive_prompt)} chars "
              f"({'custom' if positive is not None else 'DEFAULT'})")

        # Upload input image
        filename = f"trurender_v7_input_{client_id[:8]}.png"
        uploaded_name = self._upload_image(image_bytes, filename)

        # v7.4: optionally upload + wire a style reference image.
        # bytes win over name — if bytes are provided, we always upload them
        # and ignore the caller-supplied name. This is because ComfyUI's
        # LoadImage node can only load files from its input dir; a name
        # pointing at a non-existent file is rejected with HTTP 400.
        if style_image_bytes is not None:
            style_filename = f"trurender_v7_style_{client_id[:8]}.jpg"
            uploaded_style_name = self._upload_image(style_image_bytes, style_filename)
            print(f"[TruRender v7.4] style image uploaded: {uploaded_style_name} "
                  f"({len(style_image_bytes)} bytes)")
        else:
            uploaded_style_name = style_image_name  # may be None (v7.3 BASE)

        # Build and submit workflow
        workflow = build_workflow(
            image_name=uploaded_name,
            seed=seed,
            target_width=target_width,
            target_height=target_height,
            positive=positive_prompt,
            negative=negative_prompt,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            model_sampling_shift=model_sampling_shift,
            cfgnorm_strength=cfgnorm_strength,
            use_fp8=use_fp8,
            use_lightning_lora=use_lightning_lora,
            depth_strength=depth_strength,
            style_image_name=uploaded_style_name,
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
    seed: int = DEFAULT_SEED,
    steps: int = 40,
    cfg: float = 4.0,
    use_fp8: bool = True,
    use_lightning_lora: bool = False,
    depth_strength: float = 0.3,
    target_width: int = None,
    target_height: int = None,
    positive: str = None,
    negative: str = None,
    # v7.4: style-reference conditioning.
    # style_image_name=None → v7.3 BASE (backward compatible).
    # Set to a filename from the unified 19 (or pass --style-image-b64 to
    # upload ad-hoc). space_type is used by the smart selector when
    # style_image_name is None. Defaults to "kitchen" if both unset.
    style_image_name: str = None,
    space_type: str = None,
    style_image_b64: str = None,
):
    """Render an Enscape image with the v7 (Qwen-Image-Edit) pipeline.

    Usage:
        modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png
        modal run trurender_qwen_comfyui.py::render --input-path /path/to/enscape.png --output-path out.png --seed 1234
        modal run trurender_qwen_comfyui.py::render --input-path ... --positive "..." --cfg 3.5 --seed 42
        modal run trurender_qwen_comfyui.py::render --input-path ... --depth-strength 0.0   # disable v7.3 backstop

        # v7.4: smart selector picks the best CTAI/Catherine ref for the room
        modal run trurender_qwen_comfyui.py::render --input-path ... --space-type kitchen

        # v7.4: explicit style ref
        modal run trurender_qwen_comfyui.py::render --input-path ... --style-image-name Charlestown-12-scaled.jpg

        # v7.4: ad-hoc style image (base64 in the request)
        modal run trurender_qwen_comfyui.py::render --input-path ... --style-image-b64 "$(base64 -i ref.jpg)"

    By default, the v7 hardcoded DEFAULT_POSITIVE/DEFAULT_NEGATIVE are used. Pass
    --positive / --negative to override (used by parameter sweeps). When passing
    --positive from a shell, use single quotes; the CLI forwards verbatim.
    When --style-image-name is set, --positive is auto-replaced with
    DEFAULT_POSITIVE_STYLE unless the caller also passes --positive (caller
    wins; the v7.4 trigger is what we want for raw style transfer but a custom
    prompt is still allowed).

    depth_strength (default 0.3 in v7.3) is the DiffSynth depth ControlNet
    backstop. Pass --depth-strength 0.0 to disable (preserves v7.2 behavior).
    """
    import base64 as _b64
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

    # v7.4: resolve style image
    style_image_bytes = None
    if style_image_b64:
        style_image_bytes = _b64.b64decode(style_image_b64)
        print(f"[TruRender v7.4] style-image-b64: decoded {len(style_image_bytes)} bytes")

    if style_image_name is None and style_image_bytes is None and space_type is None:
        # Backward compat: no style ref requested
        effective_space_type = None
        effective_style_name = None
    else:
        effective_space_type = space_type or "kitchen"
        if style_image_name is None and style_image_bytes is None:
            # Run the smart selector locally to resolve a filename
            try:
                pick = select_style_ref(effective_space_type, override_filename=None)
                effective_style_name = pick["filename"]
                print(f"[TruRender v7.4] smart selector ({effective_space_type}) → "
                      f"{pick['filename']} (source={pick['source']}, score={pick['score']:.3f})")
                print(f"[TruRender v7.4] why: {pick['why']}")
                # If the picked ref has a local path (canonical set), read
                # the bytes and pass them — ComfyUI's input dir doesn't
                # have these files staged, so we must upload them.
                if pick.get("local_path") and os.path.exists(pick["local_path"]):
                    with open(pick["local_path"], "rb") as _f:
                        style_image_bytes = _f.read()
                    print(f"[TruRender v7.4] read {len(style_image_bytes)} bytes from "
                          f"{pick['local_path']}")
                else:
                    # Catherine pick or missing local file. Surface a
                    # clear warning.
                    print(f"[TruRender v7.4] WARNING: picked ref {pick['filename']!r} "
                          f"has no local_path; cannot upload to ComfyUI without "
                          f"volume-staged copy. (selector source={pick['source']})")
                    effective_style_name = None
            except Exception as e:
                print(f"[TruRender v7.4] selector error: {e}")
                effective_style_name = None
        else:
            effective_style_name = style_image_name

    # Auto-swap positive to v7.4 trigger when style ref is in play and caller
    # didn't supply a custom positive.
    if effective_style_name is not None and positive is None:
        positive = DEFAULT_POSITIVE_STYLE
        print(f"[TruRender v7.4] positive auto-set to DEFAULT_POSITIVE_STYLE")

    # Call the remote class
    output_bytes, elapsed = TruRenderQwen()._render_single.remote(
        image_bytes, seed=seed, target_width=target_width, target_height=target_height,
        steps=steps, cfg=cfg, use_fp8=use_fp8, use_lightning_lora=use_lightning_lora,
        depth_strength=depth_strength,
        positive=positive, negative=negative,
        style_image_bytes=style_image_bytes,
        style_image_name=effective_style_name,
    )

    output_path.write_bytes(output_bytes)
    print(f"[TruRender v7] ✓ Rendered in {elapsed:.1f}s → {output_path} ({len(output_bytes)} bytes)")


@app.local_entrypoint(name="render_options")
def render_options(
    input_path: str,
    output_dir: str = None,
    num_options: int = 5,
    anchor_seed: int = DEFAULT_SEED,
    steps: int = 40,
    cfg: float = 4.0,
    use_fp8: bool = True,
    use_lightning_lora: bool = False,
    depth_strength: float = 0.3,
    positive: str = None,
    negative: str = None,
    # v7.4: style-reference conditioning (same as render)
    style_image_name: str = None,
    space_type: str = None,
    style_image_b64: str = None,
):
    """Render N seed-variation options in ONE warm Modal container (production entrypoint).

    This is the end-user "give me options" path. Generates `num_options`
    independent renders of the same input image, each at a different seed, and
    writes them to `output_dir/{input_stem}_opt_{i}.png` (1-indexed).

    The first option always uses `anchor_seed` (default DEFAULT_SEED = 1234, the
    v7.2 chosen hero seed). Remaining options use random seeds sampled with
    `random.randint(0, 2**31 - 1)`. The seed list is printed at startup so it's
    reproducible if the user re-runs with the same inputs.

    All options are rendered sequentially within a single warm container
    (cold-start amortized across N renders — same pattern as `sweep()`). Cost
    is ~$0.55 + $0.03 per option at v7.1 2MP on A100-80GB.

    Usage:
        modal run trurender_qwen_comfyui.py::render_options \\
            --input-path /path/to/enscape.png \\
            --output-dir /path/to/outputs \\
            --num-options 5

    Add `--positive` or `--negative` to override the default prompts for
    A/B-style prompt-vs-prompt comparisons; otherwise the v7 hardcoded
    DEFAULT_POSITIVE/DEFAULT_NEGATIVE are used (the 6-item trimmed negative
    is the v7.2 default; the 100-word blocklist has been retired).

    Pass --depth-strength 0.0 to disable the v7.3 backstop (preserves v7.2
    behavior bit-identically). Default 0.3 applies the validated
    noise-reduction ControlNet.

    Returns the list of written output paths.
    """
    import json as _json
    import random as _random
    from datetime import datetime as _dt

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir is None:
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/tmp/trurender_v7_options_{ts}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = input_path.read_bytes()
    print(f"[TruRender v7 render_options] Local: read {len(image_bytes)} bytes from {input_path}")
    print(f"[TruRender v7 render_options] Output dir: {output_dir}")
    print(f"[TruRender v7 render_options] num_options={num_options} anchor_seed={anchor_seed}")

    # Build the seed list: anchor first, then num_options-1 random.
    seeds = [anchor_seed] + [_random.randint(0, 2**31 - 1) for _ in range(num_options - 1)]
    print(f"[TruRender v7 render_options] seeds: {seeds}")

    service = TruRenderQwen()
    stem = input_path.stem
    written = []
    started = time.time()

    # v7.4: resolve style image once for the whole options run
    import base64 as _b64
    style_image_bytes = None
    if style_image_b64:
        style_image_bytes = _b64.b64decode(style_image_b64)
    if style_image_name is None and style_image_bytes is None and space_type is None:
        effective_space_type = None
        effective_style_name = None
    else:
        effective_space_type = space_type or "kitchen"
        if style_image_name is None and style_image_bytes is None:
            try:
                pick = select_style_ref(effective_space_type, override_filename=None)
                effective_style_name = pick["filename"]
                print(f"[TruRender v7.4] smart selector ({effective_space_type}) → "
                      f"{pick['filename']} (source={pick['source']}, score={pick['score']:.3f})")
                # If picked ref has a local_path, read the bytes for upload.
                if pick.get("local_path") and os.path.exists(pick["local_path"]):
                    with open(pick["local_path"], "rb") as _f:
                        style_image_bytes = _f.read()
                    print(f"[TruRender v7.4] read {len(style_image_bytes)} bytes from "
                          f"{pick['local_path']}")
                else:
                    print(f"[TruRender v7.4] WARNING: picked ref {pick['filename']!r} "
                          f"has no local_path; cannot upload to ComfyUI without "
                          f"volume-staged copy. (selector source={pick['source']})")
                    effective_style_name = None
            except Exception as e:
                print(f"[TruRender v7.4] selector error: {e}")
                effective_style_name = None
        else:
            effective_style_name = style_image_name
    if effective_style_name is not None and positive is None:
        positive = DEFAULT_POSITIVE_STYLE

    for i, seed in enumerate(seeds, start=1):
        out_name = f"{stem}_opt_{i:02d}_seed{seed}.png"
        out_path = output_dir / out_name
        cell_start = time.time()
        try:
            output_bytes, render_elapsed = service._render_single.remote(
                image_bytes,
                seed=seed,
                steps=steps,
                cfg=cfg,
                use_fp8=use_fp8,
                use_lightning_lora=use_lightning_lora,
                depth_strength=depth_strength,
                positive=positive,
                negative=negative,
                style_image_bytes=style_image_bytes,
                style_image_name=effective_style_name,
            )
            out_path.write_bytes(output_bytes)
            wall = time.time() - cell_start
            print(f"[TruRender v7 render_options] {i}/{num_options} seed={seed} "
                  f"render={render_elapsed:.1f}s wall={wall:.1f}s -> {out_path} "
                  f"({len(output_bytes)} bytes)")
            written.append(str(out_path))
        except Exception as e:
            print(f"[TruRender v7 render_options] {i}/{num_options} seed={seed} FAILED: {e}")
            raise

    total = time.time() - started
    # Write a manifest of the seed list + paths so the run is reproducible
    manifest = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_options": num_options,
        "anchor_seed": anchor_seed,
        "seeds": seeds,
        "outputs": written,
        "elapsed_s": total,
    }
    manifest_path = output_dir / f"{stem}_options_manifest.json"
    manifest_path.write_text(_json.dumps(manifest, indent=2))
    print(f"[TruRender v7 render_options] ✓ {num_options} options in {total:.1f}s "
          f"({total / num_options:.1f}s/option avg) -> {output_dir}")
    print(f"[TruRender v7 render_options] manifest: {manifest_path}")
    return written


@app.local_entrypoint(name="sweep")
def sweep(
    spec_path: str,
):
    """Run a parameter sweep in a SINGLE warm Modal container.

    Each `modal run script.py::entrypoint` spawns an ephemeral container that
    tears down at the end. For sweeps with many cells, cold-start + container
    spinup overhead per cell is wasteful (each render would include ~90s of
    container + ComfyUI startup). This entrypoint reads a JSON spec and calls
    `_render_single` N times within ONE container, keeping the model warm in
    VRAM between cells.

    Spec JSON shape:
      {
        "input_path": "/abs/path/to/enscape.png",
        "output_dir": "/abs/path/to/outputs",
        "common": {"seed": 42, "steps": 40},   # applied to every cell
        "cells": [
          {"code": "A7", "output": "A7.png", "positive": "...", "cfg": 3.5},
          ...
        ]
      }

    Each cell is rendered in order. Results are written to
    `{output_dir}/{cell.output}` as PNG bytes returned from `_render_single`.
    A `results.json` is written alongside with per-cell timings.
    """
    import json as _json
    from datetime import datetime as _dt

    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"spec file not found: {spec_path}")

    with open(spec_path) as f:
        spec = _json.load(f)

    input_path = Path(spec["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"input image not found: {input_path}")

    image_bytes = input_path.read_bytes()
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = spec["cells"]
    common = spec.get("common", {})
    seed = common.get("seed", 42)
    steps = common.get("steps", 40)
    sampler_name = common.get("sampler_name", "euler")
    scheduler = common.get("scheduler", "simple")
    use_fp8 = common.get("use_fp8", True)
    use_lightning_lora = common.get("use_lightning_lora", False)
    common_depth_strength = common.get("depth_strength", 0.3)
    # v7.4: common-block style-ref params (per-cell overrides still possible)
    common_style_image_name = common.get("style_image_name", None)
    common_space_type = common.get("space_type", None)
    common_style_image_b64 = common.get("style_image_b64", None)

    print(f"[TruRender v7 sweep] input: {input_path} ({len(image_bytes)} bytes)")
    print(f"[TruRender v7 sweep] output_dir: {output_dir}")
    print(f"[TruRender v7 sweep] cells: {len(cells)}")
    print(f"[TruRender v7 sweep] common: seed={seed} steps={steps} "
          f"sampler={sampler_name}/{scheduler} fp8={use_fp8} lightning={use_lightning_lora} "
          f"depth_strength={common_depth_strength}")
    if common_style_image_name or common_space_type or common_style_image_b64:
        print(f"[TruRender v7 sweep] v7.4 common: style_image_name={common_style_image_name} "
              f"space_type={common_space_type} style_image_b64={'<set>' if common_style_image_b64 else None}")

    # Instantiate the class ONCE — keeps the warm ComfyUI server alive across cells
    service = TruRenderQwen()

    # v7.4: decode common-block style_image_b64 once (used for cells that
    # don't override it)
    import base64 as _b64
    common_style_image_bytes = None
    if common_style_image_b64:
        common_style_image_bytes = _b64.b64decode(common_style_image_b64)
        print(f"[TruRender v7 sweep] common style_image_b64: {len(common_style_image_bytes)} bytes")

    sweep_start = time.time()
    results = []
    for i, cell in enumerate(cells, 1):
        code = cell["code"]
        out_name = cell.get("output", f"{code}.png")
        out_path = output_dir / out_name
        positive = cell["positive"]
        # Per-cell overrides — fall back to common block, then hard defaults.
        # Fixes the BF16 sub-agent's reported bug: `cfg = cell.get("cfg", 4.0)`
        # did not honor spec.common.cfg. Apply the same fallback pattern to every
        # common-kwarg pass.
        cfg = cell.get("cfg", common.get("cfg", 4.0))
        negative = cell.get("negative", None)
        # Per-cell seed override (falls back to common.seed). Fixes seed-variation
        # trial pattern (S-1, S-7, S-100, S-1234) where each cell needs its own
        # seed within a single warm container.
        cell_seed = cell.get("seed", common.get("seed", 42))
        cell_steps = cell.get("steps", common.get("steps", 40))
        cell_sampler_name = cell.get("sampler_name", common.get("sampler_name", "euler"))
        cell_scheduler = cell.get("scheduler", common.get("scheduler", "simple"))
        cell_use_fp8 = cell.get("use_fp8", common.get("use_fp8", True))
        cell_model_sampling_shift = cell.get("model_sampling_shift",
                                             common.get("model_sampling_shift", 3.1))
        cell_cfgnorm_strength = cell.get("cfgnorm_strength",
                                          common.get("cfgnorm_strength", 1.0))
        cell_depth_strength = cell.get("depth_strength",
                                        common.get("depth_strength", 0.3))
        # v7.4 style-ref per-cell overrides
        cell_style_image_name = cell.get("style_image_name", common_style_image_name)
        cell_space_type = cell.get("space_type", common_space_type)
        cell_style_image_b64 = cell.get("style_image_b64", common_style_image_b64)
        cell_style_image_bytes = None
        if cell_style_image_b64:
            cell_style_image_bytes = _b64.b64decode(cell_style_image_b64)
        # If a name was passed explicitly, try to resolve local bytes for it
        # (canonical refs have a local_path; Catherine picks don't and need
        # volume staging). Run the selector with override to get metadata.
        if cell_style_image_name and cell_style_image_bytes is None:
            try:
                _pick = select_style_ref(
                    cell.get("space_type", common_space_type) or "kitchen",
                    override_filename=cell_style_image_name,
                )
                if _pick.get("local_path") and os.path.exists(_pick["local_path"]):
                    with open(_pick["local_path"], "rb") as _f:
                        cell_style_image_bytes = _f.read()
                    print(f"[TruRender v7.4] cell {code}: read {len(cell_style_image_bytes)} bytes "
                          f"from {_pick['local_path']}")
            except Exception as e:
                print(f"[TruRender v7.4] cell {code}: override-name resolution error: {e}")

        # v7.4: resolve smart selector if no explicit name + no b64
        if (cell_style_image_name is None
                and cell_style_image_bytes is None
                and (cell_space_type is not None or common_space_type is not None)):
            effective_space_type = cell_space_type or "kitchen"
            try:
                pick = select_style_ref(effective_space_type, override_filename=None)
                cell_style_image_name = pick["filename"]
                # If the picked ref has a local path (canonical set), read the
                # bytes and pass them — ComfyUI's input dir doesn't have these
                # files staged, and the LoadImage node can't fetch from
                # /Users/...; the bytes must be uploaded by _render_single.
                if pick.get("local_path") and os.path.exists(pick["local_path"]):
                    with open(pick["local_path"], "rb") as _f:
                        cell_style_image_bytes = _f.read()
                    if i == 1:
                        print(f"[TruRender v7.4] smart selector ({effective_space_type}) → "
                              f"{pick['filename']} (source={pick['source']}, score={pick['score']:.3f})")
                        print(f"[TruRender v7.4] why: {pick['why']}")
                        print(f"[TruRender v7.4] read {len(cell_style_image_bytes)} bytes from "
                              f"{pick['local_path']}")
                else:
                    # Catherine pick (no local_path) — can't upload from
                    # local; would need volume-staged copy. Surface as a
                    # clear error rather than silently failing.
                    if i == 1:
                        print(f"[TruRender v7.4] WARNING: picked ref {pick['filename']!r} "
                              f"has no local_path; cannot upload to ComfyUI without "
                              f"volume-staged copy. (selector source={pick['source']})")
                    cell_style_image_name = None
            except Exception as e:
                print(f"[TruRender v7.4] selector error: {e}")
                cell_style_image_name = None

        # Auto-swap positive to v7.4 trigger when style ref is in play
        # and the cell's positive is the v7 default (i.e. not custom).
        v7_default_positive = DEFAULT_POSITIVE
        is_cell_positive_default = (positive == v7_default_positive)
        if cell_style_image_name is not None and is_cell_positive_default:
            positive = DEFAULT_POSITIVE_STYLE
            if i == 1:
                print(f"[TruRender v7.4] positive auto-swapped to DEFAULT_POSITIVE_STYLE")

        print(f"\n=== cell {i}/{len(cells)}: {code} (seed={cell_seed} cfg={cfg} "
              f"steps={cell_steps} {cell_sampler_name}/{cell_scheduler} "
              f"fp8={cell_use_fp8} shift={cell_model_sampling_shift} "
              f"cfgnorm={cell_cfgnorm_strength} depth={cell_depth_strength}"
              f"{' style=' + cell_style_image_name if cell_style_image_name else ''}"
              f"{' style_b64' if cell_style_image_bytes else ''}"
              f") ===")
        cell_start = time.time()
        try:
            output_bytes, render_elapsed = service._render_single.remote(
                image_bytes,
                seed=cell_seed,
                steps=cell_steps,
                cfg=cfg,
                sampler_name=cell_sampler_name,
                scheduler=cell_scheduler,
                model_sampling_shift=cell_model_sampling_shift,
                cfgnorm_strength=cell_cfgnorm_strength,
                use_fp8=cell_use_fp8,
                use_lightning_lora=use_lightning_lora,
                depth_strength=cell_depth_strength,
                positive=positive,
                negative=negative,
                style_image_bytes=cell_style_image_bytes,
                style_image_name=cell_style_image_name,
            )
            out_path.write_bytes(output_bytes)
            wall = time.time() - cell_start
            ok = True
            err = None
            print(f"[TruRender v7 sweep] ✓ {code}: render={render_elapsed:.1f}s wall={wall:.1f}s "
                  f"→ {out_path} ({len(output_bytes)} bytes)")
        except Exception as e:
            wall = time.time() - cell_start
            ok = False
            err = repr(e)
            print(f"[TruRender v7 sweep] ✗ {code}: FAILED after {wall:.1f}s — {err}")

        results.append({
            "code": code,
            "output": out_name,
            "seed": cell_seed,
            "cfg": cfg,
            "steps": cell_steps,
            "sampler_name": cell_sampler_name,
            "scheduler": cell_scheduler,
            "model_sampling_shift": cell_model_sampling_shift,
            "cfgnorm_strength": cell_cfgnorm_strength,
            "use_fp8": cell_use_fp8,
            "depth_strength": cell_depth_strength,
            "style_image_name": cell_style_image_name,
            "space_type": cell_space_type,
            "ok": ok,
            "wall_s": wall,
            "render_s": (render_elapsed if ok else None),
            "size_bytes": (out_path.stat().st_size if out_path.exists() else 0),
            "error": err,
        })

    total_sweep_s = time.time() - sweep_start
    ok_count = sum(1 for r in results if r["ok"])

    manifest = {
        "spec_path": str(spec_path),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "common": {"seed": seed, "steps": steps, "sampler_name": sampler_name,
                   "scheduler": scheduler, "use_fp8": use_fp8,
                   "use_lightning_lora": use_lightning_lora,
                   "depth_strength": common_depth_strength,
                   # v7.4
                   "style_image_name": common_style_image_name,
                   "space_type": common_space_type,
                   "style_image_b64": ("<set>" if common_style_image_b64 else None)},
        "finished_at": _dt.now().astimezone().isoformat(timespec="seconds"),
        "total_sweep_s": total_sweep_s,
        "ok_count": ok_count,
        "fail_count": len(results) - ok_count,
        "results": results,
    }
    manifest_path = output_dir / "results_manifest.json"
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)

    print(f"\n[TruRender v7 sweep] DONE — {ok_count}/{len(results)} ok, "
          f"{total_sweep_s:.1f}s total")
    print(f"[TruRender v7 sweep] manifest: {manifest_path}")


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

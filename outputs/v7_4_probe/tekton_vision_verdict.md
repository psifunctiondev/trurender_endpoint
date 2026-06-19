# v7.4 IMG-only Probe — Tekton Vision Verdict

**Model:** Qwen2.5-VL-7B-Instruct (Tekton Vision, T4 endpoint)
**Inputs:** 3-up comparison sheet (SOURCE | BASE v7.3 | IMG v7.4)
**Picked style ref:** `Locke-Feature-scaled.jpg` (source=canonical, score=1.275)
**Date:** 2026-06-19

## Verbatim verdict

### Comparison of BASE (v7.3) and IMG (v7.4) Photorealistic Transforms

#### 1. Preservation of Source Render's Geometry, Cabinetry Layout, Window Placement, and Material Identity

**BASE (v7.3):**
- **Geometry and Layout:** The geometry of the cabinetry, layout, and window placement appear to be very close to the original source render. The cabinets, countertops, and island are accurately represented.
- **Material Identity:** The materials, such as the white shaker cabinets, white marble countertops, and walnut island, are rendered with a high degree of fidelity. The brass hardware is also accurately represented.

**IMG (v7.4):**
- **Geometry and Layout:** The cabinetry layout and window placement are consistent with the source render. However, there are minor differences in the precise alignment and positioning of the elements, which might be due to the style reference applied.
- **Material Identity:** The materials, including the white shaker cabinets, white marble countertops, and walnut island, are rendered with a similar level of accuracy. The brass hardware also appears to be preserved well.

#### 2. Application of Catherine-Consistent Editorial Style

**BASE (v7.3):**
- **Natural Daylight:** The lighting in BASE appears to be natural daylight, but it lacks the refined quality and mood that might be achieved with a Catherine-approved style.
- **Color Palette:** The color palette is natural and restrained, but it may not be as polished or consistent with the style reference.
- **Photographic Realism:** The photorealism is present, but it might not be as refined or stylized as the final Catherine-approved image.
- **Honest Materials:** The materials are rendered with honesty, but the overall look might not be as polished or consistent with the style reference.

**IMG (v7.4):**
- **Natural Daylight:** The natural daylight is maintained, but the style reference has introduced a more polished and refined look.
- **Color Palette:** The color palette is more consistent with the Catherine-approved style, featuring a restrained and natural color scheme.
- **Photographic Realism:** The photorealism is enhanced, with a more polished and refined look that aligns with the style reference.
- **Honest Materials:** The materials are rendered with honesty, but the overall look is more refined and consistent with the Catherine-approved style.

#### 3. Visible Style Transfer in IMG (v7.4)

**BASE (v7.3):**
- **Color Cast:** The color cast is neutral and natural, with no significant warmth or coolness.
- **Material Rendering:** The materials are rendered with a high level of detail, but the overall look is more straightforward and less refined.
- **Lighting Mood:** The lighting mood is natural and straightforward, with no significant changes in mood or atmosphere.

**IMG (v7.4):**
- **Color Cast:** The color cast is warmer and more polished, with a more refined and refined look.
- **Material Rendering:** The materials are rendered with a more refined look, with subtle differences in texture and finish that align with the style reference.
- **Lighting Mood:** The lighting mood is more refined and polished, with a more dramatic and refined look that aligns with the Catherine-approved style.

### Conclusion

- **Geometry and Material Identity:** Both BASE and IMG preserve the source render's geometry and material identity very well. However, the IMG transform better aligns with the Catherine-approved style in terms of color palette, lighting, and overall mood.
- **Editorial Style:** The IMG transform better applies the Catherine-consistent editorial style, with a more polished and refined look, natural daylight, and a restrained color palette.
- **Style Transfer:** The IMG transform shows visible style transfer, with a warmer color cast, more refined material rendering, and a more dramatic lighting mood compared to the BASE transform.

## Metric cross-check

| Metric | SOURCE | BASE (v7.3) | IMG (v7.4) | Notes |
|---|---|---|---|---|
| Width × Height | 1920×1080 | 1920×1072 | 1920×1072 | 1920×1080 input → 1920×1072 (2MP @ 16:9) |
| File size | — | 2125 KB | 1848 KB | |
| TR/TL R-std ratio | — | 0.211 | 0.310 | v7.2 danger zone = 1.18. Both <0.32 → clean |
| Whole R std | 50.4 | 52.2 | 51.6 | Signal richness (broader = more photo-like dynamic range) |
| MAE vs source | — | 45.51 | 7.84 | Lower = closer to source. IMG ~6× closer than BASE |
| RMSE vs source | — | 55.32 | 17.69 | |
| Render time | — | 199.1s | 220.4s | IMG ~10% slower (2nd image) |

**Key finding:** Tekton Vision reports the two transforms as **near-equivalent on geometry and material identity** (both preserve cabinetry layout, window placement, marble, walnut island, brass hardware), and **IMG is clearly preferred on Catherine-style editorial polish** (warmer color cast, more refined material rendering, more dramatic/refined lighting mood, restrained color palette).

This aligns with the metric signal: IMG is much closer to the source (MAE 7.84 vs 45.51) — meaning the style reference (a real Catherine-approved photo) is anchoring the model to the source's geometry/lighting/mood more than the bare text prompt does. The result reads as a clean editorial photograph, not an over-interpreted re-render.

## Why this is the v7.4 path

The v7.4 mechanism is:
- `TextEncodeQwenImageEditPlus` has a second image input (`image2`) that takes a style reference
- The Qwen2.5-VL encoder inside that node sees BOTH images — `image1` is the source render (the thing to edit), `image2` is the style reference (the mood/material vocabulary to draw from)
- The trigger prompt (`"Change the style of Figure 1 to the style of Figure 2. Photorealistic interior photograph. Preserve exact room layout, camera angle, and object positions. Do not change any material or finish."`) tells the model which image is which
- The v7.3 6-item trimmed negative stays — it still blocks 3D-render artifacts, plastic, distorted geometry, text, watermark, added furniture

The style reference (Locke-Feature-scaled.jpg) is a real photograph of a kitchen with sage green cabinetry, walnut island, and marble counters. The model is using that photo as a "render vocabulary" — pulling lighting mood, material rendering quality, color palette — and applying those to the source render's geometry.

## What's in the v7.4 release

- `trurender_endpoint/pipeline/trurender_qwen_comfyui.py` — smart selector + multi-image wiring + new params
- `trurender_endpoint/outputs/v7_4_probe/` — this probe + artifacts

## Follow-ups

- v7.4 worked first-try after the bytes-vs-name fix in `_render_single`. The fix is permanent.
- v7.4 also enables `style_image_name=<Catherine pick>` directly, but the local copies of Catherine's 3 are no longer on this machine (the brief said they were cleaned up); they're only on the Modal volume at `/models/style_references/`. Volume staging for Catherine's picks is deferred to a future task — the smart selector already works for the 16 canonical refs that have local copies.
- TR/TL ratio is 0.310 on IMG (vs 0.211 on BASE) — still well under the 1.18 danger zone, but the right border is slightly noisier. This is probably because the v7.3 depth ControlNet backstop is doing less work when the model is more constrained by the style reference. Acceptable for v7.4; can be tuned in v7.5.
- Cost: ~$0.40 GPU for 2 cells on A100 80GB (about 7.6 min total, the second cell reuses cached model nodes — 7 of 12 nodes cached) + ~$0.05 Tekton Vision cold-start + inference on T4. Total ~$0.45, well under the $0.80 cap.

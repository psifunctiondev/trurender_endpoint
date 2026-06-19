# Tier-B Probe — Perceptual Notes

**Source input:** `inputs/enscape_input.png` — modern kitchen/dining interior, marble-topped island, dark dining chairs, pendant lights, white cabinetry, large windows with cityscape view.

**Subject of comparison:** Each cell vs BASE (the v7.1 control).

## BASE (control)
Clean photorealistic render. Neutral warm lighting, balanced exposure, good detail. TR std=7.04, TL std=20.77, ratio=0.339 (TR less busy than TL — consistent asymmetry signature across most cells).

## MS-2.5 — model_sampling_shift=2.5
Indistinguishable from BASE. The shift axis barely moves the needle within this range; ratio 0.416 sits comfortably in the healthy band.

## MS-4.0 — model_sampling_shift=4.0
Indistinguishable from BASE. Same scene, same exposure, same detail. ratio 0.369 confirms symmetry holds.

## MS-5.0 — model_sampling_shift=5.0
Indistinguishable from BASE. ratio 0.348. **Conclusion: the model_sampling_shift axis is essentially a no-op for this prompt at the 2.5–5.0 sweep range.** Worth a wider sweep (1.0–8.0) before drawing final conclusions.

## CN-0.5 — cfgnorm_strength=0.5
**Catastrophic degradation.** Output is an essentially flat olive-green field with no recognizable scene structure. Both TR std and TL std collapse to ~3.4 (from BASE's 7/21). cfgnorm_strength=0.5 is destroying the render — likely under-conditioning the CFG guidance. **Do not use this value.** ratio 1.005 only because both sides collapsed equally.

## CN-1.5 — cfgnorm_strength=1.5
**Catastrophic degradation in the opposite direction.** Blown-out white frame, only the dark chairs and island remain as silhouettes against overexposed background, with a red wash in the lower-left. TR/TL std both near zero (0.07/0.05). ratio 1.291 is technically "near 1.0" but both are collapsed. cfgnorm_strength=1.5 is destroying the render — over-conditioning is saturating the latent. **Do not use this value.** File is also anomalously small at 373 KB.

## AS-uni — sampler=uni_pc, scheduler=simple
Indistinguishable from BASE. Visually identical. ratio 0.387. uni_pc is a safe swap that doesn't move the look at this resolution/prompt.

## AS-sde — sampler=dpmpp_sde, scheduler=karras
**Same scene, but with a strong yellow/sepia color cast and slightly hazy/washed look.** Recognizably the same architectural interior but the warm tint is unmistakable. Reduced contrast (TR std 3.81, TL std 13.16 — both lower than BASE). SDE/Karras shifts the color palette noticeably; render also took ~2× as long (312s vs ~156s) — expected for SDE. ratio 0.290 is the lowest of the sweep.

## NP-trim — trimmed 6-item negative prompt
Indistinguishable from BASE. Actually marginally **higher** contrast than BASE (TL std=24.42, highest in the sweep). This validates the hypothesis that the original 100-word blocklist was overkill — the 6 highest-signal negatives (3D render, plastic surfaces, distorted geometry, text, watermark, added furniture) carry the full load. ratio 0.355.

## Summary axes

| Axis | Verdict |
|------|---------|
| model_sampling_shift (2.5 / 4.0 / 5.0) | **No-op** within this range. BASE already at the sweet spot. |
| cfgnorm_strength (0.5 / 1.5) | **Both break the render.** Stick with the default 1.0. |
| sampler swap (uni_pc vs dpmpp_sde/karras) | uni_pc is safe. dpmpp_sde/karras adds a yellow cast + 2× cost. Avoid unless that look is wanted. |
| trimmed negative prompt | **Validates simplification.** 6-item negative matches or exceeds the verbose one. |

## What this tells us for v7.2

- BASE settings are validated; nothing in the sweep is better than BASE at the default params.
- The cfgnorm_strength axis is a **trap**: both off-default values are broken, so this axis should probably be locked at 1.0 unless a future calibration wants to widen it (e.g. CN-0.8, CN-1.2).
- model_sampling_shift deserves a wider probe before being locked.
- NP-trim confirms the prompt-cull is safe; can ship to production.

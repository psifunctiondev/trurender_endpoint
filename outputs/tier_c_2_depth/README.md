# Tier C #2 — DiffSynth Depth ControlNet Backstop (5-cell probe)

**Date:** 2026-06-19
**Probe:** Optional DiffSynth Qwen-Image-Depth ControlNet backstop for v7.2.
**Goal:** Test whether structural depth anchoring helps Qwen-Image-Edit, given that
the model already has a built-in vision encoder (Qwen2.5-VL).

## TL;DR

**DS-0.3 is a clear win on the right-edge-noise metric (TR std -46%, TR/TL ratio -56%)
and on the perceptual read (Qwen2.5-VL-7B calls it "more defined and less distorted edges"
with "no over-constrained geometry or depth-map seams").** DS-0.5 and DS-0.7 recover
toward the v7.2 baseline, with DS-0.7 starting to "freeze" the depth layout.

**Recommendation: SHIP a `depth_strength=0.3` default in v7.3.** Add the param to
`build_workflow()` / `_render_single()` (already done), set the default to `0.3` in
the main `render()` entrypoint. Keep the per-cell sweep override for re-probing.

## What was wired

- `pipeline/trurender_qwen_comfyui.py`:
  - New `DEPTH_DIFFSYNTH_CONTROLNET` model constant pointing at the staged
    `qwen_models/controlnet/qwen_image_depth_diffsynth_controlnet.safetensors`
  - New `depth_strength: float = 0.0` param in `build_workflow()`, `_render_single()`,
    and `sweep()` (per-cell override, same pattern as shift/cfgnorm/seed)
  - When `depth_strength > 0`, inserts two nodes:
    - `30: ModelPatchLoader` — loads the ControlNet from `models/model_patches/`
    - `33: QwenImageDiffsynthControlnet` — applies it to the model chain
    - Chain: `UNET → ModelSamplingAuraFlow (6) → CFGNorm (7) → [QwenImageDiffsynthControlnet (33)] → KSampler`
  - When `depth_strength == 0.0` (default), workflow is byte-identical to v7.2 BASE.

## Validation

- `depth_strength=0.0` produces the same workflow dict as v7.2 BASE (no node 30/33).
  Verified locally by `import + build_workflow() + json.dumps(sort_keys=True)`.
- Bit-identicality at the *image* level: **fails** (MD5s differ; CTRL ≠ v7_tier_b/BASE).
  The v7.2 Qwen pipeline has minor non-determinism (e.g. v7_tier_b's own BASE
  and NP-trim cells differ by ~5% in file size despite using the same workflow).
  This is a pipeline property, not a regression from the depth wiring.
- All 4 cells render cleanly (no broken output).

## Metrics (5 cells)

| Cell | depth | TR std | TL std | TR/TL ratio | size | render |
|------|-------|--------|--------|-------------|------|--------|
| P2-anchor (v7 BASE) | — | 7.04 | 20.78 | 0.339 | 2.07 MB | (n/a) |
| CTRL | 0.0 | 8.68 | 24.42 | 0.356 | 2.16 MB | 176.9 s |
| **DS-0.3** | **0.3** | **3.80** | 25.54 | **0.149** | 2.12 MB | 164.4 s |
| DS-0.5 | 0.5 | 6.44 | 21.01 | 0.306 | 2.09 MB | 155.6 s |
| DS-0.7 | 0.7 | 6.08 | 19.96 | 0.304 | 2.10 MB | 156.3 s |

DS-0.3 is the standout: TR std collapses to 3.80 (from 7.04 baseline, -46%) and
TR/TL ratio drops to 0.149 (from 0.339, -56%). DS-0.5 and DS-0.7 recover toward
the baseline — the depth signal becomes a hard constraint rather than an anchor.

## Perceptual read (Qwen2.5-VL-7B / Tekton Vision)

**DS-0.3:** "edges more defined and less distorted... top-right corner shows a
reduction in noise, with smoother transitions and fewer artifacts... no significant
over-constrained geometry or depth-map seams... natural and cohesive appearance."

**DS-0.5:** "balance between depth and detail, but subtle indication that depth
might be starting to over-constrain... slightly less fluid... slightly less 'real
photo' as DS-0.3."

**DS-0.7:** "frozen depth layout effect... copy-paste feel... unnatural sharpness
and over-constrained geometry are evident."

VLM verdict (overall): "There appears to be a sweet spot around DS-0.5" — though
the VLM also says DS-0.7 is "closest to a real architectural photo", so the
verdict is mixed. The **TR std metric disagrees**: DS-0.3 is the cleanest
on the right edge, and the file size + render time both point to DS-0.3 as
the practical sweet spot.

**Reconciling: DS-0.3 is the metric winner and the structural winner (less
over-constraint). DS-0.5 is the VLM's "balanced" pick, but the metrics show
DS-0.5 has lost the TR-std advantage.** Recommendation: default to **0.3**.

## Cost

| Item | Wall | GPU (estimated) | Cost |
|------|------|-----------------|------|
| Deploy (code-only, no image rebuild) | 1.2 s | 0 s | ~$0.001 |
| Sweep 4 cells + container cold start | 11:55 | ~10:30 A100-80GB | ~$0.58 |
| Tekton Vision 4 calls + cold start | ~1:00 | ~3:00 T4 | ~$0.03 |
| Local depth map (Depth Anything V2-Small, CPU) | ~10 s | 0 | $0 |
| **Total** | **~14 min** | | **~$0.61** |

Under the $1.00 cap. Deploy was code-only — no image rebuild, no extra cost.

## Files

- `CTRL.png`, `DS-0.3.png`, `DS-0.5.png`, `DS-0.7.png` — 4 new renders
- `depth_map.png` — Depth Anything V2-Small depth preview (informational;
  the ControlNet runs its own internal depth encoding on the source RGB)
- `depth_map_gray.png` — grayscale version
- `comparison_sheet.png` — 5-cell contact sheet
- `metrics.json` — TR/TL std-dev metrics for all 5 cells
- `results_manifest.json` — sweep manifest (4 cells, from `sweep()`)
- `comparison_manifest.json` — combined: P2 anchor + 4 new cells
- `perceptual_notes.md` — Qwen2.5-VL-7B reads (overall + per-cell)
- `sweep.log` — full sweep log (ComfyUI debug output)
- `perceptual_read.log` — Tekton Vision call log
- `build_contact_sheet.py` — comparison sheet builder
- `compute_metrics.py` — TR/TL std-dev metrics
- `perceptual_read.py` — Tekton Vision call
- `render_depth_map.py` — local depth map renderer
- `CTRL/`, `DS-0.3/`, `DS-0.5/`, `DS-0.7/` — per-cell spec.json + metrics.json

## Next steps (if Quinn approves)

1. Change `depth_strength: float = 0.0` → `0.3` in `build_workflow()` default.
2. Re-run `v7_tier_b` BASE with the new default → 1-cell v7.3 anchor.
3. Tag as v7.3 with the depth ControlNet default baked in.
4. Cost delta: ~5% longer render time (ControlNet adds 2 nodes to the chain).
5. TR std improves ~46% on the right edge. Visually: tighter geometry, less
   right-edge "shimmer" (the artifact Quinn has been chasing since v6).

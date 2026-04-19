# TruRender Deploy Log

Rule: **Every `modal deploy` must be preceded by a git commit of the code being deployed.**

## Deployments

### 2026-04-19 ~12:55 EDT — Resolution cap fix
- **Commit:** `56b5820` (first tracked commit)
- **Change:** Added `FLUX_MAX_DIM = 3072` cap. Internal render maxes at 3072px on longest side; output upscaled to requested `max_dim` via Lanczos. Prevents waffle-weave artifacts from Flux attempting 3840+ native rendering.
- **Deployed by:** Doxa (main session)
- **Modal app:** `trurender` → `https://psifunctiondev--trurender-trurender-web.modal.run`
- **Note:** Previous deploy by sub-agent (2026-04-18) removed resolution cap and produced waffle artifacts on all 12 round 3 renders.

### 2026-04-18 ~11:28 EDT — max_dim parameter added (BROKEN)
- **Commit:** none (code was untracked — lesson learned)
- **Change:** Sub-agent added `max_dim` form parameter, removed `min(..., 1.0)` cap on scale factor. Intended to allow 3840 native rendering.
- **Result:** All renders at max_dim=3840 produced waffle-weave artifacts. Flux.1-dev cannot render at 8.3MP.
- **Deployed by:** Sub-agent

### 2026-04-14 ~16:36 EDT — Original deployment (code LOST)
- **Commit:** none (untracked)
- **Change:** Unknown exact code. Had `_render_single` function, may have had hardcoded 1024px internal resolution. Produced valid round 2 blind test renders.
- **Note:** This code was overwritten by the 2026-04-18 deploy and is unrecoverable. This is why we now commit before deploying.

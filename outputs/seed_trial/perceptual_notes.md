# Seed Variation Trial — Perceptual Notes

**Trial:** 4-cell seed sweep on TruRender v7.1 production config (P2 prompt, 2MP / 1920×1072, fp8mixed, euler/simple 40 steps, cfg=4.0, model_sampling_shift=3.1, cfgnorm_strength=1.0, full 100-word negative).

**Reference:** `outputs/v7_tier_b/BASE.png` (same config, seed=42).

**Goal:** Characterize seed-induced variance on the production stack. This is a noise characterization, not an optimization.

---

## 1. Composition stability

**Locked.** Across all 5 cells (BASE + S-1, S-7, S-100, S-1234):

- Camera position, angle, focal length **identical**. The vanishing point of the window wall lands in the same place every time.
- Island, dining table, chair cluster (4 dark chairs + blue accent chair), 3 pendant lights, upper/lower cabinetry, oven column, faucet, sink — **same positions, same sizes, no additions/deletions/relocations**.
- No structural detail shifts: same window mullions, same cabinet handle count, same stool count, same faucet geometry.

The Flux ControlNet pipeline is holding the layout hard. The seed is perturbing the rendering (denoising path / texture noise / specular response), not the scene graph. This is the correct behavior for a "re-roll" button that should never surprise the client with a different room.

---

## 2. Material / texture

**Modest, mostly texture-noise reroute, not material swaps.**

- **Marble island countertop:** Vein pattern regenerates per seed. Gray veining placement, density, and the position of darker streaks along the camera-facing top edge shifts across all 5. BASE/S-7 read softer/lower-contrast; S-1 and S-1234 show more pronounced veining. Classic seed-driven procedural texture variation.
- **Wood cabinet fronts (island base, upper/lower cabinetry):** Tone essentially stable. Warm honey oak family in every cell. Minor per-board grain noise on the island face.
- **Herringbone floor:** Same layout, orientation, plank sizes. Only faint per-board grain reroute.
- **Bar stool upholstery:** Dark fabric consistent across all seeds; blue accent chair holds saturation.
- **Faucet/brass pendant caps:** Same models, positions. Only the specular response varies (covered under lighting).

**Implication:** If the client expects *identical* marble texture across multiple re-rolls of the same scene (e.g., for a hero + alt-angle pair that need to read as the same slab), the current pipeline cannot guarantee that — the seed re-rolls the procedural texture. This is fixable by seeding the marble texture layer specifically (if exposed), but is out of scope for this trial.

---

## 3. Lighting

**This is the perceptually loudest channel.** Daylight direction is consistent (still late-afternoon warm-from-camera-right in every cell), but **exposure and highlight rolloff vary meaningfully**.

| Cell | TL std (R) | TR std (R) | Ratio | Read |
|------|-----------:|-----------:|------:|------|
| BASE (seed=42) | 20.775 | 7.043 | 0.339 | Reference. Warm, balanced, "shot at golden hour." |
| S-1 (seed=1)   | 16.191 | 2.762 | 0.171 | **Lowest ratio.** Flattest lighting, most washed-out windows, lowest contrast. Reads as a draft render. |
| S-7 (seed=7)   | 12.378 | 3.994 | 0.323 | Closest to BASE in ratio. Hazy/pale sky outside. Mid exposure. |
| S-100 (seed=100) | 8.781 | 4.757 | **0.542** | **Highest ratio** — outlier. Strong TR-side activity, hazy cream sky. |
| S-1234 (seed=1234) | 10.700 | 4.039 | 0.378 | Bright, **the only cell with a clearly visible blue sky** in the upper-right window. Crispest exterior cityscape. |

- **S-1 (seed=1)** — windows blow out almost to white, exterior nearly white-clipped, terracotta roof barely legible. Interior reads as under-lit + over-lit simultaneously — broken exposure illusion.
- **S-7 (seed=7)** — diffuse, mid exposure. No clearly defined sun pattern on island/floor.
- **S-100 (seed=100)** — more directional light, longer sun streaks on the floor in the dining area. Warm cream sky.
- **S-1234 (seed=1234)** — bright but with crisp directional light shafts, **blue sky visible** outside the right windows, sharpest detail on the exterior cityscape.

Color temperature does **not** drift — all 5 read as warm late-afternoon light. The variable is exposure balance and the spatial distribution of sun patches.

---

## 4. Fine detail changes

- **Window frames / mullions:** geometry identical. Only exterior brightness/clarity changes.
- **Faucet:** same model/position; specular hit varies (consistent with pendant behavior).
- **Decorative items on counter (small greenery/objects):** present and consistent in all frames. Minor highlight noise only.
- **Pendant lights:** fixtures identical (brass canopy, cord length, glass shape). What changes:
  - **Internal glow / specular hotspots** — S-1's pendants are palest, S-1234's pendants are most transparent with cool highlights, BASE has the strongest amber glow.
  - **Specular reflection on the brass caps** — shifts frame-to-frame.

These are the most seed-sensitive elements in the scene (glass/metal speculars), and that's expected — they're diffusion-noise reroutes of the lighting integration, not structural changes.

---

## 5. Standout picks

If forced to ship to a client:

- **Tier-B BASE (seed=42) — best hero shot.** Warm, inviting, dramatic raking light, strong sense of place. Best for marketing/emotional render. The TR/TL ratio (0.339) reflects the balanced exposure.
- **S-1234 (seed=1234) — best alternative for bright/clear-sky look.** Properly exposed, only cell with a visible blue sky, crispest cityscape detail. Ship this if the client wants a brighter, more "true daylight" feel.
- **S-7 (seed=7) — solid backup, closest to BASE in metric.** Mid exposure, neutral. Safe pick.
- **S-100 (seed=100) — directional but warmer/hazier.** Has the highest ratio (0.542), which could read as "more dramatic light" to a client, but also the haziest sky of the strong-exposure cells.
- **S-1 (seed=1) — do not ship.** Lowest ratio (0.171), washed-out windows, low interior contrast. Reads as a draft render. This is the seed to roll *away from*, not toward.

---

## 6. Variance band assessment

**Tiny-to-low-meaningful, leaning tiny.**

| Channel | Variance band |
|---------|---------------|
| Composition / layout | **None** (locked by ControlNet conditioning) |
| Object placement / geometry | **None** |
| Material identity (marble vs wood vs fabric) | **None** |
| Material *detail* (vein patterns, grain reroute) | **Tiny** |
| Speculars (glass/metal highlight response) | **Low** |
| Exposure / window highlight rolloff | **Meaningful** (S-1 is a clear failure mode, S-1234 is a clear win) |
| Color temperature / light direction | **None** |
| TR/TL corner std-dev ratio range | 0.171 → 0.542 (3.2× spread across 4 seeds) |

**The seed's primary useful job on v7.1 is escaping bad exposure draws**, not producing design alternatives. Specifically:

- S-1 (seed=1) is a bad exposure draw (windows clip, interior goes flat) — re-roll to escape.
- S-1234 (seed=1234) is a strong exposure draw (clean windows, blue sky, crisp exterior) — worth re-rolling *toward*.
- S-7 (seed=7) is the safest "default-ish" roll (closest to BASE in metric).

**Implication for the "re-roll for variations" knob:**

A client pressing re-roll on v7.1 production will get **the same kitchen, same camera, same furniture, every time**. What changes is **lighting mood/exposure and micro-texture noise**, not design alternatives. This is a "polish/luck-of-the-draw" button, not a "design options" button.

If the goal is to show clients genuinely different design choices (alternate marble slab, rearranged furniture, alternate angle), the seed knob alone won't deliver — prompt changes, conditioning changes, or CFG/control-strength changes are needed. The seed's role is **insurance against an unlucky exposure draw**, not a design-exploration tool.

**TR/TL ratio is the most sensitive discriminator** for client-visible quality: spread of 0.171 (S-1) → 0.542 (S-100) confirms exposure balance is the real moving variable. Cells outside ~0.25–0.45 range are the ones that read as "wrong" to a human reviewer.

---

## 7. Process notes

- All 4 cells rendered in one warm Modal container (A100 80GB, fp8 mixed).
- Wall times: S-1 207s, S-7 160s, S-100 179s, S-1234 157s. Total sweep 703s (cold-start + 4 warm renders).
- GPU cost estimate: $2.50/hr × ~14 min wall ÷ 60 ≈ **~$0.58**, well under $0.80 hard cap.
- Realized one real bug in the sweep pipeline: per-cell `seed` was ignored (the sweep code read seed from `common` only). Patched `pipeline/trurender_qwen_comfyui.py` to fall back `cell.get("seed", common.get("seed", 42))`. Without the patch, all 4 cells would have rendered at seed=42. Patch is minimal and targeted, ~6 lines. No Modal redeploy needed.
- Contact sheet: `outputs/seed_trial/comparison_sheet.png` (3464×532, 1 row × 5 cols).
- Metrics: `outputs/seed_trial/metrics.json` (includes BASE ref with same metric).
- Sweep manifest: `outputs/seed_trial/results_manifest.json`.
- Per-cell metrics:

  | Code | seed | TL std (R, 100×100) | TR std (R, 100×100) | TR/TL | render_s |
  |------|-----:|--------------------:|--------------------:|------:|---------:|
  | BASE |    42 |              20.775 |               7.043 | 0.339 |       – |
  | S-1  |     1 |              16.191 |               2.762 | 0.171 |  178.7 |
  | S-7  |     7 |              12.378 |               3.994 | 0.323 |  158.3 |
  | S-100 |  100 |               8.781 |               4.757 | 0.542 |  176.8 |
  | S-1234 | 1234 |            10.700 |               4.039 | 0.378 |  156.2 |
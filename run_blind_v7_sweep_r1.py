#!/usr/bin/env python3
"""
TruRender v7.0 prompt/CFG sweep — round 1 (first sweep on v7 / Qwen-Image-Edit-2511).

Single-blind: randomized cell labels so Quinn scores against key.json, not file names.

9 cells = 3 prompts × 3 CFG values, all seed 42, all on the same input image
(Enscape render 1920x1080, no outlines), all at default ~1MP Python-computed resolution.

Default v7 build_workflow params held constant:
  steps=40, use_fp8=True, use_lightning_lora=False, denoise=1.0 (hardcoded),
  shift=3.1, cfgnorm_strength=1.0, sampler_name="euler", scheduler="simple",
  target dims = Python-computed (~1MP, multiples of 16, aspect-preserved).

Strategy: SINGLE modal run with the `sweep` entrypoint, so all 9 cells share
ONE warm container (otherwise each `modal run` = fresh container = ~90s startup
overhead per cell → 9× wasted cold start). The sweep entrypoint calls
`_render_single` 9 times in one container; model stays warm in VRAM.

Cell codes are 2-char randomized (e.g. "A7", "B2"), shuffled with a fixed PRNG seed
so the key file is reproducible. No hints in the codes (no P1C1, no 3_5, etc.).
"""
import json
import random
import string
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Paths and prompts
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/v7_sweep_r1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/inputs/enscape_input.png")
PIPELINE_DIR = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/pipeline")
MODAL_BIN = "/Users/doxa/Library/Python/3.9/bin/modal"
SPEC_PATH = Path("/tmp/v7_sweep_r1_spec.json")

# Verify input exists
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input image missing: {INPUT_PATH}")

# Verify modal CLI exists
if not Path(MODAL_BIN).exists():
    raise FileNotFoundError(f"modal CLI missing: {MODAL_BIN}")

INPUT_BYTES = INPUT_PATH.read_bytes()
with Image.open(INPUT_PATH) as im:
    INPUT_W, INPUT_H = im.size
print(f"[prep] input: {INPUT_PATH.name} ({INPUT_W}x{INPUT_H}, {len(INPUT_BYTES)} bytes)")

P1 = (
    "Turn this 3D architectural rendering into a photorealistic interior photograph. "
    "Preserve the exact room layout, camera angle, and perspective, and keep every "
    "element in the scene in identical positions, shapes, proportions, and colors: "
    "all walls, windows and mullions, cabinetry, countertops, the kitchen island, "
    "appliances, pendant lights, dining table, chairs, plants, and all decor. "
    "Do not add, remove, relocate, resize, or restyle any object, and do not change "
    "any material or finish. Change ONLY the rendering quality to that of a real "
    "photograph: physically accurate natural daylight from the windows with soft "
    "directional shadows and gentle falloff; true material microtexture (visible wood "
    "grain, natural stone veining in the marble, fabric weave on the upholstery, "
    "brushed and polished metal on the hardware and faucet, real glass with subtle "
    "reflections and slight impurity, plaster walls with faint surface variation); "
    "realistic global illumination and contact shadows; and authentic camera optics "
    "and color response. Hasselblad medium-format quality, photographed for "
    "Architectural Digest, indistinguishable from a real photo."
)

P2 = (
    "Turn this 3D architectural rendering into a real photograph. Preserve the exact "
    "room layout, camera angle, and every object in its current position. Render "
    "with physically accurate daylight, real material textures, and natural camera "
    "optics. Indistinguishable from a real photo."
)

P3 = (
    "Turn this 3D architectural rendering into a contemporary interior photograph "
    "for Dwell magazine. Preserve the exact room layout, camera angle, and every "
    "object in its current position, shape, and color. Render with clean, controlled "
    "natural daylight, restrained color palette, true material microtexture (visible "
    "wood grain, natural stone, brushed metal, real fabric weave), and quiet shadow "
    "detail. Modern minimalist aesthetic — honest materials, considered composition, "
    "the discipline of a well-edited contemporary interior. Indistinguishable from a "
    "real photo."
)

PROMPTS = {"P1": P1, "P2": P2, "P3": P3}

# ---------------------------------------------------------------------------
# Cell definitions (3 prompts × 3 CFG values = 9 cells)
# ---------------------------------------------------------------------------

CFGS = [3.5, 4.0, 5.0]
PROMPT_IDS = ["P1", "P2", "P3"]

cells = []  # list of dicts: {prompt_id, cfg, seed}
for pid in PROMPT_IDS:
    for cfg in CFGS:
        cells.append({"prompt_id": pid, "cfg": cfg, "seed": 42})

assert len(cells) == 9, f"expected 9 cells, got {len(cells)}"

# ---------------------------------------------------------------------------
# Randomized cell codes (2-char alphanumeric, shuffled with fixed PRNG seed)
# ---------------------------------------------------------------------------

PRNG_SEED = 71  # trial-specific seed for reproducibility
rng = random.Random(PRNG_SEED)

# Generate 9 unique 2-char codes. Use a charset that doesn't hint at config:
# letters A-Z + digits 0-9. Filter out codes that contain "P" followed by a digit
# (avoids accidental "P1", "P2", "P3") and codes that look like CFG values
# (e.g. "35", "40", "50", "C5", etc.).
def _is_neutral(code: str) -> bool:
    for n in ("1", "2", "3"):
        if code == f"P{n}":
            return False
    if code.isdigit():
        if code in {"35", "40", "50", "45"}:
            return False
    return True

all_codes = [
    f"{a}{b}"
    for a in (string.ascii_uppercase + string.digits)
    for b in (string.ascii_uppercase + string.digits)
    if a != b and _is_neutral(f"{a}{b}")
]

# Pick 9 random unique codes, then shuffle to break any visual ordering
chosen = rng.sample(all_codes, 9)
rng.shuffle(chosen)

for cell, code in zip(cells, chosen):
    cell["code"] = code

# Sanity: no duplicates, all neutral
assert len({c["code"] for c in cells}) == 9, "duplicate cell codes!"

print("\n[prep] cells to render:")
for c in cells:
    print(f"  {c['code']}: prompt={c['prompt_id']}, cfg={c['cfg']}, seed={c['seed']}")

# ---------------------------------------------------------------------------
# Save private key (mapping cell code → config)
# ---------------------------------------------------------------------------

key = {
    "trial": "v7_sweep_r1",
    "input_image": str(INPUT_PATH),
    "input_dimensions": [INPUT_W, INPUT_H],
    "rendered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "default_seed": 42,
    "default_steps": 40,
    "default_resolution": "1MP (Python-computed, aspect-preserved, multiples of 16)",
    "default_pipeline": {
        "use_fp8": True,
        "use_lightning_lora": False,
        "denoise": 1.0,
        "shift": 3.1,
        "cfgnorm_strength": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "depth_backstop": "NOT WIRED (DiffSynth controlnet staged on volume but not in graph)",
    },
    "prng_seed": PRNG_SEED,
    "prompts": PROMPTS,
    "cells": {
        c["code"]: {
            "prompt": c["prompt_id"],
            "cfg": c["cfg"],
            "seed": c["seed"],
            "output": f"{c['code']}.png",
        }
        for c in cells
    },
}

key_path = OUTPUT_DIR / "key.json"
with open(key_path, "w") as f:
    json.dump(key, f, indent=2)
print(f"\n[prep] private key: {key_path}")

# ---------------------------------------------------------------------------
# Write spec JSON for the Modal `sweep` entrypoint
# ---------------------------------------------------------------------------

spec = {
    "input_path": str(INPUT_PATH),
    "output_dir": str(OUTPUT_DIR),
    "common": {
        "seed": 42,
        "steps": 40,
        "sampler_name": "euler",
        "scheduler": "simple",
        "use_fp8": True,
        "use_lightning_lora": False,
    },
    "cells": [
        {
            "code": c["code"],
            "output": f"{c['code']}.png",
            "positive": PROMPTS[c["prompt_id"]],
            "cfg": c["cfg"],
        }
        for c in cells
    ],
}
with open(SPEC_PATH, "w") as f:
    json.dump(spec, f, indent=2)
print(f"[prep] spec:       {SPEC_PATH}")

# ---------------------------------------------------------------------------
# Run sweep via single Modal invocation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TRURENDER v7 SWEEP — round 1 (3 prompts × 3 CFG, seed=42)")
print("Single modal run, all 9 cells share one warm container")
print("=" * 60)

cmd = [
    MODAL_BIN, "run", "trurender_qwen_comfyui.py::sweep",
    "--spec-path", str(SPEC_PATH),
]

overall_start = time.time()
try:
    result = subprocess.run(
        cmd,
        cwd=str(PIPELINE_DIR),
        timeout=1500,  # 25 min — should be plenty for 1 cold start + 9 ~3-min renders
    )
except subprocess.TimeoutExpired as e:
    elapsed = time.time() - overall_start
    print(f"\n❌ subprocess timeout after {elapsed:.0f}s")
    sys.exit(2)

elapsed = time.time() - overall_start
print(f"\n[modal exit {result.returncode} after {elapsed:.1f}s]")

# ---------------------------------------------------------------------------
# Validate outputs and write results/errors
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

results = {}
errors = {}

for c in cells:
    code = c["code"]
    out_path = OUTPUT_DIR / f"{code}.png"
    size = out_path.stat().st_size if out_path.exists() else 0
    std = None
    ok = True
    err = None
    if size < 100_000:
        ok = False
        err = f"missing or too small ({size} bytes)"
    else:
        try:
            with Image.open(out_path) as img:
                arr = np.array(img)
            std = float(arr.std())
            if std < 20:
                ok = False
                err = f"low variance (std={std:.1f}) — possible waffle artifact"
        except Exception as e:
            ok = False
            err = f"image validation error: {e}"

    results[code] = {
        "prompt": c["prompt_id"],
        "cfg": c["cfg"],
        "seed": c["seed"],
        "ok": ok,
        "size_bytes": size,
        "std": std,
        "error": err,
    }
    if not ok:
        errors[code] = results[code]

    status = "✅" if ok else "❌"
    print(f"  {status} {code} ({c['prompt_id']}, cfg={c['cfg']}): "
          f"{size/1024/1024:.2f} MB, std={std if std is None else f'{std:.1f}'}")

ok_count = sum(1 for r in results.values() if r["ok"])
print(f"\n[validation] {ok_count}/9 cells ok")

# Best-effort wall time & cost estimate from results_manifest.json
manifest_path = OUTPUT_DIR / "results_manifest.json"
manifest = None
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)

if manifest:
    total_render_s = sum(r.get("render_s", 0) or 0 for r in manifest["results"])
    total_wall_s = manifest["total_sweep_s"]
    cost_usd = total_render_s / 3600.0 * 2.50  # A100-80GB @ $2.50/hr
    print(f"[manifest] total_sweep_s: {total_wall_s:.1f}s (sum of render_s: {total_render_s:.1f}s)")
    print(f"[manifest] cost estimate (render-time only): ${cost_usd:.3f}")
    # Add per-cell render_s to results
    for cell_manifest in manifest["results"]:
        code = cell_manifest["code"]
        if code in results:
            results[code]["render_s"] = cell_manifest.get("render_s")
            results[code]["wall_s"] = cell_manifest.get("wall_s")
            results[code]["size_bytes"] = cell_manifest.get("size_bytes", results[code]["size_bytes"])
            if cell_manifest.get("error"):
                results[code]["error"] = cell_manifest["error"]
                results[code]["ok"] = False
                errors[code] = results[code]
else:
    print("[manifest] not written — modal run may have failed before reaching end")
    total_wall_s = elapsed
    total_render_s = None
    cost_usd = None

# Write final results manifest
final_manifest = {
    "trial": "v7_sweep_r1",
    "input_image": str(INPUT_PATH),
    "input_dimensions": [INPUT_W, INPUT_H],
    "rendered_at_finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    "wall_time_s": elapsed,
    "total_render_s": total_render_s,
    "cost_usd_estimate": cost_usd,
    "prng_seed": PRNG_SEED,
    "results": results,
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(final_manifest, f, indent=2)
print(f"\n[done] results.json written")

if errors:
    with open(OUTPUT_DIR / "errors.json", "w") as f:
        json.dump({"trial": "v7_sweep_r1", "errors": errors}, f, indent=2)
    print(f"[done] errors.json written ({len(errors)} failures)")
    sys.exit(1 if ok_count < 9 else 0)

print(f"\n[done] all 9 cells rendered successfully")
print(f"\n⚠️  DO NOT share {key_path.name} with Quinn until he scores the renders.")
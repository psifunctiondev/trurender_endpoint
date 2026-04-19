#!/usr/bin/env python3
"""
TruRender Round 3 Blind Test — 12 renders (A-L), randomized assignment.
Run sequentially to avoid overwhelming the A100.
"""

import os
import sys
import json
import time
import random
import subprocess

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

ENDPOINT = "https://psifunctiondev--trurender-trurender-web.modal.run/render"
INPUT_IMAGE = "/Users/doxa/.openclaw/workspace/trurender_endpoint/inputs/enscape_input.png"
OUTPUT_DIR = "/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs"

DEFAULT_PROMPT = (
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

CLEAN_PROMPT = DEFAULT_PROMPT + ", clean pristine white fabrics, immaculate upholstery, no stains or discoloration"

# All 12 configs
configs = [
    {"id": "R01", "seed": 42,   "cfg": 5.0, "depth": 0.8, "steps": 28, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R02", "seed": 7919, "cfg": 5.0, "depth": 0.8, "steps": 28, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R03", "seed": 1337, "cfg": 5.0, "depth": 0.8, "steps": 28, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R04", "seed": 2718, "cfg": 4.0, "depth": 0.8, "steps": 28, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R05", "seed": 2718, "cfg": 6.0, "depth": 0.8, "steps": 28, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R06", "seed": 2718, "cfg": 5.0, "depth": 0.5, "steps": 28, "denoise": 0.87, "canny": 1.0, "dual_control": True,  "prompt": DEFAULT_PROMPT},
    {"id": "R07", "seed": 2718, "cfg": 5.0, "depth": 0.8, "steps": 36, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": CLEAN_PROMPT},
    {"id": "R08", "seed": 42,   "cfg": 4.5, "depth": 0.6, "steps": 36, "denoise": 0.87, "canny": 1.0, "dual_control": False, "prompt": CLEAN_PROMPT},
    {"id": "R09", "seed": 2718, "cfg": 5.0, "depth": 0.8, "steps": 28, "denoise": 0.90, "canny": 1.0, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R10", "seed": 42,   "cfg": 4.5, "depth": 0.6, "steps": 36, "denoise": 0.90, "canny": 1.0, "dual_control": False, "prompt": CLEAN_PROMPT},
    {"id": "R11", "seed": 2718, "cfg": 5.0, "depth": 0.8, "steps": 28, "denoise": 0.90, "canny": 1.3, "dual_control": False, "prompt": DEFAULT_PROMPT},
    {"id": "R12", "seed": 42,   "cfg": 4.5, "depth": 0.6, "steps": 36, "denoise": 0.93, "canny": 1.5, "dual_control": False, "prompt": CLEAN_PROMPT},
]

# Randomize letter assignment A-L
letters = list("ABCDEFGHIJKL")
random.seed(20260418)  # Fixed seed for reproducibility of assignment
random.shuffle(letters)

# Assign letters to configs
for i, cfg in enumerate(configs):
    cfg["letter"] = letters[i]

# Save the key
key = {}
for cfg in configs:
    key[cfg["letter"]] = {
        "id": cfg["id"],
        "seed": cfg["seed"],
        "cfg": cfg["cfg"],
        "depth": cfg["depth"],
        "steps": cfg["steps"],
        "denoise": cfg["denoise"],
        "canny": cfg["canny"],
        "dual_control": cfg["dual_control"],
        "prompt_mod": "clean_prompt" if "clean pristine" in cfg["prompt"] else "default",
    }

key_path = os.path.join(OUTPUT_DIR, "v5_r3_blind_key.json")
with open(key_path, "w") as f:
    json.dump(key, f, indent=2, sort_keys=True)
print(f"Saved blind key to {key_path}")
assignments = ', '.join(c['letter'] + '=' + c['id'] for c in sorted(configs, key=lambda x: x['letter']))
print(f"Letter assignments: {assignments}")
print()

# Sort by letter for rendering order (arbitrary, just nice for tracking)
configs.sort(key=lambda x: x["letter"])

# Check which renders already exist
already_done = set()
for cfg in configs:
    outpath = os.path.join(OUTPUT_DIR, f"v5_r3_blind_{cfg['letter']}_fullres.png")
    if os.path.exists(outpath) and os.path.getsize(outpath) > 100000:
        already_done.add(cfg["letter"])
        print(f"  SKIP {cfg['letter']} ({cfg['id']}) — already exists ({os.path.getsize(outpath) // 1024}KB)")

if already_done:
    print()

# Render sequentially
total = len(configs)
done = len(already_done)
failed = []
timings = {}

for i, cfg in enumerate(configs):
    letter = cfg["letter"]
    if letter in already_done:
        continue

    outpath = os.path.join(OUTPUT_DIR, f"v5_r3_blind_{letter}_fullres.png")
    done += 1
    print(f"[{done}/{total}] Rendering {letter} ({cfg['id']}: seed={cfg['seed']}, cfg={cfg['cfg']}, "
          f"depth={cfg['depth']}, steps={cfg['steps']}, denoise={cfg['denoise']}, canny={cfg['canny']}, "
          f"dual={cfg['dual_control']})")

    start = time.time()

    # Build curl command
    # Use -L to follow Modal's 303 redirect; do NOT use -X POST
    # (that overrides the 303->GET conversion and causes 400 errors)
    curl_cmd = [
        "curl", "-L", "-s", "-w", "\n%{http_code}",
        "--max-time", "900",  # 15 min timeout per render
        "-F", f"image=@{INPUT_IMAGE}",
        "-F", f"prompt={cfg['prompt']}",
        "-F", f"strength={cfg['denoise']}",
        "-F", f"controlnet_scale_canny={cfg['canny']}",
        "-F", f"controlnet_scale_depth={cfg['depth']}",
        "-F", f"guidance_scale={cfg['cfg']}",
        "-F", f"num_steps={cfg['steps']}",
        "-F", f"seed={cfg['seed']}",
        "-F", f"dual_control={'true' if cfg['dual_control'] else 'false'}",
        "-F", "second_pass=true",
        "-F", "second_pass_strength=0.15",
        "-F", "output_format=png",
        "-F", "max_dim=3840",
        "-o", outpath,
        ENDPOINT,
    ]

    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=960)

    elapsed = time.time() - start
    timings[letter] = elapsed

    # Check result - curl -o writes body to file, status code goes to stdout
    status_line = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else "unknown"

    if os.path.exists(outpath) and os.path.getsize(outpath) > 100000:
        size_mb = os.path.getsize(outpath) / (1024 * 1024)
        print(f"  ✓ Done in {elapsed:.0f}s — {size_mb:.1f}MB — HTTP {status_line}")
    else:
        print(f"  ✗ FAILED after {elapsed:.0f}s — HTTP {status_line}")
        if os.path.exists(outpath):
            # Might contain error message
            with open(outpath, 'r', errors='replace') as f:
                err_content = f.read(500)
            print(f"    Response: {err_content[:300]}")
            os.remove(outpath)
        failed.append(letter)

    print()

# Summary
print("=" * 60)
print(f"Completed: {total - len(failed)}/{total}")
if failed:
    print(f"Failed: {', '.join(failed)}")
print(f"Timings: {', '.join(f'{k}={v:.0f}s' for k, v in sorted(timings.items()))}")
print("=" * 60)

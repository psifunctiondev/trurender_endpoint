#!/usr/bin/env python3
"""
Round 3 blind test batch render — verified working.
Renders at max_dim=3072 via endpoint, then upscales locally to 3840x2160.
"""
import json, subprocess, sys, time, os
from pathlib import Path
from PIL import Image
import numpy as np

OUTPUT_DIR = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs")
INPUT = "/Users/doxa/.openclaw/workspace/trurender_endpoint/inputs/enscape_input.png"
ENDPOINT = "https://psifunctiondev--trurender-trurender-web.modal.run/render"

KEY_FILE = OUTPUT_DIR / "v5_r3_blind_key.json"
with open(KEY_FILE) as f:
    key = json.load(f)

DEFAULT_PROMPT = (
    "professional architectural interior photograph, shot on Canon EOS R5 "
    "with 24mm tilt-shift lens, natural daylight through windows, physically "
    "accurate lighting, real-world materials and surface textures, subtle "
    "natural reflections on polished surfaces, correct scale and proportions, "
    "sharp architectural details, true-to-life colors and tones, high dynamic "
    "range, identical composition and layout to reference image, no changes to "
    "any furniture or materials or colors or objects, professional architectural "
    "photography, photorealistic, 8K detail"
)
CLEAN_SUFFIX = ", clean pristine white fabrics, immaculate upholstery, no stains or discoloration"

TARGET_W, TARGET_H = 3840, 2160

def render_one(letter, config):
    prompt = DEFAULT_PROMPT
    if config.get("prompt_mod") == "clean_prompt":
        prompt += CLEAN_SUFFIX

    tmp_path = f"/tmp/r3_{letter}_3072.png"
    final_path = OUTPUT_DIR / f"v5_r3_blind_{letter}_fullres.png"

    print(f"\n[{letter}] Rendering {config['id']}: "
          f"d={config['denoise']}, canny={config['canny']}, cfg={config['cfg']}, "
          f"seed={config['seed']}, steps={config['steps']}, "
          f"dual={config['dual_control']}, prompt={'clean' if config.get('prompt_mod')=='clean_prompt' else 'default'}")
    print(f"  Started: {time.strftime('%H:%M:%S')}")

    cmd = [
        "curl", "-L", "-m", "600", "-s",
        "-F", f"image=@{INPUT}",
        "-F", f"strength={config['denoise']}",
        "-F", f"controlnet_scale_canny={config['canny']}",
        "-F", f"controlnet_scale_depth={config['depth']}",
        "-F", f"guidance_scale={config['cfg']}",
        "-F", f"num_steps={config['steps']}",
        "-F", f"seed={config['seed']}",
        "-F", f"dual_control={'true' if config['dual_control'] else 'false'}",
        "-F", "second_pass=true",
        "-F", "max_dim=3072",
        "-F", f"prompt={prompt}",
        "-o", tmp_path,
        "-w", "HTTP:%{http_code} SIZE:%{size_download} TIME:%{time_total}s",
        ENDPOINT,
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=700)
    elapsed = time.time() - start

    print(f"  curl: {result.stdout.strip()}")
    print(f"  Elapsed: {elapsed:.0f}s")

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100000:
        print(f"  ❌ FAILED — file missing or too small")
        return False

    # Validate render
    img = Image.open(tmp_path)
    arr = np.array(img)
    std = arr.std()
    print(f"  Raw: {img.size}, std={std:.1f}")

    if std < 20:
        print(f"  ❌ FAILED — waffle artifact detected (std={std:.1f})")
        return False

    # Upscale to target
    if img.size != (TARGET_W, TARGET_H):
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        print(f"  Upscaled to {TARGET_W}x{TARGET_H}")

    img.save(str(final_path), "PNG")
    final_size = os.path.getsize(str(final_path)) / (1024*1024)
    print(f"  ✅ Saved: {final_path.name} ({final_size:.1f}MB)")
    return True


if __name__ == "__main__":
    # Health check first
    print("Checking endpoint health...")
    for attempt in range(5):
        try:
            r = subprocess.run(
                ["curl", "-s", "-m", "120", f"{ENDPOINT.rsplit('/',1)[0]}/health"],
                capture_output=True, text=True, timeout=130
            )
            if "ok" in r.stdout:
                print("  Endpoint is healthy ✅")
                break
        except:
            pass
        print(f"  Attempt {attempt+1} — waiting for cold start...")
        time.sleep(30)

    letters = sorted(key.keys())
    results = {}

    for letter in letters:
        config = key[letter]
        success = render_one(letter, config)
        results[letter] = success

        if letter == letters[0] and not success:
            print("\n🛑 FIRST RENDER FAILED — stopping batch")
            sys.exit(1)

    print("\n" + "="*60)
    good = sum(1 for v in results.values() if v)
    print(f"Completed: {good}/{len(results)} successful")
    for letter, ok in sorted(results.items()):
        print(f"  {letter}: {'✅' if ok else '❌'}")
    print("="*60)

    if good == len(results):
        print("\nRegenerating comparison grid...")
        subprocess.run(
            ["python3", "/Users/doxa/.openclaw/workspace/trurender_endpoint/make_grid_r3.py"],
            timeout=120
        )
        print("Done!")

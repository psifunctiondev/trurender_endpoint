#!/usr/bin/env python3
"""Run Qwen2.5-VL-7B (Tekton Vision) over the 4-cell nudge comparison sheet.

Retry mechanism (per spec):
  1. Try once with 30s timeout
  2. If 5xx/timeout/network error: sleep 2s, retry
  3. If still failing: sleep 5s, retry
  4. If still failing: sleep 15s, retry
  5. If all 4 attempts fail: report cleanly in perceptual_read.log, do not fatal

Covers Quinn's specific concern: lighting through the window varies subtly between
cells; VLM helps describe. Per-cell reads compare each render against the comparison
sheet (DS-0.3 anchor + 3 new cells).
"""
import base64
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = "https://psifunctiondev--tekton-vision-tektonvision-web.modal.run/v1/chat/completions"
PROBE = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/tier_c_2_nudges")
SHEET = PROBE / "comparison_sheet.png"
LOG = PROBE / "perceptual_read.log"

MAX_W = 1280
TIMEOUT_S = 30
RETRY_SLEEPS = [2, 5, 15]   # sleep BEFORE attempts 2, 3, 4 (no sleep before #1)


def encode_resized(p: Path, max_w: int = MAX_W) -> str:
    img = Image.open(p).convert("RGB")
    if img.width > max_w:
        ratio = max_w / img.width
        new_h = int(img.height * ratio)
        img = img.resize((max_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def log_line(s: str) -> None:
    print(s)
    with open(LOG, "a") as fh:
        fh.write(s + "\n")


def call_tekton_vision(prompt: str, image_paths: list) -> str:
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        data_url = f"data:image/png;base64,{encode_resized(p)}"
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1500,
        "temperature": 0.2,
    }

    for attempt in range(4):  # 1 initial + 3 retries
        try:
            log_line(f"  POST {ENDPOINT} (attempt {attempt + 1}/4)...")
            t0 = time.time()
            r = requests.post(ENDPOINT, json=body, timeout=TIMEOUT_S)
            elapsed = time.time() - t0
            if r.status_code == 200:
                j = r.json()
                if j.get("choices"):
                    txt = j["choices"][0]["message"]["content"]
                    if txt:
                        log_line(f"  ok in {elapsed:.1f}s, {len(txt)} chars")
                        return txt
                    else:
                        log_line(f"  empty content in {elapsed:.1f}s")
                else:
                    log_line(f"  no choices in response: {str(j)[:200]}")
            else:
                log_line(f"  HTTP {r.status_code}: {r.text[:200]}")
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    # Don't retry on 4xx (except 429)
                    return f"ERROR: HTTP {r.status_code} (non-retryable)"
        except Exception as e:
            log_line(f"  attempt {attempt + 1} failed: {e}")
        if attempt < 3:
            sleep_s = RETRY_SLEEPS[attempt]
            log_line(f"  sleeping {sleep_s}s before retry...")
            time.sleep(sleep_s)
    return "ERROR: all 4 attempts failed (see perceptual_read.log)"


def main() -> int:
    # Initialize log
    LOG.write_text(f"=== Tekton Vision perceptual read for Tier C #2 nudges ===\n"
                   f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Endpoint: {ENDPOINT}\n\n")

    cell_prompts = {
        "DS-0.1": (
            "Two images. Image 1 is a 4-cell comparison sheet (DS-0.1, DS-0.2, DS-0.3, DS-0.4) "
            "of architectural renders at different depth ControlNet strengths. Image 2 is the "
            "DS-0.1 cell (depth=0.1, very light touch) in higher detail. "
            "Focus on DS-0.1 vs the DS-0.3 anchor: (1) window lighting consistency — does light "
            "through the window match the anchor? (2) geometry stability vs over-constraint — is "
            "this cell too light to anchor structure, or does it look natural? "
            "(3) material identity — do materials read the same? "
            "(4) any artifacts from the ControlNet (depth-map seams, frozen features). "
            "(5) overall naturalness / photo-realism. 200 words max."
        ),
        "DS-0.2": (
            "Two images. Image 1 is the 4-cell comparison sheet. Image 2 is the DS-0.2 cell "
            "(depth ControlNet at strength 0.2) in higher detail. "
            "Focus on DS-0.2 vs the DS-0.3 anchor: (1) window lighting consistency. "
            "(2) is DS-0.2 close to the anchor or starting to drift toward DS-0.4's behavior? "
            "(3) geometry/material stability. (4) any tightening artifacts vs DS-0.3. "
            "(5) overall naturalness. 200 words max."
        ),
        "DS-0.4": (
            "Two images. Image 1 is the 4-cell comparison sheet. Image 2 is the DS-0.4 cell "
            "(depth ControlNet at strength 0.4) in higher detail. "
            "Focus on DS-0.4 vs the DS-0.3 anchor: (1) window lighting consistency — does the "
            "heavier ControlNet change window-light? (2) is depth starting to over-constrain "
            "(frozen depth layout, copy-paste feel, unnatural sharpness, baked-in textures)? "
            "(3) geometry/material preservation. (4) any new artifacts vs DS-0.3. "
            "(5) overall naturalness — is DS-0.4 closer to a real photo, or past the sweet spot? "
            "200 words max."
        ),
    }

    notes = {}
    for code, prompt in cell_prompts.items():
        img = PROBE / f"{code}.png"
        if not img.exists():
            notes[code] = f"(image not found: {img})"
            log_line(f"\n=== Tekton Vision per-cell read: {code} — SKIP (missing image) ===")
            continue
        log_line(f"\n=== Tekton Vision per-cell read: {code} ===")
        notes[code] = call_tekton_vision(prompt, [SHEET, img])

    # Header read — comparison sheet only
    header_prompt = (
        "4 architectural renders of the same scene at different depth ControlNet strengths "
        "(L→R: DS-0.1, DS-0.2, DS-0.3, DS-0.4). "
        "(a) which cell looks closest to a real architectural photo? "
        "(b) does the depth ControlNet change geometry or just texture? "
        "(c) is there a sweet-spot among these 4 cells? "
        "(d) does window lighting vary across cells? "
        "(e) any obvious over-constraint at the higher end? 300 words max."
    )
    log_line("\n=== Tekton Vision overall read ===")
    notes["__overall__"] = call_tekton_vision(header_prompt, [SHEET])

    # Write perceptual_notes.md
    md = [
        "# Tier C #2 Nudges — Perceptual Read (Qwen2.5-VL-7B / Tekton Vision)",
        "",
        f"**Endpoint:** `{ENDPOINT}`  ",
        f"**Comparison sheet:** `outputs/tier_c_2_nudges/comparison_sheet.png`  ",
        f"**Note:** per-cell reads compare each render against the 4-cell comparison sheet. "
        f"DS-0.3 is the anchor from the prior Tier C #2 probe (not re-rendered).",
        "",
        "## Overall read",
        "",
        notes["__overall__"],
        "",
    ]
    for code in ["DS-0.1", "DS-0.2", "DS-0.4"]:
        md.append(f"## {code}")
        md.append("")
        md.append(notes[code])
        md.append("")
    out = PROBE / "perceptual_notes.md"
    out.write_text("\n".join(md))
    log_line(f"\nWrote {out}")
    log_line(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
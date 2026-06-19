#!/usr/bin/env python3
"""Run Qwen2.5-VL-7B (Tekton Vision) over the 5-cell comparison sheet + per-cell reads.

Sequential per-image calls to avoid OOM. Image size is reduced before send.
"""
import base64
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = "https://psifunctiondev--tekton-vision-tektonvision-web.modal.run/v1/chat/completions"
PROBE = Path("/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs/tier_c_2_depth")
SHEET = PROBE / "comparison_sheet.png"

# Resize images to <= 1280 wide to keep prompt tokens manageable
MAX_W = 1280


def encode_resized(p: Path, max_w: int = MAX_W) -> str:
    img = Image.open(p).convert("RGB")
    if img.width > max_w:
        ratio = max_w / img.width
        new_h = int(img.height * ratio)
        img = img.resize((max_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_tekton_vision(prompt: str, image_paths: list, max_retries: int = 4) -> str:
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

    for attempt in range(max_retries):
        try:
            print(f"  POST {ENDPOINT} (attempt {attempt + 1})...")
            t0 = time.time()
            r = requests.post(ENDPOINT, json=body, timeout=300)
            elapsed = time.time() - t0
            if r.status_code == 200:
                j = r.json()
                if j.get("choices"):
                    txt = j["choices"][0]["message"]["content"]
                    if txt:
                        print(f"  ok in {elapsed:.1f}s, {len(txt)} chars")
                        return txt
                    else:
                        print(f"  empty content in {elapsed:.1f}s, retrying...")
                else:
                    print(f"  no choices in response: {j}")
            else:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
        time.sleep(8)
    return "ERROR: all retries failed"


def main() -> int:
    # Per-cell reads — single image + comparison sheet, prompt kept short
    cell_prompts = {
        "DS-0.3": (
            "Two images. Image 1 is a 5-cell comparison sheet (P2-anchor, CTRL, DS-0.3, DS-0.5, DS-0.7). "
            "Image 2 is the DS-0.3 cell (depth ControlNet at strength 0.3) in higher detail. "
            "Focus on DS-0.3 vs the P2 anchor: (1) geometry — does depth anchoring hold structure tighter? "
            "(2) right-edge noise — is the top-right corner less noisy? "
            "(3) any artifacts from the ControlNet — over-constrained geometry, depth-map seams, "
            "frozen features, unnatural flattening, copy-paste feel. 200 words max."
        ),
        "DS-0.5": (
            "Two images. Image 1 is the 5-cell comparison sheet. Image 2 is the DS-0.5 cell "
            "(depth ControlNet at strength 0.5) in higher detail. "
            "Focus on DS-0.5: does depth start to over-constrain here? Are edges/textures getting "
            "baked in or frozen? Is the look less 'real photo' than DS-0.3? 200 words max."
        ),
        "DS-0.7": (
            "Two images. Image 1 is the 5-cell comparison sheet. Image 2 is the DS-0.7 cell "
            "(depth ControlNet at strength 0.7) in higher detail. "
            "Focus on DS-0.7: does the depth ControlNet start to dominate (frozen depth layout, "
            "copy-paste feel, unnatural sharpness, geometry over-constraint)? 200 words max."
        ),
    }

    notes = {}
    for code, prompt in cell_prompts.items():
        img = PROBE / f"{code}.png"
        if not img.exists():
            notes[code] = f"(image not found: {img})"
            continue
        print(f"\n=== Tekton Vision per-cell read: {code} ===")
        notes[code] = call_tekton_vision(prompt, [SHEET, img])

    # Header read — comparison sheet only
    header_prompt = (
        "5 architectural renders, same input scene, different 'depth ControlNet' strengths. "
        "L→R: P2-anchor (v7.2 baseline), CTRL (depth=0.0 gate), DS-0.3, DS-0.5, DS-0.7. "
        "(a) which cell looks closest to a real architectural photo? "
        "(b) does the depth ControlNet change geometry or just texture? "
        "(c) is there a sweet-spot among the DS-* cells? "
        "(d) any obvious over-constraint at higher strengths? 300 words max."
    )
    print("\n=== Tekton Vision overall read ===")
    notes["__overall__"] = call_tekton_vision(header_prompt, [SHEET])

    # Write the perceptual_notes.md
    md = ["# Tier C #2 — Perceptual Read (Qwen2.5-VL-7B / Tekton Vision)", ""]
    md.append(f"**Endpoint:** `{ENDPOINT}`  ")
    md.append(f"**Comparison sheet:** `outputs/tier_c_2_depth/comparison_sheet.png`  ")
    md.append(f"**Note:** per-cell reads compare each render against the comparison sheet (P2 anchor + 4 new cells).")
    md.append("")
    md.append("## Overall read")
    md.append("")
    md.append(notes["__overall__"])
    md.append("")
    for code in ["DS-0.3", "DS-0.5", "DS-0.7"]:
        md.append(f"## {code}")
        md.append("")
        md.append(notes[code])
        md.append("")
    out = PROBE / "perceptual_notes.md"
    out.write_text("\n".join(md))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

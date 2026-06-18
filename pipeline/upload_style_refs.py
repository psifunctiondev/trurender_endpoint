#!/usr/bin/env python3
"""Upload Catherine's style reference photos to the TruRender Modal volume.

Usage:
    cd agents/tekton/modal
    /Users/doxa/Library/Python/3.9/bin/modal run upload_style_refs.py
"""
import base64
import os

import modal

VOLUME_NAME = "tekton-model-cache"
VOLUME_PATH = "/models"

volume = modal.Volume.from_name(VOLUME_NAME)

app = modal.App("trurender-upload-styles")

STYLE_REFS = [
    {
        "local_path": os.path.expanduser(
            "~/.openclaw/workspace/trurender_endpoint/outputs/best_photo_1_library_9.jpg"
        ),
        "volume_name": "style_ref_1_library.jpg",
    },
    {
        "local_path": os.path.expanduser(
            "~/.openclaw/workspace/trurender_endpoint/outputs/best_photo_3_staircase_dining_85.jpg"
        ),
        "volume_name": "style_ref_2_staircase.jpg",
    },
    {
        "local_path": os.path.expanduser(
            "~/.openclaw/workspace/trurender_endpoint/outputs/best_photo_5_kitchen_8.jpg"
        ),
        "volume_name": "style_ref_3_kitchen.jpg",
    },
]


@app.function(volumes={VOLUME_PATH: volume}, timeout=120)
def do_upload(images_b64: list[str], filenames: list[str]):
    style_dir = os.path.join(VOLUME_PATH, "style_references")
    os.makedirs(style_dir, exist_ok=True)

    for b64, fname in zip(images_b64, filenames):
        data = base64.b64decode(b64)
        path = os.path.join(style_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"  ✓ {fname} ({size_kb:.0f} KB)")

    volume.commit()
    print(f"\n{len(filenames)} style references committed to volume.")

    for f in sorted(os.listdir(style_dir)):
        fpath = os.path.join(style_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")


@app.local_entrypoint()
def main():
    images_b64 = []
    filenames = []
    for ref in STYLE_REFS:
        path = ref["local_path"]
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path, "rb") as f:
            data = f.read()
        images_b64.append(base64.b64encode(data).decode("utf-8"))
        filenames.append(ref["volume_name"])
        size_kb = len(data) / 1024
        print(f"  Read {ref['volume_name']} ({size_kb:.0f} KB)")

    if not images_b64:
        print("ERROR: No style reference images found!")
        return

    print(f"\nUploading {len(images_b64)} style references to Modal volume...")
    do_upload.remote(images_b64, filenames)
    print("Done!")

"""
upload_ctai_styles.py — Upload CTAI style references to the TruRender Modal volume.

Usage:
    # Upload all refs
    python upload_ctai_styles.py

    # Upload only refs matching a space type tag (e.g. kitchen, living, bath, exterior)
    python upload_ctai_styles.py --space kitchen

    # Dry run (show what would be uploaded, don't actually upload)
    python upload_ctai_styles.py --dry-run

    # Composite mode: blend top-N refs into a single style image for IP-Adapter
    python upload_ctai_styles.py --space kitchen --composite --top 4

This script reads style-refs.json and uploads the selected images to the Modal volume
under the style_references/ directory, renaming them with a ctai_ prefix so they coexist
with any other style refs already on the volume.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent.parent  # agents/tekton/modal/ -> workspace/
STYLE_REFS_DIR = WORKSPACE / "assets" / "trurender" / "style-refs"
MANIFEST_PATH = STYLE_REFS_DIR / "style-refs.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def select_refs(manifest: dict, space_type: str = None, top: int = None) -> list[dict]:
    """Select refs from manifest, optionally filtered and ranked by space type."""
    refs = manifest["refs"]
    tag_weights = manifest.get("tag_weights", {})

    if space_type and space_type in tag_weights:
        multipliers = tag_weights[space_type]
        scored = []
        for ref in refs:
            # Compute score: base weight × best matching tag multiplier
            best_mult = max(
                (multipliers.get(tag, 0.0) for tag in ref["tags"]),
                default=0.0
            )
            score = ref["weight"] * best_mult
            if score > 0:
                scored.append((score, ref))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [r for _, r in scored]
    else:
        # No filter — return all, sorted by weight
        selected = sorted(refs, key=lambda r: r["weight"], reverse=True)

    if top:
        selected = selected[:top]

    return selected


def composite_refs(selected: list[dict], style_refs_dir: Path) -> bytes:
    """
    Blend multiple style refs into a single image by tiling them side by side.
    Returns JPEG bytes. Requires Pillow.
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        print("ERROR: Pillow is required for composite mode. pip install Pillow", file=sys.stderr)
        sys.exit(1)

    images = []
    for ref in selected:
        img_path = style_refs_dir / ref["filename"]
        if not img_path.exists():
            print(f"WARNING: {img_path} not found, skipping")
            continue
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    if not images:
        print("ERROR: No images loaded for composite.", file=sys.stderr)
        sys.exit(1)

    # Resize all to same height (512px), tile horizontally
    target_h = 512
    resized = []
    for img in images:
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    total_w = sum(i.width for i in resized)
    composite = Image.new("RGB", (total_w, target_h))
    x = 0
    for img in resized:
        composite.paste(img, (x, 0))
        x += img.width

    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def upload_refs(selected: list[dict], style_refs_dir: Path, dry_run: bool = False):
    """Upload selected refs to the Modal volume via the trurender_comfyui module."""
    if not selected:
        print("No refs to upload.")
        return

    # Import the Modal app
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trurender_comfyui",
        SCRIPT_DIR / "trurender_comfyui.py"
    )
    mod = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(mod)

    items = []
    for ref in selected:
        img_path = style_refs_dir / ref["filename"]
        if not img_path.exists():
            print(f"WARNING: {img_path} not found, skipping")
            continue
        dest_name = f"ctai_{ref['filename']}"
        data = img_path.read_bytes()
        items.append({"name": dest_name, "data_b64": base64.b64encode(data).decode()})
        print(f"  {'[DRY RUN] ' if dry_run else ''}Queued: {dest_name} ({len(data)/1024:.0f} KB) — tags: {', '.join(ref['tags'])}")

    if dry_run:
        print(f"\n[DRY RUN] Would upload {len(items)} files to Modal volume style_references/")
        return

    if not items:
        print("Nothing to upload.")
        return

    print(f"\nUploading {len(items)} style references to Modal volume...")
    mod.upload_style_references.remote(style_data=items)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Upload CTAI style refs to TruRender Modal volume")
    parser.add_argument("--space", help="Filter by space type tag (kitchen, living, bath, exterior, entry, dining)")
    parser.add_argument("--top", type=int, help="Limit to top N refs by score")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    parser.add_argument("--composite", action="store_true", help="Blend selected refs into a single composite image")
    parser.add_argument("--list", action="store_true", help="List refs (with optional space filter) and exit")
    args = parser.parse_args()

    manifest = load_manifest()
    selected = select_refs(manifest, space_type=args.space, top=args.top)

    print(f"Style refs selected: {len(selected)}" + (f" (space={args.space})" if args.space else ""))
    for ref in selected:
        print(f"  {ref['filename']} — weight {ref['weight']} — tags: {', '.join(ref['tags'])}")
        print(f"    {ref['notes']}")

    if args.list:
        return

    if args.composite:
        print("\nBuilding composite style image...")
        composite_bytes = composite_refs(selected, STYLE_REFS_DIR)
        out_path = STYLE_REFS_DIR / f"ctai_composite{'_' + args.space if args.space else ''}.jpg"
        out_path.write_bytes(composite_bytes)
        print(f"Composite saved to {out_path} ({len(composite_bytes)/1024:.0f} KB)")
        print("You can upload this as style_image in a render call.")
        return

    upload_refs(selected, STYLE_REFS_DIR, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

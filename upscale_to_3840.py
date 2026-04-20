"""Upscale a 3072-max render to 3840 equivalent, preserving aspect ratio."""
import sys
from PIL import Image

def upscale(input_path, output_path, target_max_dim=3840):
    img = Image.open(input_path)
    w, h = img.size
    scale = target_max_dim / max(w, h)
    if scale <= 1.0:
        print(f"Already at or above target: {w}x{h}")
        img.save(output_path)
        return
    new_w = int(w * scale) // 16 * 16
    new_h = int(h * scale) // 16 * 16
    result = img.resize((new_w, new_h), Image.LANCZOS)
    result.save(output_path)
    print(f"Upscaled {w}x{h} -> {new_w}x{new_h}")

if __name__ == "__main__":
    upscale(sys.argv[1], sys.argv[2])

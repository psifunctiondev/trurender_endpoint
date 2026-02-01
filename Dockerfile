# syntax=docker/dockerfile:1.6

FROM runpod/worker-comfyui:5.7.1-flux1-dev

# Make bash the default shell for RUN and fail fast + print commands
SHELL ["/bin/bash", "-lc"]

# Basic build-time metadata + sanity checks
RUN set -euxo pipefail; \
    echo "===================="; \
    echo "[CHAT] Build reached first RUN step"; \
    echo "[CHAT] UTC time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"; \
    echo "[CHAT] Kernel: $(uname -a)"; \
    echo "[CHAT] OS release:"; (cat /etc/os-release || true); \
    echo "[CHAT] User: $(id)"; \
    echo "[CHAT] PWD: $(pwd)"; \
    echo "[CHAT] Disk usage:"; df -h; \
    echo "[CHAT] Memory:"; (free -h || true); \
    echo "[CHAT] Python:"; (python --version || true); \
    echo "[CHAT] Pip:"; (pip --version || true); \
    echo "===================="

# Show key ComfyUI model paths (verifies the base image contents you care about)
RUN set -euxo pipefail; \
    echo "===================="; \
    echo "[CHAT] ComfyUI model directory listing"; \
    for d in \
      /comfyui \
      /comfyui/models \
      /comfyui/models/unet \
      /comfyui/models/clip \
      /comfyui/models/vae \
    ; do \
      echo "--- $d"; \
      ls -la "$d" || true; \
    done; \
    echo "[CHAT] Expected baseline files (if present):"; \
    test -f /comfyui/models/unet/flux1-dev.safetensors && echo "OK: unet flux1-dev.safetensors" || echo "MISSING: unet flux1-dev.safetensors"; \
    test -f /comfyui/models/clip/clip_l.safetensors && echo "OK: clip clip_l.safetensors" || echo "MISSING: clip clip_l.safetensors"; \
    test -f /comfyui/models/clip/t5xxl_fp8_e4m3fn.safetensors && echo "OK: clip t5xxl_fp8_e4m3fn.safetensors" || echo "MISSING: clip t5xxl_fp8_e4m3fn.safetensors"; \
    test -f /comfyui/models/vae/ae.safetensors && echo "OK: vae ae.safetensors" || echo "MISSING: vae ae.safetensors"; \
    echo "===================="

# Optional: quick check that ComfyUI exists where the worker expects it
RUN set -euxo pipefail; \
    echo "===================="; \
    echo "[CHAT] ComfyUI install sanity"; \
    (ls -la /comfyui || true); \
    (python -c "import sys; print('[CHAT] python ok:', sys.version)" || true); \
    echo "===================="

# IMPORTANT: Do not override ENTRYPOINT/CMD.
# The base image defines what RunPod needs.

# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM ghcr.io/runpod-workers/worker-comfyui:5.5.1-base

# Install comfyui_controlnet_aux (provides DepthAnythingPreprocessor and other preprocessors)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

RUN pip install \
    --no-cache-dir \
    -r /comfyui/custom_nodes/comfyui_controlnet_aux/requirements.txt

# install custom nodes into comfyui (first node with --mode remote to fetch updated cache)
# Could not resolve custom node 'LoadImage' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'UNETLoader' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'VAELoader' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'DualCLIPLoader' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'CLIPTextEncode' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'CLIPTextEncode' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'ControlNetLoader' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'DepthAnythingPreprocessor' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'ControlNetApplyAdvanced' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'VAEEncode' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'KSampler' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'KSampler' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'VAEDecode' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'SaveImage' from unknown_registry (no aux_id provided)
# Could not resolve custom node 'Note' from unknown_registry (no aux_id provided)

# download models into comfyui
RUN comfy model download \
    --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors \
    --relative-path models/diffusion_models \
    --filename flux1-dev.safetensors

RUN comfy model download \
    --url https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors \
    --relative-path models/vae \
    --filename ae.safetensors

RUN comfy model download \
    --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors \
    --relative-path models/text_encoders \
    --filename t5xxl_fp16.safetensors

RUN comfy model download \
    --url https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_l.safetensors \
    --relative-path models/text_encoders \
    --filename clip_l.safetensors

RUN comfy model download \
    --url https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-depth-controlnet-v3.safetensors \
    --relative-path models/controlnet \
    --filename flux-depth-controlnet-v3.safetensors

RUN comfy model download \
    --url https://huggingface.co/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth \
    --relative-path models/annotators \
    --filename depth_anything_vitl14.pth


# copy all input data (like images or videos) into comfyui (uncomment and adjust if needed)
# COPY input/ /comfyui/input/

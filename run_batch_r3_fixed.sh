#!/bin/bash
# Batch render R3 blind test — renders at 3072 then upscales to 3840 locally
# This avoids the waffle artifact caused by the endpoint's 3840 render path

ENDPOINT="https://psifunctiondev--trurender-trurender-web.modal.run/render"
INPUT="/Users/doxa/.openclaw/workspace/trurender_endpoint/inputs/enscape_input.png"
OUTDIR="/Users/doxa/.openclaw/workspace/trurender_endpoint/outputs"
UPSCALE="/Users/doxa/.openclaw/workspace/trurender_endpoint/upscale_to_3840.py"

DEFAULT_PROMPT="professional architectural interior photograph, shot on Canon EOS R5 with 24mm tilt-shift lens, natural daylight through windows, physically accurate lighting, real-world materials and surface textures, subtle natural reflections on polished surfaces, correct scale and proportions, sharp architectural details, true-to-life colors and tones, high dynamic range, identical composition and layout to reference image, no changes to any furniture or materials or colors or objects, professional architectural photography, photorealistic, 8K detail"
CLEAN_SUFFIX=", clean pristine white fabrics, immaculate upholstery, no stains or discoloration"

render_one() {
    local LETTER=$1
    local DENOISE=$2
    local CANNY=$3
    local DEPTH=$4
    local CFG=$5
    local STEPS=$6
    local SEED=$7
    local DUAL=$8
    local PROMPT_MOD=$9

    local PROMPT="$DEFAULT_PROMPT"
    if [ "$PROMPT_MOD" = "clean_prompt" ]; then
        PROMPT="${PROMPT}${CLEAN_SUFFIX}"
    fi

    local TMPOUT="/tmp/r3_${LETTER}_3072.png"
    local FINALOUT="${OUTDIR}/v5_r3_blind_${LETTER}_fullres.png"

    echo "=== Rendering $LETTER (denoise=$DENOISE canny=$CANNY depth=$DEPTH cfg=$CFG steps=$STEPS seed=$SEED dual=$DUAL prompt=$PROMPT_MOD) ==="
    echo "Started: $(date)"

    curl -L -m 600 -s \
        -F "image=@${INPUT}" \
        -F "strength=${DENOISE}" \
        -F "controlnet_scale_canny=${CANNY}" \
        -F "controlnet_scale_depth=${DEPTH}" \
        -F "guidance_scale=${CFG}" \
        -F "num_steps=${STEPS}" \
        -F "seed=${SEED}" \
        -F "dual_control=${DUAL}" \
        -F "second_pass=true" \
        -F "max_dim=3072" \
        -F "prompt=${PROMPT}" \
        -o "${TMPOUT}" \
        -w "HTTP:%{http_code} SIZE:%{size_download} TIME:%{time_total}s\n" \
        "${ENDPOINT}"

    if [ $? -ne 0 ]; then
        echo "FAILED: curl error for $LETTER"
        return 1
    fi

    # Upscale to 3840
    python3 "${UPSCALE}" "${TMPOUT}" "${FINALOUT}"
    echo "Completed $LETTER: $(date)"
    echo ""
}

# Skip A - already done
# B: denoise=0.9, canny=1.3, depth=0.8, cfg=5.0, steps=28, seed=2718, dual=false, default
render_one B 0.9 1.3 0.8 5.0 28 2718 false default

# C: denoise=0.87, canny=1.0, depth=0.8, cfg=5.0, steps=28, seed=7919, dual=false, default
render_one C 0.87 1.0 0.8 5.0 28 7919 false default

# D: denoise=0.9, canny=1.0, depth=0.8, cfg=5.0, steps=28, seed=2718, dual=false, default
render_one D 0.9 1.0 0.8 5.0 28 2718 false default

# E: denoise=0.87, canny=1.0, depth=0.8, cfg=5.0, steps=28, seed=1337, dual=false, default
render_one E 0.87 1.0 0.8 5.0 28 1337 false default

# F: denoise=0.87, canny=1.0, depth=0.8, cfg=5.0, steps=36, seed=2718, dual=false, clean
render_one F 0.87 1.0 0.8 5.0 36 2718 false clean_prompt

# G: denoise=0.93, canny=1.5, depth=0.6, cfg=4.5, steps=36, seed=42, dual=false, clean
render_one G 0.93 1.5 0.6 4.5 36 42 false clean_prompt

# H: denoise=0.87, canny=1.0, depth=0.5, cfg=5.0, steps=28, seed=2718, dual=true, default
render_one H 0.87 1.0 0.5 5.0 28 2718 true default

# I: denoise=0.87, canny=1.0, depth=0.8, cfg=4.0, steps=28, seed=2718, dual=false, default
render_one I 0.87 1.0 0.8 4.0 28 2718 false default

# J: denoise=0.9, canny=1.0, depth=0.6, cfg=4.5, steps=36, seed=42, dual=false, clean
render_one J 0.9 1.0 0.6 4.5 36 42 false clean_prompt

# K: denoise=0.87, canny=1.0, depth=0.8, cfg=5.0, steps=28, seed=42, dual=false, default
render_one K 0.87 1.0 0.8 5.0 28 42 false default

# L: denoise=0.87, canny=1.0, depth=0.8, cfg=6.0, steps=28, seed=2718, dual=false, default
render_one L 0.87 1.0 0.8 6.0 28 2718 false default

echo "=== ALL RENDERS COMPLETE ==="
echo "Finished: $(date)"

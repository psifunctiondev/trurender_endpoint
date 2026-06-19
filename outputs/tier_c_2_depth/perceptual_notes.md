# Tier C #2 — Perceptual Read (Qwen2.5-VL-7B / Tekton Vision)

**Endpoint:** `https://psifunctiondev--tekton-vision-tektonvision-web.modal.run/v1/chat/completions`  
**Comparison sheet:** `outputs/tier_c_2_depth/comparison_sheet.png`  
**Note:** per-cell reads compare each render against the comparison sheet (P2 anchor + 4 new cells).

## Overall read

(a) The cell that looks closest to a real architectural photo is the one with the highest depth strength, which is DS-0.7. This cell shows the most realistic depth perception and lighting, with a clear sense of depth and a natural look to the scene.

(b) The depth ControlNet primarily affects the texture and lighting of the scene. It enhances the realism by adding depth cues and adjusting the lighting to match the perceived depth, rather than altering the geometry significantly. The changes are subtle but noticeable, especially in how shadows and highlights are rendered.

(c) There appears to be a sweet spot among the DS-* cells, which is around DS-0.5. This cell strikes a balance between realism and over-constraint, providing a good level of depth without losing the architectural details. The DS-0.3 cell is slightly less realistic, while DS-0.7 starts to show signs of over-constraint, with the scene appearing less natural and more artificial.

(d) Yes, there is an obvious over-constraint at higher strengths, particularly in the DS-0.7 cell. The scene becomes overly detailed and artificial, with exaggerated depth cues that make the image look less realistic. The DS-0.5 cell is the most balanced, providing a good level of depth without over-constraint. The DS-0.3 cell is less realistic but still maintains a good balance, making it a viable option for a more subtle depth effect.

## DS-0.3

Comparing the DS-0.3 cell (Image 2) with the P2 anchor (Image 1), the depth anchoring in DS-0.3 appears to hold the structure slightly tighter, as evidenced by the more defined and less distorted edges of the kitchen elements. The top-right corner of the DS-0.3 cell shows a reduction in noise, with smoother transitions and fewer artifacts compared to the P2 anchor. No significant over-constrained geometry or depth-map seams are apparent, suggesting that the ControlNet at strength 0.3 effectively guides the scene without causing unnatural flattening or a copy-paste feel. The DS-0.3 cell maintains a natural and cohesive appearance, with the depth anchoring enhancing the realism and detail of the kitchen environment.

## DS-0.5

The DS-0.5 cell in Image 2 appears to show a balance between depth and detail, but there is a subtle indication that depth might be starting to over-constrain the scene. The edges and textures, while still sharp and defined, seem to have a slightly less fluid appearance compared to DS-0.3. This could be due to the increased depth strength, which might be causing the model to prioritize depth over the fine details of the textures and edges. The overall look is still quite realistic, but it might not be as 'real photo' as DS-0.3, which seems to have a more natural and less constrained appearance. The DS-0.5 cell maintains a good level of realism, but the depth strength might be pushing the model towards a more structured, less organic look.

## DS-0.7

The DS-0.7 cell in Image 2, when compared to the other cells in the TruRender v7.2 comparison sheet, exhibits a noticeable shift in depth control. As the depth strength increases, the image becomes more heavily anchored to the depth information, leading to a "frozen depth layout" effect. This can result in a copy-paste feel, where details are less varied and more uniform across the scene. The unnatural sharpness and over-constrained geometry are evident, as the depth information dominates the rendering process, potentially leading to a loss of the nuanced interplay between light, shadow, and texture that is characteristic of a more dynamic depth control. The DS-0.7 cell shows a significant departure from the more flexible depth control seen in earlier cells, suggesting a shift towards a more rigid, depth-focused rendering approach.

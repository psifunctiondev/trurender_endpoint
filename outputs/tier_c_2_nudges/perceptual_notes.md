# Tier C #2 Nudges — Perceptual Read (Qwen2.5-VL-7B / Tekton Vision)

**Endpoint:** `https://psifunctiondev--tekton-vision-tektonvision-web.modal.run/v1/chat/completions`  
**Comparison sheet:** `outputs/tier_c_2_nudges/comparison_sheet.png`  
**Note:** per-cell reads compare each render against the 4-cell comparison sheet. DS-0.3 is the anchor from the prior Tier C #2 probe (not re-rendered).

## Overall read

(a) The cell with the highest depth strength (DS-0.4) appears closest to a real architectural photo. The depth and perspective are more pronounced, and the scene looks more three-dimensional, which is characteristic of high-quality architectural photography.

(b) The depth ControlNet primarily affects the geometry of the scene. As the depth strength increases, the objects in the scene appear to be more anchored in space, creating a more realistic depth perception. The texture and lighting remain relatively consistent, but the geometry becomes more defined.

(c) There is a sweet spot among these four cells, which is around DS-0.3. This is where the scene appears most realistic and balanced. The depth is sufficient to create a sense of space without being overly exaggerated, and the geometry is well-anchored without appearing stiff or unrealistic.

(d) The window lighting does not vary significantly across the cells. The lighting remains consistent, with natural light streaming in from the windows, creating a bright and inviting atmosphere. The differences are more subtle in terms of the depth and geometry rather than the lighting.

(e) There is no obvious over-constraint at the higher end (DS-0.4). The scene still looks realistic and well-rendered, with the depth and geometry appearing natural. The only noticeable difference is the increased depth and perspective, which enhances the realism of the scene without compromising the overall quality.

## DS-0.1

ERROR: all 4 attempts failed (see perceptual_read.log)

## DS-0.2

Comparing the DS-0.2 cell (Image 2) with the DS-0.3 anchor (Image 1), the window lighting consistency in DS-0.2 appears slightly less uniform, with a hint of shadowing that suggests a slight drift toward the more dynamic lighting of DS-0.4. DS-0.2 is closer to the anchor, maintaining a balance between the baseline and the more aggressive DS-0.4. The geometry and material stability in DS-0.2 are comparable to DS-0.3, with no noticeable tightening artifacts. The overall naturalness of DS-0.2 is slightly less refined than DS-0.3, indicating that it is not yet fully anchored but is close to the anchor's behavior.

## DS-0.4

Comparing DS-0.4 with DS-0.3, the DS-0.4 cell (depth ControlNet at strength 0.4) in Image 2 exhibits a noticeable shift in window lighting consistency. While DS-0.3 maintains a balanced and natural light distribution, DS-0.4 appears to have slightly more direct sunlight streaming through the windows, suggesting a potential over-constraint in depth control. This could lead to a "frozen depth layout" effect, where the scene feels less dynamic and more like a copy-paste image, potentially losing the subtle interplay of light and shadow that DS-0.3 captures.

In terms of geometry and material preservation, DS-0.4 seems to maintain a high level of detail, preserving the textures and shapes of the kitchen's elements, such as the marble countertops and wooden flooring. However, the increased depth strength might introduce new artifacts, such as unnatural sharpness or baked-in textures, which could detract from the overall realism.

Overall, DS-0.4 appears to be closer to a real photo in terms of naturalness, but it may be pushing towards the "sweet spot" where the depth control becomes too strong, leading to a less dynamic and potentially artificial-looking scene. The transition from DS-0.3 to DS-0.4 suggests a fine balance between depth control and naturalness, with DS-0.4 potentially over-constraining the depth, resulting in a scene that, while detailed, may lack the subtle nuances of a real-world environment.

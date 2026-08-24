# YCB-V multi-object pipeline

Generates 3D objects from a multi-object YCB-V photo and composes them back into
one metric scene. Also runs the single-shot merged-mask baseline for comparison.

Run everything from the repo root.

## Pipeline

```bash
# 1. one masked RGBA crop per object instance (BOP mask_visib as alpha,
#    using Pixal3D's own tight-bbox x1.1 square crop rule)
python scripts/ycb/ycb_prep.py --root datasets/ycbv --scene 48 --outdir work/s48

# 2. generate a mesh per crop. --fovs carries the exact per-crop FOV,
#    2*atan(S / 2*fx), derived from the scene's single shared intrinsic matrix.
python scripts/ycb/batch_infer.py \
    --images work/s48/*.png --fovs 0.355241,0.197151,... \
    --outdir work/glb --low_vram --resolution 1024

# 3. compose into a metric scene + score placement against BOP 6D-pose GT
python scripts/ycb/ycb_eval.py --scenedirs "work/s*" --glbdir work/glb \
    --outdir outputs/ycb_scenes

# 4. baseline: merge all instance masks into one object, generate in a single shot
python scripts/ycb/ycb_merged_prep.py --scenedirs "work/s*" --outdir work/merged
python scripts/ycb/batch_infer.py --images work/merged/*.png --fovs ... \
    --outdir work/merged_glb --low_vram --resolution 1024
python scripts/ycb/ycb_merged_eval.py --merged work/merged/merged.json \
    --mergedglb work/merged_glb --scenesdir outputs/ycb_scenes

# 5. novel-view render sheet for any composed scene
python scripts/ycb/render_scene.py outputs/ycb_scenes/scene_000048_000001/scene.glb out.png
```

## Coordinate conventions

Established empirically by reprojecting generated meshes through Pixal3D's own
camera model and matching silhouettes against the input crops:

```
exported GLB : image-right = -X, image-up = +Y, away-from-camera = +Z
GLB -> OpenCV camera : diag(-1, -1, +1)   (det = +1, a proper rotation)
OpenCV camera -> glTF: diag(+1, -1, -1)
```

The determinant check matters: silhouette matching alone cannot resolve the depth
sign (a front/back flip barely changes an outline), and the wrong choice silently
mirrors every object.

## Off-axis (ray) correction

`ProjGrid` assumes the principal point is the image centre
(`x_pixel = f*x/(-z) + resolution/2`, `image_conditioned_proj.py:96`) and a fixed
on-axis camera whose only free parameter is distance (`:212-215`). But
`preprocess_image` centres its crop on the *object's* bbox, so any off-centre
object yields a decentred window. On these scenes the true rays are tilted a
median of 7.16 deg (max 13.45 deg) off the optical axis.

`ycb_eval.py` compensates at composition time by rotating each object by the
minimal rotation carrying the optical axis onto its viewing ray. Measured over 44
objects: mask IoU 0.814 -> 0.835 overall, and +0.044 for the 13 objects beyond
8 deg off-axis versus +0.012 for the rest.

## Results (10 YCB-V scenes, 44 objects)

Per-object generation + composition, scored against BOP 6D-pose GT and CAD models.

| stage | file | ADD-S mean | <0.1*d |
|-------|------|-----------|--------|
| placement only            | `scene_no_ray_correction.glb` | -        | -   |
| + off-axis ray correction | `scene.glb`                   | 1.14 cm  | 82% |
| + per-object median lambda| `scene_final.glb`             | **0.92 cm** | **89%** |
| union-mask, no separation | `merged_single_shot.glb`      | -        | -   |

- **Ray correction**: mask IoU 0.814 -> 0.835. Gain concentrates where it should:
  +0.044 for the 13 objects beyond 8 deg off-axis vs +0.012 for the other 31.
- **lambda** (uniform scale about the camera centre, projection-invariant):
  median 0.9872, and 38/44 objects have lambda < 1 -- a systematic ~1.3% over-distance
  in the placement rule. ADD-S 1.14 -> 0.92 cm (-19.6%); the strict 0.05*d threshold
  moves most, 52% -> 73%. Median beats RANSAC (0.92 vs 0.94 cm) because outlier
  contamination is only ~28%, well inside the median's 50% breakdown point.
- **Occlusion is the dominant failure mode**: ADD-S 0.65 cm (visib>=0.9) vs 1.63 cm
  (visib<0.9); mask IoU 0.878 vs 0.721. lambda spread is 3x wider when occluded
  (std 0.014 vs 0.042), so the fit starts chasing noise -- the two worst regressions
  are both heavily occluded objects.
- **Merged (union-mask) baseline**: slightly better union silhouette (IoU 0.745 vs
  0.723) but it destroys instance structure -- 177 connected components for 44 true
  objects, 0/10 scenes recovering the right count. Not fusion, *fragmentation*.

## Relation to SAM 3D Objects

SAM 3D crops the same way: `crop_around_mask_with_padding(box_size_factor=1.0,
padding_factor=0.1)` then `rembg` (`image * mask`), i.e. tight bbox of the *visible*
mask, x1.1, squared, background blacked out - the same recipe as Pixal3D's
`preprocess_image`. Neither sees the occluder, so both must complete amodally from
the prior. SAM 3D does additionally accept an optional `pointmap` (geometry input);
Pixal3D has no equivalent and discards MoGe's depth, keeping only the focal length.

## Caveats

- Masks are BOP **ground truth**, so segmentation is free here. A real deployment
  needs SAM or similar and mask error would propagate into both crop and placement.
- YCB-V GT masks are themselves misaligned with the RGB by ~2 px (measured: 91% of
  objects need >= 2 px shift), which caps achievable IoU below 1.0.
- Objects below 0.9 visibility degrade sharply (IoU 0.721 vs 0.878, placement error
  2.2 cm vs 1.2 cm) - the visible-mask limitation this baseline cannot overcome.

"""Pick a YCB-V scene and emit one masked RGBA crop per visible object instance.

Writes crops/<scene>_<im>_<objid>_<idx>.png plus a crops.json manifest holding the
crop square in original-image pixels, which is what the placement step needs.
"""
import json, os, sys, argparse
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="dir containing test/<scene>/")
ap.add_argument("--scene", type=int, default=None)
ap.add_argument("--im", type=int, default=None)
ap.add_argument("--outdir", required=True)
ap.add_argument("--min_pixels", type=int, default=2500)
ap.add_argument("--min_visib_frac", type=float, default=0.35)
ap.add_argument("--max_objects", type=int, default=6)
args = ap.parse_args()

test = os.path.join(args.root, "test")
scenes = sorted(os.listdir(test))
scene = f"{args.scene:06d}" if args.scene is not None else scenes[0]
sdir = os.path.join(test, scene)

cam = json.load(open(os.path.join(sdir, "scene_camera.json")))
gt = json.load(open(os.path.join(sdir, "scene_gt.json")))
gti = json.load(open(os.path.join(sdir, "scene_gt_info.json")))

# choose the frame with the most sufficiently-visible instances, unless pinned
def score(imk):
    return sum(1 for g in gti[imk]
               if g["px_count_visib"] >= args.min_pixels and g["visib_fract"] >= args.min_visib_frac)

imk = str(args.im) if args.im is not None else max(gt.keys(), key=lambda k: (score(k), -int(k)))
print(f"scene={scene} im={imk}  usable instances={score(imk)}")

K = np.array(cam[imk]["cam_K"], float).reshape(3, 3)
depth_scale = cam[imk]["depth_scale"]
print(f"cam_K fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f} depth_scale={depth_scale}")

rgb_path = os.path.join(sdir, "rgb", f"{int(imk):06d}.png")
if not os.path.exists(rgb_path):
    rgb_path = os.path.join(sdir, "rgb", f"{int(imk):06d}.jpg")
rgb = np.array(Image.open(rgb_path).convert("RGB"))
depth = np.array(Image.open(os.path.join(sdir, "depth", f"{int(imk):06d}.png"))).astype(np.float32)
depth_m = depth * depth_scale / 1000.0  # BOP depth is in mm after depth_scale

os.makedirs(args.outdir, exist_ok=True)
manifest = {"scene": scene, "im": imk, "cam_K": K.tolist(), "depth_scale": depth_scale,
            "rgb": rgb_path, "image_size": [rgb.shape[1], rgb.shape[0]], "objects": []}

cands = []
for idx, (g, info) in enumerate(zip(gt[imk], gti[imk])):
    if info["px_count_visib"] < args.min_pixels or info["visib_fract"] < args.min_visib_frac:
        continue
    cands.append((info["px_count_visib"], idx, g, info))
cands.sort(reverse=True)
cands = cands[: args.max_objects]

for _, idx, g, info in cands:
    mp = os.path.join(sdir, "mask_visib", f"{int(imk):06d}_{idx:06d}.png")
    m = np.array(Image.open(mp)) > 0
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    S = int(max(x1 - x0, y1 - y0) * 1.1)          # same 1.1 margin preprocess_image uses
    box = (int(cx - S // 2), int(cy - S // 2), int(cx - S // 2) + S, int(cy - S // 2) + S)

    rgba = np.dstack([rgb, (m * 255).astype(np.uint8)])
    crop = Image.fromarray(rgba).crop(box)        # PIL pads out-of-bounds with zeros
    name = f"{scene}_{int(imk):06d}_obj{g['obj_id']:02d}_i{idx}"
    crop.save(os.path.join(args.outdir, name + ".png"))

    d = depth_m[m]
    d = d[d > 1e-6]
    manifest["objects"].append({
        "name": name, "obj_id": g["obj_id"], "inst_idx": idx,
        "crop_box": list(map(int, box)), "crop_size": S,
        "crop_center": [float(cx), float(cy)],
        "px_count_visib": info["px_count_visib"], "visib_fract": info["visib_fract"],
        "depth_front_m": float(np.percentile(d, 5)) if d.size else None,
        "depth_med_m": float(np.median(d)) if d.size else None,
        "cam_R_m2c": g["cam_R_m2c"], "cam_t_m2c": g["cam_t_m2c"],  # GT, for comparison only
    })
    print(f"  obj{g['obj_id']:02d} i{idx}: S={S}px visib={info['visib_fract']:.2f} "
          f"depth_front={manifest['objects'][-1]['depth_front_m']:.3f}m")

json.dump(manifest, open(os.path.join(args.outdir, "crops.json"), "w"), indent=2)
print(f"\nwrote {len(manifest['objects'])} crops -> {args.outdir}")

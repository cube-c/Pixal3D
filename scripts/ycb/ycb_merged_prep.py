"""Build ONE union-mask crop per scene: all selected instances merged into a single
object, so Pixal3D generates the whole group in one shot (no per-object separation)."""
import json, glob, math, os, argparse
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--scenedirs", required=True)
ap.add_argument("--root", default="datasets/ycbv")
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

manifest = []
for d in sorted(glob.glob(a.scenedirs)):
    man = json.load(open(f"{d}/crops.json"))
    K = np.array(man["cam_K"], float); fx = K[0, 0]
    sdir = os.path.join(a.root, "test", man["scene"]); imk = int(man["im"])
    rgb = np.array(Image.open(man["rgb"]).convert("RGB")); H, W = rgb.shape[:2]

    # union of exactly the instances we generated per-object, for a fair comparison
    union = np.zeros((H, W), bool)
    for o in man["objects"]:
        m = np.array(Image.open(os.path.join(sdir, "mask_visib",
                     f"{imk:06d}_{o['inst_idx']:06d}.png"))) > 0
        union |= m

    ys, xs = np.nonzero(union)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    S = int(max(x1 - x0, y1 - y0) * 1.1)                  # same rule preprocess_image uses
    box = (int(cx - S // 2), int(cy - S // 2), int(cx - S // 2) + S, int(cy - S // 2) + S)

    rgba = np.dstack([rgb, (union * 255).astype(np.uint8)])
    name = f"{man['scene']}_{imk:06d}_MERGED"
    Image.fromarray(rgba).crop(box).save(os.path.join(a.outdir, name + ".png"))

    depth = np.array(Image.open(os.path.join(sdir, "depth", f"{imk:06d}.png"))).astype(np.float32)
    dm = depth * man["depth_scale"] / 1000.0
    dv = dm[union]; dv = dv[dv > 1e-6]
    fov = 2 * math.atan(S / (2 * fx))
    manifest.append(dict(name=name, scene=man["scene"], im=man["im"], cam_K=man["cam_K"],
                         rgb=man["rgb"], image_size=man["image_size"],
                         crop_box=list(map(int, box)), crop_size=S,
                         crop_center=[float(cx), float(cy)], fov=fov,
                         n_objects=len(man["objects"]),
                         depth_front_m=float(np.percentile(dv, 5)),
                         depth_med_m=float(np.median(dv)),
                         inst_idxs=[o["inst_idx"] for o in man["objects"]]))
    print(f"{man['scene']}: {len(man['objects'])} objects merged -> S={S}px  "
          f"FOV={math.degrees(fov):.2f}deg  depth {manifest[-1]['depth_front_m']:.3f}m")

json.dump(manifest, open(os.path.join(a.outdir, "merged.json"), "w"), indent=2)
print(f"\nwrote {len(manifest)} merged crops -> {a.outdir}")

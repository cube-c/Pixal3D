"""Export the ground-truth scene as GLB: YCB CAD models at their BOP 6D poses.

Same camera frame and same glTF convention as scene_final.glb, and restricted to
the same instances, so the two GLBs can be loaded together and compared directly.
"""
import json, glob, os, argparse
import numpy as np, trimesh
from PIL import Image

CV2GLTF = np.diag([1., -1., -1.])   # OpenCV camera -> glTF world

ap = argparse.ArgumentParser()
ap.add_argument("--scenedirs", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--models", default="/app/Pixal3D/datasets/ycbv/models")
ap.add_argument("--name", default="scene_gt.glb")
a = ap.parse_args()

for d in sorted(glob.glob(a.scenedirs)):
    man = json.load(open(f"{d}/crops.json"))
    sc, imk = man["scene"], int(man["im"])
    od = os.path.join(a.outdir, f"scene_{sc}_{imk:06d}")
    if not os.path.isdir(od):
        continue
    scene = trimesh.Scene()
    for o in man["objects"]:
        cad = trimesh.load(os.path.join(a.models, f"obj_{o['obj_id']:06d}.ply"))
        # attach the model's texture image if trimesh did not pick it up
        tex = os.path.join(a.models, f"obj_{o['obj_id']:06d}.png")
        try:
            if getattr(cad.visual, "uv", None) is not None and os.path.exists(tex):
                cad.visual.material = trimesh.visual.material.SimpleMaterial(
                    image=Image.open(tex).convert("RGB"))
        except Exception:
            pass
        V = np.asarray(cad.vertices, float) / 1000.0           # BOP models are mm
        R = np.array(o["cam_R_m2c"], float).reshape(3, 3)
        t = np.array(o["cam_t_m2c"], float) / 1000.0           # BOP translation is mm
        cad.vertices = (V @ R.T + t) @ CV2GLTF.T
        scene.add_geometry(cad, node_name=f"gt_{o['name']}")
    out = os.path.join(od, a.name)
    scene.export(out)
    n = len(man["objects"])
    print(f"{sc}: {n} GT objects -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

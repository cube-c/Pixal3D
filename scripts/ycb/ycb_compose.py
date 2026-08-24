"""Place per-object Pixal3D GLBs back into one metric scene using YCB-V depth + K.

Frame conventions (established empirically, see README notes):
  exported GLB : image-right = -X, image-up = +Y, away-from-camera = +Z
  OpenCV camera: X right, Y down, Z away    ->  cam = diag(-1,-1,+1) @ G   (det=+1)
  glTF world   : X right, Y up,  -Z forward ->  gltf = diag(1,-1,-1) @ cam
"""
import json, os, sys, argparse
import numpy as np, trimesh
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--crops", required=True, help="crops.json from ycb_prep.py")
ap.add_argument("--glbdir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--overlay", default=None, help="write a reprojection overlay PNG here")
args = ap.parse_args()

man = json.load(open(args.crops))
K = np.array(man["cam_K"], float)
fx, fy, ux, uy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

GLB_TO_CV = np.diag([-1.0, -1.0, 1.0])
CV_TO_GLTF = np.diag([1.0, -1.0, -1.0])

scene = trimesh.Scene()
placed = []
for o in man["objects"]:
    p = os.path.join(args.glbdir, o["name"] + ".glb")
    if not os.path.exists(p):
        print(f"  !! missing {p}"); continue
    g = trimesh.load(p)
    mesh = trimesh.util.concatenate(list(g.geometry.values())) if hasattr(g, "geometry") else g
    G = np.asarray(mesh.vertices, float)

    # mesh extent in the image plane (GLB X,Y) <-> object tight bbox = crop_size/1.1 px
    ext = max(np.ptp(G[:, 0]), np.ptp(G[:, 1]))
    px_extent = o["crop_size"] / 1.1
    Zref = o["depth_med_m"]
    s = (px_extent * Zref / fx) / max(ext, 1e-9)          # metres per GLB unit

    V = G @ GLB_TO_CV.T * s                                # camera-frame, metric, centred at 0
    # anchor the nearest surface to the observed front depth
    Z0 = o["depth_front_m"] - V[:, 2].min()
    cxg, cyg = (V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2
    czg = (V[:, 2].min() + V[:, 2].max()) / 2
    Zc = Z0 + czg
    X0 = (o["crop_center"][0] - ux) * Zc / fx - cxg
    Y0 = (o["crop_center"][1] - uy) * Zc / fy - cyg

    V = V + np.array([X0, Y0, Z0])
    mesh.vertices = V @ CV_TO_GLTF.T
    scene.add_geometry(mesh, node_name=o["name"])
    placed.append((o, V))
    size_cm = (np.ptp(V, axis=0) * 100).round(1)
    print(f"  {o['name']}: scale={s:.4f} m/unit  centre_depth={Zc:.3f}m  size={size_cm}cm")

scene.export(args.out)
print(f"\nwrote {args.out}  ({len(placed)} objects)")

if args.overlay and placed:
    img = Image.open(man["rgb"]).convert("RGB")
    dr = ImageDraw.Draw(img)
    colors = [(255,80,80),(80,255,80),(80,160,255),(255,220,60),(255,80,255),(80,255,255)]
    for i, (o, V) in enumerate(placed):
        sub = V[np.random.default_rng(0).choice(len(V), min(len(V), 4000), replace=False)]
        u = fx * sub[:, 0] / sub[:, 2] + ux
        v = fy * sub[:, 1] / sub[:, 2] + uy
        for a, b in zip(u, v):
            dr.point((a, b), fill=colors[i % len(colors)])
    img.save(args.overlay)
    print(f"wrote overlay {args.overlay}")

"""Compare lambda strategies by ADD-S: none / median / RANSAC.

mask_visib already excludes pixels occluded by other objects. What it does NOT
exclude are outliers: the ~2px YCB mask-vs-RGB misalignment dragging background
depth inside the mask, depth-sensor mixed pixels at occlusion edges, and regions
where the generated shape is simply wrong.

lambda is a 1-parameter fit -- uniform scale about the camera centre, which is
projection-invariant -- so the RANSAC minimal sample is a single pixel and the
inlier test is on the depth residual |d_obs - lambda * z_render|.
"""
import json, glob, os, argparse
import numpy as np, trimesh
from scipy.spatial import cKDTree
from PIL import Image

GLB2CV = np.diag([-1., -1., 1.])


def ray_rotation(dx, dy):
    d = np.array([dx, dy, 1.0]); d /= np.linalg.norm(d)
    z = np.array([0., 0., 1.]); v = np.cross(z, d); s = np.linalg.norm(v); c = float(z @ d)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def zbuffer(V, F, K, shape):
    fx, fy, ux, uy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; H, W = shape
    z = np.full((H, W), np.inf); tri = V[F]
    for k in range(3):
        p = tri[:, k, :]; ok = p[:, 2] > 1e-6
        u = (fx * p[ok, 0] / p[ok, 2] + ux).astype(int)
        v = (fy * p[ok, 1] / p[ok, 2] + uy).astype(int)
        dd = p[ok, 2]; m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        np.minimum.at(z, (v[m], u[m]), dd[m])
    return z


def ransac_lambda(z, d, tau, iters, rng):
    """Robust fit of d ~= lambda*z. Minimal sample = 1 pixel, refit on inliers."""
    n = len(z)
    if n < 20:
        return float(np.median(d / z)), 1.0
    best_in, best_lam = None, float(np.median(d / z))
    for i in rng.integers(0, n, size=min(iters, n)):
        if z[i] <= 1e-6:
            continue
        lam = d[i] / z[i]
        inl = np.abs(d - lam * z) < tau
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_lam = inl, lam
    if best_in is not None and best_in.sum() >= 20:
        zz, dd = z[best_in], d[best_in]
        return float((zz @ dd) / (zz @ zz)), float(best_in.mean())   # LSQ on inliers
    return best_lam, 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--scenedirs", required=True); ap.add_argument("--glbdir", required=True)
ap.add_argument("--models", required=True); ap.add_argument("--root", default="/app/Pixal3D/datasets/ycbv")
ap.add_argument("--npts", type=int, default=6000)
ap.add_argument("--tau", type=float, default=0.010, help="inlier threshold on depth residual (m)")
ap.add_argument("--iters", type=int, default=300)
a = ap.parse_args()
info = json.load(open(os.path.join(a.models, "models_info.json")))

recs = []
for d_ in sorted(glob.glob(a.scenedirs)):
    man = json.load(open(f"{d_}/crops.json")); K = np.array(man["cam_K"], float)
    fx, fy, ux, uy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    sdir = os.path.join(a.root, "test", man["scene"]); imk = int(man["im"])
    H, W = man["image_size"][1], man["image_size"][0]
    dm = np.array(Image.open(os.path.join(sdir, "depth", f"{imk:06d}.png"))).astype(np.float32) * man["depth_scale"] / 1000.0
    for o in man["objects"]:
        p = os.path.join(a.glbdir, o["name"] + ".glb")
        if not os.path.exists(p):
            continue
        g = trimesh.load(p); mesh = trimesh.util.concatenate(list(g.geometry.values()))
        G = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces)
        Fs = F[np.random.default_rng(0).choice(len(F), min(len(F), 60000), replace=False)]
        ext = max(np.ptp(G[:, 0]), np.ptp(G[:, 1]))
        s = ((o["crop_size"] / 1.1) * o["depth_med_m"] / fx) / max(ext, 1e-9)
        V = G @ GLB2CV.T * s
        V = V @ ray_rotation((o["crop_center"][0] - ux) / fx, (o["crop_center"][1] - uy) / fy).T
        Z0 = o["depth_front_m"] - V[:, 2].min()
        cx_ = (V[:, 0].min() + V[:, 0].max()) / 2; cy_ = (V[:, 1].min() + V[:, 1].max()) / 2
        cz_ = (V[:, 2].min() + V[:, 2].max()) / 2; Zc = Z0 + cz_
        V = V + np.array([(o["crop_center"][0] - ux) * Zc / fx - cx_,
                          (o["crop_center"][1] - uy) * Zc / fy - cy_, Z0])

        mv = np.array(Image.open(os.path.join(sdir, "mask_visib", f"{imk:06d}_{o['inst_idx']:06d}.png"))) > 0
        zr = zbuffer(V, Fs, K, (H, W))
        base = mv & np.isfinite(zr) & (dm > 1e-6)
        if base.sum() >= 50:
            lam_med = float(np.median(dm[base] / zr[base]))
            lam_rs, inl = ransac_lambda(zr[base].astype(float), dm[base].astype(float),
                                        a.tau, a.iters, np.random.default_rng(0))
        else:
            lam_med, lam_rs, inl = 1.0, 1.0, 0.0

        cad = trimesh.load(os.path.join(a.models, f"obj_{o['obj_id']:06d}.ply"))
        Q = cad.sample(a.npts) / 1000.0
        R = np.array(o["cam_R_m2c"], float).reshape(3, 3); t = np.array(o["cam_t_m2c"], float) / 1000.0
        Q = Q @ R.T + t

        def adds(lmb):
            m2 = mesh.copy(); m2.vertices = V * lmb
            return float(cKDTree(m2.sample(a.npts)).query(Q)[0].mean()) * 100

        recs.append(dict(scene=man["scene"], obj=o["obj_id"], visib=o["visib_fract"],
                         diam=info[str(o["obj_id"])]["diameter"] / 10.0,
                         lam_med=lam_med, lam_rs=lam_rs, inlier=inl, px=int(base.sum()),
                         a_none=adds(1.0), a_med=adds(lam_med), a_rs=adds(lam_rs)))
        r = recs[-1]
        print(f"  {man['scene']} obj{o['obj_id']:02d} v={o['visib_fract']:.2f} "
              f"lam med {lam_med:.4f} ransac {lam_rs:.4f} (inl {inl:.0%} of {base.sum()})  "
              f"ADD-S none {r['a_none']:.2f} med {r['a_med']:.2f} rs {r['a_rs']:.2f}")

by = {}
for r in recs:
    by.setdefault(r["scene"], []).append(r["lam_rs"])
json.dump(dict(per_object=recs, per_scene={k: float(np.median(v)) for k, v in by.items()}),
          open("/app/Pixal3D/outputs/ycb_scenes/lambda_variants.json", "w"), indent=2)

print(f"\n=== {len(recs)} objects ===")
dia = np.array([r["diam"] for r in recs])
for k, lbl in (("a_none", "lambda = 1 (no fit)"), ("a_med", "per-object lambda, median"),
               ("a_rs", "per-object lambda, RANSAC")):
    v = np.array([r[k] for r in recs]); rt = v / dia
    print(f"  {lbl:32s} ADD-S mean {v.mean():.2f} cm  median {np.median(v):.2f} cm  <0.1d {(rt<0.1).mean():.0%}")
vi = np.array([r["visib"] for r in recs]); il = np.array([r["inlier"] for r in recs])
for k, lbl in (("a_none", "none"), ("a_med", "median"), ("a_rs", "RANSAC")):
    v = np.array([r[k] for r in recs])
    print(f"  {lbl:8s} visib>=0.9 {v[vi>=0.9].mean():.2f} cm | visib<0.9 {v[vi<0.9].mean():.2f} cm")
print(f"  RANSAC inlier fraction: mean {il.mean():.0%}  min {il.min():.0%}")

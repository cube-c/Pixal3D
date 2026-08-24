"""Tiny z-buffer renderer: orbit the composed scene to show it is real 3D."""
import sys, numpy as np, trimesh
from PIL import Image

path, out = sys.argv[1], sys.argv[2]
RES = 640
scene = trimesh.load(path)

geoms = []
for g in scene.geometry.values():
    V = np.asarray(g.vertices, float); F = np.asarray(g.faces)
    # per-vertex colour from the baseColor texture via UVs
    C = np.full((len(V), 3), 180, np.uint8)
    try:
        uv = np.asarray(g.visual.uv)
        tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
        h, w = tex.shape[:2]
        px = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        py = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        C = tex[py, px]
    except Exception as e:
        print("  (no texture:", e, ")")
    geoms.append((V, F, C))

allV = np.vstack([v for v, _, _ in geoms])
centre = (allV.min(0) + allV.max(0)) / 2
radius = np.linalg.norm(allV.max(0) - allV.min(0)) / 2

def render(yaw, pitch, fname):
    cy, sy = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R = Rx @ Ry
    eye_d = radius * 3.0
    f = RES * 0.9
    img = np.zeros((RES, RES, 3), np.uint8); img[:] = (26, 28, 34)
    zbuf = np.full((RES, RES), 1e9)
    for V, F, C in geoms:
        P = (V - centre) @ R.T
        P[:, 2] += eye_d
        ok = P[:, 2] > 1e-3
        u = f * P[:, 0] / np.maximum(P[:, 2], 1e-3) + RES / 2
        v = -f * P[:, 1] / np.maximum(P[:, 2], 1e-3) + RES / 2
        # splat vertices with a z-test (dense meshes -> effectively a point render)
        gi = np.clip(u.astype(int), 0, RES - 1); gj = np.clip(v.astype(int), 0, RES - 1)
        order = np.argsort(-P[:, 2])           # far to near
        gi, gj, z, c, ok = gi[order], gj[order], P[order, 2], C[order], ok[order]
        m = ok
        for dj in (0, 1):
            for di in (0, 1):
                jj = np.clip(gj[m] + dj, 0, RES - 1); ii = np.clip(gi[m] + di, 0, RES - 1)
                sel = z[m] < zbuf[jj, ii]
                zbuf[jj[sel], ii[sel]] = z[m][sel]
                img[jj[sel], ii[sel]] = c[m][sel]
    Image.fromarray(img).save(fname)
    return fname

views = [(0.0, 0.25, "front"), (0.7, 0.35, "right"), (-0.7, 0.35, "left"), (0.0, 1.1, "top")]
tiles = []
for yaw, pitch, name in views:
    tiles.append(render(yaw, pitch, f"/tmp/_v_{name}.png"))
    print("  rendered", name)
sheet = Image.new("RGB", (RES * 2, RES * 2))
for k, t in enumerate(tiles):
    sheet.paste(Image.open(t), ((k % 2) * RES, (k // 2) * RES))
sheet.save(out)
print("wrote", out)

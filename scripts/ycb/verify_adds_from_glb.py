"""Verify the reported ADD-S by recomputing it from the exported GLB files.

The headline numbers came from lambda_variants.py, which applies placement inline
and never opens the GLBs. This reads scene_final.glb and scene_gt.glb off disk
instead and checks the two agree.
"""
import json, glob, os
import numpy as np, trimesh
from scipy.spatial import cKDTree

SP = "/tmp/claude-0/-app-Pixal3D/e89d816d-465e-4b7d-bb3e-144d1d81853a/scratchpad"
OUT = "/app/Pixal3D/outputs/ycb_scenes"
ref = {}
for r in json.load(open(f"{OUT}/lambda_variants.json"))["per_object"]:
    ref.setdefault(r["scene"], []).append(r)

NP = 6000
rows = []
for d in sorted(glob.glob(f"{SP}/ycb10/s*")):
    man = json.load(open(f"{d}/crops.json")); sc, imk = man["scene"], int(man["im"])
    od = f"{OUT}/scene_{sc}_{imk:06d}"
    fp, gp = f"{od}/scene_final.glb", f"{od}/scene_gt.glb"
    if not (os.path.exists(fp) and os.path.exists(gp)):
        continue
    fin = list(trimesh.load(fp).geometry.values())      # order == manifest order
    gtm = list(trimesh.load(gp).geometry.values())
    if len(fin) != len(gtm) or len(fin) != len(man["objects"]):
        print(f"  !! {sc}: {len(fin)} final / {len(gtm)} gt / {len(man['objects'])} manifest"); continue
    for i, o in enumerate(man["objects"]):
        P = fin[i].sample(NP)                           # generated, as exported
        Q = gtm[i].sample(NP)                           # CAD @ GT pose, as exported
        a_glb = float(cKDTree(P).query(Q)[0].mean()) * 100
        a_ref = ref[sc][i]["a_med"]
        rows.append((sc, o["obj_id"], a_glb, a_ref))

g = np.array([r[2] for r in rows]); v = np.array([r[3] for r in rows])
print(f"recomputed from GLB files vs reported, n={len(rows)}\n")
print(f"  mean ADD-S  from GLBs {g.mean():.3f} cm   reported {v.mean():.3f} cm")
print(f"  median      from GLBs {np.median(g):.3f} cm   reported {np.median(v):.3f} cm")
print(f"  max |difference| per object: {np.abs(g-v).max():.4f} cm")
print(f"  mean |difference|          : {np.abs(g-v).mean():.4f} cm")
bad = [r for r in rows if abs(r[2]-r[3]) > 0.05]
print(f"  objects differing by >0.05 cm: {len(bad)}")
for r in bad[:5]:
    print(f"    {r[0]} obj{r[1]:02d}: glb {r[2]:.3f} vs reported {r[3]:.3f}")

"""Fit ONE lambda (uniform scale about the camera centre) to a whole composed scene.

Projection-invariant, so the scene's silhouette never moves; only depth does.
Scored by depth RMSE/MAE against the sensor depth map over the union GT mask -
the quantity lambda actually minimises.
"""
import json, glob, os, argparse
import numpy as np, trimesh
from PIL import Image

CV2GLTF = np.diag([1.,-1.,-1.])

def zbuffer(V,F,K,shape):
    fx,fy,ux,uy=K[0,0],K[1,1],K[0,2],K[1,2]; H,W=shape
    z=np.full((H,W),np.inf)
    tri=V[F]
    for k in range(3):
        p=tri[:,k,:]; ok=p[:,2]>1e-6
        u=(fx*p[ok,0]/p[ok,2]+ux).astype(int); v=(fy*p[ok,1]/p[ok,2]+uy).astype(int)
        dd=p[ok,2]; m=(u>=0)&(u<W)&(v>=0)&(v<H)
        np.minimum.at(z,(v[m],u[m]),dd[m])
    return z

ap=argparse.ArgumentParser()
ap.add_argument("--scenesdir",required=True); ap.add_argument("--scenedirs",required=True)
ap.add_argument("--root",default="/app/Pixal3D/datasets/ycbv"); ap.add_argument("--write",action="store_true")
a=ap.parse_args()

rows=[]
for d in sorted(glob.glob(a.scenedirs)):
    man=json.load(open(f"{d}/crops.json")); K=np.array(man["cam_K"],float)
    imk=int(man["im"]); sdir=os.path.join(a.root,"test",man["scene"])
    sp=os.path.join(a.scenesdir,f"scene_{man['scene']}_{imk:06d}","scene.glb")
    if not os.path.exists(sp): continue
    H,W=man["image_size"][1],man["image_size"][0]
    dm=np.array(Image.open(os.path.join(sdir,"depth",f"{imk:06d}.png"))).astype(np.float32)*man["depth_scale"]/1000.0
    union=np.zeros((H,W),bool)
    for o in man["objects"]:
        union|=np.array(Image.open(os.path.join(sdir,"mask_visib",f"{imk:06d}_{o['inst_idx']:06d}.png")))>0

    sc=trimesh.load(sp)
    # instance-aware: render each object separately and pair its depth ONLY with
    # the pixels where that same object is genuinely visible (its own mask_visib).
    # A whole-scene z-buffer over the union mask would pair object B's sensor depth
    # against object A's reconstructed surface wherever placement error overlaps them.
    geo=list(sc.geometry.values())
    # GLB export drops node names (they become geometry_0..N) but preserves the
    # order they were added, which is manifest order.
    if len(geo)!=len(man["objects"]):
        print(f"  !! {man['scene']}: {len(geo)} geoms vs {len(man['objects'])} objects, skipping"); continue
    ratios=[]; zr_all=np.full((H,W),np.inf); sel=np.zeros((H,W),bool)
    for oi,o in enumerate(man["objects"]):
        gm=geo[oi]
        Vi=np.asarray(gm.vertices,float)@np.linalg.inv(CV2GLTF).T
        Fi=np.asarray(gm.faces)
        Fi=Fi[np.random.default_rng(0).choice(len(Fi),min(len(Fi),60000),replace=False)]
        zi=zbuffer(Vi,Fi,K,(H,W))
        mv=np.array(Image.open(os.path.join(sdir,"mask_visib",
                    f"{imk:06d}_{o['inst_idx']:06d}.png")))>0
        si=mv & np.isfinite(zi) & (dm>1e-6)
        if si.sum()<50: continue
        ratios.append(dm[si]/zi[si])
        zr_all[si]=zi[si]; sel |= si
    if not ratios: continue
    zr=zr_all
    lam=float(np.median(np.concatenate(ratios)))
    e0=dm[sel]-zr[sel]; e1=dm[sel]-lam*zr[sel]
    rows.append(dict(scene=man["scene"],lam=lam,px=int(sel.sum()),
        rmse0=float(np.sqrt((e0**2).mean()))*100, rmse1=float(np.sqrt((e1**2).mean()))*100,
        mae0=float(np.abs(e0).mean())*100, mae1=float(np.abs(e1).mean())*100,
        bias0=float(e0.mean())*100, bias1=float(e1.mean())*100))
    r=rows[-1]
    print(f"{man['scene']}  lambda={lam:.4f}  depth MAE {r['mae0']:5.2f} -> {r['mae1']:5.2f} cm | "
          f"RMSE {r['rmse0']:5.2f} -> {r['rmse1']:5.2f} cm | bias {r['bias0']:+5.2f} -> {r['bias1']:+5.2f} cm")
    if a.write:
        for g in sc.geometry.values():
            g.vertices = (np.asarray(g.vertices,float)@np.linalg.inv(CV2GLTF).T*lam)@CV2GLTF.T
        sc.export(os.path.join(os.path.dirname(sp),"scene_depth_fitted.glb"))

l=np.array([r["lam"] for r in rows])
for k in ("mae","rmse"):
    b=np.array([r[k+"0"] for r in rows]); af=np.array([r[k+"1"] for r in rows])
    print(f"\n{k.upper():5s} mean {b.mean():.2f} -> {af.mean():.2f} cm   improved {(af<b).sum()}/{len(rows)}")
print(f"lambda median {np.median(l):.4f}  range {l.min():.3f}..{l.max():.3f}")
json.dump(rows,open("/app/Pixal3D/outputs/ycb_scenes/scene_depth_fit.json","w"),indent=2)

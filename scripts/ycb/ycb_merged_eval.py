"""Head-to-head: one merged whole-group mesh vs the composed per-object scene.

Scores both by reprojected silhouette IoU against the UNION of GT visible masks,
and counts connected components to test whether objects remain separable.
"""
import json, os, glob, argparse
import numpy as np, trimesh
from PIL import Image, ImageDraw

GLB2CV, CV2GLTF = np.diag([-1.,-1.,1.]), np.diag([1.,-1.,-1.])

def ray_rotation(dx, dy):
    d = np.array([dx, dy, 1.0]); d /= np.linalg.norm(d)
    z = np.array([0.,0.,1.]); v = np.cross(z,d); s = np.linalg.norm(v); c = float(z@d)
    if s < 1e-12: return np.eye(3)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx@vx*((1-c)/s**2)

def raster(V, F, K, shape):
    fx,fy,ux,uy = K[0,0],K[1,1],K[0,2],K[1,2]
    ok = V[:,2] > 1e-6
    u = fx*V[:,0]/np.maximum(V[:,2],1e-6)+ux; v = fy*V[:,1]/np.maximum(V[:,2],1e-6)+uy
    im = Image.new("1",(shape[1],shape[0]),0); dr = ImageDraw.Draw(im)
    for t in F:
        if ok[t].all():
            dr.polygon([(u[t[0]],v[t[0]]),(u[t[1]],v[t[1]]),(u[t[2]],v[t[2]])], fill=1)
    return np.array(im, bool)

ap = argparse.ArgumentParser()
ap.add_argument("--merged", required=True); ap.add_argument("--mergedglb", required=True)
ap.add_argument("--scenesdir", required=True); ap.add_argument("--root", default="datasets/ycbv")
ap.add_argument("--out", default="outputs/ycb_scenes/merged_vs_composed.json")
a = ap.parse_args()

rows=[]
for m in json.load(open(a.merged)):
    p = os.path.join(a.mergedglb, m["name"]+".glb")
    if not os.path.exists(p): continue
    K = np.array(m["cam_K"], float); fx,fy,ux,uy = K[0,0],K[1,1],K[0,2],K[1,2]
    H,W = m["image_size"][1], m["image_size"][0]
    sdir = os.path.join(a.root,"test",m["scene"]); imk=int(m["im"])
    union = np.zeros((H,W), bool)
    for i in m["inst_idxs"]:
        union |= np.array(Image.open(os.path.join(sdir,"mask_visib",f"{imk:06d}_{i:06d}.png")))>0

    g = trimesh.load(p); mesh = trimesh.util.concatenate(list(g.geometry.values()))
    G = np.asarray(mesh.vertices,float); F = np.asarray(mesh.faces)
    ext = max(np.ptp(G[:,0]), np.ptp(G[:,1]))
    s = ((m["crop_size"]/1.1)*m["depth_med_m"]/fx)/max(ext,1e-9)
    V = G @ GLB2CV.T * s
    V = V @ ray_rotation((m["crop_center"][0]-ux)/fx,(m["crop_center"][1]-uy)/fy).T
    Z0 = m["depth_front_m"]-V[:,2].min()
    cxg=(V[:,0].min()+V[:,0].max())/2; cyg=(V[:,1].min()+V[:,1].max())/2; czg=(V[:,2].min()+V[:,2].max())/2
    Zc = Z0+czg
    V = V + np.array([(m["crop_center"][0]-ux)*Zc/fx-cxg,(m["crop_center"][1]-uy)*Zc/fy-cyg, Z0])
    Fs = F[np.random.default_rng(0).choice(len(F), min(len(F),60000), replace=False)]
    iou_m = (raster(V,Fs,K,(H,W)) & union).sum()/max((raster(V,Fs,K,(H,W)) | union).sum(),1)

    # components of the merged mesh (are objects still separable?)
    comps = mesh.split(only_watertight=False)
    big = [c for c in comps if len(c.faces) > 0.01*len(mesh.faces)]

    # composed per-object scene, scored against the same union mask
    sc = os.path.join(a.scenesdir, f"scene_{m['scene']}_{imk:06d}", "scene.glb")
    iou_c, ncomp_c = None, None
    if os.path.exists(sc):
        cs = trimesh.load(sc)
        parts = list(cs.geometry.values()); ncomp_c = len(parts)
        cm = trimesh.util.concatenate(parts)
        CV = np.asarray(cm.vertices,float) @ np.linalg.inv(CV2GLTF).T
        CF = np.asarray(cm.faces)
        CFs = CF[np.random.default_rng(0).choice(len(CF), min(len(CF),60000), replace=False)]
        R = raster(CV,CFs,K,(H,W))
        iou_c = (R & union).sum()/max((R | union).sum(),1)

    rows.append(dict(scene=m["scene"], n_obj=m["n_objects"], iou_merged=iou_m,
                     iou_composed=iou_c, merged_components=len(big), composed_parts=ncomp_c,
                     merged_faces=len(mesh.faces)))
    print(f"{m['scene']}  objects={m['n_objects']}  IoU merged {iou_m:.3f}  "
          f"composed {iou_c if iou_c is None else round(iou_c,3)}  "
          f"| merged mesh components={len(big)}  composed parts={ncomp_c}")

im_ = np.array([r["iou_merged"] for r in rows])
ic_ = np.array([r["iou_composed"] for r in rows if r["iou_composed"] is not None])
nc = np.array([r["merged_components"] for r in rows]); no = np.array([r["n_obj"] for r in rows])
print(f"\n=== {len(rows)} scenes ===")
print(f"IoU vs union mask : merged  mean {im_.mean():.3f}  median {np.median(im_):.3f}")
print(f"                    composed mean {ic_.mean():.3f}  median {np.median(ic_):.3f}")
print(f"separability: merged mesh yields {nc.sum()} components for {no.sum()} true objects "
      f"({(nc==no).sum()}/{len(rows)} scenes match)")
json.dump(rows, open(a.out,"w"), indent=2)

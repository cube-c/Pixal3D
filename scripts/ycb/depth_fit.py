"""Refine placement with ONE parameter: uniform scale about the camera centre.

Scaling P -> lambda*P leaves the projection u = f*X/Z + c unchanged, so the
silhouette is invariant and only depth moves. That makes lambda fittable against
the sensor depth map in closed form from a single rasterisation:
    z_render(lambda) = lambda * z_render(1)   =>   lambda* = median(z_obs / z_render)
"""
import json, glob, os, argparse
import numpy as np, trimesh
from PIL import Image

GLB2CV = np.diag([-1.,-1.,1.])

def ray_rotation(dx, dy):
    d = np.array([dx,dy,1.0]); d/=np.linalg.norm(d)
    z = np.array([0.,0.,1.]); v=np.cross(z,d); s=np.linalg.norm(v); c=float(z@d)
    if s < 1e-12: return np.eye(3)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*((1-c)/s**2)

def zbuffer(V, F, K, shape):
    """Nearest-surface depth per pixel (painter's z-test on triangle centroids)."""
    fx,fy,ux,uy = K[0,0],K[1,1],K[0,2],K[1,2]
    H,W = shape
    z = np.full((H,W), np.inf)
    tri = V[F]                                   # [T,3,3]
    zc  = tri[:,:,2].mean(1)
    order = np.argsort(-zc)                      # far -> near
    tri, zc = tri[order], zc[order]
    for k in range(3):                           # splat the 3 verts of each tri
        p = tri[:,k,:]
        ok = p[:,2] > 1e-6
        u = (fx*p[ok,0]/p[ok,2]+ux).astype(int); v=(fy*p[ok,1]/p[ok,2]+uy).astype(int)
        d = p[ok,2]
        m = (u>=0)&(u<W)&(v>=0)&(v<H)
        np.minimum.at(z, (v[m],u[m]), d[m])
    return z

ap=argparse.ArgumentParser()
ap.add_argument("--scenedirs",required=True); ap.add_argument("--glbdir",required=True)
ap.add_argument("--root",default="/app/Pixal3D/datasets/ycbv")
a=ap.parse_args()

rows=[]
for d in sorted(glob.glob(a.scenedirs)):
    man=json.load(open(f"{d}/crops.json")); K=np.array(man["cam_K"],float)
    fx,fy,ux,uy=K[0,0],K[1,1],K[0,2],K[1,2]
    sdir=os.path.join(a.root,"test",man["scene"]); imk=int(man["im"])
    H,W=man["image_size"][1],man["image_size"][0]
    depth=np.array(Image.open(os.path.join(sdir,"depth",f"{imk:06d}.png"))).astype(np.float32)
    dm=depth*man["depth_scale"]/1000.0
    for o in man["objects"]:
        p=os.path.join(a.glbdir,o["name"]+".glb")
        if not os.path.exists(p): continue
        g=trimesh.load(p); mesh=trimesh.util.concatenate(list(g.geometry.values()))
        G=np.asarray(mesh.vertices,float); F=np.asarray(mesh.faces)
        F=F[np.random.default_rng(0).choice(len(F),min(len(F),60000),replace=False)]
        ext=max(np.ptp(G[:,0]),np.ptp(G[:,1]))
        s=((o["crop_size"]/1.1)*o["depth_med_m"]/fx)/max(ext,1e-9)
        V=G@GLB2CV.T*s
        V=V@ray_rotation((o["crop_center"][0]-ux)/fx,(o["crop_center"][1]-uy)/fy).T
        Z0=o["depth_front_m"]-V[:,2].min()
        cxg=(V[:,0].min()+V[:,0].max())/2; cyg=(V[:,1].min()+V[:,1].max())/2
        czg=(V[:,2].min()+V[:,2].max())/2; Zc=Z0+czg
        V=V+np.array([(o["crop_center"][0]-ux)*Zc/fx-cxg,(o["crop_center"][1]-uy)*Zc/fy-cyg,Z0])

        gt=np.array(o["cam_t_m2c"],float)/1000.0
        cen=lambda X: np.array([(X[:,0].min()+X[:,0].max())/2,(X[:,1].min()+X[:,1].max())/2,
                                (X[:,2].min()+X[:,2].max())/2])
        err_before=float(np.linalg.norm(cen(V)-gt))*100

        # --- single-parameter fit about the camera centre ---
        zr=zbuffer(V,F,K,(H,W))
        msk=np.array(Image.open(os.path.join(sdir,"mask_visib",
                     f"{imk:06d}_{o['inst_idx']:06d}.png")))>0
        sel=msk & np.isfinite(zr) & (dm>1e-6)
        if sel.sum()<50:
            lam=1.0
        else:
            lam=float(np.median(dm[sel]/zr[sel]))
        Vf=V*lam
        err_after=float(np.linalg.norm(cen(Vf)-gt))*100
        rows.append(dict(scene=man["scene"],obj=o["obj_id"],visib=o["visib_fract"],
                         lam=lam,before=err_before,after=err_after,px=int(sel.sum())))
        print(f"  {man['scene']} obj{o['obj_id']:02d}  lambda={lam:.4f}  "
              f"3Derr {err_before:5.1f} -> {err_after:5.1f} cm  ({int(sel.sum())} px)")

b=np.array([r["before"] for r in rows]); af=np.array([r["after"] for r in rows])
l=np.array([r["lam"] for r in rows])
print(f"\n=== {len(rows)} objects ===")
print(f"lambda: median {np.median(l):.4f}  mean {l.mean():.4f}  range {l.min():.3f}..{l.max():.3f}")
print(f"3D error before: median {np.median(b):.2f} cm  mean {b.mean():.2f} cm")
print(f"3D error after : median {np.median(af):.2f} cm  mean {af.mean():.2f} cm")
print(f"improved on {(af<b).sum()}/{len(rows)} objects")
json.dump(rows, open("/app/Pixal3D/outputs/ycb_scenes/depth_fit_lambda.json","w"), indent=2)

"""Final composition: per-object placement + per-object lambda (median, default).

lambda is a uniform scale about the camera centre. It is projection-invariant, so
it cannot change the silhouette -- it only corrects metric depth. Fitted per object
from that object's own mask_visib (which already excludes pixels occluded by other
objects) against the sensor depth map.

Median is the default estimator: for a 1-parameter fit it has a 50% breakdown
point, and measured outlier contamination here is ~28%, so RANSAC bought nothing
(ADD-S 0.94 vs 0.92 cm).
"""
import json, glob, os, argparse
import numpy as np, trimesh
from PIL import Image

GLB2CV, CV2GLTF = np.diag([-1.,-1.,1.]), np.diag([1.,-1.,-1.])

def ray_rotation(dx,dy):
    d=np.array([dx,dy,1.0]); d/=np.linalg.norm(d)
    z=np.array([0.,0.,1.]); v=np.cross(z,d); s=np.linalg.norm(v); c=float(z@d)
    if s<1e-12: return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*((1-c)/s**2)

def zbuffer(V,F,K,shape):
    fx,fy,ux,uy=K[0,0],K[1,1],K[0,2],K[1,2]; H,W=shape
    z=np.full((H,W),np.inf); tri=V[F]
    for k in range(3):
        p=tri[:,k,:]; ok=p[:,2]>1e-6
        u=(fx*p[ok,0]/p[ok,2]+ux).astype(int); v=(fy*p[ok,1]/p[ok,2]+uy).astype(int)
        dd=p[ok,2]; m=(u>=0)&(u<W)&(v>=0)&(v<H)
        np.minimum.at(z,(v[m],u[m]),dd[m])
    return z

def fit_lambda(z,d,how,tau=0.010,iters=300):
    if len(z)<50: return 1.0,1.0
    if how=="none": return 1.0,1.0
    if how=="median": return float(np.median(d/z)),1.0
    rng=np.random.default_rng(0); best=None; lam=float(np.median(d/z))
    for i in rng.integers(0,len(z),size=min(iters,len(z))):
        if z[i]<=1e-6: continue
        L=d[i]/z[i]; inl=np.abs(d-L*z)<tau
        if best is None or inl.sum()>best.sum(): best,lam=inl,L
    if best is not None and best.sum()>=20:
        zz,dd=z[best],d[best]; return float((zz@dd)/(zz@zz)),float(best.mean())
    return lam,0.0

ap=argparse.ArgumentParser()
ap.add_argument("--scenedirs",required=True); ap.add_argument("--glbdir",required=True)
ap.add_argument("--outdir",required=True); ap.add_argument("--root",default="/app/Pixal3D/datasets/ycbv")
ap.add_argument("--lambda-fit",dest="lfit",choices=["none","median","ransac"],default="median")
a=ap.parse_args()

log=[]
for d_ in sorted(glob.glob(a.scenedirs)):
    man=json.load(open(f"{d_}/crops.json")); K=np.array(man["cam_K"],float)
    fx,fy,ux,uy=K[0,0],K[1,1],K[0,2],K[1,2]
    sdir=os.path.join(a.root,"test",man["scene"]); imk=int(man["im"])
    H,W=man["image_size"][1],man["image_size"][0]
    dm=np.array(Image.open(os.path.join(sdir,"depth",f"{imk:06d}.png"))).astype(np.float32)*man["depth_scale"]/1000.0
    scene=trimesh.Scene(); lams=[]
    for o in man["objects"]:
        p=os.path.join(a.glbdir,o["name"]+".glb")
        if not os.path.exists(p): continue
        g=trimesh.load(p); mesh=trimesh.util.concatenate(list(g.geometry.values()))
        G=np.asarray(mesh.vertices,float); F=np.asarray(mesh.faces)
        Fs=F[np.random.default_rng(0).choice(len(F),min(len(F),60000),replace=False)]
        ext=max(np.ptp(G[:,0]),np.ptp(G[:,1]))
        s=((o["crop_size"]/1.1)*o["depth_med_m"]/fx)/max(ext,1e-9)
        V=G@GLB2CV.T*s
        V=V@ray_rotation((o["crop_center"][0]-ux)/fx,(o["crop_center"][1]-uy)/fy).T
        Z0=o["depth_front_m"]-V[:,2].min()
        cx_=(V[:,0].min()+V[:,0].max())/2; cy_=(V[:,1].min()+V[:,1].max())/2
        cz_=(V[:,2].min()+V[:,2].max())/2; Zc=Z0+cz_
        V=V+np.array([(o["crop_center"][0]-ux)*Zc/fx-cx_,(o["crop_center"][1]-uy)*Zc/fy-cy_,Z0])
        mv=np.array(Image.open(os.path.join(sdir,"mask_visib",f"{imk:06d}_{o['inst_idx']:06d}.png")))>0
        zr=zbuffer(V,Fs,K,(H,W)); sel=mv&np.isfinite(zr)&(dm>1e-6)
        lam,_=fit_lambda(zr[sel].astype(float),dm[sel].astype(float),a.lfit)
        lams.append(lam)
        mesh.vertices=(V*lam)@CV2GLTF.T
        scene.add_geometry(mesh,node_name=o["name"])
    if not lams: continue
    od=os.path.join(a.outdir,f"scene_{man['scene']}_{imk:06d}")
    os.makedirs(od,exist_ok=True)
    scene.export(os.path.join(od,"scene_final.glb"))
    log.append(dict(scene=man["scene"],n=len(lams),lam_min=min(lams),lam_max=max(lams),
                    lam_med=float(np.median(lams))))
    print(f"{man['scene']}: {len(lams)} objects  lambda {min(lams):.4f}..{max(lams):.4f} "
          f"(median {np.median(lams):.4f})  -> {od}/scene_final.glb")
json.dump(log,open(os.path.join(a.outdir,"final_lambda.json"),"w"),indent=2)
print(f"\nwrote {len(log)} scenes with --lambda-fit {a.lfit}")

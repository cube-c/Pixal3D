"""ADD-S between the GT-posed YCB CAD model and our reconstruction.

Standard ADD-S assumes the *correct* CAD model and scores pose only:
    ADD-S = mean_{x in M} min_{y in M} || (R_gt x + t_gt) - (R_est y + t_est) ||
Here the prediction is a *generated* mesh, not the CAD model under an estimated
pose, so this measures pose AND shape error together. Reported as symmetric
chamfer as well, and against the usual 0.1*diameter threshold.
"""
import json, glob, os, argparse
import numpy as np, trimesh
from scipy.spatial import cKDTree
from PIL import Image

GLB2CV, CV2GLTF = np.diag([-1.,-1.,1.]), np.diag([1.,-1.,-1.])

def ray_rotation(dx,dy):
    d=np.array([dx,dy,1.0]); d/=np.linalg.norm(d)
    z=np.array([0.,0.,1.]); v=np.cross(z,d); s=np.linalg.norm(v); c=float(z@d)
    if s<1e-12: return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*((1-c)/s**2)

YCB={1:"master_chef_can",2:"cracker_box",3:"sugar_box",4:"tomato_soup_can",5:"mustard_bottle",
     6:"tuna_fish_can",7:"pudding_box",8:"gelatin_box",9:"potted_meat_can",10:"banana",
     11:"pitcher_base",12:"bleach_cleanser",13:"bowl",14:"mug",15:"power_drill",16:"wood_block",
     17:"scissors",18:"large_marker",19:"large_clamp",20:"extra_large_clamp",21:"foam_brick"}

ap=argparse.ArgumentParser()
ap.add_argument("--scenedirs",required=True); ap.add_argument("--glbdir",required=True)
ap.add_argument("--models",required=True); ap.add_argument("--root",default="/app/Pixal3D/datasets/ycbv")
ap.add_argument("--npts",type=int,default=8000)
ap.add_argument("--lam",type=str,default=None,help="scene_depth_fit.json to apply per-scene lambda")
a=ap.parse_args()

info=json.load(open(os.path.join(a.models,"models_info.json")))
lam_map={}
if a.lam and os.path.exists(a.lam):
    lam_map={r["scene"]: r["lam"] for r in json.load(open(a.lam))}

rows=[]
for d in sorted(glob.glob(a.scenedirs)):
    man=json.load(open(f"{d}/crops.json")); K=np.array(man["cam_K"],float)
    fx,fy,ux,uy=K[0,0],K[1,1],K[0,2],K[1,2]
    lam=lam_map.get(man["scene"],1.0)
    for o in man["objects"]:
        p=os.path.join(a.glbdir,o["name"]+".glb")
        if not os.path.exists(p): continue
        # --- prediction: our generated mesh, placed
        g=trimesh.load(p); mesh=trimesh.util.concatenate(list(g.geometry.values()))
        G=np.asarray(mesh.vertices,float)
        ext=max(np.ptp(G[:,0]),np.ptp(G[:,1]))
        s=((o["crop_size"]/1.1)*o["depth_med_m"]/fx)/max(ext,1e-9)
        V=G@GLB2CV.T*s
        V=V@ray_rotation((o["crop_center"][0]-ux)/fx,(o["crop_center"][1]-uy)/fy).T
        Z0=o["depth_front_m"]-V[:,2].min()
        cxg=(V[:,0].min()+V[:,0].max())/2; cyg=(V[:,1].min()+V[:,1].max())/2
        czg=(V[:,2].min()+V[:,2].max())/2; Zc=Z0+czg
        V=V+np.array([(o["crop_center"][0]-ux)*Zc/fx-cxg,(o["crop_center"][1]-uy)*Zc/fy-cyg,Z0])
        V=V*lam
        m2=mesh.copy(); m2.vertices=V
        P=m2.sample(a.npts)

        # --- GT: CAD model under GT pose (BOP models & t are in mm)
        cad=trimesh.load(os.path.join(a.models,f"obj_{o['obj_id']:06d}.ply"))
        Q=cad.sample(a.npts)/1000.0
        R=np.array(o["cam_R_m2c"],float).reshape(3,3); t=np.array(o["cam_t_m2c"],float)/1000.0
        Q=Q@R.T+t

        dgt=cKDTree(P).query(Q)[0]           # GT -> pred  (this is ADD-S)
        dpr=cKDTree(Q).query(P)[0]           # pred -> GT
        diam=info[str(o["obj_id"])]["diameter"]/1000.0
        rows.append(dict(scene=man["scene"],obj=YCB.get(o["obj_id"],o["obj_id"]),
            visib=o["visib_fract"], adds=float(dgt.mean())*100,
            chamfer=float((dgt.mean()+dpr.mean())/2)*100,
            diameter=diam*100, ratio=float(dgt.mean())/diam))
        r=rows[-1]
        print(f"  {r['scene']} {r['obj']:20s} visib {r['visib']:.2f}  ADD-S {r['adds']:5.2f} cm  "
              f"chamfer {r['chamfer']:5.2f} cm  d={r['diameter']:5.1f} cm  ADD-S/d={r['ratio']:.3f}")

ad=np.array([r["adds"] for r in rows]); ch=np.array([r["chamfer"] for r in rows])
rt=np.array([r["ratio"] for r in rows]); vi=np.array([r["visib"] for r in rows])
print(f"\n=== {len(rows)} objects ===")
print(f"ADD-S   : mean {ad.mean():.2f} cm  median {np.median(ad):.2f} cm")
print(f"chamfer : mean {ch.mean():.2f} cm  median {np.median(ch):.2f} cm")
print(f"ADD-S < 0.1*diameter : {(rt<0.1).mean():.0%}   < 0.2*d : {(rt<0.2).mean():.0%}")
for lbl,m in (("visib>=0.9",vi>=0.9),("visib<0.9",vi<0.9)):
    if m.sum(): print(f"  {lbl:11s} N={m.sum():3d}  ADD-S {ad[m].mean():.2f} cm  ADD-S/d {rt[m].mean():.3f}")
json.dump(rows,open("/app/Pixal3D/outputs/ycb_scenes/adds_eval.json","w"),indent=2)

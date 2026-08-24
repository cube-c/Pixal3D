"""One summary figure per scene: input, instance masks, reprojection, and renders
of both the composed (separable) and merged (single-shot) reconstructions."""
import json, glob, os, argparse
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

CV2GLTF = np.diag([1.,-1.,-1.])
COLORS = [(255,80,80),(80,255,80),(80,160,255),(255,220,60),(255,80,255),(120,255,220)]
YCB={1:"master_chef_can",2:"cracker_box",3:"sugar_box",4:"tomato_soup_can",5:"mustard_bottle",
 6:"tuna_fish_can",7:"pudding_box",8:"gelatin_box",9:"potted_meat_can",10:"banana",11:"pitcher_base",
 12:"bleach_cleanser",13:"bowl",14:"mug",15:"power_drill",16:"wood_block",17:"scissors",
 18:"large_marker",19:"large_clamp",20:"extra_large_clamp",21:"foam_brick"}

def render(path, res=460, yaw=0.6, pitch=0.3):
    """Cheap vertex-splat render with a z-test; textures via UV lookup."""
    sc = trimesh.load(path)
    geoms=[]
    for g in sc.geometry.values():
        V=np.asarray(g.vertices,float); C=np.full((len(V),3),170,np.uint8)
        try:
            uv=np.asarray(g.visual.uv); tex=np.asarray(g.visual.material.baseColorTexture.convert("RGB"))
            h,w=tex.shape[:2]
            C=tex[np.clip(((1-uv[:,1])*(h-1)).astype(int),0,h-1),
                  np.clip((uv[:,0]*(w-1)).astype(int),0,w-1)]
        except Exception: pass
        geoms.append((V,C))
    allV=np.vstack([v for v,_ in geoms])
    ctr=(allV.min(0)+allV.max(0))/2; rad=np.linalg.norm(allV.max(0)-allV.min(0))/2
    cy,sy=np.cos(yaw),np.sin(yaw); cp,sp=np.cos(pitch),np.sin(pitch)
    R=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    img=np.full((res,res,3),22,np.uint8); zb=np.full((res,res),1e9)
    f=res*0.85; eye=rad*3.0
    for V,C in geoms:
        P=(V-ctr)@R.T; P[:,2]+=eye
        ok=P[:,2]>1e-3
        u=(f*P[:,0]/np.maximum(P[:,2],1e-3)+res/2).astype(int)
        v=(-f*P[:,1]/np.maximum(P[:,2],1e-3)+res/2).astype(int)
        o=np.argsort(-P[:,2]); u,v,z,c,ok=u[o],v[o],P[o,2],C[o],ok[o]
        for dv in (0,1):
            for du in (0,1):
                jj=np.clip(v[ok]+dv,0,res-1); ii=np.clip(u[ok]+du,0,res-1)
                s=z[ok]<zb[jj,ii]
                zb[jj[s],ii[s]]=z[ok][s]; img[jj[s],ii[s]]=c[ok][s]
    return img

ap=argparse.ArgumentParser()
ap.add_argument("--scenedirs",required=True); ap.add_argument("--outdir",required=True)
ap.add_argument("--root",default="/app/Pixal3D/datasets/ycbv")
ap.add_argument("--name",default="summary.png")
a=ap.parse_args()

adds={}
if os.path.exists(f"{a.outdir}/lambda_variants.json"):
    for r in json.load(open(f"{a.outdir}/lambda_variants.json"))["per_object"]:
        adds.setdefault(r["scene"],[]).append(r["a_med"])
mvc={}
if os.path.exists(f"{a.outdir}/merged_vs_composed.json"):
    for r in json.load(open(f"{a.outdir}/merged_vs_composed.json")): mvc[r["scene"]]=r

for d in sorted(glob.glob(a.scenedirs)):
    man=json.load(open(f"{d}/crops.json")); sc_=man["scene"]; imk=int(man["im"])
    od=os.path.join(a.outdir,f"scene_{sc_}_{imk:06d}")
    if not os.path.exists(os.path.join(od,"scene_final.glb")): continue
    root=os.path.join(a.root,"test",sc_)
    rgb=Image.open(man["rgb"]).convert("RGB")

    # panel 2: instance masks
    mimg=np.array(rgb).copy()//3
    for i,o in enumerate(man["objects"]):
        m=np.array(Image.open(f"{root}/mask_visib/{imk:06d}_{o['inst_idx']:06d}.png"))>0
        mimg[m]=(np.array(COLORS[i%len(COLORS)])*0.75+mimg[m]*0.25).astype(np.uint8)

    # panel 3: reprojection of the final composed scene
    ov=Image.open(os.path.join(od,"overlay.png")).convert("RGB") \
        if os.path.exists(os.path.join(od,"overlay.png")) else rgb

    r_fin=render(os.path.join(od,"scene_final.glb"))
    r_mrg=render(os.path.join(od,"merged_single_shot.glb")) \
        if os.path.exists(os.path.join(od,"merged_single_shot.glb")) else np.full((460,460,3),22,np.uint8)

    fig,ax=plt.subplots(1,5,figsize=(23,5.2))
    names=", ".join(YCB.get(o["obj_id"],str(o["obj_id"])) for o in man["objects"])
    panels=[(np.array(rgb),"(a) input  YCB-V %s / frame %d"%(sc_,imk)),
            (mimg,"(b) GT visible masks (mask_visib)"),
            (np.array(ov),"(c) reprojection of composed scene"),
            (r_fin,"(d) composed, per-object + lambda  [separable]"),
            (r_mrg,"(e) merged union-mask  [1 blob]")]
    for k,(im,t) in enumerate(panels):
        ax[k].imshow(im); ax[k].set_title(t,fontsize=10); ax[k].axis("off")
    sub=f"{len(man['objects'])} objects: {names}"
    if sc_ in adds: sub+=f"   |   ADD-S {np.mean(adds[sc_]):.2f} cm"
    if sc_ in mvc: sub+=f"   |   merged mesh: {mvc[sc_]['merged_components']} components for {mvc[sc_]['n_obj']} objects"
    fig.suptitle(sub,fontsize=11,y=0.04)
    plt.tight_layout(rect=[0,0.05,1,1])
    out=os.path.join(od,a.name); plt.savefig(out,dpi=110); plt.close()
    print("wrote",out)

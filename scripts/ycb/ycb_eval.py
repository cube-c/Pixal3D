"""Compose YCB scenes, with and without the off-axis ray correction, and score
both by reprojected-silhouette IoU against the GT visible mask."""
import json, os, glob, math, argparse
import numpy as np, trimesh
from PIL import Image, ImageDraw

YCB = {1:"master_chef_can",2:"cracker_box",3:"sugar_box",4:"tomato_soup_can",5:"mustard_bottle",
       6:"tuna_fish_can",7:"pudding_box",8:"gelatin_box",9:"potted_meat_can",10:"banana",
       11:"pitcher_base",12:"bleach_cleanser",13:"bowl",14:"mug",15:"power_drill",16:"wood_block",
       17:"scissors",18:"large_marker",19:"large_clamp",20:"extra_large_clamp",21:"foam_brick"}
GLB2CV  = np.diag([-1., -1., 1.])
CV2GLTF = np.diag([1., -1., -1.])
COLORS = [(255,80,80),(80,255,80),(80,160,255),(255,220,60),(255,80,255),(120,255,220)]

def ray_rotation(dx, dy):
    """Minimal rotation taking the optical axis (0,0,1) onto the ray (dx,dy,1)."""
    d = np.array([dx, dy, 1.0]); d /= np.linalg.norm(d)
    z = np.array([0., 0., 1.])
    v = np.cross(z, d); s = np.linalg.norm(v); c = float(z @ d)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx @ vx * ((1-c)/s**2)

def place(o, G, K, use_ray):
    fx, fy, ux, uy = K[0,0], K[1,1], K[0,2], K[1,2]
    ext = max(np.ptp(G[:,0]), np.ptp(G[:,1]))
    s = ((o["crop_size"]/1.1) * o["depth_med_m"] / fx) / max(ext, 1e-9)
    V = G @ GLB2CV.T * s
    if use_ray:
        cx, cy = o["crop_center"]
        V = V @ ray_rotation((cx-ux)/fx, (cy-uy)/fy).T
    Z0 = o["depth_front_m"] - V[:,2].min()
    cxg = (V[:,0].min()+V[:,0].max())/2; cyg = (V[:,1].min()+V[:,1].max())/2
    czg = (V[:,2].min()+V[:,2].max())/2
    Zc = Z0 + czg
    X0 = (o["crop_center"][0]-ux)*Zc/fx - cxg
    Y0 = (o["crop_center"][1]-uy)*Zc/fy - cyg
    return V + np.array([X0, Y0, Z0]), Zc

def reproj_mask(V, F, K, shape):
    fx, fy, ux, uy = K[0,0], K[1,1], K[0,2], K[1,2]
    ok = V[:,2] > 1e-6
    u = fx*V[:,0]/np.maximum(V[:,2],1e-6) + ux
    v = fy*V[:,1]/np.maximum(V[:,2],1e-6) + uy
    im = Image.new("1", (shape[1], shape[0]), 0); dr = ImageDraw.Draw(im)
    for t in F:
        if ok[t].all():
            dr.polygon([(u[t[0]],v[t[0]]),(u[t[1]],v[t[1]]),(u[t[2]],v[t[2]])], fill=1)
    return np.array(im, bool)

ap = argparse.ArgumentParser()
ap.add_argument("--scenedirs", required=True); ap.add_argument("--glbdir", required=True)
ap.add_argument("--outdir", required=True); ap.add_argument("--root", default="datasets/ycbv")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

rows = []
for d in sorted(glob.glob(a.scenedirs)):
    man = json.load(open(f"{d}/crops.json")); K = np.array(man["cam_K"], float)
    sdir = os.path.join(a.root, "test", man["scene"]); imk = int(man["im"])
    H, W = man["image_size"][1], man["image_size"][0]
    scenes = {False: trimesh.Scene(), True: trimesh.Scene()}
    placed = []
    for o in man["objects"]:
        p = os.path.join(a.glbdir, o["name"] + ".glb")
        if not os.path.exists(p): continue
        g = trimesh.load(p); mesh = trimesh.util.concatenate(list(g.geometry.values()))
        G = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces)
        Fs = F[np.random.default_rng(0).choice(len(F), min(len(F), 40000), replace=False)]
        gtm = np.array(Image.open(os.path.join(sdir, "mask_visib",
                        f"{imk:06d}_{o['inst_idx']:06d}.png"))) > 0
        res = {}
        for use_ray in (False, True):
            V, Zc = place(o, G, K, use_ray)
            M = reproj_mask(V, Fs, K, (H, W))
            res[use_ray] = (V, (M & gtm).sum()/max((M | gtm).sum(), 1))
            m2 = mesh.copy(); m2.vertices = V @ CV2GLTF.T
            scenes[use_ray].add_geometry(m2, node_name=o["name"])
        placed.append((o, res[True][0]))
        gt = np.array(o["cam_t_m2c"], float)/1000.0
        V = res[True][0]
        est = np.array([(V[:,0].min()+V[:,0].max())/2,(V[:,1].min()+V[:,1].max())/2,
                        (V[:,2].min()+V[:,2].max())/2])
        cx, cy = o["crop_center"]
        rows.append(dict(scene=man["scene"], obj=YCB.get(o["obj_id"],str(o["obj_id"])),
            visib=o["visib_fract"],
            fov=math.degrees(2*math.atan(o["crop_size"]/(2*K[0,0]))),
            offaxis=math.degrees(math.atan(math.hypot((cx-K[0,2])/K[0,0],(cy-K[1,2])/K[1,1]))),
            iou_plain=res[False][1], iou_ray=res[True][1],
            d3=float(np.linalg.norm(est-gt))*100))
    nm = f"scene_{man['scene']}_{imk:06d}"
    if placed:
        sdir_out = os.path.join(a.outdir, nm)
        os.makedirs(os.path.join(sdir_out, "objects"), exist_ok=True)
        scenes[True].export(os.path.join(sdir_out, "scene.glb"))
        scenes[False].export(os.path.join(sdir_out, "scene_no_ray_correction.glb"))
        import shutil
        for o, _ in placed:
            shutil.copy(os.path.join(a.glbdir, o["name"] + ".glb"),
                        os.path.join(sdir_out, "objects", o["name"] + ".glb"))
        shutil.copy(f"{d}/crops.json", os.path.join(sdir_out, "crops.json"))
        img = Image.open(man["rgb"]).convert("RGB"); dr = ImageDraw.Draw(img)
        for i,(o,V) in enumerate(placed):
            sub = V[np.random.default_rng(0).choice(len(V), min(len(V),3500), replace=False)]
            uu = K[0,0]*sub[:,0]/sub[:,2]+K[0,2]; vv = K[1,1]*sub[:,1]/sub[:,2]+K[1,2]
            for p1,p2 in zip(uu,vv): dr.point((p1,p2), fill=COLORS[i%len(COLORS)])
        img.save(os.path.join(sdir_out, "overlay.png"))
    print(f"{man['scene']}: {len(placed)} objects")

json.dump(rows, open(os.path.join(a.outdir,"placement_eval.json"),"w"), indent=2)
ip = np.array([r["iou_plain"] for r in rows]); ir = np.array([r["iou_ray"] for r in rows])
d3 = np.array([r["d3"] for r in rows]); oa = np.array([r["offaxis"] for r in rows])
vi = np.array([r["visib"] for r in rows]); fv = np.array([r["fov"] for r in rows])
print(f"\n=== {len(rows)} objects, {len(set(r['scene'] for r in rows))} scenes ===")
print(f"mask IoU  no ray-correction : mean {ip.mean():.3f}  median {np.median(ip):.3f}")
print(f"mask IoU  with ray-correction: mean {ir.mean():.3f}  median {np.median(ir):.3f}")
print(f"improved on {(ir>ip).sum()}/{len(rows)} objects")
print(f"3D translation err vs GT: median {np.median(d3):.1f} cm  mean {d3.mean():.1f} cm")
for lbl, m in (("off-axis >8deg", oa>8), ("off-axis <=8deg", oa<=8),
               ("FOV <10deg", fv<10), ("FOV >=10deg", fv>=10),
               ("visib >=0.9", vi>=0.9), ("visib <0.9", vi<0.9)):
    if m.sum():
        print(f"  {lbl:16s} N={m.sum():3d}  IoU {ir[m].mean():.3f} (plain {ip[m].mean():.3f})  3Derr {np.median(d3[m]):.1f}cm")

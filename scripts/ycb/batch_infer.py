"""Batch runner: generate a GLB for every image in a directory using one loaded pipeline."""
import os
import sys
import time
import argparse
import traceback

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import inference as inf  # noqa: E402  (sets env vars on import)
import o_voxel  # noqa: E402


# --- transformers>=5 compat -------------------------------------------------
# Pixal3D pins transformers==4.57.3, where DINOv3ViTModel exposes `.layer`
# directly. In transformers 5.x the blocks moved into the nested encoder
# (`DINOv3ViTModel.model.layer`), so `extract_features` raises
# AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'.
# Everything else in the block loop is unchanged between the two versions.
import torch.nn.functional as _F
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor as _DinoExtractor,
)


def _extract_features_compat(self, image):
    image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = self.model.embeddings(image, bool_masked_pos=None)
    position_embeddings = self.model.rope_embeddings(image)

    layers = getattr(self.model, "layer", None)
    if layers is None:
        layers = self.model.model.layer

    for layer_module in layers:
        hidden_states = layer_module(
            hidden_states,
            position_embeddings=position_embeddings,
        )

    return _F.layer_norm(hidden_states, hidden_states.shape[-1:])


_DinoExtractor.extract_features = _extract_features_compat
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, nargs="+", required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fov", type=float, default=-1.0)
    ap.add_argument("--fovs", type=str, default=None,
                    help="comma-separated per-image FOV in radians, same order as --images")
    ap.add_argument("--low_vram", action="store_true")
    ap.add_argument("--resolution", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fov_list = [float(x) for x in args.fovs.split(",")] if args.fovs else None
    if fov_list is not None:
        assert len(fov_list) == len(args.images), "--fovs must match --images"

    pipeline = inf.init_pipeline(inf.MODEL_PATH, low_vram=args.low_vram)
    moge_model = None
    if args.fov <= 0 and fov_list is None:
        moge_model = inf.load_moge_model(device="cuda")

    pipeline_type = f"{args.resolution if args.resolution > 0 else (1024 if args.low_vram else 1536)}_cascade"
    mesh_scale, extend_pixel, image_resolution = 1.0, 0, 512

    ok, fail = [], []
    for i, image_path in enumerate(args.images):
        name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.outdir, f"{name}.glb")
        print(f"\n===== [{i+1}/{len(args.images)}] {image_path} -> {out_path} =====", flush=True)
        t0 = time.time()
        try:
            img = Image.open(image_path)
            image_preprocessed = pipeline.preprocess_image(img)

            tmp_path = os.path.join(args.outdir, f"_tmp_{name}.png")
            image_preprocessed.save(tmp_path)
            fov_i = fov_list[i] if fov_list is not None else args.fov
            if moge_model is not None:
                camera_params = inf.get_camera_params_wild_moge(
                    tmp_path, moge_model, device="cuda", mesh_scale=mesh_scale,
                    extend_pixel=extend_pixel, image_resolution=image_resolution)
            else:
                distance = inf.distance_from_fov(
                    fov_i, torch.tensor([-1.0, 0.0, 0.0]),
                    torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
                    mesh_scale, image_resolution)["distance_from_x"]
                camera_params = {"camera_angle_x": fov_i, "distance": distance, "mesh_scale": mesh_scale}
            os.remove(tmp_path)
            print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f} distance={camera_params['distance']:.4f}", flush=True)

            torch.manual_seed(args.seed)
            mesh_list, (_, _, res) = pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=args.seed,
                sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5,
                                                 "guidance_rescale": 0.7, "rescale_t": 5.0},
                shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5,
                                           "guidance_rescale": 0.5, "rescale_t": 3.0},
                tex_slat_sampler_params={"steps": 12, "guidance_strength": 1.0,
                                         "guidance_rescale": 0.0, "rescale_t": 3.0},
                preprocess_image=False,
                return_latent=True,
                pipeline_type=pipeline_type,
                max_num_tokens=49152,
            )
            mesh = mesh_list[0]
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
                grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=200000, texture_size=1024,
                remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
            )
            rot = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
            glb.apply_transform(rot)
            glb.export(out_path, extension_webp=True)
            dt = time.time() - t0
            print(f"[OK] {name} in {dt:.1f}s -> {out_path}", flush=True)
            ok.append(name)
        except Exception:
            traceback.print_exc()
            print(f"[FAILED] {name}", flush=True)
            fail.append(name)
        finally:
            torch.cuda.empty_cache()

    print(f"\n===== SUMMARY: {len(ok)} ok, {len(fail)} failed =====", flush=True)
    if fail:
        print("failed: " + ", ".join(fail), flush=True)


if __name__ == "__main__":
    main()

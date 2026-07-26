"""
Render novel views from test_poses.csv for competition submission.

Usage:
    python render_test_poses.py \
        -m output/bonsai \
        --test_poses /mnt/passport/VAI_NVS_DATA_ROUND2/bonsai/test/test_poses.csv \
        --output_dir output/bonsai/test_renders
"""

import os
import csv
import torch
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from gaussian_renderer import render, GaussianModel
from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import focal2fov
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def load_test_cameras(test_poses_csv, resolution_scale=1, data_device="cuda"):
    """Build Camera objects from test_poses.csv (no ground-truth image needed)."""
    cameras = []
    with open(test_poses_csv, newline="") as f:
        reader = csv.DictReader(f)
        for uid, row in enumerate(reader):
            # Quaternion (qw, qx, qy, qz) and translation in COLMAP convention
            qvec = np.array([float(row["qw"]), float(row["qx"]),
                             float(row["qy"]), float(row["qz"])])
            tvec = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])

            # R is the world-to-camera rotation matrix transposed (as in COLMAP convention)
            R = np.transpose(qvec2rotmat(qvec))
            T = tvec

            orig_w = int(row["width"])
            orig_h = int(row["height"])
            width  = round(orig_w / resolution_scale)
            height = round(orig_h / resolution_scale)
            fx = float(row["fx"]) / resolution_scale
            fy = float(row["fy"]) / resolution_scale
            FoVx = focal2fov(fx, width)
            FoVy = focal2fov(fy, height)

            # Dummy black PIL image as placeholder (not used for loss or GT saving)
            dummy_pil = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))

            cam = Camera(
                resolution=(width, height),
                colmap_id=uid,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                depth_params=None,
                image=dummy_pil,
                invdepthmap=None,
                image_name=os.path.splitext(row["image_name"])[0],
                uid=uid,
                data_device=data_device,
                train_test_exp=False,
                is_test_dataset=False,
                is_test_view=False,
            )
            cameras.append((row["image_name"], cam))

    return cameras


def render_test(model_path, test_poses_csv, output_dir, iteration, pipeline, bg_color, resolution_scale=1):
    os.makedirs(output_dir, exist_ok=True)

    # Load gaussians
    gaussians = GaussianModel(3)  # sh_degree=3 matches training default

    # Find the right checkpoint iteration
    if iteration == -1:
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        iterations = sorted([
            int(d.replace("iteration_", ""))
            for d in os.listdir(point_cloud_dir)
            if d.startswith("iteration_")
        ])
        if not iterations:
            raise RuntimeError(f"No saved iterations found in {point_cloud_dir}")
        iteration = iterations[-1]

    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    print(f"Loading gaussians from {ply_path}")
    gaussians.load_ply(ply_path)

    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = load_test_cameras(test_poses_csv, resolution_scale=resolution_scale)
    print(f"Rendering {len(cameras)} test views -> {output_dir}")

    with torch.no_grad():
        for i, (image_name, cam) in enumerate(tqdm(cameras, desc="Rendering")):
            rendering = render(cam, gaussians, pipeline, background,
                               separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            out_path = os.path.join(output_dir, image_name)
            torchvision.utils.save_image(rendering, out_path)
            del rendering
            if i % 10 == 0:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Render from test_poses.csv")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--test_poses", type=str, required=True,
                        help="Path to test_poses.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save rendered images (default: <model>/test_renders)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Which iteration to load (-1 = latest)")
    parser.add_argument("--resolution_scale", type=float, default=1.0,
                        help="Downscale factor: 1=full, 2=half, etc.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    if args.output_dir is None:
        args.output_dir = os.path.join(dataset.model_path, "test_renders")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    render_test(
        model_path=dataset.model_path,
        test_poses_csv=args.test_poses,
        output_dir=args.output_dir,
        iteration=args.iteration,
        pipeline=pipe,
        bg_color=bg_color,
        resolution_scale=args.resolution_scale,
    )
    print("Done.")

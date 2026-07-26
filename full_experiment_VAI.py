"""
Full experiment pipeline for VAI_NVS_DATA_ROUND2 using the gaussian-splatting repo.

For each scene:
  1. Train using train.py  (skipped if output already contains a saved point_cloud)
  2. Render test poses from test/test_poses.csv using render_test_poses.py
     -> rendered images are saved inside the output folder:
        <output_root>/<scene>/test_renders/<image_name>
        where <image_name> is taken directly from the test_poses.csv `image_name` column.

Usage:
    python full_experiment_VAI.py \
        --data_root ../VAI_NVS_DATA_ROUND2 \
        --output_root output

Optional:
    --scenes bonsai chair HCM0421   # run only specific scenes
    --skip_train                    # skip training, only render
    --skip_render                   # skip rendering, only train
    --iterations 30000              # number of training iterations (default: 30000)
    --train_resolution 2            # train image downscale: 1=full (default), 2=half, 4=quarter
    --resolution_scale 1.0          # render downscale factor (1=full res)
"""

import os
import sys
import subprocess
import argparse
import copy


def has_finished_training(output_dir):
    """Return True if the output dir already contains at least one saved point_cloud iteration."""
    pc_dir = os.path.join(output_dir, "point_cloud")
    if not os.path.isdir(pc_dir):
        return False
    iterations = [d for d in os.listdir(pc_dir) if d.startswith("iteration_")]
    return len(iterations) > 0


def run(cmd, desc="", extra_env=None):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    env = None
    if extra_env:
        env = copy.copy(os.environ)
        env.update(extra_env)
    subprocess.run(cmd, check=True, env=env)


def main():
    PRESETS = {
        "runpod": {
            "data_root": "/workspace/VAI_NVS_DATA_ROUND2",
            "output_root": "/workspace/Demo_3DGS/output",
        },
        "local": {
            "data_root": "/mnt/passport/VAI_NVS_DATA_ROUND2",
            "output_root": "/mnt/passport/gaussian-splatting/output",
        },
    }

    parser = argparse.ArgumentParser(
        description="Train and render all VAI_NVS_DATA_ROUND2 scenes with gaussian-splatting."
    )
    parser.add_argument("--preset", choices=["runpod", "local"], default=None,
                        help="Environment preset: 'runpod' or 'local'. "
                             "Sets --data_root and --output_root automatically. "
                             "Explicit --data_root / --output_root override the preset.")
    parser.add_argument("--data_root", default=None,
                        help="Root directory containing per-scene folders.")
    parser.add_argument("--output_root", default=None,
                        help="Root directory for trained model outputs.")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Subset of scene names to process. Default: all scenes in data_root.")
    # parser.add_argument("--skip_train", action="store_true",
    #                     help="Skip training; only run rendering.")
    # parser.add_argument("--skip_render", action="store_true",
    #                     help="Skip rendering; only run training.")
    parser.add_argument("--iterations", type=int, default=30_000,
                        help="Number of training iterations (default: 30000). "
                             "~14-15h for 7 scenes on RTX 4090. Use 15000 for ~8h at lower quality.")
    parser.add_argument("--train_resolution", "-r", type=int, default=1,
                        help="Training image downscale factor passed to train.py via -r (default: 1 = full res). "
                             "Use 1 for full res (requires more VRAM), 2 for half res, 4 for quarter res.")
    parser.add_argument("--resolution_scale", type=float, default=1.0,
                        help="Render resolution downscale factor: 1=full, 2=half, etc.")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002,
                        help="Gradient threshold for Gaussian densification (default: 0.0002). "
                             "Lower = more Gaussians = finer detail but more VRAM. Upstream default is 0.0002.")
    parser.add_argument("--densify_until_iter", type=int, default=15_000,
                        help="Stop densification after this iteration (default: 15000). "
                             "Set to half of --iterations for best quality. Upstream default is 15000.")
    parser.add_argument("--optimizer_type", type=str, default="default",
                        choices=["default", "sparse_adam"],
                        help="Optimizer type for training (default: default). "
                             "sparse_adam gives ~2.7x speedup with the accelerated rasterizer.")
    parser.add_argument("--antialiasing", action="store_true", default=False,
                        help="Enable EWA antialiasing during training and rendering (requires accelerated rasterizer).")
    parser.add_argument("--lambda_lpips", type=float, default=0.0,
                        help="Weight for LPIPS perceptual loss during training (default: 0, disabled). "
                             "Costs ~40%% speed. Enable with 0.05 only if you have time budget. "
                             "LPIPS is 40%% of the competition score.")
    parser.add_argument("--lambda_dssim", type=float, default=0.3,
                        help="Weight for SSIM loss during training (default: 0.3). "
                             "SSIM is 30%% of the competition score. Upstream default is 0.2.")
    parser.add_argument("--opacity_reset_interval", type=int, default=3000,
                        help="Reset opacity every N iterations (default: 3000). "
                             "Lower = more aggressive size-pruning = fewer Gaussians = less VRAM. Upstream default is 3000.")
    parser.add_argument("--position_lr_max_steps", type=int, default=None,
                        help="Steps over which position LR decays (default: equals --iterations). "
                             "Must match iterations or positions stop being refined early. Upstream default is 30000.")
    args = parser.parse_args()

    # Apply preset paths, then let explicit args override
    if args.preset is not None:
        p = PRESETS[args.preset]
        if args.data_root is None:
            args.data_root = p["data_root"]
        if args.output_root is None:
            args.output_root = p["output_root"]

    # Fall back to sibling-directory defaults if nothing specified
    if args.data_root is None:
        args.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "VAI_NVS_DATA_ROUND2")
    if args.output_root is None:
        args.output_root = "output"

    # Position LR schedule must span the full training run
    if args.position_lr_max_steps is None:
        args.position_lr_max_steps = args.iterations

    # Densify for first half of training if not explicitly set
    if args.densify_until_iter == 15_000 and args.iterations != 30_000:
        args.densify_until_iter = args.iterations // 2

    data_root   = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root)

    # Discover scenes: every sub-directory in data_root that has a 'train' folder
    all_scenes = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d, "train"))
    ])

    scenes = args.scenes if args.scenes else all_scenes
    if not scenes:
        print(f"No scenes found in {data_root}. Exiting.")
        sys.exit(1)

    print(f"Scenes to process: {scenes}")

    for scene in scenes:
        scene_train_dir = os.path.join(data_root, scene, "train")
        scene_test_csv  = os.path.join(data_root, scene, "test", "test_poses.csv")
        output_dir      = os.path.join(output_root, scene)
        render_out_dir  = os.path.join(output_dir, "test_renders")

        print(f"\n{'#' * 60}")
        print(f"  SCENE: {scene}")
        print(f"{'#' * 60}")

        # ------------------------------------------------------------------ #
        # TRAIN
        # ------------------------------------------------------------------ #
        if True:  # not args.skip_train
            if has_finished_training(output_dir):
                print(f"[SKIP TRAIN] {scene}: trained model already exists at {output_dir}")
            else:
                train_cmd = [
                    sys.executable, "train.py",
                    "-s", scene_train_dir,
                    "-m", output_dir,
                    "--iterations", str(args.iterations),
                    # Save checkpoints at final iteration so rendering can load them
                    "--save_iterations", str(args.iterations),
                    # Downsample images to reduce VRAM usage
                    "-r", str(args.train_resolution),
                    # GT images loaded to GPU on-demand — saves 1-3GB VRAM with zero quality impact
                    "--data_device", "cpu",
                    # Limit Gaussian count growth to avoid rasterizer OOM
                    "--densify_grad_threshold", str(args.densify_grad_threshold),
                    "--densify_until_iter", str(args.densify_until_iter),
                    "--optimizer_type", args.optimizer_type,
                    "--lambda_lpips", str(args.lambda_lpips),
                    "--lambda_dssim", str(args.lambda_dssim),
                    "--opacity_reset_interval", str(args.opacity_reset_interval),
                    "--position_lr_max_steps", str(args.position_lr_max_steps),
                    "--disable_viewer",
                ] + (["--antialiasing"] if args.antialiasing else []) + [
                    "--quiet",
                ]
                # Reduce CUDA memory fragmentation; garbage_collection_threshold frees unused
                # cached blocks more aggressively to avoid fragmentation OOM
                extra_env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8"}
                run(train_cmd, desc=f"Training {scene}", extra_env=extra_env)

        # ------------------------------------------------------------------ #
        # RENDER
        # ------------------------------------------------------------------ #
        if True:  # not args.skip_render
            if not os.path.isfile(scene_test_csv):
                print(f"[SKIP RENDER] {scene}: no test_poses.csv found at {scene_test_csv}")
                continue

            if not has_finished_training(output_dir):
                print(f"[SKIP RENDER] {scene}: no trained model found at {output_dir}, cannot render.")
                continue

            render_cmd = [
                sys.executable, "render_test_poses.py",
                "-m", output_dir,
                "--test_poses", scene_test_csv,
                "--output_dir", render_out_dir,
                "--resolution_scale", str(args.resolution_scale),
            ] + (["--antialiasing"] if args.antialiasing else []) + [
                "--quiet",
            ]
            extra_env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}
            run(render_cmd, desc=f"Rendering test poses for {scene}", extra_env=extra_env)

    print("\nAll done.")


if __name__ == "__main__":
    main()

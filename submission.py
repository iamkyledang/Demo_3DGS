"""
Format rendered results into the required submission folder structure.

Output layout:
  submission/
  ├── <scene>/
  │   ├── 0001.png
  │   ├── 0002.png
  │   └── ...
  └── ...

Images are numbered in the order they appear in each scene's test_poses.csv.

Usage:
    python submission.py
    python submission.py --data_root /mnt/passport/VAI_NVS_DATA_ROUND2 \
                         --output_root output \
                         --submission_dir submission
    python submission.py --scenes bonsai chair HCM0421
"""

import os
import csv
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/passport/VAI_NVS_DATA_ROUND2",
                        help="Root directory containing scene folders with test_poses.csv")
    parser.add_argument("--output_root", default="output",
                        help="Root directory containing rendered outputs per scene")
    parser.add_argument("--submission_dir", default="submission",
                        help="Destination folder for the formatted submission")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Subset of scenes to process. Default: all scenes in data_root.")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root)
    submission_dir = os.path.abspath(args.submission_dir)

    # Discover scenes from data_root
    all_scenes = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    scenes = args.scenes if args.scenes else all_scenes

    print(f"Scenes to package: {scenes}")
    print(f"Submission directory: {submission_dir}\n")

    total_copied = 0
    total_missing = 0

    for scene in scenes:
        test_csv = os.path.join(data_root, scene, "test", "test_poses.csv")
        renders_dir = os.path.join(output_root, scene, "test_renders")
        scene_out_dir = os.path.join(submission_dir, scene)

        print(f"--- {scene} ---")

        if not os.path.isfile(test_csv):
            print(f"  [SKIP] test_poses.csv not found: {test_csv}")
            continue

        if not os.path.isdir(renders_dir):
            print(f"  [SKIP] renders directory not found: {renders_dir}")
            continue

        os.makedirs(scene_out_dir, exist_ok=True)

        # Read the ordered list of images from test_poses.csv
        with open(test_csv, newline="") as f:
            reader = csv.DictReader(f)
            image_names = [row["image_name"] for row in reader]

        copied = 0
        missing = 0
        for image_name in image_names:
            src = os.path.join(renders_dir, image_name)
            dst = os.path.join(scene_out_dir, image_name)

            if not os.path.isfile(src):
                print(f"  [MISSING] {src}")
                missing += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

        print(f"  Copied: {copied}/{len(image_names)}  Missing: {missing}")
        total_copied += copied
        total_missing += missing

    print(f"\nDone. Total copied: {total_copied}  Total missing: {total_missing}")
    if total_missing == 0:
        print("All images copied successfully. Submission folder is ready.")
    else:
        print("WARNING: Some images were missing. Check the output above.")


if __name__ == "__main__":
    main()

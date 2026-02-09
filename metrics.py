#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.image_utils import lpips_score, ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
import faulthandler
faulthandler.enable()


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)


def mse(a, b):
    return torch.mean((a - b) ** 2)


# -------------------------------------------------
# Image Readers
# -------------------------------------------------

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    for fname in os.listdir(renders_dir):
        if fname.endswith('.mp4'):
            continue

        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())

        image_names.append(fname)

    return renders, gts, image_names


def readDepthImages(depth_dir, gt_depth_dir):
    depths = []
    gt_depths = []
    image_names = []

    if not depth_dir.exists() or not gt_depth_dir.exists():
        return depths, gt_depths, image_names

    for fname in os.listdir(depth_dir):
        if fname.endswith('.mp4'):
            continue

        depth = np.array(Image.open(depth_dir / fname))
        gt_depth = np.array(Image.open(gt_depth_dir / fname))

        # Do NOT clamp to 3 channels — depth may be single channel
        depths.append(tf.to_tensor(depth).unsqueeze(0).cuda())
        gt_depths.append(tf.to_tensor(gt_depth).unsqueeze(0).cuda())

        image_names.append(fname)

    return depths, gt_depths, image_names


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}

    print("")

    with torch.no_grad():
        for scene_dir in model_paths:

            print("Scene:", scene_dir)

            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):

                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method

                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                depth_dir = method_dir / "depth"
                gt_depth_dir = method_dir / "gt_depth"

                # ---------------- RGB metrics ----------------
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="RGB Metric evaluation"):
                    render, gt = renders[idx], gts[idx]

                    psnrs.append(psnr(render, gt))
                    ssims.append(ssim(render, gt))
                    lpipss.append(lpips_score(render, gt))

                mean_ssim = torch.tensor(ssims).mean()
                mean_psnr = torch.tensor(psnrs).mean()
                mean_lpips = torch.tensor(lpipss).mean()

                print("Scene:", scene_dir, "SSIM :", f"{mean_ssim:>12.7f}")
                print("Scene:", scene_dir, "PSNR :", f"{mean_psnr:>12.7f}")
                print("Scene:", scene_dir, "LPIPS:", f"{mean_lpips:>12.7f}")

                # ---------------- Depth metrics ----------------
                depths, gt_depths, depth_names = readDepthImages(depth_dir, gt_depth_dir)

                depth_mses = []

                if len(depths) > 0:
                    for idx in tqdm(range(len(depths)), desc="Depth Metric evaluation"):
                        d, gtd = depths[idx], gt_depths[idx]
                        depth_mses.append(mse(d, gtd))

                    mean_depth_mse = torch.tensor(depth_mses).mean()

                    print("Scene:", scene_dir, "DEPTH MSE:", f"{mean_depth_mse:>12.7f}")
                else:
                    mean_depth_mse = torch.tensor(float("nan"))
                    print("No depth data found.")

                print("")

                # ---------------- Save results ----------------
                full_dict[scene_dir][method].update({
                    "SSIM": mean_ssim.item(),
                    "PSNR": mean_psnr.item(),
                    "LPIPS": mean_lpips.item(),
                    "DEPTH_MSE": mean_depth_mse.item()
                })

                per_view_dict[scene_dir][method].update({
                    "SSIM": {n: v for n, v in zip(image_names, torch.tensor(ssims).tolist())},
                    "PSNR": {n: v for n, v in zip(image_names, torch.tensor(psnrs).tolist())},
                    "LPIPS": {n: v for n, v in zip(image_names, torch.tensor(lpipss).tolist())},
                    "DEPTH_MSE": {n: v for n, v in zip(depth_names, torch.tensor(depth_mses).tolist())}
                    if len(depth_mses) > 0 else {}
                })

            # Save per scene
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)

            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluation script")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])

    args = parser.parse_args()
    evaluate(args.model_paths)

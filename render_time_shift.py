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

import os
import imageio
import numpy as np
import torch
from time import time
from argparse import ArgumentParser

from scene import Scene
from scene import GaussianModel
from gaussian_renderer import render

from utils.general_utils import safe_state
from arguments import (
    ModelParams,
    PipelineParams,
    ModelHiddenParams,
    OptimizationParams,
    get_combined_args,
)
from utils.image_utils import psnr, lpips_score
from utils.loss_utils import ssim
import json

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def render_time_shifted_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
):
    """
    For each view i:
      - render with camera of view i
      - use time from view i + shift
      - save GT from view i + shift
      - skip last shift views
    """

    base_path = os.path.join(
        model_path, name, f"ours_{iteration}", "time_shifted"
    )
    render_path = os.path.join(base_path, "renders_time_shift")
    gt_path = os.path.join(base_path, "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    if views is None or len(views) < 2:
        print(f"[{name}] Not enough views, skipping.")
        return

    render_images = []
    gt_images = []

    time1 = time()

    shift = 3

    for idx in range(len(views) - shift):
        view_curr = views[idx]
        view_next = views[idx + shift]

        # --- GT from shifted view ---
        gt = view_next.original_image[0:3, :, :]
        gt_np = to8b(gt.permute(1, 2, 0))
        gt_images.append(gt)

        imageio.imwrite(
            os.path.join(gt_path, f"{idx:05d}.png"),
            gt_np,
        )

        # --- Render with current camera, shifted time ---
        time_view = torch.tensor(view_next.time, device="cuda")

        rendering = render(
            view_curr,
            gaussians,
            pipeline,
            background,
            time=time_view,
        )

        render_img = rendering["render"].cpu()
        render_images.append(render_img)

        render_np = to8b(render_img.permute(1, 2, 0))

        imageio.imwrite(
            os.path.join(render_path, f"{idx:05d}.png"),
            render_np,
        )

    time2 = time()
    print(f"[{name}] FPS:", (len(views) - shift) / (time2 - time1))

    # ---- Save rendered video ----
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array * 255).clip(0, 255).numpy().astype(np.uint8)
    render_array = np.ascontiguousarray(render_array)[..., :3]

    imageio.mimwrite(
        os.path.join(base_path, "render_time_shifted.mp4"),
        render_array,
        fps=30,
        quality=8,
    )

    # ---- Save GT video ----
    gt_array = torch.stack(gt_images, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array * 255).clip(0, 255).numpy().astype(np.uint8)
    gt_array = np.ascontiguousarray(gt_array)[..., :3]

    imageio.mimwrite(
        os.path.join(base_path, "gt_current_view.mp4"),
        gt_array,
        fps=30,
        quality=8,
    )

    # ==============================
    # Metric computation (SSIM / PSNR / LPIPS)
    # ==============================
    if name =="test":
        metrics = {
            "SSIM": [],
            "PSNR": [],
            "LPIPS": [],
        }

        per_view_metrics = {}

        for idx, (render_img, gt_img) in enumerate(zip(render_images, gt_images)):
            render_t = render_img.unsqueeze(0).cuda()
            gt_t = gt_img.unsqueeze(0).cuda()

            psnr_val = psnr(render_t, gt_t).item()
            ssim_val = ssim(render_t, gt_t).item()
            lpips_val = lpips_score(render_t, gt_t).item()

            metrics["PSNR"].append(psnr_val)
            metrics["SSIM"].append(ssim_val)
            metrics["LPIPS"].append(lpips_val)

            per_view_metrics[f"{idx:05d}.png"] = {
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "LPIPS": lpips_val,
            }

        mean_metrics = {
            "PSNR": float(np.mean(metrics["PSNR"])),
            "SSIM": float(np.mean(metrics["SSIM"])),
            "LPIPS": float(np.mean(metrics["LPIPS"])),
        }

        # ---- Save JSONs ----
        with open(os.path.join(base_path, "metrics.json"), "w") as f:
            json.dump(mean_metrics, f, indent=4)

        with open(os.path.join(base_path, "per_view_metrics.json"), "w") as f:
            json.dump(per_view_metrics, f, indent=4)

        print(
            f"[{name}] Metrics — "
            f"PSNR: {mean_metrics['PSNR']:.4f}, "
            f"SSIM: {mean_metrics['SSIM']:.4f}, "
            f"LPIPS: {mean_metrics['LPIPS']:.4f}"
        )


def render_sets(
    dataset: ModelParams,
    optimization,
    hyperparam,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        if not skip_train:
            render_time_shifted_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_time_shifted_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_video:
            render_time_shifted_set(
                dataset.model_path,
                "video",
                scene.loaded_iter,
                scene.getVideoCameras(),
                gaussians,
                pipeline,
                background,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Time-shifted rendering script")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)

    args = get_combined_args(parser)

    print("Rendering", args.model_path)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        op.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
    )

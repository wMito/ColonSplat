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
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
from scene import GaussianModel
import json
from utils.image_utils import psnr, lpips_score
from utils.loss_utils import ssim

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)



def mse(a, b):
    return torch.mean((a - b) ** 2)




def render_custom_trajectory(lookat_folder, white_background=False):
    """Load camera transforms from a lookat subfolder."""
    from scene.endo_loader import ColonSplat_Dataset
    
    transform_file = os.path.join(lookat_folder, "transforms.json")
    if not os.path.exists(transform_file):
        return None
    
    # Use the readCamerasFromTransforms logic
    dataset = ColonSplat_Dataset(lookat_folder, downsample=1.0)
    cameras = dataset.readCamerasFromTransforms(lookat_folder, "transforms.json", white_background)
    return cameras


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, source_path):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "custom_trajectory")
    makedirs(render_path, exist_ok=True)
    
    test_folder = os.path.join(source_path, "test")
    
        
    test_cameras = render_custom_trajectory(test_folder, white_background=(background == torch.tensor([1,1,1], device="cuda")).all())
        
    render_images = []
    gt_images = []
    render_depths = []
    gt_depths = []
    for cam_idx, test_cam in enumerate(tqdm(test_cameras, desc=f"  Rendering test", leave=False)):
        small_scales = None
        time_view = test_cam.time
        
        rendering = render(test_cam, gaussians, pipeline, background, time=time_view, override_scales=small_scales)
        render_images.append(rendering["render"].cpu())
        render_depths.append((rendering["depth"]/ rendering["depth"].max()).cpu())
        gt_depth = test_cam.depth / test_cam.depth.max()
        gt_depths.append(gt_depth)
        gt_img = test_cam.original_image[0:3, :, :]
        gt_images.append(gt_img)
    

    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).numpy().astype(np.uint8)
    render_array = np.ascontiguousarray(render_array)
    render_array = render_array[..., :3]
    
    output_video = os.path.join(render_path, f'custom_trajectory_video.mp4')
    imageio.mimwrite(output_video, render_array, fps=30, quality=8)
    print(f"  Saved video to {output_video}")
    count = 0
    render_path_lookat = os.path.join(render_path, f"custom_trajectory_renders")
    makedirs(render_path_lookat, exist_ok=True)
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path_lookat, '{0:05d}'.format(count) + ".png"))
            count +=1
    

    metrics = {
        "SSIM": [],
        "PSNR": [],
        "LPIPS": [],
        "MSE": []
    }

    per_view_metrics = {}

    for idx, (render_img, gt_img, render_depth, gt_depth) in enumerate(zip(render_images, gt_images, render_depths, gt_depths)):
        render_t = render_img.unsqueeze(0).cuda()
        gt_t = gt_img.unsqueeze(0).cuda()

        psnr_val = psnr(render_t, gt_t).item()
        ssim_val = ssim(render_t, gt_t).item()
        lpips_val = lpips_score(render_t, gt_t).item()
        mse_val = mse(render_depth, gt_depth).item()

        metrics["PSNR"].append(psnr_val)
        metrics["SSIM"].append(ssim_val)
        metrics["LPIPS"].append(lpips_val)
        metrics["MSE"].append(mse_val)

        per_view_metrics[f"{idx:05d}.png"] = {
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "LPIPS": lpips_val,
            "MSE": mse_val
        }

    mean_metrics = {
        "PSNR": float(np.mean(metrics["PSNR"])),
        "SSIM": float(np.mean(metrics["SSIM"])),
        "LPIPS": float(np.mean(metrics["LPIPS"])),
        "MSE": float(np.mean(metrics["MSE"]))
    }



    # ---- Save JSONs ----
    with open(os.path.join(render_path, "metrics.json"), "w") as f:
        json.dump(mean_metrics, f, indent=4)

    with open(os.path.join(render_path, "per_view_metrics.json"), "w") as f:
        json.dump(per_view_metrics, f, indent=4)

    print(
        f"[{name}] Metrics — "
        f"PSNR: {mean_metrics['PSNR']:.4f}, "
        f"SSIM: {mean_metrics['SSIM']:.4f}, "
        f"LPIPS: {mean_metrics['LPIPS']:.4f}, "
        f"MSE: {mean_metrics['MSE']:.4f}"
    )

def render_sets(dataset : ModelParams, optimization, hyperparam, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        

        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.source_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), 
                op.extract(args), 
                hyperparam.extract(args), 
                args.iteration, 
                pipeline.extract(args))
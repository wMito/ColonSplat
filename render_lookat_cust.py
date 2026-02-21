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
from time import time
import glob
import re

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def load_lookat_cameras(lookat_folder, white_background=False):
    """Load camera transforms from a lookat subfolder."""
    from scene.endo_loader import ColonSplat_Dataset
    
    transform_file = os.path.join(lookat_folder, "transforms.json")
    if not os.path.exists(transform_file):
        return None
    
    # Use the readCamerasFromTransforms logic
    dataset = ColonSplat_Dataset(lookat_folder, downsample=1.0)
    cameras = dataset.readCamerasFromTransforms(lookat_folder, "transforms.json", white_background)
    return cameras

def discover_lookat_folders(source_path):
    """Discover all lookat_{num} subfolders in source_path/lookat_cams."""
    lookat_cams_dir = os.path.join(source_path, "lookat_cams")
    if not os.path.exists(lookat_cams_dir):
        print(f"Warning: {lookat_cams_dir} does not exist")
        return []
    
    # Find all lookat_{num} directories
    lookat_folders = glob.glob(os.path.join(lookat_cams_dir, "lookat_*"))
    lookat_folders = [f for f in lookat_folders if os.path.isdir(f)]
    
    # Sort by number
    def extract_num(path):
        match = re.search(r'lookat_(\d+)$', path)
        return int(match.group(1)) if match else 0
    
    lookat_folders.sort(key=extract_num)
    return lookat_folders

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, source_path):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lookat_cameras")
    makedirs(render_path, exist_ok=True)
    
    lookat_folders = discover_lookat_folders(source_path)
    
    if not lookat_folders:
        print("No lookat camera folders found, skipping rendering")
        return
    
    print(f"Found {len(lookat_folders)} lookat camera folders")
    
    
    for lookat_folder in tqdm(lookat_folders, desc="Processing lookat cameras"):
        lookat_name = os.path.basename(lookat_folder)
        print(f"\nProcessing {lookat_name}...")
        
        lookat_cameras = load_lookat_cameras(lookat_folder, white_background=(background == torch.tensor([1,1,1], device="cuda")).all())
        
        if lookat_cameras is None:
            print(f"  Skipping {lookat_name} - no valid transforms.json found")
            continue
        
        print(f"  Loaded {len(lookat_cameras)} cameras from {lookat_name}")
        
        render_images = []
        
        for cam_idx, lookat_cam in enumerate(tqdm(lookat_cameras, desc=f"  Rendering {lookat_name}", leave=False)):
            small_scales = None
            time_view = lookat_cam.time
            
            rendering = render(lookat_cam, gaussians, pipeline, background, time=time_view, override_scales=small_scales)
            render_images.append(rendering["render"].cpu())
        
        render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
        render_array = (render_array*255).clip(0, 255).numpy().astype(np.uint8)
        render_array = np.ascontiguousarray(render_array)
        render_array = render_array[..., :3]
        
        output_video = os.path.join(render_path, f'{lookat_name}_video.mp4')
        imageio.mimwrite(output_video, render_array, fps=30, quality=8)
        print(f"  Saved video to {output_video}")
        count = 0
        render_path_lookat = os.path.join(render_path, f"{lookat_name}_renders")
        makedirs(render_path_lookat, exist_ok=True)
        print("writing rendering images.")
        if len(render_images) != 0:
            for image in tqdm(render_images):
                torchvision.utils.save_image(image, os.path.join(render_path_lookat, '{0:05d}'.format(count) + ".png"))
                count +=1

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
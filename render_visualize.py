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
import open3d as o3d
from utils.graphics_utils import fov2focal
import json
from utils.lookat_utils import generateLookAtCams
import cv2

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "lookat_cameras")
    makedirs(render_path, exist_ok=True)
    

    gt_list = []
    test_times = 1
    time_view=None

    
    # this is camera from first frame
    view_zero = views[0]

    ## this generates  lookAt cameras settings to override:
    ## new camera location and rotation so that camera look at colon from far away
    xyz = gaussians.get_xyz
    radius = 100
    n_directions = 10 #here we set from how many directions (camers) we want to see the colon. set as many as you wish
    lookat_cameras = generateLookAtCams(xyz, view_zero, radius, n_directions = n_directions)

    lookat_cameras.append(None) # lets also append None, which results in original settings from camera zero
    ##

    for lookat_idx, lookat_cam_settings in enumerate(lookat_cameras):
        render_images = []
    
        for i in range(test_times):
            if views is None:
                break
            for idx, view in enumerate(views):
                if idx == 0 and i == 0:
                    time1 = time()
                
                small_scales = None #gaussians.get_scaling.clamp_max(gaussians.spatial_lr_scale*1.0)
                time_view = torch.tensor(view.time)
                rendering = render(view_zero, gaussians, pipeline, background, time=time_view, \
                                   override_scales=small_scales, override_raster_settings=lookat_cam_settings)
                
                if i == test_times-1:
                    render_images.append(rendering["render"].cpu())
                    
                    if name in ["train", "test", "video"]:
                        gt = view.original_image[0:3, :, :]
                        gt_list.append(gt)

        time2=time()
        print("FPS:",(len(views))*test_times/(time2-time1))
                
        render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
        render_array = (render_array*255).clip(0, 255).numpy().astype(np.uint8)
        render_array = np.ascontiguousarray(render_array)
        render_array = render_array[..., :3]
        # [cv2.putText(f, f"Static camera {lookat_idx}", (20,40),
        #      cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2) for f in render_array]
        imageio.mimwrite(os.path.join(render_path, f'ours_video_gaussians_from_cameralookat{lookat_idx}.mp4'), render_array, fps=30, quality=8)
        
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).numpy().astype(np.uint8)
    gt_array = np.ascontiguousarray(gt_array)
    gt_array = gt_array[..., :3]
    # [cv2.putText(f, f"GT - test video", (20,40),
    #         cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2) for f in gt_array]
    imageio.mimwrite(os.path.join(render_path, 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    


def render_sets(dataset : ModelParams, optimization, hyperparam, iteration : int, pipeline : PipelineParams, \
    skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
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
                pipeline.extract(args), 
                args.skip_train, 
                args.skip_test,
                args.skip_video)
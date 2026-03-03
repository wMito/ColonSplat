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
import cv2


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    videos_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    pcd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pcd")
    render_no_dcol_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_no_dcol")

    makedirs(render_path, exist_ok=True)
    makedirs(render_no_dcol_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    makedirs(pcd_path, exist_ok=True)
    
    
    render_images = []
    render_depths = []
    render_images_no_dcol = []
    gt_list = []
    gt_depths = []
    mask_list = []

    test_times = 1
    time_view=None

    
    for i in range(test_times):
        if views is None:
            break
        for idx, view in enumerate(views):
            if idx == 0 and i == 0:
                time1 = time()
            
            rendering = render(view, gaussians, pipeline, background, time=time_view, show_no_dcol=True)
            
            if i == test_times-1:
                render_depths.append((rendering["depth"]/ rendering["depth"].max()).cpu())
                render_images.append(rendering["render"].cpu())
                render_images_no_dcol.append(rendering["render_no_dcol"].cpu())

                current_means = rendering["transformed_points"]
                pts = current_means.cpu().numpy()
                pcd_out = o3d.geometry.PointCloud()
                pcd_out.points = o3d.utility.Vector3dVector(pts)
                ply_path = os.path.join(pcd_path, 'frame_{:05d}.ply'.format(idx))
                o3d.io.write_point_cloud(ply_path, pcd_out)
                
                if name in ["train", "test", "video"]:
                    gt = view.original_image[0:3, :, :]
                    gt_list.append(gt)
                    mask = view.mask
                    mask_list.append(mask)
                    gt_depth = view.depth / view.depth.max()
                    gt_depths.append(gt_depth)

    time2=time()
    print("FPS:",(len(views))*test_times/(time2-time1))
    # import pdb; pdb.set_trace()
    count = 0
    print("writing images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
            
    count = 0
    print("writing rendering images (no dcol).")
    if len(render_images_no_dcol) != 0:
        for image in tqdm(render_images_no_dcol):
            torchvision.utils.save_image(
                image,
                os.path.join(render_no_dcol_path, '{0:05d}'.format(count) + ".png")
            )
            count += 1

    # count = 0
    # print("writing mask images.")
    # if len(mask_list) != 0:
    #     for image in tqdm(mask_list):
    #         image = image.float()
    #         torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
    #         count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = image.float() / image.max()
            torchvision.utils.save_image(image, os.path.join(depth_path, '{0:05d}'.format(count) + ".png"))
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.float() / image.max()
            torchvision.utils.save_image(image, os.path.join(gt_depth_path, '{0:05d}'.format(count) + ".png"))
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).numpy().astype(np.uint8)
    render_array = np.ascontiguousarray(render_array)
    render_array = render_array[..., :3]
    # [cv2.putText(f, f"OURS - reconstruction", (20,40),
    #         cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2) for f in render_array]
    imageio.mimwrite(f"{videos_path}/ours_video_reconstruction.mp4", render_array, fps=30, quality=8)

    render_array = torch.stack(render_images_no_dcol, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array * 255).clip(0, 255).numpy().astype(np.uint8)
    render_array = np.ascontiguousarray(render_array)[..., :3]
    # [cv2.putText(f, f"OURS - no dcol", (20,40),
    #         cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2) for f in render_array]
    imageio.mimwrite(
        os.path.join(f"{videos_path}/ours_video_no_dcol.mp4"), render_array, fps=30, quality=8)

    render_array = torch.stack(render_depths, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).numpy().astype(np.uint8)
    render_array = np.ascontiguousarray(render_array)
    render_array = render_array[..., :3]
    # [cv2.putText(f, f"OURS - depth", (20,40),
    #         cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2) for f in render_array]
    imageio.mimwrite(f"{videos_path}/ours_depth.mp4", render_array, fps=30, quality=8)

    render_array = torch.stack(gt_depths, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).numpy().astype(np.uint8)
    render_array = np.ascontiguousarray(render_array)
    render_array = render_array[..., :3]
    # [cv2.putText(f, f"GT - depth", (20,40),
    #         cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2) for f in render_array]
    imageio.mimwrite(f"{videos_path}/gt_depth.mp4", render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).numpy().astype(np.uint8)
    gt_array = np.ascontiguousarray(gt_array)
    gt_array = gt_array[..., :3]
    imageio.mimwrite(f"{videos_path}/gt_video.mp4", gt_array, fps=30, quality=8)



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
                args.skip_video,)
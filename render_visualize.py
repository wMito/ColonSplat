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



to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, reconstruct=False, embeddings=None,
               embedding_idx=-1, illu_type=None, wo_restoration=False):
    render_path = os.path.join(model_path, name, "ours_vis_{}".format(iteration), "renders")
    render_restored_path = os.path.join(model_path, name, "ours_vis_{}".format(iteration), "render_restored")
    render_path_small = os.path.join(model_path, name, "ours_vis_{}".format(iteration), "renders_small")
    render_restored_path_small = os.path.join(model_path, name, "ours_vis_{}".format(iteration), "render_restored_small")


    makedirs(render_path, exist_ok=True)
    makedirs(render_restored_path, exist_ok=True)    
    makedirs(render_path_small, exist_ok=True)
    makedirs(render_restored_path_small, exist_ok=True) 

    with open(os.path.join(model_path, 'embedding_info.json'), 'r', encoding='utf-8') as file:
        embedding_dict = json.load(file)
    
    render_images = []
    render_images_restored = []
    render_images_small = []
    render_images_restored_small = []
    gt_list = []

    test_times = 1
    
    time_view=None
    embedding=None
    # illu_type = 'low_light' #cutting 'over_exposure' #pulling
    # 
    if illu_type is None and embedding_idx != -1:
        for key, sub_dict in embedding_dict:
            if sub_dict['train_count'] == embedding_idx:
                illu_type = sub_dict['illu_type']
    
    view_zero = views[0]
    for i in range(test_times):
        if views is None:
            break
        for idx, view in enumerate(views):
            if idx == 0 and i == 0:
                time1 = time()
            if embedding_idx==-1 and illu_type is None:
                if wo_restoration:
                    # varying illumination reconstruction, use the previous neighbour
                    embedding_idx = embedding_dict[str(view.uid-1)]['train_count']
                    embedding = embeddings[view.id][None]
                    illu_type = None
                    
                    # embedding=None
                    # embedding_idx = 7
                    # illu_type = 'low_light' #cutting 'over_exposure' #pulling 
                    # print('time_view', time_view)
                    # print('json embedding_idx', embedding_idx)

                else:
                    # default testing restoration reference
                    embedding_idx = 7
                    illu_type = 'low_light' if "cutting" in model_path else 'over_exposure' #pulling
                    time_view=None
            
            small_scales = gaussians.get_scaling.clamp_max(gaussians.spatial_lr_scale*0.01)
            time_view = torch.tensor(view.time)
            rendering_small = render(view_zero, gaussians, pipeline, background, embedding_idx=embedding_idx, \
                embedding=embedding, illu_type=illu_type, time=time_view, override_scales=small_scales)
            rendering = render(view_zero, gaussians, pipeline, background, embedding_idx=embedding_idx, \
                embedding=embedding, illu_type=illu_type, time=time_view)
            if i == test_times-1:
                render_images_small.append(rendering_small["render"].cpu())
                render_images_restored_small.append(rendering_small["render_restored"].cpu())
                render_images.append(rendering["render"].cpu())
                render_images_restored.append(rendering["render_restored"].cpu())

                current_means = rendering["transformed_points"]
                pts = current_means.cpu().numpy()
                pcd_out = o3d.geometry.PointCloud()
                pcd_out.points = o3d.utility.Vector3dVector(pts)

                
                if name in ["train", "test", "video"]:
                    gt = view.original_image[0:3, :, :]
                    gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views))*test_times/(time2-time1))
    # import pdb; pdb.set_trace()
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
            
    count = 0
    print("writing rendering images restored.")
    if len(render_images_restored) != 0:
        for image in tqdm(render_images_restored):
            torchvision.utils.save_image(image, os.path.join(render_restored_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendering images small.")
    if len(render_images_small) != 0:
        for image in tqdm(render_images_small):
            torchvision.utils.save_image(image, os.path.join(render_path_small, '{0:05d}'.format(count) + ".png"))
            count +=1
            
    count = 0
    print("writing rendering images restored small.")
    if len(render_images_restored_small) != 0:
        for image in tqdm(render_images_restored_small):
            torchvision.utils.save_image(image, os.path.join(render_restored_path_small, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255)
    imageio.mimwrite(os.path.join(model_path, name, "ours_vis_{}".format(iteration), 'ours_video_gaussians.mp4'), render_array, fps=30, quality=8)

    render_array = torch.stack(render_images_restored, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255)
    imageio.mimwrite(f"{render_restored_path}/render_gaussians.mp4", render_array, fps=30, quality=8)

    render_array = torch.stack(render_images_small, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255)
    imageio.mimwrite(os.path.join(model_path, name, "ours_vis_{}".format(iteration), 'ours_video_small_gaussians.mp4'), render_array, fps=30, quality=8)

    render_array = torch.stack(render_images_restored_small, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255)
    imageio.mimwrite(f"{render_restored_path_small}/render_small_gaussians.mp4", render_array, fps=30, quality=8)

    
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255)
    imageio.mimwrite(os.path.join(model_path, name, "ours_vis_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    


def render_sets(dataset : ModelParams, optimization, hyperparam, iteration : int, pipeline : PipelineParams, \
    skip_train : bool, skip_test : bool, skip_video: bool, pc: bool, embedding_idx=-1, illu_type=None, wo_restoration=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, reconstruct=pc,
                       embedding_idx=embedding_idx, illu_type=illu_type, wo_restoration=wo_restoration)
        if not skip_test:
            # test_embeddings = gaussians.optimize_embeddings(scene.getTestCameras(), dataset, optimization, pipeline)
            test_embeddings = None
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, reconstruct=pc, \
                embeddings=test_embeddings, embedding_idx=embedding_idx, illu_type=illu_type, wo_restoration=wo_restoration)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, reconstruct=pc,
                       embedding_idx=embedding_idx, illu_type=illu_type, wo_restoration=wo_restoration)

def reconstruct_point_cloud(images, masks, depths, camera_parameters, name, pcd_path):
    import cv2
    import copy
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        depth_np = depth_np.squeeze(0)
        mask = masks[i_frame]
        mask = mask.squeeze(0).cpu().numpy()
        
        rgb_new = copy.deepcopy(rgb_np)

        depth_smoother = (128, 64, 64)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
        
        close_depth = np.percentile(depth_np[depth_np!=0], 5)
        inf_depth = np.percentile(depth_np, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, width / 2),
            project_valid_depth_only=True
        )
        pc_path = os.path.join(pcd_path, 'frame_{}.ply'.format(i_frame))
        print('pcd:', pc_path)
        o3d.io.write_point_cloud(pc_path, pcd)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--embedding_idx", default=-1, type=int)
    #parser.add_argument("--illumination_type", default=None, type=str)
    parser.add_argument("--wo_restoration", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pc", action="store_true")
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
                args.skip_video,
                args.pc,
                args.embedding_idx,
                #args.illumination_type,
                args.wo_restoration)
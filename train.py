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
import torchvision
import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import cv2
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from torchmetrics.functional.regression import pearson_corrcoef
from utils import helper
from utils.resnet_swag import content_loss
from torchvision import transforms, models


import lpips
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# resnet = models.resnet50(pretrained=True)
# resnet.to('cuda').eval().float()


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    
    # centerline = scene.centerline
    # if centerline is not None:
    #     cl = torch.from_numpy(centerline).float().cuda()
    #     cl_tangent = cl[1:] - cl[:-1]
    #     cl_tangent = cl_tangent / (cl_tangent.norm(dim=1, keepdim=True) + 1e-8)
    # scene.centerline_tangent = cl_tangent
    # scene.centerline_points = cl[:-1]


    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration, stage=stage)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            # idx = 0
            idx = randint(0, len(viewpoint_stack)-1)
            # idx = iteration % len(viewpoint_stack)
        else:
            idx = randint(0, len(viewpoint_stack)-1)
            # idx = iteration % len(viewpoint_stack)
            
        viewpoint_cams = [viewpoint_stack[idx]]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        images_concealing = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        transformed_means_list = []
        transformed_rotations_list =[]
        transformed_scales_list =[]
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii, transformed_means, \
                transformed_rotations, transformed_scales, dcol = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], \
                    render_pkg["visibility_filter"], render_pkg["radii"], \
                        render_pkg["transformed_points"], render_pkg["transformed_rotations"], \
                            render_pkg["transformed_scales"], render_pkg["dcol"]
                        
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask
            if mask:
                mask = mask.cuda()
            
            image_concealing = render_pkg['render']
            transformed_means_list.append(transformed_means)
            transformed_rotations_list.append(transformed_rotations)
            transformed_scales_list.append(transformed_scales)
            images_concealing.append(image_concealing.unsqueeze(0))
            images.append(image.unsqueeze(0))
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if gt_depth.ndim == 2:
                gt_depth = gt_depth.unsqueeze(0)
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            if mask:
                masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        transformed_means = torch.cat(transformed_means_list, 0)
        transformed_rotations = torch.cat(transformed_rotations_list, 0)
        transformed_scales = torch.cat(transformed_scales_list, 0)
        images_concealing = torch.cat(images_concealing, 0)
        if mask:
            masks = torch.cat(masks, 0)
        
        loss = 0

        ## Ll1 loss
        if mask:
            Ll1 = l1_loss(images_concealing, gt_images, masks)
        else:
            Ll1 = l1_loss(images_concealing, gt_images)
        loss += Ll1

        ### DEPTH LOSS
        if mask:
            depth_loss = opt.depth_weight * l1_loss(torch.clamp(rendered_depths/(rendered_depths.max()+1e-6), 0, 1), \
            torch.clamp(gt_depths/(gt_depths.max()+1e-6), 0, 1), masks, True)
        else:
            depth_loss = opt.depth_weight * l1_loss(torch.clamp(rendered_depths/(rendered_depths.max()+1e-6), 0, 1), \
            torch.clamp(gt_depths/(gt_depths.max()+1e-6), 0, 1))
        loss +=depth_loss

        ## XYZ CLUSTER LOSS
        idx = gaussians.closest_point_indices
        K = int(idx.max().item()) + 1

        cluster_pts = transformed_means[idx]
        cluster_means = cluster_pts.mean(dim=1)
        center_points = transformed_means[torch.arange(cluster_means.shape[0], device=cluster_means.device)]
        per_point_sq_dist = ((center_points - cluster_means) ** 2).sum(dim=1)

        loss_clusters = opt.knn_weight * per_point_sq_dist.mean()
        loss += loss_clusters

        ## NORMAL CLUSTER LOSS - very slow so comment out
        # dir_pp_camera = (transformed_means - viewpoint_cam.camera_center.repeat(transformed_means.shape[0], 1).cuda())
        # transformed_normals = gaussians.get_gaussian_normal(transformed_rotations,transformed_scales, view_dir = dir_pp_camera)

        # render_tmp = render(viewpoint_cam, gaussians, pipe, background, stage=stage, override_color=(transformed_normals+1/2))
        # render_tmp["render"]
        # rendering = render_tmp["render"].clamp(0,1)
        # torchvision.utils.save_image(rendering, os.path.join("/home/jk/colon_dynamic/thrash", 'normal_iter.png'))
        

        # cluster_q = transformed_normals[idx]                         # (N, K, 4)
        # cluster_q_mean = cluster_q.mean(dim=1)     # (N,4)
        # cluster_q_mean = cluster_q_mean / (cluster_q_mean.norm(dim=-1, keepdim=True) + 1e-8)

        # center_q = transformed_normals[torch.arange(cluster_q_mean.shape[0], device=transformed_normals.device)]

        # dot = torch.sum(center_q * cluster_q_mean, dim=-1).abs().clamp(max=1.0)
        # rot_dist = 1.0 - dot

        # loss_rot_smooth = opt.knn_weight * rot_dist.mean()/5

        ### MINIMIZE DCOLOR UPDATE - DCOL LOSS
        loss_dcol = torch.mean(dcol ** 2)*opt.dcol_weight
        loss += loss_dcol

        
        # ### CENTERLINE LOSS
        # xyz0 = gaussians._xyz          # (N,3)
        # xyz1 = transformed_means      # (N,3)

        # delta = xyz1 - xyz0            # Δxyz
        # dist2 = torch.cdist(
        #     xyz0,
        #     scene.centerline_points
        # )

        # closest_idx = torch.argmin(dist2, dim=1)
        # tangent = scene.centerline_tangent[closest_idx]
        # delta_parallel = torch.sum(delta * tangent, dim=1)
        # loss_centerline = (delta_parallel ** 2).mean()*opt.centerline_weight #0.05
        # loss += loss_centerline

        ###  LOSS TV, breaks our colon so opt.tv_weight set to zero by default
        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = opt.tv_weight * (img_tvloss) # + depth_tvloss)
        loss += tv_loss

        if mask:
            psnr_ = psnr(images_concealing, gt_images, masks).mean().double()      
        else:
            psnr_ = psnr(images_concealing, gt_images).mean().double()  
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ Loss NaN/Inf at iteration {iteration}")
            
        loss.backward()
        for group in gaussians.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    torch.nn.utils.clip_grad_value_(param, clip_value=0.5)

            # torch.nn.utils.clip_grad_norm_(param, max_norm=1)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                string_dict = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                "psnr": f"{psnr_:.{2}f}",
                                "point":f"{total_point}"}
                # if stage == "fine" and hyper.time_smoothness_weight != 0:
                progress_bar.set_postfix(string_dict)
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            metrics = {'Ll1':Ll1, 'depth_loss':depth_loss, 'elapsed':iter_start.elapsed_time(iter_end), \
                'psnr':psnr_}
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                            (pipe, background, 1.,))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - \
                        opt.opacity_threshold_fine_after)/(opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - \
                        opt.densify_grad_threshold_after)/(opt.densify_until_iter )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # print('densify', iteration, opt.densification_interval, opt.densify_from_iter)
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    # print('pruning', iteration, opt.pruning_interval, opt.pruning_from_iter)
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    if not args.no_fine:
        
        opt.pruning_interval=opt.pruning_interval_fine
        opt.pruning_from_iter=opt.pruning_from_iter
        opt.densification_interval=opt.densification_interval_fine
        opt.densify_from_iter=opt.densify_from_iter_fine
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000, 15000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")

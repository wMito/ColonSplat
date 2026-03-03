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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
    override_color = None, stage="fine", \
    time=None, override_scales = None, override_raster_settings=None, show_no_dcol=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
        
    if override_raster_settings is None:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            require_coord = True, #for rade-gs rasterizer
            require_depth = True,
            kernel_size = 0
        )
    
    else:
        raster_settings = override_raster_settings

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means3D = pc.get_xyz
    color_emb = pc.get_deformation_table

    if time is None:
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        time = time.to(means3D.device).repeat(means3D.shape[0],1)

    means2D = screenspace_points
    opacity = pc._opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    
    
    colors_precomp = pc.features
    
    if stage == "coarse" :
        means3D_final, scales_final, rotations_final, opacity_final, color_final, dcol = means3D, scales, rotations, opacity, colors_precomp, colors_precomp*0 #hack to return 0 instead of "dcol"
    else:
        means3D_final, scales_final, rotations_final, opacity_final, (color_final, dcol) = pc._deformation(means3D, scales, 
                                                                            rotations, opacity, colors_precomp, time, color_emb)
    
    scales_final = pc.scaling_activation(scales_final)

    if override_scales is not None:
        scales_final = override_scales
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
        
   
    
    #scale s0 clamping for games - optional
    # eps_s0 = 10e-6
    # keep_axes = [i for i in range(3) if i != pc.games_flatten_axis]
    # s0 = torch.ones(pc._scaling.shape[0], 1).cuda() * eps_s0
    scales_final_flat = scales_final.clamp_max(0.05*pc.spatial_lr_scale) #0.5) #torch.cat([scales_final[:, keep_axes], s0], dim=1)

    if override_color is not None:
        color_final = override_color

    # rendered_image, radii, depth = rasterizer(
    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer( #for radegs rasterizer
        means3D = means3D_final,
        means2D = means2D,
        shs = None, #shs*pc.get_concealing[:, None, :]
        colors_precomp = color_final,
        opacities = opacity,
        scales = scales_final_flat,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image_no_dcol = None
    if show_no_dcol:
        # rendered_image_no_dcol,_,_ = rasterizer(
        rendered_image_no_dcol, _, _, _, _, _, _, _ = rasterizer( #for radegs rasterizer
            means3D = means3D_final,
            means2D = means2D,
            shs = None, #shs*pc.get_concealing[:, None, :]
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales_final_flat,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)

    depth = rendered_expected_depth if pipe.depth_mode == "expected" else rendered_median_depth
    return {"render": rendered_image,
            "render_no_dcol": rendered_image_no_dcol,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transformed_points": means3D_final,
            "transformed_color": color_final,
            "dcol": dcol}


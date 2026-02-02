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
    override_color = None, stage="fine", embedding_idx=-1, embedding=None, illu_type=None, time=None, override_scales = None, override_raster_settings=None, show_clusters=False):
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
            debug=pipe.debug
        )
    
    else:
        raster_settings = override_raster_settings

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    if time is None:
        # print('time', viewpoint_camera.time)
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        # print('use pre_time', time.item())
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

    
    if pc.use_deformation_filt:
        deformation_point = pc.get_deformation_table > 0.3
    
    if stage == "coarse" :
        means3D_final, scales_final, rotations_final, opacity_final = means3D, scales, rotations, opacity
        
    else:
        if pc.use_deformation_filt:
            means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                            rotations[deformation_point], opacity[deformation_point],
                                                                            time[deformation_point])
            means3D_final = torch.zeros_like(means3D)
            rotations_final = torch.zeros_like(rotations)
            scales_final = torch.zeros_like(scales)
            opacity_final = torch.zeros_like(opacity)
            means3D_final[deformation_point] =  means3D_deform
            rotations_final[deformation_point] =  rotations_deform
            scales_final[deformation_point] =  scales_deform
            opacity_final[deformation_point] = opacity_deform
            means3D_final[~deformation_point] = means3D[~deformation_point]
            rotations_final[~deformation_point] = rotations[~deformation_point]
            scales_final[~deformation_point] = scales[~deformation_point]
            opacity_final[~deformation_point] = opacity[~deformation_point]
        else:
            means3D_final, scales_final, rotations_final, opacity_final = pc._deformation(means3D, scales, 
                                                                            rotations, opacity,
                                                                            time)
    # print(time.max())
    # with torch.no_grad():
    #     pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])



    scales_final = pc.scaling_activation(scales_final)

    if override_scales is not None:
        scales_final = override_scales
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    shs = None
    colors_precomp = pc.features
    
        
   
    if show_clusters:
        scales_all = scales_final.clone()
        colors_precomp = torch.ones_like(colors_precomp).to('cuda')
        scales_final = scales_final.clamp_max(pc.spatial_lr_scale*0.01)
        colors_precomp[pc.closest_point_indices[50000]] = torch.tensor([1, 0, 0.]).to('cuda')
        opacity[pc.closest_point_indices[50000]] = 1.
        scales_final[pc.closest_point_indices[50000]] = scales_all[pc.closest_point_indices[50000]] 
        colors_precomp[pc.closest_point_indices[20000]] = torch.tensor([0, 1, 0.]).to('cuda')
        opacity[pc.closest_point_indices[20000]] = 1.
        scales_final[pc.closest_point_indices[20000]] = scales_all[pc.closest_point_indices[20000]]
        colors_precomp[pc.closest_point_indices[100]] = torch.tensor([0, 0, 1.]).to('cuda')
        opacity[pc.closest_point_indices[100]] = 1.
        scales_final[pc.closest_point_indices[100]] = scales_all[pc.closest_point_indices[100]]
        # for i in [0, 10, 20, 30, 40, 50]:
        #     colors_precomp[pc.closest_point_indices[i]] = torch.tensor([1, 0, 0.]).to('cuda')
        #     opacity[pc.closest_point_indices[i]] = 1.0

    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None, #shs*pc.get_concealing[:, None, :]
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # for ablation
    # rendered_image_concealing = rendered_image
    
    # rendered_image_concealing, radii, depth = rasterizer(
    #     means3D = means3D_final,
    #     means2D = means2D,
    #     shs = None, #shs*pc.get_concealing[:, None, :]
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales_final,
    #     rotations = rotations_final,
    #     cov3D_precomp = cov3D_precomp)
    
    # for ablation
    # rendered_image_concealing = pc.spatial(rendered_image_concealing.permute(1,2,0)).permute(2,0,1)
    
    # rendered_image_concealing = pc.spatial(rendered_image_concealing, \
    #     app_embeddings)
    
    return {"render": rendered_image,
            "render_restored": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transformed_points": means3D_final}


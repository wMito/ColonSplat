from utils.graphics_utils import getProjectionMatrix, look_at
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings
from types import SimpleNamespace
from tqdm import tqdm
import math
import numpy as np

def fibonacci_sphere(n):
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~2.399963

    for i in range(n):
        y = 1 - (i / (n - 1)) * 2        # y goes 1 → -1
        radius = np.sqrt(1 - y * y)

        theta = golden_angle * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append(np.array([x, y, z]))

    return points


def position_cam(object_center, direction, radius):
    direction = direction / np.linalg.norm(direction)
    cam_position = object_center + direction * radius
    return cam_position

def generateLookAtCams(xyz, pattern_camera, radius, n_directions = 10, up_dir = np.array([0,0, -1]), downscale_ratio =1.0):
    scene_points = xyz.detach().cpu().numpy()
    object_center=scene_points.mean(axis=0)
    viewpoint_camera = pattern_camera
    scaling_modifier = 1.0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # calculate the fov and projmatrix of cam
    fx_origin = viewpoint_camera.image_width / (2. * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2. * tanfovy)

    w = int(viewpoint_camera.image_width)
    h = int(viewpoint_camera.image_height)


    raster_settings_cam_list =[]
    n_cam=0
    directions = fibonacci_sphere(n_directions)
    # for n_cam in tqdm(range(N_cameras), desc="Generating LookAt cameras"):
    for i_dir in tqdm(range(n_directions), desc="Generating LookAt cameras"):
        
        random_direction = directions[i_dir]
        cam_position = position_cam(object_center, random_direction, radius*1.5)

        ## Needed in gs3, not sure why they scaled Field of View 
        camera_position = viewpoint_camera.camera_center.detach().cpu().numpy()
        f_scale_ratio = np.sqrt(np.sum(cam_position * cam_position) / np.sum(camera_position * camera_position))
        
        fx_far = fx_origin * f_scale_ratio
        fy_far = fy_origin * f_scale_ratio

        # fx_far = fx_origin
        # fy_far = fy_origin
        
        tanfovx_far = 0.5 * w / fx_far
        tanfovy_far = 0.5 * h / fy_far

        fovx_far = 2 * math.atan(tanfovx_far)
        fovy_far = 2 * math.atan(tanfovy_far)
        

        # calculate the project matrix of LookAt camera
        world_view_transform_cam=look_at(cam_position,
                                        object_center,
                                        up_dir=up_dir)
        world_view_transform_cam=torch.tensor(world_view_transform_cam,
                                                device=viewpoint_camera.world_view_transform.device,
                                                dtype=viewpoint_camera.world_view_transform.dtype).cuda()
        cam_position = torch.tensor(cam_position,
                                      device=viewpoint_camera.world_view_transform.device, 
                                      dtype=viewpoint_camera.world_view_transform.dtype)
        
        cam_prjection_matric = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=fovx_far, fovY=fovy_far).transpose(0,1).cuda()
        full_proj_transform_cam = (world_view_transform_cam.unsqueeze(0).bmm(cam_prjection_matric.unsqueeze(0))).squeeze(0)
        

        raster_settings_cam = GaussianRasterizationSettings(
            image_height = int(h//downscale_ratio),
            image_width = int(w//downscale_ratio),
            tanfovx = tanfovx_far,
            tanfovy = tanfovy_far,
            bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier = scaling_modifier,
            viewmatrix = world_view_transform_cam,
            projmatrix = full_proj_transform_cam,
            sh_degree = 0, #will be overwritten
            campos = cam_position,
            prefiltered = False,
            debug = False,
            require_coord = True,
            require_depth = True,
            kernel_size = 0
        )

        raster_settings_cam_list.append(raster_settings_cam)

       
    return raster_settings_cam_list

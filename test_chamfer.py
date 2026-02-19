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
from typing import NamedTuple
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from scene.gaussian_model import GaussianModel
from time import time
from plyfile import PlyData
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


def save_xyz_to_ply(xyz_points, filename, normals_points=None, chunk_size=10**6, quiet=False):
    """
    Save a series of XYZ points to a PLY file.

    Args:
        xyz_points: An array of shape (N, 3) containing XYZ coordinates.
        filename: The name of the output PLY file.
        rgb_colors: An array of shape (N, 3) containing RGB colors. Defaults to white.
        chunk_size: Sizes of chunks that will be saved iteratively (reduces chances of out of memory errors)
    """

    # Ensure the points are in the correct format
    assert xyz_points.shape[1] == 3, "Input points should be in the format (N, 3)"


    total_points = xyz_points.shape[0]

    num_chunks = (total_points + chunk_size - 1) // chunk_size

    with open(filename, 'wb') as ply_file:

            # Write PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {total_points}
property float x
property float y
property float z
end_header
""" 

        ply_file.write(header.encode('utf-8'))

        for i in tqdm(range(num_chunks), position=0, leave=True, disable=quiet):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)

            points_chunk = xyz_points[start_idx:end_idx].cpu().detach().numpy()
                # Create a structured array directly
            vertex = np.zeros(points_chunk.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

            vertex['x'] = points_chunk[:, 0]
            vertex['y'] = points_chunk[:, 1]
            vertex['z'] = points_chunk[:, 2]

            ply_file.write(vertex.tobytes())


def hd95(pc1, pc2):
    dist_a_to_b, _, _ = knn_points(pc1, pc2, K=1)
    dist_a_to_b = torch.sqrt(dist_a_to_b) 
    
    dist_b_to_a, _, _ = knn_points(pc2, pc1, K=1)
    dist_b_to_a = torch.sqrt(dist_b_to_a)
    
    v95_a_to_b = torch.quantile(dist_a_to_b, 0.95, dim=1)
    v95_b_to_a = torch.quantile(dist_b_to_a, 0.95, dim=1)
    
    hd95 = torch.max(v95_a_to_b, v95_b_to_a)
    
    return hd95


def get_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return torch.from_numpy(positions).float().cuda()

class FormatedGaussians(NamedTuple):
    xyz: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor

def get_gaussians_for_time(pc, time):
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc._scaling
    rotations = pc._rotation
    colors_precomp = pc.features
    color_emb = pc.get_deformation_table
    time = time.to(means3D.device).repeat(means3D.shape[0],1)
    means3D_final, scales_final, rotations_final, opacity_final, (color_final, dcol) = pc._deformation(means3D, scales, 
                                                                            rotations, opacity, colors_precomp, time, color_emb)

    scales_final = pc.scaling_activation(scales_final)
    scales_final_flat = scales_final.clamp_max(0.05*pc.spatial_lr_scale)

    #covariances = pc.covariance_activation(scales_final, 1., rotations_final)

    return FormatedGaussians(
        xyz=means3D_final,
        scales=scales_final_flat, 
        rotations=rotations_final, 
        opacities=opacity_final # N, 1
    )




def test_model(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        test_plys = Scene(dataset, gaussians, load_iteration=iteration).gt_plys["test"]
        path_to_plys = f"{dataset.model_path}/test/ours_30000/plys"
        os.makedirs(path_to_plys, exist_ok=True)
        all_chamfer_distances = []
        all_hausdorff_distances = []
        all_hd95_distances = []
        max_time = len(os.listdir(os.path.split(test_plys[list(test_plys.keys())[0]])[0])) # get number of all plys so it can be used as time
        for time in tqdm(test_plys.keys()):
            gaussians_formated = get_gaussians_for_time(gaussians, time)
            gaussians_for_time = convert_3dgs_to_pc(gaussians_formated).unsqueeze(0)

            gt_ply = get_ply(test_plys[time]).unsqueeze(0)

            chamf_dist = chamfer_distance(gaussians_for_time, gt_ply)
            haus_dist = chamfer_distance(gaussians_for_time, gt_ply, point_reduction='max')
            haus95_dist = hd95(gaussians_for_time, gt_ply)
            print(f"Hausdorff distance for t{int(time*max_time)}:", haus_dist)
            print(f"Hausdorff 95 distance for t{int(time*max_time)}:", haus95_dist)
            print(f"Chamfer distance for t{int(time*max_time)}:", chamf_dist)

            save_xyz_to_ply(gaussians_for_time.squeeze(0), f"{path_to_plys}/{int(time*max_time)}.ply", quiet=True)
            
            all_chamfer_distances.append(chamf_dist[0])
            all_hd95_distances.append(haus95_dist)
            all_hausdorff_distances.append(haus_dist[0])
        chamfer_distance_final = torch.stack(all_chamfer_distances).mean()
        hausdorff_distance_final = torch.stack(all_hausdorff_distances).mean()
        hd95_distance_final = torch.stack(all_hd95_distances).mean()
        print("Chamfer distance is ", chamfer_distance_final.item())
        print("Hausdorff distance is ", hausdorff_distance_final.item())
        print("Hausdorff 95 distance is ", hd95_distance_final.item())
        import json
        with open(f"{dataset.model_path}/test/ours_30000/distance_metrics.json", "w") as f:
            json.dump({
                "chamfer_distance": chamfer_distance_final.item(),
                "hausdorff_distance": hausdorff_distance_final.item(),
                "hd95_distance": hd95_distance_final.item()
            }, f, indent=4)
        
def convert_3dgs_to_pc(gaussians_formated, num_samples_per_gaussian=10, opacity_threshold=0.0):
    """
    Convert 3D Gaussians to point cloud by sampling points from each Gaussian.
    
    Args:
        gaussians_formated: FormatedGaussians containing xyz, scales, rotations, opacities
        num_samples_per_gaussian: Base number of points to sample per Gaussian
        opacity_threshold: Minimum opacity to consider a Gaussian
    
    Returns:
        torch.Tensor: Sampled point cloud of shape (N, 3)
    """
    xyz = gaussians_formated.xyz
    scales = gaussians_formated.scales
    rotations = gaussians_formated.rotations
    opacities = gaussians_formated.opacities.squeeze()
    
    # Filter out low opacity Gaussians
    valid_mask = opacities > opacity_threshold
    xyz = xyz[valid_mask]
    scales = scales[valid_mask]
    rotations = rotations[valid_mask]
    opacities = opacities[valid_mask]
    
    if xyz.shape[0] == 0:
        return torch.empty((0, 3), device=xyz.device)
    
    normalized_opacities = (opacities - opacities.min()) / (opacities.max() - opacities.min() + 1e-8)
    samples_per_gaussian = (normalized_opacities * num_samples_per_gaussian).long()
    samples_per_gaussian = torch.clamp(samples_per_gaussian, min=1)
    
    
    # Build rotation matrices from quaternions
    from utils.general_utils import build_rotation
    rotation_matrices = build_rotation(rotations)  # Shape: (N, 3, 3)
    
    # Apply exponential activation to scales
    #scales_activated = torch.exp(scales)  # Shape: (N, 3)
    scales_activated = scales
    
    all_samples = []
    
    for i in range(xyz.shape[0]):
        n_samples = samples_per_gaussian[i].item()
        samples = torch.randn(n_samples, 3, device=xyz.device)
        samples = samples * scales_activated[i:i+1]
        samples = torch.matmul(samples, rotation_matrices[i].T)
        samples = samples + xyz[i:i+1]
        all_samples.append(samples)
    
    # Concatenate all samples
    point_cloud = torch.cat(all_samples, dim=0)
    
    return point_cloud

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--reconstruct_train", action="store_true")
    parser.add_argument("--reconstruct_test", action="store_true")
    parser.add_argument("--reconstruct_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    # if args.configs:
    #     import mmcv
    #     from utils.params_utils import merge_hparams
    #     config = mmcv.Config.fromfile(args.configs)
    #     args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    test_model(model.extract(args), hyperparam.extract(args), args.iteration, 
        pipeline.extract(args), 
        args.skip_train, args.skip_test, args.skip_video,
        args.reconstruct_train,args.reconstruct_test,args.reconstruct_video)
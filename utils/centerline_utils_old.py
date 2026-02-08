import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter1d

def simple_extrapolate_ends(centerline, trim=3, extend=10):
    """
    Remove noisy endpoints, then linearly extrapolate using local tangents.
    """
    cl = centerline.copy()

    # 1. trim bad endpoints
    cl = cl[trim:-trim]

    # spacing
    step = np.mean(np.linalg.norm(cl[1:] - cl[:-1], axis=1))

    # 2. start extrapolation
    t0 = cl[1] - cl[0]
    t0 = t0 / (np.linalg.norm(t0) + 1e-8)
    start_extra = [cl[0] - i * step * t0 for i in range(extend, 0, -1)]

    # 3. end extrapolation
    t1 = cl[-1] - cl[-2]
    t1 = t1 / (np.linalg.norm(t1) + 1e-8)
    end_extra = [cl[-1] + i * step * t1 for i in range(1, extend + 1)]

    return np.vstack([start_extra, cl, end_extra]).astype(np.float32)


def compute_centerline_from_points(points, num_slices=80, smooth_sigma=8.0):
    
    # 1. Determine the 'long' axis. 
    # For a colon segment, this is usually the axis with the highest variance.
    # We'll use PCA to find the principal direction.
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 2. Project points onto the main axis to find the "length"
    projections = centered_points @ main_axis
    min_proj, max_proj = np.min(projections), np.max(projections)
    
    # 3. Slice and Find Centroids
    line_points = []
    slice_edges = np.linspace(min_proj, max_proj, num_slices + 1)
    
    for i in range(num_slices):
        # Mask points within this Z-slice
        mask = (projections >= slice_edges[i]) & (projections < slice_edges[i+1])
        slice_pts = points[mask]
        
        if len(slice_pts) > 0:
            # The center of the cylinder slice is the mean of its points
            centroid = np.mean(slice_pts, axis=0)
            line_points.append(centroid)
    
    centerline = np.array(line_points).astype(np.float32)
    smooth_sigma = 8.0
    # smooth
    if smooth_sigma > 0 and len(centerline) > 5:
        centerline = gaussian_filter1d(centerline, sigma=smooth_sigma, axis=0)
    
    #fix weird endings
    if len(centerline) > 11:
        centerline = simple_extrapolate_ends(centerline, trim=15, extend=10)

    return centerline



def save_pcd_with_centerline(pcd, centerline, save_path):
    """
    pcd: BasicPointCloud OR open3d.geometry.PointCloud
    centerline: (N, 3) numpy array
    """

    # --- original points ---
    if hasattr(pcd, "points"):  # BasicPointCloud
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)
        if cols.max() > 1.0:
            cols = cols / 255.0
    else:  # open3d PointCloud
        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts)
    pcd_o3d.colors = o3d.utility.Vector3dVector(cols)

    # --- centerline points ---
    cl_pcd = o3d.geometry.PointCloud()
    cl_pcd.points = o3d.utility.Vector3dVector(centerline)

    # red centerline
    cl_colors = np.zeros((centerline.shape[0], 3))
    cl_colors[:, 0] = 1.0
    cl_pcd.colors = o3d.utility.Vector3dVector(cl_colors)

    # --- merge ---
    merged = pcd_o3d + cl_pcd

    # save
    o3d.io.write_point_cloud(save_path, merged)

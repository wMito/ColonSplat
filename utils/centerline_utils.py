import numpy as np
from scipy.ndimage import gaussian_filter1d
import open3d as o3d


def extract_camera_centers(cam_infos):
    centers = []
    for cam in cam_infos:
        centers.append(cam.camera_center)

    centers = np.asarray(centers, dtype=np.float32)
    if len(centers) < 2:
        raise ValueError("Not enough camera centers found.")
    return centers

def pca_axis(points):
    mean = points.mean(axis=0)
    X = points - mean
    cov = np.cov(X.T)
    evals, evecs = np.linalg.eigh(cov)   # eigh = stable for symmetric matrices
    axis = evecs[:, np.argmax(evals)]
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return axis, mean

def compute_centerline_sliced_along_axis(points, axis, origin=None, num_slices=80, smooth_sigma=8.0):
    points = np.asarray(points, dtype=np.float32)
    if origin is None:
        origin = points.mean(axis=0)

    proj = (points - origin) @ axis
    mn, mx = float(proj.min()), float(proj.max())

    edges = np.linspace(mn, mx, num_slices + 1)
    line = []
    for i in range(num_slices):
        m = (proj >= edges[i]) & (proj < edges[i+1])
        if np.any(m):
            line.append(points[m].mean(axis=0))

    centerline = np.asarray(line, dtype=np.float32)
    if smooth_sigma > 0 and len(centerline) > 5:
        centerline = gaussian_filter1d(centerline, sigma=smooth_sigma, axis=0)
    return centerline

def compute_centerline_from_points(points, cam_infos, num_slices=80, smooth_sigma=8.0):
    cam_centers = extract_camera_centers(cam_infos)
    cam_centers = gaussian_filter1d(cam_centers, sigma=2.0, axis=0)  # smooth jitter a bit

    axis, cam_mean = pca_axis(cam_centers)
    cl = compute_centerline_sliced_along_axis(points, axis, origin=cam_mean,
                                              num_slices=num_slices, smooth_sigma=smooth_sigma)
    return cl #, axis, cam_centers


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

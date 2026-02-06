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

def compute_centerline_from_points(
    pts,
    voxel_size=2.0,
    k=30,
    num_slices=80,
    smooth_sigma=8.0,
    min_points_per_slice=10,
):
    """
    pts: (N, 3) numpy array, world-space colon point cloud
    returns: (M, 3) centerline
    """

    # downsample + denoise
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)

    points = np.asarray(pcd.points)

    # local PCA directions
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, idxs = nbrs.kneighbors(points)

    dirs = np.zeros_like(points)
    for i in range(points.shape[0]):
        neigh = points[idxs[i]]
        cov = np.cov((neigh - neigh.mean(0)).T)
        w, v = np.linalg.eigh(cov)
        dirs[i] = v[:, np.argmax(w)]

    # global axis
    axis = dirs.mean(0)
    axis /= np.linalg.norm(axis)

    # project + slice
    t = points @ axis
    bins = np.linspace(t.min(), t.max(), num_slices)

    centerline = []
    for i in range(len(bins) - 1):
        m = (t >= bins[i]) & (t < bins[i + 1])
        if m.sum() >= min_points_per_slice:
            centerline.append(points[m].mean(0))

    centerline = np.array(centerline)

    # smooth
    if smooth_sigma > 0 and len(centerline) > 5:
        centerline = gaussian_filter1d(centerline, sigma=smooth_sigma, axis=0)
    
    #fix weird endings
    if len(centerline) > 11:
        centerline = simple_extrapolate_ends(centerline, trim=15, extend=10)


    return centerline.astype(np.float32)


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

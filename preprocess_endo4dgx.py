# ============================================================
# Endo-4DGX preprocessing using DUAL illumination estimation
# Reference: Zhang et al. 2019 (DUAL) — used as D(I)
# We use this script ONLY to test endo-4dgx baseline.
# ============================================================

import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from scipy.spatial import distance
from scipy.ndimage import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------
# Utility: sparse neighbors (no external dependency)
# ------------------------------------------------------------
def get_sparse_neighbor(p, n, m):
    neighbors = {}
    i, j = divmod(p, m)

    if j + 1 < m:
        neighbors[p + 1] = (i, j + 1, 1)
    if j - 1 >= 0:
        neighbors[p - 1] = (i, j - 1, 1)
    if i + 1 < n:
        neighbors[p + m] = (i + 1, j, 0)
    if i - 1 >= 0:
        neighbors[p - m] = (i - 1, j, 0)

    return neighbors

# ------------------------------------------------------------
# DUAL illumination estimation (paper reference [29])
# ------------------------------------------------------------
def create_spacial_affinity_kernel(spatial_sigma, size=15):
    kernel = np.zeros((size, size))
    c = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(
                -0.5 * (distance.euclidean((i, j), (c, c)) ** 2) /
                (spatial_sigma ** 2)
            )
    return kernel


def compute_smoothness_weights(L, x, kernel, eps=1e-3):
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode="constant")
    T = T / (np.abs(convolve(Lp, kernel, mode="constant")) + eps)
    return T / (np.abs(Lp) + eps)


def refine_illumination_map_linear(L, gamma, lambda_, kernel, eps=1e-3):
    wx = compute_smoothness_weights(L, 1, kernel, eps)
    wy = compute_smoothness_weights(L, 0, kernel, eps)

    n, m = L.shape
    L_1d = L.flatten()

    row, col, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, is_x) in get_sparse_neighbor(p, n, m).items():
            w = wx[k, l] if is_x else wy[k, l]
            row.append(p)
            col.append(q)
            data.append(-w)
            diag += w
        row.append(p)
        col.append(p)
        data.append(diag)

    F = csr_matrix((data, (row, col)), shape=(n * m, n * m))
    A = diags([np.ones(n * m)], [0]) + lambda_ * F
    L_refined = spsolve(A.tocsr(), L_1d).reshape(n, m)

    return np.clip(L_refined, eps, 1) ** gamma


def correct_underexposure(im, gamma, lambda_, kernel, eps=1e-3):
    L = np.max(im, axis=-1)
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)
    return im / np.repeat(L_refined[..., None], 3, axis=-1)


def fuse_multi_exposure_images(im, under, over, bc=1, bs=1, be=1):
    merge = cv2.createMergeMertens(bc, bs, be)
    imgs = [np.clip(x * 255, 0, 255).astype(np.uint8) for x in [im, under, over]]
    return merge.process(imgs)


def enhance_image_exposure(im, gamma, lambda_, dual=True, sigma=3):
    kernel = create_spacial_affinity_kernel(sigma)
    im_n = im.astype(np.float32) / 255.0

    under = correct_underexposure(im_n, gamma, lambda_, kernel)

    if dual:
        inv = 1 - im_n
        over = 1 - correct_underexposure(inv, gamma, lambda_, kernel)
        out = fuse_multi_exposure_images(im_n, under, over)
    else:
        out = under

    return np.clip(out * 255, 0, 255).astype(np.uint8)

# ------------------------------------------------------------
# D(I) wrapper (paper notation)
# ------------------------------------------------------------
def D_dual(I, gamma=0.6, lambda_=0.15, sigma=3):
    return enhance_image_exposure(
        im=I,
        gamma=gamma,
        lambda_=lambda_,
        dual=True,
        sigma=sigma
    )

# ------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------
def read_rgb(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_rgb(path, img):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ------------------------------------------------------------
# Dataset preprocessing
# ------------------------------------------------------------

def preprocess_sequence(color_dir):
    color_dir = Path(color_dir)

    img_dir_adj = color_dir.parent / f"{color_dir.name}_adjusted"

    img_dir_adj.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg"))
    )

    print(f"Processing {len(image_paths)} images from {color_dir}")
    stats = []

    for p in tqdm(image_paths):

        # generate DUAL illumination prior → images_mix_adjusted
        I = read_rgb(p)
        P = D_dual(I)
        save_rgb(img_dir_adj / p.name, P)

        stats.append((I.mean(), P.mean()))

    return stats


DATASETS=[
    "c1_ascending_t4_v4",
    "c1_cecum_t1_v4",
    "c1_descending_t4_v4",
    "c1_sigmoid1_t4_v4",
    "c1_sigmoid2_t4_v4",
    "c1_transverse1_t1_v4",
    "c1_transverse1_t4_v4",
    "c2_cecum_t1_v4",
    "c2_transverse1_t1_v4",
]


for folder in DATASETS:
    stats = preprocess_sequence(
            f"/data/coloncrafter/{folder}/color"
        )

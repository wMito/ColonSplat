import torch


# ------------------ Quaternion ------------------

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    quaternions: (N,4) in (w, x, y, z) or (r,i,j,k) format
    returns: (N,3,3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp_min(1e-12)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# ------------------ Edge Construction ------------------

def build_edge_matrices(xyz: torch.Tensor, idx: torch.Tensor):
    """
    xyz: (N,3)
    idx: (N,K) neighbor indices
    Returns P: (N,K,3) where P[i,k] = xi - xj
    """
    return xyz[:, None, :] - xyz[idx]


# ------------------ Weights ------------------

def compute_edge_weights(P: torch.Tensor, adaptive=True):
    """
    P: (N,K,3) edge vectors from init shape
    returns weight (N,K)
    """
    N, K, _ = P.shape
    device = P.device

    if adaptive:
        dist2 = (P ** 2).sum(-1)  # (N,K)
        mean_dist = dist2[dist2 > 0].mean().clamp_min(1e-12)
        weight = torch.exp(-dist2 / mean_dist)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        weight = torch.ones((N, K), device=device) / K

    return weight


# ------------------ Rotation Solve (Procrustes) ------------------

def solve_rotations(P: torch.Tensor, P_prime: torch.Tensor, weight: torch.Tensor):
    """
    Solves per-point optimal rotation Ri
    """
    with torch.no_grad():
        D = torch.diag_embed(weight)                  # (N,K,K)
        S = torch.bmm(P.transpose(1, 2), torch.bmm(D, P_prime))  # (N,3,3)

        U, sig, Vh = torch.linalg.svd(S)
        R = torch.bmm(Vh, U.transpose(1, 2))

        # Reflection fix
        det = torch.det(R)
        mask = det <= 0
        if mask.any():
            U2 = U.clone()
            U2[mask, :, -1] *= -1
            R[mask] = torch.bmm(Vh[mask], U2[mask].transpose(1, 2))

    return R


# ------------------ ARAP Geometry ------------------

def arap_geometry_loss(xyz_init, xyz_target, idx, adaptive_weight=True):
    P       = build_edge_matrices(xyz_init, idx)
    P_prime = build_edge_matrices(xyz_target, idx)

    weight = compute_edge_weights(P, adaptive_weight)

    R = solve_rotations(P, P_prime, weight)

    RP = torch.einsum("nij,nkj->nki", R, P)
    arap_error = (weight[..., None] * (P_prime - RP)).square().mean()

    return arap_error, R


# ------------------ Rotation Supervision ------------------

def arap_rotation_loss(R, rot_init, rot_target):
    init_R = quaternion_to_matrix(rot_init)
    tar_R  = quaternion_to_matrix(rot_target)
    R_pred = torch.bmm(R, init_R)
    return (R_pred - tar_R).square().mean()


# ------------------ Full ARAP Loss ------------------

def arap_loss(
    xyz_init,
    xyz_target,
    idx,
    rot_init=None,
    rot_target=None,
    with_rot=True,
    adaptive_weight=True,
    rot_weight=1e2
):
    geom, R = arap_geometry_loss(xyz_init, xyz_target, idx, adaptive_weight)

    if with_rot and (rot_init is not None) and (rot_target is not None):
        rot = arap_rotation_loss(R, rot_init, rot_target) * rot_weight
    else:
        rot = xyz_init.new_tensor(0.0)

    return geom, rot

import torch
import pytorch3d.ops


# ---------------- Quaternion ----------------

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp_min(1e-12)
    o = torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# ---------------- Connectivity (same logic as SC-GS) ----------------

def cal_connectivity_from_points(points, radius, K, least_edge_num=3, adaptive_weighting=True):
    """
    points: (N,3)
    returns: ii, jj, nn, weight
    """
    Nv = points.shape[0]
    device = points.device

    knn_res = pytorch3d.ops.knn_points(points[None], points[None], None, None, K=K+1)
    nn_dist = knn_res.dists[0, :, 1:]   # remove self
    nn_idx  = knn_res.idx[0, :, 1:]

    # radius cutoff
    nn_idx[:, least_edge_num:] = torch.where(
        nn_dist[:, least_edge_num:] < radius ** 2,
        nn_idx[:, least_edge_num:],
        -torch.ones_like(nn_idx[:, least_edge_num:])
    )

    nn_dist[:, least_edge_num:] = torch.where(
        nn_dist[:, least_edge_num:] < radius ** 2,
        nn_dist[:, least_edge_num:],
        torch.ones_like(nn_dist[:, least_edge_num:]) * torch.inf
    )

    if adaptive_weighting:
        nn_dist_1d = nn_dist.reshape(-1)
        mean_dist = nn_dist_1d[~torch.isinf(nn_dist_1d)].mean().clamp_min(1e-12)
        weight = torch.exp(-nn_dist / mean_dist)
    else:
        weight = torch.exp(-nn_dist)

    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    ii = torch.arange(Nv, device=device)[:, None].expand(Nv, K).reshape(-1)
    jj = nn_idx.reshape(-1)
    nn = torch.arange(K, device=device)[None].expand(Nv, K).reshape(-1)

    mask = jj != -1
    return ii[mask], jj[mask], nn[mask], weight


# ---------------- Edge Matrix ----------------

def produce_edge_matrix_nfmt(verts, edge_shape, ii, jj, nn):
    E = torch.zeros(edge_shape, device=verts.device)
    E[ii, nn] = verts[ii] - verts[jj]
    return E


# ---------------- Rotation Solve ----------------

def solve_rotations(P, P_prime, weight):
    with torch.no_grad():
        D = torch.diag_embed(weight)
        S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))
        U, sig, Vh = torch.linalg.svd(S)
        R = torch.bmm(Vh, U.permute(0, 2, 1))

        det = torch.det(R)
        mask = det <= 0
        if mask.any():
            U2 = U.clone()
            U2[mask, :, -1] *= -1
            R[mask] = torch.bmm(Vh[mask], U2[mask].permute(0, 2, 1))
    return R


# ---------------- ARAP Loss ----------------

def arap_loss(
    xyz_init,
    xyz_target,
    rot_init=None,
    rot_target=None,
    K=50,
    with_rot=True
):
    N = xyz_init.shape[0]
    radius = torch.linalg.norm(xyz_init.max(0).values - xyz_init.min(0).values) / 8

    with torch.no_grad():
        ii, jj, nn, weight = cal_connectivity_from_points(xyz_init, radius, K)

    P       = produce_edge_matrix_nfmt(xyz_init,  (N, K, 3), ii, jj, nn)
    P_prime = produce_edge_matrix_nfmt(xyz_target,(N, K, 3), ii, jj, nn)

    R = solve_rotations(P, P_prime, weight)

    arap_error = (weight[..., None] *
                  (P_prime - torch.einsum('bxy,bky->bkx', R, P))
                 ).square().mean()

    if with_rot and rot_init is not None and rot_target is not None:
        init_R = quaternion_to_matrix(rot_init)
        tar_R  = quaternion_to_matrix(rot_target)
        R_rot = torch.bmm(R, init_R)
        rot_error = (R_rot - tar_R).square().mean() * 1e2
    else:
        rot_error = xyz_init.new_tensor(0.)

    return arap_error, rot_error

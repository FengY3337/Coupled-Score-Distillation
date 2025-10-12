import math
import torch

import numpy as np
import torch.nn.functional as F

from typing import NamedTuple

# Compute the rotation matrix from quaternions
def build_rotation(r, device):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r, device):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = build_rotation(r, device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def strip_lowerdiag(L, device):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym, device):
    return strip_lowerdiag(sym, device)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    def helper(step):
        if lr_init == lr_final:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        n = 1.0
        log_lerp = np.exp(np.log(lr_init) * (1 - t ** n) + np.log(lr_final) * t ** n)
        return delay_rate * log_lerp

    return helper

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def certain_pose(elevation, phi, radius=1):
    x = radius * np.cos(elevation) * np.sin(phi)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(phi)

    target = np.zeros(3, dtype=np.float32)
    center = np.array([x, y, z])

    pose = np.eye(4, dtype=np.float32)
    forward_vector = safe_normalize(center - target)
    up_vector = np.array([0, 1, 0], dtype=np.float32)
    right_vector = safe_normalize(np.cross(up_vector, forward_vector))
    up_vector = safe_normalize(np.cross(forward_vector, right_vector))

    pose[:3, :3] = np.stack([right_vector, up_vector, forward_vector], axis=1)
    pose[:3, 3] = center
    return pose

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, device):
        self.image_width = width
        self.image_height = height
        self.FoVy = np.deg2rad(fovy)
        self.FoVx = np.deg2rad(fovx)
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).to(device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3], dtype=torch.float32).to(device)

def perspective(fovy, W, H, near, far):
    y = np.tan(fovy / 2)
    aspect = W / H
    return np.array(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, -1 / y, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0]], dtype=np.float32,)


def mipmap_linear_grid_put_2d(H, W, coords, values, min_resolution=32, return_count=False):

    C = values.shape[-1]

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]

    cur_H, cur_W = H, W

    while min(cur_H, cur_W) > min_resolution:

        mask = (count.squeeze(-1) == 0)
        if not mask.any():
            break

        cur_result, cur_count = linear_grid_put_2d(cur_H, cur_W, coords, values, return_count=True)
        result[mask] = result[mask] + \
                       F.interpolate(cur_result.permute(2, 0, 1).unsqueeze(0).contiguous(), (H, W), mode='bilinear',
                                     align_corners=False).squeeze(0).permute(1, 2, 0).contiguous()[mask]
        count[mask] = count[mask] + F.interpolate(cur_count.view(1, 1, cur_H, cur_W), (H, W), mode='bilinear',
                                                  align_corners=False).view(H, W, 1)[mask]
        cur_H //= 2
        cur_W //= 2

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def linear_grid_put_2d(H, W, coords, values, return_count=False):

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices_00 = indices.floor().long()  # [N, 2]
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor(
        [0, 1], dtype=torch.long, device=indices.device
    )
    indices_10 = indices_00 + torch.tensor(
        [1, 0], dtype=torch.long, device=indices.device
    )
    indices_11 = indices_00 + torch.tensor(
        [1, 1], dtype=torch.long, device=indices.device
    )

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(result, count, indices_00, values * w_00.unsqueeze(1),
                                              weights * w_00.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_01, values * w_01.unsqueeze(1),
                                              weights * w_01.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_10, values * w_10.unsqueeze(1),
                                              weights * w_10.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_11, values * w_11.unsqueeze(1),
                                              weights * w_11.unsqueeze(1))

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result
def scatter_add_nd_with_count(input, count, indices, values, weights=None):

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    count = count.view(-1, 1)

    flatten_indices = (indices * torch.tensor(stride, dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    if weights is None:
        weights = torch.ones_like(values[..., :1])

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)

    return input.view(*size, C), count.view(*size, 1)

def stride_from_shape(shape):
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x)
    return list(reversed(stride))


def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

def normal_consistency(face_normals, t_pos_idx):

    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))

def laplacian_uniform(verts, faces):
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()


def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss

def certain_mvp(elevation, phi, height, width, fov, near, far, radius=1):

    x = radius * np.cos(elevation) * np.sin(phi)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(phi)

    target = np.tile(np.zeros(3), (1, 1))
    center = np.transpose(np.array([x, y, z])) + target

    poses_mvp = np.tile(np.eye(4), (1, 1, 1))
    forward_vector = safe_normalize(center - target)
    up_vector_mvp = np.array([0, 1, 0], dtype=np.float32)

    right_vector_mvp = safe_normalize(np.cross(up_vector_mvp, forward_vector))
    # right_vector_mvp = safe_normalize(np.cross(forward_vector, up_vector_mvp))
    up_vector_mvp = safe_normalize(np.cross(forward_vector, right_vector_mvp))
    # up_vector_mvp = safe_normalize(np.cross(right_vector_mvp, forward_vector))

    poses_mvp[:, :3, :3] = np.stack([right_vector_mvp, up_vector_mvp, forward_vector], axis=2)
    poses_mvp[:, :3, 3] = center
    poses_mvp = torch.tensor(poses_mvp, dtype=torch.float32)

    focal = height / (2 * np.tan(np.deg2rad(fov) / 2))

    projection = torch.tensor([
        [2 * focal / width, 0, 0, 0],
        [0, -2 * focal / height, 0, 0],
        [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
        [0, 0, -1, 0]
    ], dtype=torch.float32).unsqueeze(0)

    mvp = projection @ torch.inverse(poses_mvp)

    return mvp


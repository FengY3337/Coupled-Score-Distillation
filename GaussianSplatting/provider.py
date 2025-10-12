import random
import trimesh
import torch
import numpy as np

from torch.utils.data import DataLoader

import GaussianSplatting.Gaussian_utils as G_utils


DIR_COLORS = np.array([
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
    [255, 0, 255, 255],
    [0, 255, 255, 255],
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(elevations, phis, overhead, front, is_train):
    # overhead:-60；front-60

    # front = 0         [front/2, 180-front/2)
    # side (left) = 1   [180-front/2, 180+front/2)
    # back = 2          [180+front/2, -front/2)
    # side (right) = 3  [-front/2, front/2)
    # top = 4           [-90, overhead]
    # bottom = 5        [-overhead, 90]
    if is_train:
        res = np.zeros(elevations.shape[0])
    else:
        res = np.zeros(1)
    res[(phis < front / 2) | (phis >= 2 * torch.pi - front / 2)] = 3
    res[(phis >= front / 2) & (phis < torch.pi - front / 2)] = 0
    res[(phis >= torch.pi - front / 2) & (phis < torch.pi + front / 2)] = 1
    res[(phis >= torch.pi + front / 2) & (phis < 2 * torch.pi - front / 2)] = 2

    res[elevations <= overhead] = 4
    res[elevations >= (-overhead)] = 5

    return res

def rand_poses(size, radius_range=[2.0, 2.5], elevation_range=[0, 120], phi_range=[0, 360], return_dirs=False,
               angle_overhead=-60, angle_front=60):

    elevation_range = np.deg2rad(elevation_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = np.random.uniform(0, 1, size=size) * (radius_range[1] - radius_range[0]) + radius_range[0]

    elevations = np.random.uniform(0, 1, size=size) * (elevation_range[1] - elevation_range[0]) + elevation_range[0]
    phis = np.random.uniform(0, 1, size=size) * (phi_range[1] - phi_range[0]) + phi_range[0]

    # (size,)
    x = radius * np.cos(elevations) * np.sin(phis)
    y = - radius * np.sin(elevations)
    z = radius * np.cos(elevations) * np.cos(phis)

    # (size, 3)
    targets = np.tile(np.zeros(3), (size, 1))
    # (size, 3)
    centers = np.transpose(np.array([x, y, z])) + targets

    # (size, 4, 4)
    poses = np.tile(np.eye(4), (size, 1, 1))
    poses_mvp = np.tile(np.eye(4), (size, 1, 1))

    forward_vector = G_utils.safe_normalize(centers - targets)

    up_vector = np.array([0, 1, 0], dtype=np.float32)
    up_vector_mvp = np.array([0, 1, 0], dtype=np.float32)

    right_vector = G_utils.safe_normalize(np.cross(up_vector, forward_vector))
    right_vector_mvp = G_utils.safe_normalize(np.cross(up_vector_mvp, forward_vector))
    # right_vector_mvp = G_utils.safe_normalize(np.cross(forward_vector, up_vector_mvp))

    up_vector = G_utils.safe_normalize(np.cross(forward_vector, right_vector))
    up_vector_mvp = G_utils.safe_normalize(np.cross(forward_vector, right_vector_mvp))
    # up_vector_mvp = G_utils.safe_normalize(np.cross(right_vector_mvp, forward_vector))

    poses[:, :3, :3] = np.stack([right_vector, up_vector, forward_vector], axis=2)
    poses[:, :3, 3] = centers

    poses_mvp[:, :3, :3] = np.stack([right_vector_mvp, up_vector_mvp, forward_vector], axis=2)
    poses_mvp[:, :3, 3] = centers
    poses_mvp = torch.tensor(poses_mvp, dtype=torch.float32)

    if return_dirs:
        dirs = get_view_direction(elevations, phis, angle_overhead, angle_front, True)
    else:
        dirs = None

    return poses, dirs, elevations, phis, radius, poses_mvp

# 用于生成valid/test的相机pose和text编码信息
def circle_poses(radius=2.5, elevation=-30, phi=0, return_dirs=False, angle_overhead=-60, angle_front=60):
    elevation = np.deg2rad(elevation)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = np.array(radius, dtype=np.float32)

    # 计算相机中心坐标，[3, ]
    center = np.array([radius * np.cos(elevation) * np.sin(phi),
                        -radius * np.sin(elevation),
                        radius * np.cos(elevation) * np.cos(phi)])
    target = np.zeros([3], dtype=np.float32)

    forward_vector = G_utils.safe_normalize(center - target)
    up_vector = np.array([0, 1, 0], dtype=np.float32)
    up_vector_mvp = np.array([0, 1, 0], dtype=np.float32)

    right_vector = G_utils.safe_normalize(np.cross(up_vector, forward_vector))
    right_vector_mvp = G_utils.safe_normalize(np.cross(up_vector_mvp, forward_vector))
    # right_vector_mvp = G_utils.safe_normalize(np.cross(forward_vector, up_vector_mvp))

    up_vector = G_utils.safe_normalize(np.cross(forward_vector, right_vector))
    up_vector_mvp = G_utils.safe_normalize(np.cross(forward_vector, right_vector_mvp))
    # up_vector_mvp = G_utils.safe_normalize(np.cross(right_vector_mvp, forward_vector))

    poses = np.eye(4, dtype=np.float32)
    poses_mvp = np.eye(4, dtype=np.float32)

    poses[:3, :3] = np.stack([right_vector, up_vector, forward_vector], axis=1)
    poses[:3, 3] = center

    poses_mvp[:3, :3] = np.stack([right_vector_mvp, up_vector_mvp, forward_vector], axis=1)
    poses_mvp[:3, 3] = center
    poses_mvp = torch.tensor(poses_mvp, dtype=torch.float32)

    if return_dirs:
        dirs = get_view_direction(elevation, phi, angle_overhead, angle_front, False)
    else:
        dirs = None

    return poses, dirs, poses_mvp
    

class GSDataset:
    def __init__(self, opt, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.type = type

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']
        self.near = opt.near
        self.far = opt.far

        # [debug]
        # poses, dirs = rand_poses(100, radius_range=self.opt.radius_range, return_dirs=self.opt.dir_text,
        #                          angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def collate(self, index):
        B = len(index)
        if self.training:
            poses, dirs, elevations, phis, radius, poses_mvp = rand_poses(B, radius_range=self.opt.radius_range,
                                                                          elevation_range=self.opt.elevation_range,
                                                                          return_dirs=self.opt.dir_text,
                                                                          angle_overhead=self.opt.angle_overhead,
                                                                          angle_front=self.opt.angle_front)

            fovs = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]

            if self.opt.GS_dmtet:
                focal = self.H / (2 * np.tan(np.deg2rad(fovs) / 2))
                projection = torch.tensor([
                    [2 * focal / self.W, 0, 0, 0],
                    [0, -2 * focal / self.H, 0, 0],
                    [0, 0, -(self.far + self.near) / (self.far - self.near),
                     -(2 * self.far * self.near) / (self.far - self.near)],
                    [0, 0, -1, 0]
                ], dtype=torch.float32).unsqueeze(0)

                mvps = projection @ torch.inverse(poses_mvp)
            else:
                mvps = None

        else:
            phi = (index[0] / self.size) * 360
            poses, dirs, poses_mvp = circle_poses(radius=self.opt.val_radius, elevation=self.opt.val_elevation, phi=phi,
                                                  return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead,
                                                  angle_front=self.opt.angle_front)
            fovs = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2
            elevations = None
            phis = None
            radius = None

            if self.opt.GS_dmtet:
                focal = self.H / (2 * np.tan(np.deg2rad(fovs) / 2))

                projection = torch.tensor([
                    [2 * focal / self.W, 0, 0, 0],
                    [0, -2 * focal / self.H, 0, 0],
                    [0, 0, -(self.far + self.near) / (self.far - self.near), -(2 * self.far * self.near) / (self.far - self.near)],
                    [0, 0, -1, 0]
                ], dtype=torch.float32)

                mvps = projection @ torch.inverse(poses_mvp)
            else:
                mvps = None

        data = {
            'H': self.H,
            'W': self.W,
            'fovy': fovs,
            'fovx': fovs,
            'near': self.near,
            'far': self.far,
            'elevations': elevations,
            'phis': phis,
            'radius': radius,
            'dir': dirs,
            'poses': poses,
            'poses_mvp': poses_mvp,
            'mvps': mvps,
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        return loader

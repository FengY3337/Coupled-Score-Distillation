import math
import torch

import numpy as np
import torch.nn as nn
import nvdiffrast.torch as dr
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt
from diff_gaussian_rasterization import (GaussianRasterizationSettings, GaussianRasterizer,)

import GaussianSplatting.Gaussian_utils as G_utils

from GaussianSplatting.gridencoder.grid import GridEncoder
from GaussianSplatting.Gaussian_model import GaussianModel
from GaussianSplatting.sh_utils import eval_sh, SH2RGB, RGB2SH

class DMTet():
    def __init__(self, device):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [ 1,  0,  2, -1, -1, -1],
            [ 4,  0,  3, -1, -1, -1],
            [ 1,  4,  2,  1,  3,  4],
            [ 3,  1,  5, -1, -1, -1],
            [ 2,  3,  0,  2,  5,  3],
            [ 1,  4,  0,  1,  5,  4],
            [ 4,  2,  5, -1, -1, -1],
            [ 4,  5,  2, -1, -1, -1],
            [ 4,  1,  0,  4,  5,  1],
            [ 3,  2,  0,  3,  5,  2],
            [ 1,  3,  5, -1, -1, -1],
            [ 4,  1,  2,  4,  3,  1],
            [ 3,  0,  4, -1, -1, -1],
            [ 2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=device)
        self.num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device=device)
        self.base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=device)

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)

        return torch.stack([a, b], -1)

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=self.device)
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        return verts, faces


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class GS_DMTET(nn.Module):
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device

        ##### MLP predict albedo
        self.encoder = GridEncoder(input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19,
                                   desired_resolution=self.opt.hash_resolution, align_corners=False,
                                   interpolation='smoothstep')
        self.in_dim = self.encoder.output_dim
        self.albedo_net = MLP(self.in_dim, 3, self.opt.hidden_dim, self.opt.num_layers, bias=True)

        ##### dmtet
        tets = np.load('tets/{}_tets.npz'.format(self.opt.tet_grid_size))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device=self.device) * 2
        if self.opt.tet_grid_size == 256:
            self.verts = self.verts - 1.0
        self.verts = - self.verts

        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device=self.device)
        self.dmtet = DMTet(self.device)

        # vert sdf and deform
        self.sdf = torch.nn.Parameter(torch.zeros_like(self.verts[..., 0]), requires_grad=True)
        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)

        edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=self.device)
        all_edges = self.indices[:, edges].reshape(-1, 2)  # [M * 6, 2]
        all_edges_sorted = torch.sort(all_edges, dim=1)[0]
        self.all_edges = torch.unique(all_edges_sorted, dim=0)
        self.glctx = dr.RasterizeCudaContext()

    def verts_faces(self):
        deform = torch.tanh(self.deform) / self.opt.tet_grid_size
        verts, faces = self.dmtet(self.verts + deform, self.sdf, self.indices)
        return verts, faces

    def get_params(self, dmtet_lr, albedo_lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': albedo_lr * 10},
            {'params': self.albedo_net.parameters(), 'lr': albedo_lr},
        ]
        if not self.opt.dmtet_finetune:
            params.append({'params': self.sdf, 'lr': dmtet_lr})
            params.append({'params': self.deform, 'lr': dmtet_lr})
        return params

def get_sdf_from_voxel(vertices, sdf_voxel, resolution, range_min, range_max):
    sdf_values = []

    for v in vertices:
        mapped_idx = np.floor(((v - range_min) / (range_max - range_min)) * (resolution - 1)).astype(int)
        mapped_idx = np.clip(mapped_idx, 0, (resolution - 1))

        sdf_value = sdf_voxel[tuple(mapped_idx)]
        sdf_values.append(sdf_value)

    return np.array(sdf_values)


class Renderer:
    def __init__(self, opt, device):
        super().__init__()

        self.opt = opt
        self.device = device
        self.use_dmtet = opt.GS_dmtet
        # define 3D-GS
        self.gaussians = GaussianModel(opt.sh_degree, device)
        # default background color
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)

        # define dmtet
        if self.opt.GS_dmtet:
            self.GS_DMTET = GS_DMTET(self.opt, self.device)
        else:
            self.GS_DMTET = None

    def pred_albedo(self, x):
        enc = self.GS_DMTET.encoder(x)
        h = self.GS_DMTET.albedo_net(enc)
        albedo = torch.sigmoid(h)
        return albedo

    # initialize dmtet with 3D-GS
    def init_tet(self, resolution=128):

        density_thresh = self.opt.density_thresh

        # convert 3D-GS into voxel
        occ = self.gaussians.extract_fields(resolution).detach().cpu().numpy()
        binary_occ = (occ >= density_thresh).astype(np.uint8)

        # calculate SDF
        internal_distances = distance_transform_edt(binary_occ)
        external_distances = distance_transform_edt(1 - binary_occ)
        sdf = external_distances - internal_distances

        vertices = self.GS_DMTET.verts.detach().cpu().numpy()
        range_min = (vertices.min(axis=0)).min()
        range_max = (vertices.max(axis=0)).max()

        vertices_sdf = get_sdf_from_voxel(vertices, sdf, resolution, range_min, range_max)
        self.GS_DMTET.sdf.data = torch.tensor(vertices_sdf, dtype=torch.float32).to(self.device)

    # initialize 3D-GS
    def initialize(self, input=None, num_pts=1000, radius=0.5, scale=None):
        if input is None:
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = G_utils.BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            self.gaussians.create_from_pcd(pcd, 10)
        elif isinstance(input, G_utils.BasicPointCloud):
            self.gaussians.create_from_pcd(input, 1)
        else:
            self.gaussians.load_ply(input, scale)

    # render dmtet
    def dmtet_render(self, mvp, h, w, shading, bg_color=None):

        results = {}
        verts, faces = self.GS_DMTET.verts_faces()

        # get normals
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

        faces = faces.int()
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = G_utils.safe_normalize(face_normals)
        vn = torch.zeros_like(verts)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

        # rasterization
        verts_clip = torch.matmul(F.pad(verts, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0)

        rast, rast_db = dr.rasterize(self.GS_DMTET.glctx, verts_clip, faces, (h, w))

        alpha, _ = dr.interpolate(torch.ones_like(verts[:, :1]).unsqueeze(0), rast, faces)
        xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces)   # [1, H, W, 3]
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = G_utils.safe_normalize(normal)

        xyzs = xyzs.view(-1, 3)
        mask = (alpha > 0).view(-1).detach()

        albedo = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            masked_albedo = self.pred_albedo(xyzs[mask])
            albedo[mask] = masked_albedo.float()
        albedo = albedo.view(1, h, w, 3)

        if shading == 'normal':
            color = (normal + 1) / 2
        else:
            color = albedo

        color = dr.antialias(color, rast, verts_clip, faces).squeeze(0).clamp(0, 1)
        alpha = dr.antialias(alpha, rast, verts_clip, faces).squeeze(0).clamp(0, 1)

        depth = rast[0, :, :, [2]]
        color = color + (1 - alpha) * bg_color

        results['depth'] = depth.permute(2, 0, 1)
        results['image'] = color.permute(2, 0, 1)

        # regularizations
        if self.GS_DMTET.training:
            if self.opt.lambda_normal > 0:
                results['normal_loss'] = G_utils.normal_consistency(face_normals, faces)
            if self.opt.lambda_lap > 0:
                results['lap_loss'] = G_utils.laplacian_smooth_loss(verts, faces)

        return results

    # render 3D-GS
    def render(self, viewpoint_camera, scaling_modifier=1.0, bg_color=None, override_color=None,
               compute_cov3D_python=False, convert_SHs_python=False,):

        screenspace_points = torch.zeros_like(self.gaussians.get_xyz, dtype=self.gaussians.get_xyz.dtype,
                                              requires_grad=True, device=self.device,) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(-1, 3, (self.gaussians.max_sh_degree + 1) ** 2)
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(self.gaussians.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

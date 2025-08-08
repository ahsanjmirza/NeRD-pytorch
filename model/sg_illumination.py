import torch
import torch.nn as nn
from utils import math_utils
from model.renderer import Renderer
import numpy as np

class SG_Illumination(nn.Module):
    def __init__(self, 
        n_sg_lobes = 24,
        compress_sharpness = False,
        compress_amplitude = False
    ):
        super().__init__()
        self.n_lobes = n_sg_lobes
        self.compress_sharpness = compress_sharpness
        self.compress_amplitude = compress_amplitude
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean = torch.from_numpy(
            np.load(
                './utils/mean_sgs.npy')
            )[None, ...].to(self.device)

        if self.compress_amplitude or self.compress_sharpness:
            mean = torch.concat(
                [
                    math_utils.safe_log(mean[..., :3])
                    if self.compress_amplitude 
                    else mean[..., :3],
                    mean[..., 3:6],
                    math_utils.safe_log(mean[..., 6:][..., None])
                    if self.compress_sharpness 
                    else mean[..., 6:][..., None],
                ], 
                dim = -1
            )
        self.sgs = nn.Parameter(mean, requires_grad = True)
        self.renderer = Renderer()
        return
                    
    def forward(self, 
        requires_grad = False,
        rotating_object_pose = None
    ):
        if rotating_object_pose is not None:
            self.sgs = self.__rotate_object_pose(rotating_object_pose)

        if requires_grad: 
            self.sgs.requires_grad_(True)
            return self.sgs
        else: 
            return self.sgs.detach()

    def __rotate_object_pose(self, rotating_object_pose):
        rotation_matrix = torch.linalg.inv(rotating_object_pose[:3, :3])
        environment_ampl = self.sgs[..., :3]
        environment_axis = self.sgs[..., 3:6]
        environment_sharpness = self.sgs[..., 6:]
        environment_axis = environment_axis[..., None, :] * rotation_matrix
        environment_axis = torch.sum(environment_axis, dim = -1)
        return torch.concat([environment_ampl, environment_axis, environment_sharpness], dim = -1)
    
    def apply_wb(self, 
        wb_value,
        rays_o,
        ev100, 
        clip_range = (0.99, 1.01),
        grayscale = False
    ):
        exp_val = math_utils.ev100_to_exp(ev100)[0, :]
        normal_dir = math_utils.normalize(rays_o[0, :][None, ...])
        wb_scene = (
            self.renderer.forward(
                sg_illumination = self.forward(requires_grad = True),
                basecolor = torch.ones([normal_dir.shape[0], 3]).to(self.device) * 0.8,
                metallic = torch.zeros([normal_dir.shape[0], 1]).to(self.device),
                roughness = torch.ones([normal_dir.shape[0], 1]).to(self.device),
                normal = normal_dir,
                alpha = torch.ones([1]).to(self.device),
                view_dirs = normal_dir
            )
            * exp_val
        )

        factor = wb_value / torch.nan_to_num(wb_scene, nan = 1e-1)

        if grayscale: factor = torch.mean(factor, dim = -1, keepdim = True)
        if clip_range is not None: factor = torch.clip(factor, clip_range[0], clip_range[1])
        with torch.no_grad():
            if self.compress_amplitude:
                self.sgs[..., :3] = math_utils.safe_exp(self.sgs[0, :, :3]) * factor
            else:
                self.sgs[..., :3] = self.sgs[0, :, :3] * factor
        return
import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.positional_encoder import PositionalEncoder
from model.volumetric_renderer import volumetric_render

class Coarse(nn.Module):
    def __init__(self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        n_sg_lobes: int = 24,
        n_sg_condense: int = 16,
        coarse_samples: int = 64
    ):
        super().__init__()
        self.d_input = d_input
        self.n_samples = coarse_samples

        self.positional_encoder = PositionalEncoder(d_input, 10, False)
        d_input = self.positional_encoder.d_output
        self.theta1 = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter), nn.ReLU()] +
            [nn.Linear(d_filter, d_filter), nn.ReLU() for i in range(n_layers - 1)]
        )
        self.density_net = nn.Linear(d_filter, 1)
        self.theta2 = nn.Linear(n_sg_lobes * 7, n_sg_condense)
        self.theta3 = nn.ModuleList(
            [nn.Linear(d_filter + n_sg_condense, d_filter), 
            nn.ReLU(),
            nn.Linear(d_filter, 3)]
        )
        return
                    
    def forward(self, 
        rays_o, 
        rays_d, 
        sg_lobes,
        near,
        far
    ):
        pts, z_vals = self.__stratified_sampling(
            rays_o, 
            rays_d, 
            near, 
            far, 
            self.n_samples
        )
        pts = self.positional_encoder(pts)
        backbone = self.theta1(pts)
        density = self.density_net(backbone)
        em_sglobes = self.theta2(sg_lobes)
        rgb = self.theta3(torch.concat([backbone, em_sglobes], dim = -1))
        raw = nn.concat([rgb, density], dim = -1)
        rgb_map, depth_map, acc_map, weights = volumetric_render(raw, z_vals, rays_d)
        return rgb_map, depth_map, acc_map, z_vals, weights 
    
    def __stratified_sampling(self, 
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True
    ):
        t_vals = torch.linspace(0., 1., n_samples, device = rays_d.device)
        z_vals = near * (1.-t_vals) + far * (t_vals)

        if perturb:
            mids = .5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[-1:]], dim = -1)
            lower = torch.concat([z_vals[:1], mids], dim = -1)
            t_rand = torch.rand([n_samples], device = z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts, z_vals
    
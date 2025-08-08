import torch
import torch.nn as nn
from model.positional_encoder import PositionalEncoder
from utils import math_utils

class Coarse(nn.Module):
    def __init__(self,
        d_input = 3,
        n_layers = 8,
        d_filter = 256,
        n_sg_lobes = 24,
        n_sg_condense = 16,
        coarse_samples = 64
    ):
        super().__init__()
        self.d_input = d_input
        self.n_samples = coarse_samples

        self.positional_encoder = PositionalEncoder(d_input, 10, False)
        self.d_input = self.positional_encoder.d_output

        theta1 = [nn.Linear(self.d_input, d_filter), nn.ReLU()]
        
        for _ in range(n_layers - 1):
            theta1.append(nn.Linear(d_filter, d_filter))
            theta1.append(nn.ReLU())
        
        self.theta1 = nn.Sequential(*theta1)

        self.density_net = nn.Sequential(
            nn.Linear(d_filter + self.d_input, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter, 1)
        )
        
        self.theta2 = nn.Linear(n_sg_lobes * 7, n_sg_condense)
        
        self.rgb_net = nn.Sequential(
            nn.Linear(d_filter + n_sg_condense, d_filter), 
            nn.ReLU(),
            nn.Linear(d_filter, 3),
            nn.Sigmoid()
        )
        return
                    
    def forward(self, 
        rays_o, 
        rays_d, 
        sg_lobes,
        near,
        far,
        jitter = 0.0,
        perturb = False
    ):
        pts, z_vals = self.__stratified_sampling(
            rays_o, 
            rays_d, 
            near, 
            far, 
            self.n_samples,
            perturb
        )
        
        pts_enc = self.positional_encoder(pts)
        emb = self.theta1(pts_enc)
        density = self.density_net(torch.concat([emb, pts_enc], dim = -1))
        em_sglobes = self.theta2(torch.flatten(sg_lobes, 1, -1))
        rgb = self.rgb_net(
            torch.concat(
                [
                    emb, 
                    em_sglobes.expand(
                        emb.shape[0], 
                        emb.shape[1], 
                        -1
                    )
                ], 
                dim = -1)
            )
        
        raw = {
            'rgb': rgb,
            'density': density
        }

        rgb_map, acc_alpha, weights, alpha = self.__volumetric_render(raw, z_vals, rays_d, jitter)
        coarse = {
            'volumetric_rgb': rgb_map,
            'acc_alpha': acc_alpha,
            'individual_alphas': alpha
        }
        return coarse, z_vals, weights
    
    def __stratified_sampling(self, 
        rays_o,
        rays_d,
        near,
        far,
        n_samples,
        perturb = True
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
    
    def __volumetric_render(
        self,
        raw,
        z_vals,
        rays_d,
        raw_noise_std = 0.0
    ):
        
        eps = 1e-10
        density = raw['density'][..., 0]
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, eps * torch.ones_like(dists[..., :1])], dim = -1)  
        dists = dists * torch.norm(rays_d[..., None, :], dim = -1)  
        
        noise = 0.
        if raw_noise_std > 0.:
            noise = (((torch.rand(density.shape) * 2) - 1) * raw_noise_std).to(density.device)    
        
        alpha = 1.0 - torch.exp(-nn.functional.relu(density + noise) * dists)   

        def cumprod_exclusive(tensor):
            cumprod = torch.cumprod(tensor, -1)
            cumprod = torch.roll(cumprod, 1, -1)
            cumprod[..., 0] = 1.
            return cumprod
        
        weights = alpha * cumprod_exclusive(1. - alpha + eps) 
        
        acc_alpha = torch.sum(weights, dim=-1)    
        
        rgb_map = torch.sum(weights[..., None] * raw['rgb'], dim = -2)  
        rgb_map = math_utils.white_background_compose(rgb_map, acc_alpha[..., None])
        
        return rgb_map, acc_alpha, weights, alpha
import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.positional_encoder import PositionalEncoder
from model.volumetric_renderer import volumetric_render
from model.sg_renderer import Renderer
from model.brdf import BDRFAutoencoder
from utils import math_utils

class Fine(nn.Module):
    def __init__(self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        d_brdf_latent: int = 2,
        coarse_samples: int = 64,
        fine_samples: int = 64
    ):
        super().__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.d_filter = d_filter
        self.d_brdf_latent = d_brdf_latent
        self.coarse_samples = coarse_samples
        self.fine_samples = fine_samples

        self.positional_encoder = PositionalEncoder(d_input, 10, False)
        d_input = self.positional_encoder.d_output

        phi1 = nn.ModuleList(
            [nn.Linear(d_input, d_filter), nn.ReLU()] +
            [nn.Linear(d_filter, d_filter), nn.ReLU() for i in range((n_layers // 2) - 1)]
        )

        self.phi1_1 = nn.Sequential(
            phi1
        )

        phi1 = nn.ModuleList(
            [nn.Linear(d_input + d_filter, d_filter), nn.ReLU()] +
            [nn.Linear(d_filter, d_filter), nn.ReLU() for i in range((n_layers // 2) - 1)]
        )

        self.phi1_2 = nn.Sequential(
            phi1
        )

        self.phi1_3 = nn.Linear(d_filter, 4)

        self.phi2 = BDRFAutoencoder(d_filter, d_brdf_latent)

        self.renderer = Renderer(eval_background=True)

        return
                    
    def forward(self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        coarse_z_vals: torch.Tensor,
        coarse_weights: torch.Tensor
    ):
        pts_combined, z_vals_combined = self.__hierachical_sampling(
            rays_o, 
            rays_d, 
            coarse_z_vals, 
            coarse_weights, 
            self.fine_samples
        )
        
        pts_enc = self.positional_encoder(pts_combined)
        emb = self.phi1_1(pts_enc)
        emb = self.phi1_2(torch.concat([pts_enc, emb], dim = -1))

        raw = self.phi1_3(emb)
        normal = -math_utils.normalize(
            torch.gradient(
                raw[..., 3][..., None], spacing = pts_combined, dim = -1
            )
        )

        brdf, brdf_latent = self.phi2(emb)
        rgb_map, depth_map, acc_map, weights = volumetric_render(raw, z_vals_combined, rays_d)

        

        return rgb_map, depth_map, acc_map, z_vals_combined, weights, normal, brdf, brdf_latent
    
    def __sample_pdf(self,
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
    ):
        # Normalize weights to get PDF.
        pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

        # Convert PDF to CDF.
        cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
        cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

        # Take sample positions to grab from CDF. Linear when perturb == 0.
        if not perturb:
            u = torch.linspace(0., 1., n_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

        # Find indices along CDF where values in u would be placed.
        u = u.contiguous() # Returns contiguous tensor with same values.
        inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

        # Clamp indices that are out of bounds.
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

        # Sample from cdf and the corresponding bin centers.
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                            index=inds_g)

        # Convert samples to ray length.
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples # [n_rays, n_samples]
    
    def __hierachical_sampling(self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
    ):
        # Draw samples from PDF using z_vals as bins and weights as probabilities.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_z_samples = self.__sample_pdf(
            z_vals_mid, 
            weights[..., 1:-1], 
            n_samples,
            perturb = perturb
        )
        new_z_samples = new_z_samples.detach()

        # Resample points from ray based on PDF.
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
        return pts, z_vals_combined
    
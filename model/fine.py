import torch
import torch.nn as nn
from utils import math_utils
from model.renderer import Renderer
from model.positional_encoder import PositionalEncoder
from model.brdf_autoencoder import BDRFAutoencoder

class Fine(nn.Module):
    def __init__(self,
        d_input = 3,
        n_layers = 8,
        d_filter = 256,
        d_brdf_latent = 2,
        fine_samples = 128
    ):
        super().__init__()
        self.d_input = d_input
        self.fine_samples = fine_samples

        self.positional_encoder = PositionalEncoder(d_input, 10, False)
        d_input = self.positional_encoder.d_output

        phi1 = [nn.Linear(d_input, d_filter), nn.ReLU()]

        for _ in range((n_layers // 2) - 1):
            phi1.append(nn.Linear(d_filter, d_filter))
            phi1.append(nn.ReLU())

        self.phi1_1 = nn.Sequential(*phi1)

        phi1 = [nn.Linear(d_input + d_filter, d_filter), nn.ReLU()]

        for _ in range((n_layers // 2) - 1):
            phi1.append(nn.Linear(d_filter, d_filter))
            phi1.append(nn.ReLU())

        self.phi1_2 = nn.Sequential(*phi1)

        self.density_net = nn.Sequential(
            nn.Linear(d_filter, 1)
        )

        self.rgb_net = nn.Sequential(
            nn.Linear(d_filter, 3),
            nn.Sigmoid()
        )

        self.phi2 = BDRFAutoencoder(d_filter, d_brdf_latent)

        self.renderer = Renderer(eval_background = True)

        return
    
    def forward(self,
        rays_o,
        rays_d,
        coarse_z_vals,
        coarse_weights,
        sg_lobes,
        ev100,
        jitter = 0.0,
        perturb = False
    ):
        pts_combined, z_vals_combined = self.__hierachical_sampling(
            rays_o, 
            rays_d, 
            coarse_z_vals, 
            coarse_weights, 
            self.fine_samples,
            perturb
        )

        pts_enc = self.positional_encoder(pts_combined)
        emb = self.phi1_1(pts_enc)
        emb = self.phi1_2(torch.concat([pts_enc, emb], dim = -1))
        density = self.density_net(emb)
        rgb = self.rgb_net(emb)
        brdf_net, brdf_latent = self.phi2(emb)

        raw = {
            'rgb': rgb,
            'density': density
        }

        rgb_map, acc_alpha, weights, alpha = self.__volumetric_render(raw, z_vals_combined, rays_d, jitter)
        brdf_dict = self.__brdf_process(brdf_net, weights, acc_alpha)
        brdf_dict['latent'] = brdf_latent
        normal = self.__normal_process(density, pts_combined, weights, acc_alpha)

        hdr_rgb_map = self.renderer.forward(
            sg_illumination = sg_lobes,
            basecolor = brdf_dict['basecolor'],
            metallic = brdf_dict['metallic'],
            roughness = brdf_dict['roughness'],
            normal = normal,
            alpha = acc_alpha,
            view_dirs = math_utils.normalize(-1 * rays_d)
        )
        
        ldr_rgb_map = math_utils.white_background_compose(
            self.__camera_post_processing(hdr_rgb_map, ev100),
            acc_alpha[..., None]
        )

        fine = {
            'volumetric_rgb': rgb_map,
            'rendered_rgb': ldr_rgb_map,
            'brdf_dict': brdf_dict,
            'normal': normal,
            'acc_alpha': acc_alpha,
            'individual_alphas': alpha
        }

        return fine
                    
    def __sample_pdf(self,
        bins,
        weights,
        n_samples,
        perturb = False
    ):
        pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims = True) 

        cdf = torch.cumsum(pdf, dim = -1) 
        cdf = torch.concat([torch.zeros_like(cdf[..., :1], requires_grad = True), cdf], dim = -1) 

        if not perturb:
            u = torch.linspace(0., 1., n_samples, device = cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [n_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device = cdf.device) 

        u = u.contiguous() 
        inds = torch.searchsorted(cdf, u, right = True) 

        below = torch.clamp(inds - 1, min = 0)
        above = torch.clamp(inds, max = cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim = -1) 

        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim = -1,
                        index = inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim = -1,
                            index = inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples
    
    def __hierachical_sampling(self,
        rays_o,
        rays_d,
        z_vals,
        weights,
        n_samples,
        perturb = False
    ):
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_z_samples = self.__sample_pdf(
            z_vals_mid, 
            weights[..., 1:-1], 
            n_samples,
            perturb = perturb
        )
        new_z_samples = new_z_samples.detach()

        z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim = -1), dim = -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None] 
        return pts, z_vals_combined
    
    def __brdf_process(self, brdf, weights, acc_alpha):
        
        brdf_dict = {
            'basecolor': brdf[..., :3],
            'metallic': brdf[..., 3][..., None],
            'roughness': brdf[..., 4][..., None]
        }

        brdf_dict['basecolor'] = torch.sum(brdf_dict['basecolor'] * weights[..., None], dim = -2)
        brdf_dict['basecolor'] = math_utils.white_background_compose(brdf_dict['basecolor'], acc_alpha[..., None])

        brdf_dict['metallic'] = torch.sum(brdf_dict['metallic'] * weights[..., None], dim = -2)
        brdf_dict['metallic'] = math_utils.white_background_compose(brdf_dict['metallic'], acc_alpha[..., None])
        
        brdf_dict['roughness'] = torch.sum(brdf_dict['roughness'] * weights[..., None], dim = -2)
        brdf_dict['roughness'] = math_utils.white_background_compose(brdf_dict['roughness'], acc_alpha[..., None])

        return brdf_dict
    
    def __normal_process(self, density, pts, weights, acc_alpha):

        normal = torch.autograd.grad(
            density,
            pts,
            grad_outputs = torch.ones_like(density),
            create_graph = True,
            retain_graph = True,
            only_inputs = True
        )[0]

        normal = math_utils.normalize(torch.sum(normal * weights[..., None], dim = -2, keepdim = False))
        normal = math_utils.white_background_compose(-normal, acc_alpha[..., None])
        return normal

    def __camera_post_processing(self, hdr_rgb, ev100):
        exp_val = 1.0 if ev100 is None else math_utils.ev100_to_exp(ev100)
        return math_utils.linear_to_srgb(math_utils.saturate(hdr_rgb * exp_val))
    
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
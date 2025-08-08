import torch
import torch.nn as nn
from model.sg_illumination import SG_Illumination
from model.coarse import Coarse
from model.fine import Fine

class NeRD(nn.Module):
    def __init__(self,
        d_input,
        n_layers_coarse,
        d_filter_coarse,
        coarse_samples,
        n_layers_fine,
        d_filter_fine,
        fine_samples,
        d_brdf_latent,
        n_sg_lobes,
        n_sg_condense
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.illumination = SG_Illumination(
            n_sg_lobes = n_sg_lobes
        )
        self.coarse = Coarse(
            d_input = d_input,
            n_layers = n_layers_coarse,
            d_filter = d_filter_coarse,
            n_sg_lobes = n_sg_lobes,
            n_sg_condense = n_sg_condense,
            coarse_samples = coarse_samples
        )
        self.fine = Fine(
            d_input = d_input,
            n_layers = n_layers_fine,
            d_filter = d_filter_fine,
            d_brdf_latent = d_brdf_latent,
            fine_samples = fine_samples
        )
        return
                    
    def forward(self, 
        rays_o, 
        rays_d,
        near, 
        far,
        ev100,
        train_dict = None
    ):
        
        if train_dict is not None:
            optimize_sgs = train_dict['optimize_sgs']
            jitter = train_dict['jitter']
            perturb = True
        else:
            optimize_sgs = False
            jitter = 0.0
            perturb = False

        clip_range = (0.99, 1.01) if optimize_sgs else None

        self.illumination.apply_wb(
            torch.Tensor([[0.8, 0.8, 0.8]]).to(self.device),
            rays_o,
            ev100,
            clip_range = clip_range,
            grayscale = True
        )
        
        coarse, z_vals, weights = self.coarse.forward(
            rays_o, 
            rays_d, 
            self.illumination(
                requires_grad = False
            ),
            near, 
            far,
            jitter,
            perturb
        )

        fine = self.fine.forward(
            rays_o,
            rays_d,
            z_vals,
            weights,
            self.illumination(
                requires_grad = optimize_sgs
            ),
            ev100,
            jitter,
            perturb
        )

        return coarse, fine
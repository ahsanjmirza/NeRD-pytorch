import torch
import torch.nn as nn
from model.coarse import Coarse
from model.fine import Fine
from model.sg_renderer import Renderer
from model.illumination import SG_Illumination

class NeRD(nn.Module):
    def __init__(self,
        n_lobes: int = 24,
        optimize: bool = True
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.illumination = SG_Illumination(n_lobes, optimize)
        self.coarse = Coarse()
        self.fine = Fine()
        self.renderer = Renderer()
        return
                    
    def forward(self, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor,
        near: float, 
        far: float
    ):
        
          
        return 
    
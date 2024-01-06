import torch
import torch.nn as nn
from typing import Optional, Tuple

class SG_Illumination(nn.Module):
    def __init__(self,
        n_sg_lobes: int = 24
    ):
        super().__init__()
        self.n_sg_lobes = n_sg_lobes
        self.sgs = nn.Parameter(torch.randn(1, self.n_sg_lobes, 7))
        return
                    
    def forward(self, 
        stop_gradient: bool = False,
        rotating_object_pose: Optional[torch.Tensor] = None
    ):
        self.sgs.requires_grad = not stop_gradient
        
        if rotating_object_pose is not None:
            rotation_matrix = torch.linalg.inv(rotating_object_pose[:3, :3])
            environment_ampl = self.sgs[..., :3]
            environment_axis = self.sgs[..., 3:6]
            environment_sharpness = self.sgs[..., 6:]
            environment_axis = environment_axis[..., None, :] * rotation_matrix
            environment_axis = torch.sum(environment_axis, dim = -1)
            return torch.concat([environment_ampl, environment_axis, environment_sharpness], dim = -1)

        return self.sgs
    
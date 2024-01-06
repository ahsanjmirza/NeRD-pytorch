import numpy as np
import os
from dataflow.dataset import RayGenerator
import torch
# data = RayGenerator("./data/real/GoldCape/data_all_16")

# for (rays_o, rays_d), (img, mask), (near, far) in data:
#     print(rays_o.shape, rays_d.shape, img.shape, mask.shape, near, far)
#     break

# from model.positional_encoder import PositionalEncoder

# pe = PositionalEncoder(3, 10, False)

# print(pe(torch.randn(1, 256, 256, 3)).shape)
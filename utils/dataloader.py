import numpy as np
import torch
from torch.utils.data import Dataset

class NeRDDataloader(Dataset):
    def __init__(self, dir):
        self.data = np.load(dir, mmap_mode='r')
        return
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = np.float32(np.copy(self.data[idx]))
        rays_o = torch.from_numpy(data[:3])
        rays_d = torch.from_numpy(data[3:6])
        imgs = torch.from_numpy(data[6:9])
        masks = torch.from_numpy(np.expand_dims(data[9], 0))
        ev100 = torch.from_numpy(np.expand_dims(data[10], 0))
        near = torch.from_numpy(np.expand_dims(data[11], 0))
        far = torch.from_numpy(np.expand_dims(data[12], 0))
        return rays_o, rays_d, imgs, masks, ev100, near, far
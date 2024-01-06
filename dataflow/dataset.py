import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os

class RayGenerator(Dataset):
    def __init__(self, dir):
        """
        Initializes the Dataset object.

        Args:
            dir (str): The directory path where the dataset is located.

        Returns:
            None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_pths = [os.path.join(dir, pth) for pth in os.listdir(dir)]
        return
    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data_pths)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing three tuples:
                - The first tuple contains the rays origin and direction.
                - The second tuple contains the image and mask.
                - The third tuple contains the bounding values.
        """
        data = np.load(self.data_pths[idx])
        
        image = torch.permute(torch.from_numpy(np.float32(data["image"]) / 255.), (2, 0, 1)).to(self.device)
        mask = torch.permute(torch.from_numpy(np.float32(data["mask"]) / 255.), (2, 0, 1)).to(self.device)
        
        height, width = int(data["height"]), int(data["width"])
        
        c2w = torch.from_numpy(np.float32(data["c2w"])).to(self.device)
        
        focal = float(data["focal"])
        
        near, far = float(data["near_global"]), float(data["far_global"])
        
        i, j = torch.meshgrid(
                    torch.arange(width, dtype=torch.float32),
                    torch.arange(height, dtype=torch.float32),
                    indexing = 'ij'
                )
        
        i, j = i.transpose(-1, -2).to(self.device), j.transpose(-1, -2).to(self.device)
        
        directions = torch.stack([(i - width * .5) / focal,
                                  -(j - height * .5) / focal,
                                  -torch.ones_like(i)], dim=-1).to(self.device)
        
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1).to(self.device)
        rays_o = c2w[:3, -1].expand(rays_d.shape).to(self.device)

        return (rays_o, rays_d), (image, mask), (near, far)
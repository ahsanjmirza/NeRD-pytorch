import torch
import torch.nn as nn

class BDRFAutoencoder(nn.Module):
    def __init__(self,
        d_input: int,
        d_latent: int = 2,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent

        self.encoder = nn.Sequential(
            nn.Linear(self.d_input, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.bottleneck = nn. Linear(16, self.d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(self.d_latent, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
        return 
                    
    def forward(self, 
        x: torch.Tensor
    ):
        z = self.bottleneck(self.encoder(x))
        dec = self.decoder(z)
        return dec, z
    
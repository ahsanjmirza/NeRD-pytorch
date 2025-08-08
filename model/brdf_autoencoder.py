import torch
import torch.nn as nn

class BDRFAutoencoder(nn.Module):
    def __init__(self,
        d_input,
        d_latent = 2,
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
            nn.Linear(16, 5),
            nn.Sigmoid()
        )
        return 
                    
    def forward(self, x):
        z = self.encoder(x)
        dec = self.decoder(torch.clip(self.bottleneck(z), -40, 40))
        return dec, z
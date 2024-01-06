import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    Positional Encoder module for NeRF.
    
    Args:
        d_input (int): The input dimension.
        n_freqs (int): The number of frequency bands.
        log_space (bool, optional): Whether to use logarithmic spacing for frequency bands. 
            Defaults to False.
    """
    def __init__(self,
        d_input: int,
        n_freqs: int,
        log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
    def forward(self, x):
        """
        Forward pass of the PositionalEncoder module.
        
        Args:
            x: The input tensor.
        
        Returns:
            The encoded tensor.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim = -1)

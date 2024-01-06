import torch
import torch.nn as nn

def cumprod_exclusive(
    tensor: torch.Tensor
):
    """
    Computes the exclusive cumulative product of a tensor along the last dimension.
    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The exclusive cumulative product tensor.
    """
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def volumetric_render(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False
):
    """
    Converts the raw output of a NeRF network into RGB map, depth map, accumulated map, and weights.

    Args:
        raw (torch.Tensor): Raw output of the NeRF network. Shape: [n_rays, n_samples, 4].
        z_vals (torch.Tensor): Sampled depth values along each ray. Shape: [n_rays, n_samples].
        rays_d (torch.Tensor): Direction vectors of the rays. Shape: [n_rays, 3].
        raw_noise_std (float, optional): Standard deviation of the noise added to the raw predictions for density. Defaults to 0.0.
        white_bkgd (bool, optional): Flag indicating whether to composite onto a white background. Defaults to False.

    Returns:
        rgb_map (torch.Tensor): RGB map of the scene. Shape: [n_rays, 3].
        depth_map (torch.Tensor): Depth map of the scene. Shape: [n_rays].
        acc_map (torch.Tensor): Accumulated map of the scene. Shape: [n_rays].
        weights (torch.Tensor): Weights for each sample along each ray. Shape: [n_rays, n_samples].
    """
    
    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim = -1)  

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim = -1)  

    # Add noise to model's predictions for density. Can be used to 
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std    

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    # print(raw.shape)
    alpha = 1.0 - torch.exp(-nn.functional.softplus(raw[..., 3] + noise - 1) * dists)   

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10) 

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim = -2)  # [n_rays, 3] 

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim = -1) 

    # Disparity map is inverse depth.
    # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
    #                           depth_map / torch.sum(weights, -1))   
    
    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)    

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None]) 

    return rgb_map, depth_map, acc_map, weights
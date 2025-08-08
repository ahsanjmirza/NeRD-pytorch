import torch
from utils import math_utils

mse = torch.nn.MSELoss()
def calc_coarse_loss(payload, target, mask, lambda_advanced_loss):
    inverse_advanced = 1 - lambda_advanced_loss
    
    with torch.no_grad():
        target_mask = math_utils.white_background_compose(target, mask)

    alpha_loss = 0
    alpha_loss = math_utils.segmentation_mask_loss(
        payload['individual_alphas'], 
        payload['acc_alpha'], 
        mask, 
        fg_factor = 0
    ) 

    image_loss = mse(target_mask, payload['volumetric_rgb'])

    losses = {
        "final_loss": image_loss + alpha_loss * 0.4 * inverse_advanced,
        "volumetric_img_loss": image_loss,
        "alpha_loss": alpha_loss
    }

    return losses

def calc_fine_loss(payload, target, mask, lambda_advanced_loss, lambda_color_loss):

    inverse_advanced = 1 - lambda_advanced_loss
    inverse_color = 1 - lambda_color_loss

    with torch.no_grad():
        target_mask = math_utils.white_background_compose(target, mask)

    alpha_loss = 0
    alpha_loss = math_utils.segmentation_mask_loss(
        payload['individual_alphas'], 
        payload['acc_alpha'], 
        mask, 
        fg_factor = 0
    ) 

    image_loss = 0
    image_loss = mse(target_mask, payload['rendered_rgb'])

    direct_img_loss = 0
    direct_img_loss = mse(target_mask, payload['volumetric_rgb'])

    brdf_embedding_loss = 0
    brdf_embedding_loss = (payload['brdf_dict']['latent'] ** 2).mean() * 0.1

    final_loss = (
            image_loss * max(inverse_color, 0.01)
            + alpha_loss * inverse_advanced
            + direct_img_loss * lambda_color_loss
            + brdf_embedding_loss
    )

    losses = {
        "final_loss": final_loss,
        "rendered_image_loss": image_loss,
        "volumetric_img_loss": direct_img_loss,
        "alpha_loss": alpha_loss,
        "brdf_embedding_loss": brdf_embedding_loss
    }

    return losses

def calc_psnr(payload, target, mask):
    
    with torch.no_grad():
        target_mask = math_utils.white_background_compose(target, mask)
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse(target_mask, payload['rendered_rgb'])))
    return psnr
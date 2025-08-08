#!/usr/bin/env python3
import numpy as np
from model.nerd import NeRD
import json
from utils.dataloader import NeRDDataloader
from torch.utils.data import DataLoader
import torch
from utils.losses import calc_coarse_loss, calc_fine_loss, calc_psnr
from utils.misc import get_shape
import os
from imageio.v2 import imwrite

EXPERIMENT_DIR = './experiments/GoldCape'

config = json.load(open(f'{EXPERIMENT_DIR}/config.json', 'r'))

train_data = DataLoader(
    NeRDDataloader(
        f'{EXPERIMENT_DIR}/train.npy'
    ), 
    batch_size = config['train_batch_size'], 
    shuffle = True
)

h, w = get_shape(os.path.join(EXPERIMENT_DIR, 'images' + '_{}'.format(config['factor'])))
val_data = DataLoader(
    NeRDDataloader(
        f'{EXPERIMENT_DIR}/val.npy'
    ), 
    batch_size = w, 
    shuffle = False
)
val_dir = os.path.join(EXPERIMENT_DIR, 'val_logs')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeRD(
    d_input             = config['d_input'],
    n_layers_coarse     = config['n_layers_coarse'],
    d_filter_coarse     = config['d_filter_coarse'],
    coarse_samples      = config['coarse_samples'],
    n_layers_fine       = config['n_layers_fine'],
    d_filter_fine       = config['d_filter_fine'],
    fine_samples        = config['fine_samples'],
    d_brdf_latent       = config['d_brdf_latent'],
    n_sg_lobes          = config['n_sg_lobes'],
    n_sg_condense       = config['n_sg_condense'],
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = config['initial_learning_rate'])
lambda_ = lambda step: config['decay_rate'] ** (step / config['decay_steps_upto'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)

steps = 0

if config['continue_train']:
    continue_dict = torch.load(f'{EXPERIMENT_DIR}/weights.pth')
    model.load_state_dict(
        continue_dict['model_state_dict']
    )
    optimizer.load_state_dict(
        continue_dict['optimizer_state_dict']
    )
    scheduler.load_state_dict(
        continue_dict['scheduler_state_dict']
    )
    steps = continue_dict['steps']

while steps < config['total_steps']:
    for rays_o, rays_d, imgs, masks, ev100, near, far in train_data:
        model.train()
        rays_o, rays_d = rays_o.to(device), rays_d.to(device)
        imgs, masks, ev100 = imgs.to(device), masks.to(device), ev100.to(device)
        optimizer.zero_grad()

        coarse_payload, fine_payload = model.forward(
            rays_o, 
            rays_d, 
            near[0, 0], 
            far[0, 0], 
            ev100, 
            {
                'optimize_sgs': True if steps > 1000 else False, 
                'jitter': config['jitter']
            }
        )

        lambda_advanced_loss = 0.9 ** (steps / 5000) if steps < 5000 else 0.9
        lambda_color_loss = 0.75 ** (steps / 1500) if steps < 1500 else 0.75

        coarse_loss_dict = calc_coarse_loss(
            coarse_payload, 
            imgs, 
            masks,
            lambda_advanced_loss
        )

        fine_loss_dict = calc_fine_loss(
            fine_payload, 
            imgs, 
            masks,
            lambda_advanced_loss,
            lambda_color_loss
        )

        loss = coarse_loss_dict['final_loss'] + fine_loss_dict['final_loss']
        # loss = (0.6 * fine_loss_dict['rendered_image_loss']) + (0.4 * (0.4 * coarse_loss_dict['volumetric_img_loss']) + (0.6 * fine_loss_dict['volumetric_img_loss']))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        
        if steps < config['decay_steps_upto']: scheduler.step()
        steps += 1
        
        if steps % config['verbose'] == 0:
            print(
                "Steps: ", steps,
                "\tTotal Loss: ", round(loss.item(), 4), 
                "\tCoarse Volumetric Loss: ", round(coarse_loss_dict['volumetric_img_loss'].item(), 4), 
                "\tCoarse Alpha Loss: ", round(coarse_loss_dict['alpha_loss'].item(), 4), 
                "\tRendered Image Loss: ", round(fine_loss_dict['rendered_image_loss'].item(), 4),
                "\tFine Volumetric Loss: ", round(fine_loss_dict['volumetric_img_loss'].item(), 4),
                "\tFine Alpha Loss: ", round(fine_loss_dict['alpha_loss'].item(), 4),
                "\tBRDF Latent Loss: ", round(fine_loss_dict['brdf_embedding_loss'].item(), 5),
                "\tGrad Norm: ", round(grad_norm.item(), 4),
            )

        if steps % config['val_interval'] == 0:
            torch.save(
                { 
                    'steps': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(EXPERIMENT_DIR, 'weights.pth')
            )
            model.eval()
            val_loss = 0
            fine_ldr_rgb = np.zeros((h * w, 3), dtype = np.float32)
            fine_vol_rgb = np.zeros((h * w, 3), dtype = np.float32)
            normal = np.zeros((h * w, 3), dtype = np.float32)
            brdf_base_rgb = np.zeros((h * w, 3), dtype = np.float32)
            brdf_metallic = np.zeros((h * w), dtype = np.float32)
            brdf_roughness = np.zeros((h * w), dtype = np.float32)
            coarse_vol_rgb = np.zeros((h * w, 3), dtype = np.float32)
            i = 0 
            for rays_o, rays_d, _, _, ev100, near, far in val_data:
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)
                ev100 = ev100.to(device)
                coarse, fine = model.forward(
                    rays_o, 
                    rays_d, 
                    near[0, 0], 
                    far[0, 0], 
                    ev100, 
                    None
                )
                fine_ldr_rgb[i:i+h, ...] = fine['rendered_rgb'].detach().cpu().numpy()
                fine_vol_rgb[i:i+h, ...] = fine['volumetric_rgb'].detach().cpu().numpy()
                normal[i:i+h, ...] = (fine['normal'].detach().cpu().numpy() + 1) / 2
                brdf_base_rgb[i:i+h, ...] = fine['brdf_dict']['basecolor'].detach().cpu().numpy()
                brdf_metallic[i:i+h] = fine['brdf_dict']['metallic'].detach().cpu().numpy()[..., 0]
                brdf_roughness[i:i+h] = fine['brdf_dict']['roughness'].detach().cpu().numpy()[..., 0]
                coarse_vol_rgb[i:i+h, ...] = coarse['volumetric_rgb'].detach().cpu().numpy()
                i += h
            fine_ldr_rgb = np.uint8((np.clip(fine_ldr_rgb, 0, 1) * 255).reshape((h, w, 3)))
            fine_vol_rgb = np.uint8((np.clip(fine_vol_rgb, 0, 1) * 255).reshape((h, w, 3)))
            normal = np.uint8((np.clip(normal, 0, 1) * 255).reshape((h, w, 3)))
            brdf_base_rgb = np.uint8((np.clip(brdf_base_rgb, 0, 1) * 255).reshape((h, w, 3)))
            brdf_metallic = np.uint8((np.clip(brdf_metallic, 0, 1) * 255).reshape((h, w)))
            brdf_roughness = np.uint8((np.clip(brdf_roughness, 0, 1) * 255).reshape((h, w)))
            coarse_vol_rgb = np.uint8((np.clip(coarse_vol_rgb, 0, 1) * 255).reshape((h, w, 3)))
            imwrite(os.path.join(val_dir, "fine_rendered_img", str(steps)+'.png'), fine_ldr_rgb)
            imwrite(os.path.join(val_dir, "fine_volumetric_img", str(steps)+'.png'), fine_vol_rgb)
            imwrite(os.path.join(val_dir, "normal", str(steps)+'.png'), normal)
            imwrite(os.path.join(val_dir, "brdf_base_rgb", str(steps)+'.png'), brdf_base_rgb)
            imwrite(os.path.join(val_dir, "brdf_metallic", str(steps)+'.png'), brdf_metallic)
            imwrite(os.path.join(val_dir, "brdf_roughness", str(steps)+'.png'), brdf_roughness)
            imwrite(os.path.join(val_dir, "coarse_volumetric_img", str(steps)+'.png'), coarse_vol_rgb)

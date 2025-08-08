#!/usr/bin/env python3
import numpy as np
import os
from utils.dataflow import (
    _minify, 
    handle_exif, 
    ensureNoChannel, 
    recenter_poses, 
    spherify_poses, 
    poses_avg,
    normalize,
    render_path_spiral
)
import imageio.v2 as imageio
from tempfile import mkdtemp
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import json


EXPERIMENT_DIR = './experiments/GoldCape'


config = json.load(open(os.path.join(EXPERIMENT_DIR, 'config.json')))
IMAGES_DIR = os.path.join(EXPERIMENT_DIR, 'images')
MASKS_DIR = os.path.join(EXPERIMENT_DIR, 'masks')
POSES_PTH = os.path.join(EXPERIMENT_DIR, 'poses_bounds.npy')
TRAIN_DIR = os.path.join(EXPERIMENT_DIR, 'train.npy')
VAL_DIR = os.path.join(EXPERIMENT_DIR, 'val.npy')
FACTOR = config['factor']
RECENTER = config['recenter']
SPHERIFY = config['spherify']
PATH_ZFLAT = config['path_zflat']

print("Preparing dataset..." )
poses_arr = np.load(POSES_PTH)
poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
# print(poses.shape)
bds = poses_arr[:, -2:].transpose([1, 0])

img0 = [
    os.path.join(EXPERIMENT_DIR, "images", f)
    for f in sorted(os.listdir(os.path.join(EXPERIMENT_DIR, "images")))
    if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
][0]
sh = imageio.imread(img0).shape

ev100s = handle_exif(EXPERIMENT_DIR)

sfx = ""
if FACTOR > 1:
    sfx = "_{}".format(FACTOR)
    _minify(EXPERIMENT_DIR, FACTOR)

IMAGES_DIR = os.path.join(EXPERIMENT_DIR, 'images' + sfx)
MASKS_DIR = os.path.join(EXPERIMENT_DIR, 'masks' + sfx)

imgfiles = [
        os.path.join(IMAGES_DIR, f)
        for f in sorted(os.listdir(IMAGES_DIR))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
]

masksfiles = [
    os.path.join(MASKS_DIR, f)
    for f in sorted(os.listdir(MASKS_DIR))
    if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
]

sh = imageio.imread(imgfiles[0]).shape
poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
poses[2, 4, :] = poses[2, 4, :] * 1.0 / FACTOR

def imread(f):
    if f.endswith("png"):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

temp_imgs = os.path.join(mkdtemp(), 'temp_imgs.dat')
temp_masks = os.path.join(mkdtemp(), 'temp_masks.dat')

imgs = np.memmap(temp_imgs, dtype = np.uint8, mode = 'w+', shape = (sh[0], sh[1], 3, len(imgfiles)))
masks = np.memmap(temp_masks, dtype = np.uint8, mode = 'w+', shape = (sh[0], sh[1], len(masksfiles)))

win = 0
for f in tqdm(zip(imgfiles, masksfiles)):
    imgs[..., win] = imread(f[0])
    masks[..., win] = ensureNoChannel(imread(f[1]))
    win += 1

imgs = np.float32(imgs) / 255.0
masks = np.float32(masks) / 255.0

# From RH coordinates to our LH coordinates
poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
poses = np.moveaxis(poses, -1, 0).astype(np.float32)
imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
masks = np.expand_dims(np.moveaxis(masks, -1, 0), -1).astype(np.float32)
bds = np.moveaxis(bds, -1, 0).astype(np.float32)

if RECENTER:
    print("Recentering poses...")
    poses = recenter_poses(poses)

if SPHERIFY:
    print("Spherifying poses...")
    poses, render_poses, bds = spherify_poses(poses, bds)
else:
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))
    close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
    dt = 0.75
    mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    zdelta = close_depth * 0.2
    tt = poses[:, :3, 3]  
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    
    if PATH_ZFLAT:
        zloc = -close_depth * 0.1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.0
        N_rots = 1
        N_views /= 2
    
    render_poses = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
    )

render_poses = np.array(render_poses).astype(np.float32)
poses = poses.astype(np.float32)

c2w = poses[:, :3, :4]
hwf = poses[0, :3, -1]

h, w, focal = hwf
h, w = int(h), int(w)

dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in c2w])
origins = c2w[:, :3, -1]

print("Saving pose plot...")
fig = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')
fig.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(), length=0.5, normalize=True
)

fig.set_xlabel('X')
fig.set_ylabel('Y')
fig.set_zlabel('Z')
plt.savefig(os.path.join(EXPERIMENT_DIR, "poses_plot.png"))
plt.clf()
plt.cla()
plt.close()

near_global = bds.min()
far_global = bds.max()

train = np.memmap(os.path.join(mkdtemp(), 'temp_train.dat'), dtype = np.float32, mode = 'w+', shape = (h * w * (imgs.shape[0] - 1), 13))
val = np.memmap(os.path.join(mkdtemp(), 'temp_val.dat'), dtype = np.float32, mode = 'w+', shape = (h * w, 13))

print("Preparing training dataset...")
win = 0
for idx in tqdm(range(1, imgs.shape[0])):
    img_ = imgs[idx]
    mask_ = masks[idx]
    c2w_ = torch.from_numpy(c2w[idx])
    ev100_ = ev100s[idx]

    i, j = torch.meshgrid(
                torch.arange(w, dtype=torch.float32),
                torch.arange(h, dtype=torch.float32),
                indexing = 'ij'
            )
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack(
        [
            (i - w * 0.5) / focal,
            -(j - h * 0.5) / focal,
            -torch.ones_like(i)
        ], 
        dim = -1
    )

    rays_d = torch.sum(directions[..., None, :] * c2w_[:3, :3], dim=-1)
    rays_o = c2w_[:3, -1].expand(rays_d.shape)

    rays_o = rays_o.flatten(0, 1).numpy()
    rays_d = rays_d.flatten(0, 1).numpy()
    img_ = torch.from_numpy(img_).flatten(0, 1).numpy()
    mask_ = np.expand_dims(mask_.flatten(), -1)

    train[win * h * w:(win + 1) * h * w, :] = np.concatenate(
        (
            rays_o, 
            rays_d, 
            img_, 
            mask_, 
            ev100_ * np.ones_like(mask_), 
            near_global * np.ones_like(mask_), 
            far_global * np.ones_like(mask_)
        ),
        axis = 1
    )
    win += 1
print("Saving training dataset...")
np.save(TRAIN_DIR, train)

print("Preparing validation dataset...")
img_ = imgs[0]
mask_ = masks[0]
c2w_ = torch.from_numpy(c2w[0])
ev100_ = ev100s[0]

i, j = torch.meshgrid(
            torch.arange(w, dtype=torch.float32),
            torch.arange(h, dtype=torch.float32),
            indexing = 'ij'
        )
i, j = i.transpose(-1, -2), j.transpose(-1, -2)
directions = torch.stack(
    [
        (i - w * 0.5) / focal,
        -(j - h * 0.5) / focal,
        -torch.ones_like(i)
    ], 
    dim = -1
)

rays_d = torch.sum(directions[..., None, :] * c2w_[:3, :3], dim=-1)
rays_o = c2w_[:3, -1].expand(rays_d.shape)

rays_o = rays_o.flatten(0, 1).numpy()
rays_d = rays_d.flatten(0, 1).numpy()
img_ = torch.from_numpy(img_).flatten(0, 1).numpy()
mask_ = np.expand_dims(mask_.flatten(), -1)

val[:] = np.concatenate(
    (
        rays_o, 
        rays_d, 
        img_, 
        mask_, 
        ev100_ * np.ones_like(mask_), 
        near_global * np.ones_like(mask_), 
        far_global * np.ones_like(mask_)
    ),
    axis = 1
)
print("Saving validation dataset...")
np.save(VAL_DIR, val)
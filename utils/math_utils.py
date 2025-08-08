import torch
import torch.nn as nn

EPS = 1e-10

def saturate(x, low = 0.0, high = 1.0):
    return torch.clip(x, low, high)

def fill_like(x, val):
    return torch.ones_like(x) * val

def safe_sqrt(x):
    return torch.sqrt(torch.maximum(x, torch.ones_like(x) * EPS))

def safe_exp(x):
    return torch.exp(torch.minimum(x, torch.ones_like(x) * 87.5))

def safe_log(x):
    return torch.log(torch.minimum(x, torch.ones_like(x) * 33e37))

def magnitude(x):
    return safe_sqrt(torch.sum(x ** 2, dim = -1))

def dot(x, y):
    return torch.sum(x * y, dim = -1, keepdim=True)

def l2Norm(x):
    return dot(x, x)

def normalize(x):
    mag = magnitude(x)[..., None]
    return torch.where(mag <= torch.zeros_like(mag), torch.zeros_like(x), torch.nan_to_num(x / mag, 0))
    # return torch.nn.functional.normalize(x, dim = -1)

def srgb_to_linear(x):
    x = saturate(x)
    switch_val = 0.04045
    return torch.where(
        torch.ge(x, switch_val),
        torch.pow((torch.maximum(x, torch.ones_like(x) * switch_val) + 0.055) / 1.055, 2.4),
        x / 12.92
    )

def linear_to_srgb(x):
    x = saturate(x)
    switch_val = 0.0031308
    return torch.where(
        torch.ge(x, switch_val),
        1.055 * torch.pow(torch.maximum(x, torch.ones_like(x) * switch_val), 1.0 / 2.4) - 0.055,
        x * 12.92,
    )

def mix(x, y, a):
    a = saturate(a)
    return (x * (1 - a)) + (y * a)


def repeat(x, n, axis):
    repeat = [1 for _ in range(len(x.shape))]
    repeat[axis] = n
    return torch.tile(x, tuple(repeat))

def to_vec3(x):
    return repeat(x, 3, -1)

def ev100_to_exp(ev100): 
    maxL = 1.2 * (2.0 ** ev100)
    # return torch.nan_to_num(1.0 / maxL, EPS)
    return torch.maximum(1.0 / torch.maximum(maxL, torch.ones_like(maxL) * EPS), torch.ones_like(maxL) * EPS) 

def reflect(d, n):
    return d - 2 * dot(d, n) * n

def background_compose(x, y, mask):
    maskClip = saturate(mask)
    out = x * maskClip + (1.0 - maskClip) * y
    return out

def white_background_compose(x, mask):
    return background_compose(x, torch.ones_like(x), mask)

def segmentation_mask_loss(alpha, acc_map, mask, fg_factor=1):
    loss_background = torch.sum(
        torch.abs(alpha - mask), dim = -1, keepdim = True
    )

    loss_foreground = (
        torch.abs(acc_map - mask) if fg_factor > 0 else torch.zeros_like(mask)
    )

    mask_binary = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))

    return torch.mean(
        torch.where(
            mask > 0.5,
            loss_foreground * mask_binary * fg_factor,
            loss_background * (1 - mask_binary)
        )
    )

def div_no_nan(a, b):
    return torch.nan_to_num(a / b, 0)

def recp_no_nan(a):
    return torch.nan_to_num(1 / a, 0)
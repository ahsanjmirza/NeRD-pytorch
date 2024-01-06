import torch
import torch.nn as nn

EPS = 1e-7

def saturate(x, low = 0.0, high = 1.0):
    return torch.clip(x, low, high)

def fill_like(x, val):
    return torch.ones_like(x) * val

def safe_sqrt(x):
    return torch.sqrt(torch.maximum(x, EPS))

def safe_exp(x):
    return torch.exp(torch.minimum(x, 87.5))

def magnitude(x):
    return torch.safe_sqrt(torch.sum(x ** 2, dim = -1))

def dot(x, y):
    return torch.sum(x * y, dim = -1, keepdim=True)

def l2Norm(x):
    return dot(x, x)

def normalize(x):
    mag = magnitude(x)
    torch.where(mag <= safe_sqrt(float(0)), torch.zeros_like(x), x / mag)

def srgb_to_linear(x):
    x = saturate(x)

    switch_val = 0.04045
    return torch.where(
        torch.ge(x, switch_val),
        torch.pow((torch.maximum(x, switch_val) + 0.055) / 1.055, 2.4),
        x / 12.92
    )

def linear_to_srgb(x):
    x = saturate(x)

    switch_val = 0.0031308
    return torch.where(
        torch.ge(x, switch_val),
        1.055 * torch.pow(torch.maximum(x, switch_val), 1.0 / 2.4) - 0.055,
        x * 12.92,
    )

def mix(x, y, a):
    a = torch.clip(a, 0.0, 1.0)
    return x * (1 - a) + y * a


def repeat(x, n, axis):
    repeat = [1 for _ in range(len(x.shape))]
    repeat[axis] = n
    return torch.tile(x, tuple(repeat))

def to_vec3(x):
    return repeat(x, 3, -1)



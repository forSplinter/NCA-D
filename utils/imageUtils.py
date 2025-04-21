import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def make_seed(shape, n_channel):
    H, W = shape
    seed = np.zeros((H, W, n_channel), np.float32)
    seed[H // 2, W // 2, 0] = 1.0
    return seed


def make_circle_masks(n, h, w, min_radius=5, max_radius=15):
    masks = np.ones((n, h, w), np.float32)
    for i in range(n):
        x = np.random.randint(max_radius, w - max_radius)
        y = np.random.randint(max_radius, h - max_radius)
        r = np.random.randint(min_radius, max_radius)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        masks[i][dist_from_center <= r] = 0.0
    return masks


def to_rgb(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if x.shape[1] == 3:
        x = np.transpose(x, (0, 2, 3, 1)) if x.shape[-1] == 3 else x
        rgb = x
    else:
        rgb = np.transpose(x[:, :3], (0, 2, 3, 1))

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

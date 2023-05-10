import numpy as np
import torch


def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def masked_mean(tensor, mask):
    return torch.sum(tensor * mask[:, :, None], dim=[0, 1]) / torch.sum(mask)


def masked_std(tensor, mask):
    mean = masked_mean(tensor, mask)
    return torch.sqrt(torch.sum(tensor ** 2 * mask[:, :, None], dim=[0, 1]) / torch.sum(mask) - mean ** 2)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

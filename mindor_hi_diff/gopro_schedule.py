import math
import torch
import numpy as np

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    linear_start = float(linear_start)
    linear_end = float(linear_end)

    if schedule == "linear":
        return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

    if schedule == "quad":
        return np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2

    if schedule == "const":
        return linear_end * np.ones(n_timestep, dtype=np.float64)

    if schedule == "jsd":
        return 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)

    if schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]

        return betas.clamp(max=0.999)

    raise NotImplementedError(f"Unsupported schedule: {schedule!r}")

import torch
import numpy as np
from functools import partial
from .gopro_schedule import make_beta_schedule

class GoProPrior(torch.nn.Module):
    def __init__(self, condition, denoise, steps):
        super().__init__()

        self.net_le_dm = condition
        self.net_d = denoise

        self.register_schedule(timesteps=steps, linear_start=0.1, linear_end=0.99)

    def register_schedule(self, timesteps, linear_start, linear_end):
        to_torch = partial(torch.tensor, dtype=torch.float32, device="cpu")

        betas = make_beta_schedule(schedule="linear", n_timestep=timesteps, linear_start=linear_start, linear_end=linear_end)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t

        return posterior_mean

    def p_sample(self, x, t, condition):
        noise = self.net_d(x, condition, torch.full(x.shape, t + 1, device=self.betas.device, dtype=torch.long))
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        x_recon.clamp_(-1., 1.)

        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    def forward(self, image):
        condition = self.net_le_dm(image)
        prior = torch.randn_like(condition)

        for i in reversed(range(0, self.num_timesteps)):
            prior = self.p_sample(prior, i, condition)

        return prior

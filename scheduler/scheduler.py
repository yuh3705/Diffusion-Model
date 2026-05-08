import torch
import numpy as np


class Scheduler:
    def __init__(self, num_steps, beta_start, beta_end, schedule='linear'):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        if schedule == 'linear':
            self.betas = np.linspace(beta_start, beta_end, num_steps)

        elif schedule == 'quad':
            self.betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_steps,
                    dtype=np.float64,
                ) ** 2
            )

        elif schedule == "const":
            self.betas = beta_end * np.ones(num_steps, dtype=np.float64)

        elif schedule == "jsd":
            self.betas = 1.0 / np.linspace(
                num_steps, 1, num_steps, dtype=np.float64
            )

        elif schedule == "sigmoid":

            def sigmoid(x):
                return 1 / (np.exp(-x) + 1)

            self.betas = np.linspace(-6, 6, num_steps)
            self.betas = (
                sigmoid(self.betas) * (beta_end - beta_start)
                + beta_start
            )

        else:
            raise NotImplementedError(schedule)

        self.betas = torch.tensor(self.betas, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

    def add_noise(self, x0, noise, t):
        device = x0.device
        batch_size = x0.shape[0]

        sqrt_alpha_cumprod = (
            self.sqrt_alphas_cumprod.to(device)[t]
            .reshape(batch_size, 1, 1, 1)
        )

        sqrt_one_minus_alpha_cumprod = (
            self.sqrt_one_minus_alphas_cumprod.to(device)[t]
            .reshape(batch_size, 1, 1, 1)
        )

        xt = (
            sqrt_alpha_cumprod * x0
            + sqrt_one_minus_alpha_cumprod * noise
        )

        return xt

    def sample_prev_step(self, xt, noise_pred, t):
        device = xt.device

        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)

        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)

        sqrt_one_minus_alphas_cumprod = (
            self.sqrt_one_minus_alphas_cumprod.to(device)
        )

        if isinstance(t, int):
            t_tensor = torch.tensor([t], device=device)
        else:
            t_tensor = t

        x0 = (
            xt
            - sqrt_one_minus_alphas_cumprod[t_tensor].reshape(-1, 1, 1, 1)
            * noise_pred
        ) / sqrt_alphas_cumprod[t_tensor].reshape(-1, 1, 1, 1)

        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = (
            xt
            - (
                betas[t_tensor].reshape(-1, 1, 1, 1)
                * noise_pred
                / sqrt_one_minus_alphas_cumprod[t_tensor].reshape(-1, 1, 1, 1)
            )
        ) / torch.sqrt(alphas[t_tensor]).reshape(-1, 1, 1, 1)

        if torch.all(t_tensor == 0):
            return mean, x0

        var = (
            betas[t_tensor].reshape(-1, 1, 1, 1)
            * (
                1.0
                - alphas_cumprod[t_tensor - 1].reshape(-1, 1, 1, 1)
            )
            / (
                1.0
                - alphas_cumprod[t_tensor].reshape(-1, 1, 1, 1)
            )
        )

        sigma = torch.sqrt(var)

        z = torch.randn_like(xt)

        return mean + sigma * z, x0
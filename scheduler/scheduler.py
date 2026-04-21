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
                )
                ** 2
            )
        elif schedule == "const":
            self.betas = beta_end * np.ones(num_steps, dtype=np.float64)
        elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            self.betas = 1.0 / np.linspace(
                num_steps, 1, num_steps, dtype=np.float64
            )
        elif schedule == "sigmoid":
            def sigmoid(x):
                return 1 / (np.exp(-x) + 1)
            self.betas = np.linspace(-6, 6, num_steps)
            self.betas = sigmoid(self.betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(schedule)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, e, t):
        x0_shape = x0.shape
        batch_size = x0_shape[0]

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].to(x0.device).reshape(batch_size)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].to(x0.device).reshape(batch_size)

        return (sqrt_alpha_cumprod.to(x0.device) * x0 + sqrt_one_minus_alpha_cumprod.to(x0.device) * e)

    def sample_prev_step(self, xt, e_pred, t):
        x0 = (xt - self.sqrt_one_minus_alphas_cumprod[t].to(xt.device)*e_pred)\
            / self.sqrt_alphas_cumprod[t].to(xt.device)
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = (xt - self.betas[t].to(xt.device) * e_pred / self.self.sqrt_one_minus_alphas_cumprod[t].to(xt.device)) \
            / self.alphas[t].to(xt.device) 
        
        if t == 0:
            return mean, x0
        else:
            var = self.betas[t].to(xt.device) * (1.0 - self.alphas_cumprod[t-1].to(xt.device)) / (1.0 - self.alphas_cumprod[t].to(xt.device))
            z = torch.randn_like(xt)
            sigma = var ** 0.5
            return mean + sigma * z, x0
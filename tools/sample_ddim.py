import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision
import yaml
from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.unet import UNet
from scheduler.scheduler import Scheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_ddim_timesteps(num_ddim_steps, num_train_timesteps, device):
    if num_ddim_steps < 1:
        raise ValueError("--ddim-steps must be >= 1")
    if num_ddim_steps > num_train_timesteps:
        raise ValueError("--ddim-steps cannot exceed diffusion num_timesteps")

    timesteps = torch.linspace(
        0,
        num_train_timesteps - 1,
        steps=num_ddim_steps,
        device=device,
    ).long()

    return torch.flip(timesteps, dims=[0])


def ddim_step(xt, noise_pred, t, prev_t, alphas_cumprod, eta):
    alpha_t = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    alpha_prev = alphas_cumprod[prev_t].reshape(-1, 1, 1, 1)

    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    x0_pred = (xt - sqrt_one_minus_alpha_t * noise_pred) / torch.sqrt(alpha_t)
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

    sigma_t = eta * torch.sqrt(
        ((1.0 - alpha_prev) / (1.0 - alpha_t))
        * (1.0 - alpha_t / alpha_prev)
    )
    pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0)) * noise_pred
    noise = torch.randn_like(xt) if eta > 0 else torch.zeros_like(xt)

    x_prev = torch.sqrt(alpha_prev) * x0_pred + pred_dir + sigma_t * noise
    return x_prev, x0_pred


def save_grid(images, output_dir, step_name, num_grid_rows):
    ims = torch.clamp(images, -1, 1).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, f"{step_name}.png"))
    img.close()


def sample(model, scheduler, train_config, dataset_config, diffusion_config, args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    num_train_timesteps = diffusion_config["num_timesteps"]
    timesteps = make_ddim_timesteps(args.ddim_steps, num_train_timesteps, device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    xt = torch.randn(
        (
            train_config["num_samples"],
            dataset_config["im_channels"],
            dataset_config["im_size"],
            dataset_config["im_size"],
        ),
        device=device,
    )

    output_dir = args.output_dir or os.path.join(train_config["task_name"], "ddim_samples")
    save_every = max(args.save_every, 1)

    for step_idx, t in enumerate(tqdm(timesteps, desc="DDIM sampling")):
        t_batch = torch.full((xt.shape[0],), t.item(), device=device, dtype=torch.long)
        noise_pred = model(xt, t_batch)

        is_last = step_idx == len(timesteps) - 1
        prev_t = torch.full_like(t_batch, 0 if is_last else timesteps[step_idx + 1].item())

        xt, x0_pred = ddim_step(
            xt,
            noise_pred,
            t_batch,
            prev_t,
            alphas_cumprod,
            eta=args.eta,
        )

        if args.save_intermediates and (is_last or step_idx % save_every == 0):
            save_grid(xt, output_dir, f"xt_{t.item()}", train_config["num_grid_rows"])

    save_grid(x0_pred, output_dir, "x0_final", train_config["num_grid_rows"])


def infer(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    diffusion_config = config["diffusion_params"]
    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    scheduler = Scheduler(
        num_steps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )

    model = UNet(
        im_channels=dataset_config["im_channels"],
        model_config=model_config,
    ).to(device)
    model.eval()

    ckpt_path = os.path.join(train_config["task_name"], train_config["ckpt_name"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading model from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    with torch.no_grad():
        sample(model, scheduler, train_config, dataset_config, diffusion_config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a trained DDPM model with DDIM.")
    parser.add_argument("--config", default="config/default.yaml", type=str)
    parser.add_argument("--ddim-steps", default=50, type=int)
    parser.add_argument("--eta", default=0.0, type=float, help="0.0 is deterministic DDIM; >0 adds stochasticity.")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--save-every", default=5, type=int)
    args = parser.parse_args()
    infer(args)

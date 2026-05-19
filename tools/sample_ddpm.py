import os
import sys
from pathlib import Path
import yaml
import torch
import torchvision
import argparse
from torchvision.utils import make_grid

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.unet import UNet
from scheduler.scheduler import Scheduler
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, train_config, dataset_config, diffusion_config):
    xt = torch.randn((train_config['num_samples'],
                      dataset_config['im_channels'],
                      dataset_config['im_size'],
                      dataset_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        xt, x0_pred = scheduler.sample_prev_step(xt, noise_pred, torch.as_tensor(i).to(device))

        ims = torch.clamp(xt, -1, 1).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.makedirs(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', f'x0_{i}.png'))
        img.close()

def infer(args):
    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params'] 
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = Scheduler(num_steps = diffusion_config['num_timesteps'],
                                beta_start=diffusion_config['beta_start'], 
                                beta_end=diffusion_config['beta_end'])
    
    model = UNet(
        im_channels=dataset_config['im_channels'],
        model_config=model_config
    ).to(device)
    model.eval()

    if os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        print(f"Loading model from {os.path.join(train_config['task_name'], train_config['ckpt_name'])}...")
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']), map_location=device))
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {os.path.join(train_config['task_name'], train_config['ckpt_name'])}"
        )

    with torch.no_grad():
        sample(model, scheduler, train_config, dataset_config, diffusion_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from a trained DDPM model.')
    parser.add_argument('--config', dest='config',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)

import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.mnist_dataset import MNISTDataset
from dataset.celebhq_dataset import CelebHQDataset
from model.unet import UNet
from scheduler.scheduler import Scheduler
from tqdm import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dataset(dataset_config):
    dataset_name = dataset_config.get('name', 'mnist').lower()
    if dataset_name == 'mnist':
        return MNISTDataset(split='train', im_path=dataset_config['im_path'])
    if dataset_name in ['celebhq', 'celeba_hq', 'celeba-hq']:
        return CelebHQDataset(
            im_path=dataset_config['im_path'],
            im_size=dataset_config['im_size'],
            im_exts=dataset_config.get('im_exts'),
        )
    raise ValueError(f"Unsupported dataset: {dataset_config.get('name')}")

def train(args):
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
    
    dataset = build_dataset(dataset_config)
    data_loader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )
    
    model = UNet(
        im_channels=dataset_config['im_channels'],
        model_config=model_config
    ).to(device)
    model.train()

    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'])

    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}/{num_epochs}")
        for im in progress_bar:
            optimizer.zero_grad()
            im = im.float().to(device)

            noise = torch.randn_like(im).to(device)

            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{np.mean(losses):.4f}")

        tmp_ckpt_path = f"{ckpt_path}.tmp"
        torch.save(model.state_dict(), tmp_ckpt_path)
        os.replace(tmp_ckpt_path, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DDPM model.')
    parser.add_argument('--config', dest='config',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)

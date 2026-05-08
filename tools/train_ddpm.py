import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.mnist_dataset import MNISTDataset
from model.unet import UNet
from scheduler.scheduler import Scheduler
from tqdm import tqdm
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    mnist = MNISTDataset(split='train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True)
    
    model = UNet(
        im_channels=dataset_config['im_channels'],
        model_config=model_config
    ).to(device)
    model.train()

    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'])

    if os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        print(f"Loading model from {os.path.join(train_config['task_name'], train_config['ckpt_name'])}...")
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']), map_location=device))

    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
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

            print(f"Epoch {epoch_idx+1}/{num_epochs}, Loss: {np.mean(losses):.4f}")

            torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['ckpt_name']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DDPM model.')
    parser.add_argument('--config', dest='config',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
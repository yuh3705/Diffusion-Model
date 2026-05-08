from torchvision.datasets import MNIST
import os

for split, train in [('train', True), ('test', False)]:
    ds = MNIST(root='./data', train=train, download=True)

    base_folder = f'./data/mnist/{split}/images'

    for label in range(10):
        os.makedirs(f'{base_folder}/{label}', exist_ok=True)

    for idx, (img, label) in enumerate(ds):
        img.save(f'{base_folder}/{label}/{idx}.png')
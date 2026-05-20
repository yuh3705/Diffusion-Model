import glob
import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset


class CelebHQDataset(Dataset):
    def __init__(self, im_path, im_size, im_exts=None):
        self.im_path = im_path
        self.im_size = im_size
        self.im_exts = im_exts or ['jpg']
        self.images = self.load_images(im_path)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(im_size),
            torchvision.transforms.CenterCrop(im_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"Image path {im_path} does not exist."
        images = []
        for ext in self.im_exts:
            images.extend(glob.glob(os.path.join(im_path, '**', f'*.{ext}'), recursive=True))
            images.extend(glob.glob(os.path.join(im_path, '**', f'*.{ext.upper()}'), recursive=True))
        images = sorted(set(images))
        assert len(images) > 0, f"No images found in {im_path}."
        print(f"Loaded {len(images)} images from {im_path}.")
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx]).convert('RGB')
        return self.transform(im)

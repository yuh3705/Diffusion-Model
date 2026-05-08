import os
import glob
 
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, split, im_path, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"Image path {im_path} does not exist."
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, f'*.{self.im_ext}')):
                ims.append(fname)
                labels.append(int(d_name))
        print(f"Loaded {len(ims)} images from {im_path}.")
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        im = Image.open(self.images[idx])
        im_tensor = torchvision.transforms.ToTensor()(im)

        im_tensor = (2 * im_tensor) - 1
        return im_tensor
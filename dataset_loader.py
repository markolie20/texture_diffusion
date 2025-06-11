import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MinecraftTextureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGBA')  # Use RGBA for 4 channels
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(4, 16, 16)

        if self.transform:
            image = self.transform(image)
        
        return image

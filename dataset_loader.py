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
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or raise an error if preferred
            return torch.zeros(3, 16, 16) # Assuming 3 channels, 16x16 size

        if self.transform:
            image = self.transform(image)
        
        return image

if __name__ == '__main__':
    data_dir = 'assets/blocks/'

    # Define the transform pipeline
    transform_pipeline = transforms.Compose([
        transforms.Resize((16, 16)),  # Ensure all images are 16x16
        transforms.ToTensor(),         # Convert PIL image to tensor and scales to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Create an instance of MinecraftTextureDataset
    dataset = MinecraftTextureDataset(root_dir=data_dir, transform=transform_pipeline)

    # Print the total number of textures found
    print(f"Total textures found: {len(dataset)}")

    if len(dataset) == 0:
        print("No textures found. Please check the data_dir path and ensure images are present.")
    else:
        # Create a DataLoader
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Iterate once through the DataLoader to get a sample batch
        try:
            sample_batch = next(iter(dataloader))
            
            # Print the shape of the sample batch tensor
            print(f"Sample batch shape: {sample_batch.shape}")

            # Print the min and max pixel values of the sample batch to verify normalization
            print(f"Sample batch min pixel value: {sample_batch.min()}")
            print(f"Sample batch max pixel value: {sample_batch.max()}")
        except StopIteration:
            print("DataLoader is empty. This might happen if batch_size is larger than dataset size and drop_last=True (not the case here), or if the dataset is truly empty after filtering errors.")
        except Exception as e:
            print(f"Error during DataLoader iteration or processing: {e}")

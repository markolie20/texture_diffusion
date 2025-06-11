import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_loader import MinecraftTextureDataset
from diffusion_process import DiffusionModel
from unet import UNet

def train_diffusion_model(data_dir, img_size=(16, 16), batch_size=64, timesteps=1000, epochs=10, learning_rate=1e-4, device=None):
    # Set device
    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transform pipeline
    transform_pipeline = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset and dataloader
    dataset = MinecraftTextureDataset(root_dir=data_dir, transform=transform_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the UNet model
    unet_model = UNet(img_channels=3, time_emb_dim=32).to(device)

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(unet_model, timesteps=timesteps, device=device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(diffusion_model.unet_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            # Sample random timesteps
            t = torch.randint(0, timesteps, (images.size(0),), device=device).long()

            # Forward diffusion process
            noised_images = diffusion_model.q_sample(images, t)

            # Predict noise using the model
            predicted_noise = diffusion_model.unet_model(noised_images, t)

            # Calculate loss
            loss = criterion(predicted_noise, images)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    print("Training complete.")

if __name__ == '__main__':
    DATA_DIR = 'assets/items/'  # Update this path as necessary
    train_diffusion_model(data_dir=DATA_DIR)
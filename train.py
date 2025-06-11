import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms 
from tqdm import tqdm

from unet import UNet #
from diffusion_process import DiffusionModel 
from dataset_loader import MinecraftTextureDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4  
BATCH_SIZE = 64      
IMAGE_SIZE = 16
IMG_CHANNELS = 4
TIME_EMB_DIM = 32    

NUM_EPOCHS = 25     
TIMESTEPS_TRAIN = 1000 

MODEL_SAVE_DIR = 'models'

SAMPLES_DIR = 'training_samples' 
SAVE_SAMPLES_EPOCH_INTERVAL = 5

def train(data_dir, model_name, model_save_path, samples_subdir):
    print(f"Starting training: {model_name}")
    print(f"Device: {DEVICE}, Epochs: {NUM_EPOCHS}, Diffusion Timesteps: {TIMESTEPS_TRAIN}")
    print(f"Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    samples_dir = os.path.join(SAMPLES_DIR, samples_subdir)
    os.makedirs(samples_dir, exist_ok=True)

    transform_pipeline = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    ])
    
    dataset = MinecraftTextureDataset(root_dir=data_dir, transform=transform_pipeline)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}. Exiting training.")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True if DEVICE == "cuda" else False)

    unet_model = UNet(img_channels=IMG_CHANNELS, time_emb_dim=TIME_EMB_DIM).to(DEVICE)
    diffusion = DiffusionModel(unet_model, timesteps=TIMESTEPS_TRAIN, device=DEVICE)
    optimizer = optim.AdamW(unet_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 

    print(f"Model: {model_name} on {DEVICE} for {NUM_EPOCHS} epochs, {TIMESTEPS_TRAIN} diffusion steps.")

    for epoch in range(NUM_EPOCHS):
        unet_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        
        for i, batch_images in enumerate(progress_bar):
            batch_images = batch_images.to(DEVICE)
            
            t = torch.randint(0, diffusion.timesteps, (batch_images.shape[0],), device=DEVICE).long()
            noise_gt = torch.randn_like(batch_images, device=DEVICE) 
            noisy_images = diffusion.q_sample(x_start=batch_images, t=t, noise=noise_gt)
            
            predicted_noise = unet_model(noisy_images, t)
            
            loss = criterion(noise_gt, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Avg Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % SAVE_SAMPLES_EPOCH_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            unet_model.eval()
            with torch.no_grad():
                print(f"Generating samples for epoch {epoch+1}...")
                sampled_images = diffusion.sample(image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=4, channels=IMG_CHANNELS)
                sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2 
                save_image(sampled_images, os.path.join(samples_dir, f"epoch_{epoch+1}_samples_T{TIMESTEPS_TRAIN}.png"))
            print(f"Samples saved for epoch {epoch+1}.")

    torch.save(unet_model.state_dict(), model_save_path)
    print(f"Training complete. Final model saved to {model_save_path}")

if __name__ == '__main__':
    BLOCKS_DATA_DIR = 'assets/blocks/' 
    ITEMS_DATA_DIR = 'assets/items/' 
    
    BLOCKS_MODEL_NAME = f"minecraft_blocks_T{TIMESTEPS_TRAIN}_E{NUM_EPOCHS}.pth"
    ITEMS_MODEL_NAME = f"minecraft_items_T{TIMESTEPS_TRAIN}_E{NUM_EPOCHS}.pth"

    MODEL_SAVE_DIR = 'models'
    BLOCKS_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, BLOCKS_MODEL_NAME)
    ITEMS_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, ITEMS_MODEL_NAME)
    
    train(BLOCKS_DATA_DIR, BLOCKS_MODEL_NAME, BLOCKS_MODEL_SAVE_PATH,samples_subdir="blocks")
    train(ITEMS_DATA_DIR, ITEMS_MODEL_NAME, ITEMS_MODEL_SAVE_PATH, samples_subdir="items")
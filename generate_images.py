import os
import torch
from torchvision.utils import save_image
from unet import UNet 
from diffusion_process import DiffusionModel

def load_unet_model(model_path, device, img_channels=4, time_emb_dim=32):
    model = UNet(img_channels=img_channels, time_emb_dim=time_emb_dim).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please check the path.")
        print("You need to train a model first or provide the correct path to an existing model.")
        exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in UNet class matches the saved model.")
        exit()
    model.eval()
    return model

def generate_images(model_path, output_dir='generated', num_images=8, img_size=(16, 16), 
                    timesteps=1000,
                    channels=4, device=None):
    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for generation.")
    os.makedirs(output_dir, exist_ok=True)

    unet = load_unet_model(model_path, device, img_channels=channels)
    
    diffusion = DiffusionModel(unet, timesteps=timesteps, device=device)
    print(f"Initializing DiffusionModel for generation with {timesteps} timesteps.")

    print(f"Generating {num_images} images of size {img_size} with {channels} channels...")
    unet.eval()
    imgs = diffusion.sample(image_size=img_size, batch_size=num_images, channels=channels)
    
    imgs = (imgs.clamp(-1, 1) + 1) / 2 

    for i in range(num_images):
        save_image(imgs[i], os.path.join(output_dir, f"generated_{i+1}.png"))

    print(f"Generated {num_images} images in '{output_dir}'.")

if __name__ == "__main__":
    BLOCKS_MODEL_PATH = "models/minecraft_blocks_T1000_E25.pth"  
    ITEMS_MODEL_PATH = "models/minecraft_items_T1000_E25.pth"
    
    BLOCKS_OUTPUT_DIR = "generated_blocks"
    ITEMS_OUTPUT_DIR = "generated_items"
    
    NUM_IMAGES = 64
    IMAGE_SIZE_HW = (16, 16) 
    IMAGE_CHANNELS = 4       
    GENERATION_TIMESTEPS = 1000 

    generate_images(BLOCKS_MODEL_PATH, 
                    output_dir=BLOCKS_OUTPUT_DIR, 
                    num_images=NUM_IMAGES, 
                    img_size=IMAGE_SIZE_HW, 
                    timesteps=GENERATION_TIMESTEPS,
                    channels=IMAGE_CHANNELS)
    
    generate_images(ITEMS_MODEL_PATH, 
                    output_dir=ITEMS_OUTPUT_DIR, 
                    num_images=NUM_IMAGES, 
                    img_size=IMAGE_SIZE_HW, 
                    timesteps=GENERATION_TIMESTEPS,
                    channels=IMAGE_CHANNELS)

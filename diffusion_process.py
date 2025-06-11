import torch
import torch.nn.functional as F
from tqdm import tqdm # Optional, for progress visualization

# Assuming unet.py is in the same directory and contains the UNet class
from unet import UNet 

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionModel:
    def __init__(self, unet_model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None):
        self.unet_model = unet_model
        self.timesteps = timesteps
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(self.device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        
        # For q_sample, also referred to as sqrt_recip_alphas_cumprod in some notations for p_sample
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(self.device)
        
        # Ensure unet_model is also on the correct device
        self.unet_model.to(self.device)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        # .gather() expects t to be of the same number of dimensions as a for the dim to gather along.
        # Since a is 1D (schedule) and t is 1D (batch_size), this should be fine.
        out = a.gather(-1, t) 
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, t_index): # t_index is the integer value of the timestep
        # Ensure model is in evaluation mode if it has layers like Dropout/BatchNorm
        # self.unet_model.eval() # Typically done outside the sampling loop if it's a one-off sample call
        
        with torch.no_grad(): # No gradients needed for sampling
            predicted_noise = self.unet_model(x_t, t) # t is already (batch_size,) tensor

        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Using self.alphas directly
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)
        
        model_mean = sqrt_recip_alphas_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0: # Last step
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t, device=self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
            
    def sample(self, image_size, batch_size=16, channels=3):
        img_h, img_w = image_size
        img = torch.randn((batch_size, channels, img_h, img_w), device=self.device)
        # imgs = [] # Optional: to store intermediate images

        # self.unet_model.eval() # Set model to eval mode before starting the loop
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop', total=self.timesteps, leave=False):
            # Current timestep 'i' as an integer
            # Create t tensor for the current step
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i) # Pass integer 'i' as t_index
            # if i % 50 == 0: # Optional: store some intermediate steps
            #     imgs.append(img.cpu())
        
        # self.unet_model.train() # Return model to train mode if needed afterwards
        return img #, imgs # Return final or all images

if __name__ == '__main__':
    IMG_SIZE = 16 # Keep small for quick testing
    BATCH_SIZE = 4 # Keep small for quick testing
    CHANNELS = 3
    TIMESTEPS = 100 # Use fewer timesteps for faster testing of the sampling loop

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Instantiate U-Net model
    unet_model = UNet(img_channels=CHANNELS, time_emb_dim=32).to(device) # time_emb_dim from unet.py default

    # Instantiate DiffusionModel
    diffusion = DiffusionModel(unet_model, timesteps=TIMESTEPS, device=device)
    print(f"DiffusionModel instantiated with {TIMESTEPS} timesteps.")

    # Test q_sample (forward diffusion)
    print("\nTesting q_sample (forward diffusion)...")
    dummy_x_start = torch.randn(BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    dummy_t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=device).long() # Ensure t is within new TIMESTEPS
    
    noised_image = diffusion.q_sample(dummy_x_start, dummy_t)
    print(f"Original image shape: {dummy_x_start.shape}")
    print(f"Timestep tensor shape: {dummy_t.shape}")
    print(f"Noised image shape: {noised_image.shape}")
    assert noised_image.shape == dummy_x_start.shape
    print("q_sample test passed.\n")

    # Test sample (generation process)
    # This will be slow if TIMESTEPS is large, but with 100 it should be manageable.
    print("Testing sample (generation process)...")
    print(f"Generating {BATCH_SIZE} images of size ({IMG_SIZE}, {IMG_SIZE})...")
    
    # Set model to eval mode for sampling if it has dropout etc.
    # For this UNet, it might not be strictly necessary if no dropout/batchnorm, but good practice.
    unet_model.eval() 
    generated_images = diffusion.sample(image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, channels=CHANNELS)
    # unet_model.train() # Set back to train mode if you were to continue training

    print(f"Generated images shape: {generated_images.shape}")
    assert generated_images.shape == (BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE)
    print("sample test passed.\n")

    print("All tests in diffusion_process.py completed successfully!")

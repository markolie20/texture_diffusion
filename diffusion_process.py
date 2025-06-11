import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionModel:
    def __init__(self, unet_model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None): # Default was 1000
        self.unet_model = unet_model
        self.timesteps = timesteps
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(self.device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(self.device)
        
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(self.device)
        
        self.unet_model.to(self.device)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t) 
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, t_index):
        with torch.no_grad():
            predicted_noise = self.unet_model(x_t, t)

        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)
        
        model_mean = sqrt_recip_alphas_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t, device=self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * noise # Add small noise
            
    def sample(self, image_size, batch_size=16, channels=4):
        img_h, img_w = image_size
        img = torch.randn((batch_size, channels, img_h, img_w), device=self.device)
        
        self.unet_model.eval() # Ensure model is in eval mode for sampling
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop', total=self.timesteps, leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        return img

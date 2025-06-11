import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # For odd dimensions, append a zero
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU() # Swish activation

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_channels),
                nn.SiLU()
            )
        
        # Residual connection if input and output channels are the same
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()


    def forward(self, x, time_emb=None):
        h = self.act1(self.norm1(self.conv1(x)))

        if self.time_mlp is not None and time_emb is not None:
            time_encoding = self.time_mlp(time_emb)
            # Reshape time_encoding to (batch_size, out_channels, 1, 1) to be added to h
            h = h + time_encoding.unsqueeze(-1).unsqueeze(-1)
            
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, img_channels=3, down_channels=(64, 128, 256, 512), up_channels=(512, 256, 128, 64), time_emb_dim=32):
        super().__init__()

        self.time_mlp = SinusoidalPositionEmbeddings(time_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(img_channels, down_channels[0], kernel_size=3, padding=1)

        # Encoder Path
        self.downs = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(len(down_channels) -1):
            self.downs.append(Block(down_channels[i], down_channels[i+1], time_emb_dim))
            self.downsamplers.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.mid_block1 = Block(down_channels[-1], down_channels[-1], time_emb_dim)
        self.mid_block2 = Block(down_channels[-1], down_channels[-1], time_emb_dim)

        # Decoder Path
        self.ups = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        # Reverse down_channels to define up_channels correctly for skip connections
        reversed_down_channels = down_channels[::-1] # e.g. (512, 256, 128, 64)

        for i in range(len(up_channels)):
            prev_up_channel = down_channels[-1] if i == 0 else up_channels[i-1] # Output from bottleneck or previous up-block
            
            # The channel size of the skip connection corresponding to this upsampling level
            # skip_connections after reversing are [d2_out, d1_out, d0_out, init_out]
            # These have channels from reversed_down_channels:
            # reversed_down_channels[0] = 512 (for i=0)
            # reversed_down_channels[1] = 256 (for i=1)
            # ...
            actual_skip_channel_size = reversed_down_channels[i]

            self.upsamplers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            # The input to this block is prev_up_channel (from upsampled x) + actual_skip_channel_size (from skip connection)
            # The output of this block is up_channels[i]
            self.ups.append(Block(prev_up_channel + actual_skip_channel_size, up_channels[i], time_emb_dim))


        # Final output layer
        self.final_conv = nn.Conv2d(up_channels[-1], img_channels, kernel_size=1)

    def forward(self, x, timestep):
        t_emb = self.time_mlp(timestep)

        # Encoder Path
        x = self.init_conv(x)
        skip_connections = [x] # Store initial conv output as first "skip"

        for i in range(len(self.downs)):
            x = self.downs[i](x, t_emb)
            skip_connections.append(x)
            x = self.downsamplers[i](x)
        
        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        # Decoder Path
        # Reverse skip connections for easier pop
        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.upsamplers[i](x)
            # The skip connection should correspond to the resolution before downsampling.
            # self.downs has N elements, self.downsamplers has N elements
            # skip_connections has N+1 elements (init_conv + N down blocks)
            # e.g., down_channels=(64, 128, 256, 512) -> len=4
            # self.downs has 3 elements: (64->128), (128->256), (256->512)
            # skip_connections: init_conv_out (64), down0_out (128), down1_out (256), down2_out (512)
            # reversed_skip_connections: down2_out (512), down1_out (256), down0_out (128), init_conv_out (64)
            
            # up_channels=(512, 256, 128, 64) -> len=4
            # up0: takes bottleneck (512), skip is down2_out (512). Concat = 1024. Block out 512.
            # up1: takes up0_out (512), skip is down1_out (256). Concat = 768. Block out 256.
            # up2: takes up1_out (256), skip is down0_out (128). Concat = 384. Block out 128.
            # up3: takes up2_out (128), skip is init_conv_out (64). Concat = 192. Block out 64.

            skip = skip_connections[i] # First skip is the deepest one from encoder
            
            # Ensure spatial dimensions match for concatenation
            if x.shape[-2:] != skip.shape[-2:]: # Check H, W
                # This can happen if MaxPool2d rounds down and Upsample rounds up differently.
                # A common fix is to use ConvTranspose2d for upsampling or ensure padding/stride in MaxPool2d
                # For now, let's try padding x to match skip. This is a simplistic fix.
                # A better solution might be to use specific output_padding in ConvTranspose2d or ensure sizes align.
                target_size = skip.shape[-2:]
                x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

            x = torch.cat((x, skip), dim=1) # Concatenate along channel dimension
            x = self.ups[i](x, t_emb)
            
        return self.final_conv(x)


if __name__ == '__main__':
    # Test SinusoidalPositionEmbeddings
    print("Testing SinusoidalPositionEmbeddings...")
    time_emb_dim = 32
    dummy_time_input = torch.randint(0, 1000, (4,)).float() # batch_size = 4
    pos_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
    time_embeddings = pos_embedder(dummy_time_input)
    print(f"Input time shape: {dummy_time_input.shape}")
    print(f"Output time embeddings shape: {time_embeddings.shape}")
    assert time_embeddings.shape == (4, time_emb_dim)
    print("SinusoidalPositionEmbeddings test passed.\n")

    # Test Block
    print("Testing Block...")
    dummy_block_input = torch.randn(4, 64, 16, 16) # batch, channels, H, W
    block_no_time = Block(64, 128)
    block_output_no_time = block_no_time(dummy_block_input)
    print(f"Block output (no time) shape: {block_output_no_time.shape}")
    assert block_output_no_time.shape == (4, 128, 16, 16)

    block_with_time = Block(64, 128, time_emb_dim=time_emb_dim)
    block_output_with_time = block_with_time(dummy_block_input, time_embeddings)
    print(f"Block output (with time) shape: {block_output_with_time.shape}")
    assert block_output_with_time.shape == (4, 128, 16, 16)
    print("Block test passed.\n")

    # Test UNet
    print("Testing UNet...")
    img_channels = 3
    model = UNet(img_channels=img_channels, time_emb_dim=time_emb_dim)
    
    batch_size = 4
    img_size = 16 # Assuming 16x16 images for this test based on dataset_loader
    dummy_x = torch.randn(batch_size, img_channels, img_size, img_size)
    dummy_t = torch.randint(0, 1000, (batch_size,)).float()
    
    output = model(dummy_x, dummy_t)
    
    print(f"Input image tensor shape: {dummy_x.shape}")
    print(f"Input timestep tensor shape: {dummy_t.shape}")
    print(f"Output tensor shape: {output.shape}")
    
    assert output.shape == (batch_size, img_channels, img_size, img_size), \
        f"Output shape {output.shape} does not match input shape {(batch_size, img_channels, img_size, img_size)}"
    print("UNet test passed: Output shape matches input image shape.")

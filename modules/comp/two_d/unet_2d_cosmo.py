import numpy as np
import torch
import torch.nn as nn
import math

## --------------------------------------------------------------------------------------------------------
## --------------------------------GibbsDIFF-2D (Cosmology Adapted)--------------------------------------
## --------------------------------------------------------------------------------------------------------

class BlockGDiff2D(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.2):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, dim_out)  # Using 1 group for instance-like normalization
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) # Ensure dropout is appropriate for your dataset size

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ResnetBlockGDiff2D(nn.Module):
    """
    ResNet block with optional time and physical parameter embeddings.
    For cosmology, physical_param_emb_dim will be the output dim of PhiEmbeddingCosmo.
    """
    def __init__(self, dim, dim_out, time_emb_dim=None, physical_param_emb_dim=None, dropout=0.):
        super().__init__()

        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out) # Simpler: directly to dim_out for adding
            )

        self.phi_mlp = None # MLP for physical parameters
        if physical_param_emb_dim is not None:
            self.phi_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(physical_param_emb_dim, dim_out) # Simpler: directly to dim_out for adding
            )
            # Alternative from paper: MLP that outputs scale and shift for physical params
            # self.phi_mlp = nn.Sequential(
            #     nn.SiLU(),
            #     nn.Linear(physical_param_emb_dim, dim_out * 2)
            # )


        self.block1 = BlockGDiff2D(dim, dim_out, dropout=dropout)
        self.block2 = BlockGDiff2D(dim_out, dim_out, dropout=dropout) # Added dropout here too
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, phi_emb=None):
        h = self.block1(x)

        # Add time embedding (if provided)
        if self.time_mlp is not None and time_emb is not None:
            time_encoding = self.time_mlp(time_emb)
            # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
            h = h + time_encoding.unsqueeze(-1).unsqueeze(-1)

        # Add physical parameter embedding (if provided)
        if self.phi_mlp is not None and phi_emb is not None:
            phi_encoding = self.phi_mlp(phi_emb)
            # Example: Additive conditioning
            h = h + phi_encoding.unsqueeze(-1).unsqueeze(-1)
            # Alternative: FiLM-like conditioning (scale and shift)
            # if self.phi_mlp outputs dim_out * 2:
            #    scale_phi, shift_phi = phi_encoding.chunk(2, dim=1)
            #    h = h * (1 + scale_phi.unsqueeze(-1).unsqueeze(-1)) + shift_phi.unsqueeze(-1).unsqueeze(-1)


        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention2D(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # q, k, v each with hidden_dim channels
        
        q, k, v = map(lambda t: t.reshape(b, self.heads, dim_head, h * w), qkv)

        q = q * self.scale
        sim = torch.einsum('b h d n, b h d m -> b h n m', q, k) # Corrected einsum
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h n m, b h d m -> b h d n', attn, v) # Corrected einsum
        out = out.reshape(b, hidden_dim, h, w)
        return self.to_out(out)

class SinusoidalTimeEmbedding(nn.Module): # Renamed for clarity
    """ Sinusoidal time embedding """
    def __init__(self, dim): # dim is the output embedding dimension
        super().__init__()
        self.dim = dim

    def forward(self, t): # t is a 1D tensor of timesteps
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :] # (batch_size, 1) * (1, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (batch_size, dim)
        return embeddings

class PhiEmbeddingCosmo(nn.Module): # Renamed for clarity and specific use
    """ Embedding for cosmological parameters Phi_CMB = (H0, ombh2) """
    def __init__(self, input_phi_dim=2, hidden_dim=128, output_emb_dim=512):
        super().__init__()
        # As per paper: "linear layer followed by an activation function"
        # And "The time and parameter embeddings are then transformed by a small MLP of depth 1
        # and added to the input of their respective ResBlocks."
        # This PhiEmbeddingCosmo will be the "linear layer + activation" part.
        # The "small MLP of depth 1" will be part of the ResnetBlockGDiff2DCosmo.
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_phi_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_emb_dim) # Output a fixed embedding size
            # No SiLU after the last linear layer if this embedding is directly added or processed by another MLP.
        )

    def forward(self, phi_cmb_params): # phi_cmb_params is [B, 3]
        return self.embedding_layer(phi_cmb_params)


class FinalConvLayer2D(nn.Module): # Renamed for clarity
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet2DGDiff_cosmo(nn.Module): # Renamed class
    def __init__(
        self,
        dim,                # Base dimension for channels
        img_channels=1,       # Number of channels in the input image (e.g., 1 for grayscale dust)
        time_embed_dim=256, # Dimension for sinusoidal time embedding
        phi_input_dim=3,    # Input dimension for Phi_CMB (sigma, H0, ombh2)
        phi_embed_dim=256,  # Output dimension of the PhiEmbeddingCosmo layer
        dropout=0.1,        # Dropout rate
        attn_dim_head=32,
        attn_heads=4,
        device=None
    ):
        super().__init__()

        self.dim = dim
        self.img_channels = img_channels
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Channel dimensions for U-Net stages
        self.dims = [dim, dim * 2, dim * 4, dim * 8] # Example: 64, 128, 256, 512
        # If you need deeper, add dim * 16. Current code uses up to dim * 8 then bottleneck
        # Let's match your original channel progression nc_1 to nc_5
        # nc_1=dim, nc_2=dim*2, nc_3=dim*4, nc_4=dim*8
        # Bottleneck uses nc_4 (dim*8), then up_blocks reduce channels.
        # Paper says "bottleneck of size 32x32 or 16x16" - this refers to spatial size.
        # "Each ResBlock: 3 convolutions, GroupNorm, SiLU..." - your BlockGDiff2D has 1 conv.
        # ResnetBlockGDiff2D has 2 BlockGDiff2D.
        
        # We'll use phi_embed_dim as the dimension of the output from PhiEmbeddingCosmo,
        # and this will be fed into the ResNet blocks.
        # The ResNet blocks will have their own MLPs to process these embeddings.

        # Initial convolution
        self.init_conv = BlockGDiff2D(img_channels, self.dims[0], dropout=dropout) # from img_channels to dim

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(dim=time_embed_dim)

        # Phi_CMB (cosmological parameters) embedding
        self.phi_embedding = PhiEmbeddingCosmo(
            input_phi_dim=phi_input_dim,
            hidden_dim=phi_embed_dim, # intermediate hidden dim for phi MLP
            output_emb_dim=phi_embed_dim # Final embedding size for phi
        )
        
        # --- Downsampling Path ---
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList() # Store attention separately for clarity
        current_channels = self.dims[0]
        for i, out_channels in enumerate(self.dims[1:]): # dim*2, dim*4, dim*8
            self.down_blocks.append(
                ResnetBlockGDiff2D(
                    current_channels, out_channels,
                    time_emb_dim=time_embed_dim,
                    physical_param_emb_dim=phi_embed_dim, # Pass the output dim of PhiEmbeddingCosmo
                    dropout=dropout
                )
            )
            # Add attention at deeper layers (e.g., last two down_blocks)
            if i >= len(self.dims) - 3: # e.g. for dim*4 and dim*8 stages
                 self.down_attns.append(SelfAttention2D(out_channels, dim_head=attn_dim_head, heads=attn_heads))
            else:
                 self.down_attns.append(nn.Identity()) # No attention
            current_channels = out_channels
            # Add Downsampling layer (e.g., Conv2d stride 2 or MaxPool2d)
            if i < len(self.dims) - 2: # Don't downsample after the last down_block before bottleneck
                self.down_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)) # Downsample

        # Bottleneck
        # The last current_channels is dims[-1] (e.g., dim*8)
        bottleneck_channels = current_channels # Let's keep it same as last down stage for now
        self.bottleneck_res1 = ResnetBlockGDiff2D(
            current_channels, bottleneck_channels,
            time_emb_dim=time_embed_dim,
            physical_param_emb_dim=phi_embed_dim,
            dropout=dropout
        )
        self.bottleneck_attn = SelfAttention2D(bottleneck_channels, dim_head=attn_dim_head, heads=attn_heads)
        self.bottleneck_res2 = ResnetBlockGDiff2D(
            bottleneck_channels, bottleneck_channels, # Stays bottleneck_channels
            time_emb_dim=time_embed_dim,
            physical_param_emb_dim=phi_embed_dim,
            dropout=dropout
        )
        current_channels = bottleneck_channels

        # --- Upsampling Path ---
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        # Iterate in reverse over dims, excluding the first one (init_conv output)
        # And excluding the last one (bottleneck input)
        reversed_skip_dims = self.dims[:-1][::-1] # e.g. [dim*4, dim*2, dim]

        for i, skip_channels in enumerate(reversed_skip_dims): # skip_channels are dim*4, dim*2, dim
            # Add Upsampling layer (e.g., ConvTranspose2d)
            # The output channels of upsampling should match skip_channels to allow ResNetBlock input
            self.up_blocks.append(nn.ConvTranspose2d(current_channels, skip_channels, kernel_size=4, stride=2, padding=1)) # Upsample
            
            self.up_blocks.append(
                ResnetBlockGDiff2D(
                    skip_channels * 2, skip_channels, # Input is skip_channels from upsample + skip_channels from skip connection
                    time_emb_dim=time_embed_dim,
                    physical_param_emb_dim=phi_embed_dim,
                    dropout=dropout
                )
            )
            if i < 2 : # Add attention at deeper up_blocks (mirroring down_blocks)
                self.up_attns.append(SelfAttention2D(skip_channels, dim_head=attn_dim_head, heads=attn_heads))
            else:
                self.up_attns.append(nn.Identity())
            current_channels = skip_channels
            
        # Final convolution
        self.final_conv = FinalConvLayer2D(self.dims[0], img_channels) # from dim to img_channels

    def forward(self, x, t, phi_cmb=None): # t is [B], phi_cmb is [B, 3]
        # Input checks for phi_cmb
        if phi_cmb is None:
            # Default to zeros if not provided (e.g., for unconditional training if ever needed)
            phi_cmb = torch.zeros(x.shape[0], 3, device=x.device, dtype=x.dtype)
        elif not isinstance(phi_cmb, torch.Tensor):
            phi_cmb = torch.tensor(phi_cmb, device=x.device, dtype=x.dtype)
        
        if phi_cmb.ndim == 1: # If a single [sigma, H0, wb] is passed for the batch
            phi_cmb = phi_cmb.unsqueeze(0).repeat(x.shape[0], 1)
        elif phi_cmb.shape[0] != x.shape[0] or phi_cmb.shape[1] != 3:
            raise ValueError(f"phi_cmb shape must be [batch_size, 3] or [3] for broadcasting. Got {phi_cmb.shape}")

        # 1. Initial Convolution
        x = self.init_conv(x) # (B, dim, H, W)
        
        # 2. Embeddings
        time_e = self.time_embedding(t)       # (B, time_embed_dim)
        phi_e = self.phi_embedding(phi_cmb)   # (B, phi_embed_dim)

        # 3. Downsampling Path
        skip_connections = []
        block_idx = 0 # For ResNet and Attention
        downsample_idx = 0 # For Conv2d downsampling layers

        num_resnet_stages_down = len(self.dims) -1 # 3 stages for dims = [d, 2d, 4d, 8d]

        # x is (B, dims[0], H, W)
        for i in range(num_resnet_stages_down): # 0, 1, 2
            # ResNetBlock + Attention
            res_block = self.down_blocks[block_idx]
            attn_block = self.down_attns[block_idx]
            x = res_block(x, time_e, phi_e)
            x = attn_block(x)
            skip_connections.append(x)
            block_idx += 1
            
            # Downsampling Conv (if not the last stage before bottleneck)
            if i < num_resnet_stages_down - 1:
                downsampler = self.down_blocks[block_idx] # This is the Conv2d for downsampling
                x = downsampler(x)
                block_idx +=1


        # 4. Bottleneck
        x = self.bottleneck_res1(x, time_e, phi_e)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_res2(x, time_e, phi_e)
        # x is now (B, bottleneck_channels, H_bottleneck, W_bottleneck)

        # 5. Upsampling Path
        block_idx = 0 # For Upsample ConvT and ResNet
        attn_idx = 0
        skip_connections = skip_connections[::-1] # Reverse for easy popping

        num_resnet_stages_up = len(self.dims) -1

        for i in range(num_resnet_stages_up): # 0, 1, 2
            upsampler = self.up_blocks[block_idx]
            x = upsampler(x) # Upsample (e.g. ConvTranspose2d)
            block_idx +=1

            skip = skip_connections[i]
            x = torch.cat((x, skip), dim=1) # Concatenate with skip connection

            res_block = self.up_blocks[block_idx]
            attn_block = self.up_attns[attn_idx]

            x = res_block(x, time_e, phi_e)
            x = attn_block(x)
            
            block_idx +=1
            attn_idx +=1

        # 6. Final Convolution
        out = self.final_conv(x)
        return out

    @torch.no_grad()
    def summary(self, x_shape, t_shape, phi_shape): # Added phi_shape
        # (Summary code from your original, adapted for phi_shape)
        x = torch.zeros(x_shape).to(self.device)
        t = torch.zeros(t_shape, dtype=torch.float32).to(self.device) # t is often float for continuous time
        phi = torch.zeros(phi_shape).to(self.device)

        layer_info = []
        def get_layer_info(name):
            def hook(module, input, output):
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                output_shape = None
                if isinstance(output, torch.Tensor): output_shape = output.shape
                elif isinstance(output, tuple) and output: output_shape = output[0].shape
                
                layer_info.append({
                    'name': name, 'type': module.__class__.__name__,
                    'output_shape': tuple(output_shape) if output_shape else None, 'params': params
                })
            return hook
        
        hooks = []
        # Simplified hook registration: iterate through direct children for top-level summary
        for name, module in self.named_children():
            hooks.append(module.register_forward_hook(get_layer_info(name)))
        
        try:
            _ = self.forward(x, t, phi) # Run forward pass
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print("_" * 100)
            print(f"Model: {self.__class__.__name__}")
            print("=" * 100)
            print(f"{'Layer (type)':<40}{'Output Shape':<30}{'Param #':<15}")
            print("=" * 100)
            print(f"{'input_x (InputLayer)':<40}{str(x_shape):<30}{'0':<15}")
            print(f"{'input_t (InputLayer)':<40}{str(t_shape):<30}{'0':<15}")
            print(f"{'input_phi (InputLayer)':<40}{str(phi_shape):<30}{'0':<15}")
            
            current_x_shape = x_shape # To track shape through network for summary (simplified)
            for layer in layer_info:
                print(f"{layer['name']} ({layer['type']})".ljust(40) +
                      f"{str(layer['output_shape'])}".ljust(30) +
                      f"{layer['params']:,}".ljust(15))
                if layer['output_shape'] and len(layer['output_shape']) == 4: # Assume Conv2D like layer
                    current_x_shape = layer['output_shape']


            print("=" * 100)
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {total_params:,}") # Assuming all are trainable
            print(f"Non-trainable params: {0:,}")
            print("_" * 100)
            
        finally:
            for hook in hooks: hook.remove()

# --- Example Usage ---
if __name__ == '__main__':
    print('Running Unet2DCosmoGDiff example...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parameters for the U-Net
    base_dim = 64       # Base number of channels
    img_channels = 1    # Input image channels (e.g., 1 for grayscale dust map)
    time_emb_dim = 256  # Dimension for time embeddings
    phi_input_dim = 3   # For (sigma_CMB, H0, ombh2)
    phi_emb_output_dim = 256 # Output dimension of the PhiEmbeddingCosmo layer

    # Create the U-Net model
    model = Unet2DCosmoGDiff(
        dim=base_dim,
        img_channels=img_channels,
        time_embed_dim=time_emb_dim,
        phi_input_dim=phi_input_dim,
        phi_embed_dim=phi_emb_output_dim, # Pass the output dim for phi MLP in ResBlocks
        dropout=0.1,
        attn_dim_head=32, # Default from your code
        attn_heads=4,     # Default from your code
        device=device
    ).to(device)

    # --- Test with dummy data ---
    batch_size = 2
    img_size = 256 # Assuming 256x256 maps as per cosmology application

    dummy_x = torch.randn(batch_size, img_channels, img_size, img_size).to(device)
    dummy_t = torch.randint(0, 1000, (batch_size,)).float().to(device) # Timesteps
    
    # Dummy Phi_CMB parameters: (sigma_CMB, H0, ombh2)
    # Example: sigma_CMB ~ U(0.1, 1.2), H0 ~ U(50,90), ombh2 ~ U(0.0075, 0.0567)
    dummy_sigma_cmb = torch.rand(batch_size, 1) * (1.2 - 0.1) + 0.1
    dummy_h0 = torch.rand(batch_size, 1) * (90 - 50) + 50
    dummy_ombh2 = torch.rand(batch_size, 1) * (0.0567 - 0.0075) + 0.0075
    dummy_phi_cmb = torch.cat([dummy_h0, dummy_ombh2], dim=1).to(device)

    print(f"\nInput x shape: {dummy_x.shape}")
    print(f"Input t shape: {dummy_t.shape}")
    print(f"Input phi_cmb shape: {dummy_phi_cmb.shape}")

    # Forward pass
    try:
        output = model(dummy_x, dummy_t, dummy_phi_cmb)
        print(f"Output shape: {output.shape}")
        assert output.shape == dummy_x.shape
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Print model summary
    print("\n--- Model Summary ---")
    model.summary(
        x_shape=(batch_size, img_channels, img_size, img_size),
        t_shape=(batch_size,),
        phi_shape=(batch_size, phi_input_dim)
    )
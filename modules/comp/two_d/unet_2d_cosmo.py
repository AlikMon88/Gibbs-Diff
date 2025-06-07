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
        self.hidden_dim = dim_head * heads
        self.dim_head = dim_head

        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # q, k, v each with hidden_dim channels
        
        q, k, v = map(lambda t: t.reshape(b, self.heads, self.dim_head, h * w), qkv)

        q = q * self.scale
        sim = torch.einsum('b h d n, b h d m -> b h n m', q, k) # Corrected einsum
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h n m, b h d m -> b h d n', attn, v) # Corrected einsum
        out = out.reshape(b, self.hidden_dim, h, w)
        return self.to_out(out)

class SinusoidalTimeEmbedding(nn.Module): # Renamed for clarity
    """ Sinusoidal time embedding """
    def __init__(self, channel): # t-indexing along depth/channel dimension
        super().__init__()
        self.channel = channel

    def forward(self, t): # t is a 1D tensor of timesteps
        device = t.device
        half_dim = self.channel // 2
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
            nn.Linear(hidden_dim, hidden_dim), # Output a fixed embedding size
            nn.SiLU(),
            nn.Linear(hidden_dim, output_emb_dim)
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
        channels=1,       # Number of channels in the input image (e.g., 1 for grayscale dust)
        phi_input_dim=2,    # Input dimension for Phi_CMB (H0, ombh2)
        dropout=0.1,        # Dropout rate
        attn_dim_head=32,
        attn_heads=4,
        device=None
    ):
        super().__init__()

        self.dim = dim
        self.img_channels = channels
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Channel dimensions for U-Net stages
        self.dims = [dim, dim * 2, dim * 4, dim * 8] # Example: 64, 128, 256, 512
        
        H = 32  # Typical spatial size for image features
        time_emb_dim = int(H)
        phi_embed_dim = int(H)

        nc_1, nc_2, nc_3, nc_4, nc_5 = dim, dim * 2, dim * 4, dim * 8, dim * 16

        # Initial convolution
        self.init_conv = BlockGDiff2D(channels, nc_1)

        # Down blocks
        self.down_blocks = nn.ModuleList([
            ResnetBlockGDiff2D(nc_1, nc_2),
            ResnetBlockGDiff2D(nc_2, nc_3),
            ResnetBlockGDiff2D(nc_3, nc_4),
            ResnetBlockGDiff2D(nc_4, nc_5)
        ])

        # Attention blocks
        self.attn_blocks = nn.ModuleList([
            None,
            None,
            SelfAttention2D(nc_4, heads=attn_heads, dim_head=attn_dim_head),
            SelfAttention2D(nc_5, heads=attn_heads, dim_head=attn_dim_head)
        ])

        # Embeddings
        self.time_embeddings = nn.ModuleList([
            SinusoidalTimeEmbedding(nc_2),
            SinusoidalTimeEmbedding(nc_3),
            SinusoidalTimeEmbedding(nc_4),
            SinusoidalTimeEmbedding(nc_5)
        ])

        self.alpha_embeddings = nn.ModuleList([
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_2),
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_3),
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_4),
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_5)
        ])

        # Bottleneck
        self.bottleneck = BlockGDiff2D(nc_5, nc_5)
        self.att_bottleneck = SelfAttention2D(nc_5, heads=attn_heads, dim_head=attn_dim_head)
        self.bottl_down = BlockGDiff2D(nc_5, nc_4)

        # Up blocks
        self.up_blocks = nn.ModuleList([
            ResnetBlockGDiff2D(nc_5 + nc_4, nc_4),
            ResnetBlockGDiff2D(nc_4 + nc_4, nc_3),
            ResnetBlockGDiff2D(nc_3 + nc_3, nc_2),
            ResnetBlockGDiff2D(nc_2 + nc_2, nc_1)
        ])

        self.time_embeddings_up = nn.ModuleList([
            SinusoidalTimeEmbedding(nc_4),
            SinusoidalTimeEmbedding(nc_3),
            SinusoidalTimeEmbedding(nc_2),
            SinusoidalTimeEmbedding(nc_1)
        ])

        self.alpha_embeddings_up = nn.ModuleList([
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_4),
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_3),
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_2),
            PhiEmbeddingCosmo(phi_input_dim, hidden_dim=phi_embed_dim, output_emb_dim=nc_1)
        ])

        self.outc = FinalConvLayer2D(nc_1, channels)

    def forward(self, x, t, phi_cmb=None): # t is [B], phi_cmb is [B, 2]
        # Input checks for phi_cmb
        if phi_cmb is None:
            # Default to zeros if not provided (e.g., for unconditional training if ever needed)
            phi_cmb = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)
        elif not isinstance(phi_cmb, torch.Tensor):
            phi_cmb = torch.tensor(phi_cmb, device=x.device, dtype=x.dtype)
        
        if phi_cmb.ndim == 1: # If a single [sigma, H0, wb] is passed for the batch
            phi_cmb = phi_cmb.unsqueeze(0).repeat(x.shape[0], 1)
        elif phi_cmb.shape[0] != x.shape[0] or phi_cmb.shape[1] != 2:
            raise ValueError(f"phi_cmb shape must be [batch_size, 2] or [2] for broadcasting. Got {phi_cmb.shape}")

        # Initial Convolution
        x = self.init_conv(x) # (B, dim, H, W)
        
        residuals = []

        # Down path
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            emb = self.time_embeddings[i](t) + self.alpha_embeddings[i](phi_cmb)
            x = x + emb.unsqueeze(-1).unsqueeze(-1)
            if self.attn_blocks[i] is not None:
                x = self.attn_blocks[i](x)
            residuals.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = x + self.time_embeddings[-1](t).unsqueeze(-1).unsqueeze(-1)
        x = self.att_bottleneck(x)
        x = self.bottl_down(x)
        
        # Up path
        for i, up in enumerate(self.up_blocks):
            res = residuals[-(i + 1)]
            x = torch.cat([x, res], dim=1)
            x = up(x)
            emb = self.time_embeddings_up[i](t) + self.alpha_embeddings_up[i](phi_cmb)
            x = x + emb.unsqueeze(-1).unsqueeze(-1)

        return self.outc(x)
    
    @torch.no_grad()
    def summary(self, x_shape, t_shape): ## phi gets traced automatically
        """
        Kears-style Summary of the Unet1D model architecture.
        
        Args:
            x_shape: Shape of input data tensor (batch_size, channels, seq_length)
            # t_shape: Shape of timestep tensor (batch_size,)
        """
        
        # Create example inputs
        x = torch.zeros(x_shape).to(self.device)
        t = torch.zeros(t_shape, dtype=torch.long).to(self.device)
        
        # Dictionary to store layer info
        layer_info = []
        
        # Hook to capture layer info
        def get_layer_info(name):
            def hook(module, input, output):
                # Get parameters
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # For modules that have multiple outputs (like tuples), get the first tensor shape
                if isinstance(output, tuple):
                    output_shape = output[0].shape if output else None
                else:
                    output_shape = output.shape

                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'output_shape': tuple(output_shape) if output_shape else None,
                    'params': params
                })
            return hook
        
        # Register hooks for each module
        hooks = []
        for name, module in self.named_modules():
            if name and '.' in name and not any(name.endswith(x) for x in ['.weight', '.bias']):
                hooks.append(module.register_forward_hook(get_layer_info(name)))
        
        # Run a forward pass
        try:
            with torch.no_grad():
                output = self.forward(x, t)
            
            # Calculate total params
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            trainable_params = total_params
            non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            
            # Print model summary in Keras style
            print("_" * 100)
            print("Model: Unet1D")
            print("=" * 100)
            print(f"{'Layer (type)':<40}{'Output Shape':<25}{'Param #':<15}")
            print("=" * 100)
            
            # Input layers
            print(f"{'input_1 (InputLayer)':<40}{str(x_shape):<25}{'0':<15}")
            print(f"{'input_2 (InputLayer)':<40}{str(t_shape):<25}{'0':<15}")
            
            # Display layer information
            for layer in layer_info:
                if layer['output_shape'] is not None:
                    print(f"{layer['name']} ({layer['type']})".ljust(40) + 
                        f"{str(layer['output_shape'])}".ljust(25) + 
                        f"{layer['params']}".ljust(15))
            
            print("=" * 100)
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_params:,}")
            print(f"Non-trainable params: {non_trainable_params:,}")
            print("_" * 100)
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

# --- Example Usage ---
if __name__ == '__main__':
    print('Running Unet2DCosmoGDiff example...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parameters for the U-Net
    base_dim = 64       # Base number of channels
    img_channels = 1    # Input image channels (e.g., 1 for grayscale dust map)
    time_emb_dim = 256  # Dimension for time embeddings
    phi_input_dim = 2   # For (sigma_CMB, H0, ombh2)
    phi_emb_output_dim = 256 # Output dimension of the PhiEmbeddingCosmo layer

    # Create the U-Net model
    model = Unet2DGDiff_cosmo(
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
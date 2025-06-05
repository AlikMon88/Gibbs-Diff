import numpy as np
import torch 
import torch.nn as nn
import math

from ...utils.helper import *
from einops import rearrange


## --------------------------------------------------------------------------------------------------------
## --------------------------------GibbsDIFF-1D---------------------------------------------------------------
## --------------------------------------------------------------------------------------------------------

# Phi Embedding Module
class PhiEmbedding(nn.Module):
    def __init__(self, alpha_dim, in_dim=100, out_dim=100):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(alpha_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, alpha):
        return self.mlp(alpha)

# Basic Convolutional Block 
## GDiff paper takes 2 conv-layers - this has 1
class BlockGDiff(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c n -> b n c')
        x = self.norm(x)
        x = rearrange(x, 'b n c -> b c n')
        x = self.act(x)
        return x

# ResNet Block
class ResnetBlockGDiff(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = BlockGDiff(dim, dim_out, dropout=dropout)
        self.block2 = BlockGDiff(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale, shift = time_emb.chunk(2, dim=1)
            scale_shift = (scale, shift)

        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)

# Time Embedding Module
class TimeEmbedding(nn.Module):
    def __init__(self, channels, embed_size):
        super().__init__()
        self.channels = channels
        self.embed_size = embed_size

    def forward(self, t):
        device = t.device
        half_dim = self.channels // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        t = t.unsqueeze(-1)  # [B, 1]
        sinusoid = t * inv_freq  # [B, D/2]
        pos_enc = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)  # [B, D]
        return pos_enc.unsqueeze(-1)  # [B, D, 1]

# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class final_regress(nn.Module):
    def __init__(self, init_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.final_regress_layer = nn.Conv1d(
            in_channels=init_dim,
            out_channels=self.out_dim,
            kernel_size=1
        )

    def forward(self, x):
        return self.final_regress_layer(x)


class Unet1DGDiff(nn.Module):
    def __init__(
        self,
        dim,
        channels,
        init_dim=None,
        out_dim=None,
        dropout=0.,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4
    ):
        super().__init__()

        self.dim = dim
        self.channels = channels
        self.self_condition = self_condition
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        phi_dim = 1

        ## attn_dim
        H = math.sqrt(1024)
        embed_size = int(H / 2)

        nc_1, nc_2, nc_3, nc_4, nc_5 = dim, dim * 2, dim * 4, dim * 8, dim * 16
        print('Dimension-Cluster: ', nc_1, nc_2, nc_3, nc_4, nc_5)

        # Initial convolution
        self.init_conv = BlockGDiff(channels, nc_1)

        # Down blocks
        self.down_blocks = nn.ModuleList([
            ResnetBlockGDiff(nc_1, nc_2),  ## (64, 128) ..
            ResnetBlockGDiff(nc_2, nc_3),
            ResnetBlockGDiff(nc_3, nc_4),
            ResnetBlockGDiff(nc_4, nc_5) ## (512, 1024)
        ])

        self.attn_blocks = nn.ModuleList([
            None,  # No attention in first two blocks
            None, 
            SelfAttention(nc_4, int(H / 8)),  ## (512, int(H/2))
            SelfAttention(nc_5, int(H / 16)) ## (1024, int(H/2))
        ])

        # Embeddings for time and phi
        self.time_embeddings = nn.ModuleList([
            TimeEmbedding(nc_2, embed_size), ## (128, )
            TimeEmbedding(nc_3, embed_size), 
            TimeEmbedding(nc_4, embed_size),
            TimeEmbedding(nc_5, embed_size) ## (1024, )
        ])

        self.alpha_embeddings = nn.ModuleList([
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_2), ## (, 128)
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_3),
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_4),
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_5) ## (, 1024)
        ])

        # Bottleneck
        self.bottleneck = BlockGDiff(nc_5, nc_5) ## (1024, 1024)
        self.att_bottleneck = SelfAttention(nc_5, int(H / 16))
        self.bottl_down = BlockGDiff(nc_5, nc_4) ## (1024, 512)

        # Up blocks
        self.up_blocks = nn.ModuleList([
            ResnetBlockGDiff(nc_5 + nc_4, nc_4), ## (1024 + 512, 512)
            ResnetBlockGDiff(nc_5, nc_3), ## (512 * 2, 256)
            ResnetBlockGDiff(nc_4, nc_2), ## (256 * 2, 128)
            ResnetBlockGDiff(nc_3, nc_1) ## (128 * 2, 64)
        ])

        self.time_embeddings_up = nn.ModuleList([
            TimeEmbedding(nc_4, embed_size), ## (512, )
            TimeEmbedding(nc_3, embed_size),
            TimeEmbedding(nc_2, embed_size),
            TimeEmbedding(nc_1, embed_size)
        ])

        self.alpha_embeddings_up = nn.ModuleList([
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_4), ## (512, )
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_3),
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_2),
            PhiEmbedding(phi_dim, in_dim=100, out_dim=nc_1)
        ])

        self.outc = final_regress(nc_1, channels)

    def forward(self, x, t, phi_ps=None):
        if phi_ps is None:
            phi_ps = torch.zeros(x.shape[0], 1, device=x.device).float()
            print('Careful, no alpha was given, set to 0')
        elif isinstance(phi_ps, (int, float)):
            phi_ps = torch.tensor(phi_ps, device=x.device).reshape(-1, 1).float()

        x = self.init_conv(x)
        residuals = []

        ## NOTE: We are perfoming temporal embedding to the MCMC sequence of DDPM model - not on the sequential elements of each data sample - to inject t, phi info in x_t state
        ## BUT: We perform self-attention on the sequential elements of data samples

        # Down path
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            x += self.time_embeddings[i](t) + self.alpha_embeddings[i](phi_ps).unsqueeze(-1)
            if self.attn_blocks[i] is not None:
                x = self.attn_blocks[i](x)
            residuals.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        x += self.time_embeddings[-1](t)
        x = self.att_bottleneck(x)
        x = self.bottl_down(x)
        
        # Up path
        for i, up in enumerate(self.up_blocks):
            res = residuals[-(i + 1)]
            x = torch.cat([x, res], dim=1)
            x = up(x)
            x += self.time_embeddings_up[i](t) + self.alpha_embeddings_up[i](phi_ps).unsqueeze(-1)

        return self.outc(x)    

    @torch.no_grad()
    def summary(self, x_shape, t_shape):
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


if __name__ == '__main__':
    print('running __unet_1d.py__')
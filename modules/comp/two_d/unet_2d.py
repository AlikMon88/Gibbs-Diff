import numpy as np
import torch 
import torch.nn as nn
import math

## --------------------------------------------------------------------------------------------------------
## --------------------------------GibbsDIFF-2D---------------------------------------------------------------
## --------------------------------------------------------------------------------------------------------

## add 2-conv2d layers
class BlockGDiff2D(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.2):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1) ## padding = 'same'?
        self.norm = nn.GroupNorm(1, dim_out)  # Suitable for spatial data
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ResnetBlockGDiff2D(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = BlockGDiff2D(dim, dim_out, dropout=dropout)
        self.block2 = BlockGDiff2D(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale, shift = time_emb.chunk(2, dim=1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            x = x * (1 + scale) + shift

        h = self.block1(x)
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
        qkv = self.to_qkv(x).reshape(b, self.heads * 3, -1, h * w)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.heads, -1, h * w)
        k = k.view(b, self.heads, -1, h * w)
        v = v.view(b, self.heads, -1, h * w)

        q = q * self.scale
        sim = torch.einsum('bhcn,bhcm->bhnm', q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class TimeEmbedding2D(nn.Module):
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
        return pos_enc  # Shape [B, D]

class PhiEmbedding2D(nn.Module):
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


class final_regress2D(nn.Module):
    def __init__(self, init_dim, out_dim):
        super().__init__()
        self.final_regress_layer = nn.Conv2d(
            in_channels=init_dim,
            out_channels=out_dim,
            kernel_size=1
        )

    def forward(self, x):
        return self.final_regress_layer(x)


class Unet2DGDiff(nn.Module):
    def __init__(
        self,
        dim,
        channels,
        dropout=0.2,
        phi_dim=1,
        attn_dim_head=32,
        attn_heads=4, 
        device=None
    ):
        super().__init__()

        self.dim = dim
        self.channels = channels
        if not device:
            self.device = 'cpu' if not torch.cuda.is_available() else 'cpu'

        # Embed size for time/phi embeddings
        H = 32  # Typical spatial size for image features
        embed_size = int(H)

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
            TimeEmbedding2D(nc_2, embed_size),
            TimeEmbedding2D(nc_3, embed_size),
            TimeEmbedding2D(nc_4, embed_size),
            TimeEmbedding2D(nc_5, embed_size)
        ])

        self.alpha_embeddings = nn.ModuleList([
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_2),
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_3),
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_4),
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_5)
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
            TimeEmbedding2D(nc_4, embed_size),
            TimeEmbedding2D(nc_3, embed_size),
            TimeEmbedding2D(nc_2, embed_size),
            TimeEmbedding2D(nc_1, embed_size)
        ])

        self.alpha_embeddings_up = nn.ModuleList([
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_4),
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_3),
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_2),
            PhiEmbedding2D(phi_dim, in_dim=100, out_dim=nc_1)
        ])

        self.outc = final_regress2D(nc_1, channels)

    def forward(self, x, t, phi_ps=None):
        if phi_ps is None:
            phi_ps = torch.zeros(x.shape[0], 1, device=x.device)
        elif isinstance(phi_ps, (int, float)):
            phi_ps = torch.tensor(phi_ps, device=x.device).reshape(-1, 1).float()

        x = self.init_conv(x)
        residuals = []

        # Down path
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            emb = self.time_embeddings[i](t) + self.alpha_embeddings[i](phi_ps)
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
            emb = self.time_embeddings_up[i](t) + self.alpha_embeddings_up[i](phi_ps)
            x = x + emb.unsqueeze(-1).unsqueeze(-1)

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

if __name__ == '__main__.py':
    print('running __unet_2d.py__')
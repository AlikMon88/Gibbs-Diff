'''
UNET: Noise Predictor/Estimator Model - Modularized to fit into the Diffusion Pipeline
'''

import numpy as np
import random
import matplotlib.pyplot as plt

import math
from functools import partial

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torchsummary

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from ..utils.helper import *
from ..utils.noise_create import create_1d_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Residual-Component
class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

### DeConvolution
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

### Convolution
def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

### Root-Mean-Squared based Normalization
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

### Pre-Normalization
class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

#### SinoShoidal based Temporal embedding
class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

### Conv-Block
class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

### ResnetBlock (Block units)
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        ## n = sequenc_len == out_dim, dim_head = dimensionality of each head
        ### (b, heads * dim_head * 3, n) 
        ### .chunk() decomposes into 3 different (b, heads * dim_head, n) vectors (q, k, v) along feature axis
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        ##reshapes into mult-head format (b, heads, dim_head, n)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class final_regress(Module):
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


class Unet1D(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        self.dim = dim
        self.dim_mults = dim_mults

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        ## creates a unpacked (*) dimensionality list
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print('Obtained-Dimensionalities: ', dims, in_out)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, ## concated cos + sine for positional info
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        ## partially applied function (pre-fills key-args like default args - so no need to pass them every call) (when default value are dynamic)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers (Conv-Deconv)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)


        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_regress_conv = final_regress(init_dim, self.out_dim)
        
    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        ### Convolution
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        ### De-Convolution
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_regress_conv(x)
        
        return x
    

    def summary(self, x_shape, t_shape):
        """
        Kears-style Summary of the Unet1D model architecture.
        
        Args:
            x_shape: Shape of input data tensor (batch_size, channels, seq_length)
            t_shape: Shape of timestep tensor (batch_size,)
        """
        
        # Create example inputs
        x = torch.zeros(x_shape).to(device)
        t = torch.zeros(t_shape, dtype=torch.long).to(device)
        
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
    print('Running ... __unet.py__ ...')

    n_samples = 10000
    rn = random.randint(0, n_samples)

    observation, signal, noise = create_1d_data(n_samples=n_samples)
    print(observation.shape, signal.shape, noise.shape)

    plt.figure(figsize=(10, 5))

    plt.plot(observation[rn], label="Observation", alpha=0.8, marker='.')
    plt.plot(signal[rn], label="Signal", linestyle="dashed", alpha=0.7, color = 'red')
    plt.legend()
    plt.show()




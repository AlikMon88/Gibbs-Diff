class Unet1DGDiff(Module):
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
    print('running __unet_1d.py__')
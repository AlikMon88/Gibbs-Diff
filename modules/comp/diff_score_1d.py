class DiffScore(pl.LightningModule):
    """
    The paper and implementation looks different??
    DDPM models are discrete-time score based models (noise-predictors) while score based SDE models simulates reverse SDE in continous-time space
    This paper implements DDPM + extra phi_embedding 
    ## I Should directly use 1D - LucidBrains DDPM execution
    ## How are we denoising without t? get_closest_timestep() --> matching based on minimum abs. t-difference across all the steps (Lookup approach)

    Attributes:
        diffustion_steps (int): Number of diffusion steps.
        beta_small (float): Small value of beta for diffusion.
        beta_large (float): Large value of beta for diffusion.
        timesteps (torch.Tensor): Timesteps for diffusion.
        beta_t (torch.Tensor): Beta values for each timestep.
        alpha_t (torch.Tensor): Alpha values for each timestep.
        alpha_bar_t (torch.Tensor): Cumulative product of alpha values.

    Methods:
        pos_encoding: Computes positional encoding of time.
        forward: Performs forward pass of the model.
        get_loss: Computes the loss for training.
        denoise_1step: Performs denoising for one diffusion step.
        training_step: Training step for the LightningModule.
        validation_step: Validation step for the LightningModule.
        configure_optimizers: Configures the optimizer for training.
        blind_denoising: Performs blind denoising with Gibbs sampling.
        blind_denoising_pmean: Performs blind denoising with the posterior mean estimator.
        get_closest_timestep: Returns the closest timestep t to the given noise level \sigma.
        denoise_samples_batch_time: Denoises a batch of images for a given number of timesteps (time can be different for the elements of the batch).
    """
    
    def __init__(self,
                 in_size, 
                 diffusion_steps, 
                 img_depth=3, 
                 lr=2e-4,
                 weight_decay=0):

        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.beta_small = 0.1 / self.diffusion_steps
        self.beta_large = 20 / self.diffusion_steps
        self.timesteps = torch.arange(0, self.diffusion_steps)
        self.beta_t = self.beta_small + (self.timesteps / self.diffusion_steps) * (self.beta_large - self.beta_small)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0) 

        self.in_size = in_size
        H = math.sqrt(in_size)
        self.img_depth = img_depth
        
        phi_dim = 1 #number of dimensions of phi
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.inc = DoubleConv(self.img_depth, 64) #img_depth is the number of channels
        nc_1 = 128
        self.down1 = Down(64, nc_1) #double the number of channels, but H/2 and W/2
        self.alpha_embed1 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_1)
        nc_2 = 256
        self.down2 = Down(128, nc_2)
        self.alpha_embed2 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_2)
        nc_3 = 512 
        self.down3 = Down(256, nc_3)
        self.alpha_embed3 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_3)
        self.sa3 = SAWrapper(512, int(H/8))
        nc_4 = 1024
        self.down4 = Down(512, nc_4)
        self.alpha_embed4 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_4)
        self.sa4 = SAWrapper(1024, int(H/16)) #if H = 256: bootleneck dim = 16
        
        self.bottleneck = DoubleConv(1024, 1024)
        self.att_bottleneck = SAWrapper(1024, int(H/16))
        nc_5 = 512
        self.up1 = Up(1024, nc_5)
        self.alpha_embed5 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_5)
        nc_6 = 256
        self.up2 = Up(512, 256)
        self.alpha_embed6 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_6)
        nc_7 = 128
        self.up3 = Up(256, 128)
        self.alpha_embed7 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_7)
        nc_8 = 64
        self.up4 = Up(128, 64)
        self.alpha_embed8 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_8)
        self.outc = OutConv(64, self.img_depth)

    def pos_encoding(self, t, channels, embed_size):
        """
        Positinal encoding of time, as in the original transformer paper. 
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1)

    def forward(self, x, t, phi_ps=None):
        """
        The model is a U-Net with added positional encodings for time, embedding of phi  and self-attention layers. The total number of parameters is ~70M.
        """
        if phi_ps is None:
            phi_ps = torch.zeros(x.shape[0],1).to(self.device).float()
            print('Careful, no alpha was given, set to 0')
        elif isinstance(phi_ps, int) or isinstance(phi_ps, float):
            phi_ps = torch.tensor(phi_ps).reshape(-1,1).to(self.device).float()
        else:
            pass
        bs, n_channels, H, W = x.shape 
        x1 = self.inc(x) #dim = 256, n_channel = 64
        x2 = self.down1(x1) + self.pos_encoding(t, 128, int(H/2)) + self.alpha_embed1(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 128, n_channel = 128, unsqueeze for broadcasting on HxW
        x3 = self.down2(x2) + self.pos_encoding(t, 256, int(H/4)) + self.alpha_embed2(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 64, n_channel = 256
        x4 = self.down3(x3) + self.pos_encoding(t, 512, int(H/8)) + self.alpha_embed3(phi_ps).unsqueeze(-1).unsqueeze(-1)#dim = 32, n_channel = 512
        x4 = self.sa3(x4) #dim = 32, n_channel = 512 
        x5 = self.down4(x4) + self.pos_encoding(t, 1024, int(H/16)) + self.alpha_embed4(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 16, n_channel = 1024
        x5 = self.sa4(x5) #dim = 16, n_channel = 1024 
        x_bottleneck = self.bottleneck(x5) + self.pos_encoding(t, 1024, int(H/16)) #dim = 16, n_channel = 1024 
        x_bottleneck = self.att_bottleneck(x_bottleneck) #dim = 16, n_channel = 1024
        x = self.up1(x_bottleneck, x4) + self.pos_encoding(t, 512, int(H/8)) + self.alpha_embed5(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 32, n_channel = 512
        x = self.up2(x, x3) + self.pos_encoding(t, 256, int(H/4)) + self.alpha_embed6(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 64, n_channel = 256
        x = self.up3(x, x2) + self.pos_encoding(t, 128, int(H/2)) + self.alpha_embed7(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 128, n_channel = 128
        x = self.up4(x, x1) + self.pos_encoding(t, 64, int(H)) + self.alpha_embed8(phi_ps).unsqueeze(-1).unsqueeze(-1)#dim = 256, n_channel = 64
        output = self.outc(x) #dim = 256, n_channel = 3

        return output

    def get_loss(self, batch, batch_idx, phi_ps=None):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020), but with colored noise.
        """
        #phi should be a tensor of the size of the batch_size, we want a phi different for each batch element
        #if batch is a list (the case of ImageFolder for ImageNet): take the first element, otherwise take batch:
        if isinstance(batch, list):
            batch = batch[0]

        bs = batch.shape[0]
        if phi_ps is None:
            #sample phi_ps between -1 and 1 | shape (batch_size, 1)
            phi_ps = torch.rand(bs, 1, device=self.device)*2 - 1

        #if phi is a scalar, cast to batch dimension. For training on a single phi.
        if isinstance(phi_ps, float) or isinstance(phi_ps, int):
            phi_ps = phi_ps * torch.ones(bs,1).to(self.device) 

        ts = torch.randint(0, self.diffusion_steps, (bs, 1)).float().to(self.device)
        noise_imgs = []

        epsilons = get_colored_noise_2d(batch.shape, phi_ps, device= self.device) #B x C x H x W

        a_hat = self.alpha_bar_t[ts.squeeze(-1).int().cpu()].reshape(-1, 1, 1, 1).to(self.device)
        noise_imgs = torch.sqrt(a_hat) * batch + torch.sqrt(1 - a_hat) * epsilons ## x_t

        e_hat = self.forward(noise_imgs, ts, phi_ps=phi_ps) ## since noise is parameterised by (phi)
        loss = nn.functional.mse_loss(e_hat, epsilons)

        return loss

    def denoise_1step(self, x, t, phi_ps=None):
        """
        Denoises one step given an phi (if phi_ps=None, it is sampled uniformly in [-1,1]) and a timestep t.
        x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * e_hat) + sqrt(beta_t) * z_phi
        """
        #phi should be a tensor of the size of the batch_size, we want a different phi for each batch element
        if phi_ps is None:5
            # If no phi_ps is given, assume it's white noise, i.e. phi_ps = 0
            phi_ps = torch.zeros(x.shape[0],1, device=self.device).float()
        
        #if phi is a scalar, cast to batch dimension
        if isinstance(phi_ps, int) or isinstance(phi_ps, float):
            phi_ps = phi_ps * torch.ones(x.shape[0],1, device=self.device).float() 
        
        else: 
            phi_ps = phi_ps.to(self.device).float()
        
        with torch.no_grad():
            if t > 1:
                z = get_colored_noise_2d(x.shape, phi_ps, device= self.device)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1), phi_ps=phi_ps)
            pre_scale = 1 / math.sqrt(self.alpha_t[t])
            e_scale = (self.beta_t[t]) / math.sqrt(1 - self.alpha_bar_t[t])
            post_sigma = math.sqrt(self.beta_t[t]) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x
            
    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def denoise_samples_batch_time(self, noisy_batch, timesteps, batch_origin=None, return_sample=False, phi_ps=None):
        """
        Denoises a batch of images for a given number of timesteps (which can be different across the batch).
        """
        max_timesteps = torch.max(timesteps)
        mask = torch.ones(noisy_batch.shape[0], max_timesteps+1).to(self.device)
        for i in range(noisy_batch.shape[0]):
            mask[i, timesteps[i]+1:] = 0

        ## Reverse-Diffusion
        for t in range(max_timesteps, 0, -1):
            noisy_batch = self.denoise_1step(noisy_batch, torch.tensor(t).cuda(), phi_ps) * (mask[:, t]).reshape(-1,1,1,1) + noisy_batch * (1 - mask[:, t]).reshape(-1,1,1,1)
        if batch_origin is None:
            return noisy_batch
        else:
            if return_sample:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), noisy_batch
            else:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), None

def get_closest_timestep(noise_level, ret_sigma=False):
        """
        Returns the closest timestep to the given noise level. If ret_sigma is True, also returns the noise level corresponding to the closest timestep.
        """
        alpha_bar_t = self.alpha_bar_t.to(noise_level.device)
        all_noise_levels = torch.sqrt((1-alpha_bar_t)/alpha_bar_t).reshape(-1, 1).repeat(1, noise_level.shape[0]) #--> (T=#timesteps_cumprod, N=#noise_levels)
        print('Noise-Levels: ', all_noise_levels)
        
        closest_timestep = torch.argmin(torch.abs(all_noise_levels - noise_level), dim=0)
        print('Closest-Timesteps: ', cloesest_timestep)

        if ret_sigma:
            return closest_timestep, all_noise_levels[closest_timestep, 0]
        else:
            return closest_timestep

if __name__ == '__main__':
    
    sigmas = [0.2]
    _ = get_closeest_timestep(sigmas)
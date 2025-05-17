import numpy as np

import math
from random import random
from functools import partial
from collections import namedtuple

import torch

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

## Custom Modules
from ..utils.helper import *
from .unet import *
from ..utils.noise_create import get_colored_noise_1d

from torch.cuda.amp import autocast


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_t', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(Module): ## We by-default perform the pred_noise implementation
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super().__init__()

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.timesteps = timesteps

        self.seq_length = seq_length

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior p(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        loss_weight = torch.ones_like(snr)
        
        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        ## There might be some problem in joint training -- t --> deterministically predicts x_t --> which trains for pred_noise --> revert to t (MIGHT break the training)
        model_output, t = self.model(x, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, t, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True): ## Ancestral sampling
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0 ## Noise injection for stochastic generatiion
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps): ## Ancestral Sampling from surrogate distribution
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))


    ### Running the Inferencing on CUDA - for acceleration 
    # @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):

        b, c, n = x_start.shape

        # Generate noise if not given
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Forward diffusion (sample x_t from q(x_t | x_0, t))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Self-conditioning
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # Model prediction (UNet output)
        pred_noise, pred_t = self.model(x, x_self_cond)  # Shape: [B, channels+1, T]
        
        # Target values
        target_noise = noise
        target_t = t.float()  # Broadcast target_t

        # Losses
        loss_noise = F.mse_loss(pred_noise, target_noise, reduction='none')
        loss_noise = reduce(loss_noise, 'b ... -> b', 'mean')

        loss_t = F.mse_loss(pred_t, target_t, reduction='none')
        loss_t = reduce(loss_t, 'b ... -> b', 'mean')

        loss = loss_noise + loss_t

        ## loss-weighted based on SNR at timestep t (batched/indexed at different t)
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() ### t ~ uniformly-sampled(0, num_time_steps) == batch_size

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

#### -----------------------------------------------------------------------------------
#### ---------------------------GibbsDIFF-----------------------------------------------
#### -----------------------------------------------------------------------------------

class GibbsDiff1D(Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        num_timesteps = 1000, 
        sampling_timesteps = None):

        super().__init__()

        self.model = model
        self.device = model.device
        self.num_timesteps = num_timesteps
        self.seq_length = seq_length

        self.beta_small = 0.1 / self.num_timesteps
        self.beta_large = 20 / self.num_timesteps
        self.timesteps_t = torch.arange(0, self.num_timesteps)
        self.beta_t = self.beta_small + (self.timesteps_t / self.num_timesteps) * (self.beta_large - self.beta_small)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0) 

        if not sampling_timesteps:
            self.sampling_timesteps = self.num_timesteps


    def get_gdiff_loss(self, batch, ts, phi_ps=None):
        
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

        noise_imgs = []

        ### non-flat PSD colored-noise
        epsilons, _ = get_colored_noise_1d(batch.shape, phi_ps, device= self.device) #B x seq_dim X seq_len
        epsilons = epsilons.view(epsilons.shape[0], 1, -1)

        a_hat = self.alpha_bar_t[ts.squeeze(-1).int().cpu()].reshape(-1, 1, 1).to(self.device)
        noise_imgs = torch.sqrt(a_hat) * batch + torch.sqrt(1 - a_hat) * epsilons ## x_t

        e_hat = self.model(noise_imgs, ts, phi_ps=phi_ps) ## since noise is parameterised by (phi)
        loss = nn.functional.mse_loss(e_hat, epsilons)

        return loss

    def forward(self, img, *args, **kwargs):
        
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long() ### t ~ uniformly-sampled(0, num_timesteps) == batch_size

        # img = self.normalize(img)
        return self.get_gdiff_loss(img, t, *args, **kwargs)

    def get_closest_timestep(self, noise_level, ret_sigma=False):
        """
        Returns the closest timestep to the given noise level. If ret_sigma is True, also returns the noise level corresponding to the closest timestep.
        """
        alpha_bar_t = self.alpha_bar_t.to(noise_level.device)
        all_noise_levels = torch.sqrt((1-alpha_bar_t)/alpha_bar_t).reshape(-1, 1).repeat(1, noise_level.shape[0]) #--> (T=#timesteps_cumprod, N=#noise_levels)
        
        closest_timestep = torch.argmin(torch.abs(all_noise_levels - noise_level), dim=0)

        if ret_sigma:
            return closest_timestep, all_noise_levels[closest_timestep, 0]
        else:
            return closest_timestep
    

    @torch.no_grad()
    def denoise_1step_gdiff(self, x, t, phi_ps=None):
        """
        Denoises one step given an phi (if phi_ps=None, it is sampled uniformly in [-1,1]) and a timestep t.
        x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * e_hat) + sqrt(beta_t) * z_phi
        """

        #phi should be a tensor of the size of the batch_size, we want a different phi for each batch element
        if phi_ps is None:
            # If no phi_ps is given, assume it's white noise, i.e. phi_ps = 0
            phi_ps = torch.zeros(x.shape[0],1, device=self.device).float()
        
        #if phi is a scalar, cast to batch dimension
        if isinstance(phi_ps, int) or isinstance(phi_ps, float):
            phi_ps = phi_ps * torch.ones(x.shape[0],1, device=self.device).float() 
        
        else: 
            phi_ps = phi_ps.to(self.device).float()
        
        with torch.no_grad():
            if t > 1:
                z, _ = get_colored_noise_1d(x.shape, phi_ps, device= self.device)
            else:
                z = 0
            
            t_ch = t.view(1).repeat(x.shape[0],)
            e_hat = self.model(x, t_ch, phi_ps=phi_ps)
            pre_scale = 1 / math.sqrt(self.alpha_t[t])
            e_scale = (self.beta_t[t]) / math.sqrt(1 - self.alpha_bar_t[t])
            post_sigma = math.sqrt(self.beta_t[t]) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x
    
    @torch.no_grad()
    def denoise_samples_batch_time(self, noisy_batch, timesteps, batch_origin=None, return_sample=False, phi_ps=None):
        """
        Denoises a batch of images for a given number of timesteps (which can be different across the batch).
        """
        
        max_timesteps = torch.max(timesteps)
        mask = torch.ones(noisy_batch.shape[0], max_timesteps+1).to(self.device)

        for i in range(noisy_batch.shape[0]):
            mask[i, timesteps[i]+1:] = 0

        ## Reverse-Diffusion (t(x_t) -> 0) --> Denoising from t to 0 | NOT Generating
        for t in range(max_timesteps, 0, -1):
            noisy_batch = self.denoise_1step_gdiff(noisy_batch, torch.tensor(t), phi_ps) * (mask[:, t]).reshape(-1,1,1) + noisy_batch * (1 - mask[:, t]).reshape(-1,1,1)
        
        # noisy_batch = self.unnormalize(noisy_batch)

        if batch_origin is None:
            return noisy_batch
        else:
            if return_sample:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), noisy_batch
            else:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), None
    



if __name__ == '__main__':
    print('Running ... __diffusion_1d.py ...')

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
from ..utils.hmc import *
from ..utils.hmc import sample_hmc
from ..utils.hmc import get_phi_all_bounds

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
    
    ## y - not t-indexed noisy image and yt - normalized t-indexed noisy image | we don't use pytorch grad_calculate
    ## SIMPLE GIBBS SAMPLER + HMC EXECUTION
    def run_gibbs_sampler(self, y, yt, num_chains_per_sample, n_it_gibbs=5, n_it_burnin=1, sigma_min=0.04, sigma_max=0.4, return_chains=False):

        device = self.model.device
        ps_model = ColoredPowerSpectrum1D(device=device)

        phi_max = 1.0
        phi_min = -phi_max

        batch_size = y.shape[0]
        chains = num_chains_per_sample
        total_chains = batch_size * chains

        # Repeat inputs for each chain
        y = y.repeat_interleave(chains, dim=0)     # (B * C, ...)
        yt = yt.repeat_interleave(chains, dim=0)   # (B * C, ...)
        # print(y.shape, yt.shape)

        # Initialize phi and sigma
        phi_init = sample_phi_prior(total_chains).unsqueeze(1)  # (B*C, 1)
        sigma_init = get_noise_estimate(y, sigma_min, sigma_max).to(device).repeat(total_chains, 1)  # (B*C, 1)
    
        phi_init_all = torch.cat([phi_init, sigma_init], dim=1)  # (B*C, 2)
        # print('phi_init: ', phi_init_all, phi_init_all.shape)
        phi_all = [phi_init_all]

        phi_all_min, phi_all_max = get_phi_all_bounds(phi_min, phi_max, sigma_min, sigma_max, device)

        x_all = []
        step_size = None
        inv_mass_matrix = None

        ##pre-filling
        ## prior(phi)
        log_prior = lambda phi: log_prior_phi_sigma(phi[:, 0], phi[:, 1])
        ## log-likelihood(eps|phi) 
        log_likelihood = lambda phi, eps: log_likelihood_eps_phi_sigma(phi[:, 0], phi[:, 1], eps, ps_model)

        for i in range(n_it_gibbs + n_it_burnin):

            phi = phi_all[-1]  # (B*C, 2)

            # print()
            # print('UPDATED-PHI: ', phi)
            # print(phi.shape)
            # print()

            # Step 1: DDOM Posterior Sampling
            t = self.get_closest_timestep(phi[:, 1])  # sigma values
            x = self.denoise_samples_batch_time(yt, t, phi_ps=phi[:, :1])  # (B*C, ...)
            epsilon = (y - x)
            # print('epsilon:')
            # print(torch.mean(epsilon, dim=-1))

            # Step 2: HMC Sampling
            def log_posterior(phi, epsilon):
                return log_likelihood(phi, epsilon) + log_prior(phi)

            ## pre-filling
            log_prob_fn = lambda phi: log_posterior(phi, epsilon)

            def gradient_log_prob(phi):

                phi_clone = phi.clone().requires_grad_(True)
                logp = log_posterior(phi_clone, epsilon)
                # print('logp:')
                # print(type(logp), logp.shape, logp)
                # print('logp.requires_grad:', logp.requires_grad)
                # print('logp.grad_fn:', logp.grad_fn)
                grad_phi = torch.autograd.grad(logp, phi_clone, grad_outputs=torch.ones_like(logp))[0]
                return grad_phi

            if i == 0:
                phi_new, step_size, inv_mass_matrix = sample_hmc(log_prob_fn=log_prob_fn, log_grad=gradient_log_prob, phi_init=phi, adapt=True)
            else:
                phi_new = sample_hmc(log_prob_fn=log_prob_fn, log_grad=gradient_log_prob, phi_init=phi, step_size=step_size, inv_mass_matrix=inv_mass_matrix, adapt=False)

            phi_all.append(phi_new)
            x_all.append(x)

        if return_chains:
            ## returns the entire gibbs chain
            return torch.stack(phi_all, dim=1).detach().cpu(), torch.stack(x_all, dim=1).detach().cpu()  # (B*C, steps, ...)
        else:
            ## returns the last gibbs state
            return phi_all[-1].detach().cpu(), x_all[-1].detach().cpu()

    ## there are 2 MCMC (HMC and Gibbs) chains we talk about the gibbs chain ofc 
    def blind_posterior_mean(self,y, yt, norm_phi_mode='compact', num_chains_per_sample=5, n_it_gibbs=5, n_it_burnin=1, avg_pmean=2): ## last avg_pmean positions
        
        '''Performs blind denoising with the posterior mean estimator. | Run Multiple MCMC chains'''

        phi_all, x_all = self.run_gibbs_sampler(y, yt, num_chains_per_sample=num_chains_per_sample, n_it_gibbs=n_it_gibbs, n_it_burnin=n_it_burnin, return_chains=True)
        # The multi-MCMC chain posterior -> (#chains, batch_size, chain_length (taking last avg_mean states), channel_depth, seq_len) 
        # We take a mean over #chains & avg_mean chain_len --> (batch_size, channel, seq_len)
        x_denoised_pmean = x_all[:, -avg_pmean:].reshape(num_chains_per_sample, -1, avg_mean, self.channels, self.seq_len).mean(dim=(0, 2)) ## Each batch-sample will have num_chain_per_sample distinct MCMC chain - each chain of (n_it_gibbs + n_it_burnin) chain length 
        
        return phi_all, x_denoised_pmean
        

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

        ## Reverse-Diffusion (t(x_t) -> 0) --> Denoising from t to 0 | NOT Generating | Ancestral Sampling
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

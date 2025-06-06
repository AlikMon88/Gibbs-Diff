import numpy as np

import math
from random import random
from functools import partial
from collections import namedtuple

import torch
import torch.nn as nn

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

## Custom Modules
from ...utils.helper import *
from .unet_1d import *
from ...utils.noise_create import get_colored_noise_1d
from ...utils.hmc import *
from ...utils.hmc import get_noise_estimate_1d, get_noise_estimate_2d
from ...utils.hmc import sample_hmc
from ...utils.hmc import get_phi_all_bounds

## HMC-verison2
from ...utils.hmc_v2 import get_noise_estimate_1d, get_noise_estimate_2d
from ...utils.hmc_v2 import sample_hmc_v2
from ...utils.hmc_v2 import get_phi_all_bounds

from torch.cuda.amp import autocast


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


class GibbsDiff1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_len,
        num_timesteps = 1000, 
        sampling_timesteps = None):

        super().__init__()

        self.model = model
        self.device = model.device
        self.num_timesteps = num_timesteps
        self.seq_len = seq_len
        self.channels = 1

        self.beta_small = 0.1 / self.num_timesteps
        self.beta_large = 20 / self.num_timesteps
        self.timesteps_t = torch.arange(0, self.num_timesteps)
        self.beta_t = self.beta_small + (self.timesteps_t / self.num_timesteps) * (self.beta_large - self.beta_small)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0) 

        self.n_leapfrog_steps = 2  
        self.chain_length =  5
        self.burnin_steps = 2
        self.n_adapt = 5

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
        
        b, c, n, device, seq_len, = *img.shape, img.device, self.seq_len
        assert n == seq_len, f'seq length must be {seq_len}'
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
    def run_gibbs_sampler(self, y, yt, num_chains_per_sample, n_it_gibbs=5, n_it_burnin=1, sigma_min=0.04, sigma_max=0.4, return_chains=False, sampler_v2=False):

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
    
        # Initialize phi and sigma
        phi_init = sample_phi_prior(total_chains).unsqueeze(1)  # (B*C, 1)
        ## high-freq component capture using low-pass filter
        sigma_init = get_noise_estimate_1d(y, sigma_min, sigma_max).to(device).repeat(total_chains, 1)  # (B*C, 1)
    
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

        ## HMC-Sampler
        if sampler_v2:
            hmc_prefill = lambda log_prob_fn, log_grad, phi_init, step_size, inv_mass_matrix, adapt: sample_hmc_v2(log_prob_fn, log_grad, phi_init, step_size=step_size, n_leapfrog_steps=self.n_leapfrog_steps, chain_length=self.chain_length, burnin_steps=self.burnin_steps, \
            inv_mass_matrix=inv_mass_matrix, adapt=adapt, n_adapt=self.n_adapt, phi_min_norm=None, phi_max_norm=None) 

        else:
            hmc_prefill = lambda log_prob_fn, log_grad, phi_init, step_size, inv_mass_matrix, adapt: sample_hmc(log_prob_fn, log_grad, phi_init, step_size=step_size, n_leapfrog_steps=self.n_leapfrog_steps, chain_length=self.chain_length, burnin_steps=self.burnin_steps, \
            inv_mass_matrix=inv_mass_matrix, adapt=adapt, n_adapt=self.n_adapt, phi_min_norm=None, phi_max_norm=None) 

        hmc_accept_list = []
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
            # print('epsilon:', epsilon.shape)
            # print(torch.mean(epsilon, dim=-1))

            # Step 2: HMC Sampling
            def log_posterior(phi, epsilon):
                return log_likelihood(phi, epsilon) + log_prior(phi)

            ## pre-filling
            log_prob_fn = lambda phi: log_posterior(phi, epsilon)

            def gradient_log_prob(phi):
                phi_clone = phi.clone().requires_grad_(True)
                logp = log_posterior(phi_clone, epsilon)
                grad_phi = torch.autograd.grad(logp, phi_clone, grad_outputs=torch.ones_like(logp))[0]
                return grad_phi

            if i == 0: ## first-gibbs-step-adaption
                # phi_new, step_size, inv_mass_matrix = sample_hmc(log_prob_fn=log_prob_fn, log_grad=gradient_log_prob, phi_init=phi, n_leapfrog_steps=self.n_leapfrog_steps, chain_length=self.chain_length, burnin_steps=self.burnin_steps, adapt=True)
                phi_new, step_size, inv_mass_matrix, _ = hmc_prefill(log_prob_fn=log_prob_fn, log_grad=gradient_log_prob, phi_init=phi, step_size=0.1, inv_mass_matrix=None, adapt=True)
           
            else:
                # phi_new = sample_hmc(log_prob_fn=log_prob_fn, log_grad=gradient_log_prob, phi_init=phi, step_size=step_size, inv_mass_matrix=inv_mass_matrix, adapt=False)
                phi_new, hmc_accept_mean = hmc_prefill(log_prob_fn=log_prob_fn, log_grad=gradient_log_prob, phi_init=phi, step_size=step_size, inv_mass_matrix=inv_mass_matrix, adapt=False)
                hmc_accept_list.append(hmc_accept_mean)

            phi_all.append(phi_new)
            x_all.append(x)
        
        ## After phi distrib convergence
        print('HMC-last-state-Acceptance-Probability: ', hmc_accept_mean)
        print('HMC-mean-accept-proba: ', np.mean(np.array(hmc_accept_list), axis=0))

        if return_chains:
            ## returns the entire gibbs chain
            return torch.stack(phi_all, dim=1).detach().cpu(), torch.stack(x_all, dim=1).detach().cpu()  # (B*C, steps, ...)
        else:
            ## returns the last gibbs state
            return phi_all[-1].detach().cpu(), x_all[-1].detach().cpu()

    ## there are 2 MCMC (HMC and Gibbs) chains we talk about the gibbs chain ofc 
    def blind_posterior_mean(self,y, yt, norm_phi_mode='compact', num_chains_per_sample=5, n_it_gibbs=5, n_it_burnin=1, avg_pmean=2, return_post=False, sampler_v2=False): ## last avg_pmean positions
        
        '''Performs blind denoising with the posterior mean estimator. | Run Multiple MCMC chains'''

        phi_all, x_all = self.run_gibbs_sampler(y, yt, num_chains_per_sample=num_chains_per_sample, n_it_gibbs=n_it_gibbs, n_it_burnin=n_it_burnin, return_chains=True, sampler_v2=sampler_v2)
        
        # The multi-MCMC chain posterior -> (#chains, batch_size, chain_length (taking last avg_mean states), channel_depth, seq_len) 
        # We take a mean over #chains & avg_mean chain_len --> (batch_size, channel, seq_len)
        
        phi_all_posterior = phi_all[:, -avg_pmean:].reshape(num_chains_per_sample, -1, avg_pmean, 2)
        x_denoised_posterior = x_all[:, -avg_pmean:].reshape(num_chains_per_sample, -1, avg_pmean, self.channels, self.seq_len) 
        
        x_denoised_pmean = x_denoised_posterior.mean(dim=(0, 2)) ## Each batch-sample will have num_chain_per_sample distinct MCMC chain - each chain of (n_it_gibbs + n_it_burnin) chain length 
        phi_all_mean = phi_all_posterior.mean(dim=(0, 2))

        if not return_post:
            return phi_all_mean, x_denoised_pmean
        else:
            return phi_all_posterior, x_denoised_posterior

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
                z = z.reshape(z.shape[0], 1, -1)
            else:
                z = 0
            
            t_ch = t.view(1).repeat(x.shape[0],)
            e_hat = self.model(x, t_ch, phi_ps=phi_ps) ## (b, dim, seq_dim)
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
            x_t = self.denoise_1step_gdiff(noisy_batch, torch.tensor(t), phi_ps)
            noisy_batch = x_t * (mask[:, t]).reshape(-1, 1, 1) + noisy_batch * (1 - mask[:, t]).reshape(-1, 1, 1)
        
        # noisy_batch = self.unnormalize(noisy_batch)

        if batch_origin is None:
            return noisy_batch
        else:
            if return_sample:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), noisy_batch
            else:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), None


if __name__ == '__main__':
    print('running __diffusion_1d.py__')
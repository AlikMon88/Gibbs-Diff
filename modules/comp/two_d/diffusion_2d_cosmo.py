# Assuming Unet2DCosmoGDiff is defined as per previous response
# from .unet_2d_cosmo import Unet2DCosmoGDiff # Or however you import it
import numpy as np

import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch.cuda.amp import autocast
import torch.nn as nn

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

## Custom Modules
from ...utils.helper import *
from .unet_2d import *
from ...utils.noise_create_2d import get_colored_noise_2d
from ...utils.hmc_cosmo import *

from pixell import enmap
from ...utils.cosmo_create import get_cmb_noise_batch

# Helper functions for noise schedule (can be kept from your original)
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
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# Priors (from paper Section 3.2)
H0_PRIOR_MIN, H0_PRIOR_MAX = 50.0, 90.0
OMBH2_PRIOR_MIN, OMBH2_PRIOR_MAX = 0.0075, 0.0567 # Note: paper uses omega_b, CAMB uses ombh2
# To convert: omega_b = ombh2 / (H0/100)^2. For priors, it's easier to sample H0 and ombh2 directly.
SIGMA_CMB_PRIOR_MIN, SIGMA_CMB_PRIOR_MAX = 0.1, 1.2 # sigma_min should be >0. Let's use 0.1 for now.


class GibbsDiff2D_cosmo(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size, # Tuple (H, W), e.g., (256, 256)
        num_timesteps_ddpm=1000, # Timesteps for the DDPM noise schedule
        sampling_timesteps_ddpm=None, # For DDIM-like acceleration if used
        ddpm_beta_schedule='linear', # 'linear' or 'cosine'
        # HMC specific parameters for Phi_CMB sampling
        hmc_n_leapfrog_steps=5,
        hmc_chain_length=1, # Number of HMC samples per Gibbs iteration
        hmc_burnin_steps=2, # Burn-in for HMC *within* each Gibbs step (usually small or 0 after initial adaptation)
        hmc_adapt_stepsize_iters = 5 # For initial HMC step size adaptation
    ):
        super().__init__()

        self.model = model
        self.device = model.device # Assumes model is already on the correct device
        self.num_timesteps_ddpm = num_timesteps_ddpm
        self.image_size_hw = image_size[1:]
        self.img_channels = image_size[0]

        # Setup DDPM noise schedule
        if ddpm_beta_schedule == 'linear':
            beta_t_ddpm = linear_beta_schedule(num_timesteps_ddpm)
        elif ddpm_beta_schedule == 'cosine':
            beta_t_ddpm = cosine_beta_schedule(num_timesteps_ddpm)
        else:
            raise ValueError(f"Unknown beta schedule: {ddpm_beta_schedule}")

        alpha_t_ddpm = 1. - beta_t_ddpm
        self.alpha_bar_t_ddpm = torch.cumprod(alpha_t_ddpm, axis=0).to(self.device) # cumulative products of alphas
        self.beta_t_ddpm = beta_t_ddpm.to(self.device)
        self.alpha_t_ddpm = alpha_t_ddpm.to(self.device)

        # For DDIM sampling if used (not strictly necessary for basic ancestral sampling)
        self.sampling_timesteps_ddpm = sampling_timesteps_ddpm if sampling_timesteps_ddpm is not None else num_timesteps_ddpm
        # assert self.sampling_timesteps_ddpm <= num_timesteps_ddpm # not used in current denoise_1step
        
        # HMC parameters (can be tuned)
        self.hmc_n_leapfrog_steps = hmc_n_leapfrog_steps
        self.hmc_chain_length = hmc_chain_length
        self.hmc_burnin_steps = hmc_burnin_steps # Burn-in for HMC adaptation *within* a Gibbs iter
        self.hmc_adapt_stepsize_iters = hmc_adapt_stepsize_iters # for the very first HMC call

        # Placeholder for lmap (initialize properly before running Gibbs)
        self.lmap_fourier = None # Must be set externally or via a method

        PIX_SIZE_ARCMIN = 8.0
        self.SHAPE, self.WCS = enmap.geometry(pos=(0,0), shape=(self.image_size_hw[0], self.image_size_hw[-1]), res=np.deg2rad(PIX_SIZE_ARCMIN/60.), proj="car")

    def set_lmap_fourier(self, lmap_tensor):
        self.lmap_fourier = lmap_tensor.to(self.device)

    def get_gdiff_loss(self, clean_dust_batch, ddpm_timesteps, phi_cmb_batch=None):
        """
        Calculates the diffusion model training loss.
        clean_dust_batch: [B, C, H, W]
        phi_cmb_batch: [B, 3] tensor of (sigma_CMB, H0, ombh2)
        ddpm_timesteps: [B] tensor of DDPM timesteps
        """
        batch_size = clean_dust_batch.shape[0]
        
        if phi_cmb_batch is None:
            # sample phi_cmb sample between MIN_MAX range | shape (batch_size, 2)
            phi_h0_batch = torch.rand(batch_size, 1, device=self.device) * (H0_PRIOR_MAX - H0_PRIOR_MIN) + H0_PRIOR_MIN
            phi_omb_batch = torch.rand(batch_size, 1, device=self.device) * (OMBH2_PRIOR_MIN - OMBH2_PRIOR_MIN) + OMBH2_PRIOR_MIN

        elif isinstance(phi_cmb_batch[0], float) or isinstance(phi_cmb_batch[0], int):
            phi_h0_batch = phi_cmb_batch[0] * torch.ones(batch_size, 1).to(self.device)
            phi_omb_batch = phi_cmb_batch[1] * torch.ones(batch_size, 1).to(self.device)

        phi_cmb_batch = torch.concatenate([phi_h0_batch, phi_omb_batch], dim=1)

        # 1. Get alpha_bar_t for the sampled DDPM timesteps
        # Squeeze ddpm_timesteps if it's [B,1]
        a_bar_t = extract(self.alpha_bar_t_ddpm, ddpm_timesteps.squeeze(-1) if ddpm_timesteps.ndim > 1 else ddpm_timesteps, clean_dust_batch.shape)

        # 2. Sample standard Gaussian noise for the DDPM forward process
        # ddpm_noise_eps = torch.randn_like(clean_dust_batch, device=self.device)

        ''' ADD CMB gaussian like distribution (noise) '''
        ddpm_noise_eps = get_cmb_noise_batch(phi_cmb_batch, device=self.device)  # Shape: (B, C, H, W)

        assert clean_dust_batch.shape == ddpm_noise_eps.shape 

        # 3. Create z_t (DDPM noised dust map)
        # x_t = sqrt(alpha_hat) * x_0 + sqrt(1-alpha_hat) * eps
        z_t = torch.sqrt(a_bar_t) * clean_dust_batch + torch.sqrt(1. - a_bar_t) * ddpm_noise_eps

        # 4. Get model prediction (predicts the DDPM noise eps)
        # The model is conditioned on ddpm_timesteps and the true phi_cmb_batch
        predicted_ddpm_noise = self.model(z_t.float(), ddpm_timesteps.float(), phi_cmb=phi_cmb_batch)

        # 5. Calculate MSE loss
        loss = nn.functional.mse_loss(predicted_ddpm_noise, ddpm_noise_eps)
        return loss

    def forward(self, clean_dust_img, *args, **kwargs): # For training wrapper
        """
        A forward pass suitable for training.
        clean_dust_img: Batch of clean dust maps [B, C, H, W]
        phi_cmb_params: Batch of corresponding true CMB parameters [B, 3]
        """
        b, c, h, w = clean_dust_img.shape
        # assert (c, h, w) == (self.img_channels, *self.image_size_hw), f'Image size mismatch'
        
        # Sample random DDPM timesteps for this batch
        t_ddpm = torch.randint(0, self.num_timesteps_ddpm, (b,), device=self.device).long()
        
        return self.get_gdiff_loss(clean_dust_img, t_ddpm)

    def get_closest_ddpm_timestep_from_sigma_cmb(self, sigma_cmb_values): # sigma_cmb_values is [B_chains]
        """
        Finds the DDPM timestep t_eff whose DDPM noise level sigma_t_ddpm
        is closest to the provided physical sigma_CMB values.
        This is based on your interpretation for starting ancestral sampling.
        """

        # print('sigma_cmb_values: ', sigma_cmb_values.shape)

        # DDPM noise schedule: sigma_t^2 = beta_t or (1-alpha_bar_t) / alpha_bar_t etc.
        ddpm_noise_levels = torch.sqrt((1. - self.alpha_bar_t_ddpm) / self.alpha_bar_t_ddpm).to(self.device) # Shape [num_timesteps_ddpm]
        
        # Expand dims for broadcasting: [1, T_ddpm] vs [B_chains, 1]
        diffs = torch.abs(ddpm_noise_levels.unsqueeze(0) - sigma_cmb_values.unsqueeze(1)) # [B_chains, T_ddpm]
        closest_ddpm_t_indices = torch.argmin(diffs, dim=1) # [B_chains]

        return closest_ddpm_t_indices # These are the t_eff to start DDPM from
        
        ###################################

        # alpha_bar_t_ddpm = self.alpha_bar_t_ddpm.to(self.device)
        # all_noise_levels = torch.sqrt((1-alpha_bar_t_ddpm)/alpha_bar_t_ddpm).reshape(-1, 1).repeat(1, sigma_cmb_values.shape[0]) #--> (T=#timesteps_cumprod, N=#noise_levels)
        # print('Closest-Timestep: ', all_noise_levels.shape, sigma_cmb_values.shape)
        # closest_timestep = torch.argmin(torch.abs(all_noise_levels - sigma_cmb_values), dim=0)

        # return closest_timestep    
        

    @torch.no_grad()
    def denoise_1step_ancestral(self, z_t, t_ddpm, phi_cmb_cond): # t_ddpm is a scalar tensor
        """ Performs one step of DDPM ancestral sampling. """
        # Ensure t_ddpm is correctly shaped for model and indexing
        t_ddpm_batch = t_ddpm.repeat(z_t.shape[0]) # If z_t is batched

        predicted_ddpm_noise = self.model(z_t.float(), t_ddpm_batch.float(), phi_cmb=phi_cmb_cond)
        
        alpha_t = self.alpha_t_ddpm[t_ddpm]
        alpha_bar_t = self.alpha_bar_t_ddpm[t_ddpm]
        
        # Denoise from z_t to predicted x_0
        # x_0_pred = (z_t - torch.sqrt(1. - alpha_bar_t) * predicted_ddpm_noise) / torch.sqrt(alpha_bar_t)
        # x_0_pred = torch.clamp(x_0_pred, -1., 1.) # Optional clamping if data was in [-1,1]

        # Standard DDPM sampling step (from Ho et al. 2020, Eq. 11 for x_t-1 from x_t)
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mean_prev_t = coeff1 * (z_t - coeff2 * predicted_ddpm_noise)
        
        variance_t = self.beta_t_ddpm[t_ddpm] # sigma_t^2 = beta_t
        
        noise_for_prev_step = torch.randn_like(z_t) if t_ddpm > 0 else torch.zeros_like(z_t)
        z_prev_t = mean_prev_t + torch.sqrt(variance_t) * noise_for_prev_step
        
        return z_prev_t

    @torch.no_grad()
    def sample_dust_posterior(self, y_observed_cmb_corrupted, # The actual observation [B_orig,C,H,W]
                              phi_cmb_current_estimate,     # Current Phi_CMB [B_chains, 3]
                              num_ddpm_ancestral_steps=None): # How many DDPM steps from t_eff
        """
        Samples a dust map x_k ~ q(x | y_observed, Phi_CMB_current)
        using DDPM ancestral sampling, starting from an effective t_eff.
        """
        # y_observed_cmb_corrupted is the "true" observation y = x_dust + eps_CMB
        # For DDPM, the input z_t should be x_0 + artificial_ddpm_noise.
        # Here, y_observed_cmb_corrupted *is* effectively our starting point,
        # considered to be at some t_eff.
        
        sigma_cmb_from_phi = phi_cmb_current_estimate[:, 0] # Extract current sigma_CMB estimate

        # Find the DDPM timestep t_eff that "matches" the current sigma_CMB
        t_eff_indices = self.get_closest_ddpm_timestep_from_sigma_cmb(sigma_cmb_from_phi) # [B_chains]
        
        # The starting "noisy image" for DDPM is the observation y.
        # If y_observed_cmb_corrupted is [B_orig, C, H, W] and phi_cmb_current_estimate is [B_chains, 3],
        # and B_chains = B_orig * num_chains_per_sample, we need to align them.
        # Typically, y_observed will already be repeated for each chain.
        z_t_eff = y_observed_cmb_corrupted.clone() # Shape [B_chains, C, H, W]

        # Determine the number of DDPM steps for each chain
        # This can be fixed (e.g., all run for max(t_eff_indices) steps)
        # or variable (each chain runs for its own t_eff_i steps).
        # For simplicity, let's make all chains run up to max(t_eff_indices)
        # and apply updates conditionally.
        # Or, more simply for ancestral sampling: each chain starts at its t_eff_i
        # and iterates down to 0. The number of steps will vary.
        
        # The paper's code (natural images) uses "mask" to handle variable timesteps in batch.
        # Let's iterate for each chain individually for clarity here, then batch it.
        # This is inefficient but illustrates the per-chain logic.
        # A batched version would run all chains for max(t_eff_indices) steps
        # and use a mask to apply updates only for t <= t_eff_i for chain i.
        
        # Simpler: Loop for the maximum t_eff found across the batch.
        # Inside the loop, only update samples whose t_eff_i is >= current t.
        
        max_t_eff = torch.max(t_eff_indices).item()
        current_z = z_t_eff

        for t_val_ddpm in range(max_t_eff, -1, -1): # from max_t_eff down to 0
            t_tensor = torch.tensor(t_val_ddpm, device=self.device)
            
            # Create a mask for which chains are still active at this t_val_ddpm
            active_mask = (t_eff_indices >= t_val_ddpm).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            if torch.sum(active_mask) > 0: # If any chain is active
                z_prev_t = self.denoise_1step_ancestral(current_z, t_tensor, phi_cmb_cond=phi_cmb_current_estimate[:, :2])
                current_z = active_mask * z_prev_t + (1.0 - active_mask) * current_z # Update only active chains
            
        return current_z # This is the sampled x_k (dust map)

    def run_gibbs_sampler(self,
                            y_observed, # Single observation or batch [B_orig, C, H, W]
                            num_chains_per_gibbs_sample, # Number of parallel MCMC chains for Phi_CMB
                            n_it_gibbs=10,
                            n_it_burnin_gibbs=2,
                            initial_phi_cmb=None, # Optional starting point for Phi_CMB [B_orig*N_chains, 3]
                            hmc_step_size_initial=0.01, # Initial HMC step size
                            hmc_inv_mass_matrix_initial=None, # Initial HMC inv mass matrix
                            adapt_hmc_during_burnin=True,
                            return_chains_history=False):
        """
        Runs the Gibbs sampler for the cosmology problem.
        y_observed: The input mixed map (dust + CMB). Shape [B_orig, C, H, W]
        """
        if self.lmap_fourier is None:
            WCS_COSMO_MAP_TEST = enmap.zeros(self.image_size_hw).wcs
            self.lmap_fourier = get_cosmo_lmap(self.image_size_hw, WCS_COSMO_MAP_TEST, device=self.device)

        device = self.device
        original_batch_size = y_observed.shape[0]
        total_chains = original_batch_size * num_chains_per_gibbs_sample

        # Repeat y_observed for each chain
        y_repeated = y_observed.repeat_interleave(num_chains_per_gibbs_sample, dim=0).to(device) # [B_total, C,H,W]

        # Initialize Phi_CMB = (sigma_CMB, H0, ombh2)
        if initial_phi_cmb is not None:
            phi_cmb_current = initial_phi_cmb.to(device)
            assert phi_cmb_current.shape == (total_chains, 3)
        else:
            # Sample from prior or a smarter initialization
            s_init = torch.rand(1, 1, device=device) * (SIGMA_CMB_PRIOR_MAX - SIGMA_CMB_PRIOR_MIN) + SIGMA_CMB_PRIOR_MIN
            s_init = s_init.repeat(total_chains, 1)

            h_init = torch.rand(1, 1, device=device) * (H0_PRIOR_MAX - H0_PRIOR_MIN) + H0_PRIOR_MIN
            h_init = h_init.repeat(total_chains, 1)

            o_init = torch.rand(1, 1, device=device) * (OMBH2_PRIOR_MIN - OMBH2_PRIOR_MIN) + OMBH2_PRIOR_MIN
            o_init = o_init.repeat(total_chains, 1)

            phi_cmb_current = torch.cat([s_init, h_init, o_init], dim=-1) # [B_total, 3]

        # print(phi_cmb_current.shape)
        phi_cmb_history = []
        x_dust_history = []
        
        # HMC parameters (will be adapted if adapt_hmc_during_burnin is True)
        current_hmc_step_size = hmc_step_size_initial
        current_hmc_inv_mass_matrix = hmc_inv_mass_matrix_initial
        if current_hmc_inv_mass_matrix is None: # Default to identity if not provided
             current_hmc_inv_mass_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(total_chains, 1, 1) # Diagonal or per chain

        phi_min_bounds, phi_max_bounds = get_phi_cmb_parameter_bounds(device=device)


        for gibbs_iter in tqdm(range(n_it_gibbs + n_it_burnin_gibbs), desc="Gibbs Iterations"):
            # --- Step 1: Sample Dust Map x_k ~ q(x | y, Phi_CMB_{k-1}) ---
            # This uses the DDPM ancestral sampling starting from t_eff based on sigma_CMB
            # The y_repeated is the observation.
            sampled_x_dust = self.sample_dust_posterior(
                y_observed_cmb_corrupted=y_repeated,
                phi_cmb_current_estimate=phi_cmb_current
            ) # Returns [B_total, C, H, W]

            # --- Step 2: Estimate CMB residual epsilon_{k-1} = y - x_k ---
            estimated_cmb_residual = y_repeated - sampled_x_dust # [B_total, C, H, W]

            # --- Step 3: Sample Phi_CMB_k ~ q(Phi | epsilon_{k-1}) using HMC ---
            def hmc_log_posterior_fn(phi_cmb_trial): # phi_cmb_trial is [B_total, 2]
                log_p = log_prior_phi_cmb(phi_cmb_trial, (phi_min_bounds, phi_max_bounds))
                # Only compute likelihood for valid priors to save computation
                
                valid_prior_mask = ~torch.isinf(log_p)
                if torch.any(valid_prior_mask):
                
                    log_l = torch.zeros_like(log_p)
                    
                    computed_log_likelihoods = log_likelihood_cmb_phi(
                        phi_cmb_trial[valid_prior_mask],
                        # phi_cmb_trial,
                        estimated_cmb_residual[valid_prior_mask], # Pass only relevant residuals
                        self.WCS,
                        self.lmap_fourier, 
                        int(1.5 * self.image_size_hw[0])
                    )

                    log_l[valid_prior_mask] = computed_log_likelihoods ## inf/null values padded by 0
                    log_p = log_p + log_l
                
                return log_p

            def hmc_gradient_log_posterior_fn(phi_cmb_trial_grad):
                phi_cmb_clone = phi_cmb_trial_grad.clone().requires_grad_(True)
                logp_val = hmc_log_posterior_fn(phi_cmb_clone)
                
                # Sum is needed if logp_val is not scalar per batch item, but it should be.
                # Handle -inf by setting grad to 0 for those, or use a large negative number.
                valid_grads_mask = ~torch.isinf(logp_val)
                grad_phi = torch.zeros_like(phi_cmb_clone)

                if torch.any(valid_grads_mask):
                    # Compute gradients only for valid logp values
                    # Need to sum valid logp values to get a scalar for autograd, then un-sum grads.
                    # This is tricky for batched autograd if logp_val is not all valid.
                    # A common way is to compute grads for each item if autograd over batch is problematic.
                    # For now, assume autograd handles batching correctly or sum and scale.
                    
                    # Simpler: if any logp_val is -inf, the gradient for that chain is effectively zero for other params
                    # or should be handled by boundary conditions in HMC. Let's try direct grad.
                    # If logp_val contains -inf, autograd might error or give NaNs.
                    # We should ensure inputs to log_likelihood are only for valid priors.
                    
                    # This is a common pain point with HMC and hard boundaries.
                    # The log_posterior_fn already returns -inf.
                    # HMC should ideally handle this by rejecting steps outside bounds.
                    
                    # Compute grad only for elements where logp_val is finite
                    if torch.any(valid_grads_mask):
                        # This is the tricky part with batched autograd and conditional computation.
                        # One way: loop (inefficient) or use more advanced autograd tricks.
                        # For now, let's assume our HMC handles boundary rejections.
                        # The HMC log_prob_fn will return -inf for out-of-bounds, leading to rejection.
                        # The gradient is only "needed" for points where log_prob is finite.
                        grad_phi[valid_grads_mask] = torch.autograd.grad(
                            outputs=logp_val[valid_grads_mask].sum(), # Sum to make scalar for grad
                            inputs=phi_cmb_clone, # Grad w.r.t. all inputs
                            create_graph=False, # For HMC, no higher order derivs needed
                            allow_unused=True # If some inputs don't affect output
                        )[0][valid_grads_mask] # Extract grad for valid inputs

                return grad_phi.detach() # Detach as HMC doesn't need graph for phi_new

            # Determine if HMC step size adaptation is needed
            adapt_step_size, adapt_mass_matrix = False, False
            current_n_adapt_hmc = 0
            if adapt_hmc_during_burnin and gibbs_iter < n_it_burnin_gibbs : # Adapt during Gibbs burn-in
                 adapt_step_size, adapt_mass_matrix = True, True
                 current_n_adapt_hmc = self.hmc_adapt_stepsize_iters // n_it_burnin_gibbs # Distribute adaptation
                 if gibbs_iter == 0: current_n_adapt_hmc = self.hmc_adapt_stepsize_iters # Full adapt on first iter

            # Run HMC (using your sample_hmc_v2 or similar)
            # We need to ensure sample_hmc_v2 can take batched phi_init, step_size, inv_mass_matrix
            # and that log_prob_fn and log_grad handle batches.
            phi_cmb_new, hmc_step_size_new, hmc_inv_mass_new, _ = sample_hmc_cosmo(
                log_prob_fn=hmc_log_posterior_fn,
                log_grad_fn=hmc_gradient_log_posterior_fn,
                phi_init=phi_cmb_current.detach(), # Start HMC from current Phi_CMB
                mass_matrix_M_input=None if current_hmc_inv_mass_matrix is None else torch.linalg.inv(current_hmc_inv_mass_matrix.mean(0)).unsqueeze(0).repeat(total_chains, 1, 1), # HMC_v2 takes M
                step_size_initial=current_hmc_step_size,
                n_leapfrog_steps=self.hmc_n_leapfrog_steps,
                num_samples_chain=self.hmc_chain_length, # Produce 1 sample per Gibbs iter
                num_burnin_steps_hmc=self.hmc_burnin_steps,  # HMC internal burn-in (usually 0 after initial adaptation)
                adapt_step_size=adapt_step_size,
                adapt_mass_matrix=adapt_mass_matrix,
                num_adapt_steps_total=current_n_adapt_hmc,
                phi_min_bounds=phi_min_bounds, # Pass bounds to HMC
                phi_max_bounds=phi_max_bounds
            )
            
            ''' Posterior-Distribution Problem'''
            # # sample_hmc_v2 returns the last sample.
            # # If chain_length > 1, phi_cmb_new would be [B_total, chain_length_hmc, 3]
            # # We want the last sample from the HMC chain.
            # if phi_cmb_new.ndim == 3 and phi_cmb_new.shape[1] == self.hmc_chain_length:
            #     phi_cmb_current = phi_cmb_new[:, -1, :].detach() # Take last HMC sample
            # elif phi_cmb_new.ndim == 2: # if chain_length was 1
            #     phi_cmb_current = phi_cmb_new.detach()
            # else:
            #     raise ValueError("Unexpected shape from HMC sampler for phi_cmb_new")

            # print('phi_cmb_current: ', phi_cmb_current.shape)
            # print('phi_cmb_new', phi_cmb_new.shape)
       
            phi_cmb_current = phi_cmb_new.detach()

            # Update HMC step size and inv mass matrix if they were adapted
            if adapt_step_size and adapt_mass_matrix:
                current_hmc_step_size = hmc_step_size_new
                current_hmc_inv_mass_matrix = hmc_inv_mass_new # This is M_inv from hmc_v2 if M was adapted

            # Store results if past Gibbs burn-in
            if gibbs_iter >= n_it_burnin_gibbs:
                phi_cmb_history.append(phi_cmb_current.reshape(original_batch_size, num_chains_per_gibbs_sample, 3))
                x_dust_history.append(sampled_x_dust.reshape(original_batch_size, num_chains_per_gibbs_sample, self.img_channels, *self.image_size_hw))

        if not phi_cmb_history: # If no samples collected (e.g. n_it_gibbs was 0)
            # Return current state as the only sample
             phi_cmb_history.append(phi_cmb_current.reshape(original_batch_size, num_chains_per_gibbs_sample, 3))
             x_dust_history.append(sampled_x_dust.reshape(original_batch_size, num_chains_per_gibbs_sample, self.img_channels, *self.image_size_hw))


        phi_cmb_out = torch.stack(phi_cmb_history, dim=2).detach().cpu() # [B_orig, N_chains, N_gibbs_samples, 3]
        x_dust_out = torch.stack(x_dust_history, dim=2).detach().cpu()   # [B_orig, N_chains, N_gibbs_samples, C,H,W]

        if return_chains_history:
            return phi_cmb_out, x_dust_out
        else:
            # Return the mean of the chains after burn-in (posterior mean estimate)
            return phi_cmb_out.mean(dim=(1,2)), x_dust_out.mean(dim=(1,2))


    def blind_posterior_mean(self, y_observed, num_chains_per_gibbs_sample=5,
                                   n_it_gibbs=10, n_it_burnin_gibbs=5,
                                   return_full_posterior_chains=False, avg_pmean=2, **hmc_kwargs):
        """
        Performs blind denoising for cosmology to get posterior mean estimates.
        y_observed: [B, C, H, W]
        """
        phi_chains, x_chains = self.run_gibbs_sampler(
            y_observed,
            num_chains_per_gibbs_sample=num_chains_per_gibbs_sample,
            n_it_gibbs=n_it_gibbs,
            n_it_burnin_gibbs=n_it_burnin_gibbs,
            return_chains_history=True, # Get all history for averaging
            **hmc_kwargs
        )
        # phi_chains: [B, N_chains_gibbs, N_gibbs_samples, 3] -> [N_chains_gibbs, B, avg_pmean, 3]
        # x_chains:   [B, N_chains_gibbs, N_gibbs_samples, C, H, W] -> [N_chains_gibbs, B, avg_pmean, channels, image_h, image_w]

        # print(phi_chains.shape, x_chains.shape)
        phi_chains = phi_chains[:, : , -avg_pmean:, :].reshape(num_chains_per_gibbs_sample, -1, avg_pmean, 3)
        x_chains = x_chains[:, :, -avg_pmean:, :, :, :].reshape(num_chains_per_gibbs_sample, -1, avg_pmean, self.img_channels, *self.image_size_hw)

        if return_full_posterior_chains:
            return phi_chains, x_chains
        else:
            # Posterior mean over chains and Gibbs samples
            phi_posterior_mean = phi_chains.mean(dim=(0, 2)) # Mean over N_chains_gibbs and N_gibbs_samples
            x_dust_posterior_mean = x_chains.mean(dim=(0, 2))
            return phi_posterior_mean, x_dust_posterior_mean
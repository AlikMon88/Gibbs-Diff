# import os
# import numpy as np
# import math
# import torch
# from autograd import grad
# import torch.nn as nn
# import torch.nn.functional as F

# sigma_min, sigma_max = 0.04, 0.4

# def get_phi_all_bounds(phi_min=-1.0, phi_max = 1.0, sigma_min = 0.04, sigma_max = 0.4, device = 'cpu'):
#     phi_min = torch.tensor([phi_min], device=device)
#     phi_max = torch.tensor([phi_max], device=device)
#     sigma_min = torch.tensor([sigma_min], device=device)
#     sigma_max = torch.tensor([sigma_max], device=device)

#     phi_min_all = torch.cat([phi_min, sigma_min])
#     phi_max_all = torch.cat([phi_max, sigma_max])
#     return phi_min_all, phi_max_all

# def get_noise_estimate_2d(y, sigma_min, sigma_max):
#     y_std = y.std()
#     sigma_est = 1.15 * y_std - 0.17  # heuristic from Imagenet
#     sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
#     return sigma_est.unsqueeze(0)

# def high_pass_filter(y, kernel_size=31):
#     # Create a moving average (low-pass) kernel
#     kernel = torch.ones(kernel_size) / kernel_size
#     kernel = kernel.to(y.device).unsqueeze(0).unsqueeze(0)  # (1, 1, K)

#     low_pass = F.conv1d(y, kernel, padding=kernel_size // 2)
#     high_pass = y - low_pass
#     return high_pass.squeeze()

# def get_noise_estimate_1d(y, sigma_min, sigma_max, kernel_size=31):
#     high_freq_part = high_pass_filter(y, kernel_size)
#     sigma_est = high_freq_part.std()
#     sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
#     return sigma_est.unsqueeze(0)

# ## standardize = [-1, 1] -> [0, 1] ## for HMC-sampling
# def normalize_phi(phi, phi_max=1.0, phi_min=-1.0, mode='compact'):
#     ret = (phi - phi_min) / (phi_max - phi_min)
#     return ret

# ## de-standardize = [0, 1] -> [-1, 1]
# def unnormalize_phi(phi, phi_max=1.0, phi_min=-1.0, mode='compact'):
#     ret = phi * (phi_max - phi_min) + phi_min
#     return ret

# def sample_phi_prior(n, phi_min=-1.0, phi_max=1.0, norm_mode='compact'):
#     phi = torch.rand(n) * (phi_max - phi_min) + phi_min
#     return normalize_phi(phi)


# def log_prior_phi(phi, norm_mode="compact"):
#     # phi is (b, 2)

#     # in_bounds = torch.logical_and(phi >= 0.0, phi <= 1.0)  # (b, 2)
#     # all_in_bounds = torch.all(in_bounds, dim=-1).float()   # (b,)
#     # logp = torch.log(all_in_bounds + 1e-30)  # add epsilon to avoid log(0)
    
#     logp = torch.log(torch.logical_and(phi >= 0.0, phi <= 1.0).float())  # (b, 2)
#     for i in range(1, phi.shape[-1]):
#         logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())

#     return logp


# def log_prior_phi_sigma(phi, sigma, sigma_min=0.04, sigma_max=0.4, norm_mode="compact"):
#     # phi: (b, 2), sigma: (b,)

#     # in_bounds_phi = torch.logical_and(phi >= 0.0, phi <= 1.0)  # (b, 2)
#     # valid_phi = torch.all(in_bounds_phi, dim=-1).float()       # (b,)
#     # logp_phi = torch.log(valid_phi + 1e-30)                    # (b,)

#     # in_bounds_sigma = torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float()  # (b,)
#     # logp_sigma = torch.log(in_bounds_sigma + 1e-30)             # (b,)

#     # logp = logp_phi + logp_sigma
#     # print('log_prior_phi_sigma: ', logp)

    
#     logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float()) #gives either 0 or -inf
#     for i in range(1, phi.shape[-1]):
#         logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())
    
#     logp += torch.log(torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float()).squeeze(0)
    
#     print('log_prior_phi_sigma | (phi, sigma): ', phi, sigma)
#     print('log_prior_phi_sigma: ', logp)
         
#     return logp


# def log_likelihood_eps_phi(phi, eps, ps_model):
    
#     ps = ps_model(phi)  # shape: same as eps (excluding batch)

#     if eps.ndim == 3:  # 1D case: (B, dim, N)
#         eps_dim = eps.shape[-1]
#         xf = torch.fft.fft(eps)
#         term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
#         term_logdet = -0.5 * torch.sum(torch.log(ps), dim=-1)
#         term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / ps, dim=-1) / eps_dim

#     elif eps.ndim == 4:  # 2D case: (B, C, H, W)
#         H, W = eps.shape[-2], eps.shape[-1]
#         eps_dim = H * W
#         xf = torch.fft.fft2(eps)
#         term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
#         term_logdet = -0.5 * torch.sum(torch.log(ps), dim=(-2, -1))  # sum over H, W
#         term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / ps, dim=(-2, -1)) / eps_dim

#     else:
#         raise ValueError("eps must be 2D (1D case) or 4D (image case)")

#     log_likelihood = term_pi + term_logdet + term_x  # (b, dim)
#     log_likelihood = log_likelihood.sum(dim=1)       # (b,) --> sum over the channel dimm
#     return log_likelihood


# def log_likelihood_eps_phi_sigma(phi, sigma, eps, ps_model):
    
#     print('before-ps-model-phi: ', phi)
#     ps = ps_model(phi)

#     if eps.ndim == 3:  # 1D case
#         eps_dim = eps.shape[-1]
#         xf = torch.fft.fft(eps)
#         sigma = sigma.view(-1, 1) if sigma.ndim == 1 else sigma  # (B, 1)
        
#         print('scaled-sigma: ', sigma)
#         print('ps: ', ps)
        
#         scaled_ps = sigma**2 * ps
#         term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
#         term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=-1)
        
#         print('term_pi: ', term_pi)
#         print('scaled_ps: ', scaled_ps)
#         print('term_logdet: ', term_logdet)
        
#         term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=-1) / eps_dim

#     elif eps.ndim == 4:  # 2D image case
#         H, W = eps.shape[-2], eps.shape[-1]
#         eps_dim = H * W
#         xf = torch.fft.fft2(eps)
#         sigma = sigma.view(-1, 1, 1, 1) if sigma.ndim == 1 else sigma
#         scaled_ps = sigma**2 * ps
#         term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
#         term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=(-2, -1))
#         term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=(-2, -1)) / eps_dim

#     else:
#         raise ValueError("eps must be 2D (1D case) or 4D (image case)")
    
#     log_likelihood = term_pi + term_logdet + term_x  # (b, dim)
#     log_likelihood = log_likelihood.sum(dim=1)       # (b,) --> sum over the channel dim
    
#     print('log_likelihood_eps_phi_sigma | (phi, sigma): ', phi, sigma)
#     print('log_likelihood_eps_phi_sigma: ', log_likelihood)
    
#     return log_likelihood


# ## We gradient-trace the ColoredPoweredSpectrum
# class ColoredPowerSpectrum2D(nn.Module):
#     def __init__(self, norm_input_phi='compact', shape=(3, 64, 64), device='cpu', sigma_eps=1e-6):
#         super().__init__()
#         assert len(shape) == 3  # (C, H, W)
#         self.norm_input_phi = norm_input_phi
#         self.device = device

#         dim, H, W = shape

#         # Create 2D isotropic wavenumber grid
#         ky = torch.fft.fftfreq(H, d=1.0).to(torch.float32).to(device)  # (H,)
#         kx = torch.fft.fftfreq(W, d=1.0).to(torch.float32).to(device)  # (W,)
#         kx, ky = torch.meshgrid(kx, ky, indexing='ij')  # (W, H)
#         k_squared = kx ** 2 + ky ** 2
#         k_magnitude = torch.sqrt(k_squared).unsqueeze(0).unsqueeze(0)  # shape: (1, C, H, W)
#         k_magnitude[:, :, 0, 0] = sigma_eps  # avoid division by zero at (0, 0)

#         self.S = k_magnitude  # shape: (1, C, H, W)

#     def forward(self, phi):
#         '''
#         phi: tensor of shape (batch_size, dim) or (batch_size, 1)
#              controls the spectral slope alpha
#         '''
#         phi = unnormalize_phi(phi, mode=self.norm_input_phi)

#         batch_size, dim = phi.shape[0], self.S.shape[1]
#         S = self.S.repeat(batch_size, dim, 1, 1)        # (batch_size, dim, H, W)
#         S = S ** phi.reshape(-1, 1, 1, 1)               # (batch_size, dim, H, W)
#         S = S / S.mean(dim=(2, 3), keepdim=True)        # Normalize spectrum per sample
#         return S

# class ColoredPowerSpectrum1D(nn.Module):
#     def __init__(self, norm_input_phi='compact', shape=(1, 100), device='cpu'):
#         super().__init__()
#         shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
#         assert len(shape) == 2  # (dim, seq_len)
        
#         dim, N = shape
#         self.norm_input_phi = norm_input_phi

#         # Create isotropic wavenumber vector for 1D
#         wn = torch.fft.fftfreq(N).to(torch.float32)  # shape: (N,)
#         wn = wn.to(device)

#         # Compute wavenumber magnitude (squared)
#         S = wn.pow(2).reshape(1, 1, N)  # (1, 1, N)
#         S = torch.sqrt(S)
#         S[:, :, 0] = sigma_eps  # avoid division by zero for k=0

#         self.S = S  # shape: (1, 1, N)

#     def forward(self, phi):
#         '''
#         phi: tensor of shape (batch_size, dim) or (batch_size, 1)
#              controls the spectral slope alpha
#         '''
        
#         print('before-unnormalize_phi: ', phi)
        
#         phi = unnormalize_phi(phi, mode=self.norm_input_phi)
        
#         print('unnormalize_phi: ', phi)

#         batch_size, dim = phi.shape[0], self.S.shape[1]
#         S = self.S.repeat(batch_size, dim, 1)  # shape: (batch_size, dim, N)
#         S = S ** phi.reshape(-1, 1, 1)         # shape: (batch_size, dim, N)
#         S = S / S.mean(dim=2, keepdim=True)    # Normalize spectrum
#         # S = S / S.mean(dim=1, keepdim=True)    # Normalize spectrum
        
#         print('PSpectrum: ', S.shape)
        
#         return S

# #### -------------------------------------------------------------
# #### ------------------------- HMC (Utils) -----------------------
# #### -------------------------------------------------------------


# sigma_eps = 1e-6

# def reflect_boundary(q, p, p_nxt, phi_min_norm, phi_max_norm):
#     p_ret = p_nxt
#     for i in range(q.shape[-1]):  # phi and sigma
#         crossed_min_boundary = q[..., i] < phi_min_norm[i]
#         crossed_max_boundary = q[..., i] > phi_max_norm[i]
#         p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
#         p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]
#     return p_ret

# def compute_inverse_mass_matrix(mass_matrix):
#     return torch.linalg.inv(mass_matrix)

# def kinetic_energy(p, inv_mass_matrix):
#     return 0.5 * torch.sum(p * (inv_mass_matrix @ p.unsqueeze(-1)).squeeze(-1), dim=1)

# def hamiltonian(q, p, log_prob_fn, inv_mass_matrix):
    
#     print('hamiltonian(q, p): ', q, p)
#     potential_energy = -log_prob_fn(q)
    
#     print('potential-energy: ', potential_energy)
    
#     return potential_energy + kinetic_energy(p, inv_mass_matrix)

# def leapfrog(q, p, step_size, n_steps, log_prob_fn, log_grad, inv_mass_matrix):
    
#     q = q.clone()
#     p = p.clone()

#     print('leap-frog-grad-I: ', p, q)

#     grad_ = log_grad(q)
    
    
#     p -= 0.5 * step_size * grad_
#     # print('p_ini: ', p)

#     for _ in range(n_steps):
#         q += step_size * (p @ inv_mass_matrix)
        
#         print('leap-frog-grad-II: ', p, q)
#         grad = log_grad(q)
#         p -= step_size * grad

#     print('leap-frog-grad-III: ', p, q)
#     p -= 0.5 * step_size * log_grad(q)
#     p = -p  # negate for symmetry

#     print('leapgrog-after: ', p, q)
    
#     return q, p

# class DualAveragingStepSize:
#     def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10, kappa=0.75):
#         self.mu = np.log(10 * initial_step_size)
#         self.log_step = np.log(initial_step_size)
#         self.h_bar = 1-6
#         self.step_bar = 1-6
#         self.t = 1
#         self.target_accept = target_accept
#         self.gamma = gamma
#         self.t0 = t0
#         self.kappa = kappa

#     def update(self, accept_prob):
        
#         print('accept_prob: ', accept_prob)
        
#         self.t += 1
#         eta = 1.0 / (self.t + self.t0)
#         self.h_bar = (1 - eta) * self.h_bar + eta * (self.target_accept - accept_prob)
#         log_step = self.mu - (np.sqrt(self.t) / self.gamma) * self.h_bar
#         self.log_step = log_step
#         self.step_bar = np.exp((self.t ** -self.kappa) * log_step + (1 - self.t ** -self.kappa) * np.log(self.step_bar or np.exp(log_step)))
        
#         print('log_step: ', log_step)
        
#         return np.exp(log_step)

# def sample_hmc(log_prob_fn, log_grad, phi_init, step_size=0.1, n_leapfrog_steps=50, chain_length=100, burnin_steps=20, inv_mass_matrix=None, adapt=True, n_adapt=100, phi_min_norm=None, phi_max_norm=None):
    
#     q = phi_init.clone()

#     batch_size, dim = q.shape

#     if inv_mass_matrix is None:
#         mass_matrix = torch.eye(dim)
#         inv_mass_matrix = compute_inverse_mass_matrix(mass_matrix)

#     step_size_adapter = DualAveragingStepSize(step_size)

#     accept_prob_list = []
#     for i in range(1, chain_length + burnin_steps + 1):
#         p = torch.randn_like(q)
#         q_new, p_new = leapfrog(q, p, step_size, n_leapfrog_steps, log_prob_fn, log_grad, inv_mass_matrix)

#         # print(q.shape, p.shape)
#         if phi_min_norm is not None and phi_max_norm is not None:
            
#             print(' --- Boundary-Reflection ---')
            
#             p_new = reflect_boundary(q, p, p_new, phi_min_norm, phi_max_norm)
            
#             print('p_new: ', p_new)
        
#         print('q, p: ', q, p)
#         print('q_new, p_new: ', q_new, p_new)

#         H_old = hamiltonian(q, p, log_prob_fn, inv_mass_matrix)
#         H_new = hamiltonian(q_new, p_new, log_prob_fn, inv_mass_matrix)

#         print('Hamiltonian:')
#         print(H_old, H_new)

#         accept_prob = torch.exp(torch.clamp(H_old - H_new, max=0.0)).reshape(-1, 1)
#         accept_prob_list.append(accept_prob.mean(dim=0).item())

#         accept = torch.rand(q.shape) < accept_prob
        
#         q[accept] = q_new[accept]

#         if adapt and i <= n_adapt:
#             step_size = step_size_adapter.update(accept_prob.mean().item())


#     mean_accept_prob = np.mean(np.array(accept_prob_list), axis=0)
#     # return torch.stack(samples[burnin_steps:], dim=1), step_size, inv_mass_matrix if adapt else torch.stack(samples[burnin_steps:], dim=1)

#     if adapt:
#         return q, step_size, inv_mass_matrix, mean_accept_prob
#     else: 
#         return q, mean_accept_prob

# if __name__ == '__main__':

#     phi = torch.randn(1, 2).reshape(1, 2)
#     epsilon = torch.randn(1, 1, 100).reshape(1, 1, 100)
    
#     print('phi: ', phi, phi.shape)
#     print(epsilon.shape)

#     def log_posterior(phi_):
#         return log_likelihood_eps_phi_sigma(phi_, epsilon) + log_prior_phi_sigma(phi_)

#     phi, _, _ = sample_hmc(log_prob_fn=log_posterior, phi_init=phi)

#     print('sampling successful!')


### --------------- CLEANED ----------------------

import os
import numpy as np
import math
import torch
from autograd import grad
import torch.nn as nn
import torch.nn.functional as F

# --- Constants and Configuration ---
# Physical range for the spectral index phi
PHI_MIN = -1.0
PHI_MAX = 1.0

# Prior range for the noise amplitude sigma
SIGMA_MIN = 1e-3
SIGMA_MAX = 1.0

# Small constant for numerical stability
EPSILON_CONST = 1e-6

# --- Normalization Utilities ---

def get_phi_all_bounds(phi_min=PHI_MIN, phi_max=PHI_MAX, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, device = 'cpu'):
    phi_min = torch.tensor([phi_min], device=device)
    phi_max = torch.tensor([phi_max], device=device)
    sigma_min = torch.tensor([sigma_min], device=device)
    sigma_max = torch.tensor([sigma_max], device=device)

    phi_min_all = torch.cat([phi_min, sigma_min])
    phi_max_all = torch.cat([phi_max, sigma_max])
    return phi_min_all, phi_max_all

def get_noise_estimate_2d(y, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MIN):
    y_std = y.std()
    sigma_est = 1.15 * y_std - 0.17  # heuristic from Imagenet
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est.unsqueeze(0)

def high_pass_filter(y, kernel_size=31):
    # Create a moving average (low-pass) kernel
    kernel = torch.ones(kernel_size) / kernel_size
    kernel = kernel.to(y.device).unsqueeze(0).unsqueeze(0)  # (1, 1, K)

    low_pass = F.conv1d(y, kernel, padding=kernel_size // 2)
    high_pass = y - low_pass
    return high_pass.squeeze()

def get_noise_estimate_1d(y, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MIN, kernel_size=31):
    high_freq_part = high_pass_filter(y, kernel_size)
    sigma_est = high_freq_part.std()
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est.unsqueeze(0)

def sample_phi_prior(n, phi_min=PHI_MIN, phi_max=PHI_MAX):
    phi = torch.rand(n) * (phi_max - phi_min) + phi_min
    return normalize_phi(phi)

def normalize_phi(phi, phi_max=PHI_MAX, phi_min=PHI_MIN):
    """Maps a physical phi from [phi_min, phi_max] to the internal [0, 1] range."""
    return (phi - phi_min) / (phi_max - phi_min)

def unnormalize_phi(phi_normalized, phi_max=PHI_MAX, phi_min=PHI_MIN):
    """Maps an internal phi from [0, 1] back to its physical [phi_min, phi_max] range."""
    return phi_normalized * (phi_max - phi_min) + phi_min

# --- Likelihood and Prior Functions ---

def log_prior_phi_sigma(phi_all, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX):
    
    # Unpack parameters
    phi = phi_all[:, 0].unsqueeze(-1) # Shape [B, 1]
    sigma = phi_all[:, 1].unsqueeze(-1) # Shape [B, 1]

    logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float()) #gives either 0 or -inf
    for i in range(1, phi.shape[-1]):
        logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())

    logp += torch.log(torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float()).squeeze(-1)
    return logp


def log_likelihood_eps_phi_sigma(phi_all, eps, ps_model):
    
    # Unpack parameters
    phi = phi_all[:, 0].unsqueeze(-1) # Shape [B, 1]
    sigma = phi_all[:, 1].unsqueeze(-1)         # Shape [B, 1]
    
    ps = ps_model(phi)

    if eps.ndim == 3:  # 1D case
        eps_dim = eps.shape[-1]
        xf = torch.fft.fft(eps)
        sigma = sigma.view(-1, 1) if sigma.ndim == 1 else sigma  # (B, 1)
        scaled_ps = sigma**2 * ps
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=(-1, -2))
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=(-1, -2)) / eps_dim

    elif eps.ndim == 4:  # 2D image case
        H, W = eps.shape[-2], eps.shape[-1]
        eps_dim = H * W
        xf = torch.fft.fft2(eps)
        sigma = sigma.view(-1, 1, 1, 1) if sigma.ndim == 1 else sigma
        scaled_ps = sigma**2 * ps
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=(-1, -2, -3))
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=(-1, -2, -3)) / eps_dim

    else:
        raise ValueError("eps must be 2D (1D case) or 4D (image case)")
    
    log_likelihood = term_pi + term_logdet + term_x  # (b, dim)
    # log_likelihood = log_likelihood.sum(dim=1)       # (b,) --> sum over the channel dim
    return log_likelihood


# --- Power Spectrum Models ---

class ColoredPowerSpectrum1D(nn.Module):
    def __init__(self, shape=(1, 100), device='cpu'):
        super().__init__()
        assert len(shape) == 2, "(dim, seq_len)"
        dim, N = shape
        
        wn = torch.fft.fftfreq(N, d=1.0, device=device, dtype=torch.float32)
        k_magnitude = torch.sqrt(wn.pow(2)).reshape(1, 1, N)
        k_magnitude[:, :, 0] = EPSILON_CONST # Avoid k=0 issues
        self.register_buffer('S', k_magnitude) # Use register_buffer for non-parameter tensors

    def forward(self, phi_normalized):
        phi_physical = unnormalize_phi(phi_normalized) # Unnormalize to physical range
        
        # Broadcasting handles batch size
        # S is [1,1,N], phi_physical is [B,1] -> reshaped to [B,1,1]
        ps = self.S ** phi_physical.reshape(-1, 1, 1)
        # Normalize each power spectrum in the batch to have a mean of 1
        ps = ps / ps.mean(dim=-1, keepdim=True)
        return ps

# (ColoredPowerSpectrum2D would be analogous)

class ColoredPowerSpectrum2D(nn.Module):
    def __init__(self, norm_input_phi='compact', shape=(3, 64, 64), device='cpu', sigma_eps=1e-6):
        super().__init__()
        assert len(shape) == 3  # (C, H, W)
        self.norm_input_phi = norm_input_phi
        self.device = device

        dim, H, W = shape

        # Create 2D isotropic wavenumber grid
        ky = torch.fft.fftfreq(H, d=1.0).to(torch.float32).to(device)  # (H,)
        kx = torch.fft.fftfreq(W, d=1.0).to(torch.float32).to(device)  # (W,)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')  # (W, H)
        k_squared = kx ** 2 + ky ** 2
        k_magnitude = torch.sqrt(k_squared).unsqueeze(0).unsqueeze(0)  # shape: (1, C, H, W)
        k_magnitude[:, :, 0, 0] = sigma_eps  # avoid division by zero at (0, 0)

        self.S = k_magnitude  # shape: (1, C, H, W)

    def forward(self, phi):
        '''
        phi: tensor of shape (batch_size, dim) or (batch_size, 1)
             controls the spectral slope alpha
        '''
        phi = unnormalize_phi(phi, mode=self.norm_input_phi)

        batch_size, dim = phi.shape[0], self.S.shape[1]
        S = self.S.repeat(batch_size, dim, 1, 1)        # (batch_size, dim, H, W)
        S = S ** phi.reshape(-1, 1, 1, 1)               # (batch_size, dim, H, W)
        S = S / S.mean(dim=(2, 3), keepdim=True)        # Normalize spectrum per sample
        return S


# --- HMC Sampler and Utilities ---
    
class DualAveragingStepSize():
    """ Dual averaging step size adaptation (Nesterov 2009). """

    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
        """        
        Args:
            initial_step_size (torch.Tensor): Initial step size.
            target_accept (float, optional): Target Metropolis acceptance rate. Must be between 0 and 1. Defaults to 0.65.
            gamma (float, optional): Adaptation regularization scale. Defaults to 0.05.
            t0 (float, optional): Adaptation iteration offset. Defaults to 10.0.
            kappa (float, optional): Adaptation relaxation exponent. Defaults to 0.75.
            nadapt (int, optional): _description_. Defaults to 0.    
        """
        
        self.initial_step_size = initial_step_size 
        self.mu = torch.log(initial_step_size) # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma * 2 #parameter to tune
        self.t = t0
        self.kappa = kappa
        self.error_sum = torch.zeros_like(self.initial_step_size)
        self.log_averaged_step = torch.zeros_like(self.initial_step_size)
        
    def update(self, p_accept):
        
        p_accept[p_accept > 1] = 1.
        p_accept[torch.isnan(p_accept)] = 0.
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return torch.exp(log_step)

    def get_final_step_size(self):
        return torch.exp(self.log_averaged_step)
    
    
def reflect_boundary(q, p, p_nxt, phi_min_norm, phi_max_norm):
    p_ret = p_nxt
    for i in range(q.shape[-1]):  # phi and sigma
        crossed_min_boundary = q[..., i] < phi_min_norm[i]
        crossed_max_boundary = q[..., i] > phi_max_norm[i]
        p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
        p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]
    return p_ret

def leapfrog(q_curr, p_curr, step_size, n_steps, log_grad_fn, inv_mass_matrix,
             q_min_bounds=None, q_max_bounds=None):

    q = q_curr.clone()
    p = p_curr.clone()

    # Step size should be a scalar for all chains or a tensor of shape [B]
    # We'll make it [B, 1] for broadcasting
    step_size_t = torch.tensor(step_size, device=q.device, dtype=q.dtype).view(-1, 1)
    if step_size_t.shape[0] == 1:
        step_size_t = step_size_t.expand(q.shape[0], 1)

    # Half step for momentum
    grad = log_grad_fn(q)
    p = p - 0.5 * step_size_t * -grad # Potential V = -logP, so grad_V = -grad_logP

    # Full steps for position and momentum
    for _ in range(n_steps - 1):
        q = q + step_size_t * (p @ inv_mass_matrix)

        # --- Boundary Reflection for q ---
        if q_min_bounds is not None and q_max_bounds is not None:
            for dim_i in range(q.shape[-1]):
                crossed_min = q[..., dim_i] < q_min_bounds[dim_i]
                crossed_max = q[..., dim_i] > q_max_bounds[dim_i]
                if torch.any(crossed_min):
                    q[crossed_min, dim_i] = q_min_bounds[dim_i] + (q_min_bounds[dim_i] - q[crossed_min, dim_i])
                    p[crossed_min, dim_i] = -p[crossed_min, dim_i]
                if torch.any(crossed_max):
                    q[crossed_max, dim_i] = q_max_bounds[dim_i] - (q[crossed_max, dim_i] - q_max_bounds[dim_i])
                    p[crossed_max, dim_i] = -p[crossed_max, dim_i]
        # --- End Boundary Reflection ---

        grad = log_grad_fn(q)
        p = p - step_size_t * -grad

    # Final full step for position and half step for momentum
    q = q + step_size_t * (p @ inv_mass_matrix)
    
    # --- Final Boundary Reflection for q ---
    if q_min_bounds is not None and q_max_bounds is not None:
        for dim_i in range(q.shape[-1]):
            crossed_min = q[..., dim_i] < q_min_bounds[dim_i]
            crossed_max = q[..., dim_i] > q_max_bounds[dim_i]
            if torch.any(crossed_min):
                # Just clip at the final step to prevent leaving the domain
                q[crossed_min, dim_i] = q_min_bounds[dim_i]
            if torch.any(crossed_max):
                q[crossed_max, dim_i] = q_max_bounds[dim_i]
    # --- End Final Boundary Reflection ---
    
    grad = log_grad_fn(q)
    p = p - 0.5 * step_size_t * -grad
    
    return q, -p  # Negate momentum for detailed balance

def sample_hmc(log_prob_fn, log_grad_fn, q_init, step_size=0.01, n_leapfrog_steps=10,
               chain_length=500, burnin_steps=250, adapt=True,
               q_min_bounds=None, q_max_bounds=None):
    
    
    q = q_init.clone()
    # print('q.shape: ', q.shape)
    batch_size, dim = q.shape
    device = q.device
    
    # HMC is simpler with identity mass matrix for this problem
    inv_mass_matrix = torch.eye(dim, device=device)

    step_size_adapter = None
    if adapt:
        ## for batch_size wise chaining
        step_size_adapter = DualAveragingStepSize(step_size * torch.ones((batch_size), device=q.device, dtype=q.dtype))

    q_chain = []
    accept_rates = []

    for i in range(chain_length + burnin_steps):
        p = torch.randn_like(q) # Sample p ~ N(0, I)

        # Leapfrog integration
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog_steps, log_grad_fn, inv_mass_matrix,
                                q_min_bounds, q_max_bounds)

        # Metropolis-Hastings acceptance
        # Potential V = -logP, Hamiltonian H = V + K
        H_old = -log_prob_fn(q) + 0.5 * (p**2).sum(dim=-1)
        H_new = -log_prob_fn(q_new) + 0.5 * (p_new**2).sum(dim=-1)

        log_accept_ratio = H_old - H_new
        log_accept_ratio = torch.nan_to_num(log_accept_ratio, nan=-torch.inf) # Handle NaNs
        
        accept_prob = torch.exp(torch.clamp(log_accept_ratio, max=0.0))
        accept_rates.append(accept_prob.mean().item())
        
        u = torch.rand(batch_size, device=device)
        accept_mask = u < accept_prob
        
        q[accept_mask.squeeze(-1)] = q_new[accept_mask.squeeze(-1)].detach() # Update state

        # print('Accepted-State: ', q)
        
        # Adapt step size during burn-in
        if adapt and i < burnin_steps:
            step_size = step_size_adapter.update(accept_prob)
        elif adapt and i == burnin_steps:
            step_size = step_size_adapter.get_final_step_size()
            p_step_size = step_size.mean().item()
            print(f"HMC adaptation finished. Final step size: {p_step_size:.4e}")

        if i >= burnin_steps:
            q_chain.append(q.clone())

    final_q_samples = torch.stack(q_chain, dim=1) if q_chain else q.unsqueeze(1)
    mean_accept_rate = np.mean(accept_rates)
    
    if adapt:
        return q, step_size, inv_mass_matrix, mean_accept_rate
    else: 
        return q, mean_accept_rate

    # return final_q_samples, step_size, inv_mass_matrix, mean_accept_rate

# --- Main execution for testing ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")


import os
import numpy as np
import math
import torch
from autograd import grad
import torch.nn as nn
import torch.nn.functional as F

sigma_min, sigma_max = 0.04, 0.4

def get_phi_all_bounds(phi_min=-1.0, phi_max = 1.0, sigma_min = 0.04, sigma_max = 0.4, device = 'cpu'):
    phi_min = torch.tensor([phi_min], device=device)
    phi_max = torch.tensor([phi_max], device=device)
    sigma_min = torch.tensor([sigma_min], device=device)
    sigma_max = torch.tensor([sigma_max], device=device)

    phi_min_all = torch.cat([phi_min, sigma_min])
    phi_max_all = torch.cat([phi_max, sigma_max])
    return phi_min_all, phi_max_all

def get_noise_estimate_2d(y, sigma_min, sigma_max):
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

def get_noise_estimate_1d(y, sigma_min, sigma_max, kernel_size=31):
    high_freq_part = high_pass_filter(y, kernel_size)
    sigma_est = high_freq_part.std()
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est.unsqueeze(0)

## standardize = [-1, 1] -> [0, 1] ## for HMC-sampling
def normalize_phi(phi, phi_max=1.0, phi_min=-1.0, mode='compact'):
    ret = (phi - phi_min) / (phi_max - phi_min)
    return ret

## de-standardize = [0, 1] -> [-1, 1]
def unnormalize_phi(phi, phi_max=1.0, phi_min=-1.0, mode='compact'):
    ret = phi * (phi_max - phi_min) + phi_min
    return ret

def sample_phi_prior(n, phi_min=-1.0, phi_max=1.0, norm_mode='compact'):
    phi = torch.rand(n) * (phi_max - phi_min) + phi_min
    return normalize_phi(phi)


def log_prior_phi(phi, norm_mode="compact"):
    # phi is (b, 2)
    in_bounds = torch.logical_and(phi >= 0.0, phi <= 1.0)  # (b, 2)
    all_in_bounds = torch.all(in_bounds, dim=-1).float()   # (b,)
    logp = torch.log(all_in_bounds + 1e-30)  # add epsilon to avoid log(0)
    return logp


def log_prior_phi_sigma(phi, sigma, sigma_min=1e-2, sigma_max=1e2, norm_mode="compact"):
    # phi: (b, 2), sigma: (b,)
    in_bounds_phi = torch.logical_and(phi >= 0.0, phi <= 1.0)  # (b, 2)
    valid_phi = torch.all(in_bounds_phi, dim=-1).float()       # (b,)
    logp_phi = torch.log(valid_phi + 1e-30)                    # (b,)

    in_bounds_sigma = torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float()  # (b,)
    logp_sigma = torch.log(in_bounds_sigma + 1e-30)             # (b,)

    logp = logp_phi + logp_sigma
    return logp

def log_likelihood_eps_phi(phi, eps, ps_model):
    
    ps = ps_model(phi)  # shape: same as eps (excluding batch)

    if eps.ndim == 3:  # 1D case: (B, dim, N)
        eps_dim = eps.shape[-1]
        xf = torch.fft.fft(eps)
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(ps), dim=-1)
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / ps, dim=-1) / eps_dim

    elif eps.ndim == 4:  # 2D case: (B, C, H, W)
        H, W = eps.shape[-2], eps.shape[-1]
        eps_dim = H * W
        xf = torch.fft.fft2(eps)
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(ps), dim=(-2, -1))  # sum over H, W
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / ps, dim=(-2, -1)) / eps_dim

    else:
        raise ValueError("eps must be 2D (1D case) or 4D (image case)")

    log_likelihood = term_pi + term_logdet + term_x  # (b, dim)
    log_likelihood = log_likelihood.sum(dim=1)       # (b,) --> sum over the channel dimm
    return log_likelihood


def log_likelihood_eps_phi_sigma(phi, sigma, eps, ps_model):
    
    ps = ps_model(phi)

    if eps.ndim == 3:  # 1D case
        eps_dim = eps.shape[-1]
        xf = torch.fft.fft(eps)
        sigma = sigma.view(-1, 1) if sigma.ndim == 1 else sigma  # (B, 1)
        scaled_ps = sigma**2 * ps
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=-1)
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=-1) / eps_dim

    elif eps.ndim == 4:  # 2D image case
        H, W = eps.shape[-2], eps.shape[-1]
        eps_dim = H * W
        xf = torch.fft.fft2(eps)
        sigma = sigma.view(-1, 1, 1, 1) if sigma.ndim == 1 else sigma
        scaled_ps = sigma**2 * ps
        term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
        term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=(-2, -1))
        term_x = -0.5 * torch.sum(torch.abs(xf).pow(2) / scaled_ps, dim=(-2, -1)) / eps_dim

    else:
        raise ValueError("eps must be 2D (1D case) or 4D (image case)")
    
    log_likelihood = term_pi + term_logdet + term_x  # (b, dim)
    log_likelihood = log_likelihood.sum(dim=1)       # (b,) --> sum over the channel dim
    return log_likelihood


## We gradient-trace the ColoredPoweredSpectrum
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

class ColoredPowerSpectrum1D(nn.Module):
    def __init__(self, norm_input_phi='compact', shape=(1, 100), device='cpu'):
        super().__init__()
        shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        assert len(shape) == 2  # (dim, seq_len)
        
        dim, N = shape
        self.norm_input_phi = norm_input_phi

        # Create isotropic wavenumber vector for 1D
        wn = torch.fft.fftfreq(N).to(torch.float32)  # shape: (N,)
        wn = wn.to(device)

        # Compute wavenumber magnitude (squared)
        S = wn.pow(2).reshape(1, 1, N)  # (1, 1, N)
        S = torch.sqrt(S)
        S[:, :, 0] = sigma_eps  # avoid division by zero for k=0

        self.S = S  # shape: (1, 1, N)

    def forward(self, phi):
        '''
        phi: tensor of shape (batch_size, dim) or (batch_size, 1)
             controls the spectral slope alpha
        '''
        phi = unnormalize_phi(phi, mode=self.norm_input_phi)

        batch_size, dim = phi.shape[0], self.S.shape[1]
        S = self.S.repeat(batch_size, dim, 1)  # shape: (batch_size, dim, N)
        S = S ** phi.reshape(-1, 1, 1)         # shape: (batch_size, dim, N)
        S = S / S.mean(dim=1, keepdim=True)    # Normalize spectrum
        return S

#### -------------------------------------------------------------
#### ------------------------- HMC (Utils) -----------------------
#### -------------------------------------------------------------


sigma_eps = 1e-6

def reflect_boundary(q, p, p_nxt, phi_min_norm, phi_max_norm):
    p_ret = p_nxt.clone()
    for i in range(2):  # phi and sigma
        crossed_min_boundary = q[..., i] < phi_min_norm[i]
        crossed_max_boundary = q[..., i] > phi_max_norm[i]
        p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
        p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]
    return p_ret

def compute_inverse_mass_matrix(mass_matrix):
    return torch.linalg.inv(mass_matrix)

def kinetic_energy(p, inv_mass_matrix):
    return 0.5 * torch.sum(p * (inv_mass_matrix @ p.unsqueeze(-1)).squeeze(-1), dim=1)

def hamiltonian(q, p, log_prob_fn, inv_mass_matrix):
    return -log_prob_fn(q) + kinetic_energy(p, inv_mass_matrix)

def leapfrog(q, p, step_size, n_steps, log_prob_fn, log_grad, inv_mass_matrix):
    
    q = q.clone()
    p = p.clone()

    grad_ = log_grad(q)
    # print('leap-frog-grad: ', p, q, grad_)
    p -= 0.5 * step_size * grad_
    # print('p_ini: ', p)

    for _ in range(n_steps):
        q += step_size * (p @ inv_mass_matrix)
        grad = log_grad(q)
        p -= step_size * grad

    p -= 0.5 * step_size * log_grad(q)
    p = -p  # negate for symmetry

    # print('leapgrog-after: ', p, q)
    return q, p

class DualAveragingStepSize:
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10, kappa=0.75):
        self.mu = np.log(10 * initial_step_size)
        self.log_step = np.log(initial_step_size)
        self.h_bar = 0
        self.step_bar = 0
        self.t = 1
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa

    def update(self, accept_prob):
        self.t += 1
        eta = 1.0 / (self.t + self.t0)
        self.h_bar = (1 - eta) * self.h_bar + eta * (self.target_accept - accept_prob)
        log_step = self.mu - (np.sqrt(self.t) / self.gamma) * self.h_bar
        self.log_step = log_step
        self.step_bar = np.exp((self.t ** -self.kappa) * log_step + (1 - self.t ** -self.kappa) * np.log(self.step_bar or np.exp(log_step)))
        return np.exp(log_step)

def sample_hmc(log_prob_fn, log_grad, phi_init, step_size=0.1, n_leapfrog_steps=50, chain_length=100, burnin_steps=20, inv_mass_matrix=None, adapt=True, n_adapt=100, phi_min_norm=None, phi_max_norm=None):
    
    q = phi_init.clone()

    batch_size, dim = q.shape

    if inv_mass_matrix is None:
        mass_matrix = torch.eye(dim)
        inv_mass_matrix = compute_inverse_mass_matrix(mass_matrix)

    step_size_adapter = DualAveragingStepSize(step_size)

    accept_prob_list = []
    for i in range(1, chain_length + burnin_steps + 1):
        p = torch.randn_like(q)
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog_steps, log_prob_fn, log_grad, inv_mass_matrix)

        if phi_min_norm is not None and phi_max_norm is not None:
            p_new = reflect_boundary(q, p, p_new, phi_min_norm, phi_max_norm)

        H_old = hamiltonian(q, p, log_prob_fn, inv_mass_matrix)
        H_new = hamiltonian(q_new, p_new, log_prob_fn, inv_mass_matrix)

        # print('Hamiltonian:')
        # print(H_old, H_new)

        accept_prob = torch.exp(torch.clamp(H_old - H_new, max=0.0)).reshape(-1, 1)
        accept_prob_list.append(accept_prob.mean(dim=0).item())

        accept = torch.rand(q.shape) < accept_prob
        
        q[accept] = q_new[accept]

        if adapt and i <= n_adapt:
            step_size = step_size_adapter.update(accept_prob.mean().item())


    mean_accept_prob = np.mean(np.array(accept_prob_list), axis=0)
    # return torch.stack(samples[burnin_steps:], dim=1), step_size, inv_mass_matrix if adapt else torch.stack(samples[burnin_steps:], dim=1)

    if adapt:
        return q, step_size, inv_mass_matrix, mean_accept_prob
    else: 
        return q, mean_accept_prob

if __name__ == '__main__':

    phi = torch.randn(1, 2).reshape(1, 2)
    epsilon = torch.randn(1, 1, 100).reshape(1, 1, 100)
    
    print('phi: ', phi, phi.shape)
    print(epsilon.shape)

    def log_posterior(phi_):
        return log_likelihood_eps_phi_sigma(phi_, epsilon) + log_prior_phi_sigma(phi_)

    phi, _, _ = sample_hmc(log_prob_fn=log_posterior, phi_init=phi)

    print('sampling successful!')

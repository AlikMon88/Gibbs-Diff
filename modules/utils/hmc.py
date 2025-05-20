import os
import numpy as np
import math
import torch
from autograd import grad
import torch.nn as nn

sigma_min, sigma_max = 0.04, 0.4

def get_phi_all_bounds(phi_min=-1.0, phi_max = 1.0, sigma_min = 0.04, sigma_max = 0.4, device = 'cpu'):
    phi_min = torch.tensor([phi_min], device=device)
    phi_max = torch.tensor([phi_max], device=device)
    sigma_min = torch.tensor([sigma_min], device=device)
    sigma_max = torch.tensor([sigma_max], device=device)

    phi_min_all = torch.cat([phi_min, sigma_min])
    phi_max_all = torch.cat([phi_max, sigma_max])
    return phi_min_all, phi_max_all

def get_noise_estimate(y, sigma_min, sigma_max):
    y_std = y.std()
    sigma_est = 1.15 * y_std - 0.17  # heuristic from Imagenet
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est.unsqueeze(0)

## standardize = [-1, 1] -> [0, 1]
def normalize_phi(phi, phi_max=1.0, phi_min=-1.0, mode='compact'):

    ret = (phi - phi_min) / (phi_max - phi_min)
    return ret

    # if mode == "compact":
    #     return (phi - phi_min) / (phi_max - phi_min)
    # elif mode == "inf":
    #     compact = (phi - phi_min) / (phi_max - phi_min)
    #     return torch.tan((compact - 0.5) * torch.pi)
    # else:
    #     return phi

## de-standardize = [0, 1] -> [-1, 1]
def unnormalize_phi(phi, phi_max=1.0, phi_min=-1.0, mode='compact'):

    ret = phi * (phi_max - phi_min) + phi_min
    return ret

    # if mode == "compact":
    #     return phi * (phi_max - phi_min) + phi_min
    # elif mode == "inf":
    #     compact = torch.atan(phi) / torch.pi + 0.5
    #     return compact * (phi_max - phi_min) + phi_min
    # else:
    #     return phi

def sample_phi_prior(n, norm_mode='compact'):

    phi = torch.rand(n) * (phi_max - phi_min) + phi_min
    return normalize_phi(phi, norm_mode)

def log_prior_phi(phi, norm_mode):

    logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float())
    for i in range(1, phi.shape[-1]):
        logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())
    
    return logp

    # if norm_mode == "compact":
    #     return torch.where((phi >= 0) & (phi <= 1), torch.zeros_like(phi), torch.full_like(phi, -float('inf')))
    # elif norm_mode == "inf":
    #     return -torch.log1p(phi ** 2)
    # else:
    #     return -torch.log(torch.tensor(phi_max - phi_min))

def log_prior_phi_sigma(phi, sigma, norm_mode="compact"): ## ~ returns 0
    
    logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float())
    for i in range(1, phi.shape[-1]):
        logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())

    logp += torch.log(torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float())

    return logp

    # prior_phi = log_prior_phi(phi, norm_mode)
    # valid_sigma = (sigma >= sigma_min) & (sigma <= sigma_max)
    # prior_sigma = torch.where(valid_sigma, torch.zeros_like(sigma), torch.full_like(sigma, -float('inf')))
    # ret = prior_phi + prior_sigma
    # print('ret-prior: ', ret)
    # return ret

### All these prior and Log-likelihood functions are returning nan
def log_likelihood_eps_phi(phi, eps, ps_model):

    eps_dim = eps.shape[-1]  # N
    ps = ps_model(phi)       # Should return same shape as eps

    xf = torch.fft.fft(eps)
    term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
    term_logdet = -0.5 * torch.sum(torch.log(ps), dim=-1)
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / ps, dim=-1) / eps_dim

    return term_pi + term_logdet + term_x

def log_likelihood_eps_phi_sigma(phi, sigma, eps, ps_model): ## ~ return nan
    
    eps_dim = eps.shape[-1]
    ps = ps_model(phi)  # Same shape as eps
    sigma = sigma.view(-1, 1) if sigma.ndim == 1 else sigma  # broadcast

    xf = torch.fft.fft(eps)
    scaled_ps = sigma**2 * ps

    term_pi = -(eps_dim / 2) * np.log(2 * np.pi)
    term_logdet = -0.5 * torch.sum(torch.log(scaled_ps), dim=-1)
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / scaled_ps, dim=-1) / eps_dim

    return term_pi + term_logdet + term_x

class ColoredPowerSpectrum1D(torch.nn.Module):

    def __init__(self, seq_len=100, device='cpu'):
        super().__init__()
        
        k = torch.fft.fftfreq(seq_len, d=1.0).to(device)
        k[0] = sigma_eps  # avoid divide by zero
        self.k = k
        self.device = device

    def forward(self, phi):
        phi = unnormalize_phi(phi, mode="compact" if phi.ndim == 1 else "inf")

        if phi.ndim == 0:
            phi = phi.reshape(1)

        batch_size = phi.shape[0]
        k = self.k.unsqueeze(0).expand(batch_size, -1)
        S = k ** phi.unsqueeze(1)
        S = S / S.mean(dim=1, keepdim=True)
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

def gradient_log_prob(log_prob_fn, q):
    q_clone = q.clone().requires_grad_(True)
    logp = log_prob_fn(q_clone)
    print(type(logp), logp.shape)
    grad_q = torch.autograd.grad(logp, q_clone, grad_outputs=torch.ones_like(logp))[0]
    return grad_q

def leapfrog(q, p, step_size, n_steps, log_prob_fn, log_grad, inv_mass_matrix):
    q = q.clone()
    p = p.clone()

    p -= 0.5 * step_size * log_grad(q)

    for _ in range(n_steps):
        q += step_size * (p @ inv_mass_matrix)
        grad = gradient_log_prob(log_prob_fn, q)
        p -= step_size * grad

    p -= 0.5 * step_size * gradient_log_prob(log_prob_fn, q)
    p = -p  # negate for symmetry

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

def sample_hmc(log_prob_fn, log_grad, phi_init, step_size=0.1, n_leapfrog_steps=50, chain_length=50, burnin_steps=10, inv_mass_matrix=None, adapt=True, n_adapt=100, phi_min_norm=None, phi_max_norm=None):
    
    q = phi_init.clone()
    samples = []

    batch_size, dim = q.shape

    if inv_mass_matrix is None:
        mass_matrix = torch.eye(dim)
        inv_mass_matrix = compute_inverse_mass_matrix(mass_matrix)

    step_size_adapter = DualAveragingStepSize(step_size)

    for i in range(1, chain_length + burnin_steps + 1):
        p = torch.randn_like(q)
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog_steps, log_prob_fn, log_grad, inv_mass_matrix)

        if phi_min_norm is not None and phi_max_norm is not None:
            p_new = reflect_boundary(q, p, p_new, phi_min_norm, phi_max_norm)

        H_old = hamiltonian(q, p, log_prob_fn, inv_mass_matrix)
        H_new = hamiltonian(q_new, p_new, log_prob_fn, inv_mass_matrix)

        accept_prob = torch.exp(torch.clamp(H_old - H_new, max=0.0))
        accept = torch.rand(batch_size) < accept_prob
        q[accept] = q_new[accept]

        if adapt and i <= n_adapt:
            step_size = step_size_adapter.update(accept_prob.mean().item())

        samples.append(q.clone())

    return torch.stack(samples[burnin_steps:], dim=1), step_size, inv_mass_matrix if adapt else torch.stack(samples[burnin_steps:], dim=1)

if __name__ == '__main__':

    phi = torch.randn(1, 2).reshape(1, 2)
    epsilon = torch.randn(1, 1, 100).reshape(1, 1, 100)
    
    print('phi: ', phi, phi.shape)
    print(epsilon.shape)

    def log_posterior(phi_):
        return log_likelihood_eps_phi_sigma(phi_, epsilon) + log_prior_phi_sigma(phi_)

    phi, _, _ = sample_hmc(log_prob_fn=log_posterior, phi_init=phi)

    print('sampling successful!')

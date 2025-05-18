### HMC Main + Utils

import numpy as np
import math
import torch
from autograd import grad


phi_min = 1.0
phi_max = 4.0
sigma_eps = 1e-6d

def reflect_boundary(q, p, p_nxt):

    p_ret = p_nxt
    for i in range(2): ## phi and sigma
        crossed_min_boundary = q[..., i] < phi_min_norm[i]
        crossed_max_boundary = q[..., i] > phi_max_norm[i]

        # Reflecting boundary conditions
        p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
        p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]
    return p_ret

def phi_all_bounds(phi_min, phi_max, sigma_min, sigma_max, device):
   
    phi_min = torch.tensor([phi_min]).to(device)
    phi_max = torch.tensor([phi_max]).to(device)
    sigma_min = torch.tensor([sigma_min]).to(device)
    sigma_max = torch.tensor([sigma_max]).to(device)

    phi_min_all = torch.concatenate([phi_min, sigma_min])
    phi_max_all = torch.concatenate([phi_max, sigma_max])

    return phi_min_all, phi_max_all


def log_likelihood(phi, epsilon):

    # Placeholder log-likelihood function
    sigma = phi[:, 1]
    return -0.5 * np.sum((epsilon / sigma[:, None, None, None]) ** 2, axis=(1, 2, 3))

def log_prior(phi):

    # Placeholder: Assuming standard normal prior over phi
    return -0.5 * np.sum(phi**2, axis=1)

def get_noise_estimate(y, sigma_min, sigma_max):

    y_std = y.std()
    sigma_est = 1.15 * y_std - 0.17 ## heuristics for Imagenet dataset
    sigma_est = torch.clamp(sigma_est, sigma_min * 1.05, sigma_max * 0.95)
    return sigma_est

def normalize_phi(phi, mode='compact'):

    if mode == "compact":
        return (phi - phi_min) / (phi_max - phi_min)
    elif mode == "inf":
        compact = (phi - phi_min) / (phi_max - phi_min)
        return torch.tan((compact - 0.5) * np.pi)
    else:
        return phi

def unnormalize_phi(phi, mode='compact'):

    if mode == "compact":
        return phi * (phi_max - phi_min) + phi_min
    elif mode == "inf":
        compact = (torch.atan(phi) / np.pi + 0.5)
        return compact * (phi_max - phi_min) + phi_min
    else:
        return phi

def generate_image_from_phi(phi, ps_model):
    
    S = ps_model.forward(phi)
    noise = torch.randn_like(S)
    x = fft.ifft2(fft.fft2(noise) * torch.sqrt(S + sigma_eps))
    return x.real

def sample_phi_prior(n, norm_mode):
    phi = torch.rand(n) * (phi_max - phi_min) + phi_min
    return normalize_phi(phi, norm_mode)

def log_prior_phi(phi, norm_mode):

    if norm_mode == "compact":
        return torch.where((phi >= 0) & (phi <= 1), torch.zeros_like(phi), torch.full_like(phi, -float('inf')))
    elif norm_mode == "inf":
        return -torch.log1p(phi ** 2)
    else:
        # Uniform prior over phi_min to phi_max
        return -torch.log(torch.tensor(phi_max - phi_min))

def log_prior_phi_sigma(phi, sigma, norm_mode="compact"):

    prior_phi = log_prior_phi(phi, norm_mode)
    valid_sigma = (sigma >= sigma_min) & (sigma <= sigma_max)
    prior_sigma = torch.where(valid_sigma, torch.zeros_like(sigma), torch.full_like(sigma, -float('inf')))
    return prior_phi + prior_sigma

def log_likelihood_eps_phi(phi, eps, ps_model):

    S = ps_model.forward(phi)
    xf = fft.fft2(eps)
    xf2 = (xf.real ** 2 + xf.imag ** 2)
    term = xf2 / (S + sigma_eps)
    log_det = torch.log(S + sigma_eps).sum(dim=[1, 2])
    return -0.5 * (log_det + term.sum(dim=[1, 2]))

def log_likelihood_eps_phi_sigma(phi, sigma, eps, ps_model):
    s
    S = ps_model.forward(phi)
    S_scaled = (sigma ** 2)[:, None, None] * S
    xf = fft.fft2(eps)
    xf2 = (xf.real ** 2 + xf.imag ** 2)
    term = xf2 / (S_scaled + sigma_eps)
    log_det = torch.log(S_scaled + sigma_eps).sum(dim=[1, 2])
    return -0.5 * (log_det + term.sum(dim=[1, 2]))


#### -------------------------------------------------------------
#### ------------------------- HMC (Utils) -----------------------
#### -------------------------------------------------------------


def compute_inverse_mass_matrix(mass_matrix):
    return np.linalg.inv(mass_matrix)

def kinetic_energy(p, inv_mass_matrix):
    return 0.5 * np.sum(p * (inv_mass_matrix @ p.T).T, axis=1)

def hamiltonian(q, p, log_prob_fn, inv_mass_matrix):
    return -log_prob_fn(q) + kinetic_energy(p, inv_mass_matrix)

## Finite-Difference
def gradient_log_prob(log_prob_fn, q, eps=1e-5):
    grad = np.zeros_like(q)
    for i in range(q.shape[1]):
        dq = np.zeros_like(q)
        dq[:, i] = eps
        grad[:, i] = (log_prob_fn(q + dq) - log_prob_fn(q - dq)) / (2 * eps)
    return grad

## Numpy-Autograd
def gradient_log_prob(log_prob_fn, q):
    
    grad_fn = grad(log_prob_fn)  # autograd grad w.r.t. input q (shape (dim,))
    grads = np.zeros_like(q)
    
    for i in range(q.shape[0]):
        grads[i] = grad_fn(q[i])  # Compute gradient for each sample independently

    return grads

def leapfrog(q, p, step_size, n_steps, log_prob_fn, inv_mass_matrix):
    q = q.copy()
    p = p.copy()

    p -= 0.5 * step_size * gradient_log_prob(log_prob_fn, q)

    for _ in range(n_steps):
        q += step_size * (p @ inv_mass_matrix)
        grad = gradient_log_prob(log_prob_fn, q)
        p -= step_size * grad

    p -= 0.5 * step_size * gradient_log_prob(log_prob_fn, q)
    p = -p  # negate for symmetry

    return q, p

class ColoredPowerSpectrum:
    def __init__(self, image_shape, device="cpu"):
        h, w = image_shape
        ky = torch.fft.fftfreq(h).reshape(-1, 1)
        kx = torch.fft.fftfreq(w).reshape(1, -1)
        self.k_norm = torch.sqrt(kx ** 2 + ky ** 2).to(device)
        self.k_norm[0, 0] = sigma_eps  # avoid log(0)

    def forward(self, phi):
        phi = unnormalize_phi(phi, mode="compact" if phi.ndim == 1 else "inf")
        if phi.ndim == 0:
            phi = phi.view(1)
        batch_size = phi.shape[0]
        S = self.k_norm[None, :, :] ** phi[:, None, None]
        S = S / S.mean(dim=[1, 2], keepdim=True)
        return S

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

## Conditional 1D sampling of phi+sigma parameters given noise from gibbs algo. 
def sample_hmc(log_prob_fn, phi_init, step_size, n_leapfrog_steps, chain_length=50, burinin_steps = 10, inv_mass_matrix=None, adapt=True, n_adapt=100):
   
    q = phi_init.copy()
    samples = []
    
    if not inv_mass_matrix:
        mass_matrix = np.ones(phi.shape[1])
        inv_mass_matrix = compute_inverse_mass_matrix(mass_matrix)
    
    step_size_adapter = DualAveragingStepSize(step_size)
    
    for i in range(1, chain_lenght + burnin_steps + 1):

        # 1. Sample momentum
        p = np.random.multivariate_normal(np.zeros(q.shape[1]), np.linalg.inv(inv_mass_matrix), size=q.shape[0])

        # 2. Leapfrog
        q_new, p_new = leapfrog(q, p, step_size, n_leapfrog_steps, log_prob_fn, inv_mass_matrix)
        p_new = reflect_boundary(q, p, p_new)

        # 3. Hamiltonians
        H_old = hamiltonian(q, p, log_prob_fn, inv_mass_matrix)
        H_new = hamiltonian(q_new, p_new, log_prob_fn, inv_mass_matrix)

        # 4. MH acceptance
        accept_prob = np.exp(np.clip(H_old - H_new, -100, 0))
        accept = np.random.rand(q.shape[0]) < accept_prob
        q[accept] = q_new[accept]  # accept/reject

        # 5. Step size adaptation
        if adapt and i <= n_adapt:
            step_size = step_size_adapter.update(np.mean(accept_prob))

        samples.append(q.copy())
    
    if adapt:
        return np.stack(samples[burnin_steps + 1:], axis=1), step_size, inv_mass_matrix  # shape: (batch, n_samples, dim)
    else:
        return np.stack(samples[burnin_steps + 1:], axis=1)  # shape: (batch, n_samples, dim)

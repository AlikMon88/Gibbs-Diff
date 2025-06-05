import os
import numpy as np
import math
import torch
from autograd import grad
import torch.nn as nn
import torch.nn.functional as F

# from tqdm import tqdm # Not used in the provided user HMC code snippet

# User's existing functions (some might be replaced or become v2)
sigma_eps = 1e-6 # Unused in this snippet, but kept from original
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


def log_prior_phi_sigma(phi, sigma, sigma_min=1e-2, sigma_max=1e2, norm_mode="compact"):

    # phi: (b, 2), sigma: (b,)
    in_bounds_phi = torch.logical_and(phi >= 0.0, phi <= 1.0)  # (b, 2)
    valid_phi = torch.all(in_bounds_phi, dim=-1).float()       # (b,)
    logp_phi = torch.log(valid_phi + 1e-30)                    # (b,)

    in_bounds_sigma = torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float()  # (b,)
    logp_sigma = torch.log(in_bounds_sigma + 1e-30)             # (b,)

    logp = logp_phi + logp_sigma
    return logp


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


def reflect_boundary(q, p_original, p_proposed, phi_min_norm, phi_max_norm):

    """
    Reflects momentum at boundaries.
    Args:
        q (torch.Tensor): Current positions (B, D), used to check boundary crossing.
        p_original (torch.Tensor): Momentum before leapfrog step that led to p_proposed (B,D).
                                    Used for reflection direction.
        p_proposed (torch.Tensor): Proposed momentum after leapfrog (B,D).
        phi_min_norm (torch.Tensor): Minimum boundary values (D) or scalar.
        phi_max_norm (torch.Tensor): Maximum boundary values (D) or scalar.
    Returns:
        torch.Tensor: Momentum after reflection (B,D).
    """
    
    p_ret = p_proposed.clone()
    for i in range(q.shape[-1]):  # Iterate over dimensions
        current_phi_min = phi_min_norm if phi_min_norm.ndim == 0 else phi_min_norm[i]
        current_phi_max = phi_max_norm if phi_max_norm.ndim == 0 else phi_max_norm[i]

        crossed_min = q[..., i] < current_phi_min
        crossed_max = q[..., i] > current_phi_max
        
        # Reflect based on the momentum component *before* the full leapfrog step proposed the new p
        # This is a common way, but some variants reflect p_proposed.
        # The original passed 'p' (pre-leapfrog p) and 'p_nxt' (post-leapfrog p).
        # If reflection means p_i_reflected = -p_i_original for that dimension:
        p_ret[crossed_min, i] = -torch.abs(p_original[crossed_min, i]) \
            if torch.all(p_original[crossed_min, i] > 0) else torch.abs(p_original[crossed_min, i])
        p_ret[crossed_max, i] = -torch.abs(p_original[crossed_max, i]) \
            if torch.all(p_original[crossed_max, i] < 0) else torch.abs(p_original[crossed_max, i])
        # A simpler reflection often just flips the sign of the proposed momentum component:
        # p_ret[crossed_min, i] = -p_proposed[crossed_min, i]
        # p_ret[crossed_max, i] = -p_proposed[crossed_max, i]
        # The user's original code was: p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
        # This means it used the momentum *before* the leapfrog step for the reflection value.
        # Let's stick to that interpretation.
        if torch.any(crossed_min):
             p_ret[crossed_min, i] = -p_original[crossed_min, i]
        if torch.any(crossed_max):
             p_ret[crossed_max, i] = -p_original[crossed_max, i]
    return p_ret



# --- V2 Functions incorporating features from the first HMC code ---

def compute_inverse_mass_v2(mass_matrix_input):

    if mass_matrix_input is None:
        return None
    if not isinstance(mass_matrix_input, torch.Tensor):
        raise TypeError("mass_matrix_input must be a torch.Tensor or None")

    if mass_matrix_input.ndim == 1: # Diagonal mass matrix (D)
        if torch.any(mass_matrix_input <= 0):
            raise ValueError("Diagonal mass matrix must be positive. Contains non-positive values.")
        return 1.0 / mass_matrix_input
    
    elif mass_matrix_input.ndim == 2:
        # Check if (D,D) for full or (B,D) for batch of diagonals
        # This heuristic might need refinement if D can be equal to B.
        # Assuming D is reasonably different from B or context implies one or the other.
        # For simplicity, let's assume if shape[0]!=shape[1], it's (B,D)
        if mass_matrix_input.shape[0] == mass_matrix_input.shape[1]: # Full mass matrix (D,D)
            try:
                # Optional: Check for positive definiteness before inverting
                # torch.linalg.cholesky(mass_matrix_input) 
                return torch.linalg.inv(mass_matrix_input)
            except torch.linalg.LinAlgError as e:
                raise ValueError(f"Full mass matrix (D,D) is singular or not invertible: {e}")
        else: # Batch of diagonals (B,D)
            if torch.any(mass_matrix_input <= 0):
                raise ValueError("Batch of diagonal mass matrices must be positive.")
            return 1.0 / mass_matrix_input
    
    elif mass_matrix_input.ndim == 3: # Batch of full mass matrices (B,D,D)
        if mass_matrix_input.shape[-1] != mass_matrix_input.shape[-2]:
            raise ValueError("Matrices in batch of full mass matrices must be square.")
        try:
            # Optional: Check PD for each matrix in batch
            return torch.linalg.inv(mass_matrix_input)
        except torch.linalg.LinAlgError as e:
            raise ValueError(f"One or more matrices in the batch are singular or not invertible: {e}")
    else:
        raise ValueError(f"Unsupported mass_matrix_input ndim: {mass_matrix_input.ndim}")


def kinetic_energy_v2(p, inv_mass_matrix):

    if inv_mass_matrix is None:
        return 0.5 * (p**2).sum(dim=-1)
    
    if inv_mass_matrix.ndim == 1: # Diagonal inverse (D), broadcasts to (B,D) with p:(B,D)
        return 0.5 * (p**2 * inv_mass_matrix).sum(dim=-1)

    elif inv_mass_matrix.ndim == 2:
        if inv_mass_matrix.shape[0] == p.shape[0] and \
           inv_mass_matrix.shape[1] == p.shape[1]: # Batch of diagonals (B,D)
            return 0.5 * (p**2 * inv_mass_matrix).sum(dim=-1)

        elif inv_mass_matrix.shape[0] == p.shape[1] and \
             inv_mass_matrix.shape[1] == p.shape[1]: # Full inverse (D,D) shared
            p_col = p.unsqueeze(-1)
            M_inv_p = (inv_mass_matrix @ p_col).squeeze(-1)
            return 0.5 * (p * M_inv_p).sum(dim=-1)
        
        else:
            raise ValueError(f"Ambiguous inv_mass_matrix shape {inv_mass_matrix.shape} for p shape {p.shape}")

    elif inv_mass_matrix.ndim == 3: # Batch of full inverse (B,D,D)
        p_col = p.unsqueeze(-1)
        M_inv_p = (inv_mass_matrix @ p_col).squeeze(-1)
        return 0.5 * (p * M_inv_p).sum(dim=-1)
    
    else:
        raise ValueError(f"Invalid inv_mass_matrix ndim: {inv_mass_matrix.ndim}")

def dKE_dp_v2(p, inv_mass_matrix):
    
    if inv_mass_matrix is None:
        return p
    
    if inv_mass_matrix.ndim == 1: # Diagonal inverse (D)
        return p * inv_mass_matrix

    elif inv_mass_matrix.ndim == 2:
        if inv_mass_matrix.shape[0] == p.shape[0] and \
           inv_mass_matrix.shape[1] == p.shape[1]: # Batch of diagonals (B,D)
            return p * inv_mass_matrix
        
        elif inv_mass_matrix.shape[0] == p.shape[1] and \
             inv_mass_matrix.shape[1] == p.shape[1]: # Full inverse (D,D) shared
            return (inv_mass_matrix @ p.unsqueeze(-1)).squeeze(-1)
        else:
        
            raise ValueError(f"Ambiguous inv_mass_matrix shape {inv_mass_matrix.shape} for p shape {p.shape}")
    
    elif inv_mass_matrix.ndim == 3: # Batch of full inverse (B,D,D)
        return (inv_mass_matrix @ p.unsqueeze(-1)).squeeze(-1)
    
    else:
        raise ValueError(f"Invalid inv_mass_matrix ndim: {inv_mass_matrix.ndim}")

def hamiltonian_v2(q, p, log_prob_fn, inv_mass_matrix):
    return -log_prob_fn(q) + kinetic_energy_v2(p, inv_mass_matrix)

def leapfrog_v2(q, p, step_size, n_steps, log_grad_fn, inv_mass_matrix):
    """
    log_grad_fn returns d(log_prob)/dq.
    Force F = d(log_prob)/dq. Momentum update p' = p + eps * F.
    """
    q_curr = q.clone()
    p_curr = p.clone()
    
    # Initial half-step for momentum
    p_curr = p_curr + 0.5 * step_size * log_grad_fn(q_curr)

    # Full steps
    for _ in range(n_steps - 1):
        q_curr = q_curr + step_size * dKE_dp_v2(p_curr, inv_mass_matrix)
        p_curr = p_curr + step_size * log_grad_fn(q_curr)
    
    # Final full step for position and half-step for momentum
    q_curr = q_curr + step_size * dKE_dp_v2(p_curr, inv_mass_matrix)
    p_curr = p_curr + 0.5 * step_size * log_grad_fn(q_curr)
    
    p_curr = -p_curr  # Negate momentum for detailed balance
    return q_curr, p_curr

class DualAveragingStepSize: # User's class with minor adjustments
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10, kappa=0.75):
        self.mu = np.log(10 * initial_step_size) 
        self.log_initial_step = np.log(initial_step_size)
        self.log_step = self.log_initial_step # Current noisy log step size
        self.log_step_bar = self.log_initial_step # Averaged log step size
        self.h_bar = 0.0
        
        self.iter_count = 0 # Iteration counter m (starts from 1 for updates)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa

    def update(self, accept_prob):
        self.iter_count += 1
        
        # Regularization for h_bar (log_step update)
        eta_h = 1.0 / (self.iter_count + self.t0)
        self.h_bar = (1 - eta_h) * self.h_bar + eta_h * (self.target_accept - accept_prob)
        
        self.log_step = self.mu - (np.sqrt(self.iter_count) / self.gamma) * self.h_bar
        
        # Averaging for log_step_bar
        eta_s = self.iter_count ** (-self.kappa)
        self.log_step_bar = eta_s * self.log_step + (1 - eta_s) * self.log_step_bar
        
        return np.exp(self.log_step) # Return current (noisy) step size

    def get_final_step_size(self):
        return np.exp(self.log_step_bar) # Return smoothed/averaged step size

def sample_hmc_v2(log_prob_fn, log_grad_fn, phi_init, inv_mass_matrix=None, step_size=0.1, n_leapfrog_steps=50, chain_length=100, burnin_steps=20, adapt=True, n_adapt=100, phi_min_norm=None, phi_max_norm=None):
    
    q = phi_init.clone()
    if q.ndim == 1: q = q.unsqueeze(0)
    batch_size, _ = q.shape

    if not inv_mass_matrix:
        inv_mass_matrix = compute_inverse_mass_v2(inv_mass_matrix)

    current_step_size = float(step_size) # Ensure it's a float for adapter
    if adapt:
        step_size_adapter = DualAveragingStepSize(initial_step_size=current_step_size)

    # samples_list = [] # Uncomment if storing all samples
    accept_prob_list = []
    q_list_for_mass_adapt = []

    for i in range(1, chain_length + burnin_steps + 1):

        ''' Might need to sample from ~ (0, M) | Read the theory better | relates to Remannian Manifold MCMC (mass-matrix: M(q) current state: q) '''
        p_original = torch.randn_like(q) # p_original ~ N(0,I) : Need to p_tilde = N(0, M) by transforming momentum variable

        q_new, p_new_proposed = leapfrog_v2(q, p_original, current_step_size, n_leapfrog_steps, log_grad_fn, inv_mass_matrix)
        
        p_final_proposal = p_new_proposed # p_final_proposal already has momentum negated from leapfrog

        if phi_min_norm is not None and phi_max_norm is not None:
            # Pass q_new to check boundary condition on the proposed state.
            # Pass p_original as the momentum before leapfrog if that's the desired reflection logic.
            p_final_proposal = reflect_boundary(q_new, p_original, p_final_proposal, phi_min_norm, phi_max_norm)

        H_old = hamiltonian_v2(q, p_original, log_prob_fn, inv_mass_matrix)
        # Hamiltonian for proposed state uses q_new and p_final_proposal (which includes negation from leapfrog)
        H_new = hamiltonian_v2(q_new, p_final_proposal, log_prob_fn, inv_mass_matrix)

        accept_prob = torch.exp(torch.clamp(H_old - H_new, max=0.0)) # (B)
        accept_prob_list.append(accept_prob.mean(dim=0).item())

        accept_mask = torch.rand(batch_size, device=q.device) < accept_prob
        q[accept_mask] = q_new[accept_mask]

        # if adapt:
        #     if i <= n_adapt:
        #         current_step_size = step_size_adapter.update(accept_prob.mean().item())
        #     elif i == n_adapt + 1: # First step after adaptation window closes
        #          current_step_size = step_size_adapter.get_final_step_size()

        ## Changing mass-matrix through state-accumulation
        if adapt:
            if i <= n_adapt:
                current_step_size = step_size_adapter.update(accept_prob.mean().item())

            if i <= (3 * n_adapt) // 4: # Collect samples for mass matrix estimation
                q_list_for_mass_adapt.append(q.clone().detach())

            # Estimate the inverse mass matrix at the specified point
            if i == (3 * n_adapt) // 4 and len(q_list_for_mass_adapt) > 1: # Ensure enough samples
                # Select a window of samples, e.g., from n_adapt//4 onwards
                start_idx = max(0, len(q_list_for_mass_adapt) - ( (3 * n_adapt) // 4 - (n_adapt // 4) ))
                relevant_samples_tensor = torch.stack(q_list_for_mass_adapt[start_idx:], dim=-2) # (B, N_samples, D)

                if relevant_samples_tensor.shape[-2] > 1: # Need at least 2 samples for covariance
                    B_s, N_s, D_s = relevant_samples_tensor.size()
                    mean_s = relevant_samples_tensor.mean(dim=-2, keepdim=True)
                    diffs_s = (relevant_samples_tensor - mean_s).reshape(B_s * N_s, D_s)
                    prods_s = torch.bmm(diffs_s.unsqueeze(-1), diffs_s.unsqueeze(-2)).reshape(B_s, N_s, D_s, D_s)
                    # Compute covariance for the mass matrix (M is cov, so inv_mass_matrix is inv_cov)
                    estimated_mass_matrix = prods_s.sum(dim=-3) / (N_s - 1) # (B, D, D)
                    
                    # Update the inv_mass_matrix being used
                    try:
                        inv_mass_matrix = compute_inverse_mass_v2(estimated_mass_matrix.detach())
                        print(f"Iteration {i}: Updated inv_mass_matrix based on sample covariance.")
                    except ValueError as e:
                        print(f"Iteration {i}: Failed to update inv_mass_matrix: {e}. Keeping previous.")
                
                q_list_for_mass_adapt = [] # Clear list after use or manage memory

            elif i == n_adapt + 1: # First step after adaptation window closes
                current_step_size = step_size_adapter.get_final_step_size()
        
    final_q = q
    final_step_size = current_step_size # This will be the smoothed one if adapt=True & i > n_adapt
    mean_accept_prob = np.mean(np.array(accept_prob_list), axis=0)

    if adapt:
        return final_q, final_step_size, inv_mass_matrix, mean_accept_prob
    else: 
        return final_q, mean_accept_prob


# --- Dummy functions for testing ---
def dummy_log_posterior(phi_tensor):
    return -0.5 * torch.sum(phi_tensor**2, dim=-1)

def dummy_log_posterior_grad(phi_tensor):
    return -phi_tensor

if __name__ == '__main__':
    print("Running HMC V2 example:")
    num_chains = 5
    dim = 2
    phi_initial = torch.randn(num_chains, dim) * 2.0 # Start further from mode

    common_sampling_params = {
        "log_prob_fn": dummy_log_posterior,
        "log_grad_fn": dummy_log_posterior_grad,
        "phi_init": phi_initial.clone(),
        "step_size": 0.2, # Adjusted for potentially better initial acceptance
        "n_leapfrog_steps": 20,
        "chain_length": 500,
        "burnin_steps": 200,
        "adapt": True,
        "n_adapt": 150 # Adaptation steps
    }

    print(f"Initial phi mean: {phi_initial.mean(dim=0)}")

    test_scenarios = {
        "Identity Mass": None,
        "Diagonal Mass": torch.tensor([0.5, 5.0]), # Values for mass matrix M
        "Full Mass": torch.tensor([[2.0, 0.8], [0.8, 1.5]]), # M
        "Batched Diagonal Mass": torch.rand(num_chains, dim) * 2.0 + 0.5, # Batch of M_diags
        "Batched Full Mass": torch.stack([ # Batch of M
            A @ A.T + torch.eye(dim) * 1e-2 
            for A in [torch.randn(dim, dim) for _ in range(num_chains)]
        ])
    }
    
    # Boundary condition test (optional)
    # phi_min = torch.tensor([-2.0, -2.0]) 
    # phi_max = torch.tensor([2.0, 2.0])
    # common_sampling_params_bc = {**common_sampling_params, "phi_min_norm": phi_min, "phi_max_norm": phi_max}


    for name, mass_config in test_scenarios.items():
        print(f"\n--- Testing with {name} ---")
        current_params = {**common_sampling_params, "mass_matrix_input": mass_config}
        if name == "Full Mass" or name == "Batched Full Mass": # May need smaller step for dense mass
            current_params["step_size"] = 0.1

        q_f, step_f, inv_m_f = sample_hmc_v2(**current_params)
        print(f"Final q mean ({name}): {q_f.mean(dim=0)}")
        print(f"Final step_size ({name}): {step_f:.4e}")
        # print(f"Inv Mass Mat sample ({name}): {inv_m_f if inv_m_f is None else inv_m_f[0] if inv_m_f.ndim > 1 else inv_m_f}")


    print("\nAll V2 tests completed!")

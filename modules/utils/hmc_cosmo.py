# hmc_cosmo_utils.py

import numpy as np
import torch
import math
from pixell import enmap # For lmap and FFT operations
import camb # For CMB power spectra
# from tqdm.auto import tqdm # Optional for HMC progress if long chains
from tqdm import tqdm


sigma_eps = 1e-6 # Unused in this snippet, but kept from original
# sigma_min, sigma_max = 0.04, 0.4

OMCH2_FID = 0.122 # Cold dark matter density omega_c * h^2
OMK_FID = 0.0    # Omega_k
TAU_FID = 0.0544 # Optical depth
NS_FID = 0.9649  # Scalar spectral index
AS_FID = 2.1e-9  # Scalar amplitude (ln(10^10 As) = 3.044 => As ~ 2.1e-9)

H0_PRIOR_MIN, H0_PRIOR_MAX = 50.0, 90.0
OMBH2_PRIOR_MIN, OMBH2_PRIOR_MAX = 0.0075, 0.0567 # Note: paper uses omega_b, CAMB uses ombh2
# To convert: omega_b = ombh2 / (H0/100)^2. For priors, it's easier to sample H0 and ombh2 directly.
SIGMA_CMB_PRIOR_MIN, SIGMA_CMB_PRIOR_MAX = 0.35, 1.0 # sigma_min should be >0. Let's use 0.1 for now.


# --- Caching ---
_camb_cls_cache = {}
_lmap_cache_cosmo = {}


physical_mins = torch.tensor([SIGMA_CMB_PRIOR_MIN, H0_PRIOR_MIN, OMBH2_PRIOR_MIN])
physical_maxs = torch.tensor([SIGMA_CMB_PRIOR_MAX, H0_PRIOR_MAX, OMBH2_PRIOR_MAX])

def normalize_phi_cmb(phi_physical, mins=physical_mins, maxs=physical_maxs):
    """Maps a physical Phi_CMB vector to the internal [0, 1] range."""
    mins, maxs = mins.to(phi_physical.device), maxs.to(phi_physical.device)
    return (phi_physical - mins) / (maxs - mins)

def unnormalize_phi_cmb(phi_normalized, mins=physical_mins, maxs=physical_maxs):
    """Maps an internal [0, 1] vector back to its physical range."""
    
    mins = mins.reshape(1, -1)  # (1, 3)
    maxs = maxs.reshape(1, -1)  # (1, 3)

    # Move to same device
    mins, maxs = mins.to(phi_normalized.device), maxs.to(phi_normalized.device)

    if phi_normalized.ndim == 4:
        # Expand to match (B, C, T, 3)
        mins = mins[None, None, None, :]  # (1, 1, 1, 3)
        maxs = maxs[None, None, None, :]  # (1, 1, 1, 3)

    return phi_normalized * (maxs - mins) + mins

# --- Cosmology Specific Likelihood & Prior ---

def get_cosmo_lmap(shape_hw, wcs, device='cpu'): # shape_hw is (H,W)
    """
    Gets or computes and caches the lmap for Fourier operations.
    A more robust key might be needed if WCS varies subtly for the same shape.
    For many use cases, shape and a simple WCS descriptor are enough.
    """
    
    if wcs is not None:
        try:
            cd_flat = tuple(wcs.wcs.cd.flatten()) if hasattr(wcs, 'wcs') and hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None else tuple(wcs.wcs.cdelt)
            crpix_flat = tuple(wcs.wcs.crpix)
            ctype_tuple = tuple(wcs.wcs.ctype)
            key_wcs_part = (cd_flat, crpix_flat, ctype_tuple)
        except AttributeError: # Fallback if wcs.wcs structure is different
            key_wcs_part = str(wcs) # Less ideal, but better than error
    else:
        key_wcs_part = "NoneWCS"

    key = (tuple(shape_hw), key_wcs_part)

    if key not in _lmap_cache_cosmo:
        if wcs is None:
            raise ValueError("WCS must be provided to generate lmap.")
        # print(f"Cache miss for lmap key: {key}. Computing lmap.") # For debugging
        _lmap_cache_cosmo[key] = enmap.lmap(shape_hw, wcs)
    # else:
        # print(f"Cache hit for lmap key: {key}") # For debugging
    return _lmap_cache_cosmo[key]

def get_camb_cls_cosmo(H0, ombh2, lmax_camb, device='cpu'):
    
    ## unnormalize H0 and ombh2 for map-creation
    phi_ = torch.tensor([0, H0, ombh2]).float().reshape(1, -1)
    unnorm_phi_ = unnormalize_phi_cmb(phi_)
    H0, ombh2 = unnorm_phi_[:, 1], unnorm_phi_[:, 2] 
    
    # print('H0, ombh2: ', H0, ombh2)
    
    param_key = (float(H0), float(ombh2), OMCH2_FID, OMK_FID, TAU_FID, AS_FID, NS_FID, lmax_camb)
    if param_key in _camb_cls_cache:
        return _camb_cls_cache[param_key].to(device)
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=float(H0), ombh2=float(ombh2), omch2=OMCH2_FID, omk=OMK_FID, tau=TAU_FID)
    pars.InitPower.set_params(As=AS_FID, ns=NS_FID, r=0)
    pars.set_for_lmax(lmax_camb, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    cl_tt_numpy = powers['total'][:, 0]
    if len(cl_tt_numpy) > lmax_camb + 1: cl_tt_numpy = cl_tt_numpy[:lmax_camb+1]
    elif len(cl_tt_numpy) < lmax_camb + 1: cl_tt_numpy = np.pad(cl_tt_numpy, (0, lmax_camb + 1 - len(cl_tt_numpy)), 'constant', constant_values=0)
    cl_tt_numpy[0:2] = 1e-30 # Small non-zero for stability if log taken before scaling. Actual map has 0 mean.
    cl_tt_tensor = torch.from_numpy(cl_tt_numpy.astype(np.float32)).to(device)
    _camb_cls_cache[param_key] = cl_tt_tensor
    return cl_tt_tensor

def cl_to_2d_power_spectrum_cosmo(cl_1d, sigma_cmb_amp, lmap_fourier, device='cpu'):
    
    lmap_fourier = torch.tensor(lmap_fourier)
    lmax_from_map = int(torch.max(torch.tensor(lmap_fourier)).item())
    cl_1d_padded = cl_1d
    if cl_1d.shape[0] <= lmax_from_map:
        padding_needed = (lmax_from_map + 1) - cl_1d.shape[0]
        cl_1d_padded = torch.cat([cl_1d, torch.zeros(padding_needed, dtype=cl_1d.dtype, device=device)])
    
    lmap_fourier = lmap_fourier.long().clamp(min=0, max=(cl_1d_padded.shape[0])-1) ### clamping 0-150 (removing negatives)
    # print(cl_1d_padded.shape, lmap_fourier.shape, lmap_fourier.long().clamp(max=(cl_1d_padded.shape[0])-1))
    ps2d_base = cl_1d_padded[lmap_fourier]
    ps2d = ps2d_base * (sigma_cmb_amp**2)
    # Ensure modes with lmap < 2 (originally C0, C1 = 0) don't cause issues if ps2d_base was 0
    # If ps2d_base[lmap<2] is 0, then ps2d[lmap<2] is also 0. Regularizer handles this in likelihood.
    return ps2d

def log_prior_phi_cmb(phi_cmb_batch, prior_bounds_tuple):
    
    mins, maxs = prior_bounds_tuple
    device = phi_cmb_batch.device
    log_p = torch.zeros(phi_cmb_batch.shape[0], device=device)
    for i in range(phi_cmb_batch.shape[1]):
        param_values = phi_cmb_batch[:, i]
        log_p += torch.where((param_values >= mins[i]) & (param_values <= maxs[i]), 0.0, -torch.inf)
    return log_p

def log_likelihood_cmb_phi(phi_cmb_batch, 
                           epsilon_cmb_batch,
                           wcs,
                           lmap_fourier, 
                           lmax_camb,
                           psd_regularizer=1e-30): # Increased regularizer
    
    batch_size = phi_cmb_batch.shape[0]
    device = epsilon_cmb_batch.device
    log_likelihood_vals = torch.zeros(batch_size, device=device)

    if epsilon_cmb_batch.ndim == 4 and epsilon_cmb_batch.shape[1] == 1:
        epsilon_numpy_batch = epsilon_cmb_batch.squeeze(1).detach().cpu().numpy() ## detach the numpy-trace
        # current_epsilon_maps = epsilon_cmb_batch.squeeze(1)
    elif epsilon_cmb_batch.ndim == 3:
        # current_epsilon_maps = epsilon_cmb_batch
        epsilon_numpy_batch = epsilon_cmb_batch.detach().cpu().numpy() ## detach the numpy-trace
    else:
        raise ValueError(f"epsilon_cmb_batch has unexpected shape: {epsilon_cmb_batch.shape}")
    
    batched_emap = enmap.ndmap(epsilon_numpy_batch, wcs)
    # enmap.fft will operate on the last two dimensions
    epsilon_fourier_numpy_batch = enmap.fft(batched_emap, normalize="phys")
    epsilon_fourier_batch_torch = torch.from_numpy(epsilon_fourier_numpy_batch.astype(np.complex64)).to(device)
    
    abs_epsilon_fourier_sq = torch.abs(epsilon_fourier_batch_torch)**2
    
    # print('phi_cmb_batch: ', phi_cmb_batch.shape)

    for i in range(batch_size):
        sigma_cmb_i, H0_i, ombh2_i = phi_cmb_batch[i, 0], phi_cmb_batch[i, 1], phi_cmb_batch[i, 2]
        cl_1d_base_i = get_camb_cls_cosmo(H0_i.item(), ombh2_i.item(), lmax_camb=lmax_camb, device=device)
        s_phi_k_2d_i = cl_to_2d_power_spectrum_cosmo(cl_1d_base_i, sigma_cmb_i, lmap_fourier, device=device)
        s_phi_k_2d_i_reg = torch.clamp(s_phi_k_2d_i, min=psd_regularizer)
        
        log_det_term = torch.sum(torch.log(s_phi_k_2d_i_reg))
        chi_sq_term  = torch.sum(abs_epsilon_fourier_sq[i] / s_phi_k_2d_i_reg)
        log_likelihood_vals[i] = -0.5 * (log_det_term + chi_sq_term)

    return log_likelihood_vals

def get_phi_cmb_parameter_bounds(sigma_min = SIGMA_CMB_PRIOR_MIN, sigma_max = SIGMA_CMB_PRIOR_MAX, h0_min = H0_PRIOR_MIN, h0_max = H0_PRIOR_MAX, ombh2_min = OMBH2_PRIOR_MIN, ombh2_max = OMBH2_PRIOR_MAX, device='cpu'):
    mins = torch.tensor([sigma_min, h0_min, ombh2_min], dtype=torch.float32, device=device)
    maxs = torch.tensor([sigma_max, h0_max, ombh2_max], dtype=torch.float32, device=device)
    return mins, maxs

# --- HMC Core Components (Adapted for p ~ N(0,M) and Cosmology) ---

# class DualAveragingStepSizeHMC: # From GDiff pyhmc.py style, adapted for our HMC
#     def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
#         self.initial_step_size = initial_step_size
#         # mu is log(10 * initial_step_size) in NUTS paper, can be just log(initial_step_size)
#         # For safety with very small step sizes, ensure initial_step_size is positive
#         self.mu = np.log(10 * self.initial_step_size) if self.initial_step_size > 1e-9 else -np.inf
#         self.target_accept = target_accept
#         self.gamma = gamma
#         self.t = t0
#         self.kappa = kappa
#         self.error_sum = 0.0 # scalar error sum
#         # log_averaged_step is for the *single* step size being adapted
#         self.log_averaged_step = np.log(self.initial_step_size) if self.initial_step_size > 1e-9 else -np.inf

#     def update(self, p_accept_scalar): # p_accept is scalar (mean acceptance over chains)
#         if self.initial_step_size < 1e-9: # Step size effectively fixed
#             return self.initial_step_size, self.initial_step_size

#         p_accept_scalar = np.clip(p_accept_scalar, 0.0, 1.0) # Defensive
#         self.error_sum += self.target_accept - p_accept_scalar
#         log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
#         eta = self.t ** -self.kappa
#         self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
#         self.t += 1.0
#         return torch.exp(torch.tensor(log_step)) # noisy, smoothed

#     def get_final_averaged_step_size(self):
#         if self.initial_step_size < 1e-9: return self.initial_step_size
#         return torch.exp(torch.tensor(self.log_averaged_step))

# def compute_mass_matrix_sqrt_from_M(mass_matrix_M):
#     if mass_matrix_M is None: return None
#     # Simplified: assumes M is (B,D,D) or (D,D) and positive definite
#     # For diagonal M (B,D) or (D), sqrt is element-wise.
#     if mass_matrix_M.ndim == 1: return mass_matrix_M ** 0.5 # (D)
#     if mass_matrix_M.ndim == 2 and mass_matrix_M.shape[0] != mass_matrix_M.shape[1]: # (B,D)
#         return mass_matrix_M ** 0.5
#     # Full or batched full
#     try: # Using Cholesky L, so M = LL^T, then L is M_sqrt
#         return torch.linalg.cholesky(mass_matrix_M)
#     except torch.linalg.LinAlgError: # Fallback to Eigendecomposition if not PD for Cholesky
#         L_eigen, Q_eigen = torch.linalg.eigh(mass_matrix_M)
#         if torch.any(L_eigen < -1e-6): raise ValueError("Mass matrix M not PSD for sqrt.")
#         L_eigen_sqrt = torch.clamp(L_eigen, min=0.0) ** 0.5
#         # M_sqrt = Q @ diag(L_sqrt) @ Q.T, but Q @ diag(L_sqrt) is also a valid sqrt if M_sqrt M_sqrt^T = M
#         # For p_actual = M_sqrt @ p_tilde, we need M_sqrt such M_sqrt M_sqrt^T = M
#         # Q @ diag(L_sqrt) is one such matrix. (Q @ diag(L_sqrt)) (Q @ diag(L_sqrt))^T = Q L Q^T = M
#         return Q_eigen @ torch.diag_embed(L_eigen_sqrt)


# def compute_inverse_mass_from_M(mass_matrix_M):
#     if mass_matrix_M is None: return None
#     if mass_matrix_M.ndim == 1: return 1.0 / mass_matrix_M # (D)
#     if mass_matrix_M.ndim == 2 and mass_matrix_M.shape[0] != mass_matrix_M.shape[1]: # (B,D)
#         return 1.0 / mass_matrix_M
#     try: # Full or batched full
#         return torch.linalg.inv(mass_matrix_M)
#     except torch.linalg.LinAlgError as e:
#         raise ValueError(f"Mass matrix M is singular: {e}")

# def _kinetic_energy_hmc(p_actual, inv_mass_matrix_M_inv): # p_actual ~ N(0,M)
    
#     # print(inv_mass_matrix_M_inv, inv_mass_matrix_M_inv.ndim)
    
#     if inv_mass_matrix_M_inv is None: # M=I
#         return 0.5 * (p_actual**2).sum(dim=-1)
#     # p_actual: (B,D), inv_mass_matrix_M_inv can be (D), (B,D), (D,D), (B,D,D)
#     if inv_mass_matrix_M_inv.ndim == 1: # (D)
#         return 0.5 * (p_actual**2 * inv_mass_matrix_M_inv).sum(dim=-1)
#     elif inv_mass_matrix_M_inv.ndim == 2:
#         if inv_mass_matrix_M_inv.shape[0] == p_actual.shape[0]: # (B,D)
#             return 0.5 * (p_actual**2 * inv_mass_matrix_M_inv).sum(dim=-1)
#         else: # (D,D)
#             M_inv_p = (inv_mass_matrix_M_inv @ p_actual.unsqueeze(-1)).squeeze(-1)
#             return 0.5 * (p_actual * M_inv_p).sum(dim=-1)
#     elif inv_mass_matrix_M_inv.ndim == 3: # (B,D,D)
#         M_inv_p = (inv_mass_matrix_M_inv @ p_actual.unsqueeze(-1)).squeeze(-1)
#         return 0.5 * (p_actual * M_inv_p).sum(dim=-1)
#     raise ValueError("Invalid inv_mass_matrix_M_inv shape")

# def _dKE_dp_actual_hmc(p, inv_mass_matrix): # dK/dp_actual = M^-1 p_actual

#     if inv_mass_matrix is None:
#         return p
    
#     if inv_mass_matrix.ndim == 1: # Diagonal inverse (D)
#         return p * inv_mass_matrix

#     elif inv_mass_matrix.ndim == 2:
#         if inv_mass_matrix.shape[0] == p.shape[0] and \
#            inv_mass_matrix.shape[1] == p.shape[1]: # Batch of diagonals (B,D)
#             return p * inv_mass_matrix
        
#         elif inv_mass_matrix.shape[0] == p.shape[1] and \
#              inv_mass_matrix.shape[1] == p.shape[1]: # Full inverse (D,D) shared
#             return (inv_mass_matrix @ p.unsqueeze(-1)).squeeze(-1)
#         else:
        
#             raise ValueError(f"Ambiguous inv_mass_matrix shape {inv_mass_matrix.shape} for p shape {p.shape}")
    
#     elif inv_mass_matrix.ndim == 3: # Batch of full inverse (B,D,D)
#         return (inv_mass_matrix @ p.unsqueeze(-1)).squeeze(-1)
    
#     else:
#         raise ValueError(f"Invalid inv_mass_matrix ndim: {inv_mass_matrix.ndim}")
    
# def _hamiltonian_hmc(q, p_actual, log_prob_fn_q, inv_mass_matrix_M_inv):
#     potential = -log_prob_fn_q(q)
#     kinetic = _kinetic_energy_hmc(p_actual, inv_mass_matrix_M_inv)
#     return potential + kinetic

# def _leapfrog_hmc(q_curr, p_actual_curr, step_size_val, n_steps,
#                   log_grad_fn_q, inv_mass_matrix_M_inv,
#                   q_min_bounds=None, q_max_bounds=None): # q_min/max are for parameters q
    
#     q = q_curr.clone()
#     p_actual = p_actual_curr.clone()

#     step_size_val_t = torch.tensor(step_size_val, device=q.device, dtype=q.dtype)

#     if q.ndim > 1 and step_size_val_t.ndim == 0 : # if step_size is scalar but q is batched
#         step_size_val_t = step_size_val_t.repeat(q.shape[0]).unsqueeze(-1) # (B,1) for broadcasting with grads
#     elif step_size_val_t.ndim == 1 and q.ndim > 1 : # step_size is (B)
#         step_size_val_t = step_size_val_t.unsqueeze(-1) # (B,1)

#     grad_potential_q = -log_grad_fn_q(q) # V_g = - d(logP)/dq; dP/dt = -V_g
#     # print('grad_potential_q: ', grad_potential_q.shape)

#     p_actual = p_actual - 0.5 * step_size_val_t * grad_potential_q # p_actual update: dp/dt = -dV/dq
#     # print('p_actual: ', p_actual.shape, step_size_val_t.shape, step_size_val_t)

#     for _ in range(n_steps - 1):
#         q = q + step_size_val_t * _dKE_dp_actual_hmc(p_actual, inv_mass_matrix_M_inv) # dq/dt = dK/dp = M^-1 p_actual
#         # Boundary checks for q (parameters)
#         if q_min_bounds is not None and q_max_bounds is not None:
#             for dim_i in range(q.shape[-1]):
#                 crossed_min = q[..., dim_i] < q_min_bounds[dim_i]
#                 crossed_max = q[..., dim_i] > q_max_bounds[dim_i]
#                 if torch.any(crossed_min):
#                     q[crossed_min, dim_i] = q_min_bounds[dim_i] + (q_min_bounds[dim_i] - q[crossed_min, dim_i]) # Reflect position
#                     p_actual[crossed_min, dim_i] = -p_actual[crossed_min, dim_i] # Reflect momentum
#                 if torch.any(crossed_max):
#                     q[crossed_max, dim_i] = q_max_bounds[dim_i] - (q[crossed_max, dim_i] - q_max_bounds[dim_i])
#                     p_actual[crossed_max, dim_i] = -p_actual[crossed_max, dim_i]

#         grad_potential_q_new = -log_grad_fn_q(q)
#         p_actual = p_actual - step_size_val_t * grad_potential_q_new
    
#     q = q + step_size_val_t * _dKE_dp_actual_hmc(p_actual, inv_mass_matrix_M_inv)
#     # print('q: ', q.shape)
#     if q_min_bounds is not None and q_max_bounds is not None: # Final boundary check for q
#         for dim_i in range(q.shape[-1]):
#             crossed_min = q[..., dim_i] < q_min_bounds[dim_i]
#             crossed_max = q[..., dim_i] > q_max_bounds[dim_i]
#             if torch.any(crossed_min):
#                 q[crossed_min, dim_i] = q_min_bounds[dim_i]
#                 # p_actual[crossed_min, dim_i] = 0 # Or reflect, but for param bounds, often just clip q
#             if torch.any(crossed_max):
#                 q[crossed_max, dim_i] = q_max_bounds[dim_i]
#                 # p_actual[crossed_max, dim_i] = 0
    
#     grad_potential_q_final = -log_grad_fn_q(q)
#     # print('grad_potential_q_final: ', grad_potential_q_final.shape)
#     p_actual = p_actual - 0.5 * step_size_val_t * grad_potential_q_final
    
#     return q, -p_actual # Negate momentum for detailed balance

# def sample_hmc_cosmo(log_prob_fn, log_grad_fn, # For target q (Phi_CMB)
#                      phi_init, # Initial q (Phi_CMB) [B_chains, D_phi]
#                      mass_matrix_M_input=None, # Mass matrix M [B_chains, D_phi, D_phi] or [D_phi,D_phi] or [D_phi] or None
#                      step_size_initial=0.01, # scalar
#                      n_leapfrog_steps=15, # int or tuple for random range
#                      num_samples_chain=30, # samples to return per chain (after burn-in)
#                      num_burnin_steps_hmc=15, # HMC's own burn-in for adaptation
#                      adapt_step_size=True,
#                      adapt_mass_matrix=False, # Whether to adapt M during HMC burn-in
#                      num_adapt_steps_total=18, # Total steps for adaptation phase
#                      phi_min_bounds=None, 
#                      phi_max_bounds=None, # Tensors [D_phi]
#                      verbose=False):
    
#     q = phi_init.clone().detach()
#     device = q.device
#     num_chains, dim_phi = q.shape

#     # Initialize M, M_sqrt, M_inv
#     current_M = mass_matrix_M_input.clone().detach() if mass_matrix_M_input is not None else None
#     # if current_M is not None and current_M.ndim == dim_phi: # Shared M [D,D] or [D]
#     #     current_M = current_M.unsqueeze(0).repeat(num_chains, *((1,)*(current_M.ndim)))
#     #     current_M = current_M.squeeze(0)
    
#     # print('current_M: ', current_M.shape)
#     inv_mass_matrix_M_inv = compute_inverse_mass_from_M(current_M)
#     mass_matrix_M_sqrt = compute_mass_matrix_sqrt_from_M(current_M)

#     # Step size adaptation kernel (one kernel, adapts based on mean acceptance)
#     # Step size itself will be a tensor [num_chains]
#     current_step_size_val = float(step_size_initial) # scalar for adapter
#     step_size_adapter = None
#     if adapt_step_size:
#         step_size_adapter = DualAveragingStepSizeHMC(current_step_size_val)
    
#     # Per-chain step sizes, initialized
#     current_step_sizes_per_chain = torch.full((num_chains,), current_step_size_val, device=device, dtype=q.dtype)

#     q_collected_for_adapt_M = []
#     collected_samples_q = []
#     acceptance_rates = []

#     total_hmc_iterations = num_burnin_steps_hmc + num_samples_chain

#     for i_iter in range(total_hmc_iterations):
#         # 1. Sample momentum p_tilde ~ N(0,I), then p_actual = M_sqrt @ p_tilde
#         p_tilde = torch.randn_like(q)
#         if mass_matrix_M_sqrt is not None:
#             if mass_matrix_M_sqrt.ndim == 1: # M_sqrt is [D_phi] (from diagonal M)
#                 p_actual = mass_matrix_M_sqrt * p_tilde
#             elif mass_matrix_M_sqrt.ndim == 2 and mass_matrix_M_sqrt.shape[0] != dim_phi : # (B,D)
#                 p_actual = mass_matrix_M_sqrt * p_tilde # M_sqrt is diag per chain
#             else: # M_sqrt is [D_phi, D_phi] or [B_chains, D_phi, D_phi]
#                 p_actual = (mass_matrix_M_sqrt @ p_tilde.unsqueeze(-1)).squeeze(-1)
#         else: # M = I
#             p_actual = p_tilde
        
#         p_actual_initial = p_actual.clone()

#         # 2. Leapfrog integration
#         # Handle random leapfrog steps
#         current_n_leap = n_leapfrog_steps
#         if isinstance(n_leapfrog_steps, tuple):
#             current_n_leap = np.random.randint(n_leapfrog_steps[0], n_leapfrog_steps[1] + 1)
        
#         # Use current_step_sizes_per_chain for leapfrog
#         q_prop, p_actual_prop = _leapfrog_hmc(
#             q, p_actual, current_step_sizes_per_chain, current_n_leap,
#             log_grad_fn, inv_mass_matrix_M_inv,
#             phi_min_bounds, phi_max_bounds
#         )

#         # 3. Metropolis-Hastings
#         H_initial = _hamiltonian_hmc(q, p_actual_initial, log_prob_fn, inv_mass_matrix_M_inv)
#         H_proposed = _hamiltonian_hmc(q_prop, p_actual_prop, log_prob_fn, inv_mass_matrix_M_inv)

#         # print(q.shape, q_prop.shape, p_actual_initial.shape, p_actual_prop.shape)
#         # print('H: ', H_initial.shape, H_proposed.shape)

#         log_accept_ratio = H_initial - H_proposed # For N(0,M) momentum, proposal is symmetric
        
#         # Handle NaNs/Infs in log_accept_ratio (e.g. from out of bounds proposals)
#         log_accept_ratio = torch.nan_to_num(log_accept_ratio, nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)

#         accept_prob = torch.exp(torch.clamp(log_accept_ratio, max=0.0)) # Numerical stability
#         u = torch.rand(num_chains, device=device)
#         accepted_mask = u < accept_prob
        
#         # print('accpt/q: ', accepted_mask.shape, q.shape)
#         q[accepted_mask] = q_prop[accepted_mask].detach() # Update q for accepted proposals
#         acceptance_rates.append(accepted_mask.float().mean().item())

#         # 4. Adaptation phase (during HMC burn-in)
#         if i_iter < num_burnin_steps_hmc:
#             if adapt_step_size and step_size_adapter:
#                 # The __call__ method for step_size_adapter from GDiff style HMC
#                 # takes (current_adapt_step, accept_prob_batch, total_adapt_steps)
#                 # Here, num_burnin_steps_hmc acts as total_adapt_steps for this HMC call
#                 current_step_sizes_per_chain = step_size_adapter.update(accept_prob.mean(dim=0).item())

#             # if adapt_mass_matrix:
#             #     q_collected_for_adapt_M.append(q.clone().detach())
#             #     # Adapt M at specified point, e.g., 3/4 of HMC burn-in
#             #     # This schedule needs to be robust.
#             #     if num_burnin_steps_hmc > dim_phi and i_iter == (3 * num_burnin_steps_hmc) // 4 :
#             #         if len(q_collected_for_adapt_M) > dim_phi: # Need enough samples
#             #             # Use a window of recent samples
#             #             window_start = max(0, len(q_collected_for_adapt_M) - num_burnin_steps_hmc // 2)
#             #             samples_for_M = torch.stack(q_collected_for_adapt_M[window_start:], dim=0) # [N_collected, B_chains, D_phi]
#             #             # Average over chains or estimate per chain if M is per chain
#             #             # For shared M, average samples across chains then compute covariance:
#             #             # samples_for_M_flat = samples_for_M.transpose(0,1).reshape(-1, dim_phi) # [N_coll*B_chains, D_phi]
#             #             # estimated_M = torch.cov(samples_for_M_flat.T) + torch.eye(dim_phi,device=device)*1e-6 # [D_phi,D_phi]
                        
#             #             # For batched M (per chain):
#             #             estimated_M_batched = torch.zeros_like(current_M) if current_M is not None else torch.eye(dim_phi, device=device).unsqueeze(0).repeat(num_chains,1,1)
#             #             any_M_updated = False
#             #             for chain_idx in range(num_chains):
#             #                 chain_samples = samples_for_M[:, chain_idx, :] # [N_collected, D_phi]
#             #                 if chain_samples.shape[0] > dim_phi : # Ensure enough samples per chain
#             #                     cov_matrix = torch.cov(chain_samples.T)
#             #                     # Ensure positive definiteness
#             #                     cov_matrix += torch.eye(dim_phi, device=device) * 1e-6 
#             #                     estimated_M_batched[chain_idx] = cov_matrix
#             #                     any_M_updated = True
                        
#             #             if any_M_updated:
#             #                 current_M = estimated_M_batched.detach()
#             #                 inv_mass_matrix_M_inv = compute_inverse_mass_from_M(current_M)
#             #                 mass_matrix_M_sqrt = compute_mass_matrix_sqrt_from_M(current_M)
#             #                 if verbose: print(f"HMC Iter {i_iter}: Mass matrix M updated.")
#             #             q_collected_for_adapt_M = [] # Reset
        
#         # Collect samples after HMC burn-in
#         if i_iter >= num_burnin_steps_hmc:
#             collected_samples_q.append(q.clone().detach())

#     final_q_samples = torch.stack(collected_samples_q, dim=1) if collected_samples_q else q.unsqueeze(1) # [B_chains, N_samples_chain, D_phi]
    
#     # print('current_step_sizes_per_chain: ', current_step_sizes_per_chain, current_step_sizes_per_chain.shape, current_step_sizes_per_chain.numel())
#     final_step_size = current_step_sizes_per_chain.item() if current_step_sizes_per_chain.numel() < 2 else current_step_sizes_per_chain[0] # Report one step size
    
#     if adapt_step_size and step_size_adapter and num_burnin_steps_hmc > 0 :
#         final_step_size = step_size_adapter.get_final_averaged_step_size()

#     # return final_q_samples[:, -1, :], final_step_size, current_M, np.mean(acceptance_rates[-num_samples_chain:]) if acceptance_rates else 0.0

#     ## only return the last state
#     return q, final_step_size, current_M, np.mean(acceptance_rates[-num_samples_chain:]) if acceptance_rates else 0.0


## ----------------------- HMCv3 (inspired) ----------------------------

class DualAveragingStepSize():
    """ Dual averaging step size adaptation (Nesterov 2009). """

    def __init__(self, initial_step_size,
                 target_accept=0.65,
                 gamma=0.05,
                 t0=10.0,
                 kappa=0.75,
                 nadapt=0):
        """Constructor

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
        self.error_sum = torch.zeros_like(self.initial_step_size).to(initial_step_size.device)
        self.log_averaged_step = torch.zeros_like(self.initial_step_size).to(initial_step_size.device)
        self.nadapt = nadapt
        
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
        return torch.exp(log_step), torch.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n" % step_size)
        else:
            step_size = torch.exp(self.log_averaged_step)
        return step_size
    


class HMC():
    
    def __init__(self, log_prob,
                 grad_log_prob=None,
                 log_prob_and_grad=None,
                 inv_mass_matrix=None,
                 precision=torch.float32):

        self.precision = precision
        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad
        
        assert not((self.grad_log_prob is None) and (self.log_prob_and_grad is None))

        # Convert to prescribed precision
        if self.log_prob is not None:
            self.log_prob = lambda x: log_prob(x).to(self.precision)
        if self.grad_log_prob is not None:
            self.grad_log_prob = lambda x: grad_log_prob(x).to(self.precision)
        if self.log_prob_and_grad is not None:
            self.log_prob_and_grad = lambda x: tuple([y.to(self.precision) for y in log_prob_and_grad(x)])

        # Set inverse mass matrix and define corresponding kinetic energy function and its gradient
        self.set_inv_mass_matrix(inv_mass_matrix)

        # Define the potential energy
        self.V = lambda x: -self.log_prob(x)

        # Collision function
        self.collision_fn = None

        # Counters
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0
    
    def reset_counters(self):
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0
    
    def set_inv_mass_matrix(self, mat, batch_dim=False):
     
        if mat is None:
            self.mass_matrix_inv = None
            self.mass_matrix_sqrt = None

            # Define kinetic energy and its gradient
            self.KE = lambda p: 0.5*(p**2).sum(-1) # Sum across parameter dimension
            self.KE_g = lambda p: p
        else:
            assert mat.ndim == 1 + int(batch_dim) or mat.ndim == 2 + int(batch_dim)
            if batch_dim:
                self.mass_matrix_batch_dim = True
            if mat.ndim == 1 + int(batch_dim):
                self.mass_matrix_inv = mat
                self.mass_matrix_sqrt = torch.stack([torch.diag(m ** -0.5) for m in mat], dim=0)

                # Define kinetic energy and its gradient
                self.KE = lambda p: 0.5*(p**2 * self.mass_matrix_inv).sum(-1) # Sum across parameter dimension
                self.KE_g = lambda p: p * self.mass_matrix_inv
            else:
                L_list, Q_list = [], []
                mass_matrix_sqrt_list = []

                for i,m in enumerate(mat):
                    assert (m == m.T).all() # Check if symmetric
                    L, Q = torch.linalg.eigh(m)
                    if (L > 0).all(): # Check if positive definite
                        L_list.append(L)
                        Q_list.append(Q)
                        mass_matrix_sqrt_list.append(Q @ torch.diag(L**-0.5) @ Q.T)
                    else:
                        m = mat.mean(dim=0)
                        L, Q = torch.linalg.eigh(m)
                        L_list.append(L)
                        Q_list.append(Q)
                        mass_matrix_sqrt_list.append(Q @ torch.diag(L**-0.5) @ Q.T)
                        mat[i] = m
                
                self.mass_matrix_inv = mat
                self.mass_matrix_sqrt = torch.stack(mass_matrix_sqrt_list, dim=0)

                # Define kinetic energy and its gradient
                self.KE = lambda p: 0.5*(p * (self.mass_matrix_inv @ p.unsqueeze(-1)).squeeze(-1)).sum(-1) # Sum across parameter dimension
                self.KE_g = lambda p: (self.mass_matrix_inv @ p.unsqueeze(-1)).squeeze(-1)

    def V_g(self, q):

        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(q)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(q)
        return -v_g.detach()

    def H(self, q, p, Vq=None):

        if Vq is None:
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)

    def set_collision_fn(self, collision_fn):
       
        self.collision_fn = collision_fn

    def leapfrog(self, q, p, nleap, step_size):
        
        self.leapcount += 1
        s = step_size.unsqueeze(-1)

        assert isinstance(nleap, int) or isinstance(nleap, tuple)
        if isinstance(nleap, tuple):
            N = np.random.randint(nleap[0], nleap[1]+1)
        else:
            N = nleap
 
        p = p - 0.5 * s * self.V_g(q)
        for i in range(N - 1):
            q = q + s * self.KE_g(p)
            if self.collision_fn is not None:
                p = self.collision_fn(q, p, p - s * self.V_g(q))
            else:
                p = p - s * self.V_g(q)
        q = q + s * self.KE_g(p)
        p = p - 0.5 * s * self.V_g(q)
        return q, p
        
    def metropolis(self, q0, p0, q1, p1, V_q0=None, V_q1=None):
        
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = torch.exp(H0 - H1)

        u = torch.rand(prob.shape[0], device=prob.device)

        qq = q1.clone()
        pp = p1.clone()
        acc = torch.ones_like(prob)

        cond1 = torch.logical_or(torch.isnan(prob), torch.isinf(prob))
        cond1 = torch.logical_or(cond1, torch.sum(q0 - q1, dim=-1) == 0)

        qq[cond1] = q0[cond1]
        pp[cond1] = p0[cond1]
        acc[cond1] = -1.0

        cond2 = torch.logical_and(u > torch.min(torch.ones_like(u), prob), ~cond1)
        qq[cond2] = q0[cond2]
        pp[cond2] = p0[cond2]
        acc[cond2] = 0.0
        
        return qq, pp, acc, torch.stack([H0, H1], dim=-1)

    def step(self, q, nleap, step_size):
       
        p = torch.randn(q.shape, device=q.device, dtype=self.precision)
        if self.mass_matrix_sqrt is not None:
            p = (self.mass_matrix_sqrt @ p.unsqueeze(-1)).squeeze(-1)
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis(q, p, q1, p1)
        return q, p, accepted, Hs, torch.tensor([self.Hcount, self.Vgcount, self.leapcount])

    def adapt_stepsize(self, q, step_size, epsadapt, nleap, verbose=False):
        
        print("Adapting step size using %d iterations" % epsadapt)
        epsadapt_kernel = DualAveragingStepSize(step_size)

        q_list = [] # We save the positions to estimate the inverse mass matrix at half the iterations
        
        for i in tqdm(range((epsadapt)), disable=not verbose):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            q = q.detach()
            Hs = Hs.detach()
            
            q_list.append(q)

            prob = torch.exp(Hs[...,0] - Hs[...,1])

            if i < epsadapt - 1:
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt - 1:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)

            # Estimate the inverse mass matrix
            if i == 3*epsadapt//4:
                q_list_tensor = torch.stack(q_list, dim=-2)[:, epsadapt//4:, :]
                assert q_list_tensor.ndim == 3

                # Batched sample covariance
                B, N, D = q_list_tensor.size()
                mean = q_list_tensor.mean(dim=-2, keepdim=True)
                diffs = (q_list_tensor - mean).reshape(B * N, D)
                prods = torch.bmm(diffs.unsqueeze(-1), diffs.unsqueeze(-2)).reshape(B, N, D, D)
                bcov = prods.sum(dim=-3) / (N - 1)  # Unbiased estimate
                self.set_inv_mass_matrix(bcov.detach(), batch_dim=True)
            step_size.detach()
        return q, step_size
    
    def sample(self, q,
               step_size=0.01,
               nsamples=20,
               burnin=10,
               nleap=30,
               skipburn=True,
               epsadapt=0,
               verbose=False,
               ret_side_quantities=False):

        """Performs HMC sampling.
        """
        if q.ndim == 1: q = q.unsqueeze(0) # We add a chain dimension if there is none.
        assert q.ndim == 2, "q must be 2D" # Shape of q is (nchains, ndim)

        self.reset_counters()

        q = q.to(self.precision)
        
        step_size = step_size * torch.ones((q.shape[0]), device=q.device, dtype=self.precision)

        # We store the samples, acceptance rates, Hamiltonian values, and misc counts
        samples_list = []
        accepts_list = []
        Hs_list = []
        counts_list = []

        if epsadapt > 0:
            q, step_size = self.adapt_stepsize(q, step_size, epsadapt, nleap, verbose=verbose)
            self.step_size = step_size
        for i in tqdm(range(nsamples + burnin), disable=not verbose):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            q.detach()
            Hs.detach()
            accepts_list.append(acc)
            if (skipburn and (i >= burnin)) or not skipburn:
                samples_list.append(q)
                Hs_list.append(Hs)
                counts_list.append(count)
            

        # To torch tensors
        samples_list = torch.stack(samples_list, dim=-2)
        accepts_list = torch.stack(accepts_list, dim=-1)
        Hs_list = torch.stack(Hs_list, dim=-2)
        counts_list = torch.stack(counts_list, dim=-2)
        
        if ret_side_quantities:
            return samples_list, accepts_list, Hs_list, counts_list
        else:
            return samples_list


if __name__ == '__main__':
    print("Running __HMC_cosmo__ ...")
    
# hmc_cosmo_utils.py

import numpy as np
import torch
import math
from pixell import enmap # For lmap and FFT operations
import camb # For CMB power spectra
# from tqdm.auto import tqdm # Optional for HMC progress if long chains

# --- Global Configurations for Cosmology (User MUST define these elsewhere and pass them) ---
# These are placeholders. Define them in your main script where you use these utils.
# NPIX_SIDE_COSMO = 64
# SHAPE_COSMO_MAP = (NPIX_SIDE_COSMO, NPIX_SIDE_COSMO) # Spatial shape for lmap
# WCS_COSMO_MAP = None # Must be a valid enmap WCS object
# LMAX_CAMB_COSMO = int(1.5 * NPIX_SIDE_COSMO)
# FIDUCIAL_COSMO_PARAMS_DICT = {
#     'omch2': 0.120, 'omk': 0.0, 'tau': 0.054,
#     'As': 2.1e-9, 'ns': 0.965
# }
# PRIOR_BOUNDS_PHI_CMB_TUPLE = ( # (mins_tensor, maxs_tensor)
#     torch.tensor([0.5, 60.0, 0.020]), # sigma_min, h0_min, ombh2_min
#     torch.tensor([1.5, 80.0, 0.025])  # sigma_max, h0_max, ombh2_max
# )
# --- End Global Configurations Placeholder ---

sigma_eps = 1e-6 # Unused in this snippet, but kept from original
sigma_min, sigma_max = 0.04, 0.4

OMCH2_FID = 0.122 # Cold dark matter density omega_c * h^2
OMK_FID = 0.0    # Omega_k
TAU_FID = 0.0544 # Optical depth
NS_FID = 0.9649  # Scalar spectral index
AS_FID = 2.1e-9  # Scalar amplitude (ln(10^10 As) = 3.044 => As ~ 2.1e-9)

H0_PRIOR_MIN, H0_PRIOR_MAX = 50.0, 90.0
OMBH2_PRIOR_MIN, OMBH2_PRIOR_MAX = 0.0075, 0.0567 # Note: paper uses omega_b, CAMB uses ombh2
# To convert: omega_b = ombh2 / (H0/100)^2. For priors, it's easier to sample H0 and ombh2 directly.
SIGMA_CMB_PRIOR_MIN, SIGMA_CMB_PRIOR_MAX = 0.1, 1.2 # sigma_min should be >0. Let's use 0.1 for now.


# --- Caching ---
_camb_cls_cache = {}
_lmap_cache_cosmo = {}

# --- Cosmology Specific Likelihood & Prior ---

def get_cosmo_lmap(shape_hw, wcs, device='cpu'): # shape_hw is (H,W)
    """
    Gets or computes and caches the lmap for Fourier operations.
    A more robust key might be needed if WCS varies subtly for the same shape.
    For many use cases, shape and a simple WCS descriptor are enough.
    """
    # Try to create a somewhat unique key from WCS properties that affect lmap
    # This is still a heuristic. The best is to precompute lmap once if WCS is fixed for a shape.
    if wcs is not None:
        # Extract some key WCS parameters that define the geometry for lmap
        # CRVAL might not always be present or consistently named for all projections
        # Using CD matrix (or CDELT) and CRPIX is more fundamental
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
    batch_size = epsilon_cmb_batch.shape[0]
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

class DualAveragingStepSizeHMC: # From GDiff pyhmc.py style, adapted for our HMC
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
        self.initial_step_size = initial_step_size
        # mu is log(10 * initial_step_size) in NUTS paper, can be just log(initial_step_size)
        # For safety with very small step sizes, ensure initial_step_size is positive
        self.mu = np.log(10 * self.initial_step_size) if self.initial_step_size > 1e-9 else -np.inf
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0.0 # scalar error sum
        # log_averaged_step is for the *single* step size being adapted
        self.log_averaged_step = np.log(self.initial_step_size) if self.initial_step_size > 1e-9 else -np.inf

    def update(self, p_accept_scalar): # p_accept is scalar (mean acceptance over chains)
        if self.initial_step_size < 1e-9: # Step size effectively fixed
            return self.initial_step_size, self.initial_step_size

        p_accept_scalar = np.clip(p_accept_scalar, 0.0, 1.0) # Defensive
        self.error_sum += self.target_accept - p_accept_scalar
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        self.t += 1.0
        return torch.exp(torch.tensor(log_step)) # noisy, smoothed

    def get_final_averaged_step_size(self):
        if self.initial_step_size < 1e-9: return self.initial_step_size
        return torch.exp(torch.tensor(self.log_averaged_step)).item()

def compute_mass_matrix_sqrt_from_M(mass_matrix_M):
    if mass_matrix_M is None: return None
    # Simplified: assumes M is (B,D,D) or (D,D) and positive definite
    # For diagonal M (B,D) or (D), sqrt is element-wise.
    if mass_matrix_M.ndim == 1: return mass_matrix_M ** 0.5 # (D)
    if mass_matrix_M.ndim == 2 and mass_matrix_M.shape[0] != mass_matrix_M.shape[1]: # (B,D)
        return mass_matrix_M ** 0.5
    # Full or batched full
    try: # Using Cholesky L, so M = LL^T, then L is M_sqrt
        return torch.linalg.cholesky(mass_matrix_M)
    except torch.linalg.LinAlgError: # Fallback to Eigendecomposition if not PD for Cholesky
        L_eigen, Q_eigen = torch.linalg.eigh(mass_matrix_M)
        if torch.any(L_eigen < -1e-6): raise ValueError("Mass matrix M not PSD for sqrt.")
        L_eigen_sqrt = torch.clamp(L_eigen, min=0.0) ** 0.5
        # M_sqrt = Q @ diag(L_sqrt) @ Q.T, but Q @ diag(L_sqrt) is also a valid sqrt if M_sqrt M_sqrt^T = M
        # For p_actual = M_sqrt @ p_tilde, we need M_sqrt such M_sqrt M_sqrt^T = M
        # Q @ diag(L_sqrt) is one such matrix. (Q @ diag(L_sqrt)) (Q @ diag(L_sqrt))^T = Q L Q^T = M
        return Q_eigen @ torch.diag_embed(L_eigen_sqrt)


def compute_inverse_mass_from_M(mass_matrix_M):
    if mass_matrix_M is None: return None
    if mass_matrix_M.ndim == 1: return 1.0 / mass_matrix_M # (D)
    if mass_matrix_M.ndim == 2 and mass_matrix_M.shape[0] != mass_matrix_M.shape[1]: # (B,D)
        return 1.0 / mass_matrix_M
    try: # Full or batched full
        return torch.linalg.inv(mass_matrix_M)
    except torch.linalg.LinAlgError as e:
        raise ValueError(f"Mass matrix M is singular: {e}")

def _kinetic_energy_hmc(p_actual, inv_mass_matrix_M_inv): # p_actual ~ N(0,M)
    
    # print(inv_mass_matrix_M_inv, inv_mass_matrix_M_inv.ndim)
    
    if inv_mass_matrix_M_inv is None: # M=I
        return 0.5 * (p_actual**2).sum(dim=-1)
    # p_actual: (B,D), inv_mass_matrix_M_inv can be (D), (B,D), (D,D), (B,D,D)
    if inv_mass_matrix_M_inv.ndim == 1: # (D)
        return 0.5 * (p_actual**2 * inv_mass_matrix_M_inv).sum(dim=-1)
    elif inv_mass_matrix_M_inv.ndim == 2:
        if inv_mass_matrix_M_inv.shape[0] == p_actual.shape[0]: # (B,D)
            return 0.5 * (p_actual**2 * inv_mass_matrix_M_inv).sum(dim=-1)
        else: # (D,D)
            M_inv_p = (inv_mass_matrix_M_inv @ p_actual.unsqueeze(-1)).squeeze(-1)
            return 0.5 * (p_actual * M_inv_p).sum(dim=-1)
    elif inv_mass_matrix_M_inv.ndim == 3: # (B,D,D)
        M_inv_p = (inv_mass_matrix_M_inv @ p_actual.unsqueeze(-1)).squeeze(-1)
        return 0.5 * (p_actual * M_inv_p).sum(dim=-1)
    raise ValueError("Invalid inv_mass_matrix_M_inv shape")

def _dKE_dp_actual_hmc(p, inv_mass_matrix): # dK/dp_actual = M^-1 p_actual

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
    
def _hamiltonian_hmc(q, p_actual, log_prob_fn_q, inv_mass_matrix_M_inv):
    potential = -log_prob_fn_q(q)
    kinetic = _kinetic_energy_hmc(p_actual, inv_mass_matrix_M_inv)
    return potential + kinetic

def _leapfrog_hmc(q_curr, p_actual_curr, step_size_val, n_steps,
                  log_grad_fn_q, inv_mass_matrix_M_inv,
                  q_min_bounds=None, q_max_bounds=None): # q_min/max are for parameters q
    
    q = q_curr.clone()
    p_actual = p_actual_curr.clone()

    step_size_val_t = torch.tensor(step_size_val, device=q.device, dtype=q.dtype)

    if q.ndim > 1 and step_size_val_t.ndim == 0 : # if step_size is scalar but q is batched
        step_size_val_t = step_size_val_t.repeat(q.shape[0]).unsqueeze(-1) # (B,1) for broadcasting with grads
    elif step_size_val_t.ndim == 1 and q.ndim > 1 : # step_size is (B)
        step_size_val_t = step_size_val_t.unsqueeze(-1) # (B,1)

    grad_potential_q = -log_grad_fn_q(q) # V_g = - d(logP)/dq; dP/dt = -V_g
    # print('grad_potential_q: ', grad_potential_q.shape)

    p_actual = p_actual - 0.5 * step_size_val_t * grad_potential_q # p_actual update: dp/dt = -dV/dq
    # print('p_actual: ', p_actual.shape, step_size_val_t.shape, step_size_val_t)

    for _ in range(n_steps - 1):
        q = q + step_size_val_t * _dKE_dp_actual_hmc(p_actual, inv_mass_matrix_M_inv) # dq/dt = dK/dp = M^-1 p_actual
        # Boundary checks for q (parameters)
        if q_min_bounds is not None and q_max_bounds is not None:
            for dim_i in range(q.shape[-1]):
                crossed_min = q[..., dim_i] < q_min_bounds[dim_i]
                crossed_max = q[..., dim_i] > q_max_bounds[dim_i]
                if torch.any(crossed_min):
                    q[crossed_min, dim_i] = q_min_bounds[dim_i] + (q_min_bounds[dim_i] - q[crossed_min, dim_i]) # Reflect position
                    p_actual[crossed_min, dim_i] = -p_actual[crossed_min, dim_i] # Reflect momentum
                if torch.any(crossed_max):
                    q[crossed_max, dim_i] = q_max_bounds[dim_i] - (q[crossed_max, dim_i] - q_max_bounds[dim_i])
                    p_actual[crossed_max, dim_i] = -p_actual[crossed_max, dim_i]

        grad_potential_q_new = -log_grad_fn_q(q)
        p_actual = p_actual - step_size_val_t * grad_potential_q_new
    
    q = q + step_size_val_t * _dKE_dp_actual_hmc(p_actual, inv_mass_matrix_M_inv)
    # print('q: ', q.shape)
    if q_min_bounds is not None and q_max_bounds is not None: # Final boundary check for q
        for dim_i in range(q.shape[-1]):
            crossed_min = q[..., dim_i] < q_min_bounds[dim_i]
            crossed_max = q[..., dim_i] > q_max_bounds[dim_i]
            if torch.any(crossed_min):
                q[crossed_min, dim_i] = q_min_bounds[dim_i]
                # p_actual[crossed_min, dim_i] = 0 # Or reflect, but for param bounds, often just clip q
            if torch.any(crossed_max):
                q[crossed_max, dim_i] = q_max_bounds[dim_i]
                # p_actual[crossed_max, dim_i] = 0
    
    grad_potential_q_final = -log_grad_fn_q(q)
    # print('grad_potential_q_final: ', grad_potential_q_final.shape)
    p_actual = p_actual - 0.5 * step_size_val_t * grad_potential_q_final
    
    return q, -p_actual # Negate momentum for detailed balance

def sample_hmc_cosmo(log_prob_fn, log_grad_fn, # For target q (Phi_CMB)
                     phi_init, # Initial q (Phi_CMB) [B_chains, D_phi]
                     mass_matrix_M_input=None, # Mass matrix M [B_chains, D_phi, D_phi] or [D_phi,D_phi] or [D_phi] or None
                     step_size_initial=0.01, # scalar
                     n_leapfrog_steps=5, # int or tuple for random range
                     num_samples_chain=1, # samples to return per chain (after burn-in)
                     num_burnin_steps_hmc=2, # HMC's own burn-in for adaptation
                     adapt_step_size=True,
                     adapt_mass_matrix=False, # Whether to adapt M during HMC burn-in
                     num_adapt_steps_total=50, # Total steps for adaptation phase
                     phi_min_bounds=None, 
                     phi_max_bounds=None, # Tensors [D_phi]
                     verbose=False):
    
    q = phi_init.clone().detach()
    device = q.device
    num_chains, dim_phi = q.shape

    # Initialize M, M_sqrt, M_inv
    current_M = mass_matrix_M_input.clone().detach() if mass_matrix_M_input is not None else None
    # if current_M is not None and current_M.ndim == dim_phi: # Shared M [D,D] or [D]
    #     current_M = current_M.unsqueeze(0).repeat(num_chains, *((1,)*(current_M.ndim)))
    #     current_M = current_M.squeeze(0)
    
    # print('current_M: ', current_M.shape)
    inv_mass_matrix_M_inv = compute_inverse_mass_from_M(current_M)
    mass_matrix_M_sqrt = compute_mass_matrix_sqrt_from_M(current_M)

    # Step size adaptation kernel (one kernel, adapts based on mean acceptance)
    # Step size itself will be a tensor [num_chains]
    current_step_size_val = float(step_size_initial) # scalar for adapter
    step_size_adapter = None
    if adapt_step_size:
        step_size_adapter = DualAveragingStepSizeHMC(current_step_size_val)
    
    # Per-chain step sizes, initialized
    current_step_sizes_per_chain = torch.full((num_chains,), current_step_size_val, device=device, dtype=q.dtype)

    q_collected_for_adapt_M = []
    collected_samples_q = []
    acceptance_rates = []

    total_hmc_iterations = num_burnin_steps_hmc + num_samples_chain

    for i_iter in range(total_hmc_iterations):
        # 1. Sample momentum p_tilde ~ N(0,I), then p_actual = M_sqrt @ p_tilde
        p_tilde = torch.randn_like(q)
        if mass_matrix_M_sqrt is not None:
            if mass_matrix_M_sqrt.ndim == 1: # M_sqrt is [D_phi] (from diagonal M)
                p_actual = mass_matrix_M_sqrt * p_tilde
            elif mass_matrix_M_sqrt.ndim == 2 and mass_matrix_M_sqrt.shape[0] != dim_phi : # (B,D)
                p_actual = mass_matrix_M_sqrt * p_tilde # M_sqrt is diag per chain
            else: # M_sqrt is [D_phi, D_phi] or [B_chains, D_phi, D_phi]
                p_actual = (mass_matrix_M_sqrt @ p_tilde.unsqueeze(-1)).squeeze(-1)
        else: # M = I
            p_actual = p_tilde
        
        p_actual_initial = p_actual.clone()

        # 2. Leapfrog integration
        # Handle random leapfrog steps
        current_n_leap = n_leapfrog_steps
        if isinstance(n_leapfrog_steps, tuple):
            current_n_leap = np.random.randint(n_leapfrog_steps[0], n_leapfrog_steps[1] + 1)
        
        # Use current_step_sizes_per_chain for leapfrog
        q_prop, p_actual_prop = _leapfrog_hmc(
            q, p_actual, current_step_sizes_per_chain, current_n_leap,
            log_grad_fn, inv_mass_matrix_M_inv,
            phi_min_bounds, phi_max_bounds
        )

        # 3. Metropolis-Hastings
        H_initial = _hamiltonian_hmc(q, p_actual_initial, log_prob_fn, inv_mass_matrix_M_inv)
        H_proposed = _hamiltonian_hmc(q_prop, p_actual_prop, log_prob_fn, inv_mass_matrix_M_inv)

        # print(q.shape, q_prop.shape, p_actual_initial.shape, p_actual_prop.shape)
        # print('H: ', H_initial.shape, H_proposed.shape)

        log_accept_ratio = H_initial - H_proposed # For N(0,M) momentum, proposal is symmetric
        
        # Handle NaNs/Infs in log_accept_ratio (e.g. from out of bounds proposals)
        log_accept_ratio = torch.nan_to_num(log_accept_ratio, nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)

        accept_prob = torch.exp(torch.clamp(log_accept_ratio, max=0.0)) # Numerical stability
        u = torch.rand(num_chains, device=device)
        accepted_mask = u < accept_prob
        
        # print('accpt/q: ', accepted_mask.shape, q.shape)
        q[accepted_mask] = q_prop[accepted_mask].detach() # Update q for accepted proposals
        acceptance_rates.append(accepted_mask.float().mean().item())

        # 4. Adaptation phase (during HMC burn-in)
        if i_iter < num_burnin_steps_hmc:
            if adapt_step_size and step_size_adapter:
                # The __call__ method for step_size_adapter from GDiff style HMC
                # takes (current_adapt_step, accept_prob_batch, total_adapt_steps)
                # Here, num_burnin_steps_hmc acts as total_adapt_steps for this HMC call
                current_step_sizes_per_chain = step_size_adapter.update(accept_prob.mean().item())

            if adapt_mass_matrix:
                q_collected_for_adapt_M.append(q.clone().detach())
                # Adapt M at specified point, e.g., 3/4 of HMC burn-in
                # This schedule needs to be robust.
                if num_burnin_steps_hmc > dim_phi and i_iter == (3 * num_burnin_steps_hmc) // 4 :
                    if len(q_collected_for_adapt_M) > dim_phi: # Need enough samples
                        # Use a window of recent samples
                        window_start = max(0, len(q_collected_for_adapt_M) - num_burnin_steps_hmc // 2)
                        samples_for_M = torch.stack(q_collected_for_adapt_M[window_start:], dim=0) # [N_collected, B_chains, D_phi]
                        # Average over chains or estimate per chain if M is per chain
                        # For shared M, average samples across chains then compute covariance:
                        # samples_for_M_flat = samples_for_M.transpose(0,1).reshape(-1, dim_phi) # [N_coll*B_chains, D_phi]
                        # estimated_M = torch.cov(samples_for_M_flat.T) + torch.eye(dim_phi,device=device)*1e-6 # [D_phi,D_phi]
                        
                        # For batched M (per chain):
                        estimated_M_batched = torch.zeros_like(current_M) if current_M is not None else torch.eye(dim_phi, device=device).unsqueeze(0).repeat(num_chains,1,1)
                        any_M_updated = False
                        for chain_idx in range(num_chains):
                            chain_samples = samples_for_M[:, chain_idx, :] # [N_collected, D_phi]
                            if chain_samples.shape[0] > dim_phi : # Ensure enough samples per chain
                                cov_matrix = torch.cov(chain_samples.T)
                                # Ensure positive definiteness
                                cov_matrix += torch.eye(dim_phi, device=device) * 1e-6 
                                estimated_M_batched[chain_idx] = cov_matrix
                                any_M_updated = True
                        
                        if any_M_updated:
                            current_M = estimated_M_batched.detach()
                            inv_mass_matrix_M_inv = compute_inverse_mass_from_M(current_M)
                            mass_matrix_M_sqrt = compute_mass_matrix_sqrt_from_M(current_M)
                            if verbose: print(f"HMC Iter {i_iter}: Mass matrix M updated.")
                        q_collected_for_adapt_M = [] # Reset
        
        # Collect samples after HMC burn-in
        if i_iter >= num_burnin_steps_hmc:
            collected_samples_q.append(q.clone().detach())

    final_q_samples = torch.stack(collected_samples_q, dim=1) if collected_samples_q else q.unsqueeze(1) # [B_chains, N_samples_chain, D_phi]
    
    # print(current_step_sizes_per_chain, current_step_sizes_per_chain.shape)
    final_step_size = current_step_sizes_per_chain.item() # Report one step size
    if adapt_step_size and step_size_adapter and num_burnin_steps_hmc > 0 :
        final_step_size = step_size_adapter.get_final_averaged_step_size()

    return final_q_samples, final_step_size, current_M, np.mean(acceptance_rates[-num_samples_chain:]) if acceptance_rates else 0.0


# --- Example Main Block (Illustrative) ---
if __name__ == '__main__':
    print("Running __HMC_cosmo__ ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Define Global constants for the test ---
    NPIX_SIDE_COSMO = 32 # Smaller for much quicker test
    SHAPE_COSMO_MAP_HW = (NPIX_SIDE_COSMO, NPIX_SIDE_COSMO)
    
    # Dummy WCS for enmap. If enmap is crucial and hard to install, skip lmap-dependent tests for now
    try:
        WCS_COSMO_MAP_TEST = enmap.zeros(SHAPE_COSMO_MAP_HW, proj="car").wcs
    except Exception as e:
        print(f"Failed to create dummy WCS with enmap: {e}. Some tests might be limited.")
        WCS_COSMO_MAP_TEST = None

    LMAX_CAMB_COSMO = int(1.5 * NPIX_SIDE_COSMO)
    FIDUCIAL_COSMO_PARAMS_DICT = {
        'omch2': 0.120, 'omk': 0.0, 'tau': 0.054, 'As': 2.1e-9, 'ns': 0.965
    }
    PRIOR_BOUNDS_PHI_CMB_TUPLE = get_phi_cmb_parameter_bounds(
        sigma_min=0.8, sigma_max=1.2, h0_min=65.0, h0_max=75.0, ombh2_min=0.021, ombh2_max=0.023, device=device
    )
    
    # --- Test HMC Sampler ---
    print("\n--- Testing HMC Sampler for Phi_CMB ---")
    num_hmc_chains = 2
    dim_phi_cmb = 3
    
    # Create a dummy target: Gaussian centered at true_phi_cmb
    true_phi_cmb = torch.tensor([1.0, 70.0, 0.022], device=device)
    
    # For HMC, log_prob_fn is the log posterior of Phi_CMB
    # We need a dummy epsilon_cmb and lmap for the likelihood part
    if WCS_COSMO_MAP_TEST:
        lmap_for_hmc_test = get_cosmo_lmap(SHAPE_COSMO_MAP_HW, WCS_COSMO_MAP_TEST, device=device)
        dummy_epsilon = torch.randn(num_hmc_chains, 1, NPIX_SIDE_COSMO, NPIX_SIDE_COSMO, device=device) * 5.0

        def target_log_prob(phi_cmb_batch): # phi_cmb_batch is [Num_HMC_Chains, 3]
            log_p = log_prior_phi_cmb(phi_cmb_batch, PRIOR_BOUNDS_PHI_CMB_TUPLE)
            valid_mask = ~torch.isinf(log_p)
            if torch.any(valid_mask):
                # Simplified: Use a Gaussian likelihood centered at true_phi_cmb for testing HMC itself
                # log_l = -0.5 * torch.sum((phi_cmb_batch[valid_mask] - true_phi_cmb.unsqueeze(0))**2 / (0.1**2), dim=1)
                # OR use the actual CMB likelihood (can be slow for many HMC steps if CAMB is slow)
                log_l_cmb = torch.zeros_like(log_p)
                log_l_cmb[valid_mask] = log_likelihood_cmb_phi(
                     phi_cmb_batch[valid_mask], dummy_epsilon[valid_mask], lmap_for_hmc_test,
                     LMAX_CAMB_COSMO, psd_regularizer=1e-20
                )
                log_p = log_p + log_l_cmb # Add log-likelihood only where prior is valid
            return log_p
    else: # Fallback if enmap/lmap is not available
        print("Warning: enmap/WCS not available, using simplified Gaussian target for HMC test.")
        def target_log_prob(phi_cmb_batch):
            log_p = log_prior_phi_cmb(phi_cmb_batch, PRIOR_BOUNDS_PHI_CMB_TUPLE)
            valid_mask = ~torch.isinf(log_p)
            if torch.any(valid_mask):
                 log_l = -0.5 * torch.sum((phi_cmb_batch[valid_mask] - true_phi_cmb.unsqueeze(0))**2 / (torch.tensor([0.1, 2.0, 0.0005], device=device)**2), dim=1)
                 log_p[valid_mask] = log_p[valid_mask] + log_l
            return log_p


    def target_log_grad(phi_cmb_batch_grad):
        phi_clone = phi_cmb_batch_grad.clone().requires_grad_(True)
        logp_val = target_log_prob(phi_clone)
        
        valid_mask = ~torch.isinf(logp_val)
        grad_phi = torch.zeros_like(phi_clone)
        if torch.any(valid_mask):
            # Summing valid logp values for a scalar input to autograd
            # Then, ensure gradient is only taken for phi_clone[valid_mask]
            # This is still tricky with autograd's current API for batched conditional gradients.
            # A loop or more careful masking for autograd might be needed for production.
            # For now, let's try the sum approach, assuming HMC rejection handles bad regions.
            grad_output = torch.autograd.grad(logp_val[valid_mask].sum(), phi_clone, allow_unused=True, retain_graph=False)[0]
            if grad_output is not None: # It could be None if no inputs affected the sum
                 grad_phi = grad_output # The grad should already be zero where logp was -inf if handled correctly
            else: # Fallback if autograd returns None for all
                 pass # grad_phi remains zeros

        return grad_phi.detach()


    initial_q_hmc = torch.stack([ # Start chains near the true value but offset
        PRIOR_BOUNDS_PHI_CMB_TUPLE[0] + (PRIOR_BOUNDS_PHI_CMB_TUPLE[1] - PRIOR_BOUNDS_PHI_CMB_TUPLE[0]) * torch.rand(dim_phi_cmb, device=device)
        for _ in range(num_hmc_chains)
    ])
    print(f"Initial q for HMC: \n{initial_q_hmc}")

    # Test with identity mass matrix first
    print("\nRunning HMC with Identity Mass Matrix...")
    q_samples, final_eps, final_M, acc_rate = sample_hmc_cosmo(
        log_prob_fn=target_log_prob,
        log_grad_fn=target_log_grad,
        q_init=initial_q_hmc.clone(),
        mass_matrix_M_input=None, # Identity M
        step_size_initial=0.01,
        n_leapfrog_steps=(5,15),
        num_samples_chain=10,
        num_burnin_steps_hmc=20, # HMC adaptation steps
        adapt_step_size=True,
        adapt_mass_matrix=False, # Keep M fixed as identity for this test
        q_min_bounds=PRIOR_BOUNDS_PHI_CMB_TUPLE[0],
        q_max_bounds=PRIOR_BOUNDS_PHI_CMB_TUPLE[1],
        verbose=False # Set to True for more HMC printouts
    )
    print(f"HMC finished. Sample shape: {q_samples.shape}") # [N_chains, N_samples, D_phi]
    print(f"Mean accepted q (last sample from each chain): \n{q_samples[:, -1, :].mean(0)}")
    print(f"Final HMC step size: {final_eps:.4e}, Acceptance rate: {acc_rate:.3f}")

    # Test with mass matrix adaptation
    print("\nRunning HMC with Mass Matrix Adaptation...")
    q_samples_adaptM, final_eps_adaptM, final_M_adaptM, acc_rate_adaptM = sample_hmc_cosmo(
        log_prob_fn=target_log_prob,
        log_grad_fn=target_log_grad,
        q_init=initial_q_hmc.clone(),
        mass_matrix_M_input=None, # Start with Identity M, let it adapt
        step_size_initial=0.01,
        n_leapfrog_steps=(5,15),
        num_samples_chain=10,
        num_burnin_steps_hmc=50, # More steps for M adaptation
        adapt_step_size=True,
        adapt_mass_matrix=True, # Enable M adaptation
        num_adapt_steps_total=50, # Matches num_burnin_steps_hmc for this HMC call
        q_min_bounds=PRIOR_BOUNDS_PHI_CMB_TUPLE[0],
        q_max_bounds=PRIOR_BOUNDS_PHI_CMB_TUPLE[1],
        verbose=False
    )
    print(f"HMC with M adapt finished. Sample shape: {q_samples_adaptM.shape}")
    print(f"Mean accepted q (last sample from each chain): \n{q_samples_adaptM[:, -1, :].mean(0)}")
    print(f"Final HMC step size: {final_eps_adaptM:.4e}, Acceptance rate: {acc_rate_adaptM:.3f}")
    if final_M_adaptM is not None:
        print(f"Adapted Mass Matrix M (first chain example): \n{final_M_adaptM[0]}")

    print("\nCosmology HMC module test complete.")
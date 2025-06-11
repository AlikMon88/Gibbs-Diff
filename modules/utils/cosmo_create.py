import numpy as np
import matplotlib.pyplot as plt
import os
import camb
from camb import model, initialpower
import pixell.enmap as enmap # For flat-sky map operations
from pixell import curvedsky # For generating GRFs from Cls
import healpy as hp # Often used by CAMB for Cls
import warnings
import time
import torch
import torch.nn as nn
import importlib
import cv2
warnings.filterwarnings('ignore')

from astropy.io import fits # Example for FITS files
import os

# --- Configuration ---
# Map properties
NPIX_SIDE = 256  # Number of pixels on a side
PIX_SIZE_ARCMIN = 8.0  # Pixel size in arcminutes
SHAPE, WCS = enmap.geometry(pos=(0,0), shape=(NPIX_SIDE, NPIX_SIDE), res=np.deg2rad(PIX_SIZE_ARCMIN/60.), proj="car")
# SHAPE will be (NPIX_SIDE, NPIX_SIDE), WCS is the world coordinate system object

# Cosmological Parameters (Fiducial values from paper's footnote 8)
H0_FID = 67.5 # Example, adjust if paper specifies exact CAMB H0
OMBH2_FID = 0.022 # Baryon density omega_b * h^2
OMCH2_FID = 0.122 # Cold dark matter density omega_c * h^2
OMK_FID = 0.0    # Omega_k
TAU_FID = 0.0544 # Optical depth
NS_FID = 0.9649  # Scalar spectral index
AS_FID = 2.1e-9  # Scalar amplitude (ln(10^10 As) = 3.044 => As ~ 2.1e-9)

# Noise parameter Phi = (sigma_cmb, H0_cosmo, ombh2_cosmo)
# Priors (from paper Section 3.2)
H0_PRIOR_MIN, H0_PRIOR_MAX = 50.0, 90.0
OMBH2_PRIOR_MIN, OMBH2_PRIOR_MAX = 0.0075, 0.0567 # Note: paper uses omega_b, CAMB uses ombh2
# To convert: omega_b = ombh2 / (H0/100)^2. For priors, it's easier to sample H0 and ombh2 directly.
SIGMA_CMB_PRIOR_MIN, SIGMA_CMB_PRIOR_MAX = 0.1, 1.2 # sigma_min should be >0. Let's use 0.1 for now.


SIMULATION_DICT = {
    "/home/am3353/Gibbs-Diff/data/cosmo/t_750/dens_t750.fits": {
        "run_id": "b.1p.01", 
        "timesteps": "t_750"
    }, 
    "/home/am3353/Gibbs-Diff/data/cosmo/t_700/dens_t700.fits": {
        "run_id": "b.1p.01", 
        "timesteps": "t_700"
    }, 
    "/home/am3353/Gibbs-Diff/data/cosmo/t_650/dens_t650.fits": {
        "run_id": "b.1p.01", 
        "timesteps": "t_650"
    }, 
    "/home/am3353/Gibbs-Diff/data/cosmo/t_600/dens_t600.fits": {
        "run_id": "b.1p.01", 
        "timesteps": "t_600"
    },
    "/home/am3353/Gibbs-Diff/data/cosmo/t_550/dens_t550.fits": {
        "run_id": "b.1p.01", 
        "timesteps": "t_550"
    }
}

OUTPUT_DIR = '/home/am3353/Gibbs-Diff/data/cosmo/created_data'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TARGET_MAP_SIZE = (NPIX_SIDE, NPIX_SIDE)

### ---------------- INTERSTELLER-DUSST-MAP ---------------------

def create_column_density_map_from_cube(density_cube_3d, axis=0):
    """
    Creates a 2D column density map by integrating along a specified axis.
    """
    if density_cube_3d.ndim != 3: # Basic check
        raise ValueError(f"Input must be a 3D density cube. Got {density_cube_3d.ndim} dimensions.")
    # Shape check will be done before calling or after augmentation
    return np.sum(density_cube_3d, axis=axis)

def augment_3d_cube(cube):
    """
    Applies random augmentations (rotations, flips) to a 3D cube.
    Returns the augmented cube.
    """
    augmented_cube = cube.copy()

    # 1. Random Flips (along each axis with 50% probability)
    if np.random.rand() > 0.5:
        augmented_cube = np.flip(augmented_cube, axis=0)
    if np.random.rand() > 0.5:
        augmented_cube = np.flip(augmented_cube, axis=1)
    if np.random.rand() > 0.5:
        augmented_cube = np.flip(augmented_cube, axis=2)

    # 2. Random Rotations (90-degree increments around random axis)
    # More complex rotations are possible but add complexity with non-cubic results
    # and interpolation. For now, stick to 90-degree rotations which preserve grid.
    num_rotations = np.random.randint(0, 4) # 0, 1, 2, or 3 ninety-degree rotations
    rot_axis_idx = np.random.randint(0, 3)  # 0 for x, 1 for y, 2 for z (axes to rotate around)
    
    axes_to_rotate = [(1,2), (0,2), (0,1)][rot_axis_idx] # (axes[0], axes[1]) for rotate function

    if num_rotations > 0:
        # Scipy.ndimage.rotate uses degrees.
        # Note: scipy.ndimage.rotate can change array shape if not careful,
        # and introduces interpolation. For exact 90-deg rotations without shape change
        # or interpolation, np.rot90 is better.
        augmented_cube = np.rot90(augmented_cube, k=num_rotations, axes=axes_to_rotate)
        
    return augmented_cube

def generate_single_dust_map(sub_shape=(64, 64), verbose=False):
    """
    Generates a single, randomly augmented 2D dust column density map.
    It randomly selects a 3D cube from the pre-scanned paths,
    applies random augmentations, and integrates along a random axis.

    Returns:
        np.ndarray: A 2D dust map, or None if no source cubes are available or an error occurs.
    """

    try:
        # 1. Randomly select a 3D density cube path
        cube_path = np.random.choice(list(SIMULATION_DICT.keys()))
        if verbose: print(f"Selected cube: {cube_path}")

        # 2. Load the 3D density cube
        with fits.open(cube_path) as hdul:
            density_cube_3d = hdul[0].data
        
        if density_cube_3d.shape != (NPIX_SIDE, NPIX_SIDE, NPIX_SIDE):
            if verbose: print(f"Warning: Cube {cube_path} has unexpected shape {density_cube_3d.shape}. Skipping.")
            return None # Or retry with another cube

        # 3. Apply random 3D augmentations
        augmented_cube = augment_3d_cube(density_cube_3d)

        # 4. Randomly select an integration axis
        integration_axis = np.random.randint(0, 3) # 0, 1, or 2
        
        # 5. Create column density map
        col_dens_map = create_column_density_map_from_cube(augmented_cube, axis=integration_axis)
        
        if col_dens_map.shape != TARGET_MAP_SIZE:
             if verbose: print(f"Warning: Generated map has shape {col_dens_map.shape}, expected {TARGET_MAP_SIZE}. This might indicate an issue in augmentation or projection.")
             # Potentially add cropping/resizing here if augmentations can change shape,
             # but np.rot90 with 90-deg steps on cubic arrays should preserve shape.
             return None
        col_dens_map = cv2.resize(col_dens_map, sub_shape)
        return col_dens_map

    except Exception as e:
        if verbose: print(f"Error generating single dust map from {cube_path if 'cube_path' in locals() else 'N/A'}: {e}")
        return None


### ------------- CMB-SIGNAL-MAP -------------

def get_camb_cls(H0, ombh2, omch2=OMCH2_FID, omk=OMK_FID, tau=TAU_FID,
                 As=AS_FID, ns=NS_FID, lmax=3*NPIX_SIDE): # lmax somewhat larger than Nyquist
    """
    Uses CAMB to calculate TT power spectra.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0) # r=0 for no tensors
    pars.set_for_lmax(lmax, lens_potential_accuracy=0) # lens_potential_accuracy=0 if not lensing

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK') # Get Cls in uK^2
    cl_tt = powers['total'][:, 0] # TT spectrum (0-indexed for l, so cl_tt[l] is C_l^TT)
    
    # CAMB returns Cls from l=0. We need l=0 to lmax.
    # Ensure cl_tt has length lmax+1. CAMB might return up to lmax_calc.
    if len(cl_tt) > lmax + 1:
        cl_tt = cl_tt[:lmax+1]
    elif len(cl_tt) < lmax + 1:
        # Pad with zeros if CAMB didn't compute up to lmax (shouldn't happen with set_for_lmax)
        cl_tt = np.pad(cl_tt, (0, lmax + 1 - len(cl_tt)), 'constant')

    # Remove monopole and dipole from Cls for map generation (often done)
    cl_tt[0] = 0 
    cl_tt[1] = 0
    return cl_tt # Units of uK^2

def generate_cmb_map(cl_tt, sigma_cmb_amp, sub_shape = (64, 64), seed=None):
    """
    Generates a flat-sky CMB map realization from Cls using pixell.
    cl_tt should be the power spectrum D_l = l(l+1)C_l/2pi or C_l.
    pixell.curvedsky.rand_map expects C_l.
    sigma_cmb_amp is the overall amplitude scaling factor mentioned in paper (Phi).
    """
    # The Cls from CAMB are C_l.
    # The sigma_cmb_amp from the paper seems to be a direct multiplier on the *covariance*,
    # so it's a multiplier on Cls (power), or on the map std. dev. if it's sqrt(power).
    # Let's assume sigma_cmb_amp scales the *standard deviation* of the CMB map.
    # So, C_l_scaled = (sigma_cmb_amp^2) * C_l_fiducial
    # However, the paper describes sigma as part of Phi, which parametrizes Sigma_Phi.
    # If Sigma_Phi = sigma^2 * Sigma_phi_base, then C_l_effective = sigma^2 * C_l_base
    
    ''' Technically - the Sigma should only scale the diag-cov in log-post calc'''
    scaled_cl_tt = cl_tt * (sigma_cmb_amp**2) # Scale power spectrum

    # pixell.curvedsky.rand_map needs an array of Cls [TT, EE, BB, TE, ...]
    # For TT only:
    cls_for_randmap = np.zeros((1, len(scaled_cl_tt))) # Shape (1, nl) for just T
    cls_for_randmap[0, :] = scaled_cl_tt
    
    cmb_map_data = curvedsky.rand_map(SHAPE, WCS, cls_for_randmap, seed=seed)
    cmb_map_data = enmap.ndmap(cmb_map_data, WCS) # Returns a single map (T)
    cmb_map_data = cv2.resize(cmb_map_data, sub_shape)

    return cmb_map_data

# def get_cmb_noise_batch(phi_cmb_batch, device):

#     H0_batch, ombh2_batch = phi_cmb_batch[:, 0].cpu().numpy(), phi_cmb_batch[:, 1].cpu().numpy()
#     cl_tt = get_camb_cls(H0_batch, ombh2_batch)

#     cmb_map_data = generate_cmb_map(cl_tt, sigma_cmb_amp, sub_shape = (64, 64), seed=None)
#     cmb_map_data = torch.tensor(np.array(cmb_map_data))
    
#     return cmb_map_data.to(device).unsqueeze(1)

### ---------- NOISE-CREATE (batched) ------------------

# ==============================================================================
# BATCHED HELPER 1: Looped function for getting a batch of CAMB power spectra
# ==============================================================================
def get_camb_cls_batch(H0_batch, ombh2_batch, omch2=OMCH2_FID, omk=OMK_FID, 
                       tau=TAU_FID, As=AS_FID, ns=NS_FID, lmax=LMAX):
    """
    Uses CAMB to calculate a batch of TT power spectra by looping over input parameters.

    Args:
        H0_batch (np.ndarray): Array of H0 values.
        ombh2_batch (np.ndarray): Array of ombh2 values.
    
    Returns:
        np.ndarray: A NumPy array of shape (batch_size, lmax + 1) containing TT power spectra.
    """
    all_cls = []
    # This loop is necessary as CAMB computes one cosmology at a time.
    for H0, ombh2 in zip(H0_batch, ombh2_batch):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, tau=tau)
        pars.InitPower.set_params(As=As, ns=ns, r=0)
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        cl_tt = powers['total'][:, 0]

        # Ensure correct length and remove monopole/dipole
        if len(cl_tt) > lmax + 1:
            cl_tt = cl_tt[:lmax+1]
        elif len(cl_tt) < lmax + 1:
            cl_tt = np.pad(cl_tt, (0, lmax + 1 - len(cl_tt)))
            
        cl_tt[0] = 0
        cl_tt[1] = 0
        all_cls.append(cl_tt)
        
    return np.array(all_cls)

# ==============================================================================
# BATCHED HELPER 2: Vectorized function for generating a batch of CMB maps
# ==============================================================================
def generate_cmb_map_batch(cl_tt_batch, sigma_cmb_amp, sub_shape=(64, 64), seed=None):
    """
    Generates a batch of flat-sky CMB map realizations from a batch of Cls.
    This operation is fully vectorized.
    """
    batch_size = cl_tt_batch.shape[0]
    
    # 1. Define the map geometry for the target shape
    # We create a shape (n_comp, height, width) where n_comp=1 for TT-only
    map_shape, wcs = enmap.geometry(pos=np.deg2rad([[0,0],[1,1]]), shape=(1, *sub_shape), proj="car")
    
    # 2. Define the full output shape for the batch
    full_batch_shape = (batch_size, *map_shape) # e.g., (16, 1, 64, 64)
    
    # 3. Scale the power spectra by the amplitude factor
    scaled_cl_tt_batch = cl_tt_batch * (sigma_cmb_amp**2)
    
    # 4. Reshape power spectra for enmap: (batch, n_comp, n_comp, lmax)
    # This shape allows enmap to generate correlated components, but here we only have one.
    ps_batch_for_randmap = scaled_cl_tt_batch[:, None, None, :]
    
    # 5. Generate all maps in a single, efficient call
    cmb_maps_batch = enmap.rand_map(full_batch_shape, wcs, ps_batch_for_randmap, seed=seed)
    
    return cmb_maps_batch


# ==============================================================================
# FINAL BATCHED FUNCTION: The main orchestrator
# ==============================================================================
def get_cmb_noise_batch(phi_cmb_batch, device, sigma_cmb_amp=1.0):
    """
    Generates a batch of CMB noise maps based on a batch of cosmological parameters.
    This version is vectorized for efficient batch processing.

    Args:
        phi_cmb_batch (torch.Tensor): Tensor of cosmological parameters of shape (batch_size, 2).
        device (torch.device): The device to place the final output tensor on.
        sigma_cmb_amp (float): Amplitude scaling factor for the power spectrum.

    Returns:
        torch.Tensor: A batch of CMB maps of shape (batch_size, 1, height, width).
    """
    # 1. Extract parameters and move to CPU for CAMB
    H0_batch, ombh2_batch = phi_cmb_batch[:, 0].cpu().numpy(), phi_cmb_batch[:, 1].cpu().numpy()
    
    # 2. Get a batch of power spectra (this part is looped internally)
    cl_tt_batch = get_camb_cls_batch(H0_batch, ombh2_batch)

    # 3. Generate all CMB maps in one vectorized operation
    #    This returns an enmap.ndmap of shape (batch_size, 1, height, width)
    cmb_map_data_batch = generate_cmb_map_batch(cl_tt_batch, sigma_cmb_amp, sub_shape=(64, 64))
    
    # 4. Convert the final numpy-like array to a torch tensor directly on the target device
    #    The channel dimension (axis 1) is already present.
    return torch.as_tensor(cmb_map_data_batch, dtype=torch.float32, device=device)


#### --------- DATASET-CREATION --------------

def generate_dataset_sample(index, seed_offset=0, verbose=False):
    """Generates one sample: dust_map, cmb_map, mixed_map, and cmb_params."""
    current_seed_dust = index + seed_offset 
    current_seed_cmb = index + seed_offset + NUM_SAMPLES_TO_GENERATE # ensure different seeds

    # 1. Get Dust map
    # dust_map = generate_dust_map(index) # Use index for dust map selection if you have a list
    dust_map = generate_single_dust_map() 

    # 2. Sample CMB parameters from prior
    h0_sample = np.random.uniform(H0_PRIOR_MIN, H0_PRIOR_MAX)
    ombh2_sample = np.random.uniform(OMBH2_PRIOR_MIN, OMBH2_PRIOR_MAX)
    sigma_cmb_sample = np.random.uniform(SIGMA_CMB_PRIOR_MIN, SIGMA_CMB_PRIOR_MAX)
    
    cmb_params = {
        'H0': h0_sample,
        'ombh2': ombh2_sample,
        'sigma_cmb': sigma_cmb_sample
    }

    # 3. Generate CMB map
    cls_tt_sample = get_camb_cls(H0=h0_sample, ombh2=ombh2_sample)
    cmb_map = generate_cmb_map(cls_tt_sample, sigma_cmb_amp=sigma_cmb_sample, seed=current_seed_cmb)

    if verbose:
    # 4. Create mixed map
        print(type(dust_map), dust_map.shape, cmb_map.shape)
        print('Dust-Map: ', min(np.ravel(dust_map)), max(np.ravel(dust_map)))
        print('CMB-Map: ', min(np.ravel(cmb_map)), max(np.ravel(cmb_map)))

    ## z-score norm (standardization)
    dust_map = (dust_map - np.mean(dust_map)) / np.std(dust_map)
    cmb_map = (cmb_map - np.mean(cmb_map)) / np.std(cmb_map)

    if verbose:
        print(type(dust_map), dust_map.shape, cmb_map.shape)
        print('Dust-Map: ', min(np.ravel(dust_map)), max(np.ravel(dust_map)))
        print('CMB-Map: ', min(np.ravel(cmb_map)), max(np.ravel(cmb_map)))

    ## min-max scaling (Norm) ~ [0, 1]/custom descaling
    # dust_map = (dust_map - np.min(dust_map)) / (np.max(dust_map) - np.min(dust_map))
    # cmb_map = (cmb_map - np.min(cmb_map)) / (np.max(cmb_map) - np.min(cmb_map))

    # print(type(dust_map), dust_map.shape, cmb_map.shape)
    # print('Dust-Map: ', min(np.ravel(dust_map)), max(np.ravel(dust_map)))
    # print('CMB-Map: ', min(np.ravel(cmb_map)), max(np.ravel(cmb_map)))

    mixed_map = dust_map + cmb_map
    if verbose:
        print('Mixed-Map: ', min(np.ravel(cmb_map)), max(np.ravel(cmb_map)))
    
    # 5. Save data
    enmap.write_map(os.path.join(OUTPUT_DIR, "dust_maps", f"dust_{index:04d}.fits"), dust_map)
    enmap.write_map(os.path.join(OUTPUT_DIR, "cmb_maps", f"cmb_{index:04d}.fits"), cmb_map)
    enmap.write_map(os.path.join(OUTPUT_DIR, "mixed_maps", f"mixed_{index:04d}.fits"), mixed_map)
    params_dir = os.path.join(OUTPUT_DIR, "params")
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    np.save(os.path.join(params_dir, f"params_{index:04d}.npy"), cmb_params)
    # Dust map is saved by load_or_generate_dust_map

    ## (1, 256, 256)
    return np.expand_dims(dust_map, axis=0), np.expand_dims(cmb_map, axis=0), np.expand_dims(mixed_map, axis=0), cmb_params


def generate_mixed_dataset(NUM_SAMPLES_TO_GENERATE, verbose=False):
    print(f"\nGenerating {NUM_SAMPLES_TO_GENERATE} dataset samples...")
    dust_maps, cmb_maps, mixed_maps, params_list = [], [], [], []
    for i in range(NUM_SAMPLES_TO_GENERATE):
        if verbose:
            print(f"Generating sample {i+1}/{NUM_SAMPLES_TO_GENERATE}")
        d_map, c_map, m_map, params = generate_dataset_sample(i)
        
        dust_maps.append(d_map) 
        cmb_maps.append(c_map) 
        mixed_maps.append(m_map)
        params_list.append(params)
    
    print("Dataset generation complete.")
    return np.array(dust_maps), np.array(cmb_maps), np.array(mixed_maps), np.array(params_list) 



if __name__ == '__main__':

    ## We test with few samples first (PASS-SUBSHAPE)
    NUM_SAMPLES_TO_GENERATE = 1000
    ft = time.time()
    dust_maps, cmb_maps, mixed_maps, params_list = generate_mixed_dataset(NUM_SAMPLES_TO_GENERATE)
    lt = time.time()
    print('time-taken (mixed-map-generation): ', (lt - ft)/60, ' mins')
    print(dust_maps.shape, cmb_maps.shape, mixed_maps.shape, params_list.shape)
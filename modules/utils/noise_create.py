import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def create_1d_data(n_depth = 100, n_samples=1000, decay=0.1):
    
    signal_arr, noise_arr, obs_arr = [], [], []

    for _ in range(n_samples):

        x_vals = np.linspace(-5, 5, n_depth)
        signal = np.sin(x_vals)
        noise = np.random.normal(0, 1, size=n_depth)

        _r = np.random.rand(n_depth)
        observation = _r * signal + (1 - _r) * decay * noise

        signal_arr.append(signal)
        noise_arr.append(noise)
        obs_arr.append(observation)

    return np.array(obs_arr), np.array(signal_arr), np.array(noise_arr) 

def get_colored_noise_1d(shape, phi=0.0, device=None):
    """
    Generate 1D colored noise using frequency-domain shaping, with proper device management.
    
    Args:
        shape: tuple (B, L) - batch size and signal length
        device: torch.device where tensors should be placed ('cpu' or 'cuda')
        phi: float or tensor of shape (B, 1) - spectral exponent
    
    Returns:
        noise: 1D colored noise tensor of shape (B, L) on the specified device
        psd: Power spectral density tensor of shape (B, L) on the specified device
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure shape is just 2D
    if len(shape) > 2:
        shape = (shape[0], shape[-1])

    B, L = shape

    # 1. Handle the spectral exponent `phi`
    if isinstance(phi, (float, int)):
        # Create the tensor directly on the target device
        phi = torch.tensor(phi, dtype=torch.float32, device=device).repeat(B, 1)
    else:
        # If phi is already a tensor, move it to the correct device
        phi = phi.to(device)
        if phi.shape != (B, 1):
            raise ValueError(f"phi tensor must have shape ({B}, 1), but got {phi.shape}")

    # 2. Create frequency basis directly on the target device
    freqs = torch.fft.fftfreq(L, device=device)
    S = freqs.pow(2).view(1, L)  # Power spectrum: f^2
    S[:, 0] = 1.0  # Avoid division by zero at DC

    # 3. Shape the spectrum (all tensors are on the correct device)
    S = S.pow(phi / 2)  # (B, L) via broadcasting
    S = S / S.mean(dim=1, keepdim=True)  # Normalize

    # 4. Generate white noise and shape it (all on the correct device)
    X_white = torch.fft.fft(torch.randn(B, L, device=device), dim=1)
    X_shaped = X_white * torch.sqrt(S)
    noise = torch.fft.ifft(X_shaped, dim=1).real

    return noise, S

def create_1d_data_colored(n_samples=1000, n_depth=100, phi=1.0, decay=0.1, sigma=0.5, device=None):
    """
    Generate 1D noisy observations from colored noise and sinusoidal signal using diffusion-style mixing.

    Args:
        n_samples: Number of samples (batch size)
        n_depth: Length of signal
        phi: Spectral exponent for colored noise
        decay: Scaling factor for the noise
        sigma: float or tensor in [0, 1]; can be:
            - scalar
            - shape (n_samples,) or (n_samples, 1)
            - shape (n_samples, n_depth)
        device: PyTorch device

    Returns:
        observations, signals, noises: each of shape (n_samples, n_depth)
    """
    
    device = 'cpu' ## force-run on cpu

    # Create deterministic signal
    x_vals = np.linspace(-5, 5, n_depth)
    signal = np.sin(x_vals)
    signal = torch.tensor(signal, dtype=torch.float32, device=device).repeat(n_samples, 1)

    # Create colored noise
    noise, _ = get_colored_noise_1d((n_samples, n_depth), phi=phi, device=device)
    noise = decay * noise.cpu()

    # Process sigma
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    if sigma.ndim == 0:
        sigma = sigma.view(1, 1).expand(n_samples, n_depth)
    elif sigma.ndim == 1:
        sigma = sigma.view(-1, 1).expand(-1, n_depth)
    elif sigma.shape != (n_samples, n_depth):
        raise ValueError(f"Incompatible sigma shape: got {sigma.shape}, expected scalar, (n_samples,), or (n_samples, n_depth)")

    # Diffusion-style mixing
    sqrt_one_minus_sigma2 = torch.sqrt(1.0 - sigma ** 2)
    observation = sqrt_one_minus_sigma2 * signal + sigma * noise

    return observation.cpu().numpy(), signal.cpu().numpy(), noise.cpu().numpy()

import torch
import numpy as np

def create_1d_data_colored_multi(n_samples=1000, n_depth=100, phi=1.0, decay=0.1, sigma=0.5, device=None):
    """
    Generate 1D noisy observations from colored noise and **randomized sinusoidal signals**.

    Args:
        n_samples: Number of samples (batch size)
        n_depth: Length of signal
        phi: Spectral exponent for colored noise
        decay: Scaling factor for the noise
        sigma: float or tensor in [0, 1]; can be:
            - scalar
            - shape (n_samples,) or (n_samples, 1)
            - shape (n_samples, n_depth)
        device: PyTorch device

    Returns:
        observations, signals, noises: each of shape (n_samples, n_depth)
    """
    
    device = 'cpu'

    x_vals = np.linspace(-5, 5, n_depth)
    x_vals = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, -1).repeat(n_samples, 1)

    # Random signal parameters
    amplitudes = torch.rand(n_samples, 1, device=device) * 1.0 + 0.5     # [0.5, 1.5]
    frequencies = torch.rand(n_samples, 1, device=device) * 2.0 + 0.5   # [0.5, 2.5]
    phases = torch.rand(n_samples, 1, device=device) * 2 * np.pi        # [0, 2π]

    # Create randomized sinusoidal signal per sample
    signal = amplitudes * torch.sin(frequencies * x_vals + phases)

    # Colored noise
    noise, _ = get_colored_noise_1d((n_samples, n_depth), phi=phi, device=device)
    noise = decay * noise.cpu()

    # Process sigma
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    if sigma.ndim == 0:
        sigma = sigma.view(1, 1).expand(n_samples, n_depth)
    elif sigma.ndim == 1:
        sigma = sigma.view(-1, 1).expand(-1, n_depth)
    elif sigma.shape != (n_samples, n_depth):
        raise ValueError(f"Incompatible sigma shape: got {sigma.shape}, expected scalar, (n_samples,), or (n_samples, n_depth)")

    # Diffusion-style mixing
    sqrt_one_minus_sigma2 = torch.sqrt(1.0 - sigma ** 2)
    observation = sqrt_one_minus_sigma2 * signal + sigma * noise

    return observation.cpu().numpy(), signal.cpu().numpy(), noise.cpu().numpy()

if __name__ == '__main__':
    
    print('Running ... __noise_create.py__ ...')

    n_samples = 10000
    rn = random.randint(0, n_samples)

    observation, signal, noise = create_1d_data(n_samples=n_samples)
    print(observation.shape, signal.shape, noise.shape)

    plt.figure(figsize=(10, 5))

    plt.plot(observation[rn], label="Observation", alpha=0.8, marker='.')
    plt.plot(signal[rn], label="Signal", linestyle="dashed", alpha=0.7, color = 'red')
    plt.legend()
    plt.show()

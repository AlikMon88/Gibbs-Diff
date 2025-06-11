import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
import cv2 as cv

def load_images_pil(image_paths, size=(64, 64)):
    
    transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()  # Converts to [0,1] range and shape (C,H,W)
    ])

    images = []
    
    for path in image_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
    
    return torch.stack(images)  # shape: (B, C, H, W)

def load_images_cv2(image_paths, size=(64, 64)):
    
    images = []
    
    for path in image_paths:
        if not os.path.isfile(path):
            print(f"Image not found: {path}")
            continue
        img = cv.imread(path)
        img = cv.resize(img, size)
        img = (img / 255.0).astype('float32')
        img = img.reshape(3, size[0], size[1])
        img = torch.tensor(img)
        images.append(img)

    images = torch.stack(images)
    return images  # shape: (B, C, H, W)


def get_colored_noise_2d(shape, phi=0, device=None):
    """
    Generate 2D colored noise with a given power spectrum exponent phi.

    Args:
        shape: (tuple or torch.Size) expected to be (B, C, H, W), with H == W.
        phi: float or torch.Tensor of shape (B, 1); power spectrum exponent.
        device: optional torch device.

    Returns:
        noises: torch.Tensor of shape (B, C, H, W) — colored noise.
        S: torch.Tensor of shape (B, C, H, W) — normalized power spectrum.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert len(shape) == 4 and shape[2] == shape[3], "Input shape must be (B, C, H, H)"
    B, C, N, _ = shape

    if isinstance(phi, (float, int)):
        phi = torch.tensor(phi, device=device).repeat(B, 1)
    else:
        phi = phi.to(device)
        assert phi.shape == (B, 1), "phi must be of shape (B, 1)"

    # Create 2D frequency grid
    freq = torch.fft.fftfreq(N, device=device).reshape(N, 1)
    freq_grid = torch.zeros((N, N), device=device)
    for i in range(2):  # For 2D grid
        freq_grid += torch.moveaxis(freq, 0, i).pow(2)

    # Expand to match shape (B, C, N, N)
    S = freq_grid[None, None, :, :].repeat(B, C, 1, 1)
    S = S.pow(phi.view(B, 1, 1, 1) / 2)
    S[:, :, 0, 0] = 1.0  # Avoid division by zero at DC

    # Normalize power spectrum
    S /= S.mean(dim=(-2, -1), keepdim=True)

    # Generate white noise in frequency domain
    X_white = torch.fft.fftn(torch.randn(shape, device=device), dim=(2, 3))
    X_shaped = X_white * torch.sqrt(S)

    # Inverse FFT to get colored noise
    noises = torch.fft.ifftn(X_shaped, dim=(2, 3)).real

    return noises, S
        
def create_2d_data_colored(image_paths, n_samples=None, phi=1.0, decay=1.0, sigma=0.5, size=(64, 64), is_plot=False):
    """
    Applies colored noise to RGB images using diffusion-style mixing. (create data on CPU)

    Args:
        image_paths: list of paths to RGB images
        phi: float or tensor (spectral exponent for colored noise)
        decay: scaling factor for noise magnitude
        sigma: scalar, (B,1), or (B,1,1,1) tensor in [0,1]
        size: tuple (H, W) to resize all images
        device: torch.device

    Returns:
        observation, original_image, noise — all tensors of shape (B, C, H, W)
    """

    if not n_samples:
        n_samples = len(image_paths)

    image_paths = image_paths[:n_samples]

    # Load and prepare images
    # images = load_images(image_paths, size=size)
    images = load_images_cv2(image_paths, size=size)
    
    B, C, H, W = images.shape

    if is_plot:

        sample_img = (images[0].view(H, W, -1).numpy() * 255).astype('uint8')
        fig = plt.figure(figsize = (3, 3))
        plt.imshow(sample_img)
        plt.show()

    # Generate 2D colored noise
    noise, _ = get_colored_noise_2d((B, C, H, W), phi=phi)
    noise = decay * noise.cpu()

    # Broadcast sigma
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(sigma, dtype=torch.float32).view(1, 1, 1, 1).expand(B, C, H, W)
    elif isinstance(sigma, torch.Tensor):
        sigma = sigma
        if sigma.ndim == 2 and sigma.shape == (B, 1):  # (B,1) → (B,1,H,W)
            sigma = sigma.view(B, 1, 1, 1).expand(B, C, H, W)
        elif sigma.shape != (B, C, H, W):
            raise ValueError(f"Invalid sigma shape: {sigma.shape}")
    else:
        raise TypeError("sigma must be a float, int, or torch.Tensor")

    # Diffusion-style mixing
    sqrt_one_minus_sigma2 = torch.sqrt(1.0 - sigma ** 2)
    observation = sqrt_one_minus_sigma2 * images + sigma * noise

    ## Linear-mixing
    # observation = images + noise

    rand_perm = torch.randperm(n_samples)

    return observation[rand_perm], images[rand_perm], noise[rand_perm]

if __name__ == '__main__':
    print('running __noise_create_2d.py__')
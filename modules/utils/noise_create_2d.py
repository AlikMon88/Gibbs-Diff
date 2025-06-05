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
    Args:
        shape: (int tuple or torch.Size) shape of the image
        phi: (float or torch.Tensor of shape (B,1)) power spectrum exponent 
        ret_psd: (bool) if True, return the power spectrum
    Returns:
        noise: colored noise
        ps: power spectrum
    """

    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    assert len(shape) == 4 # (B, C, H, W)
    assert shape[2] == shape[3] # (B, C, H, W)
    if isinstance(phi, float) or isinstance(phi, int):
        phi = torch.tensor(phi).to(device).repeat(shape[0], 1)
    else:
        assert phi.shape == (shape[0], 1)
   
    N = shape[2]

    wn = torch.fft.fftfreq(N).to(device).reshape((N, 1))
    S = torch.zeros((shape[0], shape[1], N, N))
    for i in range(2): ## we are in 2D
        S += torch.moveaxis(wn, 0, i).pow(2)

    ## phi - scaling frequency content (control frequency characteristics of noise)
    S.pow_(phi.reshape(-1, 1, 1, 1)/2)
    S[:, :, 0, 0] = 1.0
    S.div_(torch.mean(S, dim=(-1, -2), keepdim=True))  # Normalize S to keep std = 1

    X_white = torch.fft.fftn(torch.randn(shape, device=device), dim=(2,3))
    X_shaped = X_white * torch.sqrt(S)
    noises = torch.fft.ifftn(X_shaped, dim=(2,3)).real
    
    return noises, S
    
def create_2d_data_colored(image_paths, n_samples=None, phi=1.0, decay=1.0, sigma=0.5, size=(64, 64), is_plot=False):
    """
    Applies colored noise to RGB images using diffusion-style mixing.

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
    noise = decay * noise

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
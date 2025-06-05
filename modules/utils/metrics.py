import torch
import numpy as np
from skimage import metrics as skmetrics

def ensure_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def psnr_1d(signal1, signal2, data_range=1.0):
    '''
    PSNR for 1D signals of shape (B, C, L)
    signal1: clean signal
    signal2: denoised signal
    '''
    signal1 = ensure_numpy(signal1)
    signal2 = ensure_numpy(signal2)

    if signal1.ndim == 2:
        signal1 = np.expand_dims(signal1, axis=0)
        signal2 = np.expand_dims(signal2, axis=0)

    psnr_list = [
        skmetrics.peak_signal_noise_ratio(signal1[i], signal2[i], data_range=data_range)
        for i in range(signal1.shape[0])
    ]
    return np.mean(psnr_list)

def l1_loss_1d(signal1, signal2, reduction='mean'):
    '''
    L1 loss for 1D signals of shape (B, C, L)
    '''
    signal1 = ensure_numpy(signal1)
    signal2 = ensure_numpy(signal2)
    abs_diff = np.abs(signal1 - signal2)

    if reduction == 'mean':
        return np.mean(abs_diff)
    elif reduction == 'sum':
        return np.sum(abs_diff)
    else:  # 'none'
        return abs_diff

def psnr_2d(img1, img2, data_range=1.0):
    '''
    PSNR for 2D images of shape (B, C, H, W)
    '''
    img1 = ensure_numpy(img1)
    img2 = ensure_numpy(img2)

    if img1.ndim == 3:
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

    psnr_list = [
        skmetrics.peak_signal_noise_ratio(img1[i], img2[i], data_range=data_range)
        for i in range(img1.shape[0])
    ]
    return np.mean(psnr_list)

def ssim_2d(img1, img2, data_range=1.0):
    '''
    SSIM for 2D images of shape (B, C, H, W)
    '''
    img1 = ensure_numpy(img1)
    img2 = ensure_numpy(img2)

    if img1.ndim == 3:
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

    b, c, h, w = img1.shape

    # Reshape for channel-wise SSIM computation
    # img1 = img1.transpose(0, 2, 3, 1)  # (B, H, W, C)
    # img2 = img2.transpose(0, 2, 3, 1)  # (B, H, W, C)

    ssim_list = [
        skmetrics.structural_similarity(
            img1[i], img2[i], data_range=data_range, channel_axis=1, win_size=3
        )
        for i in range(b)
    ]
    return np.mean(ssim_list)

def l1_loss_2d(img1, img2, reduction='mean'):
    '''
    L1 loss for 2D images of shape (B, C, H, W)
    '''
    img1 = ensure_numpy(img1)
    img2 = ensure_numpy(img2)
    abs_diff = np.abs(img1 - img2)

    if reduction == 'mean':
        return np.mean(abs_diff)
    elif reduction == 'sum':
        return np.sum(abs_diff)
    else:  # 'none'
        return abs_diff

if __name__ == '__main__':
    print('running __metrics.py__')

    shape = (1, 1, 100)
    clean = torch.randn(shape)
    denoised = torch.randn(shape)

    print(clean.shape, denoised.shape, clean.dtype, denoised.dtype)

    psnr = psnr_1d(clean, denoised)
    l1 = l1_loss_1d(clean, denoised)

    print('PSNR:', psnr)
    print('L1:', l1)

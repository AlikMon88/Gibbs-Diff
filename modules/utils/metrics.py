import torch
import numpy as np
from skimage import metrics as skmetrics
import torch.nn.functional as F
import math

def psnr_1d(signal1, signal2, data_range=1.0):
    ''' 
    PSNR for 1D signals of shape (B, C, L)
    signal1: clean signal
    signal2: denoised signal
    '''
    if not len(signal1) == len(signal2):
        raise ValueError('Input signals must have the same number of dimensions.')
    
    if len(signal1) == 2:
        signal1 = np.expand_dims(signal1, axis = 0)
        signal2 = np.expand_dims(signal2, axis = 0)

    psnr_list = []
    for i in range(signal1.shape[0]):
        psnr_list.append(skmetrics.peak_signal_noise_ratio(
            signal1[i], signal2[i], data_range=data_range))
    
    psnr = np.mean(np.array(psnr_list))
    return psnr


def l1_loss_1d(signal1, signal2, reduction='mean'):
    '''
    L1 loss for 1D signals of shape (B, C, L)
    '''
    return F.l1_loss(torch.tensor(signal1), torch.tensor(signal2), reduction=reduction).cpu().numpy()


def psnr_2d(img1, img2, data_range=1.0):
    '''
    PSNR for 2D images of shape (B, C, H, W)
    img1: clean image
    img2: denoised image
    '''
    if not img1.ndim == img2.ndim:
        raise ValueError('Input images must have the same number of dimensions.')
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    psnr_list = []
    for i in range(img1_np.shape[0]):
        psnr_list.append(skmetrics.peak_signal_noise_ratio(
            img1_np[i], img2_np[i], data_range=data_range))
    
    return torch.tensor(psnr_list, device=img1.device)


def ssim_2d(img1, img2, data_range=1.0):
    '''
    SSIM for 2D images of shape (B, C, H, W)
    img1: clean image
    img2: denoised image
    '''
    if not img1.ndim == img2.ndim:
        raise ValueError('Input images must have the same number of dimensions.')
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    ssim_list = []
    for i in range(img1_np.shape[0]):
        # Average SSIM across channels
        ssim_ch = []
        for c in range(img1_np.shape[1]):
            ssim_ch.append(skmetrics.structural_similarity(
                img1_np[i, c], img2_np[i, c],
                data_range=data_range
            ))
        ssim_list.append(np.mean(ssim_ch))
    
    return torch.tensor(ssim_list, device=img1.device)

def l1_loss_2d(img1, img2, reduction='mean'):
    '''
    L1 loss for 2D images of shape (B, C, H, W)
    '''
    return F.l1_loss(img1, img2, reduction=reduction)


if __name__ == '__main__':
    print('running __metrics.py__')

    shape = (1, 1, 100)
    clean, denoised = torch.randn(shape).cpu().numpy(), torch.randn(shape).cpu().numpy()

    print(clean.shape, denoised.shape, clean.dtype, denoised.dtype)

    psnr, l1 = psnr_1d(clean, denoised), l1_loss_1d(clean, denoised)

    print('PSNR: ', psnr)
    print('L1: ', l1)
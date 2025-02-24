import numpy as np
import math
import random
import torch
import torch.nn.functional as F

def psnr_metric(signal1, signal2, max_value=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two 1D signals.

    Args:
        signal1 (torch.Tensor): Original signal (Batch, Length) or (Length,)
        signal2 (torch.Tensor): Reconstructed/Noisy signal (Batch, Length) or (Length,)
        max_value (float): Maximum possible value in the signals (default: 1.0 for normalized signals)

    Returns:
        float: PSNR value in dB
    """
    mse = F.mse_loss(signal1, signal2, reduction='mean')
    
    if mse == 0:
        return float('inf')  # Perfect reconstruction, PSNR is infinite

    psnr = 10 * torch.log10(max_value**2 / mse)
    return psnr.item()

if __name__ == '__main__':
    pass
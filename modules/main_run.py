'''
The Entry-Point to run the SLURM script based .py files | Scale it UP
'''

import os
import numpy as np
import argparse
import sys

import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import math
import warnings
import importlib
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import train_test_split

## ImageNet subset ~1k
import cv2 as cv
import torchvision
import torchvision.datasets

## 1D
from utils.noise_create import *
from comp.one_d.unet_1d import Unet1DGDiff
from comp.one_d.diffusion_1d import GibbsDiff1D

## 2D
from utils.noise_create_2d import *
from comp.two_d.unet_2d import Unet2DGDiff
from comp.two_d.diffusion_2d import GibbsDiff2D

from comp.diff_trainer import TrainerGDiff
from utils.metrics import *


def get_hparams(mode = '1D'):

    if mode == '1D':
        params = {
            'train_num_steps': 10000,
            'seq_len': 100,
            'diffusion_steps': 1000, ## ancestral sampling steps
            'train_batch_size': 16,
            'infer_phi': 1.0,
            'infer_sigma': 0.2,
            'init_dim': 16,
            'learning_rate':1e-5,
            'data_src': 1000
        }

    elif mode == '2D':
        extract_dir = 'data/tiny-imagenet/'
        train_image_path, _ = tiny_imagenet_file_handler(extract_dir)
        params = {
        'train_num_steps': 100000,
        'init_size': (64, 64),
        'diffusion_steps': 1000, ## ancestral sampling steps
        'train_batch_size': 16,
        'infer_phi': 1.0,
        'infer_sigma': 0.2,
        'init_dim': 8,
        'learning_rate':1e-5,
        'data_src': train_image_path
    }
    
    else: 
        raise ValueError('Wrong mode provided (1D/2D)')

    return params

def get_data(mode = '1D', **kwargs):

    if mode == '1D':
        observation, images, noise = create_1d_data_colored_multi(n_samples=data_src, phi=phi, sigma=sigma, is_plot=False, size=input_size)
    
    elif mode == '2D':
        observation, images, noise = create_1d_data_colored_multi(data_src, phi=phi, sigma=sigma, is_plot=False, size=input_size)
    
    else:
        raise ValueError('Wrong Mode Selected')

    rand_idx = torch.randperm(subset_sample_len)

    train_observation, val_observation = observation[rand_idx][:int(train_split * subset_sample_len)], observation[rand_idx][int(train_split * subset_sample_len):] 
    train_images, val_images = images[rand_idx][:int(train_split * subset_sample_len)], images[rand_idx][int(train_split * subset_sample_len):] 
    train_noise, val_noise = noise[rand_idx][:int(train_split * subset_sample_len)], noise[rand_idx][int(train_split * subset_sample_len):] 

    return train_observation, val_observation, train_images, val_images, train_noise, val_noise

def plot_curve():
    pass

def run_main(train_data, val_data, params, mode = '1D', is_plot=True):

    print('started-training ...')

    if mode == '1D':
        ## run 1D trainer

        gmodel = Unet1DGDiff(
        dim = params['input_dim'], ## ? Something is wrong here?
        channels=1, 
        )

        ## time-embedding shape -> t = torch.randint(0, self.num_timesteps, (b,), device=device).long() --> (b, )
        t_shape = (2, )
        x_shape = (2, 1, params['seq_len'])

        print(x_shape, t_shape)

        print('- UNET-summary -')
        gmodel_2d.summary(x_shape=x_shape, t_shape=t_shape)

        ### Just the diffusion framework
        gdiffusion = GibbsDiff1D(
            gmodel,
            seq_len = params['input_size'],
            num_timesteps = params['diffusion_steps'],
        )

        gtrainer = TrainerGDiff(
            gdiffusion,
            train_data,
            val_data[:50],
            train_batch_size = params['train_batch_size'],
            train_lr = params['learning_rate'],
            train_num_steps = params['train_num_steps'], # total training steps
            gradient_accumulate_every = 2,     # gradient accumulation steps
            ema_decay = 0.995,                 # exponential moving average decay
            mode = '2D'
        )

        train_loss_curve, val_loss_curve = gtrainer.train()
        
        if is_plot:
            plot_curve(train_loss_curve, val_loss_curve)

        ### save gdiffusion_2d model in /saves
 
    elif mode == '2D':
        ## run 2D trainer

        gmodel_2d = Unet2DGDiff(
        dim = params['input_dim'], ## ? Something is wrong here?
        channels=3, 
        )

        ## time-embedding shape -> t = torch.randint(0, self.num_timesteps, (b,), device=device).long() --> (b, )
        t_shape = (2, )
        x_shape = (2, 3, *params['input_size'])

        print(x_shape, t_shape)

        print('- UNET-summary -')
        gmodel_2d.summary(x_shape=x_shape, t_shape=t_shape)

        ### Just the diffusion framework
        gdiffusion_2d = GibbsDiff2D(
            gmodel_2d,
            image_size = (3, *params['input_size']),
            num_timesteps = params['diffusion_steps'],
        )

        gtrainer_2d = TrainerGDiff(
            gdiffusion_2d,
            train_data,
            val_data[:50],
            train_batch_size = params['train_batch_size'],
            train_lr = params['learning_rate'],
            train_num_steps = params['train_num_steps'], # total training steps
            gradient_accumulate_every = 2,     # gradient accumulation steps
            ema_decay = 0.995,                 # exponential moving average decay
            mode = '2D'
        )

        train_loss_curve, val_loss_curve = gtrainer_2d.train()
        
        if is_plot:
            plot_curve(train_loss_curve, val_loss_curve)

        ### save gdiffusion_2d model in /saves
 
    else: 
        raise ValueError('Wrong mode provided (1D/2D)')
    
    pass

if __name__ == '__main__':
    print('running __run_main.py__ ...')

    parser = argparse.ArgumentParser(description="train mode (1D/2D) --mode")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['1D', '2D'],  # You can customize these
        required=True,
        help='train-mode'
    )

    args_mode = parser.parse_args().mode

    params = get_hparams(args_mode)
    
    _, _, train_data, val_data, _, _ = get_data(mode=args_mode, data_src=params['data_src'])
    print(train_data.shape, val_data.shape)

    run_main(train_data, val_data, params, mode=args_mode)



     


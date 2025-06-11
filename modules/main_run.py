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
import time

## ImageNet subset ~1k
import cv2 as cv
import torchvision
import torchvision.datasets

## 1D
from .utils.noise_create import *
from .comp.one_d.unet_1d import Unet1DGDiff
from .comp.one_d.diffusion_1d import GibbsDiff1D

## 2D
from .utils.noise_create_2d import *
from .comp.two_d.unet_2d import Unet2DGDiff
from .comp.two_d.diffusion_2d import GibbsDiff2D

## Cosmo
from .utils.noise_create_2d import *
from .comp.two_d.unet_2d_cosmo import Unet2DGDiff_cosmo
from .comp.two_d.diffusion_2d_cosmo import GibbsDiff2D_cosmo

from .comp.diff_trainer import TrainerGDiff
from .utils.metrics import *
from .utils.data_file_handler import *

save_dir = 'saves'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_hparams(mode = '1D'):

    if mode == '1D':
        params = {
            'train_num_steps': 50000, # > 10
            'seq_len': 100,
            'diffusion_steps': 1000, ## ancestral sampling steps
            'train_batch_size': 32,
            'infer_phi': 1.0,
            'infer_sigma': 0.2,
            'input_dim': 32,
            'learning_rate':1e-5,
            'n_samples': 50000,
            'train_split': 0.8
        }

    elif mode == '2D':
        extract_dir = 'data/'
        train_image_path, _ = tiny_imagenet_file_handler(extract_dir)
        params = {
        'train_num_steps': 30000,
        'init_size': (64, 64),
        'diffusion_steps': 1000, ## ancestral sampling steps
        'train_batch_size': 64,
        'infer_phi': 1.0,
        'infer_sigma': 0.2,
        'input_dim': 32,
        'learning_rate':1e-5,
        'image_paths': train_image_path,
        'n_samples': 100000,
        'train_split': 0.8
    }
    
    elif mode == 'cosmo': ## (PASS-SUBSHAPE)
        cosmo_path = '/home/am3353/Gibbs-Diff/data/cosmo/created_data'
        params = {
        'train_num_steps': 50000,
        'init_size': (64, 64),
        'diffusion_steps': 1000, ## ancestral sampling steps
        'train_batch_size': 32,
        'infer_H0': 72.0, ##
        'infer_sigma': 0.40,
        'infer_ombh2': 0.02,
        'input_dim': 32,
        'learning_rate':1e-5,
        'cosmo_path': cosmo_path,
        'n_samples': 1000,
        'train_split': 0.8
    }
    

    else: 
        raise ValueError('Wrong mode provided (1D/2D)')

    print('HyParams-Retrieved')

    return params

def get_data(params, mode = '1D'):

    if mode == '1D':
        observation, images, noise = create_1d_data_colored_multi(n_samples=params['n_samples'], phi=params['infer_phi'], sigma=params['infer_sigma'])
    
    elif mode == '2D':
        observation, images, noise = create_2d_data_colored(params['image_paths'], n_samples=params['n_samples'], phi=params['infer_phi'], sigma=params['infer_sigma'], size=params['init_size'], is_plot=False)
    
    elif mode == 'cosmo':
        ## observation = mixed-map, images = intersteller-dust, noise = CMB signal
        observation, images, noise, _ = get_cosmo_data(n_samples=params['n_samples']) 
   
    else:
        raise ValueError('Wrong Mode Selected')

    subset_sample_len = params['n_samples']
    train_split = params['train_split']

    rand_idx = torch.randperm(subset_sample_len)

    train_observation, val_observation = observation[rand_idx][:int(train_split * subset_sample_len)], observation[rand_idx][int(train_split * subset_sample_len):] 
    train_images, val_images = images[rand_idx][:int(train_split * subset_sample_len)], images[rand_idx][int(train_split * subset_sample_len):] 
    train_noise, val_noise = noise[rand_idx][:int(train_split * subset_sample_len)], noise[rand_idx][int(train_split * subset_sample_len):] 

    print('Dataset-Retrieved')

    return train_observation, val_observation, train_images, val_images, train_noise, val_noise

def plot_curve(train_loss_curve, val_loss_curve_x, val_loss_curve_y, mode):
    plot_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_save_path = os.path.join(plot_dir, f'plots_curve_{mode}.png')

    fig = plt.Figure(figsize=(7, 7))

    if (len(train_loss_curve) < 10) or (len(val_loss_curve_y) < 10):
        rate = 1
    else:
        rate = 10

    plt.plot(np.arange(len(train_loss_curve))[::rate], train_loss_curve[::rate], color = 'red', label = 'train')
    plt.plot(val_loss_curve_x[::rate], val_loss_curve_y[::rate], color = 'blue', label = 'validation')

    plt.ylabel('Loss')
    plt.xlabel('#Optimization Steps (sub-sampled)')
    plt.title('Loss-Curve (Sub-Sampled)')

    plt.grid()
    plt.legend()
    plt.savefig(plot_save_path)

def run_main(train_data, val_data, params, mode = '1D', is_plot=True):

    print(f'started-training {mode} ...')

    if mode == '1D':
        ## run 1D trainer

        gmodel = Unet1DGDiff(
        dim = params['input_dim'], ## ? Something is wrong here?
        channels=1, 
        ).to(device)

        ## time-embedding shape -> t = torch.randint(0, self.num_timesteps, (b,), device=device).long() --> (b, )
        t_shape = (2, )
        x_shape = (2, 1, params['seq_len'])

        print(x_shape, t_shape)

        print('- UNET-summary -')
        gmodel.summary(x_shape=x_shape, t_shape=t_shape)

        ### Just the diffusion framework
        gdiffusion = GibbsDiff1D(
            gmodel,
            seq_len = params['seq_len'],
            num_timesteps = params['diffusion_steps'],
        ).to(device)

        ft = time.time()
        gtrainer = TrainerGDiff(
            gdiffusion,
            train_data,
            val_data[:int(0.5 * len(val_data))],
            train_batch_size = params['train_batch_size'],
            train_lr = params['learning_rate'],
            train_num_steps = params['train_num_steps'], # total training steps
            gradient_accumulate_every = 2,     # gradient accumulation steps
            ema_decay = 0.995,                 # exponential moving average decay
            mode = mode
        )
        lt = time.time()

        train_loss_curve, val_loss_curve_x, val_loss_curve_y = gtrainer.train()
        
        if is_plot:
            plot_curve(train_loss_curve, val_loss_curve_x, val_loss_curve_y, mode=mode)

        ### save gdiffusion_2d model in /saves

        save_path = os.path.join(save_dir, 'gdiffusion_1d_model.pth')

        torch.save({
        'model_state_dict': gdiffusion.model.state_dict(),
        'config': {
            'seq_len': gdiffusion.seq_len,
            'num_timesteps': gdiffusion.num_timesteps,
            'params': params,  # optional: save full params dict
        }
        }, save_path)

        print(f'trained and saved (1D) in {(lt - ft)/60} mins')
 
    elif mode == '2D':
        ## run 2D trainer

        gmodel_2d = Unet2DGDiff(
        dim = params['input_dim'], ## ? Something is wrong here?
        channels=3, 
        ).to(device)

        ## time-embedding shape -> t = torch.randint(0, self.num_timesteps, (b,), device=device).long() --> (b, )
        t_shape = (2, )
        x_shape = (2, 3, *params['init_size'])

        print(x_shape, t_shape)

        print('- UNET-summary -')
        gmodel_2d.summary(x_shape=x_shape, t_shape=t_shape)

        ### Just the diffusion framework
        gdiffusion_2d = GibbsDiff2D(
            gmodel_2d,
            image_size = (3, *params['init_size']),
            num_timesteps = params['diffusion_steps'],
        ).to(device)

        ft = time.time()
        gtrainer_2d = TrainerGDiff(
            gdiffusion_2d,
            train_data,
            val_data[:int(0.5 * len(val_data))],
            train_batch_size = params['train_batch_size'],
            train_lr = params['learning_rate'],
            train_num_steps = params['train_num_steps'], # total training steps
            gradient_accumulate_every = 2,     # gradient accumulation steps
            ema_decay = 0.995,                 # exponential moving average decay
            mode = mode
        )
        lt = time.time()

        train_loss_curve, val_loss_curve_x, val_loss_curve_y = gtrainer_2d.train()
        
        if is_plot:
            plot_curve(train_loss_curve, val_loss_curve_x, val_loss_curve_y, mode=mode)

        ### save gdiffusion_2d model in /saves
        save_path = os.path.join(save_dir, 'gdiffusion_2d_model.pth')

        torch.save({
        'model_state_dict': gdiffusion_2d.model.state_dict(),
        'config': {
            'input_size': gdiffusion_2d.image_size,
            'num_timesteps': gdiffusion_2d.num_timesteps,
            'params': params,  # optional: save full params dict
        }
        }, save_path)
 
        print(f'trained and saved (2D) in {(lt - ft)/60} mins')
    
    elif mode == 'cosmo':
        ## run cosmo trainer

        gmodel_cosmo = Unet2DGDiff_cosmo(
        dim = params['input_dim'], ## ? Something is wrong here?
        channels=1, 
        ).to(device)

        ## time-embedding shape -> t = torch.randint(0, self.num_timesteps, (b,), device=device).long() --> (b, )
        t_shape = (2, )
        x_shape = (2, 1, *params['init_size'])

        print(x_shape, t_shape)

        print('- UNET-summary -')
        gmodel_cosmo.summary(x_shape=x_shape, t_shape=t_shape)

        ### Just the diffusion framework
        gdiffusion_2d_cosmo = GibbsDiff2D_cosmo(
            gmodel_cosmo,
            image_size = (1, *params['init_size']),
            num_timesteps_ddpm = params['diffusion_steps'],
        ).to(device)

        ft = time.time()
        gtrainer_2d_cosmo = TrainerGDiff(
            gdiffusion_2d_cosmo,
            train_data,
            val_data[:int(0.5 * len(val_data))],
            train_batch_size = params['train_batch_size'],
            train_lr = params['learning_rate'],
            train_num_steps = params['train_num_steps'], # total training steps
            gradient_accumulate_every = 2,     # gradient accumulation steps
            ema_decay = 0.995,                 # exponential moving average decay
            mode = mode
        )
        lt = time.time()

        train_loss_curve, val_loss_curve_x, val_loss_curve_y = gtrainer_2d_cosmo.train()
        
        if is_plot:
            plot_curve(train_loss_curve, val_loss_curve_x, val_loss_curve_y, mode=mode)

        ### save gdiffusion_2d model in /saves
        save_path = os.path.join(save_dir, 'gdiffusion_cosmo_model.pth')

        torch.save({
        'model_state_dict': gdiffusion_2d.model.state_dict(),
        'config': {
            'input_size': gdiffusion_2d.image_size,
            'num_timesteps': gdiffusion_2d.num_timesteps,
            'params': params,  # optional: save full params dict
        }
        }, save_path)
 
        print(f'trained and saved (cosmo) in {(lt - ft)/60} mins')
 
    else: 
        raise ValueError('Wrong mode provided (1D/2D/cosmo)')
    

if __name__ == '__main__':
    print('running __run_main.py__ ...')

    parser = argparse.ArgumentParser(description="train mode (1D/2D/cosmo) --mode")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['1D', '2D', 'cosmo'],  # You can customize these
        required=True,
        help='train-mode'
    )

    args_mode = parser.parse_args().mode

    params = get_hparams(args_mode)
    
    _, _, train_data, val_data, _, _ = get_data(params, mode=args_mode)
    print(train_data.shape, val_data.shape)

    run_main(train_data, val_data, params, mode=args_mode)



     


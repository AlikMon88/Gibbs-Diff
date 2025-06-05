import torch
import math
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from einops import rearrange
from ema_pytorch import EMA
# from accelerate import Accelerator
from tqdm import tqdm
from multiprocessing import cpu_count
from tqdm import tqdm 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

__version__ = '1.0.0'

class Trainer1D:
    def __init__(
        self,
        diffusion_model: nn.Module,
        dataset,
        *,  # keyword-only separator
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 25,
        results_folder: str = './results',
        max_grad_norm: float = 1.0
    ):
        super().__init__()

        # model and channels (CPU only)
        self.model = diffusion_model
        self.channels = getattr(diffusion_model, 'channels', 1)

        # sampling & training settings
        assert num_samples**0.5 == int(num_samples**0.5), 'num_samples must have integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps

        # data loader (CPU)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        self.dl = cycle(dl)

        # optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        # EMA (CPU)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        # results folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter
        self.step = 0

    def save(self, milestone: int):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }
        torch.save(data, self.results_folder / f'model-{milestone}.pt')

    def load(self, milestone: int):
        data = torch.load(self.results_folder / f'model-{milestone}.pt', map_location='cpu')
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        self.step = data['step']
        self.ema.load_state_dict(data['ema'])

    def train(self):
        for _ in tqdm(range(self.step, self.train_num_steps), desc='Optimization Steps', initial=self.step, total=self.train_num_steps):
            self.model.train()
            total_loss = 0.0

            for _ in range(self.gradient_accumulate_every):
                
                batch = next(self.dl)
                batch = batch.reshape(batch.shape[0], 1, -1)
                
                loss = self.model(batch) / self.gradient_accumulate_every
                loss.backward()
                total_loss += loss.item()

            # gradient clipping and step
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()

            # EMA update
            self.ema.update()
            self.step += 1

            # save and sample
            # if self.step % self.save_and_sample_every == 0:
            #     self.ema.ema_model.eval()
            #     with torch.no_grad():
            #         batches = num_to_groups(self.num_samples, self.batch_size)
            #         samples = [self.ema.ema_model.sample(batch_size=n) for n in batches]
            #         all_samples = torch.cat(samples, dim=0)
            #         torch.save(all_samples, self.results_folder / f'sample-{self.step // self.save_and_sample_every}.pt')
            #         self.save(self.step // self.save_and_sample_every)

        print('Training complete.')

#### -----------------------------------------------------------------------------------
#### ---------------------------GibbsDIFF-----------------------------------------------
#### -----------------------------------------------------------------------------------

class TrainerGDiff:
    def __init__(
        self,
        diffusion_model: nn.Module,
        dataset,
        val_dataset,
        *,  ## keyword-only separator
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 25,
        results_folder: str = './results',
        max_grad_norm: float = 1.0,
        mode = '1D'
    ):
        super().__init__()

        # model and channels (CPU only)
        self.model = diffusion_model
        self.channels = getattr(diffusion_model, 'channels', 1)
        self.mode = mode

        if not (self.mode == '1D' or self.mode == '2D'):
            raise ValueError("Wrong mode supplied (1D/2D)")

        # sampling & training settings
        assert num_samples**0.5 == int(num_samples**0.5), 'num_samples must have integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps

        # data loader (CPU)
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0)
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)

        self.dl = cycle(dl)
    
        # optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        # EMA (CPU)
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        # results folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter
        self.step = 0

    def save(self, milestone: int):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }
        torch.save(data, self.results_folder / f'model-{milestone}.pt')

    def load(self, milestone: int):
        data = torch.load(self.results_folder / f'model-{milestone}.pt', map_location='cpu')
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        self.step = data['step']
        self.ema.load_state_dict(data['ema'])

    def train(self):
        
        train_loss_curve, val_loss_curve = [], []

        for _ in tqdm(range(self.step, self.train_num_steps), desc='Optimization Steps', initial=self.step, total=self.train_num_steps):
            self.model.train()

            # Get training batch and reshape
            batch = next(self.dl)
            
            if self.mode == '1D':
                batch = batch.reshape(batch.shape[0], 1, -1)
            
            # Forward and backward pass
            loss = self.model(batch)
            loss.backward()

            # Gradient clipping and optimizer step
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()

            # EMA update
            self.ema.update()
            self.step += 1

            train_loss_curve.append(loss.item())

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in self.val_dl:
                    if self.mode == '1D':
                        val_batch = val_batch.reshape(val_batch.shape[0], 1, -1)
                    val_loss += self.model(val_batch).item()
            val_loss_curve.append(val_loss)

            # Optional: Save and sample
            # if self.step % self.save_and_sample_every == 0:
            #     self.ema.ema_model.eval()
            #     with torch.no_grad():
            #         batches = num_to_groups(self.num_samples, self.batch_size)
            #         samples = [self.ema.ema_model.sample(batch_size=n) for n in batches]
            #         all_samples = torch.cat(samples, dim=0)
            #         torch.save(all_samples, self.results_folder / f'sample-{self.step // self.save_and_sample_every}.pt')
            #         self.save(self.step // self.save_and_sample_every)

        print('Training complete.')
        return train_loss_curve, val_loss_curve

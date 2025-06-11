import torch
import math
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from einops import rearrange
from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate.utils import set_seed

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

# class Trainer1D:
#     def __init__(
#         self,
#         diffusion_model: nn.Module,
#         dataset,
#         *,  # keyword-only separator
#         train_batch_size: int = 16,
#         gradient_accumulate_every: int = 1,
#         train_lr: float = 1e-4,
#         train_num_steps: int = 100000,
#         ema_update_every: int = 10,
#         ema_decay: float = 0.995,
#         adam_betas: tuple = (0.9, 0.99),
#         save_and_sample_every: int = 1000,
#         num_samples: int = 25,
#         results_folder: str = './results',
#         max_grad_norm: float = 1.0
#     ):
#         super().__init__()

#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

#         # model and channels (CPU only)
#         self.model = diffusion_model
#         self.channels = getattr(diffusion_model, 'channels', 1)

#         # sampling & training settings
#         assert num_samples**0.5 == int(num_samples**0.5), 'num_samples must have integer square root'
#         self.num_samples = num_samples
#         self.save_and_sample_every = save_and_sample_every

#         self.batch_size = train_batch_size
#         self.gradient_accumulate_every = gradient_accumulate_every
#         self.max_grad_norm = max_grad_norm
#         self.train_num_steps = train_num_steps

#         # data loader (CPU)
#         dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0)
#         self.dl = cycle(dl)

#         # optimizer
#         self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

#         # EMA (CPU)
#         self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

#         # results folder
#         self.results_folder = Path(results_folder)
#         self.results_folder.mkdir(exist_ok=True)

#         # step counter
#         self.step = 0

#     def save(self, milestone: int):
#         data = {
#             'step': self.step,
#             'model': self.model.state_dict(),
#             'opt': self.opt.state_dict(),
#             'ema': self.ema.state_dict(),
#         }
#         torch.save(data, self.results_folder / f'model-{milestone}.pt')

#     def load(self, milestone: int):
#         data = torch.load(self.results_folder / f'model-{milestone}.pt', map_location='cpu')
#         self.model.load_state_dict(data['model'])
#         self.opt.load_state_dict(data['opt'])
#         self.step = data['step']
#         self.ema.load_state_dict(data['ema'])

#     def train(self):
#         for _ in tqdm(range(self.step, self.train_num_steps), desc='Optimization Steps', initial=self.step, total=self.train_num_steps):
#             self.model.train()
#             total_loss = 0.0

#             for _ in range(self.gradient_accumulate_every):
                
#                 batch = next(self.dl)
#                 batch = batch.reshape(batch.shape[0], 1, -1)
                
#                 loss = self.model(batch) / self.gradient_accumulate_every
#                 loss.backward()
#                 total_loss += loss.item()

#             # gradient clipping and step
#             nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
#             self.opt.step()
#             self.opt.zero_grad()

#             # EMA update
#             self.ema.update()
#             self.step += 1

#             # save and sample
#             # if self.step % self.save_and_sample_every == 0:
#             #     self.ema.ema_model.eval()
#             #     with torch.no_grad():
#             #         batches = num_to_groups(self.num_samples, self.batch_size)
#             #         samples = [self.ema.ema_model.sample(batch_size=n) for n in batches]
#             #         all_samples = torch.cat(samples, dim=0)
#             #         torch.save(all_samples, self.results_folder / f'sample-{self.step // self.save_and_sample_every}.pt')
#             #         self.save(self.step // self.save_and_sample_every)

#         print('Training complete.')


## Let Accelerate handle device (CUDA/CPU) management
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
        max_grad_norm: float = 1.0,
        seed: int = 42
    ):
        super().__init__()

        # 1. Initialize Accelerator
        # This handles all device placement (CPU/GPU/TPU/Multi-GPU),
        # mixed precision, and distributed training.
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            log_with="tensorboard",
            project_dir=results_folder
        )
        set_seed(seed)

        # 2. Let Accelerator handle device placement
        self.model = diffusion_model
        self.channels = getattr(diffusion_model, 'channels', 1)

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        # DataLoader
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        # 3. Prepare everything with Accelerator
        # This wraps the model, optimizer, and dataloader for any hardware setup.
        self.model, self.opt, self.dl = self.accelerator.prepare(
            self.model, self.opt, dl
        )

        # Cycle the prepared dataloader
        self.dl = cycle(self.dl)

        # EMA: Ensure it uses the unwrapped model but runs on the correct device
        self.ema = EMA(
            self.accelerator.unwrap_model(self.model), 
            beta=ema_decay, 
            update_every=ema_update_every
        ).to(self.accelerator.device)

        # Results folder handled by Accelerator's logging directory
        self.results_folder = Path(self.accelerator.project_dir)

    @property
    def step(self):
        return self.opt.state[self.opt.param_groups[0]['params'][0]]['step']
        
    def save(self, milestone: int):
        # Use accelerator's save_state for robust saving
        self.accelerator.save_state(self.results_folder / f"state-{milestone}")
        
        # Save EMA model separately
        ema_path = self.results_folder / f'ema-{milestone}.pt'
        torch.save(self.ema.state_dict(), ema_path)

    def load(self, milestone: int):
        # Use accelerator's load_state
        self.accelerator.load_state(self.results_folder / f"state-{milestone}")

        # Load EMA model separately and move to the correct device
        ema_path = self.results_folder / f'ema-{milestone}.pt'
        self.ema.load_state_dict(torch.load(ema_path, map_location=self.accelerator.device))

    def train(self):
        # Use accelerator's tqdm for proper display in distributed training
        progress_bar = tqdm(range(self.step, self.train_num_steps), initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process)

        for step in progress_bar:
            self.model.train()
            
            # The with-statement handles gradient accumulation for us
            with self.accelerator.accumulate(self.model):
                # 4. Data is automatically moved to the correct device
                batch = next(self.dl)
                batch = batch.reshape(batch.shape[0], 1, -1)
                
                loss = self.model(batch)

                # 5. Use accelerator's backward pass for automatic scaling
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.opt.step()
                self.opt.zero_grad()
            
            # Update EMA and logging only on the main process after an optimizer step
            if self.accelerator.sync_gradients:
                self.ema.update()
                progress_bar.set_description(f"loss: {loss.item():.4f}")

                # if step % self.save_and_sample_every == 0:
                #     self.accelerator.wait_for_everyone()
                #     if self.accelerator.is_main_process:
                #         # Add sampling logic here, using self.accelerator.unwrap_model(self.ema.ema_model)
                #         self.save(step // self.save_and_sample_every)

        self.accelerator.print('Training complete.')

#### -----------------------------------------------------------------------------------
#### ---------------------------GibbsDIFF-----------------------------------------------
#### -----------------------------------------------------------------------------------

# class TrainerGDiff:
#     def __init__(
#         self,
#         diffusion_model: nn.Module,
#         dataset,
#         val_dataset,
#         *,  ## keyword-only separator
#         train_batch_size: int = 16,
#         gradient_accumulate_every: int = 1,
#         train_lr: float = 1e-4,
#         train_num_steps: int = 100000,
#         ema_update_every: int = 10,
#         ema_decay: float = 0.995,
#         adam_betas: tuple = (0.9, 0.99),
#         save_and_sample_every: int = 1000,
#         num_samples: int = 25,
#         results_folder: str = './results',
#         max_grad_norm: float = 1.0,
#         mode = '1D'
#     ):
#         super().__init__()

#         # model and channels (CPU only)
#         self.model = diffusion_model
#         self.channels = getattr(diffusion_model, 'channels', 1)
#         self.mode = mode

#         if not (self.mode == '1D' or self.mode == '2D'):
#             raise ValueError("Wrong mode supplied (1D/2D)")

#         # sampling & training settings
#         assert num_samples**0.5 == int(num_samples**0.5), 'num_samples must have integer square root'
#         self.num_samples = num_samples
#         self.save_and_sample_every = save_and_sample_every

#         self.batch_size = train_batch_size
#         self.gradient_accumulate_every = gradient_accumulate_every
#         self.max_grad_norm = max_grad_norm
#         self.train_num_steps = train_num_steps

#         # data loader (CPU)
#         dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=0)
#         self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)

#         self.dl = cycle(dl)
    
#         # optimizer
#         self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

#         # EMA (CPU)
#         self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

#         # results folder
#         self.results_folder = Path(results_folder)
#         self.results_folder.mkdir(exist_ok=True)

#         # step counter
#         self.step = 0

#     def save(self, milestone: int):
#         data = {
#             'step': self.step,
#             'model': self.model.state_dict(),
#             'opt': self.opt.state_dict(),
#             'ema': self.ema.state_dict(),
#         }
#         torch.save(data, self.results_folder / f'model-{milestone}.pt')

#     def load(self, milestone: int):
#         data = torch.load(self.results_folder / f'model-{milestone}.pt', map_location='cpu')
#         self.model.load_state_dict(data['model'])
#         self.opt.load_state_dict(data['opt'])
#         self.step = data['step']
#         self.ema.load_state_dict(data['ema'])

#     def train(self):
        
#         train_loss_curve, val_loss_curve = [], []

#         for _ in tqdm(range(self.step, self.train_num_steps), desc='Optimization Steps', initial=self.step, total=self.train_num_steps):
#             self.model.train()

#             # Get training batch and reshape
#             batch = next(self.dl)
            
#             if self.mode == '1D':
#                 batch = batch.reshape(batch.shape[0], 1, -1)
            
#             # Forward and backward pass
#             loss = self.model(batch)
#             loss.backward()

#             # Gradient clipping and optimizer step
#             nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
#             self.opt.step()
#             self.opt.zero_grad()

#             # EMA update
#             self.ema.update()
#             self.step += 1

#             train_loss_curve.append(loss.item())

#             # Validation
#             self.model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for val_batch in self.val_dl:
#                     if self.mode == '1D':
#                         val_batch = val_batch.reshape(val_batch.shape[0], 1, -1)
#                     val_loss += self.model(val_batch).item()
#             val_loss_curve.append(val_loss)

#             # Optional: Save and sample
#             # if self.step % self.save_and_sample_every == 0:
#             #     self.ema.ema_model.eval()
#             #     with torch.no_grad():
#             #         batches = num_to_groups(self.num_samples, self.batch_size)
#             #         samples = [self.ema.ema_model.sample(batch_size=n) for n in batches]
#             #         all_samples = torch.cat(samples, dim=0)
#             #         torch.save(all_samples, self.results_folder / f'sample-{self.step // self.save_and_sample_every}.pt')
#             #         self.save(self.step // self.save_and_sample_every)

#         print('Training complete.')
#         return train_loss_curve, val_loss_curve


class TrainerGDiff:
    def __init__(
        self,
        diffusion_model: nn.Module,
        dataset,
        val_dataset,
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
        max_grad_norm: float = 1.0,
        mode='1D',
        seed: int = 42
    ):
        super().__init__()

        # 1. Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            log_with="tensorboard",
            project_dir=results_folder
        )
        set_seed(seed)

        # 2. Store config and prepare models and data
        self.model = diffusion_model
        self.mode = mode
        if not (self.mode == '1D' or self.mode == '2D' or self.mode == 'cosmo'):
            raise ValueError("Wrong mode supplied (1D/2D.cosmo)")
        
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        # DataLoaders
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_dl = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=4)

        # 3. Prepare everything with Accelerator
        self.model, self.opt, self.dl, self.val_dl = self.accelerator.prepare(
            self.model, self.opt, dl, val_dl
        )
        self.dl = cycle(self.dl)

        # EMA handling
        self.ema = EMA(
            self.accelerator.unwrap_model(self.model), 
            beta=ema_decay, 
            update_every=ema_update_every
        ).to(self.accelerator.device)
        
        self.results_folder = Path(self.accelerator.project_dir)
        self.step = 0

    def save(self, milestone: int):
        if not self.accelerator.is_main_process:
            return
        
        state_path = self.results_folder / f"state-{milestone}.pt"
        self.accelerator.save_state(str(state_path)) # Use accelerator's save_state
        
        ema_path = self.results_folder / f'ema-{milestone}.pt'
        torch.save(self.ema.state_dict(), ema_path)

    def load(self, milestone: int):
        state_path = self.results_folder / f"state-{milestone}.pt"
        self.accelerator.load_state(str(state_path))
        
        ema_path = self.results_folder / f'ema-{milestone}.pt'
        self.ema.load_state_dict(torch.load(ema_path, map_location=self.accelerator.device))

        self.step = self.opt.state[self.opt.param_groups[0]['params'][0]]['step']


    def train(self):
        
        train_loss_curve, val_loss_curve_x, val_loss_curve_y = [], [], []

        progress_bar = tqdm(range(self.step, self.train_num_steps), initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process)
        
        # val_stepper = self.train_num_steps // int(0.08 * self.train_num_steps)
        val_stepper = 25
        print('Validation-Stepper: ', val_stepper)

        while self.step < self.train_num_steps:
            self.model.train()
            
            # 4. Use accelerator's context manager for gradient accumulation
            with self.accelerator.accumulate(self.model):
                batch = next(self.dl)
                if self.mode == '1D':
                    batch = batch.reshape(batch.shape[0], 1, -1)
                
                loss = self.model(batch)

                # Log training loss
                # self.accelerator.log({"train_loss": loss.item()}, step=self.step)
                
                # 5. Backward pass handled by accelerator
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.max_grad_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()
            

            # After an optimizer step, update EMA and step counter
            if self.accelerator.sync_gradients:
                self.ema.update()
                
                train_loss_curve.append(loss.item())
            
                if (self.step % val_stepper) == 0:
                    val_loss = self.validate()
                    val_loss_curve_x.append(self.step)
                    val_loss_curve_y.append(val_loss)
                    
                progress_bar.set_description(f"loss: {loss.item():.4f}")
                self.step += 1
                progress_bar.update(1)
               
        self.accelerator.print("Training complete.")
        return train_loss_curve, val_loss_curve_x, val_loss_curve_y 

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for val_batch in self.val_dl:
                if self.mode == '1D':
                    val_batch = val_batch.reshape(val_batch.shape[0], 1, -1)
                
                loss = self.model(val_batch)
                # Gather losses from all processes for accurate validation metric
                gathered_loss = self.accelerator.gather_for_metrics(loss)
                total_val_loss += gathered_loss.sum().item()
        
        avg_val_loss = total_val_loss / len(self.val_dl.dataset)
        # self.accelerator.log({"val_loss": avg_val_loss}, step=self.step)
        # self.accelerator.print(f"Step {self.step}: Validation Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss

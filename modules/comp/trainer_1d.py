import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from ema_pytorch import EMA
from tqdm.auto import tqdm

## Custom Modules
from ..utils.helper import *
from unet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------Trainer Function --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

# trainer class
class Trainer1D(object):
    def __init__(
        self,
        diffusion_model,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.device = device
        self.model.to(self.device)

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.amp = amp
        
        # setup mixed precision training if enabled
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        # dataset and dataloader
        self.dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        self.dl = cycle(self.dl)  # Create an infinite dataloader

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict() if exists(self.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(self.device)

                    if self.amp:
                        with torch.cuda.amp.autocast():
                            loss = self.model(data)
                            loss = loss / self.gradient_accumulate_every
                        
                        self.scaler.scale(loss).backward()
                    else:
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                    
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.4f}')

                # Gradient clipping
                if self.amp:
                    self.scaler.unscale_(self.opt)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update weights
                if self.amp:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                
                self.opt.zero_grad()

                # Update EMA model and save samples periodically
                self.step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                    all_samples = torch.cat(all_samples_list, dim=0)

                    # Assuming this saves some visual representation - modify as needed
                    torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.pt'))
                    self.save(milestone)

                pbar.update(1)

### Using Pytorch_lightning_wrapper
class Trainer1D_SM():
    def __inti__(self):
        pass

    def train(hparams):
        # Training hyperparameters
        diffusion_steps = hparams.diffusion_steps
        dataset_choice = hparams.dataset_choice
        max_epoch = hparams.max_epoch
        batch_size = hparams.batch_size
        n_devices = hparams.n_devices
        optimizer = hparams.optimizer
        lr = hparams.learning_rate
        weight_decay = hparams.weight_decay
        accelerator = hparams.accelerator
        enable_ckpt = hparams.enable_ckpt

        #To be saved in wandb
        hyperparameters_config = dict(
            diffusion_steps = diffusion_steps,
            dataset_choice = dataset_choice,
            max_epoch = max_epoch,
            batch_size = batch_size,
            n_devices = n_devices,
            optimizer = optimizer,
            lr = lr,
            weight_decay = weight_decay
        )

        # Loading parameters
        load_model = hparams.load_model

        #directory where to store the checkpoints:
        default_root_dir=f"../model_checkpoints/{dataset_choice}_training/{diffusion_steps}steps_{optimizer}_{lr}lr_{batch_size}bs_{max_epoch}epochs/"

        #make directory if not existing:
        if not os.path.exists(default_root_dir):
            os.makedirs(default_root_dir, exist_ok=True)
        
        # Create datasets and data loaders
        if dataset_choice == 'ImageNet':

            train_dataset = ImageDataset("imagenet_train", data_dir='/tmp/imagenet/')
            val_dataset = ImageDataset("imagenet_val", data_dir='/tmp/imagenet/')

            n_channels, H, W = train_dataset[0][0].shape
            in_size_image = H * W
        else:
            #If you train on another dataset than imagenet, you will have to define yourself the train and validation sets in data.py
            train_dataset = ImageDataset(dataset_choice, data_dir='../data/')
            val_dataset = ImageDataset(dataset_choice, data_dir='../data/')

            n_channels, H, W = train_dataset[0][0].shape
            in_size_image = H * W

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

        # Create model and trainer
        if load_model:
            model = load_gdiff_model(diffusion_steps=5000)
        else:
            model = GDiff(in_size=in_size_image, 
                        diffusion_steps=diffusion_steps, 
                        img_depth=n_channels, 
                        lr=lr, 
                        weight_decay=weight_decay)

        if hparams.wandb:
            wandb_logger = WandbLogger(project=hparams.wandb_project, 
                                    config=hyperparameters_config, 
                                    entity=hparams.wandb_entity, 
                                    group=hparams.wandb_group_name, 
                                    name=f"{dataset_choice}_steps{diffusion_steps}_lr{lr}_bs{batch_size}_opt{optimizer}_wd{weight_decay}")
        if enable_ckpt:
            checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, monitor=None)
       
        trainer = pl.Trainer(max_epochs=max_epoch, 
                            log_every_n_steps=10,
                            devices=n_devices, 
                            accelerator= accelerator, 
                            logger= wandb_logger if hparams.wandb else None,
                            callbacks=[checkpoint_callback] if enable_ckpt else None)

        trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
    print('Running ... __trainer_1d.py ...')
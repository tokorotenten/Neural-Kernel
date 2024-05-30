import os
import sys
from omegaconf import DictConfig, ListConfig
import numpy as np
from functools import partial

import lightning
import lightning_uq_box
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
import torch.nn as nn

import hydra
from hydra import utils
import mlflow
from tqdm import tqdm
from data import generate_data
from utils.evaluation import wasserstein, MSE, compute_QI

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning import LightningDataModule
from lightning_uq_box.models import MLP, ConditionalGuidedLinearModel
from lightning_uq_box.uq_methods import CARDRegression, DeterministicRegression

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


class ToyData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"input":self.X[idx], "target":self.y[idx]}

class DataModule(LightningDataModule):
    def __init__(self, X, y, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = ToyData(X, y)

    def train_dataloader(self):
        """Create the train DataLoader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    

@hydra.main(config_name='toy_CARD', version_base=None, config_path="config")
def main(cfg):
  vX, vy, vXp, vyp, DATA = generate_data(cfg)
  was = np.zeros((cfg.data.num_seeds))

  mlflow.set_experiment(cfg.mlflow.runname)
  with mlflow.start_run():
    log_params_from_omegaconf_dict(cfg)
    for i in tqdm(range(vX.shape[0])):
       X = vX[i]
       y = vy[i]
       Xp = vXp[i]
       yp = vyp[i]
       dm = DataModule(X, y, cfg.train.batch_size)
       
       network = MLP(n_inputs=X.shape[1], n_hidden=[50, 50], n_outputs=y.shape[1])
 
       cond_mean_model = DeterministicRegression(model=network, optimizer=partial(torch.optim.AdamW, lr=cfg.optimizer.MLP_lr), loss_fn=nn.MSELoss())
       trainer = Trainer(
           max_epochs=cfg.train.epoch_MLP,  # number of epochs we want to train
           devices=1,
           accelerator=cfg.device,
           enable_checkpointing=False,
           enable_progress_bar=False,
           limit_val_batches=0.0
           )
       trainer.fit(cond_mean_model, dm)
       
       guidance_model = ConditionalGuidedLinearModel(
           n_steps=cfg.model.DF_steps,
           x_dim=X.shape[1],
           y_dim=y.shape[1],
           n_hidden=[128, 128, 128],
           cat_x=True,
           cat_y_pred=True,
           )
       card_model = CARDRegression(
          cond_mean_model=cond_mean_model.model,
          guidance_model=guidance_model,
          guidance_optim=partial(torch.optim.AdamW, lr=cfg.optimizer.DF_lr),
          n_z_samples=cfg.test.num_sample,
          beta_start=0.0001,
          beta_end=0.02,
          n_steps=cfg.model.DF_steps,
          )
       
       diff_trainer = Trainer(
           max_epochs=cfg.train.epoch_DF,  # number of epochs we want to train
           accelerator=cfg.device,
           devices=1,
           enable_progress_bar=True,
           limit_val_batches=0.0,
           )
       diff_trainer.fit(card_model, dm)

       card_model = card_model.to(cfg.device)
       preds = card_model.predict_step(torch.tensor(Xp, dtype=torch.float32).to(cfg.device))
       samples = preds["samples"][-1].detach().cpu().numpy()

       was[i] = wasserstein(yp, samples.transpose(1,0,2)).mean()

  
    was_mean = np.mean(was)
    was_std = np.std(was)
    mlflow.log_metric("was_mean", was_mean)
    mlflow.log_metric("was_std", was_std)

  return was

if __name__ == '__main__':
   main()
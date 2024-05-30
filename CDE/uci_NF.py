import os
import sys
from omegaconf import DictConfig, ListConfig

import numpy as np
import math
import normflows as nf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim

import hydra
from hydra import utils
import mlflow
from tqdm import tqdm
from data import generate_data
from utils.evaluation import wasserstein, MSE, compute_QI


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


def sampling_NF(Agent, X, y, N_y):
  samples=np.zeros((y.shape[0], N_y, 1))
  for i in range(len(y)):
    context = torch.tile(X[i], (N_y,1))
    NF_samples, _ = Agent.sample(N_y, context=context)
    samples[i] = NF_samples.detach().cpu().numpy()
  return samples
  


@hydra.main(config_name='uci_NF', version_base=None, config_path="config")
def main(cfg):
  vX, vy, vXp, vyp, DATA = generate_data(cfg)
  rmse = np.zeros((cfg.data.num_seeds))
  qi = np.zeros((cfg.data.num_seeds))

  mlflow.set_experiment(cfg.mlflow.runname)
  with mlflow.start_run():
    log_params_from_omegaconf_dict(cfg)
    for i in tqdm(range(vX.shape[0])):
       device =  cfg.device
       X = vX[i]
       y = vy[i]
       Xp = vXp[i]
       yp = vyp[i]
       X_torch=torch.tensor(X, dtype=torch.float32).to(device)
       y_torch=torch.tensor(y, dtype=torch.float32).to(device)
       Xp_torch=torch.tensor(Xp, dtype=torch.float32).to(device)
       yp_torch=torch.tensor(yp, dtype=torch.float32).to(device)

       flows = []
       for j in range(cfg.model.num_flows):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(1, cfg.model.hidden_layers, cfg.model.hidden_units, num_context_channels=X.shape[1])]
        flows += [nf.flows.LULinearPermute(1)]
       # Set base distribution
       q0 = nf.distributions.DiagGaussian(1, trainable=False)
       # Construct flow model
       model = nf.ConditionalNormalizingFlow(q0, flows, y)
       model = model.to(device)

       if cfg.optimizer.weight_decay == True:
          optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.lr)
       else:
          optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

       train_ds = TensorDataset(X_torch, y_torch)
       trainloader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)

       for epoch in range(1, cfg.train.epoch + 1):
          for X, y in trainloader:
             optimizer.zero_grad()
             loss = model.forward_kld(y, X)
             loss.backward()
             optimizer.step()

       samples = sampling_NF(model, Xp_torch, yp, cfg.test.num_sample)
       points_sub = np.zeros((samples.shape[0], samples.shape[1], 1))
       for j in range(samples.shape[1]):
        points_sub[:, j, :] =  DATA[i].scaler_y.inverse_transform(samples[:, j, :])
       yp_orig = DATA[i].scaler_y.inverse_transform(yp)
       mse = MSE(yp_orig, points_sub)
       rmse[i] = np.sqrt(mse)

       n_bins=10
       y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_QI(n_bins, yp_orig, points_sub)
       qi[i] =  qice_coverage_ratio

    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)
    qi_mean = np.mean(qi)
    qi_std = np.std(qi)

    mlflow.log_metric("rmse_mean", rmse_mean)
    mlflow.log_metric("rmse_std", rmse_std)
    mlflow.log_metric("qi_mean", qi_mean)
    mlflow.log_metric("qi_std", qi_std)

  return rmse, qi

if __name__ == '__main__':
   main()
  
 



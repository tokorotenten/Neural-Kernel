import os
import sys
from omegaconf import DictConfig, ListConfig

import numpy as np
import math
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.lax as lax
import optax  
import equinox as eqx

import hydra
from hydra import utils
import mlflow
from tqdm import tqdm

from model.MDN_models import uci_NN_SN1 as NN1
from model.MDN_models import uci_NN_SN2 as NN2
from data import generate_data
from utils import sample_batch
from utils.evaluation import wasserstein, MSE, compute_QI
from utils.sampling import eval_MDN, sample_MDN

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

def lognormal(y, mean, logstd):
  logSqrtTwoPI = jnp.log(jnp.sqrt(2.0 * math.pi))
  return -0.5 * ((y - mean) / jnp.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

def compute_loss(model, state, X, Y):
  batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(None, 0, 0, 0))
  state, mu, logstd, logmix = batch_model(X, state)

  logmix = logmix - jnp.max(logmix, keepdims=True)
  
  logmix = logmix - logsumexp(logmix, 1, keepdims=True)
  v = logmix + lognormal(Y, mu, logstd)
  v = logsumexp(v, axis=1)
  loss= -jnp.mean(v)
  return loss, state

@eqx.filter_jit
def make_step_vec(model, state, optim, opt_state, X, Y, batch_size, key):
  x, y = jax.vmap(sample_batch, in_axes=(0,0,None,0))(X, Y, batch_size, key)
  grads, state = eqx.filter_vmap(eqx.filter_grad(compute_loss, has_aux=True))(model, state, x, y)
  updates, opt_state = optim.update(grads, opt_state, model)
  model = eqx.apply_updates(model, updates)
  return model, state, opt_state

def train_vec(model, state, optim, opt_state, X, Y, batch_size, num_steps, num_agents, key):
  with tqdm(range(num_steps)) as pbar_epoch:
    for steps in pbar_epoch:
       key, sub_key = jax.random.split(key, 2)
       sub_keys = jax.random.split(sub_key, num_agents)
       model, state, opt_state = make_step_vec(model, state, optim, opt_state, X, Y, batch_size, sub_keys)
  return model, state


@hydra.main(config_name='uci_MDN', version_base=None, config_path="config")
def main(cfg):
  vX, vy, vXp, vyp, DATA = generate_data(cfg)
  rmse = np.zeros((cfg.data.num_seeds))
  qi = np.zeros((cfg.data.num_seeds))
   
  seed = 5678
  key = jax.random.PRNGKey(seed)
  mkey, xkey, sample_key = jax.random.split(key, 3)
  mkeys = jax.random.split(mkey, cfg.data.num_seeds)

  num_inputs = vX.shape[-1]
  if cfg.model.type == "small":
     NN = NN1
  else: 
     NN = NN2
  model, state = eqx.filter_vmap(eqx.nn.make_with_state(NN))(num_inputs, cfg.model.mix, mkeys)

  if cfg.optimizer.weight_decay == True:
     optim = optax.adamw(cfg.optimizer.lr)
  else:
     optim = optax.adam(cfg.optimizer.lr)
  opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
  num_steps = int(vX.shape[1]/cfg.train.batch_size)*cfg.train.epoch

  mlflow.set_experiment(cfg.mlflow.runname)
  with mlflow.start_run():
    log_params_from_omegaconf_dict(cfg)
    model, state = train_vec(model, state, optim, opt_state, vX, vy, cfg.train.batch_size, num_steps, cfg.data.num_seeds, xkey)

    mu, logmix, sigma  = eqx.filter_vmap(eval_MDN)(model, state, vXp)
    for i in range(cfg.data.num_seeds):
      sample_key, sub_key = jax.random.split(sample_key, 2)
      samples = sample_MDN(mu[i], logmix[i], sigma[i], cfg.test.num_sample, sub_key)
      points_sub = np.zeros((samples.shape[0], samples.shape[1], 1))
      for j in range(samples.shape[1]):
        points_sub[:, j, :] =  DATA[i].scaler_y.inverse_transform(samples[:, j, :])
      yp_orig = DATA[i].scaler_y.inverse_transform(vyp[i])
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
  


     





    

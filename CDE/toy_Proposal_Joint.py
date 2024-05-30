import os
import sys
from omegaconf import DictConfig, ListConfig

import numpy as np
import math
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax  
import equinox as eqx

import hydra
from hydra import utils
import mlflow
from tqdm import tqdm

from model.NKME_models import toy_NN as NN
from data import generate_data
from utils import sample_batch
from utils.evaluation import wasserstein, MSE, compute_QI
from utils.sampling import eval_NKME, NKME_herding

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

def Gram(X, Y, sig):
  def pairwisedist(X,Y):
    def dist(x,y):
      z=x-y
      return jnp.sqrt(jnp.sum(jnp.square(z)))
    vmapped_dist = jax.vmap(dist, in_axes=(0, None))
    return jax.vmap(vmapped_dist, in_axes=(None, 0))(X,Y)
  S = pairwisedist(X, Y).T
  scale=jnp.sqrt(2*math.pi*(sig**2))
  return jnp.exp(- (S**2) / (2*sig**2))/scale

def compute_loss(model, state, X, Y):
  batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
  f, state, ypcl, sig = batch_model(X, state)
  loss= -2 * (Gram(Y, ypcl, sig)@f.T).diagonal().sum()+(Gram(ypcl, ypcl, sig)*(f.T@f)).sum()
  return loss/X.shape[0], state

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


@hydra.main(config_name='toy_NKME', version_base=None, config_path="config")
def main(cfg):
  vX, vy, vXp, vyp, DATA = generate_data(cfg)
  was = np.zeros((cfg.data.num_seeds))
   
  seed = 5678
  key = jax.random.PRNGKey(seed)
  mkey, xkey, xkey2 = jax.random.split(key, 3)
  mkeys = jax.random.split(mkey, cfg.data.num_seeds)

  num_inputs = vX.shape[-1]
  ymin = np.min(vy, axis=1)
  ymax= np.max(vy, axis=1)
  ypcl = jnp.array(np.linspace(ymin,ymax,cfg.model.numAtom)).transpose((1,0,2))
  sig_init=jnp.broadcast_to(cfg.model.sig_init, (cfg.data.num_seeds,1))
  model, state = eqx.filter_vmap(eqx.nn.make_with_state(NN), in_axes=(None, None, 0, 0, 0))(num_inputs, cfg.model.numAtom, ypcl, sig_init, mkeys)

  if cfg.optimizer.weight_decay == True:
     optim = optax.adamw(cfg.optimizer.lr)
  else:
     optim = optax.adam(cfg.optimizer.lr)
  opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
  num_steps = int(vX.shape[1]/cfg.train.batch_size)*cfg.train.epoch

  yc = jnp.array(np.linspace(ymin,ymax,cfg.test.num_bin)).transpose((1,0,2))

  mlflow.set_experiment(cfg.mlflow.runname)
  with mlflow.start_run():
    log_params_from_omegaconf_dict(cfg)
    model, state = train_vec(model, state, optim, opt_state, vX, vy, cfg.train.batch_size, num_steps, cfg.data.num_seeds, xkey)

    kernelmean, sig = eqx.filter_vmap(eval_NKME)(model, state, Gram, vXp, yc)
    for i in range(cfg.data.num_seeds):
      points = NKME_herding(kernelmean[i], Gram, cfg.data.N_y, yc[i], sig[i])
      was[i] = wasserstein(vyp[i], points).mean()
      
    was_mean = np.mean(was)
    was_std = np.std(was)
    mlflow.log_metric("was_mean", was_mean)
    mlflow.log_metric("was_std", was_std)

    return was
  
if __name__ == '__main__':
   main()
  


     





    

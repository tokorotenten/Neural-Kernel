import os
import sys
from omegaconf import DictConfig, ListConfig

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax  
import equinox as eqx
from typing import Any, Callable, Tuple
from collections import defaultdict
import tqdm
import gymnax

from utils import BatchManager, RolloutManager
from model.CDQN_models import NN as NN

import hydra
from hydra import utils
import mlflow
from tqdm import tqdm

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


def policy(model, model_state, obs, action_size, eps, key):
    def random(subkey):
        return jax.random.choice(subkey, jnp.arange(action_size))
    def greedy(subkey):
        pmfs, state, atoms = model(obs, model_state)
        q_vals = (pmfs * atoms).sum(axis=-1)
        action = jnp.argmax(q_vals, axis=-1)
        return action
    explore = jax.random.uniform(key)<eps
    key, subkey = jax.random.split(key)
    action = lax.cond(explore, random, greedy, operand = subkey)
    return action


def bellman_loss(p_model, p_model_state, t_model, t_model_state, obs, actions, next_obs, rewards, dones, gamma, v_min, v_max):
    t_batch_model = jax.vmap(t_model, in_axes=(0, None), out_axes=(0, None, None))
    p_model = eqx.nn.inference_mode(p_model, False)
    p_batch_model = jax.vmap(p_model, in_axes=(0, None), out_axes=(0, None, None))

    next_pmfs, t_model_state, t_atoms = t_batch_model(next_obs, t_model_state)  # (batch_size, num_actions, num_atoms)
    next_pmfs = lax.stop_gradient(next_pmfs)
    next_vals = (next_pmfs * t_atoms).sum(axis=-1)  # (batch_size, num_actions)
    next_action = jnp.argmax(next_vals, axis=-1)  # (batch_size,)
    next_pmfs = next_pmfs[jnp.arange(next_pmfs.shape[0]), next_action]
    next_atoms = rewards.reshape((-1,1)) + gamma * t_atoms * (1 - dones.reshape((-1,1)))
    # projection
    delta_z = t_atoms[1] - t_atoms[0]
    tz = jnp.clip(next_atoms, a_min=(v_min), a_max=(v_max))
    b = (tz - v_min) / delta_z
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=t_atoms.shape[0] - 1)
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=t_atoms.shape[0] - 1)
    # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
    d_m_l = (u + (l == u).astype(jnp.float32) - b) * next_pmfs
    d_m_u = (b - l) * next_pmfs
    target_pmfs = jnp.zeros_like(next_pmfs)

    def project_to_bins(i, val):
        val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
        val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
        return val

    target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs)

    #loss
    pmfs, p_model_state, p_atoms = p_batch_model(obs, p_model_state)
    pmfs = pmfs[jnp.arange(pmfs.shape[0]), actions.astype(int)]
    old_pmfs_l = jnp.clip(pmfs, a_min=1e-5, a_max=1 - 1e-5)
    loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
    
    return loss, p_model_state


@eqx.filter_jit
def make_step_vec(p_model, p_model_state, t_model, t_model_state, optim, opt_state, obs, actions, next_obs, rewards, dones, gamma, v_min, v_max):
    grads, p_model_state = eqx.filter_vmap(eqx.filter_grad(bellman_loss, has_aux=True))(p_model, p_model_state, t_model, t_model_state, obs, actions, next_obs, rewards, dones, gamma, v_min, v_max)
    updates, opt_state = optim.update(grads, opt_state, p_model)
    p_model = eqx.apply_updates(p_model, updates)
    return p_model, p_model_state, opt_state

@partial(jax.jit, static_argnums=(0,1,2,4))
def exponential_schedule(start_e: float, end_e: float, duration: int, t: int, num_env):
    eps = end_e + (start_e - end_e) * jnp.exp(-1 * t / duration)
    eps_v = jnp.broadcast_to(eps, (num_env, ))
    return eps_v

@partial(jax.jit, static_argnums=(0,1,2,4))
def linear_schedule(start_e: float, end_e: float, duration: int, t: int, num_env):
    slope = (end_e - start_e) / duration
    eps = slope * t + start_e
    eps = jnp.clip(jnp.array(eps), a_min=jnp.array(end_e))
    eps_v = jnp.broadcast_to(eps, (num_env, ))
    return eps_v


@hydra.main(config_name='CDQN', version_base=None, config_path="config")
def main(cfg):
    #managers, env
    rollout_manager = RolloutManager(cfg.env_name, cfg.env_kwargs, cfg.env_params, policy)
    batch_manager = BatchManager(cfg.buffer_size, cfg.num_env, rollout_manager.action_size, rollout_manager.observation_space)
    
    #models initialize
    seed = 5678
    key = jax.random.PRNGKey(seed)
    key, mkey = jax.random.split(key, 2)
    mkeys = jax.random.split(mkey, cfg.num_env)

    atoms = jnp.array(np.linspace(cfg.model.v_min, cfg.model.v_max, cfg.model.numAtom)).squeeze()
    v_atoms = jnp.broadcast_to(atoms, (cfg.num_env, atoms.shape[0]))
    p_model, p_model_state = eqx.filter_vmap(eqx.nn.make_with_state(NN))(rollout_manager.observation_space, cfg.model.numAtom*rollout_manager.action_size, v_atoms, mkeys)
    t_model = p_model
    t_model_state = p_model_state
    t_model = eqx.nn.inference_mode(t_model)

    #optimizer
    optim = optax.adam(learning_rate = cfg.optimizer.lr, eps = cfg.optimizer.eps)
    opt_state = optim.init(eqx.filter(p_model, eqx.is_inexact_array))

    epsilon_init = jnp.broadcast_to(jnp.array(1.0), (cfg.num_env, ))
    epsilon_test = jnp.broadcast_to(jnp.array(cfg.test.test_epsilon), (cfg.num_env, ))


    @eqx.filter_jit
    def get_transition(model, model_state, obs, state, batch, eps, num_train_envs, key):
        #inference mode
        inference_model = eqx.nn.inference_mode(model)

        key, key_act = jax.random.split(key)
        key_acts = jax.random.split(key_act, num_train_envs)
        action = rollout_manager.select_action(inference_model, model_state, obs, eps, key_acts)
        key, key_step = jax.random.split(key)
        key_steps = jax.random.split(key_step, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(key_steps, state, action)
        batch = batch_manager.append(batch, obs, action, reward, next_obs, done)
        return next_obs, next_state, done, batch
    
    #initialize managers
    batch = batch_manager.reset()
    key, key_step, key_reset, key_eval, key_update, key_buffer = jax.random.split(key, 6)
    obs, state = rollout_manager.batch_reset(jax.random.split(key_reset, cfg.num_env))
    #initialize buffer
    for i in range(cfg.buffer_size):
        key_buffer, sub_key_buffer =  jax.random.split(key_buffer, 2)
        obs, state, done, batch = get_transition(p_model, p_model_state, obs, state, batch, epsilon_init, cfg.num_env, sub_key_buffer)

    #episodes
    eval_count = 0
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        for steps in tqdm(range(cfg.train.steps)):
            #transition, eps update
            eps = linear_schedule(1.0, 0.01, 10000, steps, cfg.num_env)
            key_step, sub_key_step =  jax.random.split(key_step, 2)
            obs, state, done, batch = get_transition(p_model, p_model_state, obs, state, batch, eps, cfg.num_env, sub_key_step)
            
            #update
            if steps % cfg.train.policy_update_period == 0:
                key_update, sub_key_update =  jax.random.split(key_update, 2)
                sub_key_updates =  jax.random.split(sub_key_update, cfg.num_env)
                rb_obs, rb_actions, rb_rewards, rb_next_obs, rb_dones = batch_manager.get(batch, cfg.train.batch_size, sub_key_updates)
                p_model, p_model_state, opt_state = make_step_vec(p_model, p_model_state, t_model, t_model_state, optim, opt_state, rb_obs, rb_actions, rb_next_obs, rb_rewards, rb_dones, cfg.train.gamma, cfg.model.v_min, cfg.model.v_max)
    

            #target update
            if steps % cfg.train.target_update_period == 0:
                del t_model
                del t_model_state
                t_model = p_model
                t_model_state = p_model_state
                t_model = eqx.nn.inference_mode(t_model)
                
             #evaluate
            if steps % cfg.test.evaluate_period == 0:
                key_eval, sub_key_eval =  jax.random.split(key_eval, 2)
                rewards = rollout_manager.batch_evaluate(p_model, p_model_state, cfg.num_env, epsilon_test, sub_key_eval)
                rewards_mean = jnp.mean(rewards)
                rewards_std = jnp.std(rewards)
                mlflow.log_metric("reward_mean", rewards_mean, step = eval_count)
                mlflow.log_metric("reward_std", rewards_std, step = eval_count)
                eval_count += 1

        return p_model, p_model_state


if __name__ == '__main__':
   main()




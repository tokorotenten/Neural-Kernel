from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx
import gymnax

"""
Part of the codes here are taken from the Github repository of gymnax-blines
https://github.com/RobertTLange/gymnax-blines/blob/main/utils/ppo.py
"""

class BatchManager:
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        action_size,
        obs_space
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.action_size = action_size
        self.obs_shape = obs_space
        self.reset()

    def reset(self):
        return {
            "obs": jnp.empty((self.num_envs, self.buffer_size, self.obs_shape),dtype=jnp.float32,),
            "actions": jnp.empty((self.num_envs, self.buffer_size)),
            "rewards": jnp.empty((self.num_envs, self.buffer_size), dtype=jnp.float32),
            "next_obs": jnp.empty((self.num_envs, self.buffer_size, self.obs_shape),dtype=jnp.float32,),
            "dones": jnp.empty((self.num_envs, self.buffer_size), dtype=jnp.uint8),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, obs, action, reward, next_obs, done):
        return {
                "obs":  buffer["obs"].at[:,buffer["_p"],:].set(obs),
                "actions": buffer["actions"].at[:,buffer["_p"]].set(action),
                "rewards": buffer["rewards"].at[:,buffer["_p"]].set(reward),
                "next_obs": buffer["next_obs"].at[:,buffer["_p"],:].set(next_obs),
                "dones": buffer["dones"].at[:,buffer["_p"]].set(done),
                "_p": (buffer["_p"] + 1) % self.buffer_size,
            }

    @partial(jax.jit, static_argnums=(0,2))
    def get(self, buffer, batchsize, keys):
        def sample_batch_obs(X, batch_size, key):
             indexes = jax.random.randint(key, shape=(batch_size, ), minval=0, maxval=self.buffer_size)
             return X[indexes]

        def sample_batch(X, batch_size, key):
            indexes = jax.random.randint(key, shape=(batch_size, ), minval=0, maxval=self.buffer_size)
            return X[indexes]
        
        sample_batch_vec_obs = jax.vmap(sample_batch_obs, in_axes=(0,None,0))
        sample_batch_vec = jax.vmap(sample_batch, in_axes=(0,None,0))

        batch = (
            sample_batch_vec_obs(buffer["obs"], batchsize, keys),
            sample_batch_vec(buffer["actions"], batchsize, keys),
            sample_batch_vec(buffer["rewards"], batchsize, keys),
            sample_batch_vec_obs(buffer["next_obs"], batchsize, keys),
            sample_batch_vec(buffer["dones"], batchsize, keys),
        )
        return batch


class RolloutManager(object):
    def __init__(self, env_name, env_kwargs, env_params, policy):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.observation_space = self.env.observation_space(self.env_params).shape[0]
        self.action_size = self.env.num_actions
        self.policy = policy
        self.select_action = self.select_action

    @eqx.filter_jit
    def select_action(self, model, model_state, obs, eps, keys):
        #v_policy = eqx.filter_vmap(self.policy, in_axes=(0, None, 0, None, None, 0))
        v_policy = eqx.filter_vmap(self.policy)
        action = v_policy(model, model_state, obs, self.action_size, eps, keys)
        return action

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @eqx.filter_jit
    def batch_evaluate(self, model, model_state, num_envs, eps, rng_input):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))
        #inference mode
        inference_model = eqx.nn.inference_mode(model)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.select_action(inference_model, model_state, obs, eps, jax.random.split(rng_net, num_envs))
            next_o, next_s, reward, done, _ = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action,
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = [
                next_o,
                next_s,
                #train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ], [new_valid_mask]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                #train_state,
                rng_episode,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )

        cum_return = carry_out[-2].squeeze()
        return cum_return
    
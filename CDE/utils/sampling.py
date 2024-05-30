import numpy as np
import math
import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

def eval_NKME(model, state, Gram, Xpp, yc):
  inference_model = eqx.nn.inference_mode(model)
  batch_inference_model = jax.vmap(inference_model, in_axes=(0, None), out_axes=(0, None, None, None))
  f, state, ypcl, sig = batch_inference_model(Xpp, state)
  f=f/(f.sum(1, keepdims = True))
  kernelmean=(Gram(yc, ypcl, sig)@f.T).T
  return kernelmean,sig

def eval_DF(model, state, Gram, X, y, Xpp, yc):
  inference_model = eqx.nn.inference_mode(model)
  batch_inference_model = jax.vmap(inference_model, in_axes=(0, None), out_axes=(0, None, None, None))
  f, state, lamb, sig = batch_inference_model(X, state)
  f2, state2, lamb2, sig2 = batch_inference_model(Xpp, state)
  w = f@jnp.linalg.solve(f.T@f+lamb*jnp.eye(f.shape[1]), f2.T)
  kernelmean=(Gram(yc, y, sig)@w).T
  return kernelmean, sig

def NKME_herding(mean_embedding, Gram, N_y, yc, sig):
  super_samples = jnp.zeros((mean_embedding.shape[0], N_y, yc.shape[1]))
  @jax.jit
  def herding_objective(super_samples, yc, mu, mu_hat_sum, sig, i):
    mu_hat = mu_hat_sum / (i + 1)
    objective = mu - mu_hat
    super_samples = super_samples.at[:,i,:].set(yc[jnp.argmax(objective, axis=1)])
    mu_hat_sum_updated = mu_hat_sum + Gram(super_samples[:,i,:], yc, sig)
    return super_samples, mu_hat_sum_updated

  mu = mean_embedding
  mu_hat_sum = jnp.zeros((mu.shape[0], mu.shape[1]))
  for i in range(N_y):
    super_samples, mu_hat_sum = herding_objective(super_samples, yc, mu, mu_hat_sum, sig, i)
  return super_samples



def eval_MDN(model, state, Xpp):
  inference_model = eqx.nn.inference_mode(model)
  batch_inference_model = jax.vmap(inference_model, in_axes=(0, None), out_axes=(None, 0, 0, 0))
  state, mu, logstd, logmix = batch_inference_model(Xpp, state)
  sigma = jnp.exp(logstd)
  return mu, logmix, sigma

def sample_MDN(mu, logmix, sigma, N_y, key):
  samples=jnp.zeros((mu.shape[0], N_y, 1))
  @jax.jit
  def sample(mu, logmix, sigma, samples, i, key):
    key1, key2  = jax.random.split(key, 2)
    k = jax.random.categorical(key1, logmix)
    indices = (jnp.arange(mu.shape[0]), k)
    rn = jax.random.normal(key2, (mu.shape[0],1))
    sample = rn * jnp.expand_dims(sigma[indices],-1) + jnp.expand_dims(mu[indices], -1)
    samples =samples.at[:,i,:].set(sample)
    return samples
  for i in range(N_y):
    key, subkey  = jax.random.split(key, 2)
    samples = sample(mu, logmix, sigma, samples, i, subkey)
  return samples
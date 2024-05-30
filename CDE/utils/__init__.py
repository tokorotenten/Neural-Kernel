import numpy as np
import jax
import jax.numpy as jnp


def sample_batch(X, Y, batch_size, key):
  indexes = jax.random.randint(key, shape=(batch_size, ), minval=0, maxval=X.shape[0])
  return X[indexes], Y[indexes]
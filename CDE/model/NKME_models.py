import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx
import math


class toy_NN(eqx.Module):
  layers: list
  ypcl: jax.Array
  sig: jax.Array
  def __init__(self, num_inputs, num_outputs, ypcl, sig_init, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(num_inputs, 50, key=key1),
            eqx.nn.Linear(50, 50, key=key2),
            eqx.nn.Linear(50, num_outputs, key=key3),
        ]
        self.ypcl = ypcl
        self.sig = jnp.log(jnp.expm1(sig_init))

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x = jax.nn.relu(self.layers[1](x))
      x = self.layers[2](x)
      sig = jax.nn.softplus(self.sig)

      return x, state, lax.stop_gradient(self.ypcl), sig
  

class uci_NN_SN1(eqx.Module):
  layers: list
  ypcl: jax.Array
  sig: jax.Array
  def __init__(self, num_inputs, num_outputs, ypcl, sig_init, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.layers = [
            eqx.nn.Linear(num_inputs, 50, key=key1),
            eqx.nn.SpectralNorm(eqx.nn.Linear(50, 50, key=key2), weight_name="weight", key=key3),
            eqx.nn.Linear(50, 50, key=key4),
            eqx.nn.SpectralNorm(layer=eqx.nn.Linear(50, num_outputs, key=key5), weight_name="weight", key=key6),
        ]
        self.ypcl = ypcl
        self.sig = jnp.log(jnp.expm1(sig_init))

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x, state = self.layers[1](x, state)
      x = jax.nn.relu(x)
      x = jax.nn.relu(self.layers[2](x))
      x, state = self.layers[3](x, state)
      sig = jax.nn.softplus(self.sig)

      return x, state, lax.stop_gradient(self.ypcl), sig
  

class uci_NN_SN2(eqx.Module):
  layers: list
  ypcl: jax.Array
  sig: jax.Array
  def __init__(self, num_inputs, num_outputs, ypcl, sig_init, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.layers = [
            eqx.nn.Linear(num_inputs, 100, key=key1),
            eqx.nn.SpectralNorm(eqx.nn.Linear(100, 100, key=key2), weight_name="weight", key=key3),
            eqx.nn.Linear(100, 100, key=key4),
            eqx.nn.SpectralNorm(layer=eqx.nn.Linear(100, num_outputs, key=key5), weight_name="weight", key=key6),
        ]
        self.ypcl = ypcl
        self.sig = jnp.log(jnp.expm1(sig_init))

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x, state = self.layers[1](x, state)
      x = jax.nn.relu(x)
      x = jax.nn.relu(self.layers[2](x))
      x, state = self.layers[3](x, state)
      sig = jax.nn.softplus(self.sig)

      return x, state, lax.stop_gradient(self.ypcl), sig
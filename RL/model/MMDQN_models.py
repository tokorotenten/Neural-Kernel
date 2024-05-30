import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx
import math


class NN(eqx.Module):
  layers: list
  num_atoms:int
  num_outputs: int
  sig: jax.Array
  def __init__(self, num_inputs, num_outputs, num_atoms, sig, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(num_inputs, 50, key=key1),
            eqx.nn.Linear(50, 50, key=key2),
            eqx.nn.Linear(50, num_outputs, key=key3),
        ]
        self.num_atoms = num_atoms
        self.num_outputs = num_outputs
        self.sig = sig

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x = jax.nn.relu(self.layers[1](x))
      x = self.layers[2](x)
      x = x.reshape(int(self.num_outputs/self.num_atoms), self.num_atoms)

      return x, state, lax.stop_gradient(self.sig)
  
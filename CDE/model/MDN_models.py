import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx



class toy_NN(eqx.Module):
  layers: list

  def __init__(self, num_inputs, num_outputs, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Linear(num_inputs, 50, key=key1),
            eqx.nn.Linear(50, 50, key=key2),
            eqx.nn.Linear(50, num_outputs, key=key3),
            eqx.nn.Linear(50, num_outputs, key=key4),
            eqx.nn.Linear(50, num_outputs, key=key5),
        ]

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x = jax.nn.relu(self.layers[1](x))
      mu = self.layers[-3](x)
      logstd = self.layers[-2](x)
      logmix = self.layers[-1](x)

      return state, mu, logstd, logmix
  
  
class uci_NN_SN1(eqx.Module):
  layers: list

  def __init__(self, num_inputs, num_outputs, key):
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)
        self.layers = [
            eqx.nn.Linear(num_inputs, 50, key=key1),
            eqx.nn.Linear(50, 50, key=key2),
            eqx.nn.SpectralNorm(eqx.nn.Linear(50, 50, key=key3), weight_name="weight", key=key4),
            eqx.nn.Linear(50, num_outputs, key=key5),
            eqx.nn.Linear(50, num_outputs, key=key6),
            eqx.nn.Linear(50, num_outputs, key=key7),
        ]

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x = jax.nn.relu(self.layers[1](x))
      x, state = self.layers[2](x, state)
      x = jax.nn.relu(x)
      mu = self.layers[-3](x)
      logstd = self.layers[-2](x)
      logmix = self.layers[-1](x)

      return state, mu, logstd, logmix
  
  
class uci_NN_SN2(eqx.Module):
  layers: list

  def __init__(self, num_inputs, num_outputs, key):
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)
        self.layers = [
            eqx.nn.Linear(num_inputs, 100, key=key1),
            eqx.nn.Linear(100, 100, key=key2),
            eqx.nn.SpectralNorm(eqx.nn.Linear(100, 100, key=key3), weight_name="weight", key=key4),
            eqx.nn.Linear(100, num_outputs, key=key5),
            eqx.nn.Linear(100, num_outputs, key=key6),
            eqx.nn.Linear(100, num_outputs, key=key7),
        ]

  def __call__(self, x, state):
      x = jax.nn.relu(self.layers[0](x))
      x = jax.nn.relu(self.layers[1](x))
      x, state = self.layers[2](x, state)
      x = jax.nn.relu(x)
      mu = self.layers[-3](x)
      logstd = self.layers[-2](x)
      logmix = self.layers[-1](x)

      return state, mu, logstd, logmix
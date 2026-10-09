import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence



# A multi-layer perceptron using ELU activation functions
class MLP(nn.Module):
	'''
	A Multi-Layer Perceptron model.

	Args:
			features: A sequence of integers defining the number of neurons in each layer.
								Example: [64, 32, 2] creates two hidden layers and one output layer.
	
	Returns:
			The output of the network (logits/predictions) after the forward pass.
	'''
	features: Sequence[int]

	@nn.compact
	def __call__(self, x):
		for feat in self.features[:-1]:
			x = nn.elu(nn.Dense(feat)(x))
		x = nn.Dense(self.features[-1])(x)
		return x  
	



class ActorNetwork:
	"""
	Actor Network for the CACTO framework.
	
	This class implements a neural network responsible for generating continuous 
	control actions based on the current system state. The architecture relies on 
	a Multi-Layer Perceptron (MLP). It is optimized using JAX and Optax, ensuring 
	computational graphs are compiled only once to maximize execution speed.
	"""
	def __init__(self, name, in_dim, out_dim, layer_sizes, u_max, random_seed=0):
		self.name = name
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.u_max = u_max
		self.layer_sizes = layer_sizes # Used later to save the model architecture for future loading

		# Initialize network architecture, system dynamics, and cost functions
		self.model = MLP(layer_sizes + [out_dim])

		# Seed the random number generator and initialize network parameters
		k1, k2 = jax.random.split(jax.random.PRNGKey(random_seed))
		x = jax.random.normal(k1, (in_dim,)) 
		self.params = self.model.init(k2, x)


	def __call__(self, x):
		"""
		Forward pass for inference. Applies tanh to keep outputs strictly within physical limits.
		"""
		raw_u = self.model.apply(self.params, x)
		return self.u_max * jnp.tanh(raw_u)
	
	
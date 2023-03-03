import numpy as np
import math
import scipy.special
import random

random.seed(42)


class network:
	def __init__(self, hidden_neurons, activation, activation_deriv, learning_rate=0.1, hidden_layers_len=1):
		self.max_n = hidden_neurons
		self.controller = np.random.random((1, self.max_n))

		self.hidden_layers = []
		self.output_layer = None

		self.activation = activation
		self.activation_deriv = activation_deriv
		self.learning_rate = learning_rate
		self.layer_io = []
		self.errors = 0

	def reset_errors(self):
		self.errors = 0

	def change_hidden_neuron(self, hl, ol):
		self.hidden_layers = [hl]
		self.output_layer = ol

	def change_controller(self, values):
		self.controller = np.array(values, ndmin=2).reshape(1, self.max_n)

	def forward(self, inputs):
		self.layer_io.append(inputs)
		layer = self.activation(inputs.dot(self.hidden_layers[0])) * self.controller
		self.layer_io.append(layer)
		out = self.activation(layer.dot(self.output_layer))
		self.layer_io.append(out.T)
		self.layer_io.reverse()
		return out

	def backprop(self, targets):
		output_delta = targets - self.layer_io[0]
		output_update = self.learning_rate * np.dot(output_delta * self.activation_deriv(self.layer_io[0]), self.layer_io[1])
		layer_delta = self.output_layer.dot(output_delta)
		self.output_layer += output_update.T

		for x in range(self.max_n):
			layer_update = self.learning_rate * np.dot(layer_delta * self.activation_deriv(self.layer_io[x+1]).T, self.layer_io[x+2])
			if self.max_n > 1 and x + 1 != self.max_n:
				layer_delta = self.hidden_layers[-(x+1)].dot(layer_delta)
			self.hidden_layers[-(x+1)] += layer_update.T

		# self.errors += abs(np.sum(output_delta))
		return abs(np.sum(output_delta))

	def train(self, inputs, targets):
		"inputs shapes should be (n, 1). targets shape should be (n, 1)"
		self.forward(inputs)
		return self.backprop(targets)

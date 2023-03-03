import numpy as np
import math
import scipy.special
import random

max_n = 8
random.seed(42)


class network:
	def __init__(self, input_neurons_len, hidden_neurons_len, output_neurons_len, 
			activation, activation_deriv, learning_rate=0.1, hidden_layers_len=1):
		self.input_neurons_len = input_neurons_len
		self.hidden_neurons_len = hidden_neurons_len
		self.output_neurons_len = output_neurons_len
		self.hidden_layers_len = hidden_layers_len
		# it only works for one layer
		self.neuron1 = np.random.normal(0.0, pow(1, -0.5), (self.input_neurons_len, max_n))
		self.neuron2 = np.random.normal(0.0, pow(1, -0.5), (max_n, self.output_neurons_len))

		self.controller = np.random.random((1, max_n))

		self.hidden_layers = []
		self.output_layer = None

		self.activation = activation
		self.activation_deriv = activation_deriv
		self.learning_rate = learning_rate
		self.layer_io = []
		self.errors = 0

	def reset_weights(self):
		self.hidden_layers = [np.random.normal(0.0, pow(self.input_neurons_len, -0.5), (self.hidden_neurons_len, self.input_neurons_len))]
		# for x in range(self.hidden_layers_len-1):
			# self.hidden_layers.append(np.random.normal(0.0, pow(self.hidden_neurons_len, -0.5), (self.hidden_neurons_len, self.hidden_neurons_len)))
		self.output_layer = np.random.normal(0.0, pow(self.hidden_neurons_len, -0.5), (self.output_neurons_len, self.hidden_neurons_len))

	def reset_errors(self):
		self.errors = 0


	def change_hidden_neuron(self, count=max_n):
		count = round(count)
		if count <= 0:
			return False
		
		self.hidden_neurons_len = count
		if count == max_n:
			self.hidden_layers = [self.neuron1.copy()]
			self.output_layer = self.neuron2.copy()
		else:
			self.hidden_layers = [self.neuron1.T[:count].copy().T]
			self.output_layer = self.neuron2[:count].copy()
		
		return True

	def change_controller(self, values):
		self.controller = np.array(values, ndmin=2).reshape(1, self.hidden_neurons_len)

	def forward(self, inputs):
		self.layer_io.append(inputs)
		layer = self.activation(inputs.dot(self.hidden_layers[0])) * self.controller
		self.layer_io.append(layer)
		# for x in self.hidden_layers[1:]:
		# 	layer = self.activation(x.dot(layer))
		# 	self.layer_io.append(layer)
		out = self.activation(layer.dot(self.output_layer)).T
		self.layer_io.append(out)
		self.layer_io.reverse()
		return out

	def backprop(self, targets):
		output_delta = targets - self.layer_io[0]
		output_update = self.learning_rate * np.dot(output_delta * self.activation_deriv(self.layer_io[0]), self.layer_io[1])
		layer_delta = self.output_layer.dot(output_delta)
		self.output_layer += output_update.T

		for x in range(self.hidden_layers_len):
			layer_update = self.learning_rate * np.dot(layer_delta * self.activation_deriv(self.layer_io[x+1]).T, self.layer_io[x+2])
			if self.hidden_layers_len > 1 and x + 1 != self.hidden_layers_len:
				layer_delta = self.hidden_layers[-(x+1)].dot(layer_delta)
			self.hidden_layers[-(x+1)] += layer_update.T

		# self.errors += abs(np.sum(output_delta))
		return abs(np.sum(output_delta))

	def train(self, inputs, targets):
		"inputs shapes should be (n, 1). targets shape should be (n, 1)"
		self.forward(inputs)
		return self.backprop(targets)

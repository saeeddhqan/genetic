import numpy as np
import math
import scipy.special
import random
import concurrent.futures
import multiprocessing
import time
import utils

max_n = 100
# random.seed(42)


class network:
	def __init__(self, input_neurons_len, hidden_neurons_len, output_neurons_len, 
			activation, activation_deriv, learning_rate=0.1, hidden_layers_len=1):
		self.input_neurons_len = input_neurons_len
		self.hidden_neurons_len = hidden_neurons_len
		self.output_neurons_len = output_neurons_len
		self.hidden_layers_len = hidden_layers_len
		# it only works for one layer
		self.neuron1 = np.random.normal(0.0, pow(1, -0.5), (max_n, self.input_neurons_len))
		self.neuron2 = np.random.normal(0.0, pow(1, -0.5), (self.output_neurons_len, max_n))

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


	def change_hidden_neuron(self, count):
		count = round(count)
		if count <= 0:
			return False

		self.hidden_layers = [self.neuron1[:count].copy()]
		self.output_layer = self.neuron2.T[:count].T.copy()
		self.hidden_neurons_len = count
		return True

	def forward(self, inputs):
		self.layer_io.append(inputs)
		layer = self.activation(self.hidden_layers[0].dot(inputs))
		self.layer_io.append(layer)
		for x in self.hidden_layers[1:]:
			layer = self.activation(x.dot(layer))
			self.layer_io.append(layer)
		out = self.activation(layer.T.dot(self.output_layer.T)).T
		self.layer_io.append(out)
		self.layer_io.reverse()
		return out

	def backprop(self, targets):
		output_delta = targets - self.layer_io[0]
		output_update = self.learning_rate * np.dot(output_delta * self.activation_deriv(self.layer_io[0]), self.layer_io[1].T)
		layer_delta = (self.output_layer * output_delta).sum(0).reshape(self.hidden_neurons_len, 1) # maybe it's a mistake?
		self.output_layer += output_update

		for x in range(self.hidden_layers_len):
			layer_update = self.learning_rate * np.dot(layer_delta * sigmoid_deriv(self.layer_io[x+1]), self.layer_io[x+2].T)
			if self.hidden_layers_len > 1 and x + 1 != self.hidden_layers_len:
				layer_delta = (self.hidden_layers[-(x+1)] * layer_delta).sum(0).reshape(self.hidden_neurons_len, 1)
			self.hidden_layers[-(x+1)] += layer_update

		# self.errors += abs(np.sum(output_delta))
		return abs(np.sum(output_delta))

	def train(self, inputs, targets):
		"inputs shapes should be (n, 1). targets shape should be (n, 1)"
		self.forward(inputs)
		return self.backprop(targets)


input_neurons = 784
hidden_neurons = 1
output_neurons = 10
hidden_layers = 1
learning_rate = 0.1
epoch = 9

def sigmoid(x):
	return 1/(1+(math.e**-x))

def relu(x):
	return (x > 0) * x

def relu_deriv(x):
	return (x > 0)

def sigmoid_deriv(x):
	return x * (1 - x)

net = network(input_neurons, hidden_neurons, output_neurons, sigmoid, sigmoid_deriv, learning_rate, hidden_layers)


training_data_file = open("../neural_netwok/gradient_descent/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

test_data_file = open("../neural_netwok/gradient_descent/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

def func_error(param):
	errors = 0
	param = param * max_n
	net.change_hidden_neuron(param)
	# net.reset_errors()
	for i in range(epoch):
		for row in training_data_list:
			row = row[:-1].split(',')
			targets = np.zeros(10) + 0.01
			targets[int(row[0])] = 0.99
			targets = np.array(targets, ndmin=2).T
			inputs = (np.asfarray(row[1:]) / 255.0 * 0.99) + 0.01
			inputs = np.array(inputs, ndmin=2).T
			errors += net.train(inputs, targets)

	scorecard = []
	for record in test_data_list:
		all_values = record.split(',')
		target = int(all_values[0])
		inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		inputs = np.array(inputs, ndmin=2).T
		outputs = net.forward(inputs)
		label = np.argmax(outputs)
		if (label == target):
			scorecard.append(1)
		else:
			scorecard.append(0)
	scorecard_array = np.asarray(scorecard)
	return scorecard_array.sum()/scorecard_array.size


# def calculate_lr(epoch, w, i, lr):
# 	neu = lambda w,i: sigmoid(w*i)
# 	points = 0
# 	epsilon = 0.000001
# 	direction = +1
# 	o = neu(w,i) 
# 	a, b = func_error(o)[0], func_error(o + epsilon)[0]
# 	for x in range(epoch):
# 		direction = -1 if a - b < 0.0 else +1
# 		update = lr * b * direction * i * sigmoid_deriv(o)
# 		# if w+update < 0:
# 		# 	update = epsilon * direction

# 		w += update
# 		o = neu(w,i)
# 		a = b
# 		b, p = func_error(o)
# 		# a, b, p = b, func_error(o)
# 		# print(x, o * max_n, b, w, direction, p)
# 		points += p
# 	return points



lepoch = 5

def fitness_func(params):
	print(params, end='\r')
	return 1- (0.2 * params)
	# return func_error(params)


init_range = (0.001, 1)
pops = 100
gens = 200

inds = []
for i in range(1, 10+1):
	mx = i * 0.1
	mn = mx - 0.09
	for j in range(int(pops/10)):
		inds.append(random.uniform(mn, mx))

# inds = [random.random() for _ in range(pops)]
for generation in range(gens):
	pool = multiprocessing.Pool(20)
	fitness_values = pool.map(fitness_func, inds)
	passed_inds_idx = utils.sel_top_k(fitness_values, int(pops/2))
	passed_inds = [inds[x] for x in passed_inds_idx]
	new_offsprings = utils.cx_hole_digging(passed_inds)
	[passed_inds.append(x) for x in new_offsprings]
	if passed_inds_idx == []:
		break
	max_ind = inds[passed_inds_idx[0]]
	max_perf = fitness_values[passed_inds_idx[0]]
	print('inds_next=', len(passed_inds), ',top ind=', max_ind, ',top ind p=', max_perf, ',sum=', sum(fitness_values)/len(fitness_values),
		',inds_now=', len(inds))

	# inds = passed_inds
	inds = utils.mut_random_epsilon(passed_inds)
	# pops -= 5
print(inds)
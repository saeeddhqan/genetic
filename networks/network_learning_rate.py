import numpy as np
import math
import scipy.special
import random
import concurrent.futures
import multiprocessing
import time

import random

random.seed(42)

from networks import network
# import network


input_neurons = 3
hidden_neurons = 1
output_neurons = 3
hidden_layers = 1
learning_rate = 0.1
epoch = 1

def sigmoid(x):
	return 1/(1+(math.e**-x))

def relu(x):
	return (x > 0) * x

def relu_deriv(x):
	return (x > 0)

def sigmoid_deriv(x):
	return x * (1 - x)

net = network.network(input_neurons, hidden_neurons, output_neurons, sigmoid, sigmoid_deriv, learning_rate, hidden_layers)


training_data_file = open("networks/data")
# training_data_file = open("data")
data_list = [x for x in training_data_file.read().split('\n')[1:] if ('Red' in x or 'Green' in x or 'Blue' in x)]
training_data_file.close()
training_data_list = []
test_data_list = []

for row in data_list[:int(0.7 * len(data_list))]:
	row = row.split(',')
	if row[3] == 'Red':
		target = [0.99,0.01,0.01]
	elif row[3] == 'Green':
		target = [0.01,0.99,0.01]
	else:
		target = [0.01,0.01,0.99]
	targets = np.array(target, ndmin=2).T
	inputs = ((np.asfarray(np.array([x for x in row[:3]], ndmin=2)) / 255.0 * 0.99) + 0.01)
	training_data_list.append((inputs, targets))

for row in data_list[int(0.7 * len(data_list)):]:
	row = row.split(',')
	if row[3] == 'Red':
		target = [0.99,0.01,0.01]
	elif row[3] == 'Green':
		target = [0.01,0.99,0.01]
	else:
		target = [0.01,0.01,0.99]
	targets = np.array(target, ndmin=2).T
	inputs = ((np.asfarray(np.array([x for x in row[:3]], ndmin=2)) / 255.0 * 0.99) + 0.01)
	test_data_list.append((inputs, targets))


def func_error(param):
	errors = 0
	net.change_hidden_neuron()
	net.change_controller(param)
	for i in range(epoch):
		for row in training_data_list:
			errors += net.train(row[0], row[1])

	score = 0
	for row in test_data_list:
		outputs = net.forward(row[0])
		label = np.argmax(outputs)
		score += 1 if label == np.argmax(row[1]) else 0
	return score/len(test_data_list)

def fitness_func(params):
	return func_error(params),

# for i in range(1,2):
# 	print(fitness_func([0.12787446961849971, 0.2703588245462086, 0.6475856660926073, 0.9942801959168225, 0.5896659511394797, 0.2061116118842229, 0.7570567983156401, 0.44364691836076375, 0.6821462364782024, 0.006408612770593269, 0.33976783187043413, 0.16804943072545134]))
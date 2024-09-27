import numpy as np
import math
import scipy.special
import random
import concurrent.futures
import multiprocessing
import time

import network

max_n = 100


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

net = network.network(input_neurons, hidden_neurons, output_neurons, sigmoid, sigmoid_deriv, learning_rate, hidden_layers)
net.change_hidden_neuron(80)


training_data_file = open("../../neural_netwok/gradient_descent/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

test_data_file = open("../../neural_netwok/gradient_descent/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

def func_error(param):
	errors = 0
	net.learning_rate = param
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


def fitness_func(params):
	# print(params, end='\r')
	return func_error(params)


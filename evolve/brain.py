

import time
import numpy as np
import pickle
import math
import random
from net import network

def sigmoid(x):
	return 1/(1+(math.e**-x))
def sigmoid_deriv(x):
	return x * (1 - x)

SHAPE = (74,33)

INPUTS = 2
HIDDEN_LAYER = 11
OUTPUTS = 4

class lunar_lander:

	def __init__(self, pops, shape, random_seed=42):
		self.shape = shape
		self.pop_positions = [72/74, 3/33]
		self.seed = random_seed
		random.seed(random_seed)
		self.wall_x = 20
		self.wall_y = 20
		self.barriers = [(self.wall_x, y) for y in range(self.wall_y)]

	def __len__(self):
		return INPUTS * HIDDEN_LAYER + HIDDEN_LAYER * OUTPUTS + HIDDEN_LAYER

	def initMlp(self, netParams):
		net = network.network(HIDDEN_LAYER, sigmoid, sigmoid_deriv)
		numWeights = INPUTS * HIDDEN_LAYER + HIDDEN_LAYER * OUTPUTS
		weights = np.array(netParams[:numWeights])
		hl = weights[0:INPUTS * HIDDEN_LAYER].reshape((INPUTS, HIDDEN_LAYER))
		ol = weights[INPUTS * HIDDEN_LAYER:].reshape((HIDDEN_LAYER, OUTPUTS))
		biases = netParams[numWeights:]
		net.change_hidden_neuron(hl, ol)
		net.change_controller(biases)
		return net

	def change_pos_and_reward(self, action, position):
		"""clockwise move"""
		rewards = 0
		wall = 0
		if action == 0:
			if position[1]-1 != -1: 
				if not (position[0] == self.wall_x and position[1]-1 == self.wall_y):
					position[1] -= 1
				else:
					rewards += -1.0
				rewards +=  -0.5
			else:
				rewards += -0.5
			wall = -1
		elif action == 1:
			if position[0]+1 <= self.shape[0]: # creature shouldn't go beyond the boundary
				if not (position[0]+1 == self.wall_x and position[1] <= self.wall_y): # creature shouldn't collide with the walls
					position[0] += 1
				else:
					rewards += -1.5
				rewards += -10.5
			else:
				rewards += -0.5
			wall = 1
		elif action == 2:
			if position[1]+1 <= self.shape[1]:
				if position[0]-1 == self.wall_x and position[1] <= self.wall_y: #
					rewards += 10
				else:
					rewards += -0.5
				position[1] += 1
			else:
				rewards += -0.5
			wall = 2
		elif action == 3:
			if position[0]-1 != -1: 
				if not (position[0]-1 == self.wall_x and position[1] <= self.wall_y): # !wall collision
					position[0] -= 1
					rewards += 1.5 + (int(self.shape[0]/2) - position[0])
					wall = "yyyyy"
				else:
					wall = "xxxxxxxxxxx"
					rewards += -10.0
			else:
				rewards += -0.5
			wall = 3
		if position[0] > int(self.shape[0]/2):
			if action == 3:
				rewards += 1.5
			elif action == 1:
				rewards += -3
			else:
				rewards += -2.5

		position = [position[0]/self.shape[0], position[1]/self.shape[1]]

		return rewards, position, position[0] == 0, wall

	def getScore(self, netParams, idx):
		mlp = self.initMlp(netParams)

		totalReward = 0
		observation = self.pop_positions
		position = [round(observation[0] * self.shape[0]), round(observation[1] * self.shape[1])]

		for i in range(1, self.shape[0]+1):
			action = mlp.forward(np.array([observation[0], observation[1]], ndmin=2)).argmax()
			dosomething = self.change_pos_and_reward(action, position)
			totalReward += dosomething[0]
			observation = dosomething[1]
			position = [round(observation[0] * self.shape[0]), round(observation[1] * self.shape[1])]

			if dosomething[2]:
				break
		# totalReward = totalReward * (100/counter)
		return totalReward

	def saveParams(self, netParams):
		savedParams = []
		for param in netParams:
			savedParams.append(param)

		pickle.dump(savedParams, open("left-right-data.pickle", "wb"))

	def replayWithSavedParams(self):
		savedParams = pickle.load(open("left-right-data.pickle", "rb"))
		self.replay(savedParams)

	def replay(self, netParams):
		mlp = self.initMlp(netParams)
		from asciimatics.screen import Screen
		from time import sleep
		import numpy as np

		COLOUR_GREEN = 2

		def demo(screen):
			x = screen.dimensions[1]
			y = screen.dimensions[0]

			leftright = int(x/4)
			updown = int(y/4)
			observation = self.pop_positions
			position = [round(observation[0] * self.shape[0]), round(observation[1] * self.shape[1])]

			while True:
				# screen.print_at(f'{position}\t{self.barriers[0][0]+leftright+1}\t{leftright}', 3, 2, 3)
				for i in range(updown, y-updown):
					screen.print_at(f'▍', leftright, i, COLOUR_GREEN)
					screen.print_at(f'▍', x-leftright, i, COLOUR_GREEN)
				for j in range(leftright, x-leftright):
					screen.print_at(f'▁', j, 0, COLOUR_GREEN)
					screen.print_at(f'▁', j, y-1, COLOUR_GREEN)
				for i in self.barriers:
					screen.print_at(f'▍', i[0]+(leftright+1), i[1]+1, 3)

				screen.print_at('x', position[0]+(leftright+1), position[1]+1, 2)
				action = mlp.forward(np.array([observation[0], observation[1]], ndmin=2)).argmax()
				dosomething = self.change_pos_and_reward(action, position)
				observation = dosomething[1]
				position = [round(observation[0] * self.shape[0]), round(observation[1] * self.shape[1])]
				screen.print_at(f'{position}\t{dosomething[3]}\t{self.barriers[0][0]+leftright+1}\t{leftright}', 3, 2, 3)
				screen.refresh()
				sleep(0.3)
				screen.clear()

		Screen.wrapper(demo)



def main():
	cart = lunar_lander(1, SHAPE)
	for i in range(1000):
		cart.replayWithSavedParams()
	exit()


if __name__ == '__main__':
	main()
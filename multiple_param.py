from deap import base
from deap import creator
from deap import tools

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

# import hyperparameter_tuning_genetic_test
from networks import network_learning_rate as fitness_val_func

import elitism


# BOUNDS_LOW =  [4, 0.01]
# BOUNDS_HIGH = [50, 1.00]

BOUNDS_LOW, BOUNDS_HIGH = 0.001, 0.999

NUM_OF_PARAMS = 8

# Genetic Algorithm constants:
POPULATION_SIZE = 30
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 46
HALL_OF_FAME_SIZE = 3
CROWDING_FACTOR = 16.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# # define the hyperparameter attributes individually:
# for i in range(NUM_OF_PARAMS):
# 	# "hyperparameter_0", "hyperparameter_1", ...
# 	toolbox.register("hyperparameter_" + str(i),
# 					 random.uniform,
# 					 BOUNDS_LOW[i],
# 					 BOUNDS_HIGH[i])

# # create a tuple containing an attribute generator for each param searched:
# hyperparameters = ()
# for i in range(NUM_OF_PARAMS):
# 	hyperparameters = hyperparameters + \
# 					  (toolbox.__getattribute__("hyperparameter_" + str(i)),)


def hyperparameters():
	return [random.random() for _ in range(NUM_OF_PARAMS)]

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, hyperparameters)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


toolbox.register("evaluate", fitness_val_func.fitness_func)


# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR)

toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR, indpb=1.0/NUM_OF_PARAMS)


# Genetic Algorithm flow:
def main():

	# create initial population (generation 0):
	population = toolbox.populationCreator(n=POPULATION_SIZE)

	# prepare the statistics object:
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("max", numpy.max)
	stats.register("avg", numpy.mean)

	# define the hall-of-fame object:
	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

	# perform the Genetic Algorithm flow with hof feature added:
	population, logbook = elitism.eaSimpleWithElitism(\
		population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

	# print best solution found:
	print("- Best solution is: ")
	print("params = ", hof.items[0], sum(hof.items[0])/len(hof.items[0]))
	print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])

	# extract statistics:
	maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

	# plot statistics:
	sns.set_style("whitegrid")
	plt.plot(maxFitnessValues, color='red')
	plt.plot(meanFitnessValues, color='green')
	plt.xlabel('Generation')
	plt.ylabel('Max / Average Fitness')
	plt.title('Max and Average fitness over Generations')
	plt.show()


if __name__ == "__main__":
	main()

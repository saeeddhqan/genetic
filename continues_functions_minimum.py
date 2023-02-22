from deap import base
from deap import creator
from deap import tools

import random
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns

from networks import network_learning_rate as fitness_val_func
import elitism

# problem constants:
DIMENSIONS = 1  # number of dimensions
BOUND_LOW, BOUND_UP = 4.0, 50.0  # boundaries for all dimensions

# Genetic Algorithm constants:
POPULATION_SIZE = 10
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.4   # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 3
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# set the random seed:
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)



def randomFloat(low, up):
	return [random.uniform(low, up)]

# create an operator that randomly returns a float in the desired range and dimension:
toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

toolbox.register("evaluate", fitness_val_func.fitness_func)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)


# Genetic Algorithm flow:
def main():

	# create initial population (generation 0):
	population = toolbox.populationCreator(n=POPULATION_SIZE)

	# prepare the statistics object:
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("min", np.min)
	stats.register("avg", np.mean)

	# define the hall-of-fame object:
	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

	# perform the Genetic Algorithm flow with elitism:
	population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
											  ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

	# print info for best solution found:
	best = hof.items[0]
	print("-- Best Individual = ", best)
	print("-- Best Fitness = ", best.fitness.values[0])

	# extract statistics:
	minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

	# plot statistics:
	# sns.set_style("whitegrid")
	plt.plot(minFitnessValues, color='red')
	plt.plot(meanFitnessValues, color='green')
	plt.xlabel('Generation')
	plt.ylabel('Min / Average Fitness')
	plt.title('Min and Average fitness over Generations')

	plt.show()


main()

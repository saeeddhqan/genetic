from deap import base
from deap import creator
from deap import tools

import random
import numpy

import brain
import elitism



SHAPE = (74,33)
POPULATION_SIZE = 70
P_CROSSOVER = 0.9
P_MUTATION = 0.6
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 4
CROWDING_FACTOR = 10.0

# set the random seed:
RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

lunar = brain.lunar_lander(POPULATION_SIZE, SHAPE, RANDOM_SEED)
NUM_OF_PARAMS = len(lunar)

BOUNDS_LOW, BOUNDS_HIGH = -1.0, 1.0  # boundaries for all dimensions

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

def randomFloat(low, up):
	return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]

toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

toolbox.register("individualCreator",
				 tools.initIterate,
				 creator.Individual,
				 toolbox.attrFloat)

toolbox.register("populationCreator",
				 tools.initRepeat,
				 list,
				 toolbox.individualCreator)


def score(individual):
	return lunar.getScore(individual[0], individual[1]),


toolbox.register("evaluate", score)


toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register("mate",
				 tools.cxSimulatedBinaryBounded,
				 low=BOUNDS_LOW,
				 up=BOUNDS_HIGH,
				 eta=CROWDING_FACTOR)

toolbox.register("mutate",
				 tools.mutPolynomialBounded,
				 low=BOUNDS_LOW,
				 up=BOUNDS_HIGH,
				 eta=CROWDING_FACTOR,
				 indpb=1.0/NUM_OF_PARAMS)


# Genetic Algorithm flow:
def main():

	population = toolbox.populationCreator(n=POPULATION_SIZE)

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("max", numpy.max)
	stats.register("avg", numpy.mean)

	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

	population, logbook = elitism.eaSimpleWithElitism(
		population,toolbox,cxpb=P_CROSSOVER,mutpb=P_MUTATION,ngen=MAX_GENERATIONS,stats=stats,halloffame=hof,verbose=True)

	best = hof.items[0]
	print()
	print("Best Solution = ", best)
	print("Best Score = ", best.fitness.values[0])
	print()

	lunar.saveParams(best)

	# print("Running 100 episodes using the best solution...")
	# scores = []
	# for test in range(100):
	# 	scores.append(lunar.getScore(best))
	# print("scores = ", scores)
	# print("Avg. score = ", sum(scores) / len(scores))


if __name__ == "__main__":
	main()
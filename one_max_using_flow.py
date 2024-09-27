from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import numpy
from deap import algorithms

# ind means individual :_)

# problem constants:
ONE_MAX_LENGTH = 100 # length of bit string to be optimized
# Genetic Algorithm constants:
POPULATION_SIZE = 200 # number of individuals in population
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.1 # probability for mutating
# an individual
MAX_GENERATIONS = 50 # max number of generations for
# stopping condition
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def one_max_fitness(individual):
	return sum(individual), # it returns a tuple

toolbox = base.Toolbox()
# Here we just try to register a function named zero_one to return a random number
# between zero and one.
toolbox.register("zero_one", random.randint, 0, 1)


# 1.0 means we want to maximize our fitness function
# for more than one param, we can add it to weights' tuple
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# initializing each individual of our population
creator.create("individual", list, fitness=creator.FitnessMax)

# create first chromosomes
toolbox.register("individual_creator", tools.initRepeat, creator.individual, toolbox.zero_one, ONE_MAX_LENGTH)
toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)

# it evaluates chromosomes
toolbox.register("evaluate", one_max_fitness)

# selection method is tournament. it selects three inds and returns the fittest ind.
toolbox.register("select", tools.selTournament, tournsize=3)
# it splits two inds and create new offsprints: 11.00 and 00.11 => 1111
toolbox.register("mate", tools.cxOnePoint)
# it flips individuals' bits with the probability of 1/OML. we call this mutation :-)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


# FLOW begins here
# creating the first pop
population = toolbox.population_creator(n=POPULATION_SIZE)


# prepare the statistics object:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", numpy.max)
stats.register("avg", numpy.mean)

# algorithm flow:
population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                               stats=stats, verbose=True)


# extract statistics:
max_fitness_values, mean_fitness_values = logbook.select("max", "avg")

plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()

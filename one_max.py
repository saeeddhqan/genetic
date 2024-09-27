from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt

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
generation_counter = 0

# calculating fitness of each ind in population
fitness_values = list(map(toolbox.evaluate, population))
print(fitness_values)

for individual, fitness_value in zip(population, fitness_values):
	individual.fitness.values = fitness_value

fitness_values = [individual.fitness.values[0] for individual in population]

print(fitness_values)

max_fitness_values = []
mean_fitness_values = []

while max(fitness_values) < ONE_MAX_LENGTH and generation_counter < MAX_GENERATIONS:
	generation_counter = generation_counter + 1

	# selection
	offspring = toolbox.select(population, len(population))
	offspring = list(map(toolbox.clone, offspring))

	# offsprings
	for child1, child2 in zip(offspring[::2], offspring[1::2]):
		if random.random() < P_CROSSOVER:
			toolbox.mate(child1, child2)
			del child1.fitness.values
			del child2.fitness.values

	# mutation. removing fitness values only if ind has changed, otherwise it stays the same.
	for mutant in offspring:
		if random.random() < P_MUTATION:
			toolbox.mutate(mutant)
			del mutant.fitness.values

	# add fitness values for offsprings
	fresh_individuals = [ind for ind in offspring if not ind.fitness.valid]
	fresh_fitness_values = list(map(toolbox.evaluate, fresh_individuals))
	for individual, fitness_value in zip(fresh_individuals, fresh_fitness_values):
		individual.fitness.values = fitness_value


	population[:] = offspring
	fitness_values = [ind.fitness.values[0] for ind in population]

	max_fitness = max(fitness_values)
	mean_fitness = sum(fitness_values) / len(population)
	max_fitness_values.append(max_fitness)
	mean_fitness_values.append(mean_fitness)
	print(f"- Generation {generation_counter}: Max Fitness = {max_fitness}, Avg Fitness = {mean_fitness}")
	best_index = fitness_values.index(max(fitness_values))
	print("Best Individual = ", *population[best_index], "\n")

plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()
import random
import numpy as np

def sel_tournament(fitness_values, tourn_size=3):
	"""It only returns elements, not their index"""
	selected = []
	itr = len(fitness_values)/tourn_size
	if len(fitness_values)%tourn_size != 0:
		itr += 1
	for i in range(int(itr)):
		if len(fitness_values) <= tourn_size:
			indeces = fitness_values
		else:
			indeces = random.sample(fitness_values, tourn_size)
		selected.append(max(indeces))
		[fitness_values.pop(fitness_values.index(x)) for x in indeces]

	return selected

# This method is called hall of fame(hof)
def sel_top_k(fitness_values, k_size, reptition_killer=True, epsilon_range=(0.0001, 0.00001)):
	# fitness_values has reptition values 0.4, 0.4, ..

	if reptition_killer:
		fitness_values = [x + random.uniform(epsilon_range[0], epsilon_range[1]) for x in fitness_values]
	idxs = sorted(list(set(fitness_values)), reverse=True)[:k_size]
	selected = [fitness_values.index(x) for x in idxs]
	return selected


def cx_mean(offsprings, k_size=0.1, shuffle_prob=0.5):
	k_size = int((k_size * len(offsprings)) * 2)
	if k_size < 2:
		return []
	new_offsprings = []
	if random.random() <= shuffle_prob:
		random.shuffle(offsprings)
	parents = offsprings[:k_size]
	for i in range(int(k_size/2)):
		new_offs = (parents[i] + parents[i+1]) / 2
		new_offsprings.append(new_offs)
	return new_offsprings

def cx_hole_digging(offsprings, k_size=0.1, radian=0.19, shuffle_prob=0.0):
	"""
	Nothing to say for now
	"""
	k_size = int(k_size * len(offsprings))
	if k_size == 0:
		return []
	new_offsprings = []

	if random.random() <= shuffle_prob:
		random.shuffle(offsprings)

	parents = offsprings[:k_size]
	for i in parents:
		scale = (radian * i)
		child_up = i + scale
		child_down = i - scale
		for j in range(10):
			if child_up > 1.0:
				scale /= 2
				child_up -= scale
				if j == 9 and child_up > 1.0:
					child_up = 1.0
			else:
				break

		new_offsprings.append(child_down)
		new_offsprings.append(child_up)
	return new_offsprings

def mut_random_epsilon(inds, prob=0.1, epsilon = 0.003):
	for k,v in enumerate(inds):
		if random.random() <= prob:
			v += epsilon
			if v < 1:
				inds[k] = v
	return inds

# a = [random.random() for x in range(10)]
# print(a)
# a = [[0.9, 0.5, 0.4]]
# print(a)
# print(cx_mean(a))
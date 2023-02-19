
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
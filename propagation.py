import matplotlib.pyplot as plt
import numpy as np
#This file simulates a population under selection, mutation, genetic drift.


N=10000 #Population size
L=100 #Number of loci

theta = (.1,.2) #mutation rates (forward, backwards)
optimum_mutation = theta[0]/(theta[0]+theta[1])
s = 1 #strength of selection
rho = N #recombination rate (very high)


def generate_mutated(size):
	"""
Parameters
----------
size : the size
	"""
	return np.random.choice(2,
				size=size,
				p=(optimum_mutation,1-optimum_mutation))


def fitness(genome):
	"""Quadratic fitness"""
	return np.exp(-s*L/(2*N) * (2*np.mean(genome,axis=-1)-1)**2)


class Population():
	def __init__(self,population=None):
		if population is None:
			population = generate_mutated((N,L))
			population = np.array(population,dtype="bool")
		self.population = population
	def allele_frequencies(self):
		return np.mean(self.population,axis=0)

	def mutation(self):
		"""Mutation works by sampling a Poisson number of genes and
		   mutating them """
		number_of_mutations=np.random.poisson(L*(theta[0]+theta[1]))
		mutated_indexes=(np.random.randint(N,size=number_of_mutations),
				 np.random.randint(L,size=number_of_mutations))
		self.population[mutated_indexes]=generate_mutated(number_of_mutations)

	def selection_drift_sex(self):
		"""For each offspring we sample two parents proportionnally
		to fitness, and recombine their genomes"""
		population_fitnesses = fitness(self.population)
		sum_fitnesses = np.sum(population_fitnesses)
		population_fitnesses = population_fitnesses/sum_fitnesses

		list_parents = np.random.choice(N,p=population_fitnesses,size=2*N)
		list_crossing_overs = np.random.randint(L,size=N)
		new_population = np.array([[False]*L]*N,dtype="bool")
		for n in range(N):
			parents = (list_parents[2*n],list_parents[2*n+1])
			crossing_over = list_crossing_overs[n]
			new_genome = np.append(
					self.population[parents[0],:crossing_over],
					self.population[parents[1],crossing_over:]
					)
			new_population[n] = new_genome





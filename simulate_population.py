import matplotlib.pyplot as plt
import numpy as np
#This file contains the function Simulate which
# simulates a population under selection, mutation, genetic drift.



def generate_mutated(size,theta):
	"""
Parameters
----------
size : the size
	"""
	optimum_mutation = theta[0]/(theta[0]+theta[1])
	return np.random.choice([True,False],
				size=size,
				p=(optimum_mutation,1-optimum_mutation))

def fitness(genome,s,N,L):
	"""Quadratic fitness"""
	return np.exp(-s*L/(2*N) * (2*np.mean(genome,axis=-1)-1)**2)


def generate_pop(theta,N,L):
	"""Generates a population with allele density
	 x**(2theta1-1)(1-x)**(2theta2 - 1)"""
	x = np.linspace(1e-4,1-1e-4,1000)
	equilibrium_distribution = x**(2*theta[0]-1) * (1-x)**(2*theta[1]-1)
	equilibrium_distribution /= np.sum(equilibrium_distribution)
	frequencies = np.random.choice(x,p=equilibrium_distribution,size=L)
	population = np.array([[False]*L]*N,dtype="bool")
	for l in range(L):
		proba = (frequencies[l],1-frequencies[l])
		population[:,l] = np.random.choice([True,False],p=proba,size=N)
	return population

class Population():
	def __init__(self,theta,N,L,population=None):
		if population is None:
			population = generate_pop(theta,N,L)
		self.population = population
		self.varfitnesses = None

	def allele_frequencies(self):
		return np.mean(self.population,axis=0)

	def mutation(self,theta,N,L):
		"""Mutation works by sampling a Poisson number of genes and
		   mutating them """
		number_of_mutations=np.random.poisson(L*(theta[0]+theta[1]))
		mutated_indexes=(np.random.randint(N,size=number_of_mutations),
				 np.random.randint(L,size=number_of_mutations))
		self.population[mutated_indexes]=generate_mutated(number_of_mutations,theta)

	def selection_drift_sex(self,s,N,L):
		"""For each offspring we sample two parents proportionnally
		to fitness, and recombine their genomes"""
		population_fitnesses = fitness(self.population,s,N,L)
		sum_fitnesses = np.sum(population_fitnesses)
		population_fitnesses = population_fitnesses/sum_fitnesses
		self.varfitnesses = np.var(population_fitnesses)

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
		self.population = new_population


def Simulate(theta,s,N,L,T,record_every = 10):
	"""Simulates population evolution over time T. Returns vector of allele frequencies.
	Parameters:
	----------
	theta
	s
	N
	L
	T: optional.
	recond_every: optional:how often should the population be recorded ?

	Returns:
	list_allele_frequencies,list_varfitnesses,parameters
	"""
	nb_timesteps = int(T*N)+1
	list_allele_frequencies = np.zeros((nb_timesteps//record_every+1,L))
	list_varfitnesses = np.zeros(nb_timesteps//record_every+1)
	pop = Population(theta,N,L)

	for tN,t in enumerate(np.linspace(0,T,nb_timesteps)):
		pop.selection_drift_sex(s,N,L)
		pop.mutation(theta,N,L)
		if tN%record_every==0:
			print("Done: step "+str(tN)+" of "+str(nb_timesteps))
			list_allele_frequencies[tN//record_every] = pop.allele_frequencies()
			list_varfitnesses[tN//record_every] = pop.varfitnesses
	parameters = (theta,s,N,L,T)
	return list_allele_frequencies,list_varfitnesses,parameters

def plot_propagation_chaos(list_allele_frequencies,T=1):
	"""Plots the result of Simulate"""
	fig=plt.figure()
	ax=plt.axes()
	x = np.linspace(0,T,np.shape(list_allele_frequencies)[0])

	for l in range(L):
		ax.plot(x,list_allele_frequencies[:,l],alpha=.1,color="grey")
	ax.plot(x,np.mean(list_allele_frequencies,axis=1),color="green")
	ax.set_ylim((0,1))
	ax.set_xlabel("t")
	ax.set_ylabel("Frequency of the +1 allele")
	plt.show()
